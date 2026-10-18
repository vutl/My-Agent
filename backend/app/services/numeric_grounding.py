"""Metric-agnostic numeric grounding for generated answers.

The claim validator recognises metric/value pairs it has vocabulary for (Acc,
F1, CCC, ...). Numbers attached to anything else -- revenue, hazard ratios,
torque, doanh thu -- were never checked, so a fabricated value passed as
"no metric claims". This module checks every quantitative number in an answer
without knowing what it measures:

1. Grounding: the number (format-normalised: decimal comma, thousands
   separators, percent <-> fraction, rounding) must occur in the evidence, or
   be the difference of two numbers in one evidence table row/column.
2. Context: when the evidence occurrence is a table cell, its labels (row
   label cells, column header path, section rows) are compared with the
   labels the answer clause mentions. If the clause names a label of a table
   dimension (a row such as "Alpha", a column group such as "SQuAD") and the
   cell carries a different, unmentioned label of that same dimension, the
   value was borrowed from another row/condition.

No paper, metric or domain names are used; labels come from the evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
import hashlib
import re
import unicodedata

_NUMBER_RE = re.compile(
    # Thousands groups are exactly three digits ("0.7348" is one number);
    # a plain space never groups digits ("75 123 samples" is two numbers).
    r"(?<![\w.])[-−]?(?:\d+[^\S\r\n]+\.[^\S\r\n]+\d+|"
    r"\d{1,3}(?:[,.\u202f\u00a0]\d{3}(?!\d))+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
)
_REFERENCE_PREFIX_RE = re.compile(
    r"(?:table|tab\.|figure|figs?\.?|bảng|hình|biểu\s+đồ|sơ\s+đồ|section|mục|page|pages|"
    r"trang|eq\.?|equation|chapter|chương|step|bước|stage|phase|version|ver\.?|v|"
    r"top|layer|lớp|appendix|phụ\s+lục|exhibit|chart|no\.)\s*$",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:;|,\s|\s(?:and|while|whereas|but|và|còn|trong\s+khi|nhưng|so\s+với|versus|vs\.?)\s)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:without|w/o|w/out|excluding|except|lacking|removing|removed|minus|ablating|no|"
    r"khong\s+(?:co|dung|su\s+dung)|bo|loai\s+bo|thieu|tru)\s+(?:the\s+)?$",
)
_SEPARATOR_ROW_RE = re.compile(r"^\s*\|?\s*:?-{2,}")


@dataclass(frozen=True)
class NumberMention:
    value: str
    text: str
    reason: str  # "ungrounded" | "context_conflict"
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class NumberAssessment:
    claim_id: str
    value: str
    text: str
    start: int
    end: int
    status: str
    reason: str
    evidence_locators: tuple[dict, ...] = ()
    structured: bool = False

    def to_dict(self) -> dict:
        return {**self.__dict__, "evidence_locators": list(self.evidence_locators)}


@dataclass
class GroundingResult:
    checked: int = 0
    unsupported: list[NumberMention] = field(default_factory=list)
    # Values whose every occurrence in the answer was matched to one table
    # cell with its row and column named: positive, metric-agnostic support.
    corroborated: set[str] = field(default_factory=set)
    # Values checked and found consistent with a table cell's row/column
    # everywhere they occur (prose matches carry no structure to check).
    grounded: set[str] = field(default_factory=set)
    _structured: dict[str, list[bool]] = field(default_factory=dict)
    assessments: list[NumberAssessment] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.unsupported


@dataclass(frozen=True)
class _Occurrence:
    values: frozenset[Decimal]
    # dimension -> label (folded); empty for prose occurrences
    labels: tuple[tuple[str, str], ...]
    prose: str = ""
    locator: dict = field(default_factory=dict)
    unit: str = ""
    comparator: str = "="
    modifiers: tuple[tuple[str, str], ...] = ()


@dataclass
class _Evidence:
    occurrences: list[_Occurrence] = field(default_factory=list)
    # dimension -> set of folded labels, per document vocabulary
    dimensions: dict[str, set[str]] = field(default_factory=dict)
    differences: set[Decimal] = field(default_factory=set)
    table_values: set[Decimal] = field(default_factory=set)
    # folded label -> its written forms with diacritics ("nam" -> {"năm"}):
    # unaccented matching must not merge "Năm" (year) with "Nam" (male).
    forms: dict[str, set[str]] = field(default_factory=dict)
    next_table: int = 0
    numeric_headers: dict[str, str] = field(default_factory=dict)

    def label(self, raw: str) -> str:
        folded = _fold(raw)
        self.forms.setdefault(folded, set()).add(_mark_key(raw))
        return folded


def _mark_key(text: str) -> str:
    """Casefolded text that keeps diacritics; aligned with :func:`_fold`."""
    return re.sub(r"[_\-]+", " ", unicodedata.normalize("NFKC", str(text or "")).casefold())


def _segment_allowed(label: str, start: int, end: int, folded: str, marked: str, evidence) -> bool:
    """A folded match is real unless the written text carries diacritics that
    differ from every written form of the label ("năm" vs label "Nam")."""
    if evidence is None or len(marked) != len(folded):
        return True
    segment = marked[start:end]
    if _fold(segment) == segment:  # typed without diacritics
        return True
    forms = evidence.forms.get(label)
    if not forms:
        return True
    pieces = {piece for form in forms for piece in [form, *form.split()]}
    return segment in pieces or segment.replace(" ", "") in {piece.replace(" ", "") for piece in pieces}


def _fold(text: str) -> str:
    # NFKD maps styled math letters (CCC𝑉) to plain ones; "_"/"-" join words.
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[_\-]+", " ", text.replace("đ", "d"))


def _clean_label(cell: str) -> str:
    cell = re.sub(r"[*`~]", "", str(cell or ""))
    cell = re.sub(r"[∗†‡§¶#]+", "", cell)
    # "Wav2Vec 2.0 (baseline)" is named "Wav2Vec 2.0"; "Acc." is "Acc".
    core = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", cell)
    cell = core if core.strip() else cell
    return " ".join(cell.split()).strip(" :.;,")


def _number_candidates(raw: str) -> frozenset[Decimal]:
    """All plausible readings of a printed number (EN and VI separators)."""
    text = raw.replace("−", "-").replace(" ", "").replace(" ", "")
    sign = Decimal(-1) if text.startswith("-") else Decimal(1)
    text = re.sub(r"\s+", "", text.lstrip("-"))
    if not re.search(r"\d", text):
        return frozenset()
    readings: set[Decimal] = set()
    if re.fullmatch(r"\d+", text):
        readings.add(Decimal(text))
    elif re.fullmatch(r"\d+[.,]\d+", text):
        integer, fraction = re.split(r"[.,]", text)
        readings.add(Decimal(f"{integer}.{fraction}"))
        if len(fraction) == 3 and not integer.startswith("0"):
            # 1,234 / 1.234 may be a thousands group; 0.001 never is.
            readings.add(Decimal(integer + fraction))
    else:
        groups = re.split(r"[.,]", text)
        separators = re.findall(r"[.,]", text)
        if not separators:
            return frozenset()
        if len(set(separators)) == 1:
            readings.add(Decimal("".join(groups)))
        else:
            decimal = separators[-1]
            thousands = "," if decimal == "." else "."
            readings.add(Decimal(text.replace(thousands, "").replace(decimal, ".")))
    # More than 15 significant digits is an identifier or digits OCR ran
    # together, never a measured value; arithmetic on it overflows Decimal.
    return frozenset(sign * value for value in readings if len(value.as_tuple().digits) <= _MAX_VALUE_DIGITS)


_MAX_VALUE_DIGITS = 15


def _rounded(value: Decimal, places: int) -> Decimal | None:
    """``round`` that skips values Decimal cannot represent at ``places``."""
    try:
        return round(value, places)
    except (InvalidOperation, OverflowError):
        return None


def _decimals(raw: str) -> int:
    """Printed precision; a thousands reading (1,234) is an exact integer
    either way, so the trailing group length is a safe tolerance."""
    match = re.search(r"[.,](\d+)$", re.sub(r"\s+", "", raw))
    return len(match.group(1)) if match else 0


def _matches(answer_values: frozenset[Decimal], decimals: int, evidence: frozenset[Decimal], *, percentage_scale: bool = False, magnitude: bool = False) -> bool:
    tolerance = Decimal("0.5") * Decimal(10) ** (-decimals)
    for a in answer_values:
        for e in evidence:
            scaled = [e]
            # A fraction may be stated as a percentage and vice versa.
            if percentage_scale and abs(e) <= 1 < abs(a):
                scaled.append(e * 100)
            elif percentage_scale and abs(a) <= 1 < abs(e):
                scaled.append(e / 100)
            if any(abs((abs(a) - abs(value)) if magnitude else (a - value)) <= tolerance for value in scaled):
                return True
    return False


_UNIT_RE = re.compile(
    r"^\s*(%|‰|Nm|mmHg|GWh|kWh|mg|kg|µg|μg|g|ml|mL|L|km|mm|cm|ms|s|"
    r"million|billion|thousand|triệu|tỷ|nghìn|dollars?|USD|EUR|đồng)(?!\w)", re.I,
)


def _quantity_tags(text: str, match: re.Match, header: str = "") -> tuple[str, str]:
    suffix = text[match.end():]
    unit_match = _UNIT_RE.match(suffix)
    unit = (unit_match.group(1) if unit_match else "").casefold().replace("μ", "µ")
    if not unit:
        if "%" in header:
            unit = "%"
        header_unit = re.search(r"\(([^()]+)\)", header)
        if header_unit and not unit:
            unit_match = _UNIT_RE.match(header_unit.group(1))
            unit = (unit_match.group(1) if unit_match else "").casefold()
    prefix = text[:match.start()].rstrip()
    comparator_match = re.search(r"(<=|>=|<|>|≤|≥|±)\s*$", prefix)
    comparator = comparator_match.group(1) if comparator_match else "="
    return unit.replace("μ", "µ"), {"≤": "<=", "≥": ">="}.get(comparator, comparator)


def _quantity_compatible(occurrence: _Occurrence, unit: str, comparator: str) -> bool:
    if occurrence.comparator != comparator:
        return False
    if unit and occurrence.unit and unit != occurrence.unit and not (unit == "%" and occurrence.unit == "ratio"):
        return False
    if unit in {"%", "‰"} and occurrence.unit not in {unit, "ratio"}:
        return False
    return True


def _inclusion_modifiers(text: str) -> tuple[tuple[str, str], ...]:
    """Literal inclusion/exclusion qualifiers, not a domain/entity lexicon.

    The object comes from the source (taxes, fees, VAT, adapter, ...). This
    deliberately handles only an explicit qualifier, not free-form entailment.
    """
    folded = _fold(text)
    pattern = (r"\b(not including|with no|excluding|without|including|with|chua bao gom|chua gom|chua|"
               r"khong bao gom|khong gom|da bao gom|da gom|bao gom)\s+(?:the\s+)?([a-z]+)\b")
    excluded = {"not including", "with no", "excluding", "without", "chua bao gom", "chua gom", "chua", "khong bao gom", "khong gom"}
    return tuple((m.group(2), "excluded" if m.group(1) in excluded else "included")
                 for m in re.finditer(pattern, folded))


def _modifier_conflicts(occurrence: _Occurrence, text: str) -> bool:
    stated = dict(_inclusion_modifiers(text))
    return any(obj in stated and stated[obj] != polarity for obj, polarity in occurrence.modifiers)


def _cross_source_reference(sentence: str) -> bool:
    """An explicit non-comparability caveat does not inherit the prior paper.

    This removes only inherited heading scope, not local owners, evidence
    requirements, quantities or units. It cannot exempt fabricated values.
    """
    numbers = list(_NUMBER_RE.finditer(sentence))
    if len(numbers) < 2:
        return False
    prefix = _fold(sentence[:numbers[0].start()])
    link = _fold(sentence[numbers[0].end():numbers[-1].start()])
    return bool(re.search(r"\b(?:do not|cannot|can't|khong nen|khong the|khong)\b", prefix)
                and re.search(r"\b(?:compare|comparing|so sanh|so truc tiep|doi chieu)\b", prefix)
                and re.search(r"\b(?:with|versus|vs|and|voi|va)\b", link))


def _table_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in str(text or "").splitlines():
        if line.strip().startswith("|") or (current and line.count("|") >= 2):
            current.append(line)
        else:
            if current:
                blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [block for block in blocks if any(_SEPARATOR_ROW_RE.match(row) for row in block)]


def _cells(row: str) -> list[str]:
    # OCR sometimes splits decimals inside cells ("≈ 21 . 7").
    row = re.sub(r"(?<=\d)\s+([.,])\s+(?=\d)", r"\1", row)
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    # A cell that is one space-grouped number ("10 120") is 10120.
    return [
        cell.replace(" ", "") if re.fullmatch(r"[-−]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?", cell) else cell
        for cell in cells
    ]


def _is_numeric_cell(cell: str) -> bool:
    return bool(_NUMBER_RE.search(cell)) and not re.search(r"[A-Za-zÀ-ỹ]{3,}", cell)


def _parse_table(block: list[str], evidence: _Evidence, table_index: int) -> None:
    separator_indexes = [i for i, row in enumerate(block) if _SEPARATOR_ROW_RE.match(row)]
    width = max(len(_cells(row)) for row in block)

    def header_path(header_rows: list[list[str]]) -> list[list[str]]:
        # Column header path with group cells propagated to the right.
        headers: list[list[str]] = [[] for _ in range(width)]
        for level, row in enumerate(header_rows):
            previous = ""
            for column in range(width):
                label = re.sub(r"[*`~†‡]", "", row[column]).strip() if column < len(row) else ""
                if not label and column > 1 and previous:
                    # "| Model | SQuAD | | NQ | |": a group spans empty cells.
                    label = previous
                previous = label or previous
                if label:
                    headers[column].append(label)
        return headers

    # A table may repeat its header block (header rows + |---|) deep inside;
    # every body row uses the most recent header block above it.
    segments: list[tuple[list[list[str]], list[list[str]]]] = []
    start = 0
    for separator in separator_indexes:
        header_start = separator
        while header_start > start and not any(
            _is_numeric_cell(cell) for cell in _cells(block[header_start - 1])
        ):
            header_start -= 1
        if segments and header_start > start:
            segments[-1][1].extend(_cells(row) for row in block[start:header_start])
        segments.append(([_cells(row) for row in block[header_start:separator]], []))
        start = separator + 1
    if segments:
        segments[-1][1].extend(
            _cells(row) for i, row in enumerate(block[start:], start=start) if i not in separator_indexes
        )
    for header_rows, rows in segments:
        _parse_segment(header_path(header_rows), rows, width, evidence, table_index)


def _parse_segment(
    headers: list[list[str]], rows: list[list[str]], width: int, evidence: _Evidence, table_index: int,
) -> None:
    from app.services.evidence_validator import (_inferred_first_metric_group_qualifiers,
                                                 _metric_columns_from_header)
    flattened = [" ".join(path) for path in headers]
    inferred = _inferred_first_metric_group_qualifiers(flattened, _metric_columns_from_header(flattened))
    headers = [
        [*path[:-1], ".".join([*inferred[column], path[-1]])]
        if path and column in inferred and not all(group in _fold(" ".join(path)) for group in inferred[column])
        else path for column, path in enumerate(headers)
    ]
    numeric_columns = {
        column
        for row in rows
        for column, cell in enumerate(row)
        if _is_numeric_cell(cell)
    }
    section = ""
    numeric_rows: list[list[tuple[int, frozenset[Decimal]]]] = []
    first_header = {_fold(label) for path in headers for label in path[:1]}
    if rows and not any(_NUMBER_RE.search(cell) for cell in rows[0]) and len(
        {cell.strip() for cell in rows[0] if cell.strip()}
    ) >= 2:
        # Extractors often place the second header row ("| | EM | F1 |")
        # below the |---| separator.
        sub = rows[0]
        headers = [path + ([_clean_label(sub[column])] if column < len(sub) and sub[column].strip() else [])
                   for column, path in enumerate(headers)]
        rows = rows[1:]
    for row in rows:
        # Mixed cells ("94k hours", "≈ 21.7M") carry values and are labels too.
        values = [(column, cell) for column, cell in enumerate(row)
                  if _NUMBER_RE.search(cell) and (column > 0 or _is_numeric_cell(cell))]
        labels = [(column, _clean_label(cell)) for column, cell in enumerate(row)
                  if cell.strip() and not _is_numeric_cell(cell)]
        if not values:
            distinct = {label for _, label in labels if label}
            if len(distinct) >= 2 and {_fold(label) for label in distinct} & first_header:
                # A repeated header row without |---| starts a new schema.
                headers = [[_clean_label(row[column])] if column < len(row) and row[column].strip() else []
                           for column in range(width)]
                continue
            if len(distinct) == 1:  # "| Valence | Valence | ..." section row
                section = next(iter(distinct))
                evidence.dimensions.setdefault(f"t{table_index}:section", set()).add(evidence.label(section))
            continue
        context: list[tuple[str, str]] = []
        for column, label in labels:
            if not label or re.fullmatch(r"[-—–/ ]*", label):
                continue
            dimension = f"t{table_index}:row{column}"
            context.append((dimension, evidence.label(label)))
            evidence.dimensions.setdefault(dimension, set()).add(evidence.label(label))
        if section:
            context.append((f"t{table_index}:section", evidence.label(section)))
        # Numeric cells can qualify another measurement in the same row
        # (Week=12, Dose=50), not just supply independent answer values.
        # Retain all of them; activation requires an explicit header/value
        # phrase, so an unrelated repeated number cannot set a condition.
        numeric_context = []
        for column, cell in values:
            if not _is_numeric_cell(cell) or len(list(_NUMBER_RE.finditer(cell))) != 1 or column >= len(headers):
                continue
            header = _clean_label(" ".join(headers[column]))
            if not header:
                continue
            dimension = f"t{table_index}:cell{column}"
            evidence.numeric_headers[dimension] = header
            key = evidence.label(f"{header}: {cell}")
            evidence.dimensions.setdefault(dimension, set()).add(key)
            numeric_context.append((column, dimension, key))
        parsed_row: list[tuple[int, frozenset[Decimal]]] = []
        for column, cell in values:
            cell_context = [*context, *((d, label) for c, d, label in numeric_context if c != column)]
            for level, header in enumerate(headers[column] if column < width else []):
                dimension = f"t{table_index}:h{level}"
                cell_context.append((dimension, evidence.label(header)))
                if column in numeric_columns:
                    evidence.dimensions.setdefault(dimension, set()).add(evidence.label(header))
            for match in _NUMBER_RE.finditer(cell):
                candidates = _number_candidates(match.group(0))
                header_text = " ".join(headers[column] if column < width else [])
                unit, comparator = _quantity_tags(cell, match, header_text)
                if not unit:
                    # Units may be printed on the measure row rather than in
                    # the column header (e.g. owners are columns).
                    unit = _quantity_tags(cell, match, " ".join(row))[0]
                if not unit and re.search(r"(?i)\b(?:accuracy|acc|f1|precision|recall|ua|wa|wer|cer)\b", header_text):
                    unit = "ratio"
                evidence.occurrences.append(_Occurrence(
                    candidates, tuple(cell_context),
                    locator={"artifact_type": "table", "table": table_index, "row": len(numeric_rows),
                             "column": column, "printed_value": match.group(0)},
                    unit=unit, comparator=comparator,
                ))
                evidence.table_values.update(candidates)
                parsed_row.append((column, candidates))
        numeric_rows.append(parsed_row)
    _add_differences(numeric_rows, evidence)


def _add_differences(rows: list[list[tuple[int, frozenset[Decimal]]]], evidence: _Evidence) -> None:
    """Differences within a row or a column are legitimate derived numbers."""
    def add(group: list[frozenset[Decimal]]) -> None:
        flat = [next(iter(values)) for values in group if values][:24]
        for i, x in enumerate(flat):
            for y in flat[i + 1:]:
                difference = _rounded(abs(x - y), 6)
                if difference is not None:
                    evidence.differences.add(difference)

    for row in rows:
        add([values for _, values in row])
    columns: dict[int, list[frozenset[Decimal]]] = {}
    for row in rows:
        for column, values in row:
            columns.setdefault(column, []).append(values)
    for group in columns.values():
        add(group)


def _document_label(document: dict) -> str:
    """A document's own name: its first heading, else its filename stem."""
    for line in str(document.get("content") or "").splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            return _clean_label(heading.group(1))
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", str(document.get("filename") or ""))
    return _clean_label(re.sub(r"[_]+", " ", stem))


def _build_evidence(documents: list[dict]) -> _Evidence:
    evidence = _Evidence()
    table_index = 0
    multi_document = len({str(d.get("document_id") or id(d)) for d in documents}) >= 2
    for document in documents:
        first_occurrence = len(evidence.occurrences)
        _add_document(document, evidence, table_index)
        table_index = evidence.next_table
        doc_label = _document_label(document) if multi_document else ""
        if doc_label:
            # Several documents may report the same row/column under
            # different conditions (a 2023 vs a 2024 report).
            key = evidence.label(doc_label)
            evidence.dimensions.setdefault("doc", set()).add(key)
            evidence.occurrences[first_occurrence:] = [
                replace(o, labels=(*o.labels, ("doc", key)))
                for o in evidence.occurrences[first_occurrence:]
            ]
        # Numbers in printed headers/section labels are conditions too; do
        # not exempt every year or small ordinal in the generated answer.
        condition_labels: dict[str, set[str]] = {}
        for occurrence in evidence.occurrences[first_occurrence:]:
            for dimension, label in occurrence.labels:
                condition_labels.setdefault(dimension, set()).add(label)
        for dimension, labels in condition_labels.items():
            if dimension in evidence.numeric_headers:
                continue  # Row bindings are not free-standing value evidence.
            for label in labels:
                for match in _NUMBER_RE.finditer(label):
                    evidence.occurrences.append(_Occurrence(
                        _number_candidates(match.group(0)), ((dimension, label),),
                        locator={"artifact_type": "condition", "printed_label": label},
                    ))
        evidence.occurrences[first_occurrence:] = [
            replace(o, locator={**o.locator, "document_id": document.get("document_id"),
                                "revision_id": document.get("revision_id"),
                                "chunk_id": document.get("chunk_id"),
                                "page_number": document.get("page_number")})
            for o in evidence.occurrences[first_occurrence:]
        ]
    return evidence


def _add_facts(document: dict, evidence: _Evidence) -> list[str]:
    """Structured facts extracted from prose at ingest become labelled cells:
    entity ~ row, attribute ~ column, conditions ~ section/condition."""
    fact_quotes: list[str] = []
    for fact in document.get("facts") or []:
        from app.services.fact_extraction_service import validate_fact
        chunk_text = str(document.get("expanded_content") or document.get("content") or "")
        draft = validate_fact(fact, chunk_text)
        if draft is None or draft.validation_status != "source_bound":
            continue
        value = str(fact.get("value") or "")
        match = _NUMBER_RE.search(value)
        if not match:
            continue
        values = _number_candidates(match.group(0))
        labels: list[tuple[str, str]] = []
        for dimension, raw in (("fact:entity", fact.get("entity")), ("fact:attribute", fact.get("attribute"))):
            label = _clean_label(str(raw or ""))
            if label:
                key = evidence.label(label)
                evidence.dimensions.setdefault(dimension, set()).add(key)
                labels.append((dimension, key))
        for condition in fact.get("conditions") or []:
            label = _clean_label(str(condition or ""))
            if label:
                key = evidence.label(label)
                evidence.dimensions.setdefault("fact:cond", set()).add(key)
                labels.append(("fact:cond", key))
        quote_match = _NUMBER_RE.search(draft.quote)
        unit, comparator = _quantity_tags(draft.quote, quote_match) if quote_match else ("", "=")
        evidence.occurrences.append(_Occurrence(
            values, tuple(labels), locator={"artifact_type": "fact", "quote": draft.quote,
                                           "quote_start": draft.quote_start, "quote_end": draft.quote_end},
            unit=unit, comparator=comparator,
        ))
        fact_quotes.append(draft.quote)
    return fact_quotes


def _add_document(document: dict, evidence: _Evidence, table_index: int) -> None:
    fact_quotes = _add_facts(document, evidence)
    if True:  # keeps the original per-document block intact
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        text = "\n".join(
            str(part)
            for part in (
                document.get("filename"), document.get("caption"), document.get("content"),
                document.get("expanded_content"), metadata.get("caption"), metadata.get("content"),
            )
            if part
        )
        table_lines: set[str] = set()
        for block in _table_blocks(text):
            _parse_table(block, evidence, table_index)
            table_index += 1
            table_lines.update(block)
        owner_dimension = f"p{table_index}:{len(evidence.dimensions)}:owner"
        owner = ""
        sentences: list[tuple[str, str]] = []
        for line in text.splitlines():
            if line in table_lines:
                continue
            stripped = line.strip()
            if re.match(r"^#{1,6}\s", stripped):
                owner = ""
            # "- Model: Beta" / "**Beta**" lines own the measurements below.
            label = re.match(r"^(?:[-*+]\s+)?(?:\*\*)?[^:|*]{1,40}(?:\*\*)?:\s*\**([^:|*]{1,60}?)\**\s*$", stripped)
            bold = re.match(r"^(?:[-*+]\s+)?\*\*([^*]{1,60})\*\*:?\s*$", stripped)
            if not _NUMBER_RE.search(stripped) and (label or bold):
                owner = _clean_label((label or bold).group(1))
                if owner:
                    evidence.dimensions.setdefault(owner_dimension, set()).add(evidence.label(owner))
                continue
            for sentence in _sentence_parts(stripped, semicolon=True):
                sentences.append((sentence, owner))
        prose_owners = _prose_subjects(text)
        entity_dimension = owner_dimension + ":entity:owner"
        measure_dimension = owner_dimension + ":measure"
        if len({label for _, label in prose_owners}) >= 2:
            for _, label in prose_owners:
                evidence.dimensions.setdefault(entity_dimension, set()).add(evidence.label(label))
        else:
            prose_owners = []
        for sentence, sentence_owner in sentences:
            numbers = list(_NUMBER_RE.finditer(sentence))
            if not numbers:
                continue
            folded = _fold(sentence)
            labels = ((owner_dimension, evidence.label(sentence_owner)),) if sentence_owner else ()
            statement_label = _NUMBER_RE.sub(" ", sentence)
            if statement_label.strip():
                statement_dimension = owner_dimension + ":statement"
                key = evidence.label(_clean_label(statement_label))
                evidence.dimensions.setdefault(statement_dimension, set()).add(key)
                labels = (*labels, (statement_dimension, key))
            for match in numbers:
                candidates = _number_candidates(match.group(0))
                if any(_fact_owns_number(sentence, match, quote) for quote in fact_quotes):
                    continue  # a structured fact describes this number
                unit, comparator = _quantity_tags(sentence, match)
                a, b = next((span for span in _clause_spans(sentence)
                             if span[0] <= match.start() < span[1]), (0, len(sentence)))
                number_labels = labels
                if prose_owners:
                    # Explicit source subjects own the following measurements
                    # until another subject is named. Do not use answer text
                    # (or the numeric value itself) to infer this ownership.
                    subjects = [(pos, label) for pos, label in _prose_subjects(sentence)
                                if pos < match.start()]
                    if subjects:
                        number_labels = (*number_labels, (entity_dimension,
                            evidence.label(max(subjects, key=lambda item: item[0])[1])))
                        measure = _post_number_measure(sentence, match)
                        if measure:
                            key = evidence.label(measure)
                            evidence.dimensions.setdefault(measure_dimension, set()).add(key)
                            number_labels = (*number_labels, (measure_dimension, key))
                evidence.occurrences.append(_Occurrence(
                    candidates, number_labels, prose=folded,
                    locator={"artifact_type": "text", "quote": sentence,
                             "number_start": match.start(), "number_end": match.end()},
                    unit=unit, comparator=comparator,
                    modifiers=_inclusion_modifiers(sentence[a:b]),
                ))
            if len(numbers) >= 2:
                values = [
                    next(iter(candidates))
                    for m in numbers
                    if (candidates := _number_candidates(m.group(0)))
                ][:12]
                for i, x in enumerate(values):
                    for y in values[i + 1:]:
                        difference = _rounded(abs(x - y), 6)
                        if difference is not None:
                            evidence.differences.add(difference)
    evidence.next_table = table_index


def _prose_subjects(text: str) -> list[tuple[int, str]]:
    """Recognize explicit named subjects, not arbitrary capitalized words.

    This deliberately limited grammar supplies additional constraints when
    supported; unrecognized constructions retain the existing source guard.
    Names are read from the source, never from a paper-specific dictionary.
    """
    word = r"[A-ZÀ-ÖØ-ÞĐ][\w.-]*"
    name = rf"{word}(?:\s+{word}){{0,4}}"
    verbs = r"(?:reported|recorded|scored|registered|reached|ghi\s+nhận|ghi\s+nhan)"
    pattern = rf"(?<!\w)(?P<name>{name})\s+(?i:{verbs})(?!\w)"
    return [(match.start("name"), _clean_label(match.group("name")))
            for match in re.finditer(pattern, text)]


def _post_number_measure(sentence: str, match: re.Match) -> str:
    """Bind a counted noun phrase printed immediately after its quantity.

    Units, dates and bare numbers are not measurement names. In particular,
    '37 hospitalizations' cannot corroborate '37 confirmed cases'.
    """
    tail = sentence[match.end():]
    noun = re.match(r"\s+([A-Za-zÀ-ÖØ-öø-ÿĐđ][\w -]*?)(?=\s+(?:and|but|while|và|va)\b|[;,.!?]|$)", tail)
    if not noun:
        return ""
    label = _clean_label(noun.group(1))
    if len(label.split()) > 6 or re.match(r"(?i)^(?:in|of|on|during|năm|nam|tháng|thang)\b", label):
        return ""
    if _UNIT_RE.match(noun.group(1)):
        return ""
    # This dimension represents the counted noun, not every descriptive
    # modifier: 'confirmed influenza cases' and 'confirmed cases' both count
    # cases. Qualifiers remain in the canonical statement/modifier guards;
    # this helper does not claim to verify arbitrary noun-phrase entailment.
    return label.split()[-1]


def _fact_owns_number(sentence: str, match: re.Match, quote: str) -> bool:
    normalized = " ".join(sentence.split())
    position = normalized.find(quote)
    before = " ".join(sentence[:match.start()].split())
    start = len(before) + int(bool(before and sentence[match.start() - 1].isspace()))
    return position >= 0 and position <= start and start + len(match.group(0)) <= position + len(quote)


def _sentence_parts(text: str, *, semicolon: bool = False) -> list[str]:
    """Split punctuation outside numeric tokens, retaining original offsets.

    PDF/OCR '0 . 45' is one printed token, not two sentences. Plain-space
    integers and ordinary '3. 12' sentence/list punctuation stay separate.
    """
    numbers = list(_NUMBER_RE.finditer(text))
    parts, start = [], 0
    pattern = r"(?<=[.!?;])\s+" if semicolon else r"(?<=[.!?])\s+"
    for boundary in re.finditer(pattern, text):
        if any(match.start() <= boundary.start() < match.end() for match in numbers):
            continue
        parts.append(text[start:boundary.start()])
        start = boundary.end()
    parts.append(text[start:])
    return parts


_SHORT_COMMON_WORDS = frozenset(
    "the and for not are was all one two has had its out per via who how why can may "
    "use set top new old low big end day con cho cua mot hai voi".split()
)


def _find(pattern: str, label: str, folded: str, marked: str, evidence) -> re.Match[str] | None:
    for match in re.finditer(rf"(?<![\w-]){pattern}(?![\w-])", folded):
        if _segment_allowed(label, match.start(), match.end(), folded, marked, evidence):
            return match
    return None


def _label_mentioned(label: str, folded_text: str, raw_text: str, marked: str = "", evidence=None) -> bool:
    marked = marked or _mark_key(raw_text)
    if len(label) < 2 or label.isdigit():
        return False
    if len(label) == 3 and label.isalpha() and label not in _SHORT_COMMON_WORDS:
        return _find(re.escape(label), label, folded_text, marked, evidence) is not None
    if len(label) <= 3 and evidence is not None:
        # A short label written with diacritics ("Nữ") is distinctive as
        # written; match it exactly in the accented text.
        for form in evidence.forms.get(label, ()):
            if form != _fold(form) and re.search(rf"(?<![\w-]){re.escape(form)}(?![\w-])", marked):
                return True
    if len(label) <= 3:
        # Short labels (NQ, CE, Q2) count only when written with a capital or
        # digit, so ordinary words ("an", "no", "và") never match.
        for match in re.finditer(rf"(?<![\w-]){re.escape(label)}(?![\w-])", raw_text, re.IGNORECASE):
            if re.search(r"[A-Z0-9]", match.group(0)):
                return True
        return False
    if _find(re.escape(label), label, folded_text, marked, evidence) is not None:
        return True
    compact = label.replace(" ", "")
    return compact != label and _find(re.escape(compact), label, folded_text, marked, evidence) is not None


# Words too generic to identify one row/column on their own.
_GENERIC_LABEL_TOKENS = frozenset(
    "the of and for with model models method methods approach result results total "
    "overall average mean score value values set data dataset rate ratio group arm "
    "mo hinh phuong phap ket qua tong trung binh nhom".split()
)
_NEGATED_LABEL_RE = re.compile(r"^(?:w/o|w/out|without|no|minus|excluding|khong(?: co| dung)?|bo)\s+(.+)$")


def _tokens(label: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-z/]+", label) if token]


def _token_mentioned(
    token: str, folded_text: str, raw_text: str, marked: str = "", evidence=None, label: str = "",
) -> bool:
    if len(token) < 2 or token in _GENERIC_LABEL_TOKENS:
        return False
    marked = marked or _mark_key(raw_text)
    if len(token) <= 3 and not (len(token) == 3 and token.isalpha() and token not in _SHORT_COMMON_WORDS):
        return _label_mentioned(token, folded_text, raw_text, marked, None)
    return _find(re.escape(token), label or token, folded_text, marked, evidence) is not None


def _label_spans(
    label: str, others: set[str], folded: str, raw: str, marked: str = "", evidence=None, owner: str = "",
) -> list[int]:
    """Positions where a label is named: the whole phrase, or a token no
    other label of its dimension contains ("q1" for "Q1 2024 Revenue")."""
    owner = owner or label
    positions: list[int] = []
    if _label_mentioned(label, folded, raw, marked, evidence if owner == label else None):
        index = folded.find(label)
        positions.append(index if index >= 0 else 0)
    content_tokens = [t for t in _tokens(label) if len(t) >= 2 and t not in _GENERIC_LABEL_TOKENS]
    if not positions and len(content_tokens) >= 2 and all(
        _token_mentioned(token, folded, raw, marked, evidence, owner) for token in content_tokens
    ):
        # Every word of "IEMOCAP.WAR" is named, even if not as one phrase.
        match = re.search(rf"(?<![\w-]){re.escape(content_tokens[0])}(?![\w-])", folded)
        positions.append(match.start() if match else 0)
    shared = {token for other in others if other != label for token in _tokens(other)}
    for token in _tokens(label):
        if token in shared or not _token_mentioned(token, folded, raw, marked, evidence, owner):
            continue
        match = re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", folded)
        positions.append(match.start() if match else 0)
    return positions


def _mentions(text: str, evidence: _Evidence) -> dict[str, tuple[set[str], set[str]]]:
    """dimension -> (positively mentioned labels, negated labels).

    A label that itself denotes absence ("w/o FPN") is positively named by a
    negated mention of its core ("excluding the FPN") and contradicted by a
    plain mention of the core.
    """
    folded = _fold(text)
    marked = _mark_key(text)
    result: dict[str, tuple[set[str], set[str]]] = {}
    numeric_mentions = _numeric_binding_spans(text, evidence)
    for dimension, labels in evidence.dimensions.items():
        if dimension in evidence.numeric_headers:
            positive = {label for _, label in numeric_mentions.get(dimension, ())}
            if positive:
                result[dimension] = (positive, set())
            continue
        positive: set[str] = set()
        negative: set[str] = set()
        cores = {label: (_NEGATED_LABEL_RE.match(label).group(1) if _NEGATED_LABEL_RE.match(label) else None)
                 for label in labels}
        for label in labels:
            core = cores[label]
            target = core or label
            peers = {cores[other] or other for other in labels}
            spans = _label_spans(target, peers, folded, text, marked, evidence, label)
            if spans:
                # One mention, one polarity: a cue before any part of it
                # ("without the copy mechanism") negates the whole mention.
                negated = not dimension.endswith(":statement") and any(
                    _NEGATION_RE.search(folded[max(0, index - 24):index]) for index in spans)
                if core is None:
                    (negative if negated else positive).add(label)
                else:
                    (positive if negated else negative).add(label)
            if core is not None and _label_mentioned(label, folded, text, marked, evidence):
                positive.add(label)
                negative.discard(label)
        if positive or negative:
            result[dimension] = (positive, negative)
    return result


def _numeric_binding_spans(text: str, evidence: _Evidence) -> dict[str, list[tuple[int, str]]]:
    folded = _fold(text)
    result: dict[str, list[tuple[int, str]]] = {}
    aliases = {"week": ("week", "tuan"), "year": ("year", "nam"),
               "month": ("month", "thang"), "dose": ("dose", "lieu")}
    for dimension, raw_header in evidence.numeric_headers.items():
        header = _fold(raw_header)
        for name in aliases.get(header, (header,)):
            for found in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)\s*[:=]?\s*", folded):
                number = _NUMBER_RE.match(folded, found.end())
                if not number:
                    continue
                for label in evidence.dimensions[dimension]:
                    value = _NUMBER_RE.search(label.split(": ", 1)[-1])
                    if value and _number_candidates(number.group(0)) & _number_candidates(value.group(0)):
                        result.setdefault(dimension, []).append((found.start(), label))
    return result


def _conflicts(occurrence: _Occurrence, context) -> bool:
    """A table cell conflicts when the number's context names a label of one
    of the cell's dimensions but not the cell's own label (or negates it)."""
    grouped: dict[str, set[str]] = {}
    for dimension, label in occurrence.labels:
        grouped.setdefault(dimension, set()).add(label)
    for dimension, labels in grouped.items():
        mentioned = context.get(dimension)
        if not mentioned:
            continue
        positive, negative = mentioned
        if labels & negative:
            return True
        if positive and not labels & positive:
            # Do not treat nested labels as aliases (e.g. revenue vs net
            # revenue). Such an equivalence requires independent evidence.
            return True
    return False


def _ambiguous(occurrence: _Occurrence, context, evidence: _Evidence, text: str) -> bool:
    """The answer names a column family ("CCC", "revenue") shared by several
    labels of a header dimension but not which member this cell is."""
    folded = _fold(text)
    stated_modifiers = dict(_inclusion_modifiers(text))
    for obj, polarity in occurrence.modifiers:
        if obj in stated_modifiers:
            continue
        if any(s is not occurrence and s.prose and s.prose == occurrence.prose
               and s.locator.get("document_id") == occurrence.locator.get("document_id")
               and s.unit == occurrence.unit and not s.values & occurrence.values
               and any(other == obj and mode != polarity for other, mode in s.modifiers)
               for s in evidence.occurrences):
            return True
    if occurrence.locator.get("artifact_type") == "table":
        own_cell = f"t{occurrence.locator.get('table')}:cell{occurrence.locator.get('column')}"
        if any((m := _NUMBER_RE.search(label.split(": ", 1)[-1]))
               and _number_candidates(m.group(0)) & occurrence.values
               for label in context.get(own_cell, (set(), set()))[0]):
            return False  # The explicitly stated condition, not its outcome.
        # Missing dimensions are not resolved by the value itself. Consider
        # only the same canonical table and the *explicitly named* dimensions;
        # correlated axes (e.g. Adult + Adults) may vary together. Require the
        # same measurement header when testing an omitted owner/section.
        own_labels = dict(occurrence.labels)
        for dimension, label in occurrence.labels:
            if dimension in evidence.numeric_headers or context.get(dimension, (set(), set()))[0]:
                continue
            if not (":row" in dimension or ":h" in dimension or dimension.endswith(":section")):
                continue
            for sibling in evidence.occurrences:
                if sibling.locator.get("artifact_type") != "table" or sibling is occurrence:
                    continue
                if (sibling.locator.get("document_id"), sibling.locator.get("table")) != (
                        occurrence.locator.get("document_id"), occurrence.locator.get("table")):
                    continue
                theirs = dict(sibling.labels)
                if theirs.get(dimension) in (None, label) or sibling.values & occurrence.values:
                    continue
                if _conflicts(sibling, context) or sibling.unit != occurrence.unit:
                    continue
                if ":h" not in dimension and any(
                        theirs.get(d) != v for d, v in own_labels.items() if ":h" in d):
                    continue
                # A group header may be omitted, but changing the actual
                # measure (cost vs transit days) is not a competing condition.
                if ":h" in dimension and any(
                        theirs.get(d) != v for d, v in own_labels.items()
                        if ":h" in d and d != dimension):
                    continue
                if ":h" in dimension and not any(
                        ":h" in d and int(d.rsplit(":h", 1)[-1]) > int(dimension.rsplit(":h", 1)[-1])
                        for d in own_labels):
                    # Leaf headers need a named family, as in the existing
                    # family-name check below; don't guess that an unknown
                    # paraphrase asks for every measure in a flat table.
                    continue
                return True
    for dimension, label in occurrence.labels:
        if ":h" not in dimension:
            continue
        labels = evidence.dimensions.get(dimension, set())
        positive = context.get(dimension, (set(), set()))[0]
        if label in positive or len(labels) < 2:
            continue
        own = set(_tokens(label))
        row_key = tuple(item for item in occurrence.labels if ":h" not in item[0]
                        and item[0] not in evidence.numeric_headers)
        for other in labels - {label}:
            if not any(_token_mentioned(token, folded, text) for token in own & set(_tokens(other))):
                continue
            # Only a sibling cell of the same row with a different value
            # makes the family name ambiguous ("MAE 0.08 for both" is fine).
            for sibling in evidence.occurrences:
                if sibling is occurrence or (dimension, other) not in sibling.labels:
                    continue
                if tuple(item for item in sibling.labels if ":h" not in item[0]
                         and item[0] not in evidence.numeric_headers) != row_key:
                    continue
                if not sibling.values & occurrence.values:
                    return True
    # Same owner and column reported under another condition (a section row
    # "Năm 2023", another year's report) with a different value, and the
    # answer names no condition of that kind.
    own = {_dimension_kind(d): label for d, label in occurrence.labels if d not in evidence.numeric_headers}
    own_conditions = {label for d, label in occurrence.labels if d == "fact:cond"}
    if own_conditions and not context.get("fact:cond", (set(), set()))[0]:
        for sibling in evidence.occurrences:
            if sibling is occurrence or sibling.values & occurrence.values:
                continue
            theirs = {_dimension_kind(d): label for d, label in sibling.labels if d not in evidence.numeric_headers}
            their_conditions = {label for d, label in sibling.labels if d == "fact:cond"}
            if (
                their_conditions and their_conditions != own_conditions
                and theirs.get("entity") == own.get("entity")
                and theirs.get("attribute") == own.get("attribute")
            ):
                return True
    for dimension, label in occurrence.labels:
        kind = _dimension_kind(dimension)
        if kind not in {"section", "doc"} or context.get(dimension, (set(), set()))[0]:
            continue
        for sibling in evidence.occurrences:
            if sibling is occurrence or not sibling.labels or sibling.values & occurrence.values:
                continue
            theirs = {_dimension_kind(d): other for d, other in sibling.labels if d not in evidence.numeric_headers}
            if theirs.get(kind) in (None, label):
                continue
            if all(theirs.get(k) == v for k, v in own.items() if k != kind):
                return True
    return False


def _dimension_kind(dimension: str) -> str:
    return dimension.rsplit(":", 1)[-1]


def _specified(occurrence: _Occurrence, context, evidence: _Evidence) -> bool:
    """Row and column of the cell are both named (or unique)."""
    named_row = False
    for dimension, label in occurrence.labels:
        labels = evidence.dimensions.get(dimension, set())
        positive = context.get(dimension, (set(), set()))[0]
        if ":row" in dimension or dimension.endswith(":owner"):
            named_row = named_row or label in positive or len(labels) == 1
        elif ":h" in dimension and len(labels) >= 2 and label not in positive:
            return False
    return named_row


def _merge(*maps) -> dict[str, tuple[set[str], set[str]]]:
    merged: dict[str, tuple[set[str], set[str]]] = {}
    for mapping in maps:
        for dimension, (positive, negative) in mapping.items():
            current = merged.setdefault(dimension, (set(), set()))
            current[0].update(positive)
            current[1].update(negative)
    return merged


_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|(?:[-*+]\s+)?\*\*[^*]+\*\*:?\s*$|(?:[-*+]\s+)?[^|]{1,80}:\s*$)"
)


def _exempt(
    text: str, match: re.Match[str], question_numbers: set[Decimal], table_values: set[Decimal] | None = None,
) -> bool:
    raw = match.group(0)
    start = match.start()
    before = text[max(0, start - 24):start]
    after = text[match.end():match.end() + 3]
    if start >= 2 and text[start - 1] == "-" and text[start - 2].isalpha():
        return True  # identifier such as ResNet-50
    if _REFERENCE_PREFIX_RE.search(re.sub(r"[*_`~]", "", before)):
        return True
    if re.search(r"[A-Za-z][\w.-]*:\s*$", before) and re.match(r"[bBmMkK](?!\w)", after):
        return True  # Model tag (encoder:4b), not a measured byte size.
    if re.search(r"(?i)(?:recall|hit|precision|ndcg|mrr)\s*@\s*(?:\d+/)*$", before):
        return True  # Retrieval cutoff, including Recall@5/10.
    if re.fullmatch(r"\d+\.0", raw) and re.search(r"(?:[A-Z][\w-]*|[A-Za-z]+\d[\w-]*)\s$", before):
        return True  # version of a named model ("Wav2Vec 2.0", "Encoder 2.0")
    if re.search(r"\[\s*$", before) and re.match(r"\s*[\],]", after):
        return True  # citation [12]
    line_start = text.rfind("\n", 0, start) + 1
    if re.fullmatch(r"\s*(?:[-*+]\s+)?#*\s*", text[line_start:start]) and re.match(r"[.)]\s", after):
        return True  # list marker "1. "
    if re.fullmatch(r"\s*\d+[.)]\s*", text):
        return True  # The sentence splitter separated a leading list marker.
    # Dates, small quantities and numbers supplied by the user are factual
    # content too. Only syntactically identified references/identifiers above
    # are outside the quantitative contract.
    return False


def _check_number(
    raw: str, text_for_record: str, context, evidence: _Evidence, result: GroundingResult,
    scope_text: str = "", vietnamese: bool = False, stated: list[frozenset[Decimal]] | None = None,
    unit: str = "", comparator: str = "=", start: int = -1, end: int = -1,
) -> bool:
    """Record unsupported numbers; return True when fully corroborated."""
    values = _number_candidates(raw)
    if not values:
        # Digits that read as no plausible value (OCR runs, identifiers) are no claim.
        return True
    result.checked += 1
    if vietnamese and re.fullmatch(r"[-−]?\d{1,3}(?:\.\d{3})+", raw):
        # Vietnamese text groups thousands with dots: "5.000 mg" is 5000.
        values = frozenset(value for value in values if value == int(value) and abs(value) >= 1000) or values
    decimals = _decimals(raw)
    tolerance = Decimal("0.5") * Decimal(10) ** (-decimals)
    candidates = [o for o in evidence.occurrences
                  if _matches(values, decimals, o.values,
                              percentage_scale=bool(unit == "%" and o.unit == "ratio"),
                              magnitude=bool(re.search(r"(?i)\b(?:decreased?|reduction|reduced?|dropped?|giảm)\b", scope_text or text_for_record)))
                  and _quantity_compatible(o, unit, comparator)]
    # The number in an explicit condition phrase is bound to that column,
    # even when the sentence's main measurement has another header. This
    # changes only this occurrence's context, never the other answer numbers.
    explicit_conditions = {
        d for d, (positive, _) in context.items() if d in evidence.numeric_headers
        and any((m := _NUMBER_RE.search(label.split(": ", 1)[-1]))
                and _number_candidates(m.group(0)) & values for label in positive)
    }
    if explicit_conditions:
        bound = [o for o in candidates if f"t{o.locator.get('table')}:cell{o.locator.get('column')}"
                 in explicit_conditions and not _conflicts(o, {d: v for d, v in context.items() if ":h" not in d})]
        if bound:
            context = {d: (set(p), set(n)) for d, (p, n) in context.items()}
            for o in bound:
                for d, label in o.labels:
                    if ":h" in d:
                        context[d] = ({label}, set())
            candidates = bound
    before = len(result.unsupported)
    consistent: list[_Occurrence] = []
    if not candidates:
        # A derived difference must be the difference of the numbers the
        # sentence itself states, when it states at least two.
        others = [group for group in (stated or []) if group != values]
        if len(others) >= 2:
            differences = {
                difference
                for i, a in enumerate(others) for b in others[i + 1:]
                for x in a for y in b
                if (difference := _rounded(abs(x - y), 6)) is not None
            }
        else:
            differences = evidence.differences
        operands = [o for o in evidence.occurrences
                    if o.locator.get("artifact_type") == "table"
                    and not _conflicts(o, context)
                    and _quantity_compatible(o, unit, "=")]
        derivation = []
        if re.search(r"(?i)\b(?:difference|exceeds?|grew|growth|higher|lower|increase|decrease|drop|by|chênh|hơn|tăng|giảm)\b", text_for_record):
            for i, left in enumerate(operands):
                for right in operands[i + 1:]:
                    same_table = (left.locator.get("document_id"), left.locator.get("table")) == (
                        right.locator.get("document_id"), right.locator.get("table"))
                    same_column = left.locator.get("column") == right.locator.get("column")
                    left_headers = {token for dim, label in left.labels if ":h" in dim for token in _tokens(label) if not token.isdigit()}
                    right_headers = {token for dim, label in right.labels if ":h" in dim for token in _tokens(label) if not token.isdigit()}
                    same_measure_row = left.locator.get("row") == right.locator.get("row") and bool(left_headers & right_headers)
                    if not same_table or not (same_column or same_measure_row):
                        continue
                    if left.unit != right.unit:
                        continue
                    stated_measurements = [a for a in others if any(a & o.values for o in operands)]
                    if len(stated_measurements) >= 2 and not (any(left.values & a for a in stated_measurements) and any(right.values & b for b in stated_measurements)):
                        continue
                    if any(abs(abs(x - y) - abs(v)) <= tolerance for x in left.values for y in right.values for v in values):
                        derivation = [left.locator, right.locator]
                        break
                if derivation:
                    break
        if derivation:
            result.assessments.append(NumberAssessment(
                hashlib.sha256(f"{start}:{end}:{text_for_record}".encode()).hexdigest()[:20],
                raw, text_for_record, start, end, "supported", "located_difference_supported", tuple(derivation), True,
            ))
            return True
        if not any(abs(abs(v) - Decimal(str(d))) <= tolerance for v in values for d in differences):
            result.unsupported.append(NumberMention(raw, text_for_record, "ungrounded", start, end))
        else:
            # A difference without located operands/units is not verified.
            result.unsupported.append(NumberMention(raw, text_for_record, "unverified_derivation", start, end))
        status, reason = "unverifiable", result.unsupported[-1].reason
        result.assessments.append(NumberAssessment(
            hashlib.sha256(f"{start}:{end}:{text_for_record}".encode()).hexdigest()[:20],
            raw, text_for_record, start, end, status, reason,
        ))
        return False
    consistent = [o for o in candidates if not _conflicts(o, context)
                  and not _modifier_conflicts(o, scope_text or text_for_record)]
    if not consistent:
        result.unsupported.append(NumberMention(raw, text_for_record, "context_conflict", start, end))
    scope = scope_text or text_for_record
    clear = [o for o in consistent if not _ambiguous(o, context, evidence, scope)]
    if consistent and not clear:
        result.unsupported.append(NumberMention(raw, text_for_record, "ambiguous", start, end))
    if len(result.unsupported) > before:
        result.assessments.append(NumberAssessment(
            hashlib.sha256(f"{start}:{end}:{text_for_record}".encode()).hexdigest()[:20],
            raw, text_for_record, start, end, "contradicted" if not consistent else "unverifiable",
            result.unsupported[-1].reason,
        ))
        return False
    result.assessments.append(NumberAssessment(
        hashlib.sha256(f"{start}:{end}:{text_for_record}".encode()).hexdigest()[:20],
        raw, text_for_record, start, end, "supported", "source_value_supported",
        tuple(o.locator for o in clear), any(o.labels and _specified(o, context, evidence) for o in clear),
    ))
    result._structured.setdefault(_value_key(raw), []).append(any(o.labels for o in clear))
    return any(o.labels and _specified(o, context, evidence) for o in clear)


def ground_answer_numbers(
    answer: str,
    documents: list[dict],
    *,
    question: str | None = None,
) -> GroundingResult:
    result = GroundingResult()
    text = str(answer or "")
    if not text.strip():
        return result
    evidence = _build_evidence(documents)
    question_numbers = {
        value
        for match in _NUMBER_RE.finditer(str(question or ""))
        for value in _number_candidates(match.group(0))
    }

    # Answer tables: a cell's context is its row labels plus its own column
    # header path in the answer table (plus the section heading above).
    table_lines: set[str] = set()
    heading_mentions: dict[str, tuple[set[str], set[str]]] = {}
    corroboration: dict[str, list[bool]] = {}
    vietnamese = bool(re.search(r"[à-ỹđ]", text.casefold()))
    table_search_start = 0
    for block in _table_blocks(text):
        table_lines.update(block)
        separator_indexes = [i for i, row in enumerate(block) if _SEPARATOR_ROW_RE.match(row)]
        header_rows = [_cells(row) for row in block[:separator_indexes[0]]]
        for index, row in enumerate(block):
            row_offset = text.find(row, table_search_start)
            table_search_start = row_offset + len(row)
            if index <= separator_indexes[0] or index in separator_indexes:
                continue
            cells = _cells(row)
            row_labels = " | ".join(cell for cell in cells if not _is_numeric_cell(cell))
            row_bindings = " | ".join(
                f"{' '.join(h[c] for h in header_rows if c < len(h))}: {cell}"
                for c, cell in enumerate(cells) if _is_numeric_cell(cell))
            cell_search_start = 0
            for column, cell in enumerate(cells):
                cell_offset = row.find(cell, cell_search_start)
                cell_search_start = cell_offset + len(cell)
                if not _NUMBER_RE.search(cell):
                    continue
                header = " ".join(h[column] for h in header_rows if column < len(h))
                context = _merge(_mentions(f"{row_labels} | {header}", evidence), heading_mentions,
                    {d: v for d, v in _mentions(row_bindings, evidence).items() if d in evidence.numeric_headers})
                for match in _NUMBER_RE.finditer(cell):
                    if _exempt(cell, match, question_numbers, evidence.table_values):
                        continue
                    ok = _check_number(
                        match.group(0), row.strip(), context, evidence, result,
                        scope_text=f"{row_labels} | {header}", vietnamese=vietnamese,
                        unit=_quantity_tags(cell, match, header)[0],
                        comparator=_quantity_tags(cell, match, header)[1],
                        start=row_offset + cell_offset + match.start(),
                        end=row_offset + cell_offset + match.end(),
                    )
                    corroboration.setdefault(_value_key(match.group(0)), []).append(ok)

    line_offset = 0
    for physical_line in text.splitlines(keepends=True):
        line = physical_line.rstrip("\r\n")
        current_line_offset = line_offset
        line_offset += len(physical_line)
        if line in table_lines:
            continue
        if _HEADING_RE.match(line):
            mentions = _mentions(line, evidence)
            # A canonical document name may contain a year or copy ordinal.
            # It still changes the section owner; numeric quantities in the
            # heading continue through the ordinary occurrence checks below.
            if not _NUMBER_RE.search(line) or "doc" in mentions:
                heading_mentions = mentions
        sentence_offset = 0
        for sentence in _sentence_parts(line):
            position = line.find(sentence, sentence_offset)
            sentence_offset = position + len(sentence)
            numbers = [m for m in _NUMBER_RE.finditer(sentence) if not _exempt(sentence, m, question_numbers, evidence.table_values)]
            if not numbers:
                continue
            spans = _clause_spans(sentence)
            if _cross_source_reference(sentence):
                links = list(re.finditer(r"(?i)\b(?:with|versus|vs|với|voi)\b",
                                         sentence[numbers[0].end():numbers[-1].start()]))
                if links:
                    cuts = [(m.start() + numbers[0].end(), m.end() + numbers[0].end()) for m in links]
                    expanded = []
                    for first, last in spans:
                        cursor = first
                        for a, b in cuts:
                            if first <= a < last:
                                expanded.append((cursor, a))
                                cursor = b
                        expanded.append((cursor, last))
                    spans = expanded
            # Clauses carrying their own numbers describe those numbers; only
            # number-free clauses ("In Cohort A", "so với Base") are shared.
            shared_text = " | ".join(
                sentence[a:b] for a, b in spans if not any(a <= m.start() < b for m in numbers)
            )
            from app.services.evidence_validator import _cross_metric_reference_prefix

            # A caveat comparing two papers' different metrics ("do not compare
            # CCC 0.63 with miF1 0.81") is not attributed to the section owner.
            section_context = {} if (_cross_metric_reference_prefix(sentence.strip())
                                     or _cross_source_reference(sentence)) else heading_mentions
            condition_context = {
                dimension: mentions for dimension, mentions in _mentions(sentence, evidence).items()
                if dimension not in evidence.numeric_headers and (dimension == "doc" or ":section" in dimension
                or any(_NUMBER_RE.search(label) for label in evidence.dimensions.get(dimension, ()))
                )
            }
            stated = [_number_candidates(m.group(0)) for m in numbers]
            paired = _respectively_pairs(sentence, numbers, evidence)
            prefix_owners = {
                d: mentions for d, mentions in _mentions(sentence[:numbers[0].start()], evidence).items()
                if d.endswith(":owner")
            }
            for index, match in enumerate(numbers):
                start, end = next((s for s in spans if s[0] <= match.start() < s[1]), (0, len(sentence)))
                context = _merge(
                    _mentions(f"{sentence[start:end]} | {shared_text}", evidence), section_context, condition_context,
                )
                for dimension, mentions in prefix_owners.items():
                    if dimension not in context:
                        context[dimension] = mentions
                # Never union numeric conditions across numbered clauses.
                # Prefer the condition in this clause (even when it follows
                # the value); otherwise inherit the last explicit preceding
                # binding. A later clause cannot license an earlier value.
                local_bindings = _numeric_binding_spans(sentence[start:end], evidence)
                prior_bindings = _numeric_binding_spans(sentence[:match.start()], evidence)
                for dimension in evidence.numeric_headers:
                    context.pop(dimension, None)
                    local = local_bindings.get(dimension, [])
                    prior = prior_bindings.get(dimension, [])
                    if local:
                        before_number = [p for p in local if p[0] <= len(_fold(sentence[start:match.start()]))]
                        selected = max(before_number, default=min(local), key=lambda p: p[0])
                    elif prior:
                        selected = max(prior, key=lambda p: p[0])
                    else:
                        continue
                    context[dimension] = ({selected[1]}, set())
                for dimension, label in paired.get(index, {}).items():
                    context[dimension] = ({label}, set())
                ok = _check_number(
                    match.group(0), sentence.strip(), context, evidence, result,
                    scope_text=f"{sentence[start:end]} | {shared_text}", vietnamese=vietnamese, stated=stated,
                    unit=_quantity_tags(sentence, match)[0], comparator=_quantity_tags(sentence, match)[1],
                    start=current_line_offset + position + match.start(), end=current_line_offset + position + match.end(),
                )
                corroboration.setdefault(_value_key(match.group(0)), []).append(ok)
    result.corroborated = {key for key, flags in corroboration.items() if flags and all(flags)}
    failed = {_value_key(mention.value) for mention in result.unsupported}
    result.grounded = {
        key for key, flags in result._structured.items() if key not in failed and flags and all(flags)
    }
    if any(document.get("facts") for document in documents):
        # Facts strengthen source validation; a model-proposed label must not
        # override a contradiction already detectable in the canonical text.
        baseline = ground_answer_numbers(text, [{k: v for k, v in d.items() if k != "facts"}
                                                for d in documents], question=question)
        failed_positions = {(a.start, a.end): a for a in baseline.assessments if a.status != "supported"}
        result.assessments = [failed_positions.get((a.start, a.end), a) for a in result.assessments]
        already = {(m.start, m.end) for m in result.unsupported}
        result.unsupported.extend(m for m in baseline.unsupported if (m.start, m.end) not in already)
        result.grounded &= baseline.grounded
        result.corroborated &= baseline.corroborated
    return result


def _value_key(raw: str) -> str:
    values = sorted(_number_candidates(raw))
    rounded = _rounded(abs(values[0]), 9) if values else None
    return repr(rounded) if rounded is not None else raw


def _respectively_pairs(sentence: str, numbers, evidence: _Evidence) -> dict[int, dict[str, str]]:
    """"A, B and C reached x, y and z, respectively": the i-th named label of a
    dimension owns the i-th number."""
    if not re.search(r"(?<!\w)(?:respectively|lần\s+lượt|tương\s+ứng)(?!\w)", sentence, re.IGNORECASE):
        return {}
    folded, marked = _fold(sentence), _mark_key(sentence)
    measurements = [(i, number) for i, number in enumerate(numbers) if any(
        o.locator.get("artifact_type") == "table" and o.values & _number_candidates(number.group(0))
        for o in evidence.occurrences)] or list(enumerate(numbers))
    first_number = measurements[0][1].start() if measurements else len(sentence)
    pairs: dict[int, dict[str, str]] = {}
    for dimension, labels in evidence.dimensions.items():
        ordered = []
        for label in labels:
            spans = [i for i in _label_spans(label, labels, folded, sentence, marked, evidence) if i < first_number]
            if spans:
                ordered.append((min(spans), label))
        if len(ordered) >= 2 and len(ordered) == len(measurements):
            for (index, _number_match), (_position, label) in zip(measurements, sorted(ordered)):
                pairs.setdefault(index, {})[dimension] = label
    return pairs


def _clause_spans(sentence: str) -> list[tuple[int, int]]:
    # "A and B reach X and Y respectively" pairs across conjunctions.
    if re.search(r"(?<!\w)(?:respectively|lần\s+lượt|tương\s+ứng)(?!\w)", sentence, re.IGNORECASE):
        return [(0, len(sentence))]
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _CLAUSE_SPLIT_RE.finditer(sentence):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(sentence)))
    return spans
