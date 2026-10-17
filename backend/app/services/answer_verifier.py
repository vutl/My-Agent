"""Lightweight, non-blocking answer verification.

Replaces the buffer-then-regex gate on quantitative answers. Verification runs
after the answer has already streamed, and never by itself rewrites or blocks
it. Two levels:

1. ``find_unsourced_numbers`` -- deterministic. A number in the answer must
   occur in the evidence it could have come from. Reuses the number parsing in
   ``numeric_grounding`` (decimal commas, thousands groups, percent). No
   metric/owner/condition inference.
2. ``verify_with_model`` -- one bounded LLM call, only when level 1 flags a
   number. It returns corrections for the flagged sentences only.

Neither level decides on its own that a model-written answer is wrong. The
caller decides what to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import json
import re
from typing import Any

from app.services.numeric_grounding import _NUMBER_RE, _number_candidates

# A number inside a model citation or a list marker is not a claim.
_CITATION_RE = re.compile(r"\[S\d+\]|\[\d+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SMALL_COUNT_MAX = 12  # bare integers up to this are counts, not measurements
_LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}[.)]\s")
# Digits joined to a name by a hyphen name a model or version ("ResNet-18",
# "GPT-3.5", "COVID-19"); they are not measurements.
_IDENTIFIER_NUMBER_RE = re.compile(r"(?<![\w-])[A-Za-z][\w.]*(?:-\d+(?:\.\d+)?[A-Za-z]*)+")


@dataclass
class UnsourcedNumber:
    value: str
    sentence: str
    context: str = ""


@dataclass
class VerificationReport:
    checked: int = 0
    flagged: list[UnsourcedNumber] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.flagged


def _source_text(document: dict[str, Any]) -> str:
    """The text the answer saw: the expanded excerpt (neighbouring rows,
    sibling tables) when present, else the chunk itself."""
    content = str(document.get("content") or "")
    expanded = str(document.get("expanded_content") or "")
    if not expanded:
        return content
    return expanded if content.strip()[:200] in expanded else f"{content}\n{expanded}"


def _evidence_values(documents: list[dict[str, Any]]) -> set[float]:
    values: set[float] = set()
    for document in documents:
        text = "\n".join(
            str(part)
            for part in (
                document.get("caption"),
                document.get("content"),
                document.get("expanded_content"),
            )
            if part
        )
        for match in _NUMBER_RE.finditer(text):
            values.update(_number_candidates(match.group(0)))
    return values


def _is_claim_number(raw: str) -> bool:
    values = _number_candidates(raw)
    if not values:
        return False
    if "." in raw or "," in raw or "%" in raw:
        return True
    # A bare small integer is a count ("4 steps", "2 options"), not a measurement.
    return max(abs(v) for v in values) > _SMALL_COUNT_MAX


def find_unsourced_numbers(answer: str, documents: list[dict[str, Any]]) -> VerificationReport:
    """Flag numbers in the answer that appear nowhere in the evidence.

    Tolerances follow the printed precision of the answer's number, so a
    rounded "81%" matches an evidence "81.2" only when the rounding is exact.
    Years and table/figure references are not claims.
    """
    report = VerificationReport()
    evidence = _evidence_values(documents)
    text = _CITATION_RE.sub(" ", str(answer or ""))
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        marker = _LIST_MARKER_RE.match(sentence)
        for match in _NUMBER_RE.finditer(sentence):
            if marker is not None and match.start() < marker.end():
                continue  # "4. Heading" numbers the list item, it is not a claim
            raw = match.group(0)
            if _inside_identifier(sentence, match.start()):
                continue
            before = sentence[max(0, match.start() - 14):match.start()].casefold()
            if re.search(r"(?:table|figure|fig\.?|bảng|hình|biểu đồ|năm|year|quý|q)\s*$", before):
                continue
            if re.fullmatch(r"(?:19|20)\d\d", raw):
                continue
            if not _is_claim_number(raw):
                continue
            report.checked += 1
            if not _matches_any(raw, evidence):
                report.flagged.append(UnsourcedNumber(value=raw, sentence=sentence[:300]))
    return report


def _inside_identifier(text: str, start: int) -> bool:
    return any(
        item.start() < start < item.end() for item in _IDENTIFIER_NUMBER_RE.finditer(text)
    )


def _claim_numbers(text: str) -> list[str]:
    """Numbers in a line that state a value (not names, years or references)."""
    cleaned = _IDENTIFIER_NUMBER_RE.sub(" ", _REFERENCE_NUMBER_RE.sub(" ", _CITATION_RE.sub(" ", text)))
    return [
        match.group(0)
        for match in _NUMBER_RE.finditer(cleaned)
        if not re.fullmatch(r"(?:19|20)\d\d", match.group(0)) and _is_claim_number(match.group(0))
    ]


def _matches_any(raw: str, evidence: set[float]) -> bool:
    values = _number_candidates(raw)
    if not values:
        return True
    fraction = re.search(r"[.,](\d+)$", raw)
    decimals = len(fraction.group(1)) if fraction else 0
    tolerance = Decimal("0.5") * (Decimal(10) ** (-decimals))
    hundred = Decimal(100)
    for value in values:
        for candidate in evidence:
            for scaled in (candidate, candidate * hundred, candidate / hundred):
                if abs(abs(value) - abs(scaled)) <= tolerance:
                    return True
    return False


_VERIFY_PROMPT = """You check an answer's numeric claims against source excerpts.
Check each of these answer lines; a number may be unsupported, or attributed to
the wrong model, dataset, condition, period, row or document:

{flagged}

Sources (S1..Sn). Every passage starts with a label [S<n> · <document>]; a passage
belongs to the document in its nearest preceding label. Text in parentheses under a
listed line (its heading, table header) is context, not part of the line: copy only the line.
{sources}

Return JSON only: {{"corrections": [{{"sentence": "<flagged line copied exactly>", "action": "keep"|"rewrite"|"remove", "rewritten": "<line>"|null}}]}}.
- keep: every number in the line is supported by a source for the same entity and condition.
  A value correctly computed from source values (difference, ratio, percentage, average,
  rounding) is supported; keep it. Do not add details to a correct line.
  Recompute every computed value exactly; a wrong result is not supported.
- A line is not supported when the sources give several values for its entity (years,
  periods, splits, conditions, models) and the line does not say which one it means:
  rewrite it to name that condition, keeping the number if it is right.
- Attribution: a number must appear in a block of the document, organisation, author,
  edition or period that the line, its heading or its row names. A number found only in
  another document's block is misattributed: rewrite the line with the named document's
  value. If the named document gives no value, attribute the number to the document that
  does give it. Never move a number to a document that does not state it.
- A listed line that compares or concludes (same, agree, match, higher, differs) from
  values you rewrite must be rewritten to match them.
- rewrite: a source supports different values or a different attribution; return the corrected
  line in the same format (a Markdown table row stays a row with the same number of cells),
  using only numbers that appear in the sources or are correctly computed from them.
- remove: no source supports the line's numbers.
Never invent a number. Keep the answer's language and formatting."""


_LIST_MARKER_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)+")
MAX_VERIFY_UNITS = 24
MAX_VERIFY_DOCUMENT_CHARS = 12_000
MAX_VERIFY_SOURCE_CHARS = 40_000


_SUBJECT_PASSAGE_CHARS = 800
_SUBJECT_RADIUS = 300
_SUBJECT_PASSAGES = 3
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_LEAD_RE = re.compile(r"^\s*(?:[-*+]\s+)?\*\*(.+?)\*\*\s*:?\s*$")
MAX_UNIT_CONTEXT_CHARS = 300


def answer_unit_context(answer: str, unit: str) -> str:
    """Where a checked line sits in the answer: the heading or bold lead line
    above it and, for a table row, the table's header row. The line alone does
    not say which document, entity or column it belongs to."""
    lines = str(answer or "").splitlines()
    unit = unit.strip()
    index = next((i for i, line in enumerate(lines) if line.strip() == unit), None)
    if index is None:
        index = next((i for i, line in enumerate(lines) if unit and unit in line), None)
    if index is None:
        return ""
    parts: list[str] = []
    for line in reversed(lines[:index]):
        match = _HEADING_RE.match(line) or _BOLD_LEAD_RE.match(line)
        if match:
            parts.append(f"section: {match.group(1).strip()}")
            break
    if lines[index].count("|") >= 2:
        start = index
        while start > 0 and lines[start - 1].count("|") >= 2:
            start -= 1
        if start < index:
            parts.append(f"table header: {lines[start].strip()}")
    return "; ".join(parts)[:MAX_UNIT_CONTEXT_CHARS]


def _source_label(document: dict[str, Any]) -> str:
    labels: list[str] = []
    for key in ("filename", "document_title", "title", "caption"):
        value = " ".join(str(document.get(key) or "").split())
        if value and value not in labels:
            labels.append(value)
    return " | ".join(labels)


def _term_set(text: str) -> set[str]:
    from app.services.rag_service import _search_words

    words = _search_words(text)
    return set(words) | {f"{first} {second}" for first, second in zip(words, words[1:])}


def _subject_terms(lines: list[str], question: str = "") -> set[str]:
    """Words and adjacent word pairs of the checked lines and the question:
    what the lines are about, apart from their numbers."""
    from app.services.rag_service import _search_words

    terms: set[str] = set()
    for text in [*lines, question]:
        words = [
            word for word in _search_words(_CITATION_RE.sub(" ", str(text or "")))
            if len(word) >= 3 and not re.search(r"\d", word)
        ]
        terms.update(words)
        terms.update(f"{first} {second}" for first, second in zip(words, words[1:]))
    return terms


def _passages(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n]+", text):
        start = match.start()
        while start < match.end():
            end = min(match.end(), start + _SUBJECT_PASSAGE_CHARS)
            spans.append((start, end))
            start = end
    return spans


def _document_windows(
    text: str, targets: set[Decimal], terms: set[str], label: str, budget: int, radius: int = 600,
) -> str:
    """Passages of a whole document worth showing: around the numbers being
    checked, and the passages that talk about the checked lines' subject (they
    hold the document's own value when a line printed another document's)."""
    windows: list[list[Any]] = []  # start, end
    for match in _NUMBER_RE.finditer(text):
        if _number_candidates(match.group(0)) & targets:
            windows.append([max(0, match.start() - radius), min(len(text), match.end() + radius)])
    passages = _passages(text)
    term_sets = [_term_set(text[start:end]) for start, end in passages]
    total = len(passages)
    informative = set()
    if total:
        frequency = {term: sum(1 for words in term_sets if term in words) for term in terms}
        informative = {term for term, count in frequency.items() if 0 < count <= max(1, total * 0.4)}
    scored = sorted(
        ((len(informative & words), index) for index, words in enumerate(term_sets)),
        key=lambda item: (-item[0], item[1]),
    )
    for score, index in scored[:_SUBJECT_PASSAGES]:
        if score < 2:
            break
        start, end = passages[index]
        windows.append([max(0, start - _SUBJECT_RADIUS), min(len(text), end + _SUBJECT_RADIUS)])
    merged: list[list[Any]] = []
    for window in sorted(windows):
        if merged and window[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], window[1])
        else:
            merged.append(list(window))
    scored_windows = []
    for start, end in merged:
        piece = text[start:end]
        found: set[Decimal] = set()
        for match in _NUMBER_RE.finditer(piece):
            found.update(_number_candidates(match.group(0)) & targets)
        subject = len(informative & _term_set(piece))
        scored_windows.append((2 * len(found) + subject, start, end))
    chosen: list[tuple[int, int]] = []
    remaining = budget
    for _score, start, end in sorted(scored_windows, key=lambda item: (-item[0], item[1])):
        size = end - start + len(label) + 5
        if size > remaining:
            if chosen:
                continue
            end = start + max(remaining - len(label) - 5, 0)  # a window is better than nothing
            size = remaining
        chosen.append((start, end))
        remaining -= size
    return "\n…\n".join(f"{label}\n{text[start:end]}" for start, end in sorted(chosen))


def build_verify_prompt(
    flagged: list[UnsourcedNumber], documents: list[dict[str, Any]], question: str = "",
) -> str:
    # The verifier must see what the answer saw: a source cut short hides the
    # later rows of a table and makes correct rows look unsupported.
    # Only stated values: windows around every year a long document mentions
    # would fill the budget before the rows that matter.
    targets: set[Decimal] = set()
    for item in flagged[:MAX_VERIFY_UNITS]:
        for number in _claim_numbers(item.sentence) or ([item.value] if item.value else []):
            targets.update(_number_candidates(number))

    def _hits(document: dict[str, Any]) -> int:
        found: set[Decimal] = set()
        for match in _NUMBER_RE.finditer(_source_text(document)):
            found.update(_number_candidates(match.group(0)))
        return len(found & targets)

    # Whole documents first (they contribute only windows around the checked
    # numbers), then sources holding the checked numbers, so the budget keeps
    # them. Ranking happens before the cap: whole documents come after the
    # excerpts and were cut off when the cap came first.
    checked = flagged[:MAX_VERIFY_UNITS]
    terms = _subject_terms([item.sentence for item in checked], question)

    def _subject_score(document: dict[str, Any]) -> int:
        return len(terms & _term_set(f"{_source_label(document)}\n{_source_text(document)[:60_000]}"))

    ranked = sorted(
        documents,
        key=lambda item: (not item.get("full_document"), -_hits(item), -_subject_score(item)),
    )[:12]
    blocks: list[str] = []
    remaining = MAX_VERIFY_SOURCE_CHARS
    for index, document in enumerate(ranked, start=1):
        text = _source_text(document)
        label = _source_label(document)
        budget = min(MAX_VERIFY_DOCUMENT_CHARS, max(remaining, 0))
        if document.get("full_document"):
            content = _document_windows(text, targets, terms, f"[S{index} · {label}]", budget)
        else:
            content = text[:budget]
        remaining -= len(content)
        blocks.append(f"[S{index}] {label}\n{content}")
    sources = "\n".join(blocks)
    listing = "\n".join(
        f"- {item.sentence}" + (f"\n  ({item.context})" if item.context else "") for item in checked
    )
    if question.strip():
        listing += f"\n\nQuestion: {question.strip()[:500]}"
    return _VERIFY_PROMPT.format(flagged=listing, sources=sources)


def apply_corrections(answer: str, corrections: list[dict[str, Any]]) -> str:
    """Replace or drop only the flagged sentences; everything else is kept.

    A verdict that would delete most of the checked lines says more about the
    check (missing sources) than about the answer; it is ignored.
    """
    removals = sum(1 for correction in corrections if correction.get("action") == "remove")
    if removals >= 3 and removals * 2 > len(corrections):
        return answer
    result = answer
    for correction in corrections:
        sentence = str(correction.get("sentence") or "").strip()
        # The prompt lists units as "- unit"; the model sometimes echoes that.
        if sentence and sentence not in result:
            sentence = _LIST_MARKER_RE.sub("", sentence, count=1).strip()
        if len(sentence) < 8 or sum(char.isalpha() for char in sentence) < 3:
            continue
        # Replace the unit only where it stands as a whole sentence/line, never
        # inside a longer token ("2." inside "Wav2Vec 2.0").
        match = re.search(rf"(?<![\w.]){re.escape(sentence)}(?![\w])", result)
        if match is None:
            continue
        action = correction.get("action")
        if action == "rewrite" and str(correction.get("rewritten") or "").strip():
            replacement = str(correction["rewritten"]).strip()
        elif action == "remove":
            replacement = ""
        else:
            continue
        result = result[: match.start()] + replacement + result[match.end():]
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def keep_sourced_removals(
    corrections: list[dict[str, Any]], documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop "remove" verdicts on lines whose every claim number occurs in the
    sources. The verifier removes a line when no source supports it; when the
    values are there, the verdict comes from a source it did not see in full
    (a misattribution is a "rewrite")."""
    evidence = _evidence_values(documents)
    kept = []
    for correction in corrections:
        numbers = _claim_numbers(str(correction.get("sentence") or ""))
        sourced = not numbers or all(_matches_any(number, evidence) for number in numbers)
        action = correction.get("action")
        # A rewrite that only drops sourced values ("not established by the
        # sources") is a removal too.
        if action == "rewrite" and not _claim_numbers(str(correction.get("rewritten") or "")):
            action = "remove"
        if action == "remove" and sourced:
            continue
        kept.append(correction)
    return kept


def parse_corrections(payload: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", str(payload or ""), re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return []
    items = data.get("corrections") if isinstance(data, dict) else None
    return [item for item in items or [] if isinstance(item, dict)]


def flagged_answer_units(answer: str, values: list[str]) -> list[str]:
    """Verbatim answer lines (table rows) or sentences that print one of the
    flagged numeric ``values``; these are the only units the model may change."""
    targets: set[Decimal] = set()
    for value in values:
        targets.update(_number_candidates(str(value)))
    if not targets:
        return []
    units: list[str] = []
    in_code = False
    for line in str(answer or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        # Diagrams and code are not claims; "Wav2Vec 2.0 ->" must stay intact.
        if not stripped or in_code:
            continue
        # A list marker ("1.", "-") is not a sentence of its own.
        body = _LIST_MARKER_RE.sub("", stripped, count=1)
        pieces = [stripped] if stripped.count("|") >= 2 else [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()
        ]
        for piece in pieces:
            if len(piece) < 8 or sum(char.isalpha() for char in piece) < 3:
                continue
            numbers: set[Decimal] = set()
            for match in _NUMBER_RE.finditer(piece):
                numbers.update(_number_candidates(match.group(0)))
            if numbers & targets and piece not in units:
                units.append(piece)
    return units


_REFERENCE_NUMBER_RE = re.compile(
    r"\b(?:table|tab\.|figure|fig\.|eq\.|equation|section|sec\.|bảng|hình|mục|phương trình)"
    r"\s*\(?\d+(?:\.\d+)*\)?",
    re.IGNORECASE,
)


def numeric_answer_units(answer: str) -> list[str]:
    """Every answer line (table row) or sentence that states a value: not only
    a year, a small count, a model name or a table/figure/section reference."""
    units = flagged_answer_units(answer, _claim_numbers(str(answer or "")))
    return [unit for unit in units if _claim_numbers(unit)]
