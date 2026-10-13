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


@dataclass
class UnsourcedNumber:
    value: str
    sentence: str


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
the wrong model, dataset, condition, period or row:

{flagged}

Sources (S1..Sn):
{sources}

Return JSON only: {{"corrections": [{{"sentence": "<flagged line copied exactly>", "action": "keep"|"rewrite"|"remove", "rewritten": "<line>"|null}}]}}.
- keep: every number in the line is supported by a source for the same entity and condition.
  A value correctly computed from source values (difference, ratio, percentage, average,
  rounding) is supported; keep it. Do not add details to a correct line.
  Recompute every computed value exactly; a wrong result is not supported.
- A line is not supported when the sources give several values for its entity (years,
  periods, splits, conditions, models) and the line does not say which one it means:
  rewrite it to name that condition, keeping the number if it is right.
- rewrite: a source supports different values or a different attribution; return the corrected
  line in the same format (a Markdown table row stays a row with the same number of cells),
  using only numbers that appear in the sources or are correctly computed from them.
- remove: no source supports the line's numbers.
Never invent a number. Keep the answer's language and formatting."""


MAX_VERIFY_UNITS = 24
MAX_VERIFY_DOCUMENT_CHARS = 12_000
MAX_VERIFY_SOURCE_CHARS = 40_000


def build_verify_prompt(flagged: list[UnsourcedNumber], documents: list[dict[str, Any]]) -> str:
    # The verifier must see what the answer saw: a source cut short hides the
    # later rows of a table and makes correct rows look unsupported.
    targets: set[Decimal] = set()
    for item in flagged[:MAX_VERIFY_UNITS]:
        for match in _NUMBER_RE.finditer(item.sentence):
            targets.update(_number_candidates(match.group(0)))

    def _hits(document: dict[str, Any]) -> int:
        found: set[Decimal] = set()
        for match in _NUMBER_RE.finditer(_source_text(document)):
            found.update(_number_candidates(match.group(0)))
        return len(found & targets)

    # Sources holding the checked numbers first, so the budget keeps them.
    ranked = sorted(documents[:12], key=_hits, reverse=True)
    blocks: list[str] = []
    remaining = MAX_VERIFY_SOURCE_CHARS
    for index, document in enumerate(ranked, start=1):
        content = _source_text(document)[: min(MAX_VERIFY_DOCUMENT_CHARS, max(remaining, 0))]
        remaining -= len(content)
        blocks.append(
            f"[S{index}] {document.get('filename') or ''} {document.get('caption') or ''}\n{content}"
        )
    sources = "\n".join(blocks)
    listing = "\n".join(f"- {item.sentence}" for item in flagged[:MAX_VERIFY_UNITS])
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
        if not sentence or sentence not in result:
            continue
        action = correction.get("action")
        if action == "rewrite" and str(correction.get("rewritten") or "").strip():
            result = result.replace(sentence, str(correction["rewritten"]).strip(), 1)
        elif action == "remove":
            result = result.replace(sentence, "", 1)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


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
    for line in str(answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pieces = [stripped] if stripped.count("|") >= 2 else [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()
        ]
        for piece in pieces:
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
    """Every answer line (table row) or sentence that prints a number other
    than a table/figure/equation/section reference."""
    values = [
        match.group(0)
        for match in _NUMBER_RE.finditer(_REFERENCE_NUMBER_RE.sub(" ", str(answer or "")))
    ]
    units = flagged_answer_units(answer, values)
    return [unit for unit in units if _NUMBER_RE.search(_REFERENCE_NUMBER_RE.sub(" ", unit))]
