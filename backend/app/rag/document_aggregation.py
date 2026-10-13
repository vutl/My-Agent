"""Contracts for bounded document-wide page aggregation.

Page retrieval and answer generation solve different problems.  A ranked
frontier can contain every page candidate we found while a single answer
prompt still misses a predicate repeated across many images.  This module
keeps the map/reduce boundary explicit and deterministic: callers provide the
retrieved page frontier, a model may classify each page, and the reducer only
counts complete, validated observations from that frontier.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class PageAggregationObservation:
    """One model observation tied to a canonical document/page pair."""

    document_id: str
    page_number: int
    matches: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PageAggregationResult:
    """Deterministic reduction over complete page observations."""

    document_id: str
    question: str
    observations: tuple[PageAggregationObservation, ...]
    matched_pages: tuple[int, ...]
    unmatched_pages: tuple[int, ...]
    unobserved_pages: tuple[int, ...]
    complete: bool
    expected_page_counts: tuple[tuple[str, int], ...] = ()
    frontier_complete: bool = True

    @property
    def count(self) -> int:
        return len(self.matched_pages)

    def to_context(self) -> str:
        """Render bounded derived context for the final answer prompt."""

        matched = ", ".join(str(page) for page in self.matched_pages) or "none"
        unmatched = ", ".join(str(page) for page in self.unmatched_pages) or "none"
        unobserved = ", ".join(str(page) for page in self.unobserved_pages) or "none"
        completeness = "complete" if self.complete else "incomplete"
        expected = ", ".join(
            f"{document}: {count}"
            for document, count in self.expected_page_counts
        ) or "unknown"
        frontier = "complete" if self.frontier_complete else "partial"
        return (
            "Page-level aggregation (derived observations, not canonical text):\n"
            f"Question: {self.question}\n"
            f"Document: {self.document_id}\n"
            f"Verification status: {completeness}\n"
            f"Pages matching the requested predicate: {matched}\n"
            f"Pages checked and not matching: {unmatched}\n"
            f"Retrieved pages not successfully checked: {unobserved}\n"
            f"Derived count over matching checked pages: {self.count}\n"
            f"Document page-count expectation: {expected}\n"
            f"Retrieved frontier coverage: {frontier}\n"
            "Use this only for the requested page aggregation. Do not infer a "
            "page outside the retrieved frontier or silently treat an incomplete "
            "frontier as a complete document census."
        )


def page_key(item: dict[str, Any]) -> tuple[str, int] | None:
    """Return the stable page identity used by map/reduce and context joins."""

    document_id = str(item.get("document_id") or "").strip()
    page = item.get("page_number")
    if not document_id or page is None:
        return None
    try:
        return document_id, int(page)
    except (TypeError, ValueError):
        return None


def dedupe_page_frontier(
    pages: list[dict[str, Any]],
    *,
    max_pages: int = 32,
) -> list[dict[str, Any]]:
    """Keep one deterministic record per document/page, preserving rank order."""

    if max_pages < 1:
        return []
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for page in pages:
        key = page_key(page)
        if key is None or key in seen:
            continue
        seen.add(key)
        result.append(page)
        if len(result) >= max_pages:
            break
    return result


def parse_observations(
    raw: str,
    *,
    expected_pages: list[dict[str, Any]],
) -> list[PageAggregationObservation] | None:
    """Parse a strict JSON map response and reject missing/foreign pages.

    The model is not trusted to introduce page IDs.  Every returned page must
    belong to the batch supplied by the caller, and every expected page must
    occur exactly once.  A malformed or partial map is discarded so callers
    can use the ordinary grounded answer path instead of counting guesses.
    """

    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for page in expected_pages:
        key = page_key(page)
        if key is not None:
            expected[key] = page
    if not expected:
        return []
    payload = _extract_json_object(raw)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("pages")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    observations: list[PageAggregationObservation] = []
    seen: set[tuple[str, int]] = set()
    default_document_id = next(iter(expected))[0]
    for row in rows:
        if not isinstance(row, dict):
            return None
        document_id = str(row.get("document_id") or default_document_id).strip()
        try:
            page_number = int(row.get("page"))
        except (TypeError, ValueError):
            return None
        key = (document_id, page_number)
        if key not in expected or key in seen:
            return None
        matches = row.get("matches")
        if not isinstance(matches, bool):
            return None
        reason = " ".join(str(row.get("reason") or "").split())[:240]
        observations.append(
            PageAggregationObservation(
                document_id=document_id,
                page_number=page_number,
                matches=matches,
                reason=reason,
            )
        )
        seen.add(key)
    if seen != set(expected):
        return None
    return observations


def reduce_page_observations(
    pages: list[dict[str, Any]],
    observations: list[PageAggregationObservation],
    *,
    question: str,
    expected_page_counts: dict[str, int] | None = None,
) -> PageAggregationResult:
    """Reduce observations while retaining explicit incompleteness metadata."""

    frontier = dedupe_page_frontier(pages, max_pages=max(len(pages), 1))
    expected_keys = {key for page in frontier if (key := page_key(page)) is not None}
    by_key = {
        (item.document_id, item.page_number): item
        for item in observations
        if (item.document_id, item.page_number) in expected_keys
    }
    ordered_keys = [key for page in frontier if (key := page_key(page)) is not None]
    matched = tuple(key[1] for key in ordered_keys if by_key.get(key) and by_key[key].matches)
    unmatched = tuple(key[1] for key in ordered_keys if by_key.get(key) and not by_key[key].matches)
    unobserved = tuple(key[1] for key in ordered_keys if key not in by_key)
    document_ids = {key[0] for key in ordered_keys}
    document_id = next(iter(document_ids), "") if len(document_ids) == 1 else "multiple"
    expected = {
        str(document): int(count)
        for document, count in (expected_page_counts or {}).items()
        if str(document).strip() and int(count) > 0
    }
    frontier_counts = Counter(key[0] for key in ordered_keys)
    frontier_complete = all(
        frontier_counts.get(document, 0) >= count
        for document, count in expected.items()
    )
    expected_items = tuple(sorted(expected.items()))
    return PageAggregationResult(
        document_id=document_id,
        question=question,
        observations=tuple(observations),
        matched_pages=matched,
        unmatched_pages=unmatched,
        unobserved_pages=unobserved,
        complete=bool(expected_keys) and not unobserved and frontier_complete,
        expected_page_counts=expected_items,
        frontier_complete=frontier_complete,
    )


def _extract_json_object(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
