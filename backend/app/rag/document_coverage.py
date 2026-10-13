"""Generic policies for document-wide retrieval questions.

The ordinary retrieval path is intentionally compact: a handful of highly
ranked chunks is the right trade-off for a local fact question.  Enumeration
and aggregation questions are different.  They need bounded coverage across
pages (or other structural units), otherwise a document with many relevant
pages is silently reduced to the first few near-duplicate chunks.

This module only classifies the *shape* of a request.  It never names a paper,
reads qrels, or infers an answer.  The limits are safety budgets, not a
retrieval heuristic and can be configured by callers later.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class DocumentCoveragePolicy:
    """Bounded policy for a document-wide enumeration/aggregation request."""

    enabled: bool = False
    reason: str = ""
    unit: str = "page"
    # Coverage requests need a wider frontier than ordinary fact lookup. The
    # budgets remain bounded (and are deliberately separate from the normal
    # prompt path) so a long document can contribute page evidence without
    # allowing an unbounded prompt or image upload.
    page_limit: int = 32
    max_sources: int = 32
    max_chars: int = 20_000
    max_chars_per_source: int = 420
    max_images: int = 16
    # A page-census query may inspect more pages than can fit in the final
    # answer context.  This is a bounded map/reduce frontier, not an
    # instruction to place an entire arbitrary PDF in one prompt.
    aggregation_page_limit: int = 64


# Keep markers structural and language-paired.  A single word such as
# ``all``/``toàn bộ`` is intentionally insufficient: it is common in a normal
# comparison or table question.  The detector needs an aggregation marker plus
# a page/document-wide marker, or an explicit count/list request.
_PAGE_MARKERS = (
    "page",
    "pages",
    "trang",
    "trang nao",
    "trang nào",
)
_DOCUMENT_WIDE_MARKERS = (
    "across the document",
    "throughout the document",
    "in the document",
    "in this document",
    "throughout the paper",
    "across the paper",
    "across all pages",
    "toan bo tai lieu",
    "toàn bộ tài liệu",
    "trong ca bai",
    "trong cả bài",
    "xuyen suot tai lieu",
    "xuyên suốt tài liệu",
)
_AGGREGATION_MARKERS = (
    "how many",
    "which pages",
    "list all",
    "list the",
    "count",
    "enumerate",
    "all occurrences",
    "bao nhieu",
    "bao nhiêu",
    "nhung trang",
    "những trang",
    "liet ke",
    "liệt kê",
    "dem",
    "đếm",
    "tat ca",
    "tất cả",
    "toan bo",
    "toàn bộ",
)
_PAGE_COUNT_RE = re.compile(
    r"\b(?:how\s+many|which|list|count|bao\s+nhiêu|những|nhung|liệt\s+kê|liet\s+ke|đếm|dem)"
    r"(?:\s+\S+){0,8}\s+(?:pages?|trang)\b",
    flags=re.IGNORECASE,
)
_RANKED_REQUEST_RE = re.compile(
    r"\b(?:top|first|last|bottom)\s*\d+\b",
    flags=re.IGNORECASE,
)
_PAGE_SUPERLATIVE_RE = re.compile(
    r"\b(?:slowest|fastest|highest|lowest|largest|smallest|maximum|minimum|"
    r"greatest|least|cham nhat|nhanh nhat|cao nhat|thap nhat|lon nhat|"
    r"nho nhat|toi da|toi thieu)\b",
    flags=re.IGNORECASE,
)


def detect_document_coverage(query: str) -> DocumentCoveragePolicy:
    """Classify whether ``query`` needs bounded page-level coverage.

    The detector is deliberately conservative.  It returns disabled for a
    normal single-fact request and for a generic table inventory, which has its
    own canonical artifact path.  It is safe to call on every user query.
    """

    normalized = _normalize(query)
    if not normalized:
        return DocumentCoveragePolicy()

    has_page = any(marker in normalized for marker in _PAGE_MARKERS)
    has_wide = any(marker in normalized for marker in _DOCUMENT_WIDE_MARKERS)
    has_aggregate = any(marker in normalized for marker in _AGGREGATION_MARKERS)
    explicit_page_count = bool(_PAGE_COUNT_RE.search(normalized))
    has_ranked_request = bool(_RANKED_REQUEST_RE.search(normalized))
    has_page_superlative = bool(_PAGE_SUPERLATIVE_RE.search(normalized))

    # Explicit page aggregation is strongest.  A document-wide request only
    # activates when it also asks to count/list/identify something; this avoids
    # widening ordinary questions such as "what is the paper about?".
    if (
        (has_page and (has_aggregate or has_ranked_request))
        or (has_page and has_page_superlative)
        or explicit_page_count
        or (has_wide and (has_aggregate or has_ranked_request))
    ):
        if has_page_superlative and has_page:
            reason = "page_superlative"
        else:
            reason = "page_aggregation" if has_page or explicit_page_count else "document_wide_aggregation"
        return DocumentCoveragePolicy(enabled=True, reason=reason)
    return DocumentCoveragePolicy()


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(text.replace("đ", "d").split())
