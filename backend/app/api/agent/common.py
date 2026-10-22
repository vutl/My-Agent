import logging
import time
from typing import Annotated, Any


# Page/document-wide aggregation is deliberately bounded.  The retrieval
# projection can advertise more than the legacy 32-page frontier, but the
# map/reduce stage must use the same bounded ceiling so medium documents are
# not silently truncated before visual verification.
_DEFAULT_PAGE_AGGREGATION_FRONTIER = 32
_MAX_PAGE_AGGREGATION_FRONTIER = 64


def _page_aggregation_frontier_limit(pages: list[dict[str, Any]]) -> int:
    """Return a bounded page frontier limit derived from indexed metadata.

    ``document_page_count`` is only a sizing hint; the hard ceiling remains
    64 pages and the actual frontier is still deduplicated by document/page.
    Missing or malformed counts retain the historical 32-page default.
    """

    expected_counts: list[int] = []
    for page in pages:
        try:
            count = int(page.get("document_page_count") or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            expected_counts.append(count)
    advertised = max(expected_counts, default=_DEFAULT_PAGE_AGGREGATION_FRONTIER)
    return min(
        _MAX_PAGE_AGGREGATION_FRONTIER,
        max(_DEFAULT_PAGE_AGGREGATION_FRONTIER, advertised),
    )
# Bump whenever routing/composition policy changes.  Cached documents are a
# bounded prompt projection, so reusing an entry produced by an older context
# policy can silently discard evidence even though the underlying index did
# not change.
RETRIEVAL_CACHE_VERSION = "v57"
logger = logging.getLogger(__package__)


FULL_DOCUMENT_MAX_DOCUMENTS = 3
FULL_DOCUMENT_MAX_TOTAL_CHARS = 220_000


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _ollama_metrics(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    keys = [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ]
    metrics = {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
    eval_count = metrics.get("eval_count")
    eval_duration = metrics.get("eval_duration")
    if eval_count and eval_duration:
        metrics["tokens_per_second"] = round(float(eval_count) / (float(eval_duration) / 1_000_000_000), 2)
    return metrics

