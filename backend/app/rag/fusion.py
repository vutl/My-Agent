"""Deterministic query-variant fusion for first-stage retrieval experiments."""

from __future__ import annotations

from collections import OrderedDict
import re
import unicodedata
from typing import Any


def normalized_query(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    )


def valid_query_variants(
    *,
    original_query: str,
    standalone_query: str,
    required_entities: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return original + rewrite only when the rewrite is safe to fuse.

    A rewrite is rejected when it is empty, identical, or drops every
    meaningful token of an explicit entity.  This is a conservative guard
    against semantic query drift; it does not require the rewrite to preserve
    punctuation or casing.
    """

    original = str(original_query or "").strip()
    standalone = str(standalone_query or "").strip()
    if not original:
        return (standalone,) if standalone else ()
    if not standalone or normalized_query(original) == normalized_query(standalone):
        return (original,)
    rewrite_text = normalized_query(standalone)
    for entity in required_entities:
        tokens = _meaningful_tokens(str(entity))
        if tokens and not any(token in rewrite_text for token in tokens):
            return (original,)
    return (original, standalone)


def fuse_ranked_results(
    ranked_results: list[tuple[str, dict[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse already-ranked result lists using query-aware reciprocal rank.

    Each tuple is ``(query_origin, result_payload)``.  The result payload is
    copied and retains its original lexical/vector/rerank scores.  Fusion only
    changes final ordering and adds auditable ``query_origins`` and component
    ranks; it never invents a new text span.
    """

    if top_k < 1:
        return []
    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    components: dict[str, list[dict[str, Any]]] = {}
    for query_origin, payload in ranked_results:
        identity = _result_identity(payload)
        if not identity:
            continue
        item = merged.get(identity)
        if item is None:
            item = dict(payload)
            item["query_origins"] = []
            item["retrieval_channels"] = list(payload.get("retrieval_channels") or [])
            item["rrf_score"] = 0.0
            merged[identity] = item
            components[identity] = []
        if query_origin not in item["query_origins"]:
            item["query_origins"].append(query_origin)
        for channel in payload.get("retrieval_channels") or []:
            if channel not in item["retrieval_channels"]:
                item["retrieval_channels"].append(channel)
        # The caller supplies one tuple per ranked result.  Compute rank from
        # the stable occurrence within that origin, not from payload metadata.
        rank = _rank_for_identity(ranked_results, query_origin, identity)
        item["rrf_score"] += 1.0 / (rrf_k + rank)
        components[identity].append({"query_origin": query_origin, "rank": rank})

    ordered = []
    for identity, item in merged.items():
        item["query_rrf_components"] = components[identity]
        item["score"] = item["rrf_score"]
        ordered.append(item)
    ordered.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("raw_vector_score") or 0.0),
            str(item.get("chunk_id") or item.get("table_id") or item.get("figure_id") or ""),
        ),
        reverse=True,
    )
    return ordered[:top_k]


def _rank_for_identity(
    ranked_results: list[tuple[str, dict[str, Any]]],
    origin: str,
    identity: str,
) -> int:
    rank = 0
    for query_origin, payload in ranked_results:
        if query_origin != origin:
            continue
        candidate_identity = _result_identity(payload)
        if not candidate_identity:
            continue
        rank += 1
        if candidate_identity == identity:
            return rank
    return rank + 1


def _result_identity(payload: dict[str, Any]) -> str:
    return str(
        payload.get("evidence_id")
        or payload.get("chunk_id")
        or payload.get("table_id")
        or payload.get("figure_id")
        or payload.get("id")
        or ""
    ).strip()


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[\w][\w.-]{1,}", normalized_query(value), flags=re.UNICODE)
        if len(token) >= 2
    )
