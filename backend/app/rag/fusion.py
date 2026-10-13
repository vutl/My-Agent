"""Deterministic query-variant fusion for first-stage retrieval experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

from app.rag.evidence_identity import canonical_evidence_id


@dataclass(frozen=True, slots=True)
class QueryVariantPlan:
    """Auditable activation decision for original/rewrite retrieval."""

    variants: tuple[tuple[str, str], ...]
    reason: str


_GENERIC_FOLLOWUP_TOKENS = frozenset(
    {
        "a", "an", "and", "are", "bài", "bảng", "cái", "cho", "chúng", "có", "còn", "của", "does", "đâu", "điều",
        "đó", "đấy", "được", "gì", "hay", "hình", "it", "không", "là", "lại", "mình", "này", "nhé", "nó", "nói", "nữa",
        "paper", "please", "report", "rồi", "sao", "table", "that", "the", "there", "this", "thế", "to", "và", "vậy",
        "về", "với", "what", "which", "with", "xem", "you", "figure", "illustrate", "minh", "hoạ", "ấy",
    }
)


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

    return tuple(value for _origin, value in plan_query_variants(
        original_query=original_query,
        standalone_query=standalone_query,
        required_entities=required_entities,
    ).variants)


def plan_query_variants(
    *,
    original_query: str,
    standalone_query: str,
    required_entities: list[str] | tuple[str, ...] = (),
) -> QueryVariantPlan:
    """Choose safe query variants without treating short text as low quality.

    A raw branch is useful only when it preserves a meaningful exact token that
    the standalone rewrite changed or omitted.  Deictic follow-ups ("this
    table", "bảng đó") do not get a second dense search merely because their
    wording differs from an enriched, focused query.
    """

    original = str(original_query or "").strip()
    standalone = str(standalone_query or "").strip()
    if not original:
        return QueryVariantPlan((("rewrite", standalone),) if standalone else (), "missing_original")
    if not standalone or normalized_query(original) == normalized_query(standalone):
        return QueryVariantPlan((("original", original),), "missing_or_equivalent_rewrite")
    rewrite_text = normalized_query(standalone)
    for entity in required_entities:
        tokens = _meaningful_tokens(str(entity))
        if tokens and not any(token in rewrite_text for token in tokens):
            return QueryVariantPlan((("original", original),), "rewrite_drops_required_entity")
    if not _raw_adds_retrieval_signal(original, standalone):
        return QueryVariantPlan((("rewrite", standalone),), "raw_has_no_distinct_signal")
    return QueryVariantPlan((("original", original), ("rewrite", standalone)), "dual_distinct_signal")


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
        for obligation_id in payload.get("evidence_obligation_ids") or []:
            existing_obligations = item.setdefault("evidence_obligation_ids", [])
            if obligation_id not in existing_obligations:
                existing_obligations.append(obligation_id)
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
            # LanceDB exposes a distance: lower is closer.  RRF is the
            # primary score, but ties must not silently prefer the worse dense
            # neighbour simply because the generic sort is descending.
            -float(item["raw_vector_score"])
            if item.get("raw_vector_score") is not None
            else float("-inf"),
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
    return canonical_evidence_id(payload)


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[\w][\w.-]{1,}", normalized_query(value), flags=re.UNICODE)
        if len(token) >= 2
    )


def _raw_adds_retrieval_signal(original: str, standalone: str) -> bool:
    """Whether raw wording adds exact retrieval anchors absent from rewrite.

    This intentionally uses lexical preservation, not a language-specific
    question template or a word-count threshold.  Numbers, acronyms and
    punctuation-bearing identifiers are meaningful even when tokenization
    would otherwise normalize them away.
    """

    normalized_standalone = normalized_query(standalone)
    standalone_tokens = set(re.findall(r"[\w][\w.-]*", normalized_standalone, flags=re.UNICODE))
    for token in re.findall(r"[\w][\w._/-]*", original, flags=re.UNICODE):
        normalized = normalized_query(token)
        if not normalized or normalized in normalized_standalone:
            continue
        if any(char.isdigit() for char in token) or "_" in token or "/" in token or "-" in token:
            return True
        if token.isupper() and len(token) >= 2:
            return True
        pieces = [piece for piece in re.findall(r"[\w][\w.-]*", normalized, flags=re.UNICODE) if len(piece) >= 2]
        if any(piece not in _GENERIC_FOLLOWUP_TOKENS and piece not in standalone_tokens for piece in pieces):
            return True
    return False
