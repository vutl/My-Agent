from collections import Counter, defaultdict
import asyncio
import hashlib
import json
import re
import time
from typing import Annotated, Any
from app.core.config import Settings, get_settings
from app.rag.context import (
    _table_query_match_score,
    compose_retrieval_context,
    merge_page_coverage_sources,
    table_result_role,
)
from app.rag.document_coverage import DocumentCoveragePolicy, detect_document_coverage
from app.rag.figure_caption import (
    best_figure_caption,
    figure_relevance_score,
    requested_figure_number,
)
from app.rag.figure_quality import extract_figure_label
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.rag.evidence_identity import canonical_artifact_type
from app.rag.fusion import plan_query_variants
from app.rag.paper_facets import extract_query_facets, facet_query_terms, requested_paper_facets
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.query_rewrite_service import (
    QueryRewriteResult,
    QueryRewriteService,
    _classify_answer_depth,
    _classify_answer_intent,
    _explicit_document_target_entities,
    _is_named_entity,
    _looks_like_topic_switch,
    _query_named_entities,
    enrich_retrieval_query,
    format_recent_conversation,
    has_result_table_intent,
    has_visual_intent,
    requires_visual_pixels,
    wants_single_figure,
)
from app.services.rag_service import (
    RagService,
    caption_identifies_figure,
    requested_visual_page_numbers,
    visual_page_reservation_count,
)
from app.services.retrieval_agent_service import (
    RetrievalBranch,
    SecondRetrievalPlan,
    plan_second_retrieval_pass,
    plan_retrieval_decomposition,
    smart_retrieval_enabled,
)
from . import deps
from . import retrieval as retrieval_module
from .common import FULL_DOCUMENT_MAX_DOCUMENTS
from .common import FULL_DOCUMENT_MAX_TOTAL_CHARS
from .common import RETRIEVAL_CACHE_VERSION
from .common import _elapsed_ms
from .common import logger
from .deps import _VISUAL_PAGE_PROVIDERS
from .scope import _context_budget
from .scope import _is_library_wide_question
from .scope import _task_requires_exact_artifact
from .scope import _text_source_document_ids
from .tables import _MAIN_RESULT_TABLES_PER_DOCUMENT
from .tables import _canonical_table_number
from .tables import _described_table_numbers
from .tables import _is_canonical_result_table_request
from .tables import _requested_table_numbers
from .tables import _wants_main_result_evidence
from .tables import _wants_result_tables
from .tables import _wants_table_inventory


def _retrieval_index_configuration(settings: Settings) -> dict[str, Any]:
    """Configuration fields that change compatible retrieval/cache semantics."""

    return {
        "context_projection_version": RETRIEVAL_CACHE_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embedding_max_token_size": settings.embedding_max_token_size,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "embedding_document_prefix": settings.embedding_document_prefix,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_mode": settings.rerank_mode,
        "visual_retrieval_enabled": settings.visual_retrieval_enabled,
        "visual_retrieval_provider": settings.visual_retrieval_provider,
        "visual_retrieval_model": settings.visual_retrieval_model,
        "visual_retrieval_model_revision": settings.visual_retrieval_model_revision,
        "dual_query_rrf_enabled": settings.dual_query_rrf_enabled,
        "dual_query_rrf_k": settings.dual_query_rrf_k,
        "lightrag_enabled": settings.lightrag_enabled,
        "retrieval_engine": settings.retrieval_engine,
        "agentic_retrieval_max_hops": settings.agentic_retrieval_max_hops,
        "agentic_retrieval_max_subqueries": settings.agentic_retrieval_max_subqueries,
    }


def _merge_retrieved_documents(
    primary: list[dict[str, Any]],
    additional: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {
        identity
        for item in primary
        if (identity := _retrieval_evidence_identity(item)) is not None
    }
    merged = list(primary)
    for item in additional:
        identity = _retrieval_evidence_identity(item)
        if identity is not None and identity in seen:
            continue
        if identity is not None:
            seen.add(identity)
        merged.append(item)
    return merged


def _retrieval_evidence_identity(
    document: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Return canonical evidence identity, namespaced by document."""

    document_id = str(document.get("document_id") or "").strip()
    for kind, field in (
        ("table", "table_id"),
        ("figure", "figure_id"),
        ("parent", "parent_chunk_id"),
        ("chunk", "chunk_id"),
    ):
        value = str(document.get(field) or "").strip()
        if value:
            return kind, document_id, value
    return None


def _new_retrieval_evidence(
    *,
    initial_documents: list[dict[str, Any]],
    additional_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    initial_identities = {
        identity
        for document in initial_documents
        if (identity := _retrieval_evidence_identity(document)) is not None
    }
    new_documents: list[dict[str, Any]] = []
    seen_new: set[tuple[str, str, str]] = set()
    for document in additional_documents:
        identity = _retrieval_evidence_identity(document)
        # Without a canonical identity, progress cannot be demonstrated.
        if identity is None or identity in initial_identities or identity in seen_new:
            continue
        seen_new.add(identity)
        new_documents.append(document)
    return new_documents


def _bounded_second_retrieval_branches(
    *,
    plan: SecondRetrievalPlan,
    fallback_focus_document_ids: list[str],
    max_subqueries: int,
) -> list[RetrievalBranch]:
    limit = max(1, min(int(max_subqueries), 3))
    if plan.branches:
        return list(plan.branches[:limit])
    # Compatibility for callers constructing the former query-only plan shape.
    return [
        RetrievalBranch(
            query=plan.query,
            focus_document_ids=list(dict.fromkeys(fallback_focus_document_ids)),
            reason=f"adaptive_second_hop:{plan.reasons[0]}",
            hop=plan.hop_count,
            facets=list(plan.missing_facets),
            bridge_anchors=list(plan.bridge_anchors),
        )
    ]


def _second_retrieval_retry_payload(
    *,
    run_id: str,
    conversation_id: str,
    plan: SecondRetrievalPlan,
    branches: list[RetrievalBranch],
    max_hops: int,
    missing_entities: list[str],
    previous_focus_document_ids: list[str],
    agent_reasoning: str,
    smart_allowed: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "query": branches[0].query,
        "sub_queries": [branch.query for branch in branches],
        "branches": [
            {
                "query": branch.query,
                "focus_document_ids": branch.focus_document_ids,
                "reason": branch.reason,
                "facets": branch.facets,
                "bridge_anchors": branch.bridge_anchors,
            }
            for branch in branches
        ],
        "hop": plan.hop_count,
        "max_hops": max_hops,
        "parallel": len(branches) > 1,
        "reason": plan.reasons[0],
        "reasons": plan.reasons,
        "missing_entities": missing_entities,
        "missing_facets": plan.missing_facets,
        "bridge_anchors": plan.bridge_anchors,
        "previous_focus_document_ids": previous_focus_document_ids,
        "agent_reasoning": agent_reasoning,
        "smart_allowed": smart_allowed,
    }


def _second_hop_diagnostics_payload(
    *,
    plan: SecondRetrievalPlan,
    branches: list[RetrievalBranch],
    branch_diagnostics: list[dict[str, Any]],
    max_hops: int,
    smart_retrieval: bool,
    total_ms: float,
    new_evidence_count: int,
) -> dict[str, Any]:
    discarded = new_evidence_count == 0
    discard_reason = "no_new_evidence" if discarded else None
    parallel = len(branches) > 1
    return {
        "retry_performed": True,
        "retry_reasons": plan.reasons,
        "retry_discarded": discarded,
        "retry_discard_reason": discard_reason,
        "new_evidence_count": new_evidence_count,
        "smart_retrieval": smart_retrieval,
        "retry_branches": branch_diagnostics,
        "retry_missing_facets": plan.missing_facets,
        "retry_bridge_anchors": plan.bridge_anchors,
        "retry_timings": {
            "total_ms": total_ms,
            "branch_ms": [
                branch["timing_ms"]
                for branch in branch_diagnostics
            ],
        },
        "adaptive_second_hop": {
            "hop": plan.hop_count,
            "max_hops": max_hops,
            "parallel": parallel,
            "sub_queries": [branch.query for branch in branches],
            "branches": branch_diagnostics,
            "missing_facets": plan.missing_facets,
            "bridge_anchors": plan.bridge_anchors,
            "new_evidence_count": new_evidence_count,
            "discarded": discarded,
            "discard_reason": discard_reason,
        },
    }


async def _execute_second_retrieval_branches(
    *,
    branches: list[RetrievalBranch],
    rag: RagService,
    settings: Settings,
    collection_id: str | None,
    retrieval_mode: str,
    answer_intent: str,
    answer_depth: str,
    include_visual_boost: bool,
    prefer_legacy_tables: bool,
    coverage_policy: DocumentCoveragePolicy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a single bounded hop; provider and deadline failures propagate."""

    timeout_seconds = settings.agentic_retrieval_hop_timeout_seconds
    bounded_branches = list(
        branches[: settings.agentic_retrieval_max_subqueries]
    )

    async def run_branch(
        branch_index: int,
        branch: RetrievalBranch,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                retrieval_module._retrieve_for_agent(
                    rag=rag,
                    settings=settings,
                    query=branch.query,
                    collection_id=collection_id,
                    retrieval_mode=retrieval_mode,
                    focus_document_ids=branch.focus_document_ids,
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                    include_visual_boost=include_visual_boost,
                    prefer_legacy_tables=(
                        prefer_legacy_tables and bool(branch.focus_document_ids)
                    ),
                    allow_decomposition=False,
                    coverage_policy=coverage_policy,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "Adaptive retrieval branch "
                f"{branch_index + 1} exceeded {timeout_seconds:.1f}s"
            ) from exc
        timing_ms = _elapsed_ms(started_at)
        return result, {
            "index": branch_index,
            "query": branch.query,
            "focus_document_ids": branch.focus_document_ids,
            "reason": branch.reason,
            "hop": branch.hop,
            "facets": branch.facets,
            "bridge_anchors": branch.bridge_anchors,
            "timing_ms": timing_ms,
            "mode": result.get("mode"),
            "selected_engine": (result.get("diagnostics") or {}).get("selected_engine"),
            "policy_reason": (result.get("diagnostics") or {}).get("policy_reason"),
            "source_count": len(result.get("documents") or []),
        }

    completed = await asyncio.gather(
        *[
            run_branch(branch_index, branch)
            for branch_index, branch in enumerate(bounded_branches)
        ]
    )
    return (
        [result for result, _ in completed],
        [diagnostics for _, diagnostics in completed],
    )


def _compose_accumulated_retrieval(
    *,
    query: str,
    retrievals: list[dict[str, Any]],
    answer_intent: str | None,
    answer_depth: str | None,
    include_visual_boost: bool,
    prefer_tables: bool,
    required_document_ids: list[str] | None = None,
    coverage_policy: DocumentCoveragePolicy | None = None,
    coverage_query: str | None = None,
) -> dict[str, Any]:
    """Merge hop evidence without discarding the first retrieval frontier."""

    merged: list[dict[str, Any]] = []
    # Targeted later hops come first so newly recovered evidence is not pushed
    # out by the bounded context budget; the earlier frontier is still retained.
    for retrieval in reversed(retrievals):
        merged = _merge_retrieved_documents(
            merged,
            list(retrieval.get("documents") or []),
        )

    budget = _context_budget(answer_intent, answer_depth)
    coverage_enabled = bool(coverage_policy and coverage_policy.enabled)
    if coverage_enabled and coverage_policy is not None:
        budget = {
            **budget,
            "max_sources": max(budget["max_sources"], coverage_policy.max_sources),
            "max_chars": max(budget["max_chars"], coverage_policy.max_chars),
            "max_chars_per_source": min(
                budget["max_chars_per_source"], coverage_policy.max_chars_per_source
            ),
            "max_chunks_per_document": max(
                budget["max_chunks_per_document"], coverage_policy.page_limit
            ),
        }
    composed = compose_retrieval_context(
        merged,
        query=query,
        max_sources=max(budget["max_sources"], 8),
        max_chars=max(budget["max_chars"], 6_500),
        max_chars_per_source=(
            budget["max_chars_per_source"]
            if coverage_enabled
            else max(budget["max_chars_per_source"], 1_050)
        ),
        max_chunks_per_document=max(budget["max_chunks_per_document"], 3),
        min_figures=2 if include_visual_boost else 0,
        min_tables=1 if prefer_tables else 0,
        min_visual_pages=visual_page_reservation_count(
            query,
            [
                item
                for item in merged
                if item.get("visual_page")
                or item.get("artifact_type") == "visual_page"
            ],
        ),
        required_document_ids=required_document_ids,
        required_facets_by_document=_context_facet_requirements(
            coverage_query or query, required_document_ids, answer_intent,
        ),
        coverage_mode=coverage_enabled,
    )
    modes = [str(item.get("mode") or "unknown") for item in retrievals]
    diagnostics = {
        "adaptive_multihop": len(retrievals) > 1,
        "hop_count": 2 if len(retrievals) > 1 else 1,
        "branch_count": max(0, len(retrievals) - 1),
        "hop_modes": modes,
        "accumulated_candidate_count": len(merged),
        "selected_engines": [
            (item.get("diagnostics") or {}).get("selected_engine")
            for item in retrievals
            if (item.get("diagnostics") or {}).get("selected_engine")
        ],
        "policy_reasons": [
            (item.get("diagnostics") or {}).get("policy_reason")
            for item in retrievals
            if (item.get("diagnostics") or {}).get("policy_reason")
        ],
    }
    for retrieval in retrievals:
        diagnostics.update(
            {
                key: value
                for key, value in (retrieval.get("diagnostics") or {}).items()
                if key not in diagnostics
            }
        )
    return {
        "mode": "adaptive_multihop" if len(retrievals) > 1 else modes[0],
        "documents": composed.sources,
        "context_text": composed.context_text,
        "context_stats": composed.stats,
        "diagnostics": diagnostics,
    }


async def _retrieve_visual_assets_for_agent(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    collection_id: str | None,
    focus_document_ids: list[str] | None,
) -> dict:
    return await rag.retrieve_visual_assets(
        query=query,
        top_k=4,
        collection_id=collection_id,
        document_ids=focus_document_ids or None,
        retrieval_store=deps._retrieval_store(settings),
        embeddings=deps._embedding_provider(settings),
        rerank=settings.rerank_enabled,
        rerank_mode=settings.rerank_mode,
        cross_encoder_model_path=settings.rerank_cross_encoder_path,
        rerank_max_candidates=settings.rerank_max_candidates,
    )


def _visual_page_provider(settings: Settings) -> Any | None:
    """Return the approved isolated visual provider when the feature is on."""

    if not (
        settings.visual_retrieval_enabled
        and settings.visual_retrieval_provider == "colsmol"
    ):
        return None
    from app.rag.visual_worker_client import ColSmolWorkerClient

    key = (
        str(settings.visual_retrieval_path),
        str(settings.visual_retrieval_model),
        str(settings.visual_retrieval_model_revision),
        str(settings.visual_retrieval_base_model),
        str(settings.visual_retrieval_base_revision),
        str(settings.visual_retrieval_worker_python),
        str(settings.visual_retrieval_device),
        bool(settings.visual_retrieval_local_files_only),
        float(settings.request_timeout_seconds),
    )
    provider = _VISUAL_PAGE_PROVIDERS.get(key)
    if provider is None:
        provider = ColSmolWorkerClient(
            python_executable=settings.visual_retrieval_worker_python,
            index_root=settings.visual_retrieval_path / "colsmol",
            model_name=settings.visual_retrieval_model,
            model_revision=settings.visual_retrieval_model_revision,
            base_model_name=settings.visual_retrieval_base_model,
            base_model_revision=settings.visual_retrieval_base_revision,
            device=settings.visual_retrieval_device,
            local_files_only=settings.visual_retrieval_local_files_only,
            response_timeout_seconds=settings.request_timeout_seconds,
        )
        _VISUAL_PAGE_PROVIDERS[key] = provider
    return provider


async def _retrieve_visual_pages_for_agent(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    collection_id: str | None,
    focus_document_ids: list[str] | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Retrieve optional page images without weakening local text fallback."""

    provider = _visual_page_provider(settings)
    if provider is None:
        return [], None
    try:
        pages = await rag.search_visual_pages(
            query,
            provider=provider,
            top_k=top_k,
            collection_id=collection_id,
            document_ids=focus_document_ids,
        )
        return pages, None
    except Exception as exc:  # visual is derived; local canonical search survives
        logger.warning("Optional visual retrieval failed: %s", type(exc).__name__)
        return [], f"{type(exc).__name__}: {exc}"


def _merge_page_coverage_sources(
    text_pages: list[dict[str, Any]],
    visual_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compatibility wrapper around the shared page-context merge contract."""

    return merge_page_coverage_sources(text_pages, visual_pages)


async def _merge_focused_figures(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    collection_id: str | None,
    sources: list[dict[str, Any]],
    focus_document_ids: list[str],
    min_figures: int,
    answer_intent: str | None = None,
) -> list[dict[str, Any]]:
    existing_ids = {str(item.get("figure_id")) for item in sources if item.get("figure_id")}
    preferred_figure = requested_figure_number(query)
    if preferred_figure is not None:
        # "Figure 2 của nó": fetch the printed Figure N of the focused paper
        # when ranking did not surface it, instead of answering "not found".
        lookup = getattr(rag, "artifact_source_by_number", None)
        has_exact = any(
            item.get("figure_id") and _source_figure_number(item) == preferred_figure
            and str(item.get("document_id") or "") in set(focus_document_ids)
            for item in sources
        )
        if callable(lookup) and not has_exact:
            for document_id in focus_document_ids[:2]:
                exact = lookup(document_id, "figure", preferred_figure)
                if exact and str(exact.get("figure_id")) not in existing_ids:
                    sources = [exact, *sources]
                    existing_ids.add(str(exact.get("figure_id")))

    if answer_intent == "compare" and len(focus_document_ids) >= 2:
        merged = list(sources)
        figures_by_doc: dict[str, set[str]] = defaultdict(set)
        for item in merged:
            if item.get("figure_id") and item.get("document_id"):
                figures_by_doc[str(item["document_id"])].add(str(item["figure_id"]))

        for document_id in focus_document_ids:
            if figures_by_doc.get(document_id):
                continue
            visual = await rag.retrieve_visual_assets(
                query=query,
                top_k=6,
                collection_id=collection_id,
                document_ids=[document_id],
                retrieval_store=deps._retrieval_store(settings),
                embeddings=deps._embedding_provider(settings),
                rerank=settings.rerank_enabled,
                rerank_mode=settings.rerank_mode,
                cross_encoder_model_path=settings.rerank_cross_encoder_path,
                rerank_max_candidates=settings.rerank_max_candidates,
            )
            ranked = sorted(
                visual.get("results") or [],
                key=lambda item: figure_relevance_score(
                    item,
                    preferred_figure_number=preferred_figure,
                    query=query,
                ),
                reverse=True,
            )
            for item in ranked:
                figure_id = item.get("figure_id")
                if not figure_id or figure_id in existing_ids:
                    continue
                if _is_low_signal_figure(item):
                    continue
                merged.append(item)
                existing_ids.add(str(figure_id))
                figures_by_doc[document_id].add(str(figure_id))
                break
        return merged

    if len(existing_ids) >= min_figures:
        return sources

    visual_query = query
    if has_visual_intent(query):
        visual_query = (
            "figure diagram chart plot table visualization arousal valence CCC benchmark results "
            f"{query}"
        )

    visual = await rag.retrieve_visual_assets(
        query=visual_query,
        top_k=max(min_figures * 2, 6),
        collection_id=collection_id,
        document_ids=focus_document_ids,
        retrieval_store=deps._retrieval_store(settings),
        embeddings=deps._embedding_provider(settings),
        rerank=settings.rerank_enabled,
        rerank_mode=settings.rerank_mode,
        cross_encoder_model_path=settings.rerank_cross_encoder_path,
        rerank_max_candidates=settings.rerank_max_candidates,
    )

    merged = list(sources)
    ranked_visual = sorted(
        visual.get("results") or [],
        key=lambda item: figure_relevance_score(
            item,
            preferred_figure_number=preferred_figure,
            query=query,
        ),
        reverse=True,
    )
    for item in ranked_visual:
        figure_id = item.get("figure_id")
        if not figure_id or figure_id in existing_ids:
            continue
        if _is_low_signal_figure(item):
            continue
        merged.append(item)
        existing_ids.add(str(figure_id))
        if preferred_figure is not None:
            break
        if len(existing_ids) >= min_figures:
            break
    return merged


async def _retrieve_with_paper_evidence_cards(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    original_task: str,
    collection_id: str | None,
    retrieval_mode: str,
    focus_document_ids: list[str],
    answer_intent: str,
    answer_depth: str,
    include_visual_boost: bool,
    prefer_tables: bool,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]] | None:
    """Use canonical cards first and retrieve only demonstrably missing cells."""

    ordered_ids = list(dict.fromkeys(focus_document_ids))
    if not settings.paper_evidence_cards_enabled or not ordered_ids:
        return None
    requested_facets = requested_paper_facets(
        original_task,
        answer_intent=answer_intent,
        focused_document_count=len(ordered_ids),
    )
    service = deps.PaperEvidenceService(
        settings.sqlite_db_path,
        schema_version=settings.paper_evidence_card_schema_version,
        prompt_version=settings.paper_evidence_card_prompt_version,
    )
    cards, coverage = service.coverage_matrix(ordered_ids, requested_facets)
    card_sources = service.materialize_sources(cards, requested_facets=requested_facets)
    navigation_context = service.render_navigation_context(
        cards,
        requested_facets=requested_facets,
    )
    missing = {
        item.document_id: list(item.missing_facets)
        for item in coverage
        if item.missing_facets
    }

    raw_retrievals: list[dict[str, Any]] = []
    if missing:
        semaphore = asyncio.Semaphore(
            min(settings.agentic_retrieval_max_subqueries, max(1, len(missing)))
        )

        async def retrieve_document(document_id: str, facets: list[str]) -> dict[str, Any]:
            async with semaphore:
                missing_query = " ".join(
                    part
                    for part in (
                        query,
                        facet_query_terms(facets),
                    )
                    if part
                )
                result = await retrieval_module._retrieve_legacy_for_agent(
                    rag=rag,
                    settings=settings,
                    query=missing_query,
                    collection_id=collection_id,
                    retrieval_mode=retrieval_mode,
                    focus_document_ids=[document_id],
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                    include_visual_boost=include_visual_boost,
                    prefer_tables=prefer_tables or "benchmark_results" in facets,
                )
                for source in result.get("documents") or []:
                    existing = list(source.get("evidence_facets") or [])
                    source["evidence_facets"] = list(dict.fromkeys([*existing, *facets]))
                    source["raw_fallback_reason"] = "missing_card_facets"
                result.setdefault("diagnostics", {})["missing_card_facets"] = facets
                result["diagnostics"]["focus_document_id"] = document_id
                return result

        raw_retrievals = await asyncio.gather(
            *(retrieve_document(document_id, facets) for document_id, facets in missing.items())
        )

    merged_sources = list(card_sources)
    for result in raw_retrievals:
        merged_sources = _merge_retrieved_documents(
            merged_sources,
            list(result.get("documents") or []),
        )
    if not merged_sources:
        return None
    context_budget = _context_budget(answer_intent, answer_depth)
    composed = compose_retrieval_context(
        merged_sources,
        query=query,
        max_sources=max(context_budget["max_sources"], len(ordered_ids) * 3),
        max_chars=max(context_budget["max_chars"], len(ordered_ids) * 2_400),
        max_chars_per_source=context_budget["max_chars_per_source"],
        max_chunks_per_document=max(3, context_budget["max_chunks_per_document"]),
        min_figures=2 if include_visual_boost else 0,
        min_tables=1 if prefer_tables else 0,
        min_visual_pages=visual_page_reservation_count(
            query,
            [
                item
                for item in merged_sources
                if item.get("visual_page")
                or item.get("artifact_type") == "visual_page"
            ],
        ),
        required_document_ids=ordered_ids,
        required_facets_by_document={
            document_id: list(requested_facets)
            for document_id in ordered_ids
        },
    )
    coverage_payload = [
        {
            "document_id": item.document_id,
            "requested_facets": item.requested_facets,
            "covered_facets": item.covered_facets,
            "missing_facets": item.missing_facets,
            "stale": item.stale,
            "status": item.status,
        }
        for item in coverage
    ]
    return (
        {
            "mode": "paper_evidence_cards" if not missing else "paper_evidence_cards+raw_fallback",
            "documents": composed.sources,
            "context_text": composed.context_text,
            "context_stats": composed.stats,
            "diagnostics": {
                "selected_engine": "paper_evidence_cards",
                "policy_reason": "card_first_missing_facet_only",
                "requested_facets": requested_facets,
                "paper_facet_coverage": coverage_payload,
                "card_source_count": len(card_sources),
                "raw_fallback_document_count": len(raw_retrievals),
                "raw_fallback_document_ids": list(missing),
            },
        },
        navigation_context,
        coverage_payload,
    )


def _paper_card_validation_documents(
    *,
    settings: Settings,
    retrieval: dict[str, Any],
    retrieved_docs: list[dict[str, Any]],
    focus_document_ids: list[str],
) -> list[dict[str, Any]]:
    """Use full canonical card refs for validation, not prompt projections.

    Context composition intentionally projects long tables and excerpts to a
    bounded generation prompt. Paper-card navigation can still mention claims
    backed by other rows of the same canonical source. Re-validating only the
    projected prompt would reject those correct claims. This local SQLite path
    reconstructs the complete canonical refs without adding prompt tokens or
    making another model/retrieval call; raw fallback excerpts remain eligible
    through ``retrieved_docs``.
    """

    ordered_ids = list(
        dict.fromkeys(
            str(document_id).strip()
            for document_id in focus_document_ids
            if str(document_id).strip()
        )
    )
    diagnostics = retrieval.get("diagnostics") or {}
    requested_facets = list(diagnostics.get("requested_facets") or [])
    card_backed = bool(
        requested_facets
        and (
            diagnostics.get("selected_engine") == "paper_evidence_cards"
            or diagnostics.get("paper_facet_coverage")
            or any(document.get("card_evidence") for document in retrieved_docs)
        )
    )
    if not settings.paper_evidence_cards_enabled or not ordered_ids:
        return retrieved_docs
    service = deps.PaperEvidenceService(
        settings.sqlite_db_path,
        schema_version=settings.paper_evidence_card_schema_version,
        prompt_version=settings.paper_evidence_card_prompt_version,
    )
    validation_tables = service.materialize_validation_tables(ordered_ids)
    # A retrieval/last-source cache may legitimately omit the original card
    # diagnostics even though the focused document and its canonical tables
    # are still present locally. Validation scope follows the catalog focus,
    # not cache metadata; table materialization is therefore unconditional
    # once the feature and focus gates above are satisfied.
    if not card_backed:
        return _merge_retrieved_documents(validation_tables, retrieved_docs)
    cards, _coverage = service.coverage_matrix(ordered_ids, requested_facets)
    canonical_sources = service.materialize_sources(
        cards,
        requested_facets=requested_facets,
    )
    return _merge_retrieved_documents(
        _merge_retrieved_documents(canonical_sources, validation_tables),
        retrieved_docs,
    )


def _should_merge_visual_sources(
    *,
    query: str,
    include_visual_boost: bool,
) -> bool:
    """Reserve visual evidence only when the request actually needs visuals.

    Cross-document comparison is a reasoning shape, not a visual facet.  Treating
    every comparison as visual used to reserve figure slots during both the
    initial and adaptive retrieval hops, which could evict the result text the
    comparison explicitly asked for.  Explicit visual language and an upstream
    visual policy flag remain authoritative.
    """

    return bool(include_visual_boost or has_visual_intent(query))


def _visual_required_by_readiness(
    rag: RagService,
    document_ids: list[str] | tuple[str, ...] | None,
) -> bool:
    """Check the active projection for a visual-only document.

    Readiness can require pixels even when the query has no visual keyword.
    Keep this as a compatibility boundary for older RagService test doubles;
    the production implementation is revision-scoped and answer-blind.
    """

    resolver = getattr(rag, "documents_requiring_visual_channel", None)
    if not callable(resolver):
        return False
    try:
        return bool(resolver(list(document_ids or ())))
    except Exception:  # pragma: no cover - defensive optional capability
        logger.debug("Visual-only readiness probe unavailable", exc_info=True)
        return False


def _with_translation_variant(
    variants: list[tuple[str, str]], translation: str | None, task: str,
) -> list[tuple[str, str]]:
    """Add the router's translation of the question as a retrieval variant:
    a Vietnamese question about an English document (or the reverse) then
    matches it lexically and semantically."""
    text = " ".join(str(translation or "").split())
    if not text or text.casefold() == " ".join(str(task or "").split()).casefold():
        return list(variants)
    if any(existing.casefold() == text.casefold() for _origin, existing in variants):
        return list(variants)
    return [*variants, ("translation", text)][:3]


def _safe_query_variants(
    retrieval_query: str,
    standalone_query: str,
    required_entities: list[str] | tuple[str, ...],
) -> list[tuple[str, str]]:
    """Build at most original + rewrite variants without query drift."""

    plan = plan_query_variants(
        original_query=retrieval_query,
        standalone_query=standalone_query,
        required_entities=required_entities,
    )
    return list(plan.variants)


def _context_facet_requirements(
    query: str,
    document_ids: list[str] | None,
    answer_intent: str | None,
) -> dict[str, list[str]]:
    """Keep semantic obligations separate from appended search vocabulary.

    A rewrite can add encoder/training anchors to improve retrieval. Those
    tokens must not narrow a general comparison to those two facets, nor can a
    per-paper branch discard the original multi-paper coverage contract.
    """
    ids = list(dict.fromkeys(document_ids or []))
    if not ids:
        return {}
    # Do not manufacture an overview obligation for every implicit fact ask.
    # "Which languages?" is not a request for the abstract/contributions; a
    # speculative reservation could evict the actual table or figure. Keep
    # existing ranking when neither explicit facets nor comparison supplies
    # a semantic contract.
    if len(ids) == 1 and answer_intent != "compare" and not extract_query_facets(query):
        return {}
    facets = requested_paper_facets(
        query, answer_intent=answer_intent, focused_document_count=len(ids),
    )
    return {document_id: list(facets) for document_id in ids}


_RESULT_ASK_RE = re.compile(
    r"(?<!\w)(?:kết\s*quả|ket\s*qua|results?|scores?|accuracy|acc|f1|uar|war|wa|ua|ccc|baseline|benchmark|"
    r"bảng|table|dataset|số\s*liệu|bao\s*nhiêu)(?!\w)",
    re.IGNORECASE,
)


def _asks_library_results_for_named_terms(rag: Any, task: str, collection_id: str | None) -> bool:
    """"Cho mình kết quả của baseline WavLM": a results ask about a name the
    library's papers report, even if the router took it for general chat."""
    terms = _named_query_terms(task)
    strong = [term for term in terms if _is_strong_name(term)]
    weak = [term for term in terms if term not in strong]
    # A question names one or two things; text with many capitalised words
    # (a pasted table, an essay prompt) is not naming library entities.
    terms = strong or (weak if len(weak) <= 2 else [])
    lookup = getattr(rag, "documents_containing_all_terms", None)
    if not terms or not callable(lookup) or not _RESULT_ASK_RE.search(task):
        return False
    return bool(lookup(terms, collection_id))


def _library_mentions_named_terms(rag: Any, task: str, collection_id: str | None) -> bool:
    """A name written like an entity ("LPMN", "WavLM", "wav2vec2") that some
    library document mentions verbatim: the turn may be about that document."""
    lookup = getattr(rag, "documents_containing_all_terms", None)
    if not callable(lookup):
        return False
    strong = [term for term in _named_query_terms(task) if _is_strong_name(term)][:5]
    return any(lookup([term], collection_id) for term in strong)


def _is_strong_name(term: str) -> bool:
    return term.isupper() or any(char.isupper() for char in term[1:]) or any(char.isdigit() for char in term)


_DOCUMENT_ID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


def _document_ids_named_in(rag: Any, task: str, collection_id: str | None) -> list[str]:
    """Library document ids written in the message (UI links, API callers,
    other agents): an explicit, unambiguous document reference."""
    ids = []
    allowed = set(rag.collection_document_ids(collection_id)) if collection_id else None
    for raw in _DOCUMENT_ID_RE.findall(str(task or "")):
        document_id = raw.lower()
        if (allowed is None or document_id in allowed) and (rag.get_document(document_id) or {}):
            ids.append(document_id)
    return list(dict.fromkeys(ids))


_PAPER_BY_TERMS_RE = re.compile(
    r"(?<!\w)(?:bài|bai|paper|tài\s*liệu|tai\s*lieu|article|work|one)\s+(?:có|co|nhắc|nhac|dùng|dung|"
    r"with|that\s+(?:has|uses|mentions|compares)|mentioning|using)(?!\w)",
    re.IGNORECASE,
)
LIBRARY_SCAN_PASSAGES_PER_DOCUMENT = 2
_NAMED_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*")
_GENERIC_NAMED_TERMS = frozenset({
    "which", "what", "how", "does", "the", "paper", "papers", "and", "are", "any", "with", "only",
    "nào", "những", "các", "trong", "của", "cho", "mình", "không",
})


def _named_query_terms(query: str) -> list[str]:
    """Terms written as names or acronyms ("GPU", "Whisper", "MSP-Podcast",
    "LSSED"): worth a verbatim scan in every document."""
    text = str(query or "")
    terms = []
    for match in _NAMED_TERM_RE.finditer(text):
        token = match.group(0)
        if token.casefold() in _GENERIC_NAMED_TERMS or len(token) < 3:
            continue
        # Acronyms, inner capitals and digits mark a name anywhere
        # (GPU, WavLM, MSP-Podcast, wav2vec2). A capitalised plain word only
        # counts mid-sentence: "Let", "Find" open sentences; "Whisper" in
        # "papers that use Whisper" is a name.
        strong = token.isupper() or any(char.isupper() for char in token[1:]) or any(char.isdigit() for char in token)
        sentence_start = re.search(r"(?:^|[.!?:;\n]\s*|[\"'(]\s*)$", text[: match.start()]) is not None
        if strong or (token[0].isupper() and not sentence_start):
            terms.append(token)
    return list(dict.fromkeys(terms))
LIBRARY_SCAN_MAX_CHARS = 48_000


async def _retrieve_library_wide(
    *, rag: RagService, settings: Settings, query: str, collection_id: str | None,
    alt_queries: list[str] | None = None,
) -> dict:
    """Search every document separately and keep its best passages.

    A single global ranking returns the few papers that discuss the topic
    most, so "which papers use Whisper / which GPU did each paper use" misses
    papers that mention it once. A per-document map, plus the catalog of
    titles, lets the model answer for the whole library.
    """
    titles = rag.library_titles(collection_id)
    terms = _named_query_terms(query)
    store = deps._retrieval_store(settings)
    embeddings = deps._embedding_provider(settings)
    gate = asyncio.Semaphore(4)

    async def best_passages(document_id: str) -> list[dict[str, Any]]:
        ranked: dict[str, tuple[int, dict[str, Any]]] = {}
        for text in [query, *(alt_queries or [])][:2]:
            async with gate:
                try:
                    found = await rag.search_hybrid(
                        query=text,
                        top_k=4,
                        collection_id=collection_id,
                        document_ids=[document_id],
                        retrieval_store=store,
                        embeddings=embeddings,
                        rerank=settings.rerank_enabled,
                        rerank_mode=settings.rerank_mode,
                        cross_encoder_model_path=settings.rerank_cross_encoder_path,
                        rerank_max_candidates=settings.rerank_max_candidates,
                    )
                except Exception:  # noqa: BLE001 - one document must not fail the scan
                    continue
            for rank, item in enumerate(found.get("results") or []):
                key = str(item.get("chunk_id") or item.get("id") or rank)
                if key not in ranked or rank < ranked[key][0]:
                    ranked[key] = (rank, item)
        passages = [item for _rank, item in sorted(ranked.values(), key=lambda pair: pair[0]) if not item.get("figure_id")]
        passages = passages[:LIBRARY_SCAN_PASSAGES_PER_DOCUMENT]
        # Semantic ranking can bury a one-line mention ("trained on 8 RTX
        # 4090"); add the passage that names the asked term verbatim.
        lookup = getattr(rag, "passages_with_terms", None)
        if terms and callable(lookup):
            seen = {str(item.get("chunk_id") or "") for item in passages}
            for extra in lookup(document_id, terms, limit=1):
                if str(extra.get("chunk_id")) not in seen:
                    passages.append(extra)
        return passages

    groups = await asyncio.gather(*[best_passages(document_id) for document_id, _title in titles])
    results = [item for group in groups for item in group]
    composed = compose_retrieval_context(
        results,
        query=query,
        max_sources=max(len(results), 1),
        max_chars=LIBRARY_SCAN_MAX_CHARS,
        max_chars_per_source=1_800,
        max_chunks_per_document=LIBRARY_SCAN_PASSAGES_PER_DOCUMENT + 1,
    )
    filenames = {
        document_id: str((rag.get_document(document_id) or {}).get("filename") or "")
        for document_id, _title in titles
    }
    counter = getattr(rag, "term_counts", None)
    counts = counter([document_id for document_id, _title in titles], terms) if terms and callable(counter) else {}

    def mentions(document_id: str) -> str:
        per_term = counts.get(document_id) or {}
        if not terms or not per_term:
            return ""
        return " — mentions: " + ", ".join(f"{term} ×{per_term.get(term, 0)}" for term in terms)

    catalog = {
        "filename": "library catalog",
        "artifact_type": "catalog",
        "content": (
            f"The library has {len(titles)} documents (answer library-wide questions over all of them; "
            "the passages below are each document's best match, a document with no relevant passage "
            "does not discuss the topic"
            + ("; '×N' counts verbatim mentions of the asked terms in the whole document, so ×0 means "
               "the document never mentions it" if terms else "")
            + "):\n"
            + "\n".join(
                f"- {title}"
                + (
                    f" ({filenames.get(document_id)})"
                    if filenames.get(document_id) and filenames.get(document_id) not in title
                    else ""
                )
                + mentions(document_id)
                for document_id, title in titles
            )
        ),
        "retrieval_channels": ["library_catalog"],
    }
    sources = [catalog, *composed.sources]
    return {
        "mode": "library_scan",
        "documents": sources,
        "context_text": composed.context_text,
        "context_stats": {**composed.stats, "library_documents": len(titles)},
        "diagnostics": {"library_scan": True, "selected_document_ids": [d for d, _ in titles]},
    }


async def _retrieve_for_agent(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    collection_id: str | None,
    retrieval_mode: str,
    focus_document_ids: list[str] | None = None,
    answer_intent: str | None = None,
    answer_depth: str | None = None,
    include_visual_boost: bool = False,
    prefer_legacy_tables: bool = False,
    allow_decomposition: bool = True,
    must_cover_all_documents: bool = False,
    query_variants: list[tuple[str, str]] | None = None,
    coverage_policy: DocumentCoveragePolicy | None = None,
    coverage_query: str | None = None,
    library_scope: bool | None = None,
) -> dict:
    library_question = (
        library_scope if library_scope is not None else _is_library_wide_question(coverage_query or query)
    )
    if not focus_document_ids and library_question:
        return await _retrieve_library_wide(
            rag=rag, settings=settings, query=query, collection_id=collection_id,
            alt_queries=[text for origin, text in (query_variants or ()) if origin == "translation"],
        )
    facet_requirements = _context_facet_requirements(
        coverage_query or query, focus_document_ids, answer_intent,
    )
    branches = plan_retrieval_decomposition(
        query=query,
        answer_intent=answer_intent or "",
        focus_document_ids=focus_document_ids or [],
        enabled=(
            allow_decomposition
            and settings.agentic_retrieval_decomposition_enabled
        ),
        must_cover_all=must_cover_all_documents,
    )
    if branches:
        branch_results = await asyncio.gather(
            *[
                retrieval_module._retrieve_for_agent(
                    rag=rag,
                    settings=settings,
                    query=branch.query,
                    collection_id=collection_id,
                    retrieval_mode=retrieval_mode,
                    focus_document_ids=branch.focus_document_ids,
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                    include_visual_boost=include_visual_boost,
                    prefer_legacy_tables=prefer_legacy_tables,
                    allow_decomposition=False,
                    must_cover_all_documents=False,
                    query_variants=None,
                    coverage_policy=coverage_policy,
                    coverage_query=coverage_query,
                )
                for branch in branches
            ]
        )
        merged_sources: list[dict[str, Any]] = []
        for result in branch_results:
            merged_sources = _merge_retrieved_documents(
                merged_sources,
                list(result.get("documents") or []),
            )
        context_budget = _context_budget(answer_intent, answer_depth)
        composed = compose_retrieval_context(
            merged_sources,
            query=query,
            max_sources=max(context_budget["max_sources"], len(branches) * 3),
            max_chars=max(context_budget["max_chars"], len(branches) * 2_600),
            max_chars_per_source=context_budget["max_chars_per_source"],
            max_chunks_per_document=max(
                context_budget["max_chunks_per_document"],
                3,
            ),
            min_figures=2 if include_visual_boost else 0,
            min_tables=1 if prefer_legacy_tables else 0,
            min_visual_pages=visual_page_reservation_count(
                query,
                [
                    item
                    for item in merged_sources
                    if item.get("visual_page")
                    or item.get("artifact_type") == "visual_page"
                ],
            ),
            required_document_ids=[
                branch.focus_document_ids[0]
                for branch in branches
                if branch.focus_document_ids
            ],
            required_facets_by_document=facet_requirements,
        )
        return {
            "mode": "decomposed_compare",
            "documents": composed.sources,
            "context_text": composed.context_text,
            "context_stats": composed.stats,
            "diagnostics": {
                "decomposition": True,
                "selected_engine": "decomposed_compare",
                "policy_reason": "compare_per_document",
                "branch_count": len(branches),
                "branch_document_ids": [
                    branch.focus_document_ids[0] for branch in branches
                ],
                "branch_modes": [result.get("mode") for result in branch_results],
                "branch_selected_engines": [
                    (result.get("diagnostics") or {}).get("selected_engine")
                    for result in branch_results
                ],
                "branch_policy_reasons": [
                    (result.get("diagnostics") or {}).get("policy_reason")
                    for result in branch_results
                ],
            },
        }

    # Table/benchmark asks need LanceDB table chunks; LightRAG alone often misses them.
    if prefer_legacy_tables and focus_document_ids:
        legacy = await retrieval_module._retrieve_legacy_for_agent(
            rag=rag,
            settings=settings,
            query=query,
            collection_id=collection_id,
            retrieval_mode=retrieval_mode,
            focus_document_ids=focus_document_ids,
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            include_visual_boost=include_visual_boost,
            prefer_tables=True,
            query_variants=query_variants,
            coverage_policy=coverage_policy,
            coverage_query=coverage_query,
        )
        if legacy.get("documents"):
            legacy_diagnostics = legacy.setdefault("diagnostics", {})
            legacy_diagnostics["selected_engine"] = "legacy_table"
            legacy_diagnostics["policy_reason"] = "focused_table_evidence"
            return legacy

    engine = _select_retrieval_engine(
        configured_engine=settings.retrieval_engine,
        retrieval_mode=retrieval_mode,
        answer_intent=answer_intent,
        focus_document_ids=focus_document_ids,
        prefer_legacy_tables=prefer_legacy_tables,
    )
    policy_reason = _retrieval_engine_policy_reason(
        configured_engine=settings.retrieval_engine,
        retrieval_mode=retrieval_mode,
        answer_intent=answer_intent,
        focus_document_ids=focus_document_ids,
        prefer_legacy_tables=prefer_legacy_tables,
    )
    lightrag_uninitialized = False
    if engine in {"lightrag", "dual"} and settings.lightrag_enabled:
        try:
            bridge = deps.LightRAGBridge(settings)
            # Collection scope applies to every retrieval engine. An empty
            # collection remains an explicit empty scope; it must never widen
            # to the full LightRAG graph.
            lightrag_scope = list(focus_document_ids) if focus_document_ids else None
            if collection_id is not None:
                collection_scope = rag.collection_document_ids(collection_id)
                if lightrag_scope is None:
                    lightrag_scope = collection_scope
                else:
                    allowed = set(collection_scope)
                    lightrag_scope = [
                        document_id
                        for document_id in lightrag_scope
                        if document_id in allowed
                    ]
            lightrag_result = await bridge.retrieve(
                query,
                answer_intent=answer_intent,
                retrieval_mode=retrieval_mode if retrieval_mode.lower() in {"local", "global", "hybrid", "mix", "naive"} else None,
                focus_document_ids=lightrag_scope,
                answer_depth=answer_depth,
                include_visual_boost=include_visual_boost,
            )
            lightrag_diagnostics = lightrag_result.setdefault("diagnostics", {})
            lightrag_diagnostics["selected_engine"] = engine
            lightrag_diagnostics["policy_reason"] = policy_reason
            sources = list(lightrag_result["documents"])
            if lightrag_scope and _should_merge_visual_sources(
                query=query,
                include_visual_boost=include_visual_boost,
            ):
                sources = await _merge_focused_figures(
                    rag=rag,
                    settings=settings,
                    query=query,
                    collection_id=collection_id,
                    sources=sources,
                    focus_document_ids=lightrag_scope,
                    min_figures=4 if has_visual_intent(query) else 2,
                    answer_intent=answer_intent,
                )
                context_budget = _context_budget(answer_intent, answer_depth)
                composed = compose_retrieval_context(
                    sources,
                    query=query,
                    max_sources=context_budget["max_sources"],
                    max_chars=context_budget["max_chars"],
                    max_chars_per_source=context_budget["max_chars_per_source"],
                    max_chunks_per_document=context_budget["max_chunks_per_document"],
                    min_figures=4 if has_visual_intent(query) else (2 if include_visual_boost else 0),
                    min_visual_pages=visual_page_reservation_count(
                        query,
                        [
                            item
                            for item in sources
                            if item.get("visual_page")
                            or item.get("artifact_type") == "visual_page"
                        ],
                    ),
                )
                lightrag_result["documents"] = composed.sources
                lightrag_result["context_text"] = composed.context_text
                lightrag_result["context_stats"] = {
                    **composed.stats,
                    "figure_source_count": sum(1 for item in composed.sources if item.get("figure_id")),
                }
            if engine == "lightrag":
                if lightrag_result.get("documents"):
                    return lightrag_result
                legacy_fallback = await retrieval_module._retrieve_legacy_for_agent(
                    rag=rag,
                    settings=settings,
                    query=query,
                    collection_id=collection_id,
                    retrieval_mode=retrieval_mode,
                    focus_document_ids=focus_document_ids,
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                    include_visual_boost=include_visual_boost,
                    query_variants=query_variants,
                    coverage_policy=coverage_policy,
                    coverage_query=coverage_query,
                )
                legacy_fallback.setdefault("diagnostics", {})[
                    "lightrag_fallback_reason"
                ] = "empty_after_scope_mapping"
                legacy_fallback["diagnostics"]["selected_engine"] = "legacy_fallback"
                legacy_fallback["diagnostics"][
                    "policy_reason"
                ] = "lightrag_empty_after_scope_mapping"
                return legacy_fallback
            legacy = await retrieval_module._retrieve_legacy_for_agent(
                rag=rag,
                settings=settings,
                query=query,
                collection_id=collection_id,
                retrieval_mode=retrieval_mode,
                focus_document_ids=focus_document_ids,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                include_visual_boost=include_visual_boost,
                query_variants=query_variants,
                coverage_policy=coverage_policy,
                coverage_query=coverage_query,
            )
            merged_sources = _merge_retrieved_documents(
                list(lightrag_result.get("documents") or []),
                list(legacy.get("documents") or []),
            )
            context_budget = _context_budget(answer_intent, answer_depth)
            composed = compose_retrieval_context(
                merged_sources,
                query=query,
                max_sources=context_budget["max_sources"],
                max_chars=context_budget["max_chars"],
                max_chars_per_source=context_budget["max_chars_per_source"],
                max_chunks_per_document=context_budget["max_chunks_per_document"],
                min_figures=4 if has_visual_intent(query) else (2 if include_visual_boost else 0),
                min_visual_pages=visual_page_reservation_count(
                    query,
                    [
                        item
                        for item in merged_sources
                        if item.get("visual_page")
                        or item.get("artifact_type") == "visual_page"
                    ],
                ),
            )
            lightrag_result["mode"] = "dual"
            lightrag_result["documents"] = composed.sources
            lightrag_result["context_text"] = composed.context_text
            lightrag_result["context_stats"] = composed.stats
            lightrag_result["diagnostics"] = {
                **lightrag_result.get("diagnostics", {}),
                "dual_run": True,
                "legacy_mode": legacy.get("mode"),
                "legacy_source_count": len(legacy.get("documents") or []),
                "merged_source_count": len(composed.sources),
            }
            return lightrag_result
        except RuntimeError as exc:
            # An uninitialized optional graph store may fall back to the local
            # hybrid path.  Retrieval/provider/provenance failures must remain
            # visible; treating every RuntimeError as "LightRAG unavailable"
            # silently hides quota and data-integrity failures.
            if "lightrag is not initialized" not in str(exc).lower():
                raise
            lightrag_uninitialized = True

    legacy = await retrieval_module._retrieve_legacy_for_agent(
        rag=rag,
        settings=settings,
        query=query,
        collection_id=collection_id,
        retrieval_mode=retrieval_mode,
        focus_document_ids=focus_document_ids,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        include_visual_boost=include_visual_boost,
        query_variants=query_variants,
        coverage_policy=coverage_policy,
        coverage_query=coverage_query,
    )
    legacy_diagnostics = legacy.setdefault("diagnostics", {})
    legacy_diagnostics["selected_engine"] = "legacy"
    legacy_diagnostics["policy_reason"] = (
        "lightrag_uninitialized_local_fallback"
        if lightrag_uninitialized
        else policy_reason
    )
    return legacy


def _select_retrieval_engine(
    *,
    configured_engine: str,
    retrieval_mode: str,
    answer_intent: str | None,
    focus_document_ids: list[str] | None,
    prefer_legacy_tables: bool,
) -> str:
    """Choose a retrieval path without spending an additional router call.

    Scope resolution has already happened before this function runs.  The
    automatic policy therefore uses stable structural signals rather than an
    LLM guess: focused/direct QA uses fast hybrid retrieval, while discovery
    and cross-document reasoning use LightRAG.
    """

    configured = (configured_engine or "auto").strip().lower()
    if configured not in {"auto", "lightrag", "legacy", "dual"}:
        raise ValueError(f"Unsupported retrieval engine: {configured_engine}")

    requested_mode = (retrieval_mode or "auto").strip().lower()
    if requested_mode == "fts":
        return "legacy"
    if configured != "auto":
        return configured
    if prefer_legacy_tables or requested_mode == "hybrid":
        return "legacy"

    focus_count = len(dict.fromkeys(focus_document_ids or []))
    # A decomposed comparison branch is already scoped to exactly one
    # canonical paper.  Running LightRAG again inside each branch pays for
    # provider-backed graph keyword extraction without widening or improving
    # that known scope.  Use the fast local hybrid path for this case; retain
    # graph navigation for unscoped discovery, true multi-document requests
    # and structure inference.
    if focus_count == 1 and answer_intent == "compare":
        return "legacy"
    if answer_intent in {"compare", "infer_structure"} or focus_count >= 2:
        return "lightrag"
    if focus_count == 1:
        return "legacy"
    return "lightrag"


def _retrieval_engine_policy_reason(
    *,
    configured_engine: str,
    retrieval_mode: str,
    answer_intent: str | None,
    focus_document_ids: list[str] | None,
    prefer_legacy_tables: bool,
) -> str:
    """Explain the deterministic engine choice without another router call."""

    configured = (configured_engine or "auto").strip().lower()
    requested_mode = (retrieval_mode or "auto").strip().lower()
    if requested_mode == "fts":
        return "explicit_fts_mode"
    if configured != "auto":
        return f"configured_{configured}"
    if prefer_legacy_tables:
        return "focused_table_evidence"
    if requested_mode == "hybrid":
        return "explicit_hybrid_mode"

    focus_count = len(dict.fromkeys(focus_document_ids or []))
    if focus_count == 1 and answer_intent == "compare":
        return "scoped_compare_fast_path"
    if answer_intent in {"compare", "infer_structure"} or focus_count >= 2:
        return "cross_document_reasoning"
    if focus_count == 1:
        return "focused_single_document_fast_path"
    return "unscoped_graph_discovery"


def _exhaustive_short_document_pages(
    *,
    rag: RagService,
    focus_document_ids: list[str] | None,
    coverage_policy: DocumentCoveragePolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Project every active text page for a bounded short-document request.

    ``search_page_coverage`` is intentionally ranked and bounded.  That is
    correct for ordinary retrieval, but a document-wide question over a
    short paper needs a census rather than the first few lexical hits.  Use
    the document's active page count as a conservative signal: only replace
    the ranked frontier when every focused document fits inside the existing
    coverage budget.  Long documents keep the normal ranked behaviour.

    The methods are optional for lightweight test doubles and older service
    implementations.  Any metadata/projection failure leaves the canonical
    retrieval path untouched.
    """

    if not focus_document_ids or not coverage_policy.enabled:
        return [], {}
    counts_fn = getattr(rag, "document_page_counts", None)
    pages_fn = getattr(rag, "list_page_coverage", None)
    if not callable(counts_fn) or not callable(pages_fn):
        return [], {}
    document_ids = list(dict.fromkeys(str(item) for item in focus_document_ids if str(item).strip()))
    if not document_ids:
        return [], {}
    try:
        page_counts = {
            str(document_id): int(count)
            for document_id, count in (counts_fn(document_ids) or {}).items()
            if str(document_id).strip() and int(count) > 0
        }
        if set(page_counts) != set(document_ids):
            return [], page_counts
        aggregation_limit = max(
            int(coverage_policy.page_limit),
            min(int(coverage_policy.aggregation_page_limit), 64),
        )
        if any(count > aggregation_limit for count in page_counts.values()):
            return [], page_counts
        pages = pages_fn(
            document_ids=document_ids,
            page_limit=max(page_counts.values(), default=coverage_policy.page_limit),
        )
        return list(pages or []), page_counts
    except Exception:  # pragma: no cover - defensive compatibility boundary
        logger.debug("Exhaustive short-document page projection unavailable", exc_info=True)
        return [], {}


async def _retrieve_legacy_for_agent(
    *,
    rag: RagService,
    settings: Settings,
    query: str,
    collection_id: str | None,
    retrieval_mode: str,
    focus_document_ids: list[str] | None = None,
    answer_intent: str | None = None,
    answer_depth: str | None = None,
    include_visual_boost: bool = False,
    prefer_tables: bool = False,
    query_variants: list[tuple[str, str]] | None = None,
    coverage_policy: DocumentCoveragePolicy | None = None,
    coverage_query: str | None = None,
) -> dict:
    facet_requirements = _context_facet_requirements(
        coverage_query or query, focus_document_ids, answer_intent,
    )
    mode = retrieval_mode.lower()
    if mode not in {"auto", "hybrid", "fts"}:
        mode = "auto"

    diagnostics: dict = {}
    embeddings = deps._embedding_provider(settings)
    single_doc_focus = bool(focus_document_ids) and len(focus_document_ids) == 1
    coverage_enabled = bool(coverage_policy and coverage_policy.enabled and single_doc_focus)
    hybrid_top_k = (
        coverage_policy.page_limit
        if coverage_enabled and coverage_policy is not None
        else (10 if single_doc_focus else 6)
    )
    if mode in {"auto", "hybrid"}:
        try:
            retrieval_store = deps._retrieval_store(settings)
            has_translation = any(origin == "translation" for origin, _text in (query_variants or ()))
            if len(query_variants or ()) > 1 and (settings.dual_query_rrf_enabled or has_translation):
                hybrid = await rag.search_hybrid_queries(
                    queries=list(query_variants or ()),
                    top_k=hybrid_top_k,
                    collection_id=collection_id,
                    document_ids=focus_document_ids or None,
                    retrieval_store=retrieval_store,
                    embeddings=embeddings,
                    rerank=settings.rerank_enabled,
                    rerank_mode=settings.rerank_mode,
                    cross_encoder_model_path=settings.rerank_cross_encoder_path,
                    rerank_max_candidates=settings.rerank_max_candidates,
                    visual_boost=include_visual_boost,
                    coverage_mode=coverage_enabled,
                    coverage_page_limit=(coverage_policy.page_limit if coverage_policy else 24),
                    rrf_k=settings.dual_query_rrf_k,
                )
            else:
                hybrid = await rag.search_hybrid(
                    query=query,
                    top_k=hybrid_top_k,
                    collection_id=collection_id,
                    document_ids=focus_document_ids or None,
                    retrieval_store=retrieval_store,
                    embeddings=embeddings,
                    rerank=settings.rerank_enabled,
                    rerank_mode=settings.rerank_mode,
                    cross_encoder_model_path=settings.rerank_cross_encoder_path,
                    rerank_max_candidates=settings.rerank_max_candidates,
                    visual_boost=include_visual_boost,
                    coverage_mode=coverage_enabled,
                    coverage_page_limit=(coverage_policy.page_limit if coverage_policy else 24),
                )
            context_budget = _context_budget(answer_intent, answer_depth)
            if single_doc_focus:
                context_budget = {
                    **context_budget,
                    "max_sources": max(context_budget["max_sources"], 6),
                    "max_chunks_per_document": max(context_budget["max_chunks_per_document"], 4),
                    "max_chars": max(context_budget["max_chars"], 5_200),
                }
            coverage_results: list[dict[str, Any]] = []
            exhaustive_pages: list[dict[str, Any]] = []
            page_counts: dict[str, int] = {}
            visual_page_results: list[dict[str, Any]] = []
            visual_page_error: str | None = None
            if coverage_enabled and coverage_policy is not None:
                context_budget = {
                    **context_budget,
                    "max_sources": max(context_budget["max_sources"], coverage_policy.max_sources),
                    "max_chars": max(context_budget["max_chars"], coverage_policy.max_chars),
                    # Coverage pages are census units, not ordinary prose
                    # chunks. Keep each unit compact so a complete bounded
                    # frontier can fit in the prompt/map stage.
                    "max_chars_per_source": min(
                        context_budget["max_chars_per_source"],
                        coverage_policy.max_chars_per_source,
                    ),
                    "max_chunks_per_document": max(
                        context_budget["max_chunks_per_document"],
                        coverage_policy.page_limit,
                    ),
                }
            first_stage_results = list(
                hybrid.get("candidate_results") or hybrid.get("results") or []
            )
            visual_scope_ids = list(
                focus_document_ids
                or hybrid.get("selected_document_ids")
                or [
                    str(item.get("document_id"))
                    for item in first_stage_results
                    if item.get("document_id")
                ]
                or []
            )
            visual_readiness_required = _visual_required_by_readiness(
                rag,
                visual_scope_ids,
            )
            visual_policy_required = _should_merge_visual_sources(
                query=query,
                include_visual_boost=include_visual_boost,
            ) or visual_readiness_required
            if coverage_enabled and coverage_policy is not None:
                coverage_results = rag.search_page_coverage(
                    query,
                    page_limit=coverage_policy.page_limit,
                    document_ids=focus_document_ids,
                )
                exhaustive_pages, page_counts = _exhaustive_short_document_pages(
                    rag=rag,
                    focus_document_ids=focus_document_ids,
                    coverage_policy=coverage_policy,
                )
                if exhaustive_pages:
                    # Keep the final context bounded, but allow a short/medium
                    # document's complete page frontier to reach the generic
                    # map/reduce verifier.  The per-source truncation remains
                    # in force, so this is still a measured character budget.
                    page_budget = min(
                        32_000,
                        max(
                            context_budget["max_chars"],
                            len(exhaustive_pages)
                            * (context_budget["max_chars_per_source"] + 180),
                        ),
                    )
                    context_budget = {
                        **context_budget,
                        "max_sources": max(
                            context_budget["max_sources"], len(exhaustive_pages)
                        ),
                        "max_chars": page_budget,
                    }
                    coverage_results = _merge_retrieved_documents(
                        exhaustive_pages,
                        coverage_results,
                    )
                first_stage_results = _merge_retrieved_documents(
                    coverage_results,
                    first_stage_results,
                )
                if settings.visual_retrieval_enabled and visual_policy_required:
                    visual_page_limit = min(
                        max(
                            coverage_policy.max_images,
                            coverage_policy.page_limit,
                            max(page_counts.values(), default=0),
                        ),
                        64,
                    )
                    visual_page_results, visual_page_error = await retrieval_module._retrieve_visual_pages_for_agent(
                        rag=rag,
                        settings=settings,
                        query=query,
                        collection_id=collection_id,
                        focus_document_ids=focus_document_ids,
                        top_k=visual_page_limit,
                    )
                    first_stage_results = _merge_page_coverage_sources(
                        first_stage_results,
                        visual_page_results,
                    )
            # Visual retrieval is useful for ordinary figure/diagram turns as
            # well as document-wide page aggregation.  Keep it opt-in and
            # only activate it for an explicit visual request; the default
            # flag is off, and plain text/table turns never pay this cost.
            if (
                settings.visual_retrieval_enabled
                and not visual_page_results
                and visual_policy_required
            ):
                visual_page_results, visual_page_error = await retrieval_module._retrieve_visual_pages_for_agent(
                    rag=rag,
                    settings=settings,
                    query=query,
                    collection_id=collection_id,
                    focus_document_ids=focus_document_ids,
                    top_k=min(max(4, context_budget["max_sources"]), 12),
                )
                first_stage_results = _merge_page_coverage_sources(
                    first_stage_results,
                    visual_page_results,
                )
            # Preserve the canonical artifact/page hit, but add a bounded
            # active-revision text companion from the same page so captions or
            # visual summaries have enough surrounding context for grounded
            # inference. Older test doubles may not implement this optional
            # helper; in that case the existing path remains compatible.
            expand_page_context = getattr(rag, "expand_with_page_context", None)
            if callable(expand_page_context):
                first_stage_results = expand_page_context(
                    first_stage_results,
                    query=query,
                    max_chars=1_200,
                    max_chunks=2,
                )
            expand_page_artifacts = getattr(rag, "expand_with_page_artifacts", None)
            if callable(expand_page_artifacts):
                first_stage_results = expand_page_artifacts(
                    first_stage_results,
                    query=query,
                    max_chars=2_400,
                    max_artifacts=3,
                )
            expanded_results = rag.expand_with_neighbor_chunks(
                first_stage_results,
                window=context_budget["neighbor_window"],
                max_neighbor_chars=context_budget["max_neighbor_chars"],
                query=query,
            )
            expand_references = getattr(rag, "expand_with_referenced_artifacts", None)
            if callable(expand_references):
                expanded_results = expand_references(
                    expanded_results, query=query, max_artifacts=context_budget["max_sources"],
                )
            table_reserve = 1 if prefer_tables else 0
            if prefer_tables and focus_document_ids:
                expanded_results, inventory_count = _merge_focused_table_inventory(
                    rag=rag,
                    sources=expanded_results,
                    focus_document_ids=focus_document_ids,
                    query=query,
                    description_query=coverage_query or query,
                )
                table_reserve = max(table_reserve, inventory_count)
            composed = compose_retrieval_context(
                expanded_results,
                query=query,
                max_sources=context_budget["max_sources"],
                max_chars=context_budget["max_chars"],
                max_chars_per_source=context_budget["max_chars_per_source"],
                max_chunks_per_document=context_budget["max_chunks_per_document"],
                min_figures=4 if has_visual_intent(query) else (2 if include_visual_boost else 0),
                min_tables=table_reserve,
                # One focused document: show its tables whole (OCR'd scans
                # produce long tables; a cut table reads as "rows missing").
                max_table_chars=9_000 if single_doc_focus else 4_200,
                min_visual_pages=visual_page_reservation_count(
                    query,
                    visual_page_results,
                ),
                coverage_mode=coverage_enabled,
                required_document_ids=focus_document_ids if facet_requirements else None,
                required_facets_by_document=facet_requirements,
            )
            sources = list(composed.sources)
            # A focused result request may have retrieved only one semantically
            # nearby table (often an ablation).  Keep the complete, bounded
            # main-result inventory available to the deterministic renderer so
            # table choice is based on caption semantics rather than model
            # sampling or whichever table happened to win first-stage rank.
            if (
                prefer_tables
                and focus_document_ids
                and _is_canonical_result_table_request(query)
            ):
                sources, _ = _merge_focused_table_inventory(
                    rag=rag,
                    sources=sources,
                    focus_document_ids=focus_document_ids,
                    query=query,
                    description_query=coverage_query or query,
                )
            # Readiness-only activation needs page pixels, but does not imply
            # that canonical figure/table asset expansion should run for a
            # fact-shaped query. Keep that more expensive artifact path tied
            # to explicit visual intent/boost; the optional visual page
            # channel above still runs for image-only projections.
            if focus_document_ids and _should_merge_visual_sources(
                query=query,
                include_visual_boost=include_visual_boost,
            ):
                sources = await _merge_focused_figures(
                    rag=rag,
                    settings=settings,
                    query=query,
                    collection_id=collection_id,
                    sources=sources,
                    focus_document_ids=focus_document_ids,
                    min_figures=4 if has_visual_intent(query) else 2,
                    answer_intent=answer_intent,
                )
            return {
                "mode": "hybrid",
                "documents": sources,
                "context_text": composed.context_text,
                "context_stats": {
                    **composed.stats,
                    "figure_source_count": sum(1 for item in sources if item.get("figure_id")),
                },
                "diagnostics": {
                    "selected_document_ids": hybrid.get("selected_document_ids", []),
                    "forced_document_ids": hybrid.get("forced_document_ids", []),
                    "retrieval_channels": hybrid.get("retrieval_channels", []),
                    "document_card_count": len(hybrid.get("document_card_results", [])),
                    "context_expanded_results": sum(1 for item in expanded_results if item.get("expanded_content")),
                    "first_stage_candidate_count": len(first_stage_results),
                    "dual_query_rrf": bool(hybrid.get("dual_query_rrf")),
                    "query_origins": hybrid.get("query_origins", []),
                    "evidence_obligation_plan": hybrid.get(
                        "evidence_obligation_plan"
                    ),
                    "coverage_mode": coverage_enabled,
                    "coverage_reason": coverage_policy.reason if coverage_policy else None,
                    "coverage_page_count": sum(
                        1
                        for item in sources
                        if item.get("coverage_mode") and item.get("page_number") is not None
                    ),
                    "coverage_lexical_page_count": len(coverage_results),
                    "coverage_visual_page_count": len(visual_page_results),
                    "coverage_visual_error": visual_page_error,
                    "visual_readiness_required": visual_readiness_required,
                    "coverage_expected_page_counts": page_counts,
                    "coverage_exhaustive_page_count": len(exhaustive_pages),
                },
                **(
                    {"evidence_bundle": hybrid["evidence_bundle"]}
                    if hybrid.get("evidence_bundle")
                    else {}
                ),
            }
        except (LanceDBUnavailable, EmbeddingError) as exc:
            if mode == "hybrid":
                raise
            diagnostics["hybrid_fallback_reason"] = str(exc)

    chunks = [
        chunk.__dict__
        for chunk in rag.search(
            query,
            top_k=(coverage_policy.page_limit * 8 if coverage_enabled and coverage_policy else 6),
            document_ids=focus_document_ids or None,
        )
    ]
    context_budget = _context_budget(answer_intent, answer_depth)
    if coverage_enabled and coverage_policy is not None:
        context_budget = {
            **context_budget,
            "max_sources": max(context_budget["max_sources"], coverage_policy.max_sources),
            "max_chars": max(context_budget["max_chars"], coverage_policy.max_chars),
            "max_chars_per_source": min(
                context_budget["max_chars_per_source"], coverage_policy.max_chars_per_source
            ),
            "max_chunks_per_document": max(
                context_budget["max_chunks_per_document"], coverage_policy.page_limit
            ),
        }
    if coverage_enabled and coverage_policy is not None:
        coverage_results = rag.search_page_coverage(
            query,
            page_limit=coverage_policy.page_limit,
            document_ids=focus_document_ids,
        )
        exhaustive_pages, page_counts = _exhaustive_short_document_pages(
            rag=rag,
            focus_document_ids=focus_document_ids,
            coverage_policy=coverage_policy,
        )
        if exhaustive_pages:
            page_budget = min(
                32_000,
                max(
                    context_budget["max_chars"],
                    len(exhaustive_pages)
                    * (context_budget["max_chars_per_source"] + 180),
                ),
            )
            context_budget = {
                **context_budget,
                "max_sources": max(context_budget["max_sources"], len(exhaustive_pages)),
                "max_chars": page_budget,
            }
            coverage_results = _merge_retrieved_documents(
                exhaustive_pages,
                coverage_results,
            )
        chunks = _merge_retrieved_documents(coverage_results, chunks)
        visual_scope_ids = list(
            focus_document_ids
            or [
                str(item.get("document_id"))
                for item in chunks
                if item.get("document_id")
            ]
            or []
        )
        visual_readiness_required = _visual_required_by_readiness(
            rag,
            visual_scope_ids,
        )
        visual_policy_required = _should_merge_visual_sources(
            query=query,
            include_visual_boost=include_visual_boost,
        ) or visual_readiness_required
        if settings.visual_retrieval_enabled and visual_policy_required:
            visual_page_limit = min(
                max(
                    coverage_policy.max_images,
                    coverage_policy.page_limit,
                    max(page_counts.values(), default=0),
                ),
                64,
            )
            visual_pages, visual_error = await retrieval_module._retrieve_visual_pages_for_agent(
                rag=rag,
                settings=settings,
                query=query,
                collection_id=collection_id,
                focus_document_ids=focus_document_ids,
                top_k=visual_page_limit,
            )
            chunks = _merge_page_coverage_sources(chunks, visual_pages)
        else:
            visual_pages, visual_error = [], None
    else:
        page_counts = {}
        exhaustive_pages = []
        visual_scope_ids = list(
            focus_document_ids
            or [
                str(item.get("document_id"))
                for item in chunks
                if item.get("document_id")
            ]
            or []
        )
        visual_readiness_required = _visual_required_by_readiness(
            rag,
            visual_scope_ids,
        )
        visual_policy_required = _should_merge_visual_sources(
            query=query,
            include_visual_boost=include_visual_boost,
        ) or visual_readiness_required
        if settings.visual_retrieval_enabled and visual_policy_required:
            visual_pages, visual_error = await retrieval_module._retrieve_visual_pages_for_agent(
                rag=rag,
                settings=settings,
                query=query,
                collection_id=collection_id,
                focus_document_ids=focus_document_ids,
                top_k=min(max(4, context_budget["max_sources"]), 12),
            )
            chunks = _merge_page_coverage_sources(chunks, visual_pages)
        else:
            visual_pages, visual_error = [], None
    expand_page_context = getattr(rag, "expand_with_page_context", None)
    if callable(expand_page_context):
        chunks = expand_page_context(
            chunks,
            query=query,
            max_chars=1_200,
            max_chunks=2,
        )
    expand_page_artifacts = getattr(rag, "expand_with_page_artifacts", None)
    if callable(expand_page_artifacts):
        chunks = expand_page_artifacts(
            chunks,
            query=query,
            max_chars=2_400,
            max_artifacts=3,
        )
    expanded_chunks = rag.expand_with_neighbor_chunks(
        chunks,
        window=context_budget["neighbor_window"],
        max_neighbor_chars=context_budget["max_neighbor_chars"],
        query=query,
    )
    expand_references = getattr(rag, "expand_with_referenced_artifacts", None)
    if callable(expand_references):
        expanded_chunks = expand_references(
            expanded_chunks, query=query, max_artifacts=context_budget["max_sources"],
        )
    table_reserve = 1 if prefer_tables else 0
    if prefer_tables and focus_document_ids:
        expanded_chunks, inventory_count = _merge_focused_table_inventory(
            rag=rag,
            sources=expanded_chunks,
            focus_document_ids=focus_document_ids,
            query=query,
            description_query=coverage_query or query,
        )
        table_reserve = max(table_reserve, inventory_count)
    composed = compose_retrieval_context(
        expanded_chunks,
        query=query,
        max_sources=context_budget["max_sources"],
        max_chars=context_budget["max_chars"],
        max_chars_per_source=context_budget["max_chars_per_source"],
        max_chunks_per_document=context_budget["max_chunks_per_document"],
        min_tables=table_reserve,
        min_visual_pages=visual_page_reservation_count(query, visual_pages),
        coverage_mode=coverage_enabled,
        required_document_ids=focus_document_ids if facet_requirements else None,
        required_facets_by_document=facet_requirements,
    )
    if (
        prefer_tables
        and focus_document_ids
        and _is_canonical_result_table_request(query)
    ):
        expanded_chunks, _ = _merge_focused_table_inventory(
            rag=rag,
            sources=list(composed.sources),
            focus_document_ids=focus_document_ids,
            query=query,
            description_query=coverage_query or query,
        )
        composed = compose_retrieval_context(
            expanded_chunks,
            query=query,
            max_sources=context_budget["max_sources"],
            max_chars=context_budget["max_chars"],
            max_chars_per_source=context_budget["max_chars_per_source"],
            max_chunks_per_document=context_budget["max_chunks_per_document"],
            min_tables=max(1, table_reserve),
            min_visual_pages=visual_page_reservation_count(query, visual_pages),
            coverage_mode=coverage_enabled,
            required_document_ids=focus_document_ids if facet_requirements else None,
            required_facets_by_document=facet_requirements,
        )
    return {
        "mode": "fts",
        "documents": composed.sources,
        "context_text": composed.context_text,
        "context_stats": composed.stats,
        "diagnostics": {
            **diagnostics,
            "context_expanded_results": sum(1 for item in expanded_chunks if item.get("expanded_content")),
            "coverage_mode": coverage_enabled,
            "coverage_reason": coverage_policy.reason if coverage_policy else None,
            "coverage_page_count": sum(
                1
                for item in composed.sources
                if item.get("coverage_mode") and item.get("page_number") is not None
            ),
            "coverage_visual_page_count": len(visual_pages),
            "coverage_visual_error": visual_error,
            "visual_readiness_required": visual_readiness_required,
            "coverage_expected_page_counts": page_counts,
            "coverage_exhaustive_page_count": len(exhaustive_pages),
        },
    }


_GUESSED_ALIAS_SOURCES = frozenset({"title_description", "fuzzy_name"})


async def _evidence_backed_focus(
    rag: RagService, settings: Settings, query: str, guessed_id: str, collection_id: str | None,
) -> tuple[str, str | None]:
    """("keep" | "switch" | "drop", document) for a guessed focus, judged by
    the library-wide top 10: switch when another document holds ≥ 3 hits and
    at least twice the guess's; drop the focus when the guess has none and
    no document stands out; otherwise keep."""
    try:
        found = await rag.search_hybrid(
            query=query,
            top_k=10,
            collection_id=collection_id,
            retrieval_store=deps._retrieval_store(settings),
            embeddings=deps._embedding_provider(settings),
            rerank=settings.rerank_enabled,
            rerank_mode=settings.rerank_mode,
            cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
        )
    except Exception:  # noqa: BLE001 - the check is advisory
        return "keep", None
    ids = [str(item.get("document_id") or "") for item in (found.get("results") or [])[:10] if item.get("document_id")]
    if not ids:
        return "keep", None
    counts = Counter(ids)
    guessed = counts.get(guessed_id, 0)
    top, top_count = counts.most_common(1)[0]
    # Switching to the top-ranked document proved unsafe (long, dense
    # documents dominate rankings); only a guess retrieval never sees is
    # dropped so the library-wide search decides.
    if guessed == 0 and top_count >= 3:
        return "drop", None
    return "keep", None


def _dominant_document(documents: list[dict[str, Any]]) -> list[str]:
    """The document most retrieved passages come from (≥ 60%, ≥ 3), e.g.
    "the RAG paper's equations" answered without an explicit focus."""
    ids = [str(item.get("document_id") or "") for item in documents if item.get("document_id")]
    if len(ids) < 3:
        return []
    top, count = Counter(ids).most_common(1)[0]
    return [top] if count / len(ids) >= 0.6 else []


async def _document_priority_passages(
    rag: RagService, settings: Settings, document_ids: list[str], queries: list[str],
) -> dict[str, list[str]]:
    """Per document, the chunks a document-scoped hybrid search returns for
    the question and each sub-question (one fact each)."""
    passages: dict[str, list[str]] = {}
    for document_id in document_ids[:FULL_DOCUMENT_MAX_DOCUMENTS]:
        for query in list(dict.fromkeys(item for item in queries if item))[:5]:
            try:
                found = await rag.search_hybrid(
                    query=query, top_k=4, document_ids=[document_id],
                    retrieval_store=deps._retrieval_store(settings), embeddings=deps._embedding_provider(settings),
                )
            except Exception:  # noqa: BLE001 - priority passages are an optimisation
                continue
            for item in found.get("results") or []:
                content = str(item.get("content") or "").strip()
                if content and content not in passages.setdefault(document_id, []):
                    passages[document_id].append(content)
    return passages


def _full_document_sources(
    rag: RagService, settings: Settings, focus_document_ids: list[str] | None, query: str = "",
    priority_passages: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Whole text of up to three focused documents; a document too long for
    its share of the budget contributes the sections most related to the
    question instead (shorter documents stay whole and leave room)."""
    if not getattr(settings, "full_document_context_enabled", False):
        return []
    ids = list(dict.fromkeys(str(item) for item in focus_document_ids or [] if item))
    if not ids or len(ids) > FULL_DOCUMENT_MAX_DOCUMENTS:
        return []
    builder = getattr(rag, "document_full_text", None)
    if not callable(builder):
        return []
    whole = {document_id: builder(document_id, max_chars=10**9) for document_id in ids}
    if any(source is None for source in whole.values()):
        return []
    remaining = FULL_DOCUMENT_MAX_TOTAL_CHARS
    chosen: dict[str, dict[str, Any]] = {}
    by_length = sorted(ids, key=lambda item: len(str(whole[item].get("content") or "")))
    for position, document_id in enumerate(by_length):
        share = min(settings.full_document_context_max_chars, remaining // (len(by_length) - position))
        source = whole[document_id]
        if len(str(source.get("content") or "")) > share:
            source = (
                builder(document_id, max_chars=share, query=query,
                        priority_passages=(priority_passages or {}).get(document_id))
                if query else None
            )
        if source is None:
            return []
        chosen[document_id] = source
        remaining -= len(str(source.get("content") or ""))
    return [chosen[document_id] for document_id in ids]


def _merge_focused_table_inventory(
    *,
    rag: RagService,
    sources: list[dict[str, Any]],
    focus_document_ids: list[str],
    query: str | None = None,
    description_query: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Gather scoped exact/inventory tables or independently typed main results.

    ``description_query`` is the user's own wording. A misremembered table
    number is corrected only from what the user described, never from terms
    the retrieval query adds ("table results metrics").
    """

    requested_numbers = _requested_table_numbers(str(query or ""))
    # A structured/numeric question may be answered by a diagram or prose.
    # The table-intent hint is not authority to replace its candidate ranking
    # with the entire table inventory or discard companion figures.
    if query and not (
        requested_numbers
        or _wants_table_inventory(query)
        or _wants_main_result_evidence(query)
    ):
        return sources, 0
    existing_by_id = {
        str(item.get("table_id")): item
        for item in sources
        if item.get("table_id")
    }
    inventory: list[dict[str, Any]] = []
    inventory_ids: set[str] = set()
    for document_id in dict.fromkeys(focus_document_ids):
        document = rag.get_document(document_id) or {}
        tables = rag.list_document_tables(document_id)
        document_numbers = _described_table_numbers(
            str(description_query or query or ""), tables, requested_numbers,
        )
        if requested_numbers:
            tables = [
                table
                for table in tables
                if _canonical_table_number(table) in document_numbers
            ]
        elif query and not _wants_table_inventory(query) and _wants_main_result_evidence(query):
            main_tables = [
                table
                for table in tables
                if table_result_role(table) == "main"
            ]
            # Do not turn an unknown/main-results miss into an arbitrary
            # whole-inventory injection (e.g. corpus statistics). Keep only the
            # best-matching few per document: every reserved table displaces a
            # retrieved source from a 3-5 source context.
            main_tables.sort(
                key=lambda table: _table_query_match_score(
                    {**table, "content": str(table.get("markdown") or "")}, query=query,
                ),
                reverse=True,
            )
            tables = main_tables[:_MAIN_RESULT_TABLES_PER_DOCUMENT]

        for table in tables:
            table_id = str(table.get("id") or "")
            if not table_id or table_id in inventory_ids:
                continue
            inventory_ids.add(table_id)
            existing = existing_by_id.get(table_id)
            if existing is not None:
                inventory.append(existing)
                continue
            metadata = table.get("metadata") if isinstance(table.get("metadata"), dict) else {}
            caption = str(table.get("caption") or "").strip()
            markdown = str(table.get("markdown") or "").strip()
            content = "\n".join(
                part
                for part in [
                    f"table_type: {metadata.get('table_type')}" if metadata.get("table_type") else "",
                    f"filename: {document.get('filename')}" if document.get("filename") else "",
                    f"caption: {caption}" if caption else "",
                    f"content:\n{markdown}" if markdown else "",
                ]
                if part
            )
            inventory.append(
                {
                    "chunk_id": f"table:{table_id}",
                    "document_id": document_id,
                    "revision_id": table.get("revision_id") or document.get("active_revision_id"),
                    "source_path": document.get("source_path"),
                    "filename": document.get("filename"),
                    "content": content,
                    "page_number": table.get("page_number"),
                    "chunk_type": "table",
                    "artifact_type": "table",
                    "caption": caption or None,
                    "table_id": table_id,
                    "table_index": table.get("table_index"),
                    "image_path": table.get("image_path")
                    or metadata.get("image_path")
                    or metadata.get("table_image_path"),
                    "metadata": metadata,
                    "retrieval_channels": ["sqlite_table_inventory"],
                }
            )

    remainder = [
        item
        for item in sources
        if (
            not item.get("table_id")
            or str(item.get("table_id")) not in inventory_ids
        )
        and not (
            requested_numbers
            and (
                item.get("table_id")
                or item.get("artifact_type") == "table"
            )
        )
        and not (
            requested_numbers
            and caption_identifies_figure(str(item.get("caption") or ""))
        )
    ]
    return [*inventory, *remainder], len(inventory)


def _retrieval_has_table_sources(retrieval: dict[str, Any] | None) -> bool:
    if not retrieval:
        return False
    for document in retrieval.get("documents") or []:
        if not isinstance(document, dict):
            continue
        if document.get("table_id"):
            return True
        if str(document.get("artifact_type") or "").lower() == "table":
            return True
        if str(document.get("chunk_type") or "").lower() == "table":
            return True
        if str(document.get("chunk_id") or "").startswith("table:"):
            return True
    return False


def _is_low_signal_figure(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    quality_status = document.get("quality_status") or metadata.get("quality_status")
    asset_kind = str(document.get("asset_kind") or metadata.get("asset_kind") or "").lower()
    is_content = document.get("is_content", metadata.get("is_content"))
    if quality_status == "rejected" or is_content is False:
        return True
    if asset_kind in {"branding", "logo", "decorative", "publisher_mark"}:
        return True
    caption = str(document.get("caption") or "").strip().lower()
    if caption.startswith("figure extracted from page"):
        return True
    if "visual fallback" in caption:
        return True
    content = str(document.get("content") or "").strip().lower()
    if content.startswith("figure extracted from page"):
        return True
    return "visual fallback" in content


def _curate_figure_sources(
    documents: list[dict[str, Any]],
    *,
    answer_intent: str,
    focus_document_ids: list[str],
    query: str,
) -> list[dict[str, Any]]:
    non_figures = [document for document in documents if not document.get("figure_id")]
    figures = [document for document in documents if document.get("figure_id")]
    if not figures:
        return documents

    preferred_figure = requested_figure_number(query)
    focus_set = {str(document_id) for document_id in focus_document_ids if document_id}

    # Hard-scope to focused papers unless this is an explicit multi-doc compare.
    if focus_set and not (answer_intent == "compare" and len(focus_set) >= 2):
        focused = [figure for figure in figures if str(figure.get("document_id") or "") in focus_set]
        if focused:
            figures = focused

    ranked_figures = sorted(
        figures,
        key=lambda item: figure_relevance_score(
            item,
            preferred_figure_number=preferred_figure,
            query=query,
        ),
        reverse=True,
    )

    clean_figures = [figure for figure in ranked_figures if not _is_low_signal_figure(figure)]

    # A request for one/best/most-relevant visual is a presentation constraint:
    # apply it only after focus scoping, relevance ranking and the quality gate.
    # This remains corpus-agnostic and never maps a paper name to a Figure N.
    if wants_single_figure(query):
        return non_figures + clean_figures[:1]

    if answer_intent == "compare" and len(focus_document_ids) >= 2:
        by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for figure in clean_figures:
            document_id = str(figure.get("document_id") or "")
            if document_id:
                by_document[document_id].append(figure)

        selected: list[dict[str, Any]] = []
        for document_id in focus_document_ids:
            doc_figures = by_document.get(document_id) or []
            if not doc_figures:
                continue
            selected.append(doc_figures[0])
        return non_figures + selected

    # Exact figure number within the already-focused document (no cross-paper).
    if preferred_figure is not None:
        exact = [
            figure
            for figure in clean_figures
            if _source_figure_number(figure) == preferred_figure
        ]
        if exact:
            return non_figures + exact[:1]

    document_ids = {str(figure.get("document_id")) for figure in clean_figures if figure.get("document_id")}
    if len(document_ids) >= 2 and not focus_set:
        by_document = defaultdict(list)
        for figure in clean_figures:
            document_id = str(figure.get("document_id") or "")
            if document_id:
                by_document[document_id].append(figure)

        selected = []
        for document_id in sorted(by_document.keys()):
            selected.extend(by_document[document_id][:2])
        return non_figures + selected[:4]

    return non_figures + clean_figures[:3]


def _filter_figure_sources(
    documents: list[dict[str, Any]],
    *,
    allowed_document_ids: list[str] | None,
    answer_intent: str,
) -> list[dict[str, Any]]:
    allowed = set(allowed_document_ids or [])
    if answer_intent == "compare" and allowed:
        permitted = allowed
    elif allowed:
        permitted = allowed
    else:
        permitted = set(_text_source_document_ids(documents))

    filtered: list[dict[str, Any]] = []
    for document in documents:
        if not document.get("figure_id"):
            filtered.append(document)
            continue
        document_id = document.get("document_id")
        if _is_low_signal_figure(document) and document_id not in permitted:
            continue
        if not permitted or document_id in permitted:
            filtered.append(document)
    return filtered


def _enrich_visual_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for document in documents:
        item = dict(document)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not item.get("image_path"):
            image_path = metadata.get("image_path") or metadata.get("table_image_path")
            if image_path:
                item["image_path"] = image_path
        artifact_type = canonical_artifact_type(item)
        if artifact_type == "table" and item.get("image_path") and not item.get("image_url"):
            figure_id = metadata.get("figure_id")
            if figure_id:
                item["image_url"] = f"/rag/figures/{figure_id}/image"
        figure_id = item.get("figure_id")
        if figure_id and item.get("image_path") and not item.get("image_url"):
            item["image_url"] = f"/rag/figures/{figure_id}/image"
        if figure_id:
            for field in (
                "figure_label",
                "figure_number",
                "quality_status",
                "asset_kind",
                "is_content",
                "is_complete",
                "logical_group_id",
            ):
                if item.get(field) is None and metadata.get(field) is not None:
                    item[field] = metadata[field]
            item["caption"] = best_figure_caption(
                caption=item.get("caption"),
                content=item.get("content"),
                visual_summary=item.get("visual_summary"),
                figure_number=_source_figure_number(item),
                figure_index=item.get("figure_index"),
            )
        enriched.append(item)
    return enriched


def _source_figure_number(source: dict[str, Any]) -> int | None:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    value = source.get("figure_number") or metadata.get("figure_number")
    if isinstance(value, int):
        return value
    label = extract_figure_label(
        str(source.get("figure_label") or metadata.get("figure_label") or source.get("caption") or "")
    )
    return label.number if label else None


def _cached_retrieval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics.update(
        {
            "cache_hit": True,
            "cache_type": "last_sources",
            "cached_mode": payload.get("mode"),
        }
    )
    return {
        "mode": "last_sources_cache",
        "documents": payload.get("documents") or [],
        "context_text": payload.get("context_text") or "",
        "context_stats": payload.get("context_stats") or {},
        "diagnostics": diagnostics,
    }


def _exact_cached_retrieval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics.update(
        {
            "cache_hit": True,
            "cache_type": "exact_retrieval",
            "cached_mode": payload.get("mode"),
        }
    )
    return {
        "mode": "retrieval_cache",
        "documents": payload.get("documents") or [],
        "context_text": payload.get("context_text") or "",
        "context_stats": payload.get("context_stats") or {},
        "diagnostics": diagnostics,
    }


def _restore_cached_paper_evidence_runtime(
    *,
    settings: Settings,
    retrieval: dict[str, Any],
    focus_document_ids: list[str],
    original_task: str | None = None,
    answer_intent: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Rehydrate card coverage/contract state after a retrieval cache hit.

    Cache payloads already preserve canonical sources and card diagnostics, but
    the previous runtime initialized progressive-output state only on a fresh
    card lookup.  The same query could therefore lose per-paper validation
    merely because it became faster.  Re-read the tiny card index from SQLite
    and derive the current coverage from provenance, never from a filename or
    a query-specific paper rule.
    """

    diagnostics = retrieval.get("diagnostics") or {}
    cached_coverage = diagnostics.get("paper_facet_coverage") or []
    selected_engine = str(diagnostics.get("selected_engine") or "")
    if not cached_coverage and selected_engine != "paper_evidence_cards":
        return "", []
    ordered_ids = list(dict.fromkeys(str(value) for value in focus_document_ids if value))
    if not ordered_ids:
        return "", []
    # Retrieval caches preserve evidence, not the semantic contract of a
    # future turn. A follow-up may reuse the same sources while changing from
    # task/contribution to architecture/results. Recompute that contract from
    # the current user task; the cached list is only a backward-compatible
    # fallback for callers that do not have turn text.
    requested_facets = (
        requested_paper_facets(
            original_task,
            answer_intent=answer_intent,
            focused_document_count=len(ordered_ids),
        )
        if original_task
        else list(diagnostics.get("requested_facets") or [])
    )
    if not requested_facets:
        requested_facets = list(
            dict.fromkeys(
                str(facet)
                for source in retrieval.get("documents") or []
                for facet in source.get("evidence_facets") or []
                if facet
            )
        )
    if not requested_facets:
        return "", []
    service = deps.PaperEvidenceService(
        settings.sqlite_db_path,
        schema_version=settings.paper_evidence_card_schema_version,
        prompt_version=settings.paper_evidence_card_prompt_version,
    )
    cards, coverage = service.coverage_matrix(ordered_ids, requested_facets)
    payload = [
        {
            "document_id": item.document_id,
            "requested_facets": item.requested_facets,
            "covered_facets": item.covered_facets,
            "missing_facets": item.missing_facets,
            "stale": item.stale,
            "status": item.status,
        }
        for item in coverage
    ]
    navigation = service.render_navigation_context(
        cards,
        requested_facets=requested_facets,
    )
    return navigation, payload


def _retrieval_cache_key(
    *,
    normalized_query: str,
    original_query: str,
    collection_id: str | None,
    focus_document_ids: list[str],
    retrieval_mode: str,
    index_fingerprint: str,
) -> str:
    raw = json.dumps(
        {
            "query": normalized_query,
            # The rewritten query is intentionally compact and can be equal for
            # several artifact requests (for example Table 2 vs Table 4). Keep
            # the raw request contract in the key so an exact artifact cache
            # cannot satisfy a semantically different follow-up.
            "request_contract": _retrieval_request_contract(original_query),
            "collection_id": collection_id,
            "focus_document_ids": sorted(focus_document_ids),
            "retrieval_mode": retrieval_mode.lower(),
            "index_fingerprint": index_fingerprint,
            "cache_version": RETRIEVAL_CACHE_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_cache_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _retrieval_request_contract(query: str) -> dict[str, Any]:
    """Return cache-relevant retrieval obligations from the raw user turn.

    Rewrites are allowed to normalize wording, but they must not erase
    artifact identity. This contract deliberately records only retrieval
    semantics (modality, exact artifact numbers and inventory intent), not the
    answer text or paper-specific aliases.
    """

    return {
        "table_intent": bool(_wants_result_tables(query)),
        "table_inventory": bool(_wants_table_inventory(query)),
        "table_numbers": list(_requested_table_numbers(query)),
        "figure_number": requested_figure_number(query),
        "exact_artifact": bool(_task_requires_exact_artifact(query)),
        "visual_intent": bool(has_visual_intent(query)),
    }

