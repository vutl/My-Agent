from collections import defaultdict
from collections.abc import AsyncIterator, Callable
import asyncio
from dataclasses import replace
import hashlib
import json
import logging
import re
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.events import sse_event
from app.core.network import request_client_is_loopback
from app.llm.ollama_client import OllamaError
from app.llm.openai_client import get_llm_client
from app.rag.context import compose_retrieval_context
from app.rag.figure_caption import (
    best_figure_caption,
    figure_relevance_score,
    requested_figure_number,
)
from app.rag.figure_quality import extract_figure_label
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.rag.paper_facets import facet_query_terms, requested_paper_facets
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.agent_run_store import AgentRunStore
from app.services.agent_service import AgentService, AnswerStreamChunk
from app.services.chat_history import ChatHistory
from app.services.evidence_validator import validate_answer_claims, validate_retrieval_evidence
from app.services.document_scope_service import resolve_document_scope
from app.services.debug_trace_service import DebugTraceRecorder
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
    wants_single_figure,
)
from app.services.rag_service import RagService, caption_identifies_figure
from app.services.retrieval_agent_service import (
    RetrievalBranch,
    SecondRetrievalPlan,
    plan_second_retrieval_pass,
    plan_retrieval_decomposition,
    smart_retrieval_enabled,
)
from app.services.tool_decision_service import (
    IntentRouterService,
    ToolDecision,
    answer_needs_local_fallback,
)
from app.services.conversation_state import ConversationStateStore
from app.services.conversation_memory import ConversationMemoryStore, schedule_memory_fold
from app.services.conversation_runtime import ConversationRuntimeGate
from app.services.long_term_memory import HistoricalConversationSearch, MemoryItemStore
from app.services.paper_evidence_service import PaperEvidenceService
from app.lightrag.bridge import LightRAGBridge

router = APIRouter(prefix="/agent", tags=["agent"])
# Bump whenever routing/composition policy changes.  Cached documents are a
# bounded prompt projection, so reusing an entry produced by an older context
# policy can silently discard evidence even though the underlying index did
# not change.
RETRIEVAL_CACHE_VERSION = "v21"
logger = logging.getLogger(__name__)


class AgentRunRequest(BaseModel):
    conversation_id: str | None = None
    task: str = Field(min_length=1)
    mode: str = "research"
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    allowed_tools: list[str] = Field(default_factory=list)
    require_confirmation: bool = True
    collection_id: str | None = None
    retrieval_mode: str = "auto"
    agent_reasoning: str = "auto"
    debug_trace: bool = False


def get_agent_service(settings: Annotated[Settings, Depends(get_settings)]) -> AgentService:
    client = get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return AgentService(client=client, default_model=settings.default_model)


def get_chat_history(settings: Annotated[Settings, Depends(get_settings)]) -> ChatHistory:
    return ChatHistory(settings.sqlite_db_path)


def get_conversation_state_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationStateStore:
    return ConversationStateStore(settings.sqlite_db_path)


def get_conversation_memory_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationMemoryStore:
    return ConversationMemoryStore(settings.sqlite_db_path)


def get_conversation_runtime_gate(request: Request) -> ConversationRuntimeGate:
    return request.app.state.conversation_runtime_gate


def get_historical_conversation_search(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HistoricalConversationSearch:
    return HistoricalConversationSearch(settings.sqlite_db_path)


def get_long_term_memory_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryItemStore:
    return MemoryItemStore(settings.sqlite_db_path)


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> RagService:
    return RagService(settings.sqlite_db_path, artifact_root=settings.artifacts_path)


def get_agent_run_store(settings: Annotated[Settings, Depends(get_settings)]) -> AgentRunStore:
    return AgentRunStore(settings.sqlite_db_path)


def get_query_rewrite_service(settings: Annotated[Settings, Depends(get_settings)]) -> QueryRewriteService:
    client = get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return QueryRewriteService(client=client, default_model=settings.default_model)


def get_intent_router(settings: Annotated[Settings, Depends(get_settings)]) -> IntentRouterService:
    # Same stack as answer LLM (9router/GPT) — do not force local Ollama.
    client = get_llm_client(
        provider=settings.router_llm_provider or settings.llm_provider or "openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=min(settings.request_timeout_seconds, 45.0),
    )
    return IntentRouterService(
        client=client,
        default_model=settings.router_model or settings.default_model,
    )


@router.post("/run/stream")
async def run_agent_stream(
    request: AgentRunRequest,
    http_request: Request,
    service: Annotated[AgentService, Depends(get_agent_service)],
    history: Annotated[ChatHistory, Depends(get_chat_history)],
    state_store: Annotated[ConversationStateStore, Depends(get_conversation_state_store)],
    memory_store: Annotated[ConversationMemoryStore, Depends(get_conversation_memory_store)],
    runtime_gate: Annotated[ConversationRuntimeGate, Depends(get_conversation_runtime_gate)],
    historical_search: Annotated[
        HistoricalConversationSearch,
        Depends(get_historical_conversation_search),
    ],
    long_term_memory: Annotated[MemoryItemStore, Depends(get_long_term_memory_store)],
    rag: Annotated[RagService, Depends(get_rag_service)],
    run_store: Annotated[AgentRunStore, Depends(get_agent_run_store)],
    query_rewriter: Annotated[QueryRewriteService, Depends(get_query_rewrite_service)],
    intent_router: Annotated[IntentRouterService, Depends(get_intent_router)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    if request.debug_trace and not settings.agent_debug_trace_enabled:
        raise HTTPException(status_code=403, detail="Agent debug trace is disabled")
    if request.debug_trace and not request_client_is_loopback(http_request):
        raise HTTPException(
            status_code=403,
            detail="Agent debug trace is restricted to a loopback client",
        )
    # Allocate/resolve the thread before constructing the stream. The actual
    # history snapshot and user-message write happen under the per-thread gate,
    # so two API callers cannot build prompts from the same stale boundary.
    conversation_id = history.ensure_conversation(request.conversation_id, request.task)

    async def unlocked_event_stream():
        model = request.model or service.default_model
        run_store.fail_stale_running_runs(
            conversation_id=conversation_id,
            older_than_seconds=900,
        )
        previous_messages = history.list_messages(conversation_id)
        working_state = state_store.get_effective_working_state(
            conversation_id,
            request.task,
        )
        loaded_working_state = working_state
        scope_resolution = resolve_document_scope(
            rag,
            query=request.task,
            collection_id=request.collection_id,
            working_state=working_state,
            previous_messages=previous_messages,
        )
        if scope_resolution.authoritative and scope_resolution.document_ids:
            scoped_filenames = _filenames_for_document_ids(
                rag,
                list(scope_resolution.document_ids),
            )
            scoped_topic = (
                " / ".join(scope_resolution.labels)
                if len(scope_resolution.labels) >= 2
                else scope_resolution.labels[0]
                if scope_resolution.labels
                else working_state.active_topic
            )
            working_state = replace(
                working_state,
                active_document_ids=list(scope_resolution.document_ids),
                active_topic=scoped_topic,
                active_filenames=scoped_filenames,
                last_answer_intent=(
                    _classify_answer_intent(request.task)
                ),
                referent_document_ids=(
                    list(scope_resolution.document_ids)
                    if scope_resolution.must_cover_all
                    else working_state.referent_document_ids
                ),
                referent_filenames=(
                    scoped_filenames
                    if scope_resolution.must_cover_all
                    else working_state.referent_filenames
                ),
                referent_topic=(
                    scoped_topic
                    if scope_resolution.must_cover_all
                    else working_state.referent_topic
                ),
            )
        memory = memory_store.get_memory(conversation_id)
        conversation_context = format_recent_conversation(
            previous_messages,
            max_messages=12,
            max_chars=7200,
        )
        recent_message_ids = tuple(message.id for message in previous_messages[-12:])
        historical_block = historical_search.prompt_block_for_context(
            request.task,
            current_conversation_id=conversation_id,
            exclude_message_ids=recent_message_ids,
            limit=4,
            max_chars=4800,
        )
        long_term_block = long_term_memory.prompt_block(
            request.task,
            conversation_id=conversation_id,
            limit=10,
            max_chars=3600,
            min_confidence=0.5,
        )
        router_memory_notes = "\n\n".join(
            part
            for part in (
                memory.prompt_block(include_summary=False),
                long_term_block,
                historical_block,
            )
            if part
        )
        context_parts: list[str] = []
        working_block = working_state.prompt_block()
        if working_block:
            context_parts.append(working_block)
        if long_term_block:
            context_parts.append(long_term_block)
        memory_block = memory.prompt_block()
        if memory_block:
            context_parts.append(memory_block)
        if historical_block:
            context_parts.append(historical_block)
        context_parts.append(conversation_context)
        conversation_context = "\n\n".join(context_parts)
        user_message = history.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.task,
            model=model,
        )
        run = run_store.create_run(
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            mode=request.mode,
            metadata={
                "model": model,
                "allowed_tools": request.allowed_tools,
                "require_confirmation": request.require_confirmation,
                "collection_id": request.collection_id,
                "retrieval_mode": request.retrieval_mode,
                "debug_trace": request.debug_trace,
            },
        )
        debug_trace = DebugTraceRecorder(
            store=run_store,
            run_id=run.id,
            enabled=request.debug_trace,
            max_bytes=settings.agent_debug_trace_max_bytes,
            retention_hours=settings.agent_debug_trace_retention_hours,
            max_runs=settings.agent_debug_trace_max_runs,
            exact_secrets=tuple(
                item
                for item in (settings.openai_api_key, settings.lightrag_llm_api_key)
                if item
            ),
        )
        debug_trace.record_scope(
            loaded_working_state=loaded_working_state.to_dict(),
            effective_working_state=working_state.to_dict(),
            resolution=scope_resolution.to_dict(),
        )

        try:
            yield sse_event(
                "run.started",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "user_message_id": user_message.id,
                    "mode": request.mode,
                    "model": model,
                    "allowed_tools": request.allowed_tools,
                    "require_confirmation": request.require_confirmation,
                    "collection_id": request.collection_id,
                    "retrieval_mode": request.retrieval_mode,
                    "debug_trace_enabled": request.debug_trace,
                },
            )
        except (asyncio.CancelledError, GeneratorExit):
            run_store.cancel_run(run.id)
            debug_trace.outcome("cancelled", "stream_cancelled")
            raise

        assistant_chunks: list[str] = []
        assistant_sources: list[dict[str, Any]] = []
        pending_working_update: dict[str, Any] | None = None
        pending_retrieval_cache: dict[str, Any] | None = None
        memory_eligible = True
        try:
            scope_failure_message = _document_scope_failure_message(
                rag,
                scope_resolution,
            )
            if scope_failure_message:
                memory_eligible = False
                yield sse_event(
                    "agent.scope.rejected",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "document_scope": scope_resolution.to_dict(),
                    },
                )
                assistant_chunks.append(scope_failure_message)
                yield sse_event("message.delta", {"delta": scope_failure_message})
                assistant_record = history.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=scope_failure_message,
                    model=model,
                    sources=[],
                )
                run_store.complete_run(run.id, scope_failure_message)
                debug_trace.outcome("completed")
                yield sse_event(
                    "run.completed",
                    {"run_id": run.id, "conversation_id": conversation_id},
                )
                return
            recent_document_ids = (
                list(scope_resolution.document_ids)
                or working_state.active_document_ids
                or run_store.latest_retrieved_document_ids(
                    conversation_id,
                    expected_collection_id=request.collection_id,
                )
            )
            if request.collection_id is not None:
                collection_ids = set(rag.collection_document_ids(request.collection_id))
                recent_document_ids = [
                    document_id
                    for document_id in recent_document_ids
                    if document_id in collection_ids
                ]
            router_started_at = time.perf_counter()
            tool_decision = await intent_router.decide(
                task=request.task,
                mode=request.mode,
                previous_messages=previous_messages,
                has_recent_retrieval=bool(recent_document_ids),
                allowed_tools=request.allowed_tools,
                recent_document_ids=recent_document_ids,
                working_topic=working_state.active_topic,
                working_filenames=list(working_state.active_filenames or []),
                conversation_summary=memory.summary,
                recent_turn_notes=router_memory_notes or None,
                model=settings.router_model,
                resolved_document_ids=(
                    list(scope_resolution.document_ids)
                    if scope_resolution.authoritative
                    else None
                ),
            )
            routing_decision = tool_decision.to_dict()
            debug_trace.record_route(routing_decision)
            yield sse_event(
                "timing",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "stage": "router",
                    "ms": _elapsed_ms(router_started_at),
                    "route": routing_decision.get("route"),
                },
            )
            yield sse_event(
                "agent.route.decided",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    **routing_decision,
                    "working_state": working_state.to_dict(),
                    "loaded_working_state": loaded_working_state.to_dict(),
                    "document_scope": scope_resolution.to_dict(),
                },
            )
            if not routing_decision["use_local_retrieval"]:
                yield sse_event(
                    "retrieval.skipped",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "reason": routing_decision["reason"],
                    },
                )
                async for event in _stream_graph_and_answer(
                    service=service,
                    run_store=run_store,
                    request=request,
                    run_id=run.id,
                    conversation_id=conversation_id,
                    user_message_id=user_message.id,
                    task=request.task,
                    resolved_task=request.task,
                    conversation_context=conversation_context,
                    answer_intent=tool_decision.answer_intent,
                    answer_depth=tool_decision.answer_depth,
                    retrieved_docs=[],
                    tool_decision=routing_decision,
                    assistant_chunks=assistant_chunks,
                    debug_trace=debug_trace,
                ):
                    yield event
                if _should_run_local_fallback(
                    tool_decision=tool_decision,
                    answer="".join(assistant_chunks),
                    task=request.task,
                ):
                    fallback_state: dict[str, Any] = {"grounded": True}
                    async for event in _stream_local_rag_fallback(
                        service=service,
                        run_store=run_store,
                        rag=rag,
                        settings=settings,
                        request=request,
                        run_id=run.id,
                        conversation_id=conversation_id,
                        user_message_id=user_message.id,
                        task=request.task,
                        conversation_context=conversation_context,
                        assistant_chunks=assistant_chunks,
                        focus_document_ids=list(working_state.active_document_ids),
                        focus_topic=working_state.active_topic,
                        fallback_state=fallback_state,
                        debug_trace=debug_trace,
                    ):
                        yield event
                    memory_eligible = bool(fallback_state.get("grounded"))
                    assistant_sources = list(fallback_state.get("sources") or [])
                assistant_message = "".join(assistant_chunks)
                if not assistant_message.strip():
                    raise RuntimeError("LLM returned an empty answer")
                assistant_record = history.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=assistant_message,
                    model=model,
                    sources=assistant_sources,
                )
                run_store.complete_run(run.id, assistant_message)
                debug_trace.outcome("completed")
                if memory_eligible:
                    memory_status = _queue_l2_memory_safely(
                        memory_store=memory_store,
                        settings=settings,
                        conversation_id=conversation_id,
                        user_text=request.task,
                        assistant_text=assistant_message,
                        user_message_id=user_message.id,
                        assistant_message_id=assistant_record.id,
                        working_state=working_state,
                    )
                    yield sse_event(
                        "agent.memory.queued",
                        {
                            "run_id": run.id,
                            "conversation_id": conversation_id,
                            **memory_status,
                        },
                    )
                yield sse_event("run.completed", {"run_id": run.id, "conversation_id": conversation_id})
                return

            rewrite_started_at = time.perf_counter()
            if scope_resolution.authoritative and scope_resolution.document_ids:
                rewrite = QueryRewriteResult(
                    original_query=request.task,
                    standalone_query=request.task,
                    is_followup=scope_resolution.source == "plural_referent",
                    current_topic=(
                        " / ".join(scope_resolution.labels)
                        if len(scope_resolution.labels) >= 2
                        else scope_resolution.labels[0]
                        if scope_resolution.labels
                        else None
                    ),
                    required_entities=list(scope_resolution.labels),
                    use_last_sources=scope_resolution.source == "plural_referent",
                    answer_intent=_classify_answer_intent(request.task),
                    answer_depth=_classify_answer_depth(request.task),
                    rewrite_used=False,
                    diagnostics={"reason": "pre_resolved_document_scope"},
                )
            else:
                rewrite = await query_rewriter.rewrite(
                    query=request.task,
                    previous_messages=previous_messages,
                    model=request.model,
                    working_topic=working_state.active_topic,
                    working_document_hint=", ".join(working_state.active_filenames or []),
                )
            catalog_focus_ids = (
                list(scope_resolution.document_ids)
                if scope_resolution.authoritative
                else []
            )
            if catalog_focus_ids:
                rewrite = _apply_document_scope_to_rewrite(
                    rewrite,
                    scope=scope_resolution,
                )
            # Prefer rewrite answer policy when rewrite LLM ran; also keep strong
            # heuristic intents (structure/compare) over a generic router label.
            if rewrite.rewrite_used:
                answer_intent = rewrite.answer_intent
                answer_depth = rewrite.answer_depth
            elif rewrite.answer_intent in {"infer_structure", "compare"}:
                answer_intent = rewrite.answer_intent
                answer_depth = rewrite.answer_depth
            else:
                answer_intent = tool_decision.answer_intent
                answer_depth = tool_decision.answer_depth
            debug_trace.record_rewrite(
                {
                    "original_query": rewrite.original_query,
                    "standalone_query": rewrite.standalone_query,
                    "is_followup": rewrite.is_followup,
                    "current_topic": rewrite.current_topic,
                    "required_entities": rewrite.required_entities,
                    "use_last_sources": rewrite.use_last_sources,
                    "answer_intent": answer_intent,
                    "answer_depth": answer_depth,
                    "diagnostics": rewrite.diagnostics,
                    "focus_document_ids": catalog_focus_ids,
                }
            )
            yield sse_event(
                "timing",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "stage": "rewrite",
                    "ms": _elapsed_ms(rewrite_started_at),
                    "rewrite_used": rewrite.rewrite_used,
                    "reason": rewrite.diagnostics.get("reason"),
                },
            )
            entity_focus_ids = catalog_focus_ids or _resolve_query_document_focus(
                rag,
                rewrite=rewrite,
                collection_id=request.collection_id,
                existing_focus=[],
            )
            focus_document_ids = entity_focus_ids
            # L1 sticky: same-paper follow-ups (benchmark/tables/architecture) must
            # stay on active docs. Prefer sticky over weak entity hits like Acc/CCC.
            rewrite_topic = str(rewrite.current_topic or "").strip().lower()
            active_topic = str(working_state.active_topic or "").strip().lower()
            rewrite_preserves_working_focus = bool(
                rewrite.use_last_sources
                and active_topic
                and (
                    not rewrite_topic
                    or rewrite_topic in active_topic
                    or active_topic in rewrite_topic
                )
            )
            same_paper_followup = (
                working_state.has_active_docs
                and not catalog_focus_ids
                and (rewrite.diagnostics or {}).get("reason")
                not in {
                    "topic_switch",
                    "explicit_document_target",
                    "explicit_catalog_document_target",
                    "catalog_document_mentions",
                    "plural_document_referent",
                }
                and (
                    rewrite_preserves_working_focus
                    or not _query_names_new_paper(request.task, working_state)
                )
            )
            if same_paper_followup:
                focus_document_ids = list(working_state.active_document_ids)
            elif rewrite.use_last_sources and not focus_document_ids and working_state.active_document_ids:
                focus_document_ids = list(working_state.active_document_ids)
            if request.collection_id is not None:
                collection_ids = set(rag.collection_document_ids(request.collection_id))
                focus_document_ids = [
                    document_id
                    for document_id in focus_document_ids
                    if document_id in collection_ids
                ]
            sticky_topic = rewrite.current_topic or working_state.active_topic
            sticky_entities = list(rewrite.required_entities or [])
            if sticky_topic and sticky_topic not in sticky_entities:
                sticky_entities = [sticky_topic, *sticky_entities]
            retrieval_configuration = _retrieval_index_configuration(settings)
            reuse_index_fingerprint = run_store.index_fingerprint(
                collection_id=request.collection_id,
                document_ids=focus_document_ids or None,
                configuration=retrieval_configuration,
            )
            cached_retrieval = (
                run_store.latest_retrieval_output(
                    conversation_id,
                    expected_collection_id=request.collection_id,
                    expected_retrieval_mode=request.retrieval_mode,
                    expected_index_fingerprint=reuse_index_fingerprint,
                )
                if _should_reuse_last_retrieval(rewrite, original_query=request.task)
                else None
            )
            if cached_retrieval:
                cached_doc_ids = set(_document_ids_from_retrieval(cached_retrieval))
                if focus_document_ids and cached_doc_ids:
                    missing_focus = [
                        doc_id for doc_id in focus_document_ids if doc_id not in cached_doc_ids
                    ]
                    if missing_focus:
                        cached_retrieval = None
                elif rewrite.use_last_sources and not focus_document_ids:
                    focus_document_ids = _document_ids_from_retrieval(cached_retrieval)
            elif rewrite.is_followup and rewrite.use_last_sources:
                focus_document_ids = (
                    focus_document_ids
                    or list(working_state.active_document_ids)
                    or run_store.latest_retrieved_document_ids(
                        conversation_id,
                        expected_collection_id=request.collection_id,
                    )
                )
            retrieval_query = enrich_retrieval_query(
                rewrite.standalone_query,
                topic=sticky_topic,
                entities=sticky_entities,
                answer_intent=answer_intent,
                focus_document_ids=focus_document_ids,
            )
            if sticky_topic and sticky_topic.lower() not in retrieval_query.lower():
                retrieval_query = f"{sticky_topic} {retrieval_query}".strip()
            wants_result_tables = _wants_result_tables(request.task)
            if wants_result_tables and cached_retrieval and not _retrieval_has_table_sources(
                cached_retrieval
            ):
                # A follow-up can reuse document focus, but a previous
                # text/figure-only payload cannot satisfy a new table ask.
                cached_retrieval = None
            include_visual = (
                "retrieve_visual_assets" in routing_decision["selected_tools"]
                and (has_visual_intent(request.task) or not wants_result_tables)
            )
            normalized_retrieval_query = _normalize_cache_query(retrieval_query)
            index_fingerprint = run_store.index_fingerprint(
                collection_id=request.collection_id,
                document_ids=focus_document_ids or None,
                configuration=retrieval_configuration,
            )
            retrieval_cache_key = _retrieval_cache_key(
                normalized_query=normalized_retrieval_query,
                collection_id=request.collection_id,
                focus_document_ids=focus_document_ids,
                retrieval_mode=request.retrieval_mode,
                index_fingerprint=index_fingerprint,
            )
            exact_cached_retrieval = None
            if not cached_retrieval:
                exact_cached_retrieval = run_store.get_retrieval_cache(retrieval_cache_key)
                if (
                    wants_result_tables
                    and exact_cached_retrieval
                    and not _retrieval_has_table_sources(exact_cached_retrieval)
                ):
                    exact_cached_retrieval = None
            yield sse_event(
                "query.rewritten",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "original_query": rewrite.original_query,
                    "standalone_query": rewrite.standalone_query,
                    "is_followup": rewrite.is_followup,
                    "current_topic": rewrite.current_topic,
                    "required_entities": rewrite.required_entities,
                    "use_last_sources": rewrite.use_last_sources,
                    "answer_intent": answer_intent,
                    "answer_depth": answer_depth,
                    "focus_document_ids": focus_document_ids,
                    "rewrite_used": rewrite.rewrite_used,
                    "diagnostics": rewrite.diagnostics,
                },
            )
            yield sse_event(
                "tool.started",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "tool_name": "search_local_docs",
                    "input": {
                        "query": retrieval_query,
                        "collection_id": request.collection_id,
                        "retrieval_mode": request.retrieval_mode,
                    },
                },
            )
            if include_visual:
                yield sse_event(
                    "tool.started",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "tool_name": "retrieve_visual_assets",
                        "input": {"query": retrieval_query, "collection_id": request.collection_id},
                    },
                )
            yield sse_event(
                "retrieval.started",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "query": retrieval_query,
                    "original_task": request.task,
                    "focus_document_ids": focus_document_ids,
                    "tool_name": "search_local_docs",
                    "cache_candidate": bool(cached_retrieval or exact_cached_retrieval),
                },
            )
            retrieval_started_at = time.perf_counter()
            paper_evidence_context = ""
            paper_evidence_coverage: list[dict[str, Any]] = []
            if cached_retrieval:
                retrieval = _cached_retrieval_payload(cached_retrieval)
            elif exact_cached_retrieval:
                retrieval = _exact_cached_retrieval_payload(exact_cached_retrieval)
            else:
                card_result = None
                if not _task_requires_exact_artifact(request.task):
                    card_result = await _retrieve_with_paper_evidence_cards(
                        rag=rag,
                        settings=settings,
                        query=retrieval_query,
                        original_task=request.task,
                        collection_id=request.collection_id,
                        retrieval_mode=request.retrieval_mode,
                        focus_document_ids=focus_document_ids,
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_tables=wants_result_tables and bool(focus_document_ids),
                    )
                if card_result is not None:
                    retrieval, paper_evidence_context, paper_evidence_coverage = card_result
                    yield sse_event(
                        "evidence.card.coverage",
                        {
                            "run_id": run.id,
                            "conversation_id": conversation_id,
                            "documents": paper_evidence_coverage,
                            "mode": retrieval["mode"],
                        },
                    )
                    for paper_coverage in paper_evidence_coverage:
                        yield sse_event(
                            "evidence.paper.ready",
                            {
                                "run_id": run.id,
                                "conversation_id": conversation_id,
                                **paper_coverage,
                            },
                        )
                else:
                    retrieval = await _retrieve_for_agent(
                        rag=rag,
                        settings=settings,
                        query=retrieval_query,
                        collection_id=request.collection_id,
                        retrieval_mode=request.retrieval_mode,
                        focus_document_ids=focus_document_ids,
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_legacy_tables=wants_result_tables and bool(focus_document_ids),
                        must_cover_all_documents=scope_resolution.must_cover_all,
                    )
            yield sse_event(
                "timing",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "stage": "retrieval",
                    "ms": _elapsed_ms(retrieval_started_at),
                    "mode": retrieval.get("mode"),
                    "cache_hit": retrieval.get("diagnostics", {}).get("cache_hit", False),
                },
            )
            retrieved_docs = _enrich_visual_sources(retrieval["documents"])
            retrieved_docs = _scope_documents_to_focus(retrieved_docs, focus_document_ids)
            retrieved_docs = _filter_figure_sources(
                retrieved_docs,
                allowed_document_ids=focus_document_ids or entity_focus_ids,
                answer_intent=answer_intent,
            )
            retrieved_docs = _curate_figure_sources(
                retrieved_docs,
                answer_intent=answer_intent,
                focus_document_ids=focus_document_ids or entity_focus_ids,
                query=retrieval_query,
            )
            retrieval["documents"] = retrieved_docs
            assistant_sources = retrieved_docs
            validation = validate_retrieval_evidence(
                documents=retrieved_docs,
                required_entities=rewrite.required_entities,
                current_topic=rewrite.current_topic,
                is_followup=rewrite.is_followup,
                focus_document_ids=focus_document_ids,
                require_all_focus_documents=(
                    scope_resolution.must_cover_all
                    or (answer_intent == "compare" and len(focus_document_ids) >= 2)
                ),
            )
            smart_allowed = smart_retrieval_enabled(
                request.agent_reasoning,
                answer_intent=answer_intent,
                query=request.task,
                focus_document_ids=focus_document_ids or entity_focus_ids,
            )
            initial_diagnostics = retrieval.get("diagnostics") or {}
            graph_bridge_metadata = initial_diagnostics.get("graph_bridge_metadata")
            if not isinstance(graph_bridge_metadata, list):
                graph_bridge_metadata = None
            covered_facets = initial_diagnostics.get("covered_facets")
            if not isinstance(covered_facets, list):
                covered_facets = None
            second_pass = plan_second_retrieval_pass(
                retrieval_query=retrieval_query,
                original_task=request.task,
                topic=rewrite.current_topic,
                entities=rewrite.required_entities,
                answer_intent=answer_intent,
                focus_document_ids=focus_document_ids or entity_focus_ids,
                validation=validation,
                smart_allowed=smart_allowed,
                documents=retrieved_docs,
                graph_bridge_metadata=graph_bridge_metadata,
                covered_facets=covered_facets,
                completed_hops=1,
                previous_queries=[],
                must_cover_all=scope_resolution.must_cover_all,
                retry_budget_available=settings.agentic_retrieval_max_hops > 1,
            )
            retry_performed = False
            retry_discarded = False
            if second_pass is not None:
                initial_retrieval = retrieval
                initial_docs = list(retrieved_docs)
                initial_validation = validation
                second_hop_started_at = time.perf_counter()
                retry_branches = _bounded_second_retrieval_branches(
                    plan=second_pass,
                    fallback_focus_document_ids=(
                        focus_document_ids or entity_focus_ids
                    ),
                    max_subqueries=settings.agentic_retrieval_max_subqueries,
                )
                retry_performed = True
                yield sse_event(
                    "retrieval.retrying",
                    _second_retrieval_retry_payload(
                        run_id=run.id,
                        conversation_id=conversation_id,
                        plan=second_pass,
                        branches=retry_branches,
                        max_hops=settings.agentic_retrieval_max_hops,
                        missing_entities=validation.missing_entities,
                        previous_focus_document_ids=focus_document_ids,
                        agent_reasoning=request.agent_reasoning,
                        smart_allowed=smart_allowed,
                    ),
                )
                branch_retrievals, branch_diagnostics = (
                    await _execute_second_retrieval_branches(
                        branches=retry_branches,
                        rag=rag,
                        settings=settings,
                        collection_id=request.collection_id,
                        retrieval_mode=request.retrieval_mode,
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_legacy_tables=wants_result_tables,
                    )
                )
                prepared_branch_retrievals: list[dict[str, Any]] = []
                for branch, branch_retrieval in zip(
                    retry_branches,
                    branch_retrievals,
                    strict=True,
                ):
                    branch_documents = _enrich_visual_sources(
                        branch_retrieval["documents"]
                    )
                    branch_documents = _scope_documents_to_focus(
                        branch_documents,
                        branch.focus_document_ids,
                    )
                    branch_documents = _filter_figure_sources(
                        branch_documents,
                        allowed_document_ids=branch.focus_document_ids or None,
                        answer_intent=answer_intent,
                    )
                    branch_documents = _curate_figure_sources(
                        branch_documents,
                        answer_intent=answer_intent,
                        focus_document_ids=branch.focus_document_ids,
                        query=branch.query,
                    )
                    branch_retrieval["documents"] = branch_documents
                    branch_index = len(prepared_branch_retrievals)
                    branch_diagnostics[branch_index]["scoped_source_count"] = len(
                        branch_documents
                    )
                    branch_diagnostics[branch_index]["returned_document_ids"] = list(
                        dict.fromkeys(
                            str(document.get("document_id"))
                            for document in branch_documents
                            if document.get("document_id")
                        )
                    )
                    prepared_branch_retrievals.append(branch_retrieval)

                new_evidence = _new_retrieval_evidence(
                    initial_documents=initial_docs,
                    additional_documents=[
                        document
                        for branch_retrieval in prepared_branch_retrievals
                        for document in branch_retrieval.get("documents") or []
                    ],
                )
                if new_evidence:
                    retrieval = _compose_accumulated_retrieval(
                        query=retrieval_query,
                        retrievals=[initial_retrieval, *prepared_branch_retrievals],
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_tables=wants_result_tables,
                        required_document_ids=list(
                            focus_document_ids or entity_focus_ids
                        ),
                    )
                    retrieved_docs = list(retrieval["documents"])
                    assistant_sources = retrieved_docs
                    validation = validate_retrieval_evidence(
                        documents=retrieved_docs,
                        required_entities=rewrite.required_entities,
                        current_topic=rewrite.current_topic,
                        is_followup=rewrite.is_followup,
                        focus_document_ids=focus_document_ids or entity_focus_ids,
                        require_all_focus_documents=(
                            scope_resolution.must_cover_all
                            or (
                                answer_intent == "compare"
                                and len(focus_document_ids or entity_focus_ids) >= 2
                            )
                        ),
                    )
                else:
                    retry_discarded = True
                    retrieval = initial_retrieval
                    retrieved_docs = initial_docs
                    assistant_sources = initial_docs
                    validation = initial_validation
                second_hop_ms = _elapsed_ms(second_hop_started_at)
                parallel_second_hop = len(retry_branches) > 1
                yield sse_event(
                    "timing",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "stage": "retrieval.retry",
                        "ms": second_hop_ms,
                        "mode": retrieval.get("mode"),
                        "hop": second_pass.hop_count,
                        "parallel": parallel_second_hop,
                        "branch_count": len(retry_branches),
                        "retry_discarded": retry_discarded,
                    },
                )
                retrieval_diagnostics = retrieval.setdefault("diagnostics", {})
                retrieval_diagnostics.update(
                    _second_hop_diagnostics_payload(
                        plan=second_pass,
                        branches=retry_branches,
                        branch_diagnostics=branch_diagnostics,
                        max_hops=settings.agentic_retrieval_max_hops,
                        smart_retrieval=smart_allowed,
                        total_ms=second_hop_ms,
                        new_evidence_count=len(new_evidence),
                    )
                )
            retrieved_docs = rag.enrich_source_identities(retrieved_docs)
            retrieval["documents"] = retrieved_docs
            assistant_sources = retrieved_docs
            debug_trace.record_retrieval(
                focus_document_ids=list(focus_document_ids or entity_focus_ids),
                retrieved_document_ids=_document_ids_from_retrieval(
                    {"documents": retrieved_docs}
                ),
                validation=validation.to_dict(),
                diagnostics={
                    "mode": retrieval.get("mode"),
                    "cache_hit": bool(cached_retrieval or exact_cached_retrieval),
                    "retry_performed": retry_performed,
                    "retry_discarded": retry_discarded,
                },
            )
            visual_tool_call = None
            if include_visual:
                visual_tool_call = run_store.record_tool_call(
                    run_id=run.id,
                    tool_name="retrieve_visual_assets",
                    input_payload={
                        "query": retrieval_query,
                        "collection_id": request.collection_id,
                        "focus_document_ids": focus_document_ids,
                    },
                    output_payload={
                        "mode": "hybrid_visual_boost",
                        "results": [
                            doc
                            for doc in retrieved_docs
                            if str(doc.get("chunk_type") or "").lower() in {"figure", "table", "image"}
                        ],
                    },
                )
            if not (cached_retrieval or exact_cached_retrieval or retry_performed) and validation.valid:
                # Commit only after answer generation completes; a failed/cancelled
                # run must not seed a future retrieval cache entry.
                pending_retrieval_cache = {
                    "cache_key": retrieval_cache_key,
                    "normalized_query": normalized_retrieval_query,
                    "collection_id": request.collection_id,
                    "focus_document_ids": focus_document_ids,
                    "retrieval_mode": request.retrieval_mode,
                    "index_fingerprint": index_fingerprint,
                    "output_payload": retrieval,
                }
            tool_call = run_store.record_tool_call(
                run_id=run.id,
                tool_name="search_local_docs",
                input_payload={
                    "query": retrieval_query,
                    "original_task": request.task,
                    "focus_document_ids": focus_document_ids,
                    "top_k": 8,
                    "collection_id": request.collection_id,
                    "retrieval_mode": request.retrieval_mode,
                },
                output_payload={
                    **retrieval,
                    "query_rewrite": {
                        "original_query": rewrite.original_query,
                        "standalone_query": rewrite.standalone_query,
                        "is_followup": rewrite.is_followup,
                        "current_topic": rewrite.current_topic,
                        "required_entities": rewrite.required_entities,
                        "use_last_sources": rewrite.use_last_sources,
                        "answer_intent": answer_intent,
                        "answer_depth": answer_depth,
                        "rewrite_used": rewrite.rewrite_used,
                    },
                    "evidence_validation": validation.to_dict(),
                    "index_fingerprint": index_fingerprint,
                    "collection_id": request.collection_id,
                    "retrieval_mode": request.retrieval_mode,
                    "retry_performed": retry_performed,
                    "retry_discarded": retry_discarded,
                },
            )
            yield sse_event(
                "retrieval.completed",
                {
                    "run_id": run.id,
                    "tool_call_id": tool_call.id,
                    "conversation_id": conversation_id,
                    "query": retrieval_query,
                    "original_task": request.task,
                    "focus_document_ids": focus_document_ids,
                    "documents": retrieved_docs,
                    "retrieval_mode": retrieval["mode"],
                    "context_stats": retrieval["context_stats"],
                    "query_rewrite": {
                        "standalone_query": rewrite.standalone_query,
                        "is_followup": rewrite.is_followup,
                        "current_topic": rewrite.current_topic,
                        "required_entities": rewrite.required_entities,
                        "rewrite_used": rewrite.rewrite_used,
                        "answer_intent": answer_intent,
                        "answer_depth": answer_depth,
                    },
                    "evidence_validation": validation.to_dict(),
                    "retry_performed": retry_performed,
                    "retry_discarded": retry_discarded,
                },
            )
            yield sse_event(
                "tool.completed",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "tool_name": "search_local_docs",
                    "tool_call_id": tool_call.id,
                    "status": "completed",
                    "result_count": len(retrieved_docs),
                },
            )
            if visual_tool_call is not None:
                yield sse_event(
                    "tool.completed",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "tool_name": "retrieve_visual_assets",
                        "tool_call_id": visual_tool_call.id,
                        "status": "completed",
                        "result_count": _visual_source_count(retrieved_docs),
                    },
                )
            # Prepare L1, but commit it only after generation succeeds. A focus
            # candidate by itself is not evidence: IDs must come from returned docs.
            sticky_ids = _document_ids_from_retrieval({"documents": retrieved_docs})
            if focus_document_ids:
                sticky_set = set(sticky_ids)
                sticky_ids = [
                    document_id
                    for document_id in focus_document_ids
                    if document_id in sticky_set
                ] + [
                    document_id
                    for document_id in sticky_ids
                    if document_id not in set(focus_document_ids)
                ]
            sticky_files = _filenames_for_document_ids(rag, sticky_ids) or _filenames_from_documents(
                retrieved_docs
            )
            # Keep prior topic when rewrite topic is a metric/generic token (e.g. "benchmark").
            next_topic = rewrite.current_topic or working_state.active_topic
            if next_topic and not _is_named_entity(next_topic):
                next_topic = working_state.active_topic or next_topic
            memory_eligible = bool(validation.valid)
            if validation.valid and sticky_ids:
                pending_working_update = {
                    "document_ids": sticky_ids,
                    "topic": next_topic,
                    "filenames": sticky_files or working_state.active_filenames,
                    "answer_intent": answer_intent,
                    "source_turn_id": user_message.id,
                }
            async for event in _stream_graph_and_answer(
                service=service,
                run_store=run_store,
                request=request,
                run_id=run.id,
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                task=request.task,
                resolved_task=retrieval_query,
                conversation_context=conversation_context,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                retrieved_docs=retrieved_docs,
                tool_decision=routing_decision,
                assistant_chunks=assistant_chunks,
                focus_document_ids=focus_document_ids or entity_focus_ids or sticky_ids,
                validate_quantitative_claims=True,
                require_all_focus_documents=scope_resolution.must_cover_all,
                document_identity_resolver=lambda answer: (
                    rag.resolve_document_mentions_for_query(
                        query=answer,
                        collection_id=request.collection_id,
                        limit=32,
                    )
                ),
                debug_trace=debug_trace,
                paper_evidence_context=paper_evidence_context,
                paper_evidence_coverage=paper_evidence_coverage,
                paper_section_streaming_enabled=settings.paper_section_streaming_enabled,
            ):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            run_store.cancel_run(run.id)
            debug_trace.outcome("cancelled", "stream_cancelled")
            raise
        except OllamaError as exc:
            run_store.fail_run(run.id, str(exc))
            debug_trace.outcome("failed", str(exc))
            yield sse_event("run.failed", {"run_id": run.id, "conversation_id": conversation_id, "error": str(exc)})
            return
        except Exception as exc:
            run_store.fail_run(run.id, str(exc))
            debug_trace.outcome("failed", str(exc))
            yield sse_event("run.failed", {"run_id": run.id, "conversation_id": conversation_id, "error": str(exc)})
            return

        assistant_message = "".join(assistant_chunks)
        if not assistant_message.strip():
            error = "LLM returned an empty answer"
            run_store.fail_run(run.id, error)
            debug_trace.outcome("failed", error)
            yield sse_event(
                "run.failed",
                {"run_id": run.id, "conversation_id": conversation_id, "error": error},
            )
            return
        try:
            assistant_record = history.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_message,
                model=model,
                sources=assistant_sources,
            )
            if pending_retrieval_cache is not None:
                run_store.store_retrieval_cache(**pending_retrieval_cache)
            if pending_working_update is not None:
                working_state = state_store.update_from_retrieval(
                    conversation_id,
                    **pending_working_update,
                )
                yield sse_event(
                    "agent.working_state.updated",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        **working_state.to_dict(),
                    },
                )
            run_store.complete_run(run.id, assistant_message)
            debug_trace.outcome("completed")
        except Exception as exc:
            run_store.fail_run(run.id, f"finalization_failed:{exc}")
            debug_trace.outcome("failed", f"finalization_failed:{exc}")
            yield sse_event(
                "run.failed",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    "error": f"Could not persist completed answer: {exc}",
                },
            )
            return
        if memory_eligible:
            memory_status = _queue_l2_memory_safely(
                memory_store=memory_store,
                settings=settings,
                conversation_id=conversation_id,
                user_text=request.task,
                assistant_text=assistant_message,
                user_message_id=user_message.id,
                assistant_message_id=assistant_record.id,
                working_state=working_state,
            )
            yield sse_event(
                "agent.memory.queued",
                {
                    "run_id": run.id,
                    "conversation_id": conversation_id,
                    **memory_status,
                },
            )
        yield sse_event("run.completed", {"run_id": run.id, "conversation_id": conversation_id})

    async def event_stream():
        async with runtime_gate.turn(conversation_id):
            async for event in unlocked_event_stream():
                yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _phase4_routing_decision(
    *,
    task: str,
    mode: str,
    previous_messages: list[Any],
    has_recent_retrieval: bool,
    allowed_tools: list[str] | None = None,
    router: IntentRouterService | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Test/helper entry: inject payload or router; file_qa mode stays deterministic."""
    from app.services.tool_decision_service import decision_from_payload, decide_tools

    if payload is not None:
        return decision_from_payload(payload, allowed_tools=allowed_tools).to_dict()
    decision = await decide_tools(
        task=task,
        mode=mode,
        previous_messages=previous_messages,
        has_recent_retrieval=has_recent_retrieval,
        allowed_tools=allowed_tools,
        router=router,
    )
    return decision.to_dict()


def _schedule_l2_memory(
    *,
    memory_store: ConversationMemoryStore,
    settings: Settings,
    conversation_id: str,
    user_text: str,
    assistant_text: str,
    user_message_id: str,
    assistant_message_id: str,
    working_state: Any,
) -> Any:
    client = get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=min(settings.request_timeout_seconds, 45.0),
    )
    schedule_memory_fold(
        store=memory_store,
        conversation_id=conversation_id,
        client=client,
        model=settings.default_model,
        user_text=user_text,
        assistant_text=assistant_text,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        working_topic=getattr(working_state, "active_topic", None),
        working_filenames=list(getattr(working_state, "active_filenames", None) or []),
    )
    return memory_store.get_job(conversation_id)


def _queue_l2_memory_safely(**kwargs: Any) -> dict[str, Any]:
    """Do not turn an already-persisted answer into a failed chat on queue errors."""
    try:
        job = _schedule_l2_memory(**kwargs)
    except Exception as exc:
        logger.exception("Could not enqueue durable L2 memory")
        return {
            "status": "recovery_pending",
            "error": "durable enqueue failed; startup recovery will retry",
            "detail": " ".join(str(exc).split())[:300],
        }
    if job is None:
        return {"status": "pending"}
    return {
        "status": job.status,
        "dirty_through_seq": job.dirty_through_seq,
        "summary_through_seq": job.summary_through_seq,
        "pending_turns": max(0, job.dirty_through_seq - job.summary_through_seq),
    }


def _should_run_local_fallback(*, tool_decision: ToolDecision, answer: str, task: str) -> bool:
    if tool_decision.use_local_retrieval:
        return False
    # High-confidence chat: only honor explicit needs_fallback from the router.
    if tool_decision.route == "chat" and tool_decision.confidence == "high":
        return bool(tool_decision.needs_fallback)
    return bool(tool_decision.needs_fallback or answer_needs_local_fallback(answer, task))


def _visual_source_count(documents: list[dict[str, Any]]) -> int:
    return sum(
        1
        for document in documents
        if document.get("figure_id") or document.get("image_path") or document.get("artifact_type") == "figure"
    )


async def _stream_local_rag_fallback(
    *,
    service: AgentService,
    run_store: AgentRunStore,
    rag: RagService,
    settings: Settings,
    request: AgentRunRequest,
    run_id: str,
    conversation_id: str,
    user_message_id: str,
    task: str,
    conversation_context: str,
    assistant_chunks: list[str],
    focus_document_ids: list[str],
    focus_topic: str | None,
    fallback_state: dict[str, Any],
    debug_trace: DebugTraceRecorder | None = None,
) -> AsyncIterator[str]:
    if request.allowed_tools and "search_local_docs" not in request.allowed_tools:
        return
    fallback_state["attempted"] = True
    fallback_state["grounded"] = False

    fallback_decision = ToolDecision(
        route="file_qa",
        selected_tools=["search_local_docs"],
        reason="uncertain_answer_fallback",
        confidence="medium",
        max_tool_rounds=1,
        needs_fallback=False,
    ).to_dict()
    yield sse_event(
        "tool.fallback.started",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "tool_name": "search_local_docs",
            "reason": "uncertain_answer",
        },
    )
    separator = "\n\nTôi kiểm tra thêm trong tài liệu local một lượt.\n\n"
    assistant_chunks.append(separator)
    yield sse_event("message.delta", {"delta": separator})
    yield sse_event(
        "tool.started",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "tool_name": "search_local_docs",
            "input": {
                "query": task,
                "collection_id": request.collection_id,
                "retrieval_mode": request.retrieval_mode,
                "focus_document_ids": focus_document_ids,
                "fallback": True,
            },
        },
    )
    yield sse_event(
        "retrieval.started",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "query": task,
            "original_task": task,
            "focus_document_ids": focus_document_ids,
            "tool_name": "search_local_docs",
            "cache_candidate": False,
            "fallback": True,
        },
    )
    retrieval_started_at = time.perf_counter()
    retrieval = await _retrieve_for_agent(
        rag=rag,
        settings=settings,
        query=task,
        collection_id=request.collection_id,
        retrieval_mode=request.retrieval_mode,
        focus_document_ids=focus_document_ids,
        answer_intent="direct_answer",
        answer_depth="normal",
        must_cover_all_documents=len(dict.fromkeys(focus_document_ids)) >= 2,
    )
    yield sse_event(
        "timing",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "stage": "retrieval.fallback",
            "ms": _elapsed_ms(retrieval_started_at),
            "mode": retrieval.get("mode"),
        },
    )
    retrieved_docs = _enrich_visual_sources(retrieval["documents"])
    retrieved_docs = _scope_documents_to_focus(retrieved_docs, focus_document_ids)
    retrieved_docs = rag.enrich_source_identities(retrieved_docs)
    retrieval["documents"] = retrieved_docs
    validation = validate_retrieval_evidence(
        documents=retrieved_docs,
        required_entities=[focus_topic] if focus_topic else [],
        current_topic=focus_topic,
        is_followup=bool(focus_document_ids),
        focus_document_ids=focus_document_ids,
    )
    fallback_state["grounded"] = bool(validation.valid)
    fallback_state["sources"] = retrieved_docs
    if debug_trace is not None:
        debug_trace.record_retrieval(
            focus_document_ids=list(focus_document_ids),
            retrieved_document_ids=_document_ids_from_retrieval(
                {"documents": retrieved_docs}
            ),
            validation=validation.to_dict(),
            diagnostics={"mode": retrieval.get("mode"), "phase": "local_fallback"},
        )
    tool_call = run_store.record_tool_call(
        run_id=run_id,
        tool_name="search_local_docs",
        input_payload={
            "query": task,
            "original_task": task,
            "focus_document_ids": focus_document_ids,
            "top_k": 8,
            "collection_id": request.collection_id,
            "retrieval_mode": request.retrieval_mode,
            "fallback": True,
        },
        output_payload={
            **retrieval,
            "fallback": True,
            "evidence_validation": validation.to_dict(),
        },
    )
    yield sse_event(
        "retrieval.completed",
        {
            "run_id": run_id,
            "tool_call_id": tool_call.id,
            "conversation_id": conversation_id,
            "query": task,
            "original_task": task,
            "focus_document_ids": focus_document_ids,
            "documents": retrieved_docs,
            "retrieval_mode": retrieval["mode"],
            "context_stats": retrieval["context_stats"],
            "fallback": True,
            "evidence_validation": validation.to_dict(),
        },
    )
    yield sse_event(
        "tool.completed",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "tool_name": "search_local_docs",
            "tool_call_id": tool_call.id,
            "status": "completed",
            "result_count": len(retrieved_docs),
            "fallback": True,
        },
    )
    async for event in _stream_graph_and_answer(
        service=service,
        run_store=run_store,
        request=request,
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        task=task,
        resolved_task=task,
        conversation_context=conversation_context,
        answer_intent="direct_answer",
        answer_depth="normal",
        retrieved_docs=retrieved_docs,
        tool_decision=fallback_decision,
        assistant_chunks=assistant_chunks,
        focus_document_ids=focus_document_ids,
        validate_quantitative_claims=True,
        require_all_focus_documents=len(dict.fromkeys(focus_document_ids)) >= 2,
        document_identity_resolver=lambda answer: rag.resolve_document_mentions_for_query(
            query=answer,
            collection_id=request.collection_id,
            limit=32,
        ),
        debug_trace=debug_trace,
        generation_phase="local_fallback",
    ):
        yield event


async def _stream_graph_and_answer(
    *,
    service: AgentService,
    run_store: AgentRunStore,
    request: AgentRunRequest,
    run_id: str,
    conversation_id: str,
    user_message_id: str,
    task: str,
    resolved_task: str,
    conversation_context: str,
    answer_intent: str,
    answer_depth: str,
    retrieved_docs: list[dict],
    tool_decision: dict[str, Any],
    assistant_chunks: list[str],
    focus_document_ids: list[str] | None = None,
    validate_quantitative_claims: bool = False,
    require_all_focus_documents: bool = False,
    document_identity_resolver: Callable[[str], list[str]] | None = None,
    debug_trace: DebugTraceRecorder | None = None,
    generation_phase: str = "primary",
    paper_evidence_context: str = "",
    paper_evidence_coverage: list[dict[str, Any]] | None = None,
    paper_section_streaming_enabled: bool = False,
) -> AsyncIterator[str]:
    unique_focus_ids = list(dict.fromkeys(focus_document_ids or []))
    direct_table = (
        None
        if len(unique_focus_ids) >= 2
        else _direct_canonical_table_answer(
            task,
            retrieved_docs,
            expected_document_ids=focus_document_ids,
        )
    )
    if direct_table is not None:
        direct_started_at = time.perf_counter()
        if debug_trace is not None:
            debug_trace.record_direct_execution(
                phase=generation_phase,
                kind="direct_canonical_table",
            )
        answer, table_source = direct_table
        direct_validation = validate_answer_claims(
            answer=answer,
            documents=retrieved_docs,
            focus_document_ids=focus_document_ids,
            require_all_focus_documents=require_all_focus_documents,
            answer_document_ids=_resolve_answer_document_ids(
                document_identity_resolver,
                answer,
            ),
        )
        # Extraction is evidence, not truth by declaration.  If the canonical
        # renderer and validator disagree, fall through to the normal grounded
        # answer path instead of bypassing the safety guard.
        if direct_validation.valid:
            plan = [
                "Select the explicitly requested canonical table in the active paper.",
                "Return its extracted Markdown unchanged with provenance.",
            ]
            run_store.update_plan(run_id, plan)
            yield sse_event(
                "planner.completed",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "route": "file_qa",
                    "mode": "direct_canonical_table",
                    "plan": plan,
                    "selected_tools": ["search_local_docs"],
                },
            )
            diagnostics = _claim_validation_attempt_diagnostics(direct_validation)
            run_store.record_tool_call(
                run_id=run_id,
                tool_name="validate_answer_claims",
                input_payload={
                    "focus_document_ids": focus_document_ids or [],
                    "document_ids": table_source.get("document_ids")
                    or [table_source.get("document_id")],
                    "table_id": table_source.get("table_id"),
                    "table_ids": table_source.get("table_ids")
                    or [table_source.get("table_id")],
                    "direct_canonical_table": True,
                },
                output_payload={
                    **direct_validation.to_dict(),
                    "attempts": 0,
                    "fallback_used": False,
                    "direct_canonical_table": True,
                    "attempt_validations": [diagnostics],
                },
            )
            yield sse_event(
                "answer.evidence.validated",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    **direct_validation.to_dict(),
                    "attempts": 0,
                    "fallback_used": False,
                    "direct_canonical_table": True,
                },
            )
            first_delta_sent = False
            for offset in range(0, len(answer), 220):
                delta = answer[offset : offset + 220]
                if not first_delta_sent:
                    first_delta_sent = True
                    yield sse_event(
                        "timing",
                        {
                            "run_id": run_id,
                            "conversation_id": conversation_id,
                            "stage": "first_validated_token",
                            "ms": _elapsed_ms(direct_started_at),
                            "direct_canonical_table": True,
                            "streaming_mode": "validated_reveal",
                        },
                    )
                assistant_chunks.append(delta)
                yield sse_event("message.delta", {"delta": delta})
            yield sse_event(
                "message.finished",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "finish_reason": "direct_canonical_table",
                    "eval_count": 0,
                    "metrics": {},
                    "truncated": False,
                },
            )
            yield sse_event(
                "timing",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "stage": "generation_total",
                    "ms": _elapsed_ms(direct_started_at),
                    "validation_attempts": 0,
                    "direct_canonical_table": True,
                },
            )
            return

    graph_result = None
    route = None
    graph_mode = None
    selected_tools: list[str] = []
    planner_completed = False
    graph_started_at = time.perf_counter()
    async for graph_event in service.stream_graph_events(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        task=task,
        mode=request.mode,
        model=request.model,
        temperature=request.temperature,
        resolved_task=resolved_task,
        conversation_context=conversation_context,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        answer_style="natural_technical",
        retrieved_docs=retrieved_docs,
        tool_decision=tool_decision,
        paper_evidence_context=paper_evidence_context,
        paper_evidence_coverage=paper_evidence_coverage,
        paper_section_streaming=bool(
            paper_section_streaming_enabled
            and paper_evidence_coverage
            and len(paper_evidence_coverage) >= 2
        ),
    ):
        event_payload = {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "event": graph_event.event,
            **graph_event.payload,
        }
        yield sse_event("agent.event", event_payload)
        if graph_event.event == "router.completed":
            route = graph_event.payload.get("route")
            graph_mode = graph_event.payload.get("mode")
            selected_tools = list(graph_event.payload.get("selected_tools") or [])
        if graph_event.event == "planner.started":
            yield sse_event(
                "planner.started",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                },
            )
        if graph_event.event == "planner.completed":
            planner_completed = True
            plan = graph_event.payload.get("plan") or []
            run_store.update_plan(run_id, plan)
            yield sse_event(
                "planner.completed",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "route": route,
                    "mode": graph_mode,
                    "plan": plan,
                    "selected_tools": selected_tools,
                },
            )
        if graph_event.event == "graph.completed":
            graph_result = graph_event.result
    yield sse_event(
        "timing",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "stage": "graph",
            "ms": _elapsed_ms(graph_started_at),
        },
    )

    if graph_result is None:
        raise RuntimeError("Agent graph did not complete")
    if not planner_completed:
        run_store.update_plan(run_id, graph_result.plan)
        yield sse_event(
            "planner.completed",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "route": graph_result.route,
                "mode": graph_result.mode,
                "plan": graph_result.plan,
                "selected_tools": graph_result.selected_tools,
            },
        )

    generation_started_at = time.perf_counter()
    first_delta_sent = False
    generation_trace_index = (
        debug_trace.start_generation(
            phase=generation_phase,
            route=str(graph_result.route),
            model=request.model or service.default_model,
            temperature=request.temperature,
            prompt=graph_result.final_prompt,
        )
        if debug_trace is not None
        else None
    )
    stream = service.stream_final_answer(
        prompt=graph_result.final_prompt,
        model=request.model,
        temperature=request.temperature,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
    )
    stream = _trace_answer_stream(
        stream,
        recorder=debug_trace,
        generation_index=generation_trace_index,
        attempt_kind="initial",
    )

    if (
        paper_section_streaming_enabled
        and paper_evidence_coverage
        and len(unique_focus_ids) >= 2
    ):
        emitted_parts: list[str] = []
        final_chunk: AnswerStreamChunk | None = None
        section_diagnostics: list[dict[str, Any]] = []
        first_delta_sent = False
        async for event_type, payload in _stream_validated_paper_sections(
            stream,
            documents=retrieved_docs,
            ordered_document_ids=unique_focus_ids,
        ):
            if event_type == "finished":
                final_chunk = payload["chunk"]
                section_diagnostics = list(payload.get("validations") or [])
                break
            if event_type == "paper_validated":
                yield sse_event(
                    "answer.paper.validated",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        **payload,
                    },
                )
                continue
            delta = str(payload.get("delta") or "")
            if not delta:
                continue
            if not first_delta_sent:
                first_delta_sent = True
                yield sse_event(
                    "timing",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "stage": "first_validated_token",
                        "ms": _elapsed_ms(generation_started_at),
                        "streaming_mode": "validated_paper_sections",
                    },
                )
            emitted_parts.append(delta)
            assistant_chunks.append(delta)
            yield sse_event("message.delta", {"delta": delta})

        answer = "".join(emitted_parts)
        final_validation = validate_answer_claims(
            answer=answer,
            documents=retrieved_docs,
            focus_document_ids=unique_focus_ids,
            require_all_focus_documents=True,
            # Every emitted section is created by the parser for exactly one
            # expected document, including deterministic insufficiency blocks.
            answer_document_ids=unique_focus_ids,
        )
        run_store.record_tool_call(
            run_id=run_id,
            tool_name="validate_answer_claims",
            input_payload={
                "focus_document_ids": unique_focus_ids,
                "streaming_mode": "validated_paper_sections",
            },
            output_payload={
                **final_validation.to_dict(),
                "attempts": 1,
                "fallback_used": any(item.get("fallback_used") for item in section_diagnostics),
                "streaming_mode": "validated_paper_sections",
                "section_validations": section_diagnostics,
            },
        )
        yield sse_event(
            "answer.evidence.validated",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                **final_validation.to_dict(),
                "attempts": 1,
                "streaming_mode": "validated_paper_sections",
                "section_validations": section_diagnostics,
            },
        )
        final_chunk = final_chunk or AnswerStreamChunk(
            content="", done=True, finish_reason="validated_paper_sections_closed"
        )
        metrics = _ollama_metrics(final_chunk.metadata)
        yield sse_event(
            "message.finished",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "finish_reason": final_chunk.finish_reason,
                "eval_count": final_chunk.metadata.get("eval_count") if final_chunk.metadata else None,
                "metrics": metrics,
                "truncated": final_chunk.finish_reason == "length",
            },
        )
        yield sse_event(
            "timing",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "stage": "generation_total",
                "ms": _elapsed_ms(generation_started_at),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "validation_attempts": 1,
                "streaming_mode": "validated_paper_sections",
            },
        )
        return

    # Never expose a quantitative RAG draft before checking it against the
    # exact curated evidence. General chat keeps token streaming; grounded
    # paper/file QA is buffered because sentence-level gating leaks Markdown
    # table rows and split Acc/F1/CCC claims.
    if (
        validate_quantitative_claims
        and retrieved_docs
        and not _task_requests_quantitative_evidence(task)
        and not require_all_focus_documents
    ):
        emitted_parts: list[str] = []
        block_validations: list[dict[str, Any]] = []
        suppressed_blocks = 0
        final_chunk: AnswerStreamChunk | None = None
        async for event_type, payload in _stream_validated_grounded_blocks(
            stream,
            documents=retrieved_docs,
            focus_document_ids=focus_document_ids,
        ):
            if event_type == "finished":
                final_chunk = payload["chunk"]
                block_validations = list(payload.get("validations") or [])
                suppressed_blocks = int(payload.get("suppressed_blocks") or 0)
                break
            delta = str(payload.get("delta") or "")
            if not delta:
                continue
            if not first_delta_sent:
                first_delta_sent = True
                yield sse_event(
                    "timing",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "stage": "first_validated_token",
                        "ms": _elapsed_ms(generation_started_at),
                        "streaming_mode": "validated_blocks",
                    },
                )
            for offset in range(0, len(delta), 220):
                piece = delta[offset : offset + 220]
                emitted_parts.append(piece)
                assistant_chunks.append(piece)
                yield sse_event("message.delta", {"delta": piece})

        answer = "".join(emitted_parts)
        if not answer.strip():
            answer = _unsupported_quantitative_fallback(task)
            assistant_chunks.append(answer)
            yield sse_event("message.delta", {"delta": answer})
        final_validation = validate_answer_claims(
            answer=answer,
            documents=retrieved_docs,
            focus_document_ids=focus_document_ids,
            answer_document_ids=_resolve_answer_document_ids(
                document_identity_resolver,
                answer,
            ),
        )
        if debug_trace is not None:
            debug_trace.annotate_last_attempt(
                generation_trace_index,
                validation=final_validation.to_dict(),
            )
            debug_trace.select_output(
                generation_trace_index,
                "fallback" if not emitted_parts else "initial_sanitized" if suppressed_blocks else "initial",
            )
        run_store.record_tool_call(
            run_id=run_id,
            tool_name="validate_answer_claims",
            input_payload={
                "focus_document_ids": focus_document_ids or [],
                "document_ids": _document_ids_from_retrieval({"documents": retrieved_docs}),
                "streaming_mode": "validated_blocks",
            },
            output_payload={
                **final_validation.to_dict(),
                "attempts": 1,
                "fallback_used": not bool(emitted_parts),
                "streaming_mode": "validated_blocks",
                "suppressed_blocks": suppressed_blocks,
                "attempt_validations": block_validations,
            },
        )
        yield sse_event(
            "answer.evidence.validated",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                **final_validation.to_dict(),
                "attempts": 1,
                "fallback_used": not bool(emitted_parts),
                "streaming_mode": "validated_blocks",
                "suppressed_blocks": suppressed_blocks,
            },
        )
        final_chunk = final_chunk or AnswerStreamChunk(
            content="", done=True, finish_reason="validated_stream_closed"
        )
        metrics = _ollama_metrics(final_chunk.metadata)
        yield sse_event(
            "message.finished",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "finish_reason": final_chunk.finish_reason,
                "eval_count": final_chunk.metadata.get("eval_count") if final_chunk.metadata else None,
                "metrics": metrics,
                "truncated": final_chunk.finish_reason == "length",
            },
        )
        yield sse_event(
            "timing",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "stage": "generation_total",
                "ms": _elapsed_ms(generation_started_at),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "validation_attempts": 1,
                "streaming_mode": "validated_blocks",
            },
        )
        return

    if validate_quantitative_claims and retrieved_docs:
        draft, final_chunk = await _collect_answer_stream(stream)
        claim_validation = validate_answer_claims(
            answer=draft,
            documents=retrieved_docs,
            focus_document_ids=focus_document_ids,
            require_all_focus_documents=require_all_focus_documents,
            answer_document_ids=_resolve_answer_document_ids(
                document_identity_resolver,
                draft,
            ),
        )
        if debug_trace is not None:
            debug_trace.annotate_last_attempt(
                generation_trace_index,
                validation=claim_validation.to_dict(),
            )
        attempt_validations = [_claim_validation_attempt_diagnostics(claim_validation)]
        attempts = 1
        answer = draft
        fallback_used = False
        selected_output = "initial"
        if not claim_validation.valid:
            sanitized, sanitized_validation, sanitized_accepted = (
                _sanitize_answer_before_retry(
                    answer=draft,
                    validation=claim_validation,
                    documents=retrieved_docs,
                    focus_document_ids=focus_document_ids,
                    task=task,
                    require_all_focus_documents=require_all_focus_documents,
                    document_identity_resolver=document_identity_resolver,
                )
            )
            if sanitized_validation is not None:
                attempt_validations.append(
                    _claim_validation_attempt_diagnostics(sanitized_validation)
                )
            if sanitized_accepted:
                answer = sanitized
                claim_validation = sanitized_validation
                selected_output = "initial_sanitized"
            else:
                attempts = 2
                correction_prompt = _claim_correction_prompt(
                    graph_result.final_prompt,
                    claim_validation.to_dict(),
                )
                corrected_stream = service.stream_final_answer(
                    prompt=correction_prompt,
                    model=request.model,
                    temperature=min(request.temperature, 0.2),
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                )
                corrected_stream = _trace_answer_stream(
                    corrected_stream,
                    recorder=debug_trace,
                    generation_index=generation_trace_index,
                    attempt_kind="correction",
                )
                corrected, corrected_final = await _collect_answer_stream(corrected_stream)
                corrected_validation = validate_answer_claims(
                    answer=corrected,
                    documents=retrieved_docs,
                    focus_document_ids=focus_document_ids,
                    require_all_focus_documents=require_all_focus_documents,
                    answer_document_ids=_resolve_answer_document_ids(
                        document_identity_resolver,
                        corrected,
                    ),
                )
                if debug_trace is not None:
                    debug_trace.annotate_last_attempt(
                        generation_trace_index,
                        validation=corrected_validation.to_dict(),
                    )
                attempt_validations.append(
                    _claim_validation_attempt_diagnostics(corrected_validation)
                )
                final_chunk = corrected_final or final_chunk
                if corrected_validation.valid:
                    answer = corrected
                    claim_validation = corrected_validation
                    selected_output = "corrected"
                else:
                    corrected_sanitized, corrected_sanitized_validation, corrected_accepted = (
                        _sanitize_answer_before_retry(
                            answer=corrected,
                            validation=corrected_validation,
                            documents=retrieved_docs,
                            focus_document_ids=focus_document_ids,
                            task=task,
                            require_all_focus_documents=require_all_focus_documents,
                            document_identity_resolver=document_identity_resolver,
                        )
                    )
                    if corrected_sanitized_validation is not None:
                        attempt_validations.append(
                            _claim_validation_attempt_diagnostics(
                                corrected_sanitized_validation
                            )
                        )
                    if corrected_accepted:
                        answer = corrected_sanitized
                        claim_validation = corrected_sanitized_validation
                        selected_output = "corrected_sanitized"
                    else:
                        answer = _unsupported_quantitative_fallback(task)
                        fallback_used = True
                        selected_output = "fallback"
                        claim_validation = validate_answer_claims(
                            answer=answer,
                            documents=retrieved_docs,
                            focus_document_ids=focus_document_ids,
                            require_all_focus_documents=require_all_focus_documents,
                            answer_document_ids=_resolve_answer_document_ids(
                                document_identity_resolver,
                                answer,
                            ),
                        )
        if debug_trace is not None:
            debug_trace.select_output(generation_trace_index, selected_output)
        run_store.record_tool_call(
            run_id=run_id,
            tool_name="validate_answer_claims",
            input_payload={
                "focus_document_ids": focus_document_ids or [],
                "document_ids": _document_ids_from_retrieval({"documents": retrieved_docs}),
            },
            output_payload={
                **claim_validation.to_dict(),
                "attempts": attempts,
                "fallback_used": fallback_used,
                "attempt_validations": attempt_validations,
            },
        )
        yield sse_event(
            "answer.evidence.validated",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                **claim_validation.to_dict(),
                "attempts": attempts,
                "fallback_used": fallback_used,
            },
        )
        if not answer.strip():
            answer = _unsupported_quantitative_fallback(task)
        for offset in range(0, len(answer), 220):
            delta = answer[offset : offset + 220]
            if not first_delta_sent:
                first_delta_sent = True
                yield sse_event(
                    "timing",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "stage": "first_validated_token",
                        "ms": _elapsed_ms(generation_started_at),
                        "streaming_mode": "buffered_validation",
                    },
                )
            assistant_chunks.append(delta)
            yield sse_event("message.delta", {"delta": delta})
        final_chunk = final_chunk or AnswerStreamChunk(
            content="", done=True, finish_reason="validated_stream_closed"
        )
        metrics = _ollama_metrics(final_chunk.metadata)
        yield sse_event(
            "message.finished",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "finish_reason": final_chunk.finish_reason,
                "eval_count": final_chunk.metadata.get("eval_count") if final_chunk.metadata else None,
                "metrics": metrics,
                "truncated": final_chunk.finish_reason == "length",
            },
        )
        yield sse_event(
            "timing",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "stage": "generation_total",
                "ms": _elapsed_ms(generation_started_at),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "validation_attempts": attempts,
            },
        )
        return

    async for event_type, payload in _buffer_answer_stream(stream, min_chars=12, max_wait_ms=20):
        if event_type == "finished":
            chunk = payload["chunk"]
            metrics = _ollama_metrics(chunk.metadata)
            yield sse_event(
                "message.finished",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "finish_reason": chunk.finish_reason,
                    "eval_count": chunk.metadata.get("eval_count") if chunk.metadata else None,
                    "metrics": metrics,
                    "truncated": chunk.finish_reason == "length",
                },
            )
            yield sse_event(
                "timing",
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "stage": "generation_total",
                    "ms": _elapsed_ms(generation_started_at),
                    "tokens_per_second": metrics.get("tokens_per_second"),
                },
            )
            continue

        delta = payload["delta"]
        if delta:
            if not first_delta_sent:
                first_delta_sent = True
                yield sse_event(
                    "timing",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "stage": "first_token",
                        "ms": _elapsed_ms(generation_started_at),
                        "streaming_mode": "token_stream",
                    },
                )
            assistant_chunks.append(delta)
            yield sse_event("message.delta", {"delta": delta})

    if debug_trace is not None:
        debug_trace.select_output(generation_trace_index, "initial")


def _embedding_provider(settings: Settings) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


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
        "lightrag_enabled": settings.lightrag_enabled,
        "retrieval_engine": settings.retrieval_engine,
        "agentic_retrieval_max_hops": settings.agentic_retrieval_max_hops,
        "agentic_retrieval_max_subqueries": settings.agentic_retrieval_max_subqueries,
    }


def _retrieval_store(settings: Settings) -> LanceDBRetrievalStore:
    return LanceDBRetrievalStore(settings.lancedb_path)


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
                _retrieve_for_agent(
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
    composed = compose_retrieval_context(
        merged,
        query=query,
        max_sources=max(budget["max_sources"], 8),
        max_chars=max(budget["max_chars"], 6_500),
        max_chars_per_source=max(budget["max_chars_per_source"], 1_050),
        max_chunks_per_document=max(budget["max_chunks_per_document"], 3),
        min_figures=2 if include_visual_boost else 0,
        min_tables=1 if prefer_tables else 0,
        required_document_ids=required_document_ids,
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
        retrieval_store=_retrieval_store(settings),
        embeddings=_embedding_provider(settings),
        rerank=settings.rerank_enabled,
        rerank_mode=settings.rerank_mode,
        cross_encoder_model_path=settings.rerank_cross_encoder_path,
        rerank_max_candidates=settings.rerank_max_candidates,
    )


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
                retrieval_store=_retrieval_store(settings),
                embeddings=_embedding_provider(settings),
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
        retrieval_store=_retrieval_store(settings),
        embeddings=_embedding_provider(settings),
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
    service = PaperEvidenceService(
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
                result = await _retrieve_legacy_for_agent(
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
        required_document_ids=ordered_ids,
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
) -> dict:
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
                _retrieve_for_agent(
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
            required_document_ids=[
                branch.focus_document_ids[0]
                for branch in branches
                if branch.focus_document_ids
            ],
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
        legacy = await _retrieve_legacy_for_agent(
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
            bridge = LightRAGBridge(settings)
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
                legacy_fallback = await _retrieve_legacy_for_agent(
                    rag=rag,
                    settings=settings,
                    query=query,
                    collection_id=collection_id,
                    retrieval_mode=retrieval_mode,
                    focus_document_ids=focus_document_ids,
                    answer_intent=answer_intent,
                    answer_depth=answer_depth,
                    include_visual_boost=include_visual_boost,
                )
                legacy_fallback.setdefault("diagnostics", {})[
                    "lightrag_fallback_reason"
                ] = "empty_after_scope_mapping"
                legacy_fallback["diagnostics"]["selected_engine"] = "legacy_fallback"
                legacy_fallback["diagnostics"][
                    "policy_reason"
                ] = "lightrag_empty_after_scope_mapping"
                return legacy_fallback
            legacy = await _retrieve_legacy_for_agent(
                rag=rag,
                settings=settings,
                query=query,
                collection_id=collection_id,
                retrieval_mode=retrieval_mode,
                focus_document_ids=focus_document_ids,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                include_visual_boost=include_visual_boost,
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

    legacy = await _retrieve_legacy_for_agent(
        rag=rag,
        settings=settings,
        query=query,
        collection_id=collection_id,
        retrieval_mode=retrieval_mode,
        focus_document_ids=focus_document_ids,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        include_visual_boost=include_visual_boost,
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
) -> dict:
    mode = retrieval_mode.lower()
    if mode not in {"auto", "hybrid", "fts"}:
        mode = "auto"

    diagnostics: dict = {}
    embeddings = _embedding_provider(settings)
    single_doc_focus = bool(focus_document_ids) and len(focus_document_ids) == 1
    hybrid_top_k = 10 if single_doc_focus else 6
    if mode in {"auto", "hybrid"}:
        try:
            hybrid = await rag.search_hybrid(
                query=query,
                top_k=hybrid_top_k,
                collection_id=collection_id,
                document_ids=focus_document_ids or None,
                retrieval_store=_retrieval_store(settings),
                embeddings=embeddings,
                rerank=settings.rerank_enabled,
                rerank_mode=settings.rerank_mode,
                cross_encoder_model_path=settings.rerank_cross_encoder_path,
                rerank_max_candidates=settings.rerank_max_candidates,
                visual_boost=include_visual_boost,
            )
            context_budget = _context_budget(answer_intent, answer_depth)
            if single_doc_focus:
                context_budget = {
                    **context_budget,
                    "max_sources": max(context_budget["max_sources"], 6),
                    "max_chunks_per_document": max(context_budget["max_chunks_per_document"], 4),
                    "max_chars": max(context_budget["max_chars"], 5_200),
                }
            expanded_results = rag.expand_with_neighbor_chunks(
                hybrid["results"],
                window=context_budget["neighbor_window"],
                max_neighbor_chars=context_budget["max_neighbor_chars"],
                query=query,
            )
            table_reserve = 1 if prefer_tables else 0
            if prefer_tables and focus_document_ids:
                expanded_results, inventory_count = _merge_focused_table_inventory(
                    rag=rag,
                    sources=expanded_results,
                    focus_document_ids=focus_document_ids,
                    query=query,
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
            )
            sources = list(composed.sources)
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
                },
            }
        except (LanceDBUnavailable, EmbeddingError) as exc:
            if mode == "hybrid":
                raise
            diagnostics["hybrid_fallback_reason"] = str(exc)

    chunks = [chunk.__dict__ for chunk in rag.search(query, top_k=6, document_ids=focus_document_ids or None)]
    context_budget = _context_budget(answer_intent, answer_depth)
    expanded_chunks = rag.expand_with_neighbor_chunks(
        chunks,
        window=context_budget["neighbor_window"],
        max_neighbor_chars=context_budget["max_neighbor_chars"],
        query=query,
    )
    table_reserve = 1 if prefer_tables else 0
    if prefer_tables and focus_document_ids:
        expanded_chunks, inventory_count = _merge_focused_table_inventory(
            rag=rag,
            sources=expanded_chunks,
            focus_document_ids=focus_document_ids,
            query=query,
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
    )
    return {
        "mode": "fts",
        "documents": composed.sources,
        "context_text": composed.context_text,
        "context_stats": composed.stats,
        "diagnostics": {
            **diagnostics,
            "context_expanded_results": sum(1 for item in expanded_chunks if item.get("expanded_content")),
        },
    }


async def _buffer_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    min_chars: int = 64,
    max_wait_ms: int = 55,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    buffer: list[str] = []
    buffered_chars = 0
    last_flush = time.perf_counter()

    async for chunk in chunks:
        if chunk.done:
            if buffer:
                yield "delta", {"delta": "".join(buffer)}
            yield "finished", {"chunk": chunk}
            return

        if not chunk.content:
            continue

        buffer.append(chunk.content)
        buffered_chars += len(chunk.content)
        elapsed_ms = (time.perf_counter() - last_flush) * 1000
        if buffered_chars >= min_chars or elapsed_ms >= max_wait_ms:
            yield "delta", {"delta": "".join(buffer)}
            buffer.clear()
            buffered_chars = 0
            last_flush = time.perf_counter()


def _validated_grounded_block(
    block: str,
    *,
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> tuple[str, list[Any], bool]:
    """Validate one complete Markdown paragraph before it becomes visible."""

    validation = validate_answer_claims(
        answer=block,
        documents=documents,
        focus_document_ids=focus_document_ids,
    )
    if validation.valid:
        return block, [validation], False
    sanitized = _remove_unsupported_claim_lines(block, validation.to_dict())
    if not sanitized.strip():
        return "", [validation], True
    sanitized_validation = validate_answer_claims(
        answer=sanitized,
        documents=documents,
        focus_document_ids=focus_document_ids,
    )
    if not sanitized_validation.valid:
        return "", [validation, sanitized_validation], True
    if block.endswith("\n\n"):
        sanitized = f"{sanitized.rstrip()}\n\n"
    return sanitized, [validation, sanitized_validation], True


async def _stream_validated_paper_sections(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    documents: list[dict[str, Any]],
    ordered_document_ids: list[str],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Release each canonical paper section only after local validation.

    The delimiter is an internal transport contract.  It is never forwarded to
    the desktop. Malformed/missing sections fail closed to a deterministic,
    paper-named insufficiency block instead of leaking an unchecked draft.
    """

    ordered_ids = list(dict.fromkeys(ordered_document_ids))
    pending = ""
    next_index = 0
    validations: list[dict[str, Any]] = []
    final_chunk: AnswerStreamChunk | None = None
    synthesis_emitted = False

    async def release_ready() -> AsyncIterator[tuple[str, dict[str, Any]]]:
        nonlocal pending, next_index, synthesis_emitted
        while next_index < len(ordered_ids):
            document_id = ordered_ids[next_index]
            opener = re.compile(
                rf"<paper\s+document_id\s*=\s*['\"]{re.escape(document_id)}['\"]\s*>",
                flags=re.IGNORECASE,
            )
            match = opener.search(pending)
            if match is None:
                return
            close_at = pending.lower().find("</paper>", match.end())
            if close_at < 0:
                return
            raw_section = pending[match.end() : close_at].strip()
            pending = pending[close_at + len("</paper>") :]
            scoped_documents = _scope_documents_to_focus(documents, [document_id])
            safe_section, diagnostics = _validated_paper_section(
                raw_section,
                document_id=document_id,
                documents=scoped_documents,
            )
            validations.append(diagnostics)
            yield "paper_validated", diagnostics
            yield "delta", {"delta": f"{safe_section.rstrip()}\n\n"}
            next_index += 1

        if next_index == len(ordered_ids) and not synthesis_emitted:
            opener = re.search(r"<synthesis\s*>", pending, flags=re.IGNORECASE)
            if opener is None:
                return
            close_at = pending.lower().find("</synthesis>", opener.end())
            if close_at < 0:
                return
            raw_synthesis = pending[opener.end() : close_at].strip()
            pending = pending[close_at + len("</synthesis>") :]
            if raw_synthesis:
                validation = validate_answer_claims(
                    answer=raw_synthesis,
                    documents=documents,
                    focus_document_ids=ordered_ids,
                )
                safe = raw_synthesis
                fallback_used = False
                if not validation.valid:
                    safe = _remove_unsupported_claim_lines(raw_synthesis, validation.to_dict()).strip()
                    fallback_used = safe != raw_synthesis
                if safe:
                    yield "delta", {"delta": f"{safe.rstrip()}\n"}
                validations.append(
                    {
                        "document_id": None,
                        "section": "synthesis",
                        "valid": validation.valid,
                        "reason": validation.reason,
                        "fallback_used": fallback_used,
                    }
                )
            synthesis_emitted = True

    async for chunk in chunks:
        if chunk.content:
            pending += chunk.content
        async for event in release_ready():
            yield event
        if chunk.done:
            final_chunk = chunk
            break

    # Every missing or malformed section becomes an explicit paper-specific
    # insufficiency statement. This preserves must-cover-all atomically without
    # discarding earlier sections that already passed validation.
    while next_index < len(ordered_ids):
        document_id = ordered_ids[next_index]
        fallback = _paper_section_insufficiency(document_id, documents)
        diagnostics = {
            "document_id": document_id,
            "section": "paper",
            "valid": False,
            "reason": "missing_or_malformed_paper_section",
            "fallback_used": True,
        }
        validations.append(diagnostics)
        yield "paper_validated", diagnostics
        yield "delta", {"delta": f"{fallback}\n\n"}
        next_index += 1

    yield "finished", {
        "chunk": final_chunk
        or AnswerStreamChunk(content="", done=True, finish_reason="stream_closed"),
        "validations": validations,
    }


def _validated_paper_section(
    section: str,
    *,
    document_id: str,
    documents: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not section.strip() or not documents:
        return _paper_section_insufficiency(document_id, documents), {
            "document_id": document_id,
            "section": "paper",
            "valid": False,
            "reason": "empty_or_missing_evidence",
            "fallback_used": True,
        }
    validation = validate_answer_claims(
        answer=section,
        documents=documents,
        focus_document_ids=[document_id],
        answer_document_ids=[document_id],
    )
    safe = section.strip()
    fallback_used = False
    if not validation.valid:
        safe = _remove_unsupported_claim_lines(safe, validation.to_dict()).strip()
        sanitized = validate_answer_claims(
            answer=safe,
            documents=documents,
            focus_document_ids=[document_id],
            answer_document_ids=[document_id],
        ) if safe else validation
        if not safe or not sanitized.valid:
            safe = _paper_section_insufficiency(document_id, documents)
        fallback_used = True
    heading = _paper_section_heading(document_id, documents)
    if heading.lower() not in safe.lower():
        safe = f"**{heading}**\n\n{safe}"
    return safe, {
        "document_id": document_id,
        "section": "paper",
        "valid": validation.valid,
        "reason": validation.reason,
        "fallback_used": fallback_used,
    }


def _paper_section_heading(document_id: str, documents: list[dict[str, Any]]) -> str:
    for document in documents:
        if str(document.get("document_id") or "") == document_id:
            return str(document.get("filename") or document.get("title") or document_id)
    return document_id


def _paper_section_insufficiency(document_id: str, documents: list[dict[str, Any]]) -> str:
    heading = _paper_section_heading(document_id, documents)
    return (
        f"**{heading}**\n\n"
        "Mình chưa có đủ canonical evidence hợp lệ cho phần này, nên không suy đoán thêm."
    )


async def _stream_validated_grounded_blocks(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Progressively release complete, locally validated Markdown blocks.

    This gives non-quantitative paper QA genuine progressive output without
    exposing an unchecked metric sentence.  Quantitative/table requests still
    use the whole-answer guard because their requested rows must be assessed as
    one unit.
    """

    pending = ""
    validations: list[dict[str, Any]] = []
    suppressed_blocks = 0
    final_chunk: AnswerStreamChunk | None = None

    async for chunk in chunks:
        if chunk.content:
            pending += chunk.content
        while "\n\n" in pending:
            boundary = pending.index("\n\n") + 2
            block, pending = pending[:boundary], pending[boundary:]
            safe, block_results, sanitized = _validated_grounded_block(
                block,
                documents=documents,
                focus_document_ids=focus_document_ids,
            )
            validations.extend(
                _claim_validation_attempt_diagnostics(result)
                for result in block_results
            )
            if sanitized:
                suppressed_blocks += 1
            if safe:
                yield "delta", {"delta": safe}
        if chunk.done:
            final_chunk = chunk
            break

    if pending:
        safe, block_results, sanitized = _validated_grounded_block(
            pending,
            documents=documents,
            focus_document_ids=focus_document_ids,
        )
        validations.extend(
            _claim_validation_attempt_diagnostics(result)
            for result in block_results
        )
        if sanitized:
            suppressed_blocks += 1
        if safe:
            yield "delta", {"delta": safe}
    yield "finished", {
        "chunk": final_chunk
        or AnswerStreamChunk(content="", done=True, finish_reason="stream_closed"),
        "validations": validations[:16],
        "suppressed_blocks": suppressed_blocks,
    }


async def _collect_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
) -> tuple[str, AnswerStreamChunk | None]:
    parts: list[str] = []
    final_chunk: AnswerStreamChunk | None = None
    async for chunk in chunks:
        if chunk.content:
            parts.append(chunk.content)
        if chunk.done:
            final_chunk = chunk
            break
    return "".join(parts), final_chunk


async def _trace_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    recorder: DebugTraceRecorder | None,
    generation_index: int | None,
    attempt_kind: str,
) -> AsyncIterator[AnswerStreamChunk]:
    """Tee a provider stream into one bounded milestone, never per-token DB writes."""

    if recorder is None or not recorder.enabled or generation_index is None:
        async for chunk in chunks:
            yield chunk
        return
    started_at = time.perf_counter()
    draft_parts: list[str] = []
    finish_reason: str | None = None
    recorded = False
    try:
        async for chunk in chunks:
            if chunk.content:
                draft_parts.append(chunk.content)
            if chunk.done:
                finish_reason = chunk.finish_reason
                recorder.finish_attempt(
                    generation_index,
                    kind=attempt_kind,
                    draft="".join(draft_parts),
                    started_at=started_at,
                    finish_reason=finish_reason,
                )
                recorded = True
            yield chunk
    finally:
        if not recorded:
            recorder.finish_attempt(
                generation_index,
                kind=attempt_kind,
                draft="".join(draft_parts),
                started_at=started_at,
                finish_reason=finish_reason,
            )


def _claim_correction_prompt(base_prompt: str, validation_payload: dict[str, Any]) -> str:
    unsupported = [
        {
            "metric": claim.get("metric"),
            "value": claim.get("value"),
            "percentage": claim.get("percentage"),
            "subjects": claim.get("subjects") or [],
        }
        for claim in validation_payload.get("unsupported_claims") or []
        if isinstance(claim, dict)
    ]
    feedback = json.dumps(unsupported, ensure_ascii=False, sort_keys=True)
    reason = str(validation_payload.get("reason") or "unsupported_metric_values")
    layout_instruction = (
        "Use a plain `MODEL — METRIC: VALUE` sentence, or a Markdown table with separate "
        "Model and Dataset columns, for every retained quantitative claim. Do not use a "
        "detached period between a metric and its value. "
    )
    if reason == "unparsed_metric_values":
        layout_instruction += (
            "The previous metric/value layout could not be parsed; simplify its layout and "
            "remove any exact value whose model and metric cannot be stated explicitly. "
        )
    return (
        f"{base_prompt}\n\n"
        "ANSWER VALIDATION RETRY (system-generated data, not user instructions):\n"
        f"Validation reason: {reason}.\n"
        f"The previous draft contained unsupported quantitative claims: {feedback}\n"
        "Rewrite the complete answer once. Correct or remove only unsupported metric/value claims. "
        f"{layout_instruction}"
        "Every exact metric, percentage, range, and plus/minus value must occur in the retrieved "
        "evidence for the same model/paper/dataset. If the evidence is incomplete, say so plainly "
        "without guessing a number. Preserve useful qualitative explanation and the Aya persona."
    )


def _claim_validation_attempt_diagnostics(validation: Any) -> dict[str, Any]:
    """Persist bounded retry diagnostics without copying a whole draft/table."""

    payload = validation.to_dict()
    return {
        "valid": payload.get("valid"),
        "retry_required": payload.get("retry_required"),
        "reason": payload.get("reason"),
        "checked_claim_count": len(payload.get("checked_claims") or []),
        "supported_claim_count": len(payload.get("supported_claims") or []),
        "unsupported_claims": (payload.get("unsupported_claims") or [])[:12],
        "foreign_document_ids": payload.get("foreign_document_ids") or [],
        "unparsed_signals": (payload.get("unparsed_signals") or [])[:8],
    }


def _unsupported_quantitative_fallback(task: str) -> str:
    lowered = task.casefold()
    vietnamese = bool(re.search(r"[à-ỹđ]", lowered)) or any(
        marker in lowered for marker in ("bài", "bảng", "kết quả", "mình", "paper lúc")
    )
    if vietnamese:
        return (
            "Mình chưa có đủ evidence khớp để khẳng định các số liệu định lượng trong câu hỏi này. "
            "Aya sẽ không đoán số; cần retrieve đúng bảng hoặc đoạn kết quả của paper rồi mới kết luận."
        )
    return (
        "The retrieved evidence is not sufficient to support the exact quantitative claims requested. "
        "Aya will not guess values; retrieve the relevant result table or passage before concluding."
    )


def _task_requests_quantitative_evidence(task: str) -> bool:
    normalized = " ".join(str(task or "").casefold().split())
    if re.search(
        r"(?<!\w)(?:acc|accuracy|f1|ccc|uar|war|wer|mae|mse|rmse|"
        r"precision|recall|auc|eer|hl|jaccard|jac|exacc|biacc|"
        r"params?|parameters?)(?!\w)",
        normalized,
    ):
        return True
    return any(
        marker in normalized
        for marker in (
            "benchmark",
            "ablation",
            "bao nhiêu",
            "số liệu",
            "kết quả",
            "bảng",
            "table",
            "score",
            "metric",
            "percentage",
            "percent",
            "phần trăm",
        )
    )


_EXACT_TABLE_REQUEST_RE = re.compile(
    r"(?<!\w)(?:bảng|table)\s*(?:số\s*)?#?\s*(?P<number>\d+)(?!\w)",
    re.IGNORECASE,
)
_TABLE_NUMBER_SEQUENCE_RE = re.compile(
    r"(?<!\w)(?:bảng|tables?)\s*(?:số\s*)?#?\s*"
    r"(?P<sequence>\d+(?:\s*(?:,|;|\+|&|và|va|với|voi|and|with)\s*"
    r"(?:(?:bảng|tables?)\s*(?:số\s*)?#?\s*)?\d+)*)",
    re.IGNORECASE,
)
_TABLE_ANALYSIS_RE = re.compile(
    r"(?<!\w)(?:giải\s*thích|phân\s*tích|đánh\s*giá|nhận\s*xét|ý\s*nghĩa|"
    r"explain|analy[sz]e|evaluate|interpret)(?!\w)",
    re.IGNORECASE,
)
_GENERIC_RESULT_TABLE_REQUEST_RE = re.compile(
    r"(?:bảng|table).{0,64}(?:kết\s*quả|result|benchmark|performance)|"
    r"(?:kết\s*quả|result|benchmark|performance).{0,64}(?:bảng|table)",
    re.IGNORECASE,
)
_MAIN_RESULT_TABLE_CAPTION_RE = re.compile(
    r"(?:main\s+results?|experimental\s+results?|comparison|comparative|"
    r"performance|benchmark)",
    re.IGNORECASE,
)
_NON_MAIN_RESULT_TABLE_CAPTION_RE = re.compile(
    r"(?:ablation|distribution|statistics|dataset\s+(?:summary|distribution)|"
    r"hyperparameters?|training\s+settings?)",
    re.IGNORECASE,
)


def _select_unique_generic_result_table(
    query: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select one semantically best main table, or fail closed on a tie.

    Performance/dataset coverage and prior-method comparison are different
    result intents.  Ranking those caption semantics lets mixed asks such as
    ``which datasets + show the results table`` choose the coverage table while
    preserving ambiguity when multiple tables serve the same purpose.
    """

    normalized = " ".join(str(query or "").casefold().split())
    wants_dataset_coverage = bool(
        re.search(r"(?<!\w)(?:datasets?|bộ\s+dữ\s+liệu|dữ\s+liệu)(?!\w)", normalized)
    )
    wants_comparison = bool(
        re.search(
            r"(?<!\w)(?:so\s*sánh|compare|comparison|prior|baseline|"
            r"models?|methods?|versus|vs\.)(?!\w)",
            normalized,
        )
    )

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for candidate in candidates:
        caption = str(candidate.get("caption") or "").casefold()
        score = 0
        if re.search(r"main\s+results?", caption):
            score += 70
        if re.search(r"experimental\s+results?", caption):
            score += 65
        if re.search(r"benchmark(?:\s+results?)?", caption):
            score += 60
        if "performance" in caption:
            score += 50
        if re.search(r"compar(?:ison|ative)", caption):
            score += 40
        if wants_dataset_coverage:
            if "performance" in caption:
                score += 35
            if re.search(r"datasets?|corpora|corpus", caption):
                score += 15
        if wants_comparison and re.search(r"compar(?:ison|ative)", caption):
            score += 50
        if wants_comparison and re.search(r"prior|baseline|models?|methods?", caption):
            score += 30
        scored.append((score, str(candidate.get("table_id") or ""), candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][2]


def _direct_canonical_table_answer(
    task: str,
    documents: list[dict[str, Any]],
    *,
    expected_document_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Render an unambiguous canonical table selection without model transcription.

    Printed caption numbers are authoritative.  ``table_index + 1`` is used
    only when the extraction has no printed number, so missing/out-of-order
    captions cannot shift every later request.  Repeated/list syntax may select
    several tables, but every requested number must resolve exactly once.
    """

    query = str(task or "").strip()
    if not query or _wants_table_inventory(query) or _TABLE_ANALYSIS_RE.search(query):
        return None
    requested_numbers = _requested_table_numbers(query)
    generic_result_request = bool(_GENERIC_RESULT_TABLE_REQUEST_RE.search(query))
    if not requested_numbers and not generic_result_request:
        return None

    expected_ids = {str(item) for item in expected_document_ids or [] if item}
    candidates: list[dict[str, Any]] = []
    for document in documents:
        if not document.get("table_id") or not document.get("document_id"):
            continue
        if caption_identifies_figure(str(document.get("caption") or "")):
            continue
        if expected_ids and str(document.get("document_id")) not in expected_ids:
            continue
        content = str(document.get("content") or "").strip()
        markdown = (
            content.split("content:\n", 1)[1].strip()
            if "content:\n" in content
            else content
        )
        markdown_lines = [line for line in markdown.splitlines() if "|" in line]
        if len(markdown_lines) < 2 or not any(
            re.search(r"\|?\s*:?-{3,}", line) for line in markdown_lines[1:3]
        ):
            continue
        caption = str(document.get("caption") or "").strip()
        if not requested_numbers and (
            not _MAIN_RESULT_TABLE_CAPTION_RE.search(caption)
            or _NON_MAIN_RESULT_TABLE_CAPTION_RE.search(caption)
        ):
            continue
        canonical_number = _canonical_table_number(document)
        if requested_numbers and canonical_number not in requested_numbers:
            continue
        candidates.append(
            {
                **document,
                "_direct_markdown": markdown,
                "_canonical_table_number": canonical_number,
            }
        )

    selected_sources: list[dict[str, Any]]
    if requested_numbers:
        candidates_by_number = {
            number: [
                candidate
                for candidate in candidates
                if candidate.get("_canonical_table_number") == number
            ]
            for number in requested_numbers
        }
        if any(len(items) != 1 for items in candidates_by_number.values()):
            return None
        selected_sources = [
            candidates_by_number[number][0] for number in requested_numbers
        ]
    elif len(candidates) == 1:
        selected_sources = [candidates[0]]
    else:
        selected = _select_unique_generic_result_table(query, candidates)
        if selected is None:
            return None
        selected_sources = [selected]

    vietnamese = bool(re.search(r"[à-ỹđ]", query.casefold())) or "bảng" in query.casefold()
    rendered: list[str] = []
    for source in selected_sources:
        number = source.get("_canonical_table_number")
        caption = str(source.get("caption") or "").strip()
        label = caption or (
            f"Table {number}" if number is not None else "retrieved table"
        )
        filename = str(source.get("filename") or "tài liệu").strip()
        page = source.get("page_number")
        provenance = f"{filename} · page {page}" if page is not None else filename
        if vietnamese:
            rendered.append(
                f"Đây là **{label}**:\n\n"
                f"{source['_direct_markdown']}\n\n"
                "Mình giữ nguyên bảng đã extract, không tự tính thêm chênh lệch. "
                f"Nguồn: {provenance}."
            )
        else:
            rendered.append(
                f"Here is **{label}**:\n\n"
                f"{source['_direct_markdown']}\n\n"
                "This preserves the extracted table without calculating additional deltas. "
                f"Source: {provenance}."
            )

    clean_sources = [
        {
            key: value
            for key, value in source.items()
            if key not in {"_direct_markdown", "_canonical_table_number"}
        }
        for source in selected_sources
    ]
    result_source = {
        **clean_sources[0],
        "_direct_sources": clean_sources,
        "table_ids": [source.get("table_id") for source in clean_sources],
        "document_ids": list(
            dict.fromkeys(
                str(source.get("document_id"))
                for source in clean_sources
                if source.get("document_id")
            )
        ),
    }
    return "\n\n".join(rendered), result_source


def _requested_metric_names(task: str) -> set[str]:
    normalized = " ".join(str(task or "").casefold().split())
    aliases = {
        "accuracy": r"(?<!\w)(?:acc|accuracy)(?!\w)",
        "f1": r"(?<!\w)f1(?!\w)",
        "ccc": r"(?<!\w)ccc(?!\w)",
        "uar": r"(?<!\w)uar(?!\w)",
        "wa": r"(?<!\w)(?:wa|war)(?!\w)",
        "wer": r"(?<!\w)wer(?!\w)",
        "mae": r"(?<!\w)mae(?!\w)",
        "mse": r"(?<!\w)mse(?!\w)",
        "rmse": r"(?<!\w)rmse(?!\w)",
        "precision": r"(?<!\w)precision(?!\w)",
        "recall": r"(?<!\w)recall(?!\w)",
        "auc": r"(?<!\w)auc(?!\w)",
        "eer": r"(?<!\w)eer(?!\w)",
        "hamming_loss": r"(?<!\w)(?:hamming\s+loss|hl)(?!\w)",
        "jaccard": r"(?<!\w)(?:jaccard(?:\s+index)?|jac)(?!\w)",
        "exact_accuracy": r"(?<!\w)(?:exact\s+(?:match\s+)?accuracy|exacc)(?!\w)",
        "binary_accuracy": r"(?<!\w)(?:binary\s+accuracy|per-label\s+accuracy|biacc)(?!\w)",
        "parameters": r"(?<!\w)(?:params?|parameters?)(?!\w)",
    }
    return {
        metric
        for metric, pattern in aliases.items()
        if re.search(pattern, normalized)
    }


def _remove_unsupported_claim_lines(answer: str, validation_payload: dict[str, Any]) -> str:
    def comparable(value: str) -> str:
        compact = " ".join(str(value or "").split()).removesuffix("…").strip()
        compact = re.sub(r"\s*\[[^\]]+\]\s*$", "", compact)
        compact = compact.strip().strip("|").strip()
        return compact

    blocked_prefixes = {
        comparable(str(claim.get("text") or ""))
        for claim in validation_payload.get("unsupported_claims") or []
        if isinstance(claim, dict) and claim.get("text")
    }
    if not blocked_prefixes:
        return ""
    kept: list[str] = []
    for line in str(answer or "").splitlines():
        compact = comparable(line)
        if compact and any(
            compact.startswith(prefix) or prefix.startswith(compact)
            for prefix in blocked_prefixes
            if prefix
        ):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _sanitize_answer_before_retry(
    *,
    answer: str,
    validation: Any,
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
    task: str,
    require_all_focus_documents: bool = False,
    document_identity_resolver: Callable[[str], list[str]] | None = None,
) -> tuple[str, Any | None, bool]:
    """Remove unsupported lines and decide whether a model retry is necessary.

    A valid, still-useful answer should not be regenerated merely because an
    optional derived delta or unrelated benchmark line was unsupported.  For
    explicitly quantitative tasks, sanitization is accepted only when the
    requested metric set remains grounded; otherwise the one bounded retry is
    preserved.
    """

    sanitized = _remove_unsupported_claim_lines(answer, validation.to_dict())
    if not sanitized.strip():
        return "", None, False
    sanitized_validation = validate_answer_claims(
        answer=sanitized,
        documents=documents,
        focus_document_ids=focus_document_ids,
        require_all_focus_documents=require_all_focus_documents,
        answer_document_ids=_resolve_answer_document_ids(
            document_identity_resolver,
            sanitized,
        ),
    )
    requested_metrics = _requested_metric_names(task)
    checked_metrics = {claim.metric for claim in sanitized_validation.checked_claims}
    quantitative_satisfied = (
        not _task_requests_quantitative_evidence(task)
        or (
            bool(checked_metrics)
            and requested_metrics.issubset(checked_metrics)
        )
    )
    accepted = bool(sanitized_validation.valid and quantitative_satisfied)
    return sanitized, sanitized_validation, accepted


def _resolve_answer_document_ids(
    resolver: Callable[[str], list[str]] | None,
    answer: str,
) -> list[str] | None:
    if resolver is None:
        return None
    try:
        return list(dict.fromkeys(str(item) for item in resolver(answer) if item))
    except Exception as exc:  # pragma: no cover - defensive, validation fails closed
        logger.warning(
            "Answer document identity resolution failed: %s",
            type(exc).__name__,
        )
        return []


def _should_reuse_last_retrieval(rewrite: Any, *, original_query: str) -> bool:
    if has_visual_intent(original_query):
        return False
    return bool(
        rewrite.is_followup
        and rewrite.use_last_sources
        and rewrite.answer_intent in {"elaborate", "simplify", "example", "infer_structure"}
    )


def _context_budget(answer_intent: str | None, answer_depth: str | None) -> dict[str, int]:
    detailed = answer_depth == "detailed" or answer_intent in {"elaborate", "compare", "infer_structure"}
    if detailed:
        return {
            "max_sources": 5,
            "max_chars": 4_800,
            "max_chars_per_source": 1_050,
            "max_chunks_per_document": 3,
            "neighbor_window": 1,
            "max_neighbor_chars": 520,
        }
    if answer_depth == "brief":
        return {
            "max_sources": 3,
            "max_chars": 2_700,
            "max_chars_per_source": 750,
            "max_chunks_per_document": 2,
            "neighbor_window": 1,
            "max_neighbor_chars": 360,
        }
    return {
        "max_sources": 4,
        "max_chars": 3_600,
        "max_chars_per_source": 900,
        "max_chunks_per_document": 2,
        "neighbor_window": 1,
        "max_neighbor_chars": 450,
    }


def _document_ids_from_retrieval(retrieval: dict[str, Any] | None) -> list[str]:
    if not retrieval:
        return []
    seen: set[str] = set()
    document_ids: list[str] = []
    for document in retrieval.get("documents") or []:
        document_id = document.get("document_id")
        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


def _filenames_from_documents(documents: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    filenames: list[str] = []
    for document in documents:
        name = str(document.get("filename") or "").strip()
        if name and name not in seen:
            seen.add(name)
            filenames.append(name)
    return filenames


def _scope_documents_to_focus(
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not focus_document_ids:
        return documents
    allowed = {str(doc_id) for doc_id in focus_document_ids}
    # Never fall back to unscoped corpus — empty is safer than cross-paper leak.
    return [
        document
        for document in documents
        if str(document.get("document_id") or "") in allowed
    ]


def _wants_result_tables(query: str) -> bool:
    return has_result_table_intent(query)


def _wants_table_inventory(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    return bool(
        re.search(
            r"(?:mấy|bao\s+nhiêu|liệt\s+kê|danh\s+sách|how\s+many|list|count)"
            r"(?:\s+\S+){0,4}\s+(?:bảng|tables?)\b|"
            r"(?:bảng|tables?)(?:\s+\S+){0,4}\s+"
            r"(?:mấy|bao\s+nhiêu|liệt\s+kê|how\s+many|list|count)\b",
            normalized,
        )
        or re.search(
            r"\b(?:(?:những|các|toàn\s+bộ|tất\s+cả)\s+bảng|bảng\s+nào)\b|"
            r"\b(?:có|gồm|chứa)\s+(?:(?:những|các|toàn\s+bộ|tất\s+cả)\s+)?"
            r"bảng\s+(?:gì|nào)\b|"
            r"\b(?:what|which)\s+tables?\b|"
            r"\b(?:show|display|give)\s+(?:me\s+)?(?:all|the)\s+tables?\b|"
            r"\ball\s+tables?\b",
            normalized,
        )
    )


def _caption_table_number(caption: str) -> int | None:
    match = _EXACT_TABLE_REQUEST_RE.search(str(caption or ""))
    return int(match.group("number")) if match is not None else None


def _requested_table_numbers(query: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for match in _TABLE_NUMBER_SEQUENCE_RE.finditer(str(query or "")):
        for raw_number in re.findall(r"\d+", match.group("sequence")):
            number = int(raw_number)
            if number not in numbers:
                numbers.append(number)
    return tuple(numbers)


def _task_requires_exact_artifact(query: str) -> bool:
    """Exact artifact/quote/page asks must bypass compact semantic cards."""

    lowered = " ".join(str(query or "").lower().split())
    return bool(
        _requested_table_numbers(query)
        or requested_figure_number(query) is not None
        or re.search(r"\b(?:page|trang)\s*\d+\b", lowered)
        or any(marker in lowered for marker in ("exact quote", "verbatim", "trích nguyên", "trich nguyen"))
    )


def _canonical_table_number(table: dict[str, Any]) -> int | None:
    """Prefer the table number printed in the caption over positional index."""

    caption_number = _caption_table_number(str(table.get("caption") or ""))
    if caption_number is not None:
        return caption_number
    table_index = table.get("table_index")
    if isinstance(table_index, int) or str(table_index or "").isdigit():
        return int(table_index) + 1
    return None


def _merge_focused_table_inventory(
    *,
    rag: RagService,
    sources: list[dict[str, Any]],
    focus_document_ids: list[str],
    query: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Prepend every canonical table for an explicit scoped inventory ask."""

    requested_numbers = _requested_table_numbers(str(query or ""))
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
        if requested_numbers:
            tables = [
                table
                for table in tables
                if _canonical_table_number(table) in requested_numbers
            ]
        elif query and _GENERIC_RESULT_TABLE_REQUEST_RE.search(query):
            main_tables = [
                table
                for table in tables
                if _MAIN_RESULT_TABLE_CAPTION_RE.search(str(table.get("caption") or ""))
                and not _NON_MAIN_RESULT_TABLE_CAPTION_RE.search(
                    str(table.get("caption") or "")
                )
            ]
            if main_tables:
                tables = main_tables

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
                    "source_path": document.get("source_path"),
                    "filename": document.get("filename"),
                    "content": content,
                    "page_number": table.get("page_number"),
                    "chunk_type": "table",
                    "artifact_type": "table",
                    "caption": caption or None,
                    "table_id": table_id,
                    "table_index": table.get("table_index"),
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
        and not caption_identifies_figure(str(item.get("caption") or ""))
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


def _query_names_new_paper(query: str, working_state: Any) -> bool:
    from app.services.query_rewrite_service import _query_named_entities

    named = _query_named_entities(query)
    if not named:
        return False
    active_topic = (working_state.active_topic or "").lower()
    active_files = " ".join(working_state.active_filenames or []).lower()
    for entity in named:
        key = entity.lower()
        if key and key not in active_topic and key not in active_files:
            return True
    return False


def _canonicalize_explicit_document_target(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
) -> tuple[Any, list[str]]:
    """Make a current-turn catalog target authoritative over stale chat focus.

    The ordinary rewriter intentionally understands conversational language,
    but it cannot know every filename or alias in the user's local catalog.
    Resolve explicit ``paper/file/Table N X`` references against that catalog,
    then keep query, topic, entities and focus consistent.  A model name used
    only as a baseline inside the active paper is not treated as a document
    switch because this path requires explicit document language.
    """

    query = str(rewrite.original_query or "").strip()
    explicit_targets = _explicit_document_target_entities(query)
    has_document_marker = bool(
        re.search(
            r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
            query,
            flags=re.IGNORECASE,
        )
    )
    has_explicit_filename = bool(re.search(r"\.(?:pdf|md|txt)(?!\w)", query, re.I))
    if explicit_targets:
        lookup_entities = list(explicit_targets)
    elif has_document_marker or has_explicit_filename:
        lookup_entities = _query_named_entities(query)
    else:
        return rewrite, []

    document_ids = rag.resolve_explicit_document_ids_for_query(
        query=query,
        entities=lookup_entities,
        collection_id=collection_id,
        compare=rewrite.answer_intent == "compare",
    )
    if not document_ids:
        return rewrite, []

    if len(lookup_entities) == len(document_ids):
        canonical_entities = lookup_entities
    else:
        canonical_entities = []
        for document_id in document_ids:
            document = rag.get_document(document_id) or {}
            filename = str(document.get("filename") or document_id)
            canonical_entities.append(re.sub(r"\.[A-Za-z0-9]+$", "", filename))

    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "explicit_catalog_document_target",
            "document_ids": document_ids,
        }
    )
    return (
        replace(
            rewrite,
            standalone_query=query,
            is_followup=False,
            current_topic=canonical_entities[0] if canonical_entities else rewrite.current_topic,
            required_entities=canonical_entities,
            use_last_sources=False,
            rewrite_used=False,
            diagnostics=diagnostics,
        ),
        document_ids,
    )


def _apply_document_scope_to_rewrite(rewrite: Any, *, scope: Any) -> Any:
    """Keep rewrite facets while making the pre-router scope authoritative."""

    document_ids = list(scope.document_ids)
    labels = list(scope.labels)
    if not document_ids:
        return rewrite
    query = str(rewrite.original_query or "").strip()
    is_referent = scope.source == "plural_referent"
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": scope.source,
            "document_ids": document_ids,
            "catalog_labels": labels,
            "must_cover_all": bool(scope.must_cover_all),
        }
    )
    topic = " / ".join(labels) if len(labels) >= 2 else labels[0] if labels else None
    standalone_query = (
        f"{' '.join(labels)} {query}".strip() if is_referent else query
    )
    return replace(
        rewrite,
        standalone_query=standalone_query,
        is_followup=is_referent,
        current_topic=topic or rewrite.current_topic,
        required_entities=labels or list(rewrite.required_entities or []),
        use_last_sources=is_referent,
        answer_intent=rewrite.answer_intent,
        rewrite_used=False,
        diagnostics=diagnostics,
    )


def _document_scope_failure_message(rag: RagService, scope: Any) -> str | None:
    if not getattr(scope, "authoritative", False) or getattr(scope, "document_ids", ()):
        return None
    if scope.source == "collection_excluded":
        filenames = _filenames_for_document_ids(
            rag,
            list(scope.collection_removed_ids),
        )
        listed = ", ".join(filenames) or "tài liệu được nhắc tới"
        return (
            f"Mình nhận ra đúng nguồn ({listed}), nhưng nguồn đó nằm ngoài collection "
            "đang được giới hạn. Mình không tự mở rộng sang toàn corpus; hãy đổi collection "
            "hoặc bỏ giới hạn rồi hỏi lại."
        )
    if scope.source == "ambiguous_current_turn":
        candidate_ids: list[str] = []
        surfaces: list[str] = []
        for mention in scope.ambiguous_mentions:
            candidate_ids.extend(mention.candidate_ids)
            if mention.surface not in surfaces:
                surfaces.append(mention.surface)
        filenames = _filenames_for_document_ids(
            rag,
            list(dict.fromkeys(candidate_ids)),
        )
        target = ", ".join(surfaces) or "tên đó"
        choices = "; ".join(filenames) or "nhiều file trong catalog"
        return (
            f"Tên “{target}” khớp nhiều tài liệu: {choices}. "
            "Hãy ghi filename hoặc tiêu đề đầy đủ để mình không chọn nhầm."
        )
    return None


def _canonicalize_natural_document_mentions(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
    existing_focus: list[str],
    answer_intent_hint: str | None = None,
) -> tuple[Any, list[str]]:
    """Turn unambiguous catalog mentions into a coherent retrieval scope.

    This covers case/separator variants such as ``msf ser``, ``MSF-SER`` and
    ``msfser`` without teaching the router individual paper names.  A single
    model mention does not dislodge a different sticky paper unless the turn is
    a clear switch/comparison; two or more unique catalog identities are
    authoritative for the current turn.
    """

    query = str(rewrite.original_query or "").strip()
    resolver = getattr(rag, "resolve_document_mentions_for_query", None)
    if not callable(resolver):
        return rewrite, []
    document_ids = resolver(
        query=query,
        collection_id=collection_id,
        limit=8,
    )
    if not document_ids:
        return rewrite, []

    comparison_request = (
        rewrite.answer_intent == "compare"
        or answer_intent_hint == "compare"
        or _looks_like_multi_document_comparison(query)
    )
    active = set(existing_focus)
    if (
        len(document_ids) == 1
        and active
        and document_ids[0] not in active
        and not comparison_request
        and not _looks_like_topic_switch(query)
        and (rewrite.diagnostics or {}).get("reason") != "topic_switch"
    ):
        return rewrite, []

    labels = _document_labels_for_ids(rag, document_ids)
    if not labels:
        return rewrite, []
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "catalog_document_mentions",
            "document_ids": document_ids,
            "catalog_labels": labels,
            "multi_document": len(document_ids) >= 2,
        }
    )
    intent = "compare" if comparison_request and len(document_ids) >= 2 else rewrite.answer_intent
    topic = labels[0] if len(labels) == 1 else " vs ".join(labels)
    return (
        replace(
            rewrite,
            standalone_query=query,
            is_followup=False,
            current_topic=topic,
            required_entities=labels,
            use_last_sources=False,
            answer_intent=intent,
            rewrite_used=False,
            diagnostics=diagnostics,
        ),
        document_ids,
    )


def _canonicalize_plural_document_referent(
    rag: RagService,
    *,
    rewrite: Any,
    document_ids: list[str],
) -> Any:
    """Bind plural anaphora to the last grounded multi-document referent set."""

    unique_ids = list(dict.fromkeys(str(item) for item in document_ids if item))[:8]
    if len(unique_ids) < 2:
        return rewrite
    labels = _document_labels_for_ids(rag, unique_ids)
    if len(labels) < 2:
        return rewrite
    query = str(rewrite.original_query or "").strip()
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "plural_document_referent",
            "document_ids": unique_ids,
            "catalog_labels": labels,
        }
    )
    return replace(
        rewrite,
        standalone_query=f"{' '.join(labels)} {query}".strip(),
        is_followup=True,
        current_topic=" vs ".join(labels),
        required_entities=labels,
        use_last_sources=True,
        answer_intent="compare",
        rewrite_used=False,
        diagnostics=diagnostics,
    )


def _recover_recent_document_referents(
    rag: RagService,
    *,
    previous_messages: list[Any],
    collection_id: str | None,
) -> list[str]:
    """Backfill referents from the nearest prior user comparison only."""

    resolver = getattr(rag, "resolve_document_mentions_for_query", None)
    if not callable(resolver):
        return []
    for message in reversed(previous_messages[-40:]):
        role, content = _message_role_and_content(message)
        if role != "user" or not _looks_like_multi_document_comparison(content):
            continue
        document_ids = resolver(
            query=content,
            collection_id=collection_id,
            limit=8,
        )
        if len(document_ids) >= 2:
            return list(document_ids)
    return []


def _looks_like_multi_document_comparison(query: str) -> bool:
    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        re.search(
            r"\b(?:so\s+sánh|đối\s+chiếu|khác\s+nhau|giống\s+nhau|versus|compare|"
            r"comparison|differences?|similarities?|against)\b",
            normalized,
        )
        or re.search(r"(?<![a-z0-9])vs\.?\s", normalized)
    )


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower(), str(message.get("content") or "")
    return (
        str(getattr(message, "role", "") or "").lower(),
        str(getattr(message, "content", "") or ""),
    )


def _document_labels_for_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    labels: list[str] = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        filename = str(document.get("filename") or "").strip()
        label = re.sub(r"\.[A-Za-z0-9]+$", "", filename).strip() or str(document_id)
        if label not in labels:
            labels.append(label)
    return labels


def _filenames_for_document_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    filenames: list[str] = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        filename = str(document.get("filename") or "").strip()
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _comparison_topic_from_filenames(filenames: list[str]) -> str | None:
    labels = [re.sub(r"\.[A-Za-z0-9]+$", "", item) for item in filenames if item]
    return " vs ".join(labels[:8]) if len(labels) >= 2 else None


def _resolve_query_document_focus(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
    existing_focus: list[str],
) -> list[str]:
    explicit_targets = _explicit_document_target_entities(rewrite.original_query)
    has_document_marker = bool(
        re.search(
            r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
            rewrite.original_query,
            flags=re.IGNORECASE,
        )
    )
    has_explicit_filename = bool(
        re.search(r"\.(?:pdf|md|txt)(?!\w)", rewrite.original_query, re.I)
    )
    catalog_entities = (
        explicit_targets
        if explicit_targets
        else _query_named_entities(rewrite.original_query)
        if has_document_marker or has_explicit_filename
        else []
    )
    catalog_resolved = rag.resolve_explicit_document_ids_for_query(
        query=rewrite.original_query,
        entities=catalog_entities,
        collection_id=collection_id,
        compare=rewrite.answer_intent == "compare",
    )
    if catalog_resolved:
        return catalog_resolved[:2] if rewrite.answer_intent == "compare" else catalog_resolved[:1]

    if explicit_targets:
        entities = list(explicit_targets)
        # Do not let other names in a correction sentence pull scope back to the
        # old paper.  Resolve only the current-turn canonical target(s).
        resolution_query = " ".join(explicit_targets)
    else:
        entities = []
        if rewrite.current_topic:
            entities.append(str(rewrite.current_topic))
        for entity in rewrite.required_entities or []:
            if entity not in entities:
                entities.append(entity)
        resolution_query = "\n".join(
            part
            for part in [rewrite.original_query, rewrite.standalone_query, rewrite.current_topic]
            if part
        )

    entity_resolved = rag.resolve_document_ids_for_entities(
        entities=entities,
        collection_id=collection_id,
        query=resolution_query,
    )
    if entity_resolved:
        if rewrite.answer_intent == "compare":
            return entity_resolved[:2]
        primary = entity_resolved[:1]
        if existing_focus and rewrite.use_last_sources:
            overlap = [doc_id for doc_id in existing_focus if doc_id in entity_resolved]
            if overlap:
                return overlap[:1]
        return primary

    return existing_focus[:1] if existing_focus else []


def _text_source_document_ids(documents: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for document in documents:
        if document.get("figure_id"):
            continue
        document_id = document.get("document_id")
        if document_id:
            counts[str(document_id)] = counts.get(str(document_id), 0) + 1
    if not counts:
        return []
    return [sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]]


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
        figure_id = item.get("figure_id")
        if figure_id and item.get("image_path") and not item.get("image_url"):
            item["image_url"] = f"/rag/figures/{figure_id}/image"
        if figure_id:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
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


def _retrieval_cache_key(
    *,
    normalized_query: str,
    collection_id: str | None,
    focus_document_ids: list[str],
    retrieval_mode: str,
    index_fingerprint: str,
) -> str:
    raw = json.dumps(
        {
            "query": normalized_query,
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


@router.get("/runs/{run_id}")
async def get_agent_run(
    run_id: str,
    run_store: Annotated[AgentRunStore, Depends(get_agent_run_store)],
) -> dict:
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


@router.get("/runs/{run_id}/debug-trace")
async def get_agent_run_debug_trace(
    run_id: str,
    http_request: Request,
    response: Response,
    run_store: Annotated[AgentRunStore, Depends(get_agent_run_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    if (
        not settings.agent_debug_trace_enabled
        or not request_client_is_loopback(http_request)
    ):
        raise HTTPException(status_code=404, detail="Agent debug trace not found")
    trace = run_store.get_debug_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Agent debug trace not found")
    response.headers["Cache-Control"] = "no-store"
    return trace
