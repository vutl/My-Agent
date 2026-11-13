import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
import re
import time
from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from app.core.config import Settings, get_settings
from app.core.events import sse_event
from app.core.network import request_client_is_loopback
from app.llm.ollama_client import OllamaError
from app.rag.document_coverage import DocumentCoveragePolicy, detect_document_coverage
from app.services.agent_run_store import AgentRunStore
from app.services.agent_service import AgentService, AnswerStreamChunk
from app.services.chat_history import ChatHistory
from app.services.evidence_validator import (
    AnswerClaimValidationResult,
    validate_answer_claims,
    validate_retrieval_evidence,
)
from app.services.document_scope_service import DocumentScopeResolution, resolve_document_scope
from app.services.document_language import looks_like_document_comparison
from app.services.task_interactions import resume_waiting_task
from app.services.tasks import TaskStore
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
from app.services.tool_decision_service import (
    IntentRouterService,
    ToolDecision,
    answer_needs_local_fallback,
    _deterministic_chat_route_reason,
)
from app.services.conversation_state import ConversationStateStore
from app.services.conversation_memory import ConversationMemoryStore, schedule_memory_fold
from app.services.conversation_runtime import ConversationRuntimeGate
from app.services.long_term_memory import HistoricalConversationSearch, MemoryItemStore
from . import deps
from . import retrieval as retrieval_module
from .answer import _approved_answer_images
from .answer import _stream_graph_and_answer
from .answer import _stream_local_rag_fallback
from .answer import _visual_evidence_sufficiency_context
from .answer import _visual_source_count
from .assistant import _stream_assistant_tool_turn
from .common import FULL_DOCUMENT_MAX_DOCUMENTS
from .common import _elapsed_ms
from .common import logger
from .deps import get_agent_run_store
from .deps import get_agent_service
from .deps import get_chat_history
from .deps import get_conversation_memory_store
from .deps import get_conversation_runtime_gate
from .deps import get_conversation_state_store
from .deps import get_historical_conversation_search
from .deps import get_intent_router
from .deps import get_long_term_memory_store
from .deps import get_query_rewrite_service
from .deps import get_rag_service
from .models import AgentRunRequest
from .router_gate import library_context_signal, library_signal
from .retrieval import _GUESSED_ALIAS_SOURCES
from .retrieval import _PAPER_BY_TERMS_RE
from .retrieval import _asks_library_results_for_named_terms
from .retrieval import _library_mentions_named_terms
from .retrieval import _bounded_second_retrieval_branches
from .retrieval import _cached_retrieval_payload
from .retrieval import _compose_accumulated_retrieval
from .retrieval import _curate_figure_sources
from .retrieval import _document_ids_named_in
from .retrieval import _document_priority_passages
from .retrieval import _dominant_document
from .retrieval import _enrich_visual_sources
from .retrieval import _evidence_backed_focus
from .retrieval import _exact_cached_retrieval_payload
from .retrieval import _execute_second_retrieval_branches
from .retrieval import _filter_figure_sources
from .retrieval import _full_document_sources
from .retrieval import _named_query_terms
from .retrieval import _new_retrieval_evidence
from .retrieval import _normalize_cache_query
from .retrieval import _paper_card_validation_documents
from .retrieval import _restore_cached_paper_evidence_runtime
from .retrieval import _retrieval_cache_key
from .retrieval import _retrieval_has_table_sources
from .retrieval import _retrieval_index_configuration
from .retrieval import _retrieve_with_paper_evidence_cards
from .retrieval import _safe_query_variants
from .retrieval import _second_hop_diagnostics_payload
from .retrieval import _second_retrieval_retry_payload
from .retrieval import _with_translation_variant
from .scope import INFERRED_SCOPE_SOURCES
from .scope import _OTHER_REFERENCE_RE
from .scope import _ambiguous_reference_question
from .scope import _apply_document_scope_to_rewrite
from .scope import _asks_for_active_paper_artifact
from .scope import _catalog_scope_is_loose
from .scope import _conversation_documents
from .scope import _document_ids_from_retrieval
from .scope import _document_scope_failure_message
from .scope import _filenames_for_document_ids
from .scope import _filenames_from_documents
from .scope import _is_library_wide_question
from .scope import _labels_for_document_ids
from .scope import _query_continues_active_paper
from .scope import _query_names_new_paper
from .scope import _resolve_query_document_focus
from .scope import _safe_library_titles
from .scope import _scope_documents_to_focus
from .scope import _should_reuse_last_retrieval
from .scope import _task_requires_exact_artifact
from .scope import _with_sibling_documents
from .scope import _working_state_for_scope
from .tables import _wants_result_tables
from .verification import _answer_reports_missing_information


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
    turn_started_at = time.perf_counter()
    routing_telemetry: dict[str, Any] = {
        "router_bypass": False, "library_signal": None, "handoff": None, "first_delta_ms": None,
    }
    telemetry_context: dict[str, Any] = {}

    def record_routing_telemetry() -> None:
        if telemetry_context:
            run_store.update_metadata(telemetry_context["run_id"], routing_telemetry)
            telemetry_context["debug_trace"].record_router_bypass(routing_telemetry)

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
            working_state = _working_state_for_scope(rag, working_state, scope_resolution, request.task)
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
                **routing_telemetry,
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
        telemetry_context.update(run_id=run.id, debug_trace=debug_trace)
        record_routing_telemetry()

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
        assistant_citations: list[dict[str, Any]] = []
        pending_working_update: dict[str, Any] | None = None
        pending_retrieval_cache: dict[str, Any] | None = None
        memory_eligible = True

        agent_needs_plan: bool | None = None  # the router's plan judgement for an agent turn

        async def assistant_turn(resume_task_id: str | None = None,
                                 router_needs_plan: bool | None = None,
                                 handoff_tools: list[dict[str, Any]] | None = None,
                                 result: dict[str, Any] | None = None) -> AsyncIterator[str]:
            task_result: dict[str, Any] = result if result is not None else {}
            async with aclosing(_stream_assistant_tool_turn(
                service=service, settings=settings, rag=rag, request=request, run_id=run.id,
                conversation_id=conversation_id, model=model, previous_messages=previous_messages,
                assistant_chunks=assistant_chunks, assistant_citations=assistant_citations,
                run_store=run_store, is_disconnected=http_request.is_disconnected, task_result=task_result,
                resume_task_id=resume_task_id, router_needs_plan=router_needs_plan,
                **({"handoff_tools": handoff_tools} if handoff_tools else {}),
            )) as assistant_events:
                async for event in assistant_events:
                    yield event
            if task_result.get("handoff"):
                return
            assistant_message = "".join(assistant_chunks)
            if not assistant_message.strip():
                raise RuntimeError("LLM returned an empty answer")
            assistant_record = history.save_message(
                conversation_id=conversation_id, role="assistant", content=assistant_message,
                model=model, sources=[], citations=assistant_citations,
                message_id=task_result.get("message_id"),
            )
            run_store.complete_run(run.id, assistant_message)
            debug_trace.outcome("completed")
            memory_status = _queue_l2_memory_safely(
                memory_store=memory_store, settings=settings, conversation_id=conversation_id,
                user_text=request.task, assistant_text=assistant_message,
                user_message_id=user_message.id, assistant_message_id=assistant_record.id,
                working_state=working_state,
            )
            yield sse_event("agent.memory.queued", {"run_id": run.id, "conversation_id": conversation_id, **memory_status})
            yield sse_event("run.completed", {"run_id": run.id, "conversation_id": conversation_id})

        try:
            if "assistant_tools" in (request.allowed_tools or []) and callable(getattr(service.client, "chat_with_tools", None)):
                # A reply to a task waiting on ask_user is that task's answer: the worker continues it.
                resumed_task = await resume_waiting_task(
                    store=TaskStore(settings.sqlite_db_path), conversation_id=conversation_id,
                    text=request.task, client=service.client, model=settings.router_model,
                )
                if resumed_task:
                    routing_decision = {"route": "agent", "use_local_retrieval": False,
                                        "selected_tools": ["assistant_tools"], "reason": "task_answer",
                                        "task_id": resumed_task.id}
                    debug_trace.record_route(routing_decision)
                    yield sse_event("agent.route.decided", {"run_id": run.id, "conversation_id": conversation_id,
                                                           **routing_decision})
                    async for event in assistant_turn(resumed_task.id):
                        yield event
                    return
            scope_failure_message = _document_scope_failure_message(
                rag,
                scope_resolution,
            )
            library_titles_for_gate = None
            if settings.router_bypass:
                try:
                    gate_recent_ids = run_store.latest_retrieved_document_ids(
                        conversation_id, expected_collection_id=request.collection_id,
                    )
                    # Existing document context already rules out bypass. Keep
                    # catalog/content lookups off these turns' routing path.
                    routing_telemetry["library_signal"] = library_context_signal(
                        request=request, loaded_working_state=loaded_working_state,
                        scope_resolution=scope_resolution, recent_document_ids=gate_recent_ids,
                        scope_failure=bool(scope_failure_message),
                    )
                    if routing_telemetry["library_signal"] is None:
                        library_titles_for_gate = _safe_library_titles(rag, request.collection_id, raise_on_error=True)
                        routing_telemetry["library_signal"] = library_signal(
                            request=request, loaded_working_state=loaded_working_state,
                            scope_resolution=scope_resolution, recent_document_ids=gate_recent_ids,
                            library_titles=library_titles_for_gate,
                            named_document_ids=_document_ids_named_in(rag, request.task, request.collection_id),
                            asks_library_results=_asks_library_results_for_named_terms(rag, request.task, request.collection_id),
                            named_terms_in_library=_library_mentions_named_terms(rag, request.task, request.collection_id),
                            supports_tools=any(callable(getattr(service.client, name, None))
                                               for name in ("chat_with_tools_stream", "chat_with_tools")),
                            scope_failure=bool(scope_failure_message),
                        )
                except Exception:  # Missing catalog/state evidence must retain the established path.
                    routing_telemetry["library_signal"] = "gate_lookup_failed"
                record_routing_telemetry()
                if routing_telemetry["library_signal"] is None and not _deterministic_chat_route_reason(
                    request.task, has_active_focus=False,
                ):
                    from app.agents.handoffs import router_handoff_tools

                    routing_telemetry["router_bypass"] = True
                    record_routing_telemetry()
                    bypass_route = {"route": "agent", "reason": "router_bypass",
                                    "use_local_retrieval": False, "selected_tools": ["assistant_tools"]}
                    debug_trace.record_route(bypass_route)
                    yield sse_event("timing", {"run_id": run.id, "conversation_id": conversation_id,
                                               "stage": "router", "ms": 0, "route": "agent"})
                    yield sse_event("agent.route.decided", {"run_id": run.id, "conversation_id": conversation_id,
                                                            **bypass_route})
                    bypass_result: dict[str, Any] = {}
                    try:
                        async for event in assistant_turn(
                            router_needs_plan=False, handoff_tools=router_handoff_tools(library_titles_for_gate),
                            result=bypass_result,
                        ):
                            yield event
                    except OllamaError:
                        if routing_telemetry["first_delta_ms"] is not None:
                            raise
                        routing_telemetry["fallback"] = "gateway_before_first_delta"
                        record_routing_telemetry()
                    else:
                        handoff = bypass_result.get("handoff")
                        if not handoff:
                            return
                        routing_telemetry["handoff"] = handoff["name"]
                        # Arguments are diagnostic only. Both handoffs keep the
                        # original request, history, scope and retrieval inputs.
                        routing_telemetry["handoff_arguments"] = handoff["arguments"][:2000]
                        record_routing_telemetry()
                        if handoff["name"] == "plan_task":
                            async for event in assistant_turn(router_needs_plan=True):
                                yield event
                            return
                        # ask_library falls through to the exact existing path.
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
            # A guessed referent, or a name that only repeats the documents
            # already under discussion ("còn bản tiếng Anh… kinh tế Việt Nam"
            # while the Vietnamese article is active): the model decides.
            inferred_scope = scope_resolution.authoritative and bool(scope_resolution.document_ids) and (
                scope_resolution.source in INFERRED_SCOPE_SOURCES
                or set(scope_resolution.document_ids) <= set(loaded_working_state.active_document_ids or [])
            )
            # A name the catalog matched loosely (a title word, a phrase of a
            # file name, a description) is checked by the router, which reads
            # every title and file name: "Thông tư 83" must not stop at the one
            # document titled "Thông tư", nor "the English article" at the
            # Vietnamese one. A file name or identifier match stands alone.
            checked_catalog_scope = (
                scope_resolution.authoritative
                and bool(scope_resolution.document_ids)
                and not inferred_scope
                and _catalog_scope_is_loose(scope_resolution)
            )
            # A comparison where the catalog found one side only ("báo cáo
            # tháng 8 bản tiếng Anh" vs "báo cáo quý III"): the router adds
            # the documents it recognises from titles; the catalog's stay.
            partial_comparison_scope = (
                scope_resolution.authoritative
                and len(scope_resolution.document_ids) == 1
                and not inferred_scope
                and not checked_catalog_scope
                and looks_like_document_comparison(request.task)
            )
            library_titles_for_router = (library_titles_for_gate if library_titles_for_gate is not None
                                         else _safe_library_titles(rag, request.collection_id))
            rewrite_task: asyncio.Task[QueryRewriteResult] | None = None
            if not (scope_resolution.authoritative and scope_resolution.document_ids):
                # The rewrite needs nothing from the router; running both model
                # calls at once halves the wait before retrieval on vague
                # follow-ups. Cancelled when the router skips retrieval.
                rewrite_task = asyncio.create_task(
                    query_rewriter.rewrite(
                        query=request.task,
                        previous_messages=previous_messages,
                        model=request.model,
                        working_topic=working_state.active_topic,
                        working_document_hint=", ".join(working_state.active_filenames or []),
                    )
                )
            router_started_at = time.perf_counter()
            try:
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
                    # Named documents skip the model; a guessed referent
                    # ("nó", "bài kia") is the model's call, the guess is its fallback.
                    resolved_document_ids=(
                        list(scope_resolution.document_ids)
                        if scope_resolution.authoritative and not inferred_scope
                        and not partial_comparison_scope and not checked_catalog_scope
                        else None
                    ),
                    inferred_document_ids=(
                        list(scope_resolution.document_ids) if inferred_scope or checked_catalog_scope else None
                    ),
                    library_titles=library_titles_for_router,
                    conversation_documents=_conversation_documents(loaded_working_state),
                )
                agent_needs_plan = getattr(tool_decision, "needs_plan", None)
            except BaseException:
                if rewrite_task is not None:
                    rewrite_task.cancel()
                raise
            referent_ids = [
                document_id for document_id in getattr(tool_decision, "referent_document_ids", ()) or ()
                if request.collection_id is None
                or document_id in set(rag.collection_document_ids(request.collection_id))
            ]
            if referent_ids and (inferred_scope or not scope_resolution.authoritative):
                scope_resolution = DocumentScopeResolution(
                    document_ids=tuple(referent_ids),
                    labels=tuple(_labels_for_document_ids(rag, referent_ids)),
                    source="model_referent",
                    must_cover_all=len(referent_ids) >= 2,
                    authoritative=True,
                )
                working_state = _working_state_for_scope(rag, loaded_working_state, scope_resolution, request.task)
                recent_document_ids = list(referent_ids)
                if rewrite_task is not None:
                    rewrite_task.cancel()
                    rewrite_task = None
            added_ids = [
                document_id for document_id in (
                    [getattr(tool_decision, "library_document_id", None)]
                    + list(getattr(tool_decision, "library_document_ids", ()) or ())
                )
                if document_id and document_id not in scope_resolution.document_ids
                and (request.collection_id is None
                     or document_id in set(rag.collection_document_ids(request.collection_id)))
            ]
            router_named_ids = [
                document_id for document_id in dict.fromkeys(
                    [getattr(tool_decision, "library_document_id", None)]
                    + list(getattr(tool_decision, "library_document_ids", ()) or ())
                )
                if document_id and (request.collection_id is None
                                    or document_id in set(rag.collection_document_ids(request.collection_id)))
            ]
            if (
                checked_catalog_scope and not referent_ids and router_named_ids
                and set(router_named_ids) != set(scope_resolution.document_ids)
                # Editions sharing a name are read together; the router
                # overrides that set only with a document outside it.
                and not (
                    scope_resolution.source == "small_ambiguity_set"
                    and set(router_named_ids) <= set(scope_resolution.document_ids)
                )
            ):
                # The router read the name differently from the loose catalog
                # match; its documents are the scope. A comparison keeps the
                # catalog's side as well.
                checked_ids = (
                    [*scope_resolution.document_ids, *router_named_ids]
                    if looks_like_document_comparison(request.task)
                    else router_named_ids
                )
                checked_ids = list(dict.fromkeys(checked_ids))[:FULL_DOCUMENT_MAX_DOCUMENTS]
                scope_resolution = DocumentScopeResolution(
                    document_ids=tuple(checked_ids),
                    labels=tuple(_labels_for_document_ids(rag, checked_ids)),
                    source="router_checked_catalog",
                    must_cover_all=len(checked_ids) >= 2,
                    authoritative=True,
                )
                working_state = _working_state_for_scope(rag, loaded_working_state, scope_resolution, request.task)
                recent_document_ids = list(checked_ids)
            if partial_comparison_scope and added_ids:
                joint_ids = list(dict.fromkeys([*scope_resolution.document_ids, *added_ids]))[:FULL_DOCUMENT_MAX_DOCUMENTS]
                scope_resolution = DocumentScopeResolution(
                    document_ids=tuple(joint_ids),
                    labels=tuple(_labels_for_document_ids(rag, joint_ids)),
                    source="catalog_and_router",
                    must_cover_all=True,
                    authoritative=True,
                )
                working_state = _working_state_for_scope(rag, loaded_working_state, scope_resolution, request.task)
                recent_document_ids = list(joint_ids)
            ambiguous_ids = [
                document_id for document_id in getattr(tool_decision, "ambiguous_referent_ids", ()) or ()
                if request.collection_id is None
                or document_id in set(rag.collection_document_ids(request.collection_id))
            ]
            # "the other one" with two or more other documents in the
            # conversation is ambiguous whatever the router guessed.
            other_documents = [
                document_id for document_id, active in _conversation_documents(loaded_working_state)[-4:]
                if not active
            ]
            other_reference_is_ambiguous = (
                len(other_documents) >= 2
                and bool(_OTHER_REFERENCE_RE.search(request.task))
                and not scope_resolution.mentions
                and not _document_ids_named_in(rag, request.task, request.collection_id)
            )
            if other_reference_is_ambiguous:
                ambiguous_ids = other_documents
            if len(ambiguous_ids) >= 2 and (
                other_reference_is_ambiguous or (not referent_ids and not router_named_ids)
            ):
                # "Còn bài kia?" after three papers: ask, do not pick one.
                router_titles = dict(library_titles_for_router)
                clarification = _ambiguous_reference_question(
                    [
                        (re.sub(r"\s*\([^()]*\)\s*$", "", router_titles.get(document_id, "")) or label)[:80]
                        for document_id, label in zip(ambiguous_ids, _labels_for_document_ids(rag, ambiguous_ids))
                    ],
                    request.task,
                )
                memory_eligible = False
                yield sse_event(
                    "agent.scope.rejected",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "document_scope": {"source": "ambiguous_reference", "document_ids": ambiguous_ids},
                    },
                )
                assistant_chunks.append(clarification)
                yield sse_event("message.delta", {"delta": clarification})
                assistant_record = history.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=clarification,
                    model=model,
                    sources=[],
                )
                run_store.complete_run(run.id, clarification)
                debug_trace.outcome("completed")
                yield sse_event(
                    "run.completed",
                    {"run_id": run.id, "conversation_id": conversation_id},
                )
                return
            routing_decision = tool_decision.to_dict()
            previous_ids = list(loaded_working_state.active_document_ids or [])[:3]
            if previous_ids:
                # Lets the answer say which document a switched turn is about
                # instead of carrying the previous one over ("bài giảng kia").
                routing_decision["previous_documents"] = dict(
                    zip(previous_ids, _labels_for_document_ids(rag, previous_ids))
                )
            if (
                not routing_decision.get("use_local_retrieval")
                and routing_decision.get("route") != "agent"
                and "search_local_docs" in (request.allowed_tools or [])
                and (
                    getattr(tool_decision, "scope", None) == "library"
                    or (
                        getattr(tool_decision, "scope", None) is None
                        and _is_library_wide_question(request.task)
                    )
                    or _asks_library_results_for_named_terms(rag, request.task, request.collection_id)
                    or bool(_document_ids_named_in(rag, request.task, request.collection_id))
                    # A document missing from the library: what the library
                    # says about the same subject still answers with sources.
                    or bool(getattr(tool_decision, "unlisted_document", None))
                )
            ):
                # "How many papers in my library…" is about the library itself.
                routing_decision = {
                    **routing_decision,
                    "route": "file_qa",
                    "use_local_retrieval": True,
                    "selected_tools": ["search_local_docs"],
                    "reason": "library_reference_override",
                }
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
            if routing_decision.get("route") == "agent" and callable(getattr(service.client, "chat_with_tools", None)):
                if rewrite_task is not None:
                    rewrite_task.cancel()
                async for event in assistant_turn(router_needs_plan=agent_needs_plan):
                    yield event
                return
            if not routing_decision["use_local_retrieval"]:
                if rewrite_task is not None:
                    rewrite_task.cancel()
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
                    validate_quantitative_claims=True,
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
                rewrite = await rewrite_task
            catalog_focus_ids = (
                list(scope_resolution.document_ids)
                if scope_resolution.authoritative
                else []
            ) or _document_ids_named_in(rag, request.task, request.collection_id)
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
            router_library_id = getattr(tool_decision, "library_document_id", None)
            # "bài có Medusa với MemoCMT": a paper described by terms it contains.
            # The one document mentioning all of them outranks the router's guess.
            term_owner_ids = (
                rag.documents_containing_all_terms(
                    _named_query_terms(request.task), request.collection_id,
                )
                if not catalog_focus_ids
                and _PAPER_BY_TERMS_RE.search(request.task)
                and callable(getattr(rag, "documents_containing_all_terms", None))
                else []
            )
            if len(term_owner_ids) == 1:
                router_library_id = term_owner_ids[0]
                entity_focus_ids = [] if entity_focus_ids != term_owner_ids else entity_focus_ids
            elif (
                router_library_id
                and not catalog_focus_ids
                and entity_focus_ids
                and router_library_id not in entity_focus_ids
            ):
                # Entities the rewrite extracted ("NSO", "long") are weaker than
                # the router, which saw the titles and the conversation.
                entity_focus_ids = []
            router_library_ids = list(getattr(tool_decision, "library_document_ids", ()) or ())
            if (
                router_library_ids
                and not catalog_focus_ids
                and entity_focus_ids
                and not set(entity_focus_ids) & set(router_library_ids)
                and len(term_owner_ids) != 1
            ):
                # Rewrite entities ("Viet" → a Vietnam article) are weaker than
                # the documents the router listed for this turn.
                entity_focus_ids = []
            # Several papers described (not named) in one message: the router
            # only sees titles, so its picks are guesses. Scan the library and
            # let the answer model match the descriptions against the evidence.
            described_papers_compare = bool(
                len(router_library_ids) >= 2 and not catalog_focus_ids and len(term_owner_ids) != 1
                and len(entity_focus_ids) < 2
            )
            if not entity_focus_ids and router_library_id and not _query_continues_active_paper(
                router_library_id, working_state,
            ):
                # The router recognised a library document from a translated,
                # misspelled or described reference the catalog could not match.
                entity_focus_ids = [router_library_id]
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
            # The router sees the library and the conversation; when it names
            # the active paper, dataset or metric names ("IEMOCAP và LSSED")
            # that the rewrite heuristics read as a topic switch do not move focus.
            router_keeps_active_paper = bool(
                router_library_id and _query_continues_active_paper(router_library_id, working_state)
            )
            # ...and when it names another library document ("whisper corpus
            # ghi âm nixon" = WHiSER while MSF-SER is active), sticky focus
            # must not override it.
            router_switches_paper = bool(router_library_id and not router_keeps_active_paper) or (
                described_papers_compare
                and not set(router_library_ids) <= set(working_state.active_document_ids or [])
            )
            names_new_paper = _query_names_new_paper(
                request.task, working_state, rag=rag, collection_id=request.collection_id,
            )
            same_paper_followup = not router_switches_paper and (
                working_state.has_active_docs
                and not catalog_focus_ids
                and (
                    router_keeps_active_paper
                    or (
                        (rewrite.diagnostics or {}).get("reason") == "topic_switch"
                        and _asks_for_active_paper_artifact(request.task)
                        and not names_new_paper
                    )
                )
            ) or not router_switches_paper and (
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
                and (rewrite_preserves_working_focus or not names_new_paper)
            )
            # Library questions span every document even when a name in them
            # resolves to one ("which papers use Whisper" ~ WHiSER); only an
            # explicit list of several papers keeps the comparison scoped.
            router_scope = getattr(tool_decision, "scope", None)
            library_wide = (
                (
                    router_scope == "library"
                    if router_scope is not None
                    else _is_library_wide_question(request.task)
                )
                or described_papers_compare
            ) and not (
                scope_resolution.must_cover_all and len(catalog_focus_ids) >= 2
                # "Which of my files cover wheat?" asks the whole library even
                # when the router named the two whose titles say "wheat".
                and not (router_scope == "library" and _is_library_wide_question(request.task))
            )
            if library_wide:
                focus_document_ids = []
                same_paper_followup = False
            if same_paper_followup:
                focus_document_ids = list(working_state.active_document_ids)
            elif (
                rewrite.use_last_sources and not focus_document_ids and working_state.active_document_ids
                and not library_wide
            ):
                focus_document_ids = list(working_state.active_document_ids)
            if request.collection_id is not None:
                collection_ids = set(rag.collection_document_ids(request.collection_id))
                focus_document_ids = [
                    document_id
                    for document_id in focus_document_ids
                    if document_id in collection_ids
                ]
            # Hard focus needs an explicit reference: a catalog name/alias,
            # a document id, or a paper identified by terms only it contains
            # (plus sticky follow-ups). Anything else — a router pick from
            # titles, a description or typo match, an entity resolved to a
            # file — is a guess and must agree with library-wide retrieval.
            explicit_focus = (
                # A reference resolved from the conversation ("bài kia" → the
                # other paper discussed) carries no words of that paper, so
                # retrieval on the turn cannot confirm it; it is not a guess.
                scope_resolution.source == "model_referent"
                and bool(focus_document_ids)
                and set(focus_document_ids) <= set(scope_resolution.document_ids)
            ) or (
                # The paper already under discussion (vague follow-ups such as
                # "cái hình đó") is context, not a guess.
                # Compare with the state loaded before this turn: the working
                # state above already holds this turn's own (possibly guessed) scope.
                bool(focus_document_ids)
                and set(focus_document_ids) <= set(loaded_working_state.active_document_ids or [])
            ) or (
                bool(_document_ids_named_in(rag, request.task, request.collection_id))
                or len(term_owner_ids) == 1
                or (
                    bool(catalog_focus_ids)
                    and any(
                        getattr(mention, "alias_source", "") not in _GUESSED_ALIAS_SOURCES
                        for mention in scope_resolution.mentions
                        if getattr(mention, "document_id", None)
                    )
                )
            )
            guessed_focus = (
                len(focus_document_ids) == 1 and not same_paper_followup and not library_wide and not explicit_focus
            )
            if guessed_focus:
                verdict, other = await _evidence_backed_focus(
                    rag, settings, request.task, focus_document_ids[0], request.collection_id,
                )
                if verdict == "switch" and other:
                    focus_document_ids = [other]
                elif verdict == "drop":
                    focus_document_ids = []
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
                if _should_reuse_last_retrieval(rewrite, original_query=request.task) and not library_wide
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
            elif rewrite.is_followup and rewrite.use_last_sources and not library_wide:
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
            coverage_policy = detect_document_coverage(request.task)
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
                original_query=request.task,
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
                paper_evidence_context, paper_evidence_coverage = (
                    _restore_cached_paper_evidence_runtime(
                        settings=settings,
                        retrieval=retrieval,
                        focus_document_ids=focus_document_ids,
                        original_task=request.task,
                        answer_intent=answer_intent,
                    )
                )
            elif exact_cached_retrieval:
                retrieval = _exact_cached_retrieval_payload(exact_cached_retrieval)
                paper_evidence_context, paper_evidence_coverage = (
                    _restore_cached_paper_evidence_runtime(
                        settings=settings,
                        retrieval=retrieval,
                        focus_document_ids=focus_document_ids,
                        original_task=request.task,
                        answer_intent=answer_intent,
                    )
                )
            else:
                card_result = None
                if (
                    not _task_requires_exact_artifact(request.task)
                    and not coverage_policy.enabled
                    and not library_wide
                ):
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
                    retrieval = await retrieval_module._retrieve_for_agent(
                        rag=rag,
                        settings=settings,
                        query=retrieval_query,
                        coverage_query=request.task,
                        collection_id=request.collection_id,
                        retrieval_mode=request.retrieval_mode,
                        focus_document_ids=focus_document_ids,
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_legacy_tables=wants_result_tables and bool(focus_document_ids),
                        must_cover_all_documents=scope_resolution.must_cover_all,
                        library_scope=library_wide,
                        coverage_policy=coverage_policy,
                        query_variants=_with_translation_variant(
                            _safe_query_variants(
                                request.task,
                                retrieval_query,
                                rewrite.required_entities,
                            ),
                            getattr(tool_decision, "query_translation", None),
                            request.task,
                        ),
                    )
            if paper_evidence_coverage and (
                cached_retrieval or exact_cached_retrieval
            ):
                yield sse_event(
                    "evidence.card.coverage",
                    {
                        "run_id": run.id,
                        "conversation_id": conversation_id,
                        "documents": paper_evidence_coverage,
                        "mode": retrieval["mode"],
                        "cache_hit": True,
                    },
                )
                for paper_coverage in paper_evidence_coverage:
                    yield sse_event(
                        "evidence.paper.ready",
                        {
                            "run_id": run.id,
                            "conversation_id": conversation_id,
                            **paper_coverage,
                            "cache_hit": True,
                        },
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
                        coverage_policy=coverage_policy,
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
                        coverage_query=request.task,
                        retrievals=[initial_retrieval, *prepared_branch_retrievals],
                        answer_intent=answer_intent,
                        answer_depth=answer_depth,
                        include_visual_boost=include_visual,
                        prefer_tables=wants_result_tables,
                        required_document_ids=list(
                            focus_document_ids or entity_focus_ids
                        ),
                        coverage_policy=coverage_policy,
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
            evidence_sufficiency_context = _visual_evidence_sufficiency_context(
                request.task,
                retrieved_docs,
            )
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
            # A library-wide answer touches every paper; it is not a new focus.
            if validation.valid and sticky_ids and retrieval.get("mode") != "library_scan":
                pending_working_update = {
                    "document_ids": sticky_ids,
                    "topic": next_topic,
                    "filenames": sticky_files or working_state.active_filenames,
                    "answer_intent": answer_intent,
                    "source_turn_id": user_message.id,
                }
            answer_images = _approved_answer_images(
                retrieved_docs, settings=settings,
                max_images=(coverage_policy.max_images if coverage_policy else 4),
                query=request.task,
            )
            # Two or three documents compared in one question (named or
            # described): read them whole from the first answer, as chat
            # products do with uploaded files; excerpts per document were too
            # thin and the model rarely flagged what it was missing.
            # The router's own list first: when more candidates than the cap
            # exist, read the documents the router chose rather than none.
            comparison_ids = list(dict.fromkeys(
                [*router_library_ids, *(focus_document_ids or entity_focus_ids or [])]
            ))
            if len(router_library_ids) >= 2:
                comparison_ids = comparison_ids[:FULL_DOCUMENT_MAX_DOCUMENTS]
            # Any turn scoped to 2–3 documents — named, resolved by the
            # catalog, a small ambiguous set or the router — reads them whole
            # (or their question-focused sections) from the first answer.
            comparison_turn = 2 <= len(comparison_ids) <= FULL_DOCUMENT_MAX_DOCUMENTS
            full_document_ids = list(
                (comparison_ids if comparison_turn else [])
                or focus_document_ids
                or entity_focus_ids
                or (sticky_ids if len(sticky_ids) == 1 else [])
                or _dominant_document(retrieved_docs)
            )
            full_document_query = " ".join(
                part for part in (
                    request.task,
                    getattr(rewrite, "standalone_query", None),
                    getattr(tool_decision, "query_translation", None),
                ) if part
            )
            sub_questions = list(getattr(tool_decision, "sub_questions", ()) or ())
            priority_passages = (
                await _document_priority_passages(
                    rag, settings, full_document_ids, [request.task, *sub_questions],
                )
                if full_document_ids else {}
            )
            full_document_query = " ".join([full_document_query, *sub_questions])
            full_document_candidates = _full_document_sources(
                rag,
                settings,
                full_document_ids,
                full_document_query,
                priority_passages,
            )
            answer_kwargs = dict(
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
                validation_docs=_paper_card_validation_documents(
                    settings=settings,
                    retrieval=retrieval,
                    retrieved_docs=retrieved_docs,
                    focus_document_ids=(focus_document_ids or entity_focus_ids or sticky_ids),
                ),
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
                evidence_sufficiency_context=evidence_sufficiency_context,
                paper_evidence_coverage=paper_evidence_coverage,
                paper_section_streaming_enabled=settings.paper_section_streaming_enabled,
                image_data_urls=answer_images["urls"],
                image_attachment_labels=answer_images["labels"],
                approved_image_sources=answer_images["sources"],
                answer_verify_mode=get_settings().answer_verify_mode,
                citations_out=assistant_citations,
            )
            first_pass_full = (
                full_document_candidates
                if settings.full_document_context_mode == "always"
                or (comparison_turn and settings.full_document_context_mode != "off")
                else []
            )
            answer_signals: dict[str, Any] = {}
            async for event in _stream_graph_and_answer(
                **answer_kwargs, full_document_sources=first_pass_full, answer_signals=answer_signals,
            ):
                yield event
            escalation_sources = full_document_candidates
            if (
                settings.full_document_context_mode == "auto"
                and full_document_ids
                and (
                    answer_signals.get("needs_more_context")
                    or _answer_reports_missing_information("".join(assistant_chunks))
                )
            ):
                # Already read whole: what is missing may be in a companion of
                # the same series (the report behind a slide deck, another
                # edition); add the closest siblings, else do not repeat.
                expanded_ids = _with_sibling_documents(rag, full_document_ids, request.collection_id)
                if first_pass_full or not full_document_candidates:
                    escalation_sources = (
                        _full_document_sources(
                            rag, settings, expanded_ids, full_document_query,
                            {**priority_passages, **(await _document_priority_passages(
                                rag, settings, [item for item in expanded_ids if item not in priority_passages],
                                [request.task, *sub_questions],
                            ))},
                        )
                        if expanded_ids != full_document_ids
                        else []
                    )
            else:
                escalation_sources = []
            if escalation_sources:
                # The excerpts lacked what was asked; read the whole paper and
                # answer again, replacing the shown draft.
                assistant_chunks.clear()
                yield sse_event(
                    "message.reset",
                    {"run_id": run.id, "conversation_id": conversation_id, "reason": "full_document"},
                )
                async for event in _stream_graph_and_answer(
                    **answer_kwargs, full_document_sources=escalation_sources,
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
                citations=assistant_citations,
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
        from app.llm.scheduler import LLM_SCHEDULER

        async with runtime_gate.turn(conversation_id), LLM_SCHEDULER.interactive():
            async for event in unlocked_event_stream():
                if routing_telemetry["first_delta_ms"] is None and event.startswith((
                    "event: message.delta\n", "event: tool.call.pending\n", "event: tool.call.started\n",
                )):
                    routing_telemetry["first_delta_ms"] = _elapsed_ms(turn_started_at)
                    record_routing_telemetry()
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
    client = deps.get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=min(settings.request_timeout_seconds, 45.0),
    )
    deps.schedule_memory_fold(
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

