import asyncio
from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.config import Settings, get_settings
from app.llm.openai_client import get_llm_client
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.agent_run_store import AgentRunStore
from app.services.agent_service import AgentService, AnswerStreamChunk
from app.services.chat_history import ChatHistory
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
from . import deps
from .common import logger


# The optional visual worker is a long-lived isolated process.  Keep one client
# per immutable settings signature so a visual query does not reload ColSmol;
# the default feature flag remains disabled and therefore never starts it.
_VISUAL_PAGE_PROVIDERS: dict[
    tuple[str, str, str, str, str, str, str, bool, float], Any
] = {}


async def shutdown_visual_page_providers() -> None:
    """Close all process-backed visual providers owned by the agent module.

    The document-processing coordinator owns its own optional worker, but
    agent queries use a separate process cache.  Explicitly draining this
    cache on application shutdown prevents orphan ColSmol workers and avoids
    retaining a provider created for a previous settings fingerprint.
    """

    providers = list(_VISUAL_PAGE_PROVIDERS.values())
    _VISUAL_PAGE_PROVIDERS.clear()
    for provider in providers:
        close = getattr(provider, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover - defensive shutdown path
                logger.warning(
                    "visual_provider_shutdown_failed error_type=%s",
                    type(provider).__name__,
                )
            continue
        close_sync = getattr(provider, "close", None)
        if close_sync is not None:
            try:
                await asyncio.to_thread(close_sync)
            except Exception:  # pragma: no cover - defensive shutdown path
                logger.warning(
                    "visual_provider_shutdown_failed error_type=%s",
                    type(provider).__name__,
                )


def get_agent_service(settings: Annotated[Settings, Depends(get_settings)]) -> AgentService:
    client = deps.get_llm_client(
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
    client = deps.get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return QueryRewriteService(client=client, default_model=settings.default_model)


def get_intent_router(settings: Annotated[Settings, Depends(get_settings)]) -> IntentRouterService:
    # Same stack as answer LLM (9router/GPT) — do not force local Ollama.
    client = deps.get_llm_client(
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


def _embedding_provider(settings: Settings) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        batch_size=settings.embedding_batch_size,
        transient_retries=settings.embedding_transient_retries,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


def _retrieval_store(settings: Settings) -> LanceDBRetrievalStore:
    return LanceDBRetrievalStore(settings.lancedb_path)

