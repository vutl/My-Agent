import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import agent, catalog, chat, files, health, lightrag_routes, rag
from app.core.config import get_settings, validate_runtime_model_policy
from app.db.sqlite import init_db
from app.lightrag.client import init_lightrag, shutdown_lightrag
from app.llm.openai_client import get_llm_client
from app.services.conversation_memory import (
    ConversationMemoryStore,
    global_foreground_finished,
    global_foreground_started,
    shutdown_memory_fold_coordinators,
    start_memory_fold_coordinator,
)
from app.services.conversation_runtime import ConversationRuntimeGate
from app.services.long_term_memory import MemoryItemStore
from app.services.agent_run_store import AgentRunStore


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.lightrag_enabled:
        await init_lightrag(settings)

    memory_store = ConversationMemoryStore(settings.sqlite_db_path)
    long_term_store = MemoryItemStore(settings.sqlite_db_path)
    memory_client = get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=min(settings.request_timeout_seconds, 45.0),
    )

    async def apply_memory_ops(
        conversation_id: str,
        source_turn_seq: int,
        operations: list[dict],
    ) -> None:
        await asyncio.to_thread(
            long_term_store.apply_operations,
            operations,
            source_conversation_id=conversation_id,
            source_turn_seq=source_turn_seq,
        )

    coordinator = start_memory_fold_coordinator(
        store=memory_store,
        client=memory_client,
        model=settings.default_model,
        on_memory_ops=apply_memory_ops,
        debounce_seconds=settings.memory_fold_debounce_seconds,
    )
    app.state.conversation_memory_coordinator = coordinator
    runtime_gate: ConversationRuntimeGate = app.state.conversation_runtime_gate
    remove_busy_hook = runtime_gate.add_busy_callback(global_foreground_started)
    remove_idle_hook = runtime_gate.add_idle_callback(global_foreground_finished)

    try:
        yield
    finally:
        remove_busy_hook()
        remove_idle_hook()
        await shutdown_memory_fold_coordinators(
            timeout_seconds=settings.memory_worker_shutdown_timeout_seconds
        )
        if settings.lightrag_enabled:
            await shutdown_lightrag()


def create_app() -> FastAPI:
    settings = get_settings()
    validate_runtime_model_policy(settings)
    init_db(settings.sqlite_db_path)
    try:
        AgentRunStore(settings.sqlite_db_path).purge_debug_traces(
            max_runs=settings.agent_debug_trace_max_runs,
        )
    except Exception as exc:
        # Optional diagnostics must never prevent the assistant from starting.
        logger.warning(
            "debug_trace_startup_purge_failed error_type=%s",
            type(exc).__name__,
        )

    app = FastAPI(
        title="My AI Agent Backend",
        version=settings.app_version,
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )
    app.state.conversation_runtime_gate = ConversationRuntimeGate()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(agent.router)
    app.include_router(files.router)
    app.include_router(catalog.router)
    app.include_router(rag.router)
    app.include_router(lightrag_routes.router)
    return app


app = create_app()
