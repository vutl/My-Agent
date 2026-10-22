import asyncio
from contextlib import asynccontextmanager
import faulthandler
import logging
import threading
import signal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import agent, assistant, catalog, chat, files, health, lightrag_routes, rag

# `kill -USR1 <pid>` writes every thread's Python stack to stderr: a hung
# ingest or answer can be diagnosed without stopping the server.
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
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
from app.services.document_processing_service import DocumentProcessingService
from app.services.long_term_memory import MemoryItemStore
from app.services.agent_run_store import AgentRunStore


logger = logging.getLogger(__name__)


def _refresh_catalog_titles(db_path) -> None:
    try:
        from app.catalog.title_refresh import refresh_document_titles

        changed = refresh_document_titles(db_path)
        if changed:
            logger.info("catalog_titles_refreshed changed=%s", changed)
    except Exception as exc:  # noqa: BLE001 - catalog polish must never block startup
        logger.warning("catalog_title_refresh_failed error_type=%s", type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Data migrations run when a server starts, never on import: importing
    # app.main (tests do) must not write to the configured library.
    threading.Thread(
        target=_refresh_catalog_titles, args=(settings.sqlite_db_path,),
        name="catalog-title-refresh", daemon=True,
    ).start()
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
    document_processing = DocumentProcessingService.from_settings(settings)
    document_processing_stop = asyncio.Event()
    # Canonical local search must remain responsive when an optional visual
    # model is cold-indexing a large PDF.  The coordinator has SQLite leases,
    # so two narrowly scoped workers are safe: one for folder/local work and
    # one for derived enrichment.  This is deliberately bounded to 1 + 1 for
    # the M1 16 GB deployment rather than unbounded background concurrency.
    document_processing_tasks = [
        asyncio.create_task(
            document_processing.worker_loop(document_processing_stop, lane="local")
        ),
        asyncio.create_task(
            document_processing.worker_loop(document_processing_stop, lane="enrichment")
        ),
    ]
    app.state.document_processing_service = document_processing
    from app.services.reminders import ReminderStore, reminder_worker

    reminder_stop = asyncio.Event()
    background_tasks: list[asyncio.Task] = []
    deliverers = []
    if settings.telegram_bot_token and settings.telegram_owner_id:
        from app.channels.telegram import AyaTelegramBot

        telegram_bot = AyaTelegramBot(token=settings.telegram_bot_token, owner_id=settings.telegram_owner_id,
                                      backend_url=settings.telegram_backend_url)

        async def deliver_to_telegram(reminder) -> None:
            await telegram_bot.send_to_owner(f"⏰ Aya nhắc cậu nè~\n{reminder.text}")

        deliverers.append(deliver_to_telegram)
        background_tasks.append(asyncio.create_task(telegram_bot.run(reminder_stop)))
    background_tasks.append(asyncio.create_task(
        reminder_worker(ReminderStore(settings.sqlite_db_path), reminder_stop, deliverers=deliverers)
    ))
    runtime_gate: ConversationRuntimeGate = app.state.conversation_runtime_gate
    remove_busy_hook = runtime_gate.add_busy_callback(global_foreground_started)
    remove_idle_hook = runtime_gate.add_idle_callback(global_foreground_finished)

    try:
        yield
    finally:
        reminder_stop.set()
        for task in background_tasks:
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - shutdown must continue
                task.cancel()
        document_processing_stop.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(*document_processing_tasks), timeout=3.0
            )
        except asyncio.TimeoutError:
            for task in document_processing_tasks:
                task.cancel()
            await asyncio.gather(*document_processing_tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(document_processing.close(), timeout=10.0)
        except Exception as exc:
            logger.warning(
                "document_processing_shutdown_failed error_type=%s",
                type(exc).__name__,
            )
        try:
            await asyncio.wait_for(agent.shutdown_visual_page_providers(), timeout=10.0)
        except Exception as exc:
            logger.warning(
                "agent_visual_provider_shutdown_failed error_type=%s",
                type(exc).__name__,
            )
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
    app.include_router(assistant.router)
    app.include_router(files.router)
    app.include_router(catalog.router)
    app.include_router(rag.router)
    app.include_router(lightrag_routes.router)
    return app


app = create_app()
