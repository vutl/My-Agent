from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from lightrag import LightRAG

from app.lightrag.adapters import build_ollama_embedding_func, build_router_llm_complete

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

_rag: LightRAG | None = None
_initialized = False


async def init_lightrag(settings: Settings) -> LightRAG | None:
    global _rag, _initialized
    if not settings.lightrag_enabled:
        logger.info("LightRAG disabled via LIGHTRAG_ENABLED=false")
        return None

    working_dir = settings.lightrag_working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    # LightRAG's role queue has its own worker/watchdog timeout in addition to
    # the provider timeout enforced by the adapter.  Extraction may consume one
    # full provider timeout, then legitimately retry the exact same model once.
    # Give only the background extract role enough room for that retry; query
    # roles retain LightRAG's shorter default so foreground chat still fails
    # promptly on a provider outage. LightRAG itself doubles this role timeout
    # for the worker watchdog and adds a small health-check grace period.
    extract_role_timeout = math.ceil(
        settings.lightrag_llm_timeout_seconds
        * (settings.lightrag_llm_timeout_retries + 1)
        + 60
    )

    _rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=build_router_llm_complete(settings),
        llm_model_name=settings.lightrag_llm_model,
        llm_model_max_async=settings.lightrag_llm_max_async,
        role_llm_configs={"extract": {"timeout": extract_role_timeout}},
        chunk_token_size=settings.lightrag_chunk_token_size,
        chunk_overlap_token_size=settings.lightrag_chunk_overlap_token_size,
        embedding_func=build_ollama_embedding_func(settings),
    )
    await _rag.initialize_storages()
    _initialized = True
    logger.info(
        "LightRAG initialized (model=%s, max_async=%s, extract_timeout=%ss, "
        "chunk=%s/%s, working_dir=%s)",
        settings.lightrag_llm_model,
        settings.lightrag_llm_max_async,
        extract_role_timeout,
        settings.lightrag_chunk_token_size,
        settings.lightrag_chunk_overlap_token_size,
        working_dir,
    )
    return _rag


async def shutdown_lightrag() -> None:
    global _rag, _initialized
    if _rag is not None and _initialized:
        await _rag.finalize_storages()
        logger.info("LightRAG storages finalized")
    _rag = None
    _initialized = False


def get_lightrag() -> LightRAG:
    if _rag is None or not _initialized:
        raise RuntimeError("LightRAG is not initialized. Ensure app lifespan has run.")
    return _rag
