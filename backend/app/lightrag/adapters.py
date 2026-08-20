from __future__ import annotations

import asyncio
from functools import partial
import logging
from typing import Any, TYPE_CHECKING

import httpx
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import APIConnectionError, APITimeoutError

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)


async def _collect_completion_text(response: Any) -> str:
    """Collect LightRAG's streaming helper into the text its pipeline expects."""
    if isinstance(response, str):
        text = response
    elif hasattr(response, "__aiter__"):
        parts: list[str] = []
        async for part in response:
            if not isinstance(part, str):
                raise RuntimeError(
                    "lightrag_stream_returned_non_text_chunk:"
                    f"{type(part).__name__}"
                )
            parts.append(part)
        text = "".join(parts)
    else:
        raise RuntimeError(
            "lightrag_stream_returned_invalid_completion:"
            f"{type(response).__name__}"
        )
    if not text.strip():
        raise RuntimeError("lightrag_stream_returned_empty_completion")
    return text


def build_router_llm_complete(settings: Settings):
    """LLM for LightRAG via 9router; provider/quota errors are not model-switched."""

    async def router_llm_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        entity_extraction=False,
        **kwargs,
    ):
        history_messages = history_messages or []
        entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
        model_name = settings.lightrag_llm_model_chain[0]
        # LightRAG's OpenAI helper is decorated with three same-model retries
        # for quota/timeouts. Foreground retrieval must fail promptly so the
        # agent can surface the provider outage; call the original
        # implementation once while preserving its request/structured-output
        # semantics. This is deliberately not a model fallback.
        complete_once = getattr(
            openai_complete_if_cache,
            "__wrapped__",
            openai_complete_if_cache,
        )
        client_configs = dict(kwargs.pop("openai_client_configs", {}) or {})
        # Disabling the outer Tenacity wrapper is not sufficient: the OpenAI
        # SDK itself retries 429/5xx by default. A quota error must surface on
        # the first failed request so the batch can stop and retain progress.
        client_configs["max_retries"] = 0
        # 9router's non-stream endpoint internally converts the upstream Codex
        # stream to one JSON response and can fail long extractions with
        # "Failed to convert streaming response to JSON (reset after 30s)".
        # Consume the native stream ourselves and return the same final string
        # LightRAG expects. This also provides activity while Sol generates a
        # large entity/relation response.
        kwargs.pop("stream", None)
        for attempt in range(settings.lightrag_llm_timeout_retries + 1):
            try:
                response = await complete_once(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    keyword_extraction=keyword_extraction,
                    entity_extraction=entity_extraction,
                    base_url=settings.lightrag_llm_api_base,
                    api_key=settings.lightrag_llm_api_key or "any",
                    timeout=int(settings.lightrag_llm_timeout_seconds),
                    openai_client_configs=client_configs,
                    stream=True,
                    **kwargs,
                )
                return await _collect_completion_text(response)
            except (
                APITimeoutError,
                APIConnectionError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ):
                if attempt >= settings.lightrag_llm_timeout_retries:
                    raise
                logger.warning(
                    "LightRAG transient request failure; retrying the same model "
                    "%s (%s/%s)",
                    model_name,
                    attempt + 1,
                    settings.lightrag_llm_timeout_retries,
                )
                await asyncio.sleep(min(2**attempt, 5))

        raise RuntimeError("unreachable_lightrag_retry_state")

    return router_llm_complete


def build_ollama_embedding_func(settings: Settings) -> EmbeddingFunc:
    """Build the configured asymmetric Ollama embedder for LightRAG."""

    embed_fn = partial(
        ollama_embed.func,
        embed_model=settings.embedding_model,
        host=settings.ollama_host,
        query_prefix=settings.embedding_query_prefix or None,
        document_prefix=settings.embedding_document_prefix or None,
    )
    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.embedding_max_token_size,
        model_name=settings.embedding_model,
        supports_asymmetric=True,
        func=embed_fn,
    )
