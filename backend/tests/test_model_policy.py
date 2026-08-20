import asyncio

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from app.api import agent as agent_api
from app.api import lightrag_routes
from app.core.config import (
    Settings,
    get_settings,
    normalize_loopback_url,
    validate_runtime_model_policy,
)
from app.lightrag import adapters as lightrag_adapters
from app.lightrag import client as lightrag_client
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAICompatibleClient, get_llm_client


def test_lightrag_never_silently_switches_from_approved_gpt_model() -> None:
    settings = Settings(
        lightrag_llm_model="cx/gpt-5.6-terra",
        lightrag_llm_fallback_models=["cx/gpt-5.6-luna", "cu/another-model"],
    )

    assert settings.lightrag_llm_model_chain == ["cx/gpt-5.6-terra"]


def test_lightrag_client_applies_configured_sol_concurrency(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict = {}

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def initialize_storages(self):
            return None

        async def finalize_storages(self):
            return None

    monkeypatch.setattr(lightrag_client, "LightRAG", FakeLightRAG)
    monkeypatch.setattr(
        lightrag_client,
        "build_router_llm_complete",
        lambda _settings: "llm",
    )
    monkeypatch.setattr(
        lightrag_client,
        "build_ollama_embedding_func",
        lambda _settings: "embedding",
    )

    settings = Settings(
        data_dir=tmp_path,
        lightrag_llm_timeout_seconds=300,
        lightrag_llm_timeout_retries=1,
        lightrag_llm_max_async=1,
        lightrag_chunk_token_size=600,
        lightrag_chunk_overlap_token_size=80,
    )
    asyncio.run(lightrag_client.init_lightrag(settings))
    asyncio.run(lightrag_client.shutdown_lightrag())

    assert captured["llm_model_name"] == "cx/gpt-5.6-sol"
    assert captured["llm_model_max_async"] == 1
    assert captured["role_llm_configs"] == {"extract": {"timeout": 660}}
    assert captured["chunk_token_size"] == 600
    assert captured["chunk_overlap_token_size"] == 80


def test_lightrag_embedding_uses_configured_model_shape_and_prefixes() -> None:
    settings = Settings(
        embedding_model="embeddinggemma:300m",
        embedding_dim=768,
        embedding_max_token_size=2048,
        embedding_query_prefix="task: search result | query: ",
        embedding_document_prefix="title: none | text: ",
    )

    embedding = lightrag_adapters.build_ollama_embedding_func(settings)

    assert embedding.model_name == "embeddinggemma:300m"
    assert embedding.embedding_dim == 768
    assert embedding.max_token_size == 2048
    assert embedding.supports_asymmetric is True
    assert embedding.func.keywords["query_prefix"] == (
        "task: search result | query: "
    )
    assert embedding.func.keywords["document_prefix"] == (
        "title: none | text: "
    )


def test_invalid_lightrag_model_fails_closed_to_approved_default() -> None:
    settings = Settings(lightrag_llm_model="local/ollama-model")

    with pytest.raises(ValueError, match="approved cx/gpt-5.6 model"):
        _ = settings.lightrag_llm_model_chain


def test_llm_client_factory_never_treats_unknown_provider_as_ollama() -> None:
    with pytest.raises(ValueError, match="refusing to silently switch"):
        get_llm_client(
            provider="openai-compatbile",
            ollama_host="http://localhost:11434",
            openai_api_base="http://localhost:20128/v1",
            openai_api_key="any",
            timeout_seconds=1,
        )


def test_local_sidecar_urls_use_explicit_ipv4_without_rewriting_remote_hosts(
    monkeypatch,
) -> None:
    assert normalize_loopback_url("http://localhost:20128/v1/") == (
        "http://127.0.0.1:20128/v1"
    )
    assert normalize_loopback_url("https://router.example/v1/") == (
        "https://router.example/v1"
    )

    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:20128/v1")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("LIGHTRAG_LLM_API_BASE", "http://localhost:20128/v1")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.openai_api_base == "http://127.0.0.1:20128/v1"
    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.lightrag_llm_api_base == "http://127.0.0.1:20128/v1"
    get_settings.cache_clear()


def test_llm_client_factory_uses_only_explicit_provider() -> None:
    openai_client = get_llm_client(
        provider="openai_compatible",
        ollama_host="http://localhost:11434",
        openai_api_base="http://localhost:20128/v1",
        openai_api_key="any",
        timeout_seconds=1,
    )
    ollama_client = get_llm_client(
        provider="ollama",
        ollama_host="http://localhost:11434",
        openai_api_base="http://localhost:20128/v1",
        openai_api_key="any",
        timeout_seconds=1,
    )

    assert isinstance(openai_client, OpenAICompatibleClient)
    assert isinstance(ollama_client, OllamaClient)


def test_runtime_policy_rejects_role_drift_and_fallback_configuration() -> None:
    with pytest.raises(RuntimeError, match="ROUTER_MODEL.*FALLBACK_MODELS"):
        validate_runtime_model_policy(
            Settings(
                router_model="cx/gpt-5.5",
                lightrag_llm_fallback_models=["cu/another-model"],
            )
        )


def test_runtime_policy_accepts_fixed_9router_roles() -> None:
    validate_runtime_model_policy(Settings())


def test_runtime_policy_rejects_unknown_retrieval_engine() -> None:
    with pytest.raises(RuntimeError, match="RETRIEVAL_ENGINE"):
        validate_runtime_model_policy(Settings(retrieval_engine="mystery"))


def test_runtime_policy_requires_local_cross_encoder_only_when_enabled() -> None:
    validate_runtime_model_policy(
        Settings(
            rerank_enabled=False,
            rerank_mode="cross_encoder",
            rerank_cross_encoder_path=None,
        )
    )

    with pytest.raises(RuntimeError, match="RERANK_CROSS_ENCODER_PATH"):
        validate_runtime_model_policy(
            Settings(
                rerank_enabled=True,
                rerank_mode="cross_encoder",
                rerank_cross_encoder_path=None,
            )
        )


def test_lightrag_ingest_preflight_rejects_unavailable_gateway(
    monkeypatch,
) -> None:
    class UnavailableClient:
        def __init__(self, **_kwargs):
            pass

        async def health(self):
            return {
                "reachable": False,
                "error": "connection refused",
                "models": [],
            }

    monkeypatch.setattr(
        lightrag_routes,
        "OpenAICompatibleClient",
        UnavailableClient,
    )

    with pytest.raises(
        lightrag_routes.HTTPException,
        match="gateway unavailable",
    ):
        asyncio.run(lightrag_routes._require_lightrag_gateway(Settings()))


def test_lightrag_ingest_preflight_requires_exact_selected_model(
    monkeypatch,
) -> None:
    class WrongModelClient:
        def __init__(self, **_kwargs):
            pass

        async def health(self):
            return {
                "reachable": True,
                "models": ["cx/gpt-5.6-terra"],
            }

    monkeypatch.setattr(
        lightrag_routes,
        "OpenAICompatibleClient",
        WrongModelClient,
    )

    with pytest.raises(
        lightrag_routes.HTTPException,
        match="model unavailable: cx/gpt-5.6-sol",
    ):
        asyncio.run(lightrag_routes._require_lightrag_gateway(Settings()))


@pytest.mark.parametrize(
    "model",
    ["cx/gpt-5.6-sol", "cx/gpt-5.6-terra", "cx/gpt-5.6-luna"],
)
def test_runtime_policy_accepts_each_visible_9router_model(model: str) -> None:
    validate_runtime_model_policy(
        Settings(
            default_model=model,
            router_model=model,
            vision_model=model,
            lightrag_llm_model=model,
        )
    )


def test_lightrag_quota_error_propagates_without_trying_another_model(
    monkeypatch,
) -> None:
    requested_models: list[str] = []
    request = httpx.Request(
        "POST",
        "http://localhost:20128/v1/chat/completions",
    )
    response = httpx.Response(429, request=request)

    async def fail_with_quota(model: str, *_args, **_kwargs):
        requested_models.append(model)
        raise RateLimitError(
            "usage limit reached",
            response=response,
            body={"error": {"message": "usage limit reached"}},
        )

    monkeypatch.setattr(
        lightrag_adapters,
        "openai_complete_if_cache",
        fail_with_quota,
    )
    complete = lightrag_adapters.build_router_llm_complete(Settings())

    with pytest.raises(RateLimitError, match="usage limit reached"):
        asyncio.run(complete("extract keywords"))

    assert requested_models == ["cx/gpt-5.6-sol"]


def test_lightrag_timeout_retries_once_with_the_exact_same_model(
    monkeypatch,
) -> None:
    requested_models: list[str] = []
    request = httpx.Request(
        "POST",
        "http://localhost:20128/v1/chat/completions",
    )

    async def timeout_then_succeed(model: str, *_args, **_kwargs):
        requested_models.append(model)
        if len(requested_models) == 1:
            raise APITimeoutError(request=request)
        return "grounded extraction"

    monkeypatch.setattr(
        lightrag_adapters,
        "openai_complete_if_cache",
        timeout_then_succeed,
    )
    complete = lightrag_adapters.build_router_llm_complete(
        Settings(lightrag_llm_timeout_retries=1)
    )

    assert asyncio.run(complete("extract entities")) == "grounded extraction"
    assert requested_models == ["cx/gpt-5.6-sol", "cx/gpt-5.6-sol"]


def test_lightrag_stream_read_timeout_retries_once_with_the_exact_same_model(
    monkeypatch,
) -> None:
    requested_models: list[str] = []
    request = httpx.Request(
        "POST",
        "http://localhost:20128/v1/chat/completions",
    )

    async def stream_timeout_then_succeed(model: str, *_args, **_kwargs):
        requested_models.append(model)

        async def pieces():
            if len(requested_models) == 1:
                raise httpx.ReadTimeout("stream stalled", request=request)
            yield "grounded extraction"

        return pieces()

    monkeypatch.setattr(
        lightrag_adapters,
        "openai_complete_if_cache",
        stream_timeout_then_succeed,
    )
    complete = lightrag_adapters.build_router_llm_complete(
        Settings(lightrag_llm_timeout_retries=1)
    )

    assert asyncio.run(complete("extract entities")) == "grounded extraction"
    assert requested_models == ["cx/gpt-5.6-sol", "cx/gpt-5.6-sol"]


def test_lightrag_collects_native_stream_into_pipeline_text(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    async def stream_completion(_model: str, *_args, **kwargs):
        calls.append(kwargs)

        async def pieces():
            yield "grounded "
            yield "extraction"

        return pieces()

    monkeypatch.setattr(
        lightrag_adapters,
        "openai_complete_if_cache",
        stream_completion,
    )
    complete = lightrag_adapters.build_router_llm_complete(Settings())

    result = asyncio.run(complete("extract entities"))

    assert result == "grounded extraction"
    assert calls[0]["stream"] is True
    assert calls[0]["openai_client_configs"]["max_retries"] == 0


def test_lightrag_bypasses_library_retry_wrapper_but_preserves_call_shape(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, dict]] = []

    async def complete_once(model: str, prompt: str, **kwargs):
        calls.append((model, prompt, kwargs))
        return '{"high_level_keywords":["ASPIRE"]}'

    async def retry_wrapper(*_args, **_kwargs):
        raise AssertionError("the retry wrapper must not run")

    retry_wrapper.__wrapped__ = complete_once
    monkeypatch.setattr(
        lightrag_adapters,
        "openai_complete_if_cache",
        retry_wrapper,
    )
    complete = lightrag_adapters.build_router_llm_complete(Settings())

    result = asyncio.run(
        complete(
            "extract keywords",
            system_prompt="return JSON",
            history_messages=[{"role": "user", "content": "prior"}],
            keyword_extraction=True,
        )
    )

    assert result == '{"high_level_keywords":["ASPIRE"]}'
    assert len(calls) == 1
    model, prompt, kwargs = calls[0]
    assert model == "cx/gpt-5.6-sol"
    assert prompt == "extract keywords"
    assert kwargs["system_prompt"] == "return JSON"
    assert kwargs["history_messages"] == [{"role": "user", "content": "prior"}]
    assert kwargs["keyword_extraction"] is True
    assert kwargs["stream"] is True
    assert kwargs["openai_client_configs"]["max_retries"] == 0


def test_agent_retrieval_does_not_hide_lightrag_provider_quota(
    monkeypatch,
) -> None:
    request = httpx.Request(
        "POST",
        "http://localhost:20128/v1/chat/completions",
    )
    response = httpx.Response(429, request=request)

    class QuotaBridge:
        def __init__(self, _settings) -> None:
            pass

        async def retrieve(self, *_args, **_kwargs):
            raise RateLimitError(
                "usage limit reached",
                response=response,
                body={"error": {"message": "usage limit reached"}},
            )

    monkeypatch.setattr(agent_api, "LightRAGBridge", QuotaBridge)

    with pytest.raises(RateLimitError, match="usage limit reached"):
        asyncio.run(
            agent_api._retrieve_for_agent(
                rag=object(),
                settings=Settings(
                    lightrag_enabled=True,
                    retrieval_engine="lightrag",
                ),
                query="ASPIRE benchmark",
                collection_id=None,
                retrieval_mode="auto",
            )
        )


def test_agent_retrieval_only_falls_back_for_uninitialized_lightrag(
    monkeypatch,
) -> None:
    class BrokenBridge:
        def __init__(self, _settings) -> None:
            pass

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError("provider extraction failed")

    monkeypatch.setattr(agent_api, "LightRAGBridge", BrokenBridge)

    with pytest.raises(RuntimeError, match="provider extraction failed"):
        asyncio.run(
            agent_api._retrieve_for_agent(
                rag=object(),
                settings=Settings(
                    lightrag_enabled=True,
                    retrieval_engine="lightrag",
                ),
                query="cross paper relation",
                collection_id=None,
                retrieval_mode="auto",
            )
        )
