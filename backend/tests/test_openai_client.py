from __future__ import annotations

import asyncio

import httpx
import pytest

from app.llm.ollama_client import OllamaError
from app.llm.openai_client import OpenAICompatibleClient, _assert_reported_model


def test_stream_error_body_is_read_before_building_diagnostic(monkeypatch) -> None:
    request = httpx.Request("POST", "http://localhost:20128/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        stream=httpx.ByteStream(
            b'{"error":{"message":"usage limit reached; reset after 25m"}}'
        ),
    )

    class StreamContext:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *_args) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        def stream(self, *_args, **_kwargs):
            return StreamContext()

    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        base_url="http://localhost:20128/v1",
        api_key="any",
    )

    async def consume() -> None:
        async for _chunk in client.stream_chat(
            model="cx/gpt-5.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
        ):
            pass

    with pytest.raises(OllamaError, match=r"429.*usage limit reached"):
        asyncio.run(consume())


def test_gateway_reported_model_must_match_requested_model() -> None:
    _assert_reported_model(requested="cx/gpt-5.5", reported="cx/gpt-5.5")
    # 9router accepts a namespaced route but reports the same upstream model
    # without the route prefix in completion envelopes.
    _assert_reported_model(requested="cx/gpt-5.5", reported="gpt-5.5")
    _assert_reported_model(requested="cx/gpt-5.5", reported=None)

    with pytest.raises(OllamaError, match=r"empty model identifier"):
        _assert_reported_model(requested="cx/gpt-5.5", reported="")

    with pytest.raises(OllamaError, match=r"model mismatch.*cx/gpt-5.4"):
        _assert_reported_model(
            requested="cx/gpt-5.5",
            reported="cx/gpt-5.4",
        )


def test_nonstream_invalid_envelope_fails_closed(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            raise ValueError("broken json")

    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        base_url="http://localhost:20128/v1",
        api_key="any",
    )

    with pytest.raises(OllamaError, match="invalid JSON envelope"):
        asyncio.run(
            client.chat(
                model="cx/gpt-5.5",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.2,
            )
        )


def test_nonstream_empty_completion_fails_closed(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "cx/gpt-5.5",
                "choices": [{"message": {"content": ""}}],
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        base_url="http://localhost:20128/v1",
        api_key="any",
    )

    with pytest.raises(OllamaError, match="empty completion"):
        asyncio.run(
            client.chat(
                model="cx/gpt-5.5",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.2,
            )
        )


def test_nonstream_optional_json_response_format_is_forwarded(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "cx/gpt-5.6-sol",
                "choices": [{"message": {"content": '{"ok":true}'}}],
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **kwargs):
            captured.update(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        base_url="http://localhost:20128/v1",
        api_key="any",
    )
    completion = asyncio.run(
        client.chat(
            model="cx/gpt-5.6-sol",
            messages=[{"role": "user", "content": "return json"}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    )

    assert completion.message == '{"ok":true}'
    assert captured["response_format"] == {"type": "json_object"}


def test_stream_malformed_json_fails_instead_of_accepting_partial_answer(
    monkeypatch,
) -> None:
    request = httpx.Request("POST", "http://localhost:20128/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        stream=httpx.ByteStream(
            b": heartbeat\n"
            b"event: message\n"
            b"id: 42\n"
            b'data: {"model":"cx/gpt-5.5","choices":[{"delta":{"content":"partial"}}]}\n'
            b"data: {broken-json}\n"
        ),
    )

    class StreamContext:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *_args) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        def stream(self, *_args, **_kwargs):
            return StreamContext()

    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeAsyncClient)
    client = OpenAICompatibleClient(
        base_url="http://localhost:20128/v1",
        api_key="any",
    )

    async def consume() -> list[str]:
        chunks: list[str] = []
        async for chunk in client.stream_chat(
            model="cx/gpt-5.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
        ):
            chunks.append(chunk.content)
        return chunks

    with pytest.raises(OllamaError, match="malformed JSON"):
        asyncio.run(consume())
