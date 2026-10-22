import asyncio

import pytest

from app.llm import openai_client
from app.llm.ollama_client import OllamaError, StreamChunk
from app.llm.openai_client import OpenAICompatibleClient


def _client():
    return OpenAICompatibleClient(base_url="http://127.0.0.1:20128/v1", api_key="any", timeout_seconds=5)


def _run(client, model="cx/gpt-5.6-sol"):
    async def consume():
        return [c async for c in client.stream_chat(model=model, messages=[], temperature=0.0)]
    return asyncio.run(consume())


def test_quota_error_retries_same_model_then_succeeds(monkeypatch):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0.0)
    calls = []

    async def fake(self, **kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise OllamaError("LLM stream failed (503): [codex/gpt-5.6-sol] [429]: The usage limit has been reached")
        yield StreamChunk(content="ok", done=False)
        yield StreamChunk(content="", done=True, finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "_stream_chat_once", fake)
    chunks = _run(_client())
    assert [c.content for c in chunks if c.content] == ["ok"]
    assert calls == ["cx/gpt-5.6-sol", "cx/gpt-5.6-sol"]  # same model only


def test_non_quota_error_is_not_retried(monkeypatch):
    calls = []

    async def fake(self, **kwargs):
        calls.append(1)
        raise OllamaError("invalid api key")
        yield  # pragma: no cover

    monkeypatch.setattr(OpenAICompatibleClient, "_stream_chat_once", fake)
    with pytest.raises(OllamaError):
        _run(_client())
    assert len(calls) == 1


def test_quota_error_after_content_is_never_replayed(monkeypatch):
    calls = []

    async def fake(self, **kwargs):
        calls.append(1)
        yield StreamChunk(content="partial", done=False)
        raise OllamaError("usage limit reached")

    monkeypatch.setattr(OpenAICompatibleClient, "_stream_chat_once", fake)
    with pytest.raises(OllamaError):
        _run(_client())
    assert len(calls) == 1


def test_quota_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0.0)
    calls = []

    async def fake(self, **kwargs):
        calls.append(1)
        raise OllamaError("usage limit reached")
        yield  # pragma: no cover

    monkeypatch.setattr(OpenAICompatibleClient, "_stream_chat_once", fake)
    with pytest.raises(OllamaError):
        _run(_client())
    assert len(calls) == openai_client._QUOTA_RETRIES + 1


def test_nonstream_empty_completion_is_retried_once_on_the_same_model(monkeypatch):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0.0)
    calls = []

    async def fake(self, **kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise OllamaError("LLM request returned an empty completion")
        return openai_client.ChatCompletion(model=kwargs["model"], message="ok")

    monkeypatch.setattr(OpenAICompatibleClient, "_chat_once", fake)
    result = asyncio.run(_client().chat(model="cx/gpt-6.1-sol", messages=[], temperature=0.0))
    assert result.message == "ok"
    assert calls == ["cx/gpt-6.1-sol", "cx/gpt-6.1-sol"]


def test_tool_turn_retries_a_stalled_gateway_once(monkeypatch):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0.0)
    calls = []

    async def fake(self, **kwargs):
        calls.append(kwargs["timeout_seconds"])
        if len(calls) == 1:
            raise OllamaError("LLM request failed: ReadTimeout: ")
        return openai_client.ToolTurn(model=kwargs["model"], content="ok")

    monkeypatch.setattr(OpenAICompatibleClient, "_tool_turn_once", fake)
    result = asyncio.run(_client().chat_with_tools(model="cx/gpt-6.1-sol", messages=[], tools=[]))
    assert result.content == "ok" and calls == [90.0, 90.0]
