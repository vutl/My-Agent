"""OpenAI-compatible tool SSE assembly and retry boundaries, without a gateway."""

import asyncio
import json

import httpx
import pytest

from app.llm import openai_client
from app.llm.ollama_client import OllamaError
from app.llm.openai_client import OpenAICompatibleClient, ToolCall, ToolTurn

MODEL = "cx/gpt-6.1-sol"


async def collect(events):
    return [event async for event in events]


def client():
    return OpenAICompatibleClient("http://localhost:9/v1", "test")


def install_stream(monkeypatch, frames):
    original = httpx.AsyncClient
    requests, consumed = [], []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for frame in frames:
                consumed.append(frame)
                if isinstance(frame, Exception):
                    raise frame
                line = frame if isinstance(frame, str) else "data: " + json.dumps(frame)
                yield (line + "\n\n").encode()

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=Stream())

    monkeypatch.setattr(openai_client.httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(respond), **kwargs))
    return requests, consumed


def frame(delta, finish=None, **extra):
    return {"model": "gpt-6.1-sol", "choices": [{"delta": delta, "finish_reason": finish}], **extra}


def test_stream_yields_before_completion_and_assembles_interleaved_calls(monkeypatch):
    frames = [
        ": heartbeat", "event: message", frame({"role": "assistant"}),
        frame({"content": "Checking "}), frame({"content": "now. "}),
        frame({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "act", "arguments": '{"p":'}}]}),
        frame({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "read", "arguments": "{}"}}]}),
        frame({"tool_calls": [{"index": 1, "function": {"arguments": '1} junk'}}]}, "tool_calls"),
        "data: [DONE]",
    ]
    requests, consumed = install_stream(monkeypatch, frames)

    async def run():
        stream = client().chat_with_tools_stream(model=MODEL, messages=[], tools=[{"type": "function"}])
        first = await anext(stream)
        assert first == ("text", "Checking ") and len(consumed) == 4
        return [first, *await collect(stream)]

    events = asyncio.run(run())
    assert [data for name, data in events if name == "text"] == ["Checking ", "now. "]
    assert [(d["index"], d["id"]) for n, d in events if n == "tool_call_start"] == [(1, "b"), (0, "a")]
    assert [(d["index"], d["delta"]) for n, d in events if n == "tool_call_args"] == [(1, '{"p":'), (0, '{}'), (1, '1} junk')]
    assert events[-1] == ("turn", ToolTurn("gpt-6.1-sol", "Checking now. ", (
        ToolCall("a", "read", "{}"), ToolCall("b", "act", '{"p":1} junk'),
    )))
    assert requests[0]["stream"] is True and requests[0]["model"] == MODEL
    assert requests[0]["tool_choice"] == "auto"


def test_stream_and_whole_turn_use_same_defaults(monkeypatch):
    requests, _ = install_stream(monkeypatch, [
        frame({"tool_calls": [{"index": 0, "function": {"name": "re"}}]}),
        frame({"tool_calls": [{"index": 0, "function": {"name": "ad"}}]}, "tool_calls"),
    ])
    llm = client()
    events = asyncio.run(collect(llm.chat_with_tools_stream(model=MODEL, messages=[], tools=[])))
    expected = llm._parse_tool_turn(MODEL, {"model": "gpt-6.1-sol", "choices": [{"message": {
        "tool_calls": [{"function": {"name": "read"}}],
    }}]})
    assert events[-1] == ("turn", expected)
    assert ("tool_call_start", {"index": 0, "id": "call_0", "name": "read"}) in events
    assert "tools" not in requests[0] and "tool_choice" not in requests[0]


@pytest.mark.parametrize("error", ["usage limit reached", "ReadTimeout: stalled", "RemoteProtocolError: closed",
                                  "ConnectError: offline", "empty completion"])
@pytest.mark.parametrize("first", [None, ("text", "partial"),
                                      ("tool_call_start", {"index": 0, "id": "a", "name": "read"}),
                                      ("tool_call_args", {"index": 0, "delta": "{"})])
def test_retry_only_before_any_public_delta(monkeypatch, error, first):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0)
    requests = []

    async def once(self, **kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            if first:
                yield first
            raise OllamaError(error)
        yield "text", "ok"
        yield "turn", ToolTurn(MODEL, "ok")

    monkeypatch.setattr(OpenAICompatibleClient, "_tool_turn_stream_once", once)
    events = client().chat_with_tools_stream(model=MODEL, messages=[], tools=[])
    if first:
        with pytest.raises(OllamaError, match=error):
            asyncio.run(collect(events))
        assert len(requests) == 1
    else:
        assert asyncio.run(collect(events))[-1] == ("turn", ToolTurn(MODEL, "ok"))
        assert len(requests) == 2
    assert all(r["model"] == MODEL and r["timeout_seconds"] == 90 for r in requests)


def test_whitespace_empty_turn_is_invisible_and_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(openai_client, "_QUOTA_BACKOFF_SECONDS", 0)
    requests, _ = install_stream(monkeypatch, [frame({"content": " \n"}), "data: [DONE]"])
    received = []

    async def run():
        async for event in client().chat_with_tools_stream(model=MODEL, messages=[], tools=[]):
            received.append(event)

    with pytest.raises(OllamaError, match="empty completion"):
        asyncio.run(run())
    assert received == [] and len(requests) == openai_client._QUOTA_RETRIES + 1


def test_leading_whitespace_and_final_content_are_preserved(monkeypatch):
    install_stream(monkeypatch, [frame({"content": " \n"}), frame({"content": "Hello"}),
                                 frame({"content": "! "}, "stop")])
    events = asyncio.run(collect(client().chat_with_tools_stream(model=MODEL, messages=[], tools=[])))
    assert [d for n, d in events if n == "text"] == [" \nHello", "! "]
    assert events[-1][1].content == " \nHello! "


@pytest.mark.parametrize("bad, error", [
    ("data: broken", "malformed JSON"), ("data: []", "non-object"),
    ({"error": {"message": "invalid api key"}}, "invalid api key"),
    (frame({"content": "wrong"}, model="different-model"), "model mismatch"),
    ({"choices": "invalid"}, "invalid completion choice"),
    (frame([]), "invalid completion delta"), (frame({"content": ["bad"]}), "non-text completion"),
    (frame({"tool_calls": [{"function": {"name": "read"}}]}), "tool call index"),
    (frame({"tool_calls": [{"index": 0, "function": {"arguments": {}}}]}), "non-text tool call"),
])
def test_invalid_stream_is_rejected_without_retry(monkeypatch, bad, error):
    requests, _ = install_stream(monkeypatch, [bad])
    with pytest.raises(OllamaError, match=error):
        asyncio.run(collect(client().chat_with_tools_stream(model=MODEL, messages=[], tools=[])))
    assert len(requests) == 1


def test_stream_http_error_includes_body_and_transport_failure_after_text_is_not_retried(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(openai_client.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(401, text="invalid api key")), **kwargs))
    with pytest.raises(OllamaError, match="401.*invalid api key"):
        asyncio.run(collect(client().chat_with_tools_stream(model=MODEL, messages=[], tools=[])))
    monkeypatch.undo()
    requests, _ = install_stream(monkeypatch, [frame({"content": "partial"}), httpx.ReadTimeout("stalled")])
    with pytest.raises(OllamaError, match="ReadTimeout"):
        asyncio.run(collect(client().chat_with_tools_stream(model=MODEL, messages=[], tools=[])))
    assert len(requests) == 1


def test_heartbeats_alone_cannot_hold_a_stalled_turn_open(monkeypatch):
    original = httpx.AsyncClient

    class Heartbeats(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield ("data: " + json.dumps(frame({"role": "assistant"})) + "\n\n").encode()
            for _ in range(200):  # ~10 s of keep-alives, no data
                await asyncio.sleep(0.05)
                yield b": keep-alive\n\n"

    monkeypatch.setattr(openai_client.httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Heartbeats())), **kwargs))
    monkeypatch.setattr(openai_client, "_QUOTA_RETRIES", 0)
    with pytest.raises(OllamaError, match="Timeout"):
        asyncio.run(asyncio.wait_for(collect(client().chat_with_tools_stream(
            model=MODEL, messages=[], tools=[], timeout_seconds=0.3,
        )), timeout=5))
