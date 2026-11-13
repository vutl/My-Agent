"""Offline priority, cancellation and quota scheduling contracts."""
import asyncio
import time

import httpx
import pytest

from app.llm import openai_client
from app.llm.scheduler import LLMScheduler, ResearchDeadline, call_with_backoff


def test_background_concurrency_and_interactive_never_waits():
    async def main():
        scheduler = LLMScheduler(2)
        active = peak = 0
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            nonlocal active, peak
            async with scheduler.background():
                active += 1
                peak = max(peak, active)
                if active == 2:
                    entered.set()
                await release.wait()
                active -= 1

        jobs = [asyncio.create_task(work()) for _ in range(6)]
        await entered.wait()
        # Both permits are occupied, but chat enters immediately.
        async with scheduler.interactive():
            assert active == 2
        release.set()
        await asyncio.gather(*jobs)
        assert peak == 2
    asyncio.run(asyncio.wait_for(main(), 2))


@pytest.mark.parametrize("release_chat", [True, False])
def test_background_waits_for_chat_or_timeout(release_chat):
    async def main():
        scheduler = LLMScheduler(1, interactive_wait=0.025)
        started = asyncio.Event()

        async def background():
            async with scheduler.background():
                started.set()

        async with scheduler.interactive():
            job = asyncio.create_task(background())
            await asyncio.sleep(0)
            assert not started.is_set()
            if not release_chat:
                await asyncio.wait_for(started.wait(), 0.2)
        await job
        assert started.is_set() and scheduler._interactive == 0
    asyncio.run(main())


def test_cancelled_interactive_releases_counter_and_background_slot():
    async def main():
        scheduler = LLMScheduler(1)
        entered = asyncio.Event()
        async def chat():
            async with scheduler.interactive():
                entered.set()
                await asyncio.Event().wait()
        job = asyncio.create_task(chat())
        await entered.wait()
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        async with scheduler.background():
            assert scheduler._interactive == 0
    asyncio.run(asyncio.wait_for(main(), 1))


def test_quota_backoff_and_deadline():
    async def main():
        clock = [0.0]
        sleeps, events = [], []
        calls = 0
        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("HTTP 429 rate limit secret-vendor-key")
            return "success"
        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay
        async def emit(event):
            events.append(event)
        result = await call_with_backoff(fn, emit=emit, deadline=100, sleep=sleep, clock=lambda: clock[0])
        assert result == "success" and sleeps == [30, 60]
        assert len(events) == 2 and all(e["type"] == "waiting_quota" for e in events)
        assert "secret" not in str(events)
        async def quota():
            raise RuntimeError("HTTP 429")
        with pytest.raises(ResearchDeadline):
            await call_with_backoff(quota, emit=emit, deadline=101, sleep=sleep, clock=lambda: clock[0])
        assert sleeps[-1] == 11
    asyncio.run(main())


def test_non_quota_errors_are_not_retried():
    async def main():
        async def fail():
            raise ValueError("invalid JSON")
        async def emit(event):
            pytest.fail("unexpected quota event")
        with pytest.raises(ValueError):
            await call_with_backoff(fail, emit=emit, deadline=time.monotonic() + 1)
    asyncio.run(main())


def test_real_client_http_429_backoff_then_success(monkeypatch):
    calls, events, delays = [], [], []
    def respond(request):
        calls.append(request.url.path)
        if len(calls) <= 2:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, json={"choices": [{"message": {"content": "success"}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(respond)))
    # Exercise research backoff after the client's ordinary retries are exhausted.
    monkeypatch.setattr(openai_client, "_QUOTA_RETRIES", 0)
    client = openai_client.OpenAICompatibleClient("https://gateway.example/v1", "private-test-key")
    async def main():
        async def call():
            return await client.chat(model="cx/gpt-6.1-sol", messages=[], temperature=0)
        async def emit(event):
            events.append(event)
        async def sleep(delay):
            delays.append(delay)
        result = await call_with_backoff(call, emit=emit, deadline=time.monotonic() + 100, sleep=sleep)
        assert result.message == "success"
    asyncio.run(main())
    assert len(calls) == 3 and delays == [30, 60]
    assert len(events) == 2 and all(e["type"] == "waiting_quota" for e in events)
    assert "private-test-key" not in str(events)


@pytest.mark.parametrize("wrapped", [False, True])
def test_quota_recognises_http_status_and_wrapped_client_format(wrapped):
    error = httpx.HTTPStatusError("Request refused", request=httpx.Request("POST", "https://gateway.example"),
                                  response=httpx.Response(429))
    if wrapped:
        error = RuntimeError("LLM request failed (429): Too Many Requests")
    assert openai_client._is_quota_error(error)
