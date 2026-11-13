"""Chat never waits; background model calls yield to active chat turns."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import time
from weakref import WeakKeyDictionary

from app.llm.openai_client import _is_quota_error


class LLMScheduler:
    def __init__(self, max_concurrency=None, interactive_wait=20.0):
        self.max_concurrency = max_concurrency
        self.interactive_wait = interactive_wait
        self._loops = WeakKeyDictionary()
        self._interactive = 0

    def _state(self):
        loop = asyncio.get_running_loop()
        if loop not in self._loops:
            from app.core.config import get_settings
            idle = asyncio.Event()
            if not self._interactive:
                idle.set()
            self._loops[loop] = (asyncio.Semaphore(self.max_concurrency or get_settings().research_max_concurrency), idle)
        return self._loops[loop]

    @asynccontextmanager
    async def interactive(self):
        self._state()
        self._interactive += 1
        for _, idle in self._loops.values():
            idle.clear()
        try:
            yield
        finally:
            self._interactive -= 1
            if not self._interactive:
                for loop, (_, idle) in list(self._loops.items()):
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(lambda event=idle: event.set() if not self._interactive else None)

    @asynccontextmanager
    async def background(self):
        semaphore, idle = self._state()
        async with semaphore:
            if self._interactive:
                try:
                    await asyncio.wait_for(idle.wait(), timeout=self.interactive_wait)
                except TimeoutError:
                    pass
            yield


LLM_SCHEDULER = LLMScheduler()


class ResearchDeadline(Exception):
    """The research wall-time budget has expired."""


class ResearchHalted(Exception):
    """A research step must stop without treating it as a provider failure."""


class ResearchFinished(ResearchHalted):
    """The user requested synthesis of the evidence collected so far."""


async def call_with_backoff(fn, *, emit, deadline, sleep=asyncio.sleep, clock=time.monotonic,
                            check=None, wait_for_stop=None):
    """Retry quota failures without holding a background slot during backoff.

    fn acquires the scheduler for each attempt. Exception details never enter events.
    """
    attempt = 0
    while True:
        if check:
            check()
        remaining = deadline - clock()
        if remaining <= 0:
            raise ResearchDeadline
        try:
            return await asyncio.wait_for(fn(), timeout=remaining)
        except TimeoutError as exc:
            raise ResearchDeadline from exc
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            delay = min((30, 60, 120, 240, 600)[min(attempt, 4)], max(0, deadline - clock()))
            attempt += 1
            await emit({"type": "waiting_quota", "until": (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()})
            if wait_for_stop is None:
                await sleep(delay)
                continue
            jobs = [asyncio.create_task(sleep(delay)), asyncio.create_task(wait_for_stop())]
            try:
                done, _ = await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
                for job in done:
                    await job
            finally:
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
