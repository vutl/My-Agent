"""Loop-local worker events that can also be signalled by sync API threads."""

import asyncio
from contextlib import asynccontextmanager
import threading


class TaskWorkerWakeup:
    def __init__(self):
        self._subscribers = set()
        self._lock = threading.Lock()

    @asynccontextmanager
    async def subscribe(self):
        event = asyncio.Event()
        subscriber = (asyncio.get_running_loop(), event)
        with self._lock:
            self._subscribers.add(subscriber)
        try:
            yield event
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)

    def wake(self):
        with self._lock:
            for loop, event in self._subscribers:
                loop.call_soon_threadsafe(event.set)


TASK_WORKER_WAKEUP = TaskWorkerWakeup()
