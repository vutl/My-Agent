"""Durable event replay with process-local wakeups and transient tool cards."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from app.services.tasks import TaskStore

TERMINAL = {"done", "failed", "cancelled"}


class TaskEventHub:
    def __init__(self):
        self._watchers = defaultdict(set)

    @asynccontextmanager
    async def subscribe(self, task_id):
        queue = asyncio.Queue(maxsize=256)
        self._watchers[task_id].add(queue)
        try:
            yield queue
        finally:
            self._watchers[task_id].discard(queue)
            if not self._watchers[task_id]:
                self._watchers.pop(task_id, None)

    def attached(self, task_id):
        return bool(self._watchers.get(task_id))

    def publish(self, task_id, name, data):
        for queue in tuple(self._watchers.get(task_id, ())):
            if queue.full():
                queue.get_nowait()  # durable events still replay from SQLite
            queue.put_nowait((name, data))


TASK_EVENTS = TaskEventHub()


async def watch_task(store: TaskStore, task_id: str, *, after_seq: int = 0,
                     hub: TaskEventHub = TASK_EVENTS, interval: float = 0.2,
                     stop_when_waiting: bool = False,
                     is_disconnected: Callable[[], Awaitable[bool]] | None = None) -> AsyncIterator[tuple[str, dict]]:
    async with hub.subscribe(task_id) as queue:
        last_task = None
        while True:
            if is_disconnected is not None and await is_disconnected():
                return
            # Read status BEFORE events: if completion races this read, the next
            # iteration drains its final events before returning.
            task = store.get(task_id)
            if task is None:
                return
            records = store.events(task_id, after_seq=after_seq, limit=None)
            for event in records:
                after_seq = event["seq"]
                yield "task.event", {k: event[k] for k in ("task_id", "seq", "kind", "payload")}
                if event["kind"] == "answer" and "ask_seq" not in event["payload"]:  # not the user's reply
                    yield "answer", event["payload"]
                elif event["kind"] == "ask":
                    yield "answer", {"text": event["payload"]["question"], "sources": []}
            if task != last_task:
                yield "task.updated", {"task": task.to_dict()}
                last_task = task
            while not queue.empty():
                name, data = queue.get_nowait()
                if name not in {"task.event", "task.updated", "answer"}:
                    yield name, data
            released = task.lease_until is None or datetime.fromisoformat(task.lease_until) <= datetime.now(UTC)
            if task.status in TERMINAL or stop_when_waiting and released and task.status in {"waiting_user", "paused"}:
                return
            try:
                name, data = await asyncio.wait_for(queue.get(), interval)
                if name not in {"task.event", "task.updated", "answer"}:
                    yield name, data
            except asyncio.TimeoutError:
                pass
