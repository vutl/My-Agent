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
        self._answer_text = {}

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
        if name == "answer.delta":
            text = self._answer_text.get(task_id, "") + data["text"]
            self._answer_text[task_id] = text
            # A late subscriber or a full queue can miss deltas. Carry the
            # current prefix internally so the watcher can fill those gaps.
            data = {**data, "_answer_text": text}
        elif name == "answer" or name == "task.updated" and not data:
            self._answer_text.pop(task_id, None)
        for queue in tuple(self._watchers.get(task_id, ())):
            if queue.full():
                queue.get_nowait()  # durable events still replay from SQLite
            queue.put_nowait((name, data))


TASK_EVENTS = TaskEventHub()


def is_internal_event(event: dict) -> bool:
    return event["kind"] == "note" and "execution" in event["payload"]


async def watch_task(store: TaskStore, task_id: str, *, after_seq: int = 0,
                     hub: TaskEventHub = TASK_EVENTS, interval: float = 0.2,
                     stop_when_waiting: bool = False,
                     is_disconnected: Callable[[], Awaitable[bool]] | None = None) -> AsyncIterator[tuple[str, dict]]:
    async with hub.subscribe(task_id) as queue:
        last_task = None
        streamed_text = ""

        def transient(name, data):
            nonlocal streamed_text
            if name == "answer.delta":
                prefix = data.get("_answer_text", streamed_text + data["text"])
                text = prefix[len(streamed_text):]
                streamed_text = prefix
                if text:
                    yield name, {k: v for k, v in {**data, "text": text}.items() if k != "_answer_text"}
            elif name not in {"task.event", "task.updated", "answer"}:
                yield name, data

        while True:
            if is_disconnected is not None and await is_disconnected():
                return
            # Read status BEFORE events: if completion races this read, the next
            # iteration drains its final events before returning.
            task = store.get(task_id)
            if task is None:
                return
            records = store.events(task_id, after_seq=after_seq, limit=None)
            # Flush live report deltas before their durable final answer. The
            # worker can finish between watcher iterations (or while attached
            # code handles a yielded event), so snapshot the queue before yield.
            live = []
            while not queue.empty():
                live.append(queue.get_nowait())
            for name, data in live:
                for item in transient(name, data):
                    yield item
            for event in records:
                after_seq = event["seq"]
                if is_internal_event(event):
                    continue
                yield "task.event", {k: event[k] for k in ("task_id", "seq", "kind", "payload")}
                if event["kind"] == "answer" and "ask_seq" not in event["payload"]:  # not the user's reply
                    text = event["payload"]["text"]
                    if streamed_text and text.startswith(streamed_text):
                        # The durable report also repairs a missed final delta.
                        if tail := text[len(streamed_text):]:
                            yield "answer.delta", {"text": tail}
                        streamed_text = text
                    yield "answer", event["payload"]
                    # A later report (after a budget stop and resume) starts from
                    # nothing; keeping this report as the prefix cut its opening.
                    streamed_text = ""
                elif event["kind"] == "ask":
                    yield "answer", {"text": event["payload"]["question"], "sources": []}
            if task != last_task:
                yield "task.updated", {"task": task.to_dict()}
                last_task = task
            released = task.lease_until is None or datetime.fromisoformat(task.lease_until) <= datetime.now(UTC)
            if task.status in TERMINAL or stop_when_waiting and released and task.status in {"waiting_user", "paused"}:
                return
            try:
                name, data = await asyncio.wait_for(queue.get(), interval)
                for item in transient(name, data):
                    yield item
            except asyncio.TimeoutError:
                pass
