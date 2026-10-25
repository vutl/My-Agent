"""Leased background execution, recovery and durable completion delivery."""

import asyncio
from contextlib import aclosing
from datetime import UTC, datetime
import logging
import re

from app.agents.task_runner import execute_task
from app.services.chat_history import ChatHistory
from app.services.reminders import Reminder, ReminderStore
from app.services.task_deps import task_deps_from_settings
from app.services.task_events import TASK_EVENTS
from app.services.tasks import TaskStore

logger = logging.getLogger(__name__)


def report_message_id(task_id):
    return f"task-report:{task_id}"


def execution_options(store, task_id):
    return next((e["payload"]["execution"] for e in store.events(task_id, limit=None)
                 if e["kind"] == "note" and "execution" in e["payload"]), {})


class TaskWorker:
    def __init__(self, store, deps_factory, *, deliverers=None, hub=TASK_EVENTS,
                 clock=lambda: datetime.now(UTC), renew_interval=20.0):
        self.store, self.deps_factory = store, deps_factory
        self.deliverers, self.hub = deliverers or [], hub
        self.clock, self.renew_interval = clock, renew_interval
        self.executing = {}
        self.notifications = ReminderStore(store.db_path)

    async def notify(self, task, text, kind, suffix):
        self.notifications.notify(kind=kind, title="Aya", body=text, ref_id=task.id,
                                  conversation_id=task.conversation_id,
                                  notification_id=f"task:{task.id}:{suffix}")
        reminder = Reminder(task.id, text, self.clock(), "none", task.status, task.conversation_id)
        for deliver in self.deliverers:
            try:
                await deliver(reminder)
            except Exception:
                logger.exception("task_notification_delivery_failed task_id=%s", task.id)

    async def complete(self, task):
        records = self.store.events(task.id, limit=None)
        if any(e["kind"] == "note" and e["payload"].get("completion_delivered") for e in records):
            return
        answer = next((e["payload"] for e in reversed(records)
                       if e["kind"] == "answer" and "ask_seq" not in e["payload"]), None)
        if answer is None:
            return
        text = answer["text"]
        history = ChatHistory(self.store.db_path)
        # Conversation deletion deliberately unlinks tasks.
        if task.conversation_id:
            used = set(re.findall(r"\[(W\d+)\]", text))
            citations = [{"id": s["id"], "title": s["title"], "url": s.get("url", ""),
                          "filename": s.get("path") or None}
                         for s in answer.get("sources", []) if s["id"] in used]
            history.save_message(conversation_id=task.conversation_id, role="assistant", content=text,
                                 model=execution_options(self.store, task.id).get("model"), citations=citations,
                                 message_id=report_message_id(task.id))
        summary = " ".join(text.split())[:240]
        await self.notify(task, summary, "task", "complete")
        self.store.append_event(task.id, "note", {"completion_delivered": True})
        self.hub.publish(task.id, "avatar.state", {"state": "happy"})

    async def waiting_notification(self, task):
        """A task stopped to ask the user (or hit its budget) with no chat attached:
        keep its question/report in the conversation and tell the user once."""
        records = self.store.events(task.id, limit=None)
        # The newest thing the task said: its question, or a budget report.
        answer = next((e for e in reversed(records)
                       if e["kind"] == "ask" or e["kind"] == "answer" and "ask_seq" not in e["payload"]), None)
        if answer is None or self.hub.attached(task.id):
            return
        seq = answer["seq"]
        if any(e["kind"] == "note" and e["payload"].get("waiting_notified") == seq for e in records):
            return
        text = str(answer["payload"].get("text") or answer["payload"].get("question") or "").strip()
        if not text:
            return
        if task.conversation_id:
            ChatHistory(self.store.db_path).save_message(
                conversation_id=task.conversation_id, role="assistant", content=text,
                model=execution_options(self.store, task.id).get("model"),
                message_id=f"task-wait:{task.id}:{seq}",
            )
        await self.notify(task, "Aya đang chờ cậu: " + " ".join(text.split())[:200], "task_waiting", f"waiting:{seq}")
        self.store.append_event(task.id, "note", {"waiting_notified": seq})

    async def approval_notification(self, task):
        records = self.store.events(task.id, limit=None)
        pending = next((e for e in reversed(records) if e["kind"] == "approval"), None)
        if pending is None or pending["payload"].get("decision") is not None:
            return
        seq = pending["seq"]
        if any(e["kind"] == "note" and e["payload"].get("approval_notified") == seq for e in records):
            return
        if not self.hub.attached(task.id):
            await self.notify(task, "Aya cần cậu duyệt một thao tác", "task_approval", f"approval:{seq}")
            self.store.append_event(task.id, "note", {"approval_notified": seq})

    async def _execute(self, claimed):
        lease = [claimed.lease_until]
        executor = asyncio.current_task()

        async def heartbeat():
            while True:
                await asyncio.sleep(self.renew_interval)
                renewed = self.store.renew(claimed.id, 60, expected_lease=lease[0], now=self.clock())
                if renewed is None:
                    executor.cancel()
                    return
                lease[0] = renewed.lease_until

        renewal = asyncio.create_task(heartbeat())
        try:
            if claimed.status == "running":
                deps = self.deps_factory(claimed.conversation_id)
                deps.lease_valid = lambda: (
                    self.store.get(claimed.id).lease_until == lease[0]
                    and datetime.fromisoformat(lease[0]) > self.clock()
                )
                options = execution_options(self.store, claimed.id)
                async with aclosing(execute_task(claimed.id, deps)) as events:
                    async for name, data in events:
                        self.hub.publish(claimed.id, name, {"run_id": options.get("run_id", claimed.id), **data})
            task = self.store.get(claimed.id)
            if task.status in {"done", "failed"}:
                await self.complete(task)
            elif task.status == "waiting_approval":
                await self.approval_notification(task)
            elif task.status == "waiting_user":
                await self.waiting_notification(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed provider/delivery setup must not kill the scheduler.
            logger.exception("task_worker_execution_failed task_id=%s", claimed.id)
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            self.store.release(claimed.id, expected_lease=lease[0])
            self.executing.pop(claimed.id, None)
            self.hub.publish(claimed.id, "task.updated", {})

    async def tick(self):
        self.store.expire_approvals(self.clock())
        # Recover a crash between the terminal transition and delivery. Stable
        # message/notification IDs make local persistence idempotent.
        for task in self.store.claim(self.clock(), 60, reports=True, exclude_ids=tuple(self.executing)):
            if task.id not in self.executing:
                self.executing[task.id] = asyncio.create_task(self._execute(task))
        for task in self.store.list("waiting_approval", limit=100000):
            if task.id not in self.executing:
                await self.approval_notification(task)
        for task in self.store.claim(self.clock(), 60, exclude_ids=tuple(self.executing)):
            if task.id not in self.executing:
                self.executing[task.id] = asyncio.create_task(self._execute(task))

    async def close(self):
        tasks = list(self.executing.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def task_worker(settings, stop, *, interval=2.0, deliverers=None):
    worker = TaskWorker(TaskStore(settings.sqlite_db_path),
                        lambda cid: task_deps_from_settings(settings, cid), deliverers=deliverers)
    try:
        while not stop.is_set():
            try:
                await worker.tick()
            except Exception:
                logger.exception("task_worker_tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), interval)
            except asyncio.TimeoutError:
                pass
    finally:
        await worker.close()
