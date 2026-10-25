"""Reminders and notifications for Aya's assistant.

Reminders live in SQLite so they survive restarts. A worker in the server's
lifespan claims due reminders with a short lease (a crash mid-delivery means
the next tick retries), records a notification for the app (and any other
channel registered later, e.g. Telegram) and reschedules recurring ones.
Times are stored in UTC; the user speaks local time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sqlite3
import uuid

from app.db.sqlite import connect

logger = logging.getLogger(__name__)

REPEATS = ("none", "daily", "weekdays", "weekly")
_LEASE_SECONDS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    due_at TEXT NOT NULL,
    repeat TEXT NOT NULL DEFAULT 'none',
    status TEXT NOT NULL DEFAULT 'pending',
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    fired_at TEXT,
    lease_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at);
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    read_at TEXT,
    ref_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(read_at, created_at);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_local(text: str) -> datetime:
    """An ISO date-time the model wrote in the user's local time (or with an
    explicit offset) → aware UTC datetime."""
    value = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.astimezone()  # interpret as local wall-clock time
    return value.astimezone(timezone.utc)


def next_due(due: datetime, repeat: str) -> datetime | None:
    if repeat == "daily":
        return due + timedelta(days=1)
    if repeat == "weekly":
        return due + timedelta(days=7)
    if repeat == "weekdays":
        local = due.astimezone()
        step = local + timedelta(days=1)
        while step.weekday() >= 5:
            step += timedelta(days=1)
        return step.astimezone(timezone.utc)
    return None


@dataclass
class Reminder:
    id: str
    text: str
    due_at: datetime
    repeat: str
    status: str
    conversation_id: str | None = None

    def describe(self) -> str:
        local = self.due_at.astimezone()
        repeat = "" if self.repeat == "none" else f" (lặp lại: {self.repeat})"
        return f"{self.id[:8]} — {local:%Y-%m-%d %H:%M} — {self.text}{repeat}"


class ReminderStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        with connect(self.db_path) as connection:
            connection.executescript(_SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(notifications)")}
            if "conversation_id" not in columns:
                connection.execute("ALTER TABLE notifications ADD COLUMN conversation_id TEXT")

    def _row(self, row: sqlite3.Row) -> Reminder:
        return Reminder(id=row["id"], text=row["text"], due_at=datetime.fromisoformat(row["due_at"]),
                        repeat=row["repeat"], status=row["status"], conversation_id=row["conversation_id"])

    def create(self, *, text: str, due_at: datetime, repeat: str = "none",
               conversation_id: str | None = None) -> Reminder:
        if repeat not in REPEATS:
            raise ValueError(f"repeat must be one of {REPEATS}")
        reminder_id = str(uuid.uuid4())
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO reminders(id, text, due_at, repeat, status, conversation_id, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (reminder_id, text.strip(), _iso(due_at), repeat, conversation_id, _iso(utc_now())),
            )
        return Reminder(id=reminder_id, text=text.strip(), due_at=due_at, repeat=repeat, status="pending",
                        conversation_id=conversation_id)

    def pending(self) -> list[Reminder]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE status = 'pending' ORDER BY due_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    def cancel(self, prefix: str) -> Reminder | None:
        prefix = str(prefix or "").strip()
        if len(prefix) < 4:
            return None
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND id LIKE ?", (prefix + "%",)
            ).fetchall()
            if len(rows) != 1:
                return None
            connection.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (rows[0]["id"],))
        return self._row(rows[0])

    def claim_due(self, now: datetime | None = None) -> list[Reminder]:
        """Due reminders not leased by another tick; leases them."""
        now = now or utc_now()
        lease = _iso(now + timedelta(seconds=_LEASE_SECONDS))
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? "
                "AND (lease_until IS NULL OR lease_until < ?) ORDER BY due_at LIMIT 20",
                (_iso(now), _iso(now)),
            ).fetchall()
            for row in rows:
                connection.execute("UPDATE reminders SET lease_until = ? WHERE id = ?", (lease, row["id"]))
        return [self._row(row) for row in rows]

    def mark_fired(self, reminder: Reminder, now: datetime | None = None) -> None:
        now = now or utc_now()
        following = next_due(reminder.due_at, reminder.repeat)
        # A recurring reminder that was missed for a while (app closed) skips
        # to its next future slot instead of firing once per missed slot.
        while following is not None and following <= now:
            following = next_due(following, reminder.repeat)
        with connect(self.db_path) as connection:
            if following is None:
                connection.execute(
                    "UPDATE reminders SET status = 'fired', fired_at = ?, lease_until = NULL WHERE id = ?",
                    (_iso(now), reminder.id),
                )
            else:
                connection.execute(
                    "UPDATE reminders SET due_at = ?, fired_at = ?, lease_until = NULL WHERE id = ?",
                    (_iso(following), _iso(now), reminder.id),
                )

    # notifications -------------------------------------------------------
    def notify(self, *, kind: str, title: str, body: str, ref_id: str | None = None,
               conversation_id: str | None = None, notification_id: str | None = None) -> str:
        notification_id = notification_id or str(uuid.uuid4())
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO notifications(id, kind, title, body, created_at, ref_id, conversation_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (notification_id, kind, title, body, _iso(utc_now()), ref_id, conversation_id),
            )
        return notification_id

    def post_to_conversation(self, reminder: Reminder) -> None:
        """A lasting trace of the reminder in the chat it was set from."""
        if not reminder.conversation_id:
            return
        from app.services.chat_history import ChatHistory

        with connect(self.db_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (reminder.conversation_id,)
            ).fetchone()
        if exists:
            ChatHistory(self.db_path).save_message(
                conversation_id=reminder.conversation_id, role="assistant",
                content=f"⏰ Aya nhắc cậu nè~ **{reminder.text}**",
            )

    def unread(self) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT id, kind, title, body, created_at, conversation_id FROM notifications WHERE read_at IS NULL "
                "ORDER BY created_at LIMIT 50"
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_read(self, ids: list[str]) -> int:
        with connect(self.db_path) as connection:
            cursor = connection.executemany(
                "UPDATE notifications SET read_at = ? WHERE id = ? AND read_at IS NULL",
                [(_iso(utc_now()), item) for item in ids],
            )
            return cursor.rowcount


Deliverer = Callable[[Reminder], Awaitable[None]]


async def reminder_worker(store: ReminderStore, stop: asyncio.Event, *, interval: float = 15.0,
                          deliverers: list[Deliverer] | None = None) -> None:
    """Fire due reminders until ``stop`` is set."""
    while not stop.is_set():
        try:
            for reminder in await asyncio.to_thread(store.claim_due):
                await asyncio.to_thread(store.notify, kind="reminder", title="Aya nhắc cậu nè~",
                                        body=reminder.text, ref_id=reminder.id,
                                        conversation_id=reminder.conversation_id)
                try:
                    await asyncio.to_thread(store.post_to_conversation, reminder)
                except Exception as exc:  # noqa: BLE001 - the notification already went out
                    logger.warning("reminder_chat_post_failed error_type=%s", type(exc).__name__)
                for deliver in deliverers or []:
                    try:
                        await deliver(reminder)
                    except Exception as exc:  # noqa: BLE001 - one channel must not block the others
                        logger.warning("reminder_delivery_failed error_type=%s", type(exc).__name__)
                await asyncio.to_thread(store.mark_fired, reminder)
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            logger.warning("reminder_worker_error error_type=%s", type(exc).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
