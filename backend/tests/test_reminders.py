from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.services.reminders import ReminderStore, next_due, parse_local, reminder_worker, utc_now
from app.tools import reminders as reminder_tools
from app.tools.registry import ToolContext


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="r", conversation_id="c", workspace=tmp_path,
                       settings=SimpleNamespace(sqlite_db_path=tmp_path / "app.db"))


def test_local_time_is_stored_in_utc() -> None:
    local = parse_local("2026-10-23T15:00")
    assert local.tzinfo is not None and local == datetime(2026, 10, 23, 15, 0).astimezone().astimezone(timezone.utc)
    assert parse_local("2026-10-23T15:00+07:00") == datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


def test_recurrence() -> None:
    friday = datetime(2026, 10, 23, 9, 0).astimezone()
    assert next_due(friday, "daily") == friday + timedelta(days=1)
    assert next_due(friday, "weekly") == friday + timedelta(days=7)
    assert next_due(friday, "weekdays").astimezone().weekday() == 0  # Friday → Monday
    assert next_due(friday, "none") is None


def test_tools_create_list_cancel(tmp_path) -> None:
    context = _context(tmp_path)
    future = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
    created = asyncio.run(reminder_tools.create_reminder({"text": "uống nước", "when": future}, context))
    assert created.ok
    past = asyncio.run(reminder_tools.create_reminder({"text": "x", "when": "2020-01-01T00:00"}, context))
    assert not past.ok
    listing = asyncio.run(reminder_tools.list_reminders({}, context))
    assert "uống nước" in listing.content
    reminder_id = listing.content.split(" — ")[0]
    assert asyncio.run(reminder_tools.cancel_reminder({"id": reminder_id}, context)).ok
    assert "No pending" in asyncio.run(reminder_tools.list_reminders({}, context)).content


def test_worker_fires_due_reminders_once_and_reschedules_recurring(tmp_path) -> None:
    store = ReminderStore(tmp_path / "app.db")
    store.create(text="họp nhóm", due_at=utc_now() - timedelta(seconds=5))
    daily = store.create(text="uống thuốc", due_at=utc_now() - timedelta(days=3), repeat="daily")
    delivered: list[str] = []

    async def deliver(reminder):
        delivered.append(reminder.text)

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(reminder_worker(store, stop, interval=0.05, deliverers=[deliver]))
        await asyncio.sleep(0.3)
        stop.set()
        await task

    asyncio.run(main())
    assert sorted(delivered) == ["họp nhóm", "uống thuốc"]  # each once, missed daily slots skipped
    assert len(store.unread()) == 2
    pending = store.pending()
    assert [item.id for item in pending] == [daily.id] and pending[0].due_at > utc_now()
    assert store.mark_read([item["id"] for item in store.unread()]) == 2
