"""Reminder tools: create, list and cancel reminders (local time)."""

from __future__ import annotations

from typing import Any

from app.services.reminders import REPEATS, ReminderStore, parse_local, utc_now
from app.tools.registry import Tool, ToolContext, ToolResult


def _store(context: ToolContext) -> ReminderStore:
    return ReminderStore(context.settings.sqlite_db_path)


async def create_reminder(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    text = " ".join(str(arguments.get("text") or "").split())[:300]
    if not text:
        return ToolResult(content="Missing reminder text.", summary="Thiếu nội dung nhắc", ok=False)
    try:
        due = parse_local(str(arguments.get("when") or ""))
    except ValueError:
        return ToolResult(content="'when' must be an ISO date-time like 2026-10-23T15:00.",
                          summary="Thời điểm không hợp lệ", ok=False)
    repeat = str(arguments.get("repeat") or "none")
    if repeat not in REPEATS:
        repeat = "none"
    if due <= utc_now() and repeat == "none":
        return ToolResult(content="That time is already past.", summary="Thời điểm đã qua", ok=False)
    reminder = _store(context).create(text=text, due_at=due, repeat=repeat, conversation_id=context.conversation_id)
    return ToolResult(content=f"Reminder set: {reminder.describe()}", summary=f"Đặt nhắc: {reminder.describe()}")


async def list_reminders(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    reminders = _store(context).pending()
    listing = "\n".join(reminder.describe() for reminder in reminders) or "No pending reminders."
    return ToolResult(content=listing, summary=f"{len(reminders)} lời nhắc đang chờ")


async def cancel_reminder(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    reminder = _store(context).cancel(str(arguments.get("id") or ""))
    if reminder is None:
        return ToolResult(content="No single pending reminder matches that id (use list_reminders).",
                          summary="Không tìm thấy lời nhắc", ok=False)
    return ToolResult(content=f"Cancelled: {reminder.describe()}", summary=f"Hủy nhắc: {reminder.text[:60]}")


CREATE_REMINDER = Tool(
    name="create_reminder",
    description=(
        "Set a reminder. 'when' is an ISO date-time in the user's local time (e.g. 2026-10-23T15:00), "
        "computed from today's date and time; repeat: none | daily | weekdays | weekly. Aya notifies the user "
        "in the app (and on Telegram when connected)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "What to remind, in the user's words."},
            "when": {"type": "string"},
            "repeat": {"type": "string", "enum": list(REPEATS)},
        },
        "required": ["text", "when"],
    },
    handler=create_reminder,
    label="Đặt lời nhắc",
)
LIST_REMINDERS = Tool(name="list_reminders", description="List pending reminders with their ids.",
                      parameters={"type": "object", "properties": {}}, handler=list_reminders, label="Xem lời nhắc")
CANCEL_REMINDER = Tool(
    name="cancel_reminder", description="Cancel a pending reminder by its id (the first 8 characters suffice).",
    parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    handler=cancel_reminder, label="Hủy lời nhắc",
)

REMINDER_TOOLS = (CREATE_REMINDER, LIST_REMINDERS, CANCEL_REMINDER)
