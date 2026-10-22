"""Assistant endpoints outside the chat stream: notifications and reminders."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.services.reminders import ReminderStore

router = APIRouter(prefix="/assistant", tags=["assistant"])


def get_reminder_store(settings: Annotated[Settings, Depends(get_settings)]) -> ReminderStore:
    return ReminderStore(settings.sqlite_db_path)


class ReadNotifications(BaseModel):
    ids: list[str]


@router.get("/notifications")
def unread_notifications(store: Annotated[ReminderStore, Depends(get_reminder_store)]) -> dict[str, Any]:
    return {"notifications": store.unread()}


@router.post("/notifications/read")
def read_notifications(body: ReadNotifications,
                       store: Annotated[ReminderStore, Depends(get_reminder_store)]) -> dict[str, Any]:
    return {"updated": store.mark_read(body.ids)}


@router.get("/reminders")
def pending_reminders(store: Annotated[ReminderStore, Depends(get_reminder_store)]) -> dict[str, Any]:
    return {"reminders": [
        {"id": item.id, "text": item.text, "due_at": item.due_at.isoformat(), "repeat": item.repeat}
        for item in store.pending()
    ]}
