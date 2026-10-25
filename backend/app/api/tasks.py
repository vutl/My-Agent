"""Task inspection and explicit lifecycle actions, without an executor."""

from typing import Annotated, Any, Literal
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.services.tasks import InvalidTaskTransition, TaskStore


router = APIRouter(prefix="/tasks", tags=["tasks"])
TaskStatus = Literal[
    "planning", "running", "waiting_user", "waiting_approval", "paused", "done", "failed", "cancelled",
]


def get_task_store(settings: Annotated[Settings, Depends(get_settings)]) -> TaskStore:
    return TaskStore(settings.sqlite_db_path)


Store = Annotated[TaskStore, Depends(get_task_store)]


class CreateTask(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    budget: dict[str, Any] | None = None

    @field_validator("goal")
    @classmethod
    def non_empty_goal(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must not be blank")
        return value.strip()


@router.get("")
def list_tasks(store: Store, status: TaskStatus | None = None, conversation_id: str | None = None,
               limit: Annotated[int, Query(gt=0)] = 50) -> dict[str, Any]:
    return {"tasks": [task.to_dict() for task in store.list(status, conversation_id, limit)]}


@router.get("/{task_id}")
def get_task(task_id: str, store: Store) -> dict[str, Any]:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return {**task.to_dict(), "events": store.events(task_id, limit=None)}


@router.post("", status_code=201)
def create_task(body: CreateTask, store: Store) -> dict[str, Any]:
    try:
        return store.create(body.goal, conversation_id=body.conversation_id, budget=body.budget).to_dict()
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _transition(store: TaskStore, task_id: str, status: str,
                expected_status: str | None = None) -> dict[str, Any]:
    try:
        return store.transition(task_id, status, expected_status=expected_status).to_dict()
    except KeyError as exc:
        raise HTTPException(404, "Task not found") from exc
    except InvalidTaskTransition as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{task_id}/cancel")
def cancel_task(task_id: str, store: Store) -> dict[str, Any]:
    return _transition(store, task_id, "cancelled")


@router.post("/{task_id}/pause")
def pause_task(task_id: str, store: Store) -> dict[str, Any]:
    return _transition(store, task_id, "paused")


@router.post("/{task_id}/resume")
def resume_task(task_id: str, store: Store) -> dict[str, Any]:
    return _transition(store, task_id, "running", expected_status="paused")
