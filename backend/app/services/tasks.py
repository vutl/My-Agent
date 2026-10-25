"""Persistent task state and ordered events; execution belongs to the runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from app.db.sqlite import connect


TRANSITIONS = {
    "planning": {"running", "waiting_user", "failed", "cancelled"},
    "running": {"waiting_user", "waiting_approval", "paused", "done", "failed", "cancelled"},
    "waiting_user": {"running", "cancelled", "failed"},
    "waiting_approval": {"running", "cancelled", "failed"},
    "paused": {"running", "cancelled"},
    "done": set(),
    "failed": set(),
    "cancelled": set(),
}
STEP_STATES = {"pending", "running", "done", "skipped", "failed"}
EVENT_KINDS = {
    "plan", "plan_edit", "step_started", "step_done", "tool_call", "tool_result",
    "note", "ask", "answer", "approval", "error", "budget", "status",
}
DEFAULT_BUDGET = {"max_steps": 20, "max_tool_calls": 40, "max_seconds": 900}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class InvalidTaskTransition(ValueError):
    """The requested state change is not allowed from the current state."""


@dataclass(frozen=True)
class Task:
    id: str
    conversation_id: str | None
    parent_id: str | None
    goal: str
    status: str
    plan: list[dict[str, Any]]
    summary: str | None
    budget: dict[str, int]
    usage: dict[str, int | float]
    lease_until: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _task(row: sqlite3.Row) -> Task:
    values = dict(row)
    for name in ("plan", "budget", "usage"):
        values[name] = json.loads(values.pop(f"{name}_json"))
    return Task(**values)


def _require(connection: sqlite3.Connection, task_id: str) -> Task:
    row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise KeyError(task_id)
    return _task(row)


def _append_event(connection: sqlite3.Connection, task_id: str, kind: str,
                  payload: dict[str, Any], now: str) -> int:
    # All callers already hold the write lock from BEGIN IMMEDIATE.
    seq = connection.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO task_events(task_id, seq, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, seq, kind, json.dumps(payload, ensure_ascii=False, allow_nan=False), now),
    )
    return seq


class TaskStore:
    """Uses the application's schema, initialized by init_db at startup.

    Unknown IDs return None from get and raise KeyError from mutations/events.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def create(self, goal: str, conversation_id: str | None = None,
               parent_id: str | None = None, budget: dict[str, int] | None = None) -> Task:
        if not isinstance(goal, str) or not goal.strip() or len(goal.strip()) > 2000:
            raise ValueError("goal must contain 1 to 2000 characters")
        overrides = {} if budget is None else budget
        if not isinstance(overrides, dict) or overrides.keys() - DEFAULT_BUDGET.keys():
            raise ValueError("unknown budget keys")
        limits = {**DEFAULT_BUDGET, **overrides}
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("budget values must be positive integers")
        task_id = str(uuid.uuid4())
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO tasks(id, conversation_id, parent_id, goal, status, budget_json, "
                "usage_json, created_at, updated_at) VALUES (?, ?, ?, ?, 'planning', ?, ?, ?, ?)",
                (task_id, conversation_id, parent_id, goal.strip(), json.dumps(limits),
                 json.dumps({"steps": 0, "tool_calls": 0, "seconds": 0.0}), now, now),
            )
            return _require(connection, task_id)

    def get(self, task_id: str) -> Task | None:
        with connect(self.db_path) as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task(row) if row is not None else None

    def list(self, status: str | None = None, conversation_id: str | None = None,
             limit: int = 50) -> list[Task]:
        if status is not None and status not in TRANSITIONS:
            raise ValueError("unknown task status")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        clauses, params = [], []
        for name, value in (("status", status), ("conversation_id", conversation_id)):
            if value is not None:
                clauses.append(f"{name} = ?")
                params.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM tasks" + where + " ORDER BY created_at DESC, id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_task(row) for row in rows]

    def transition(self, task_id: str, status: str, note: str | None = None, *,
                   expected_status: str | None = None) -> Task:
        # expected_status lets API actions such as resume guard their narrower
        # precondition inside the same transaction as the state change.
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if (expected_status is not None and task.status != expected_status
                    or status not in TRANSITIONS[task.status]):
                raise InvalidTaskTransition(f"Cannot transition {task.status} to {status}")
            now = utc_now()
            connection.execute(
                "UPDATE tasks SET status = ?, updated_at = ?, lease_until = NULL WHERE id = ?",
                (status, now, task_id),
            )
            payload = {"from": task.status, "to": status}
            if note is not None:
                payload["note"] = note
            _append_event(connection, task_id, "status", payload, now)
            return _require(connection, task_id)

    def set_plan(self, task_id: str, plan: list[dict[str, Any]]) -> Task:
        if not isinstance(plan, list):
            raise ValueError("plan must be a list")
        normalized, ids = [], set()
        for step in plan:
            if (not isinstance(step, dict) or not isinstance(step.get("id"), str)
                    or not step["id"].strip() or step["id"] in ids):
                raise ValueError("plan steps must have unique non-empty string IDs")
            state = step.get("state", "pending")
            if state not in STEP_STATES:
                raise ValueError("unknown step state")
            ids.add(step["id"])
            normalized.append({**step, "state": state})
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require(connection, task_id)
            edited = connection.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'plan' LIMIT 1", (task_id,)
            ).fetchone() is not None
            now = utc_now()
            connection.execute(
                "UPDATE tasks SET plan_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(normalized, ensure_ascii=False, allow_nan=False), now, task_id),
            )
            _append_event(connection, task_id, "plan_edit" if edited else "plan", {"plan": normalized}, now)
            return _require(connection, task_id)

    def update_step(self, task_id: str, step_id: str, state: str) -> Task:
        if state not in STEP_STATES:
            raise ValueError("unknown step state")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            step = next((step for step in task.plan if step["id"] == step_id), None)
            if step is None:
                raise KeyError(step_id)
            step["state"] = state
            now = utc_now()
            connection.execute(
                "UPDATE tasks SET plan_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(task.plan, ensure_ascii=False), now, task_id),
            )
            kind = "step_started" if state == "running" else (
                "step_done" if state in {"done", "skipped"} else "plan_edit"
            )
            _append_event(connection, task_id, kind, {"step_id": step_id, "state": state}, now)
            return _require(connection, task_id)

    def append_event(self, task_id: str, kind: str, payload: dict[str, Any]) -> int:
        if kind not in EVENT_KINDS or not isinstance(payload, dict):
            raise ValueError("event requires a known kind and an object payload")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require(connection, task_id)
            now = utc_now()
            seq = _append_event(connection, task_id, kind, payload, now)
            connection.execute("UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id))
            return seq

    def events(self, task_id: str, after_seq: int = 0, limit: int | None = 500) -> list[dict[str, Any]]:
        if after_seq < 0 or limit is not None and limit <= 0:
            raise ValueError("invalid event cursor or limit")
        with connect(self.db_path) as connection:
            _require(connection, task_id)
            rows = connection.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (task_id, after_seq, -1 if limit is None else limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def add_usage(self, task_id: str, steps: int = 0, tool_calls: int = 0, seconds: float = 0.0) -> bool:
        if (any(type(value) is not int or value < 0 for value in (steps, tool_calls))
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("usage increments must be finite and non-negative; counts must be integers")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            for key, value in (("steps", steps), ("tool_calls", tool_calls), ("seconds", seconds)):
                task.usage[key] = task.usage.get(key, 0) + value
            connection.execute(
                "UPDATE tasks SET usage_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(task.usage, allow_nan=False), utc_now(), task_id),
            )
            return any(task.usage[key] > task.budget[f"max_{key}"] for key in task.usage)

    def claim(self, now: datetime, lease_seconds: int) -> list[Task]:
        """Atomically lease up to 20 running tasks, oldest updated first."""
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        now = now.astimezone(UTC)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM tasks WHERE status = 'running' "
                "AND (lease_until IS NULL OR lease_until <= ?) ORDER BY updated_at, id LIMIT 20",
                (now.isoformat(),),
            ).fetchall()
            result = []
            for row in rows:
                connection.execute(
                    "UPDATE tasks SET lease_until = ?, updated_at = ? WHERE id = ?",
                    (lease_until, now.isoformat(), row["id"]),
                )
                result.append(_require(connection, row["id"]))
            return result

    def release(self, task_id: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require(connection, task_id)
            connection.execute(
                "UPDATE tasks SET lease_until = NULL, updated_at = ? WHERE id = ?", (utc_now(), task_id)
            )
