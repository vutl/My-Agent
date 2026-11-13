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
    "research", "note", "ask", "answer", "approval", "error", "budget", "status",
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
    kind: str
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
               parent_id: str | None = None, budget: dict[str, int] | None = None,
               kind: str = "agent") -> Task:
        if kind not in {"agent", "research"}:
            raise ValueError("unknown task kind")
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
                "INSERT INTO tasks(id, conversation_id, parent_id, goal, kind, status, budget_json, "
                "usage_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'planning', ?, ?, ?, ?)",
                (task_id, conversation_id, parent_id, goal.strip(), kind, json.dumps(limits),
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
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, task_id),
            )
            payload = {"from": task.status, "to": status}
            if note is not None:
                payload["note"] = note
            _append_event(connection, task_id, "status", payload, now)
            result = _require(connection, task_id)
        if status == "running":
            from app.services.task_wakeup import TASK_WORKER_WAKEUP
            TASK_WORKER_WAKEUP.wake()
        return result

    def user_interaction(self, task_id: str, kind: str, payload: dict[str, Any]) -> Task:
        """Commit a question/answer and its state transition together.

        An ask parks a running task; an answer hands a waiting one back to the
        runner (the worker claims it, the lease is untouched). The answer's
        ``ask_seq`` must name the task's latest unanswered ask.
        """
        if kind not in {"ask", "answer"} or not isinstance(payload, dict):
            raise ValueError("interaction must be ask or answer")
        source, target = ("running", "waiting_user") if kind == "ask" else ("waiting_user", "running")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if task.status != source:
                raise InvalidTaskTransition(f"Cannot {kind} from {task.status}")
            now = utc_now()
            if kind == "answer" and connection.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'answer' "
                "AND json_extract(payload_json, '$.ask_seq') = ?", (task_id, payload.get("ask_seq")),
            ).fetchone():
                raise InvalidTaskTransition("The question was already answered")
            connection.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?", (target, now, task_id))
            _append_event(connection, task_id, kind, payload, now)
            _append_event(connection, task_id, "status", {"from": source, "to": target}, now)
            result = _require(connection, task_id)
        if target == "running":
            from app.services.task_wakeup import TASK_WORKER_WAKEUP
            TASK_WORKER_WAKEUP.wake()
        return result

    def reserve_notification(self, task_id: str) -> bool:
        """A durable, atomic five-notification budget shared by task turns."""
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require(connection, task_id)
            count = connection.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'note' "
                "AND json_extract(payload_json, '$.notification') = 1", (task_id,),
            ).fetchone()[0]
            if count >= 5:
                return False
            now = utc_now()
            _append_event(connection, task_id, "note", {"notification": True}, now)
            connection.execute("UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id))
            return True

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

    def save_report(self, task_id: str, payload: dict[str, Any], status: str) -> bool:
        """Commit the report and terminal state together, unless an API stop won."""
        if status not in {"done", "failed"}:
            raise ValueError("report status must be done or failed")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if task.status not in {"running", "waiting_user"}:
                return False
            now = utc_now()
            _append_event(connection, task_id, "answer", payload, now)
            connection.execute("UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id))
            if task.status == "running":
                _append_event(connection, task_id, "status", {"from": "running", "to": status}, now)
                connection.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                                   (status, now, task_id))
            return True

    def claim(self, now: datetime, lease_seconds: int, *, reports: bool = False,
              exclude_ids: tuple[str, ...] = ()) -> list[Task]:
        """Atomically lease up to 20 runnable tasks (or undelivered reports)."""
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        now = now.astimezone(UTC)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            status_filter = (
                "status IN ('done', 'failed') AND EXISTS (SELECT 1 FROM task_events a "
                "WHERE a.task_id = tasks.id AND a.kind = 'answer') "
                # Tasks finished by the foreground runner (no execution note)
                # already delivered their report through the chat turn.
                "AND EXISTS (SELECT 1 FROM task_events x WHERE x.task_id = tasks.id "
                "AND x.kind = 'note' AND json_extract(x.payload_json, '$.execution') IS NOT NULL) "
                "AND NOT EXISTS (SELECT 1 FROM task_events e "
                "WHERE e.task_id = tasks.id AND e.kind = 'note' "
                "AND json_extract(e.payload_json, '$.completion_delivered') = 1)"
                if reports else "status = 'running'"
            )
            excluded = " AND id NOT IN (" + ",".join("?" for _ in exclude_ids) + ")" if exclude_ids else ""
            rows = connection.execute(
                "SELECT id FROM tasks WHERE " + status_filter + " "
                "AND (lease_until IS NULL OR lease_until <= ?)" + excluded + " ORDER BY updated_at, id LIMIT 20",
                (now.isoformat(), *exclude_ids),
            ).fetchall()
            result = []
            for row in rows:
                connection.execute(
                    "UPDATE tasks SET lease_until = ?, updated_at = ? WHERE id = ?",
                    (lease_until, now.isoformat(), row["id"]),
                )
                result.append(_require(connection, row["id"]))
            return result

    def renew(self, task_id: str, lease_seconds: int, *, expected_lease: str | None = None,
              now: datetime | None = None) -> Task | None:
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        now = (now or datetime.now(UTC)).astimezone(UTC)
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if expected_lease is not None and task.lease_until != expected_lease:
                return None
            connection.execute("UPDATE tasks SET lease_until = ? WHERE id = ?",
                               ((now + timedelta(seconds=lease_seconds)).isoformat(), task_id))
            return _require(connection, task_id)

    def release(self, task_id: str, *, expected_lease: str | None = None) -> None:
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if expected_lease is not None and task.lease_until != expected_lease:
                return
            connection.execute(
                "UPDATE tasks SET lease_until = NULL, updated_at = ? WHERE id = ?", (utc_now(), task_id)
            )

    def request_approval(self, task_id: str, payload: dict[str, Any]) -> None:
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = _require(connection, task_id)
            if task.status != "running":
                raise InvalidTaskTransition("Task is no longer running")
            now = utc_now()
            _append_event(connection, task_id, "approval", {**payload, "decision": None}, now)
            _append_event(connection, task_id, "status", {"from": "running", "to": "waiting_approval"}, now)
            connection.execute("UPDATE tasks SET status = 'waiting_approval', updated_at = ? WHERE id = ?",
                               (now, task_id))

    def decide_approval(self, run_id: str, call_id: str, approved: bool, *,
                        timed_out: bool = False, now: datetime | None = None) -> bool:
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT e.* FROM task_events e JOIN tasks t ON t.id = e.task_id "
                "WHERE t.status = 'waiting_approval' AND e.kind = 'approval' "
                "AND json_extract(e.payload_json, '$.run_id') = ? "
                "AND json_extract(e.payload_json, '$.call_id') = ? ORDER BY e.seq DESC LIMIT 1",
                (run_id, call_id),
            ).fetchone()
            if row is None:
                return False
            payload = json.loads(row["payload_json"])
            if payload.get("decision") is not None:
                return False
            decided_at = (now or datetime.now(UTC)).isoformat()
            # A late decision is a timeout too, even before the worker's next tick.
            expired = datetime.fromisoformat(decided_at) >= datetime.fromisoformat(row["created_at"]) + timedelta(hours=24)
            _append_event(connection, row["task_id"], "approval",
                          {**payload, "decision": bool(approved) and not expired,
                           "timed_out": timed_out or expired}, decided_at)
            _append_event(connection, row["task_id"], "status",
                          {"from": "waiting_approval", "to": "running"}, decided_at)
            connection.execute("UPDATE tasks SET status = 'running', updated_at = ? WHERE id = ?",
                               (decided_at, row["task_id"]))
        from app.services.task_wakeup import TASK_WORKER_WAKEUP
        TASK_WORKER_WAKEUP.wake()
        return True

    def expire_approvals(self, now: datetime) -> None:
        for task in self.list("waiting_approval", limit=100000):
            event = next((e for e in reversed(self.events(task.id, limit=None)) if e["kind"] == "approval"), None)
            if event is not None and datetime.fromisoformat(event["created_at"]) + timedelta(hours=24) <= now:
                payload = event["payload"]
                self.decide_approval(payload["run_id"], payload["call_id"], False, timed_out=True, now=now)
