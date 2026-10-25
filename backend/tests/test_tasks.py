from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import itertools
import sqlite3
from threading import Barrier
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api import tasks as api
from app.core.config import Settings, get_settings
from app.db.sqlite import SCHEMA, connect, init_db
from app.services.chat_history import ChatHistory
from app.services.tasks import DEFAULT_BUDGET, InvalidTaskTransition, TaskStore


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "tasks.db"
    init_db(path)
    return TaskStore(path)


# Independent specification: changing the implementation's transition table
# must not silently change the expected state machine in these tests.
ALLOWED = {
    "planning": {"running", "waiting_user", "failed", "cancelled"},
    "running": {"waiting_user", "waiting_approval", "paused", "done", "failed", "cancelled"},
    "waiting_user": {"running", "cancelled", "failed"},
    "waiting_approval": {"running", "cancelled", "failed"},
    "paused": {"running", "cancelled"},
    "done": set(), "failed": set(), "cancelled": set(),
}


def task_at(store, status):
    task = store.create("A persistent goal")
    if status == "planning":
        return task
    if status in {"waiting_approval", "paused", "done"}:
        store.transition(task.id, "running")
    return store.transition(task.id, status)


@pytest.mark.parametrize(("source", "target"), list(itertools.product(ALLOWED, repeat=2)))
def test_state_machine(store, source, target):
    task = task_at(store, source)
    before = store.events(task.id)
    if target not in ALLOWED[source]:
        with pytest.raises(InvalidTaskTransition):
            store.transition(task.id, target)
        assert store.get(task.id) == task
        assert store.events(task.id) == before
    else:
        changed = store.transition(task.id, target, note="User requested")
        assert changed.status == target
        assert changed.updated_at >= task.updated_at
        event = store.events(task.id)[-1]
        assert event["seq"] == len(before) + 1
        assert event["kind"] == "status"
        assert event["payload"] == {"from": source, "to": target, "note": "User requested"}
        assert event["created_at"] == changed.updated_at


def test_create_roundtrip_and_filters(store):
    history = ChatHistory(store.db_path)
    conversation = history.ensure_conversation(None, "Tasks")
    first = store.create("  Write a report  ", conversation_id=conversation)
    second = store.create("Find sources", parent_id=first.id)
    assert first.goal == "Write a report" and first.status == "planning"
    assert first.plan == [] and first.summary is None and first.lease_until is None
    assert first.budget == DEFAULT_BUDGET
    assert first.usage == {"steps": 0, "tool_calls": 0, "seconds": 0.0}
    assert first.created_at == first.updated_at
    assert store.get(first.id) == first
    assert TaskStore(store.db_path).get(second.id).parent_id == first.id
    assert store.list() == [second, first]
    assert store.list(limit=1) == [second]
    assert store.list(conversation_id=conversation) == [first]
    store.transition(second.id, "running")
    assert [item.id for item in store.list(status="running")] == [second.id]
    assert store.list(status="running", conversation_id=conversation) == []
    assert store.get("missing") is None
    assert "plan_json" not in first.to_dict()
    assert first.to_dict()["plan"] == []


def test_two_threads_append_events_without_duplicate_or_lost_sequences(store):
    task = store.create("Concurrent events")
    barrier = Barrier(2)

    def append(worker):
        independent_store = TaskStore(store.db_path)
        barrier.wait()
        return [independent_store.append_event(task.id, "note", {"worker": worker, "index": i})
                for i in range(40)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, range(2)))
    assert sorted(itertools.chain.from_iterable(results)) == list(range(1, 81))
    events = store.events(task.id)
    assert [event["seq"] for event in events] == list(range(1, 81))
    assert {(event["payload"]["worker"], event["payload"]["index"]) for event in events} == {
        (worker, i) for worker in range(2) for i in range(40)
    }
    assert [event["seq"] for event in store.events(task.id, after_seq=30, limit=2)] == [31, 32]
    assert store.get(task.id).updated_at == events[-1]["created_at"]
    assert store.append_event(store.create("Another task").id, "note", {}) == 1


def test_plan_set_edit_and_step_updates(store):
    task = store.create("Plan")
    store.set_plan(task.id, [])
    plan = [{"id": "read", "title": "Read sources", "tools_hint": ["web"], "done_when": "Cited notes"}]
    planned = store.set_plan(task.id, plan)
    assert planned.plan == [{**plan[0], "state": "pending"}]
    assert "state" not in plan[0]  # no caller-owned object mutation
    assert [event["kind"] for event in store.events(task.id)] == ["plan", "plan_edit"]
    for state in ("running", "done", "pending", "skipped", "failed"):
        changed = store.update_step(task.id, "read", state)
        assert changed.plan[0]["state"] == state
        assert store.get(task.id).plan == changed.plan
        assert store.events(task.id)[-1]["payload"] == {"step_id": "read", "state": state}
    events = store.events(task.id)
    assert [event["kind"] for event in events[2:4]] == ["step_started", "step_done"]
    with pytest.raises(KeyError):
        store.update_step(task.id, "missing-step", "done")
    with pytest.raises(ValueError):
        store.update_step(task.id, "read", "unknown")
    with pytest.raises(ValueError):
        store.set_plan(task.id, [{"id": "duplicate"}, {"id": "duplicate"}])
    assert store.events(task.id) == events


@pytest.mark.parametrize("budget", [
    {"max_steps": 0}, {"max_steps": -1}, {"max_tool_calls": 1.5}, {"max_seconds": "20"},
    {"max_steps": True}, {"max_seconds": None}, {"unknown": 1}, [],
])
def test_invalid_budget(store, budget):
    with pytest.raises(ValueError):
        store.create("Goal", budget=budget)
    assert store.list() == []


@pytest.mark.parametrize(("key", "increment"), [
    ("max_steps", {"steps": 1}), ("max_tool_calls", {"tool_calls": 1}),
    ("max_seconds", {"seconds": 1.0}),
])
def test_usage_accumulates_and_exceeded_flag(store, key, increment):
    task = store.create("Budget", budget={key: 2})
    assert task.budget == {**DEFAULT_BUDGET, key: 2}
    assert not store.add_usage(task.id, **increment)
    assert not store.add_usage(task.id, **increment)  # exactly at the limit
    assert store.add_usage(task.id, **increment)
    usage_key = key.removeprefix("max_")
    assert store.get(task.id).usage[usage_key] == 3
    assert store.get(task.id).status == "planning"  # runner decides how to react
    assert store.add_usage(task.id)  # already exceeded


def test_default_budgets_and_fractional_seconds(store):
    task = store.create("Default budgets")
    assert not store.add_usage(task.id, steps=20, tool_calls=40, seconds=899.5)
    assert not store.add_usage(task.id, seconds=0.5)
    assert store.add_usage(task.id, seconds=0.25)
    assert store.get(task.id).usage == {"steps": 20, "tool_calls": 40, "seconds": 900.25}


@pytest.mark.parametrize("increments", [
    {"steps": -1}, {"steps": 0.5}, {"tool_calls": True}, {"seconds": -1},
    {"seconds": float("nan")}, {"seconds": float("inf")},
])
def test_invalid_usage_does_not_change_task(store, increments):
    task = store.create("Usage")
    with pytest.raises(ValueError):
        store.add_usage(task.id, **increments)
    assert store.get(task.id) == task


def test_claim_release_expiry_and_status_filter(store):
    for status in ALLOWED:
        task_at(store, status)
    now = datetime.now(UTC)
    running = store.list(status="running")[0]
    claimed = store.claim(now, 30)
    assert [task.id for task in claimed] == [running.id]
    assert datetime.fromisoformat(claimed[0].lease_until) == now + timedelta(seconds=30)
    assert TaskStore(store.db_path).claim(now, 30) == []
    assert store.claim(now + timedelta(seconds=29), 30) == []
    assert [task.id for task in store.claim(now + timedelta(seconds=30), 30)] == [running.id]
    store.release(running.id)
    assert store.get(running.id).lease_until is None
    assert [task.id for task in store.claim(now, 30)] == [running.id]
    store.transition(running.id, "paused")
    assert store.get(running.id).lease_until is None
    assert store.claim(now, 30) == []
    store.transition(running.id, "running")
    assert [task.id for task in store.claim(now, 30)] == [running.id]


def test_claim_is_exclusive_between_workers(store):
    task = task_at(store, "running")
    barrier = Barrier(2)
    now = datetime.now(UTC)

    def claim(_):
        barrier.wait()
        return TaskStore(store.db_path).claim(now, 30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    assert [item.id for batch in results for item in batch] == [task.id]


def test_conversation_delete_preserves_task_and_events(store):
    history = ChatHistory(store.db_path)
    conversation = history.ensure_conversation(None, "Tasks")
    history.save_message(conversation_id=conversation, role="user", content="Keep my task")
    task = store.create("Outlive the conversation", conversation_id=conversation)
    store.append_event(task.id, "note", {"text": "Keep this too"})
    assert history.delete_conversation(conversation)
    assert store.get(task.id).conversation_id is None
    assert store.list()[0].id == task.id
    assert store.events(task.id)[0]["payload"] == {"text": "Keep this too"}
    assert history.list_messages(conversation) == []


def test_parent_delete_cascades_to_children_and_events(store):
    parent = store.create("Parent")
    child = store.create("Child", parent_id=parent.id)
    store.append_event(child.id, "note", {})
    with connect(store.db_path) as connection:
        connection.execute("DELETE FROM tasks WHERE id = ?", (parent.id,))
        assert connection.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 0
    assert store.get(child.id) is None


def test_transition_and_event_are_rolled_back_together(store):
    task = store.create("Atomic status")
    # Exercise a real database failure after the state UPDATE, before commit.
    with connect(store.db_path) as connection:
        connection.executescript("""
            CREATE TRIGGER reject_event BEFORE INSERT ON task_events BEGIN
                SELECT RAISE(ABORT, 'event write failed');
            END;
        """)
    with pytest.raises(sqlite3.IntegrityError):
        store.transition(task.id, "running")
    assert store.get(task.id) == task
    assert store.events(task.id) == []


def test_unknown_ids_and_invalid_inputs(store):
    for operation in (
        lambda: store.transition("missing", "running"),
        lambda: store.set_plan("missing", []),
        lambda: store.update_step("missing", "step", "done"),
        lambda: store.append_event("missing", "note", {}),
        lambda: store.events("missing"),
        lambda: store.add_usage("missing"),
        lambda: store.release("missing"),
    ):
        with pytest.raises(KeyError):
            operation()
    task = store.create("Goal")
    with pytest.raises(InvalidTaskTransition):
        store.transition(task.id, "unknown")
    for operation in (
        lambda: store.create(" "), lambda: store.create("g" * 2001),
        lambda: store.list(status="unknown"), lambda: store.list(limit=0),
        lambda: store.append_event(task.id, "unknown", {}),
        lambda: store.events(task.id, after_seq=-1), lambda: store.events(task.id, limit=0),
        lambda: store.claim(datetime.now(UTC), 0),
    ):
        with pytest.raises(ValueError):
            operation()


def test_migration_is_additive_and_idempotent_on_existing_database(tmp_path):
    path = tmp_path / "legacy.db"
    task_start = SCHEMA.index("CREATE TABLE IF NOT EXISTS tasks (")
    task_end = SCHEMA.index("CREATE TABLE IF NOT EXISTS retrieval_cache (")
    legacy_schema = (SCHEMA[:task_start] + SCHEMA[task_end:]).replace("    task_id TEXT,\n", "")
    with sqlite3.connect(path) as connection:
        connection.executescript(legacy_schema)
        connection.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES ('c','Old','t','t')")
        connection.execute(
            "INSERT INTO agent_runs(id,conversation_id,mode,status,started_at) VALUES ('r','c','agent','done','t')"
        )
        connection.execute(
            "INSERT INTO tool_calls(id,run_id,tool_name,status,started_at) VALUES ('call','r','read','done','t')"
        )
    init_db(path)
    store = TaskStore(path)
    task = store.create("Survive repeated migration", conversation_id="c")
    store.append_event(task.id, "note", {"text": "persisted"})
    persisted = store.get(task.id)
    init_db(path)
    assert store.get(task.id) == persisted
    assert store.events(task.id)[0]["payload"] == {"text": "persisted"}
    with connect(path) as connection:
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(tool_calls)")]
        assert columns.count("task_id") == 1
        old_call = connection.execute("SELECT * FROM tool_calls WHERE id = 'call'").fetchone()
        assert old_call["tool_name"] == "read" and old_call["task_id"] is None
        connection.execute("UPDATE tool_calls SET task_id = ? WHERE id = 'call'", (task.id,))
        indexes = {row["name"] for row in connection.execute("PRAGMA index_list(tasks)")}
        assert {"idx_tasks_status", "idx_tasks_conversation"} <= indexes
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(api.router)
    # Exercise the actual settings dependency and store factory, with no
    # production paths or application lifespan workers.
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(sqlite_db_path=store.db_path)
    with TestClient(app) as client:
        yield client


def test_api_happy_paths(client, store):
    conversation = ChatHistory(store.db_path).ensure_conversation(None, "API tasks")
    response = client.post("/tasks", json={
        "goal": "  Research a topic  ", "conversation_id": conversation, "budget": {"max_steps": 3},
    })
    assert response.status_code == 201
    task = response.json()
    assert task["goal"] == "Research a topic" and task["status"] == "planning"
    assert task["conversation_id"] == conversation and task["budget"]["max_steps"] == 3
    path = f"/tasks/{task['id']}"
    assert client.get(path).json() == {**task, "events": []}
    assert client.get("/tasks", params={"conversation_id": conversation, "status": "planning"}).json() == {
        "tasks": [task],
    }
    second = client.post("/tasks", json={"goal": "Second"}).json()
    assert client.get("/tasks", params={"limit": 1}).json() == {"tasks": [second]}
    store.transition(task["id"], "running")
    assert client.post(path + "/pause").json()["status"] == "paused"
    assert client.post(path + "/resume").json()["status"] == "running"
    assert client.post(path + "/cancel").json()["status"] == "cancelled"
    events = client.get(path).json()["events"]
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert [event["payload"]["to"] for event in events] == ["running", "paused", "running", "cancelled"]


def test_task_detail_returns_all_events_not_just_first_page(client, store):
    task = store.create("Long event history")
    for i in range(505):
        store.append_event(task.id, "note", {"index": i})
    assert len(store.events(task.id)) == 500
    events = client.get(f"/tasks/{task.id}").json()["events"]
    assert [event["seq"] for event in events] == list(range(1, 506))


@pytest.mark.parametrize("action", ["", "/cancel", "/pause", "/resume"])
def test_api_unknown_id(client, action):
    response = client.post("/tasks/missing" + action) if action else client.get("/tasks/missing")
    assert response.status_code == 404


@pytest.mark.parametrize(("status", "action"), [
    ("planning", "pause"), ("planning", "resume"), ("running", "resume"),
    ("waiting_user", "resume"), ("waiting_approval", "resume"),
    ("paused", "pause"), ("done", "cancel"), ("failed", "resume"), ("cancelled", "cancel"),
])
def test_api_conflicts(client, store, status, action):
    task = task_at(store, status)
    assert client.post(f"/tasks/{task.id}/{action}").status_code == 409
    assert store.get(task.id) == task


@pytest.mark.parametrize("body", [
    {}, {"goal": ""}, {"goal": " \n "}, {"goal": "a" * 2001}, {"goal": 42},
    {"goal": "Goal", "budget": {"max_steps": 0}},
    {"goal": "Goal", "budget": {"max_steps": -2}},
    {"goal": "Goal", "budget": {"max_tool_calls": True}},
    {"goal": "Goal", "budget": {"max_seconds": 1.5}},
    {"goal": "Goal", "budget": {"max_seconds": "5"}},
    {"goal": "Goal", "budget": {"max_seconds": None}},
    {"goal": "Goal", "budget": {"typo": 2}},
    {"goal": "Goal", "budget": []},
    {"goal": "Goal", "conversation_id": "missing"},
])
def test_api_create_validation(client, body):
    assert client.post("/tasks", json=body).status_code == 422


@pytest.mark.parametrize("query", ["status=unknown", "limit=0", "limit=-1", "limit=abc"])
def test_api_list_validation(client, query):
    assert client.get("/tasks?" + query).status_code == 422


def test_tasks_router_registered_in_main(tmp_path, monkeypatch):
    from app import main

    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    app = main.create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    # No context manager: exercising routes must not start lifespan workers.
    client = TestClient(app)
    try:
        assert client.post("/tasks", json={"goal": "Registered route"}).status_code == 201
        assert len(client.get("/tasks").json()["tasks"]) == 1
    finally:
        client.close()
