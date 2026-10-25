"""Offline disconnect/restart/approval coverage with real SQLite and fake tools."""

import asyncio
from contextlib import aclosing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.agents.task_runner import execute_task, start_task
from app.api.agent.assistant import _stream_assistant_tool_turn
from app.api.agent.routes import router as agent_router
from app.api.tasks import router as task_router
from app.core.config import get_settings
from app.db.sqlite import connect
from app.services.chat_history import ChatHistory
from app.services.reminders import ReminderStore
from app.services.task_events import TaskEventHub, watch_task
from app.services.task_worker import TaskWorker, report_message_id
from app.services.tasks import TaskStore
from app.tools.registry import Tool, ToolResult
from test_task_runner import Client, MODEL, call, collect, env, status, steps, task_deps


def start(env, client, plan=None, **kwargs):
    return start_task(goal="Read, verify and save", plan=plan or steps("s1", "s2"), deps=task_deps(env, client),
                      conversation_id=env.context.conversation_id, history=env.history, run_id="run", **kwargs)


async def drain(worker):
    results = await asyncio.gather(*list(worker.executing.values()), return_exceptions=True)
    assert all(result is None or isinstance(result, asyncio.CancelledError) for result in results)


def test_chat_disconnect_mid_step_worker_finishes_and_delivers_once(env, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    client = Client([{"needs_plan": True, "steps": steps("s1", "s2", "s3")}, "Saved report [W1]."],
                    [call(), status(), status(), status()])
    delivered = []

    async def deliver(reminder):
        delivered.append(reminder.text)

    async def main():
        entered, finish = asyncio.Event(), asyncio.Event()
        original = env.registry.get("read").handler

        async def slow_tool(args, context):
            entered.set()
            await finish.wait()
            return await original(args, context)

        env.registry._tools["read"] = replace(env.registry.get("read"), handler=slow_tool)
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client), deliverers=[deliver])
        stream = _stream_assistant_tool_turn(
            service=SimpleNamespace(client=client), settings=env.settings, rag=None,
            request=SimpleNamespace(task="Read, verify and save"), run_id="run",
            conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
            assistant_chunks=[], assistant_citations=[], run_store=None,
        )
        async with aclosing(stream):
            assert "task.created" in await anext(stream)
            task_id = env.store.list()[0].id
            await worker.tick()
            await entered.wait()
        assert env.store.get(task_id).status == "running"
        finish.set()
        await drain(worker)
        assert env.store.get(task_id).status == "done"
        # Simulate the outer chat save and another worker starting after delivery.
        ChatHistory(env.store.db_path).save_message(
            conversation_id=env.context.conversation_id, role="assistant", content="Saved report [W1].",
            message_id=report_message_id(task_id),
        )
        await TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client)).tick()
        return task_id

    task_id = asyncio.run(asyncio.wait_for(main(), 2))
    messages = ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)
    assert len(messages) == 1 and messages[0].content == "Saved report [W1]."
    assert messages[0].citations[0]["id"] == "W1"
    notifications = ReminderStore(env.store.db_path).unread()
    assert len(notifications) == 1 and notifications[0]["kind"] == "task"
    assert delivered == ["Saved report [W1]."]
    assert env.store.get(task_id).lease_until is None


def test_stale_running_step_resumes_with_notes_tools_and_sources(env):
    client = Client(["Recovered report [W1]."], [status(), status()])
    task_id = start(env, client)
    env.store.update_step(task_id, "s1", "running")
    env.store.append_event(task_id, "note", {"step_id": "s1", "step_status": "blocked", "note": "Already saved output"})
    source = {"id": "W1", "title": "output", "url": "", "path": "output.txt"}
    env.store.append_event(task_id, "tool_call", {"step_id": "s1", "call_id": "before", "name": "write"})
    env.store.append_event(task_id, "tool_result", {"step_id": "s1", "call_id": "before", "ok": True,
                                                  "summary": "Saved output.txt", "sources": [source], "untrusted": True})
    stale = datetime.now(UTC) - timedelta(minutes=2)
    env.store.claim(stale, 60)

    async def main():
        worker = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client))
        await worker.tick()
        await drain(worker)

    asyncio.run(main())
    assert env.store.get(task_id).status == "done" and not env.calls
    prompt = client.tool_seen[0]["messages"][-1]["content"]
    assert "Already saved output" in prompt and "Saved output.txt" in prompt
    assert "before repeating any side effects" in prompt
    assert ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)[0].citations[0]["id"] == "W1"


def test_worker_shutdown_preserves_running_work_for_restart(env):
    client = Client(["Recovered"], [status(), status()])
    task_id = start(env, client)

    async def main():
        entered = asyncio.Event()
        original = client.chat_with_tools

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        client.chat_with_tools = blocked
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        await entered.wait()
        await worker.close()
        assert env.store.get(task_id).status == "running"
        assert env.store.get(task_id).lease_until is None
        client.chat_with_tools = original
        restarted = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client))
        await restarted.tick()
        await drain(restarted)

    asyncio.run(asyncio.wait_for(main(), 2))
    assert env.store.get(task_id).status == "done"


def test_two_workers_and_renewal_never_duplicate_execution(env):
    now = [datetime.now(UTC)]
    client = Client(["Finished"], [call(), status(), status()])
    task_id = start(env, client)

    async def main():
        entered, release, renewed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = env.registry.get("read").handler

        async def block(args, context):
            entered.set()
            await release.wait()
            return await original(args, context)

        env.registry._tools["read"] = replace(env.registry.get("read"), handler=block)
        first_store, second_store = TaskStore(env.store.db_path), TaskStore(env.store.db_path)
        original_renew = first_store.renew

        def renew(*args, **kwargs):
            result = original_renew(*args, **kwargs)
            renewed.set()
            return result

        first_store.renew = renew
        first = TaskWorker(first_store, lambda cid: task_deps(env, client), clock=lambda: now[0], renew_interval=0.01)
        second = TaskWorker(second_store, lambda cid: task_deps(env, client), clock=lambda: now[0])
        await first.tick()
        await entered.wait()
        await second.tick()
        assert not second.executing
        now[0] += timedelta(seconds=45)
        await renewed.wait()
        now[0] += timedelta(seconds=20)  # past the original lease
        await second.tick()
        assert not second.executing
        release.set()
        await drain(first)

    asyncio.run(asyncio.wait_for(main(), 2))
    assert len(env.calls) == 1 and env.store.get(task_id).status == "done"


@pytest.mark.parametrize("approved", [True, False])
def test_offline_approval_persists_restart_endpoint_decision_resumes_exact_call(env, approved):
    writes = []

    async def write(args, context):
        writes.append(args)
        return ToolResult(content="saved", summary="Saved")

    env.registry.register(Tool("guarded", "Guarded write", {"type": "object"}, write, permission="ask", side_effects=True))
    client = Client(["Final report"], [call("guarded", "approve", {"path": "chosen.txt"}), status(), status()])
    task_id = start(env, client)

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        await drain(worker)
        assert env.store.get(task_id).status == "waiting_approval" and writes == []
        assert ReminderStore(env.store.db_path).unread()[0]["kind"] == "task_approval"
        await worker.close()
        app = FastAPI()
        app.include_router(agent_router)
        app.dependency_overrides[get_settings] = lambda: env.settings
        with TestClient(app) as http:
            response = http.post("/agent/runs/run/tool-calls/approve/decision", json={"approved": approved})
            assert response.status_code == 200
            assert http.post("/agent/runs/run/tool-calls/approve/decision", json={"approved": True}).status_code == 404
        # A new executor/client receives the persisted call before asking the
        # model again. Its arguments and the user's denial cannot be lost.
        resumed = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client))
        await resumed.tick()
        await drain(resumed)
        await resumed.tick()
        await drain(resumed)

    asyncio.run(main())
    assert writes == ([{"path": "chosen.txt"}] if approved else [])
    assert env.store.get(task_id).status == "done"
    assert env.store.get(task_id).usage["tool_calls"] == 1
    assert len(ReminderStore(env.store.db_path).unread()) == 2


def test_approval_timeout_blocks_step_and_reports_without_tools(env):
    env.registry._tools["write"] = replace(env.registry.get("write"), permission="ask")
    client = Client(["Approval expired; nothing was written."], [call("write", "expired")])
    task_id = start(env, client)

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        await drain(worker)
        later = datetime.now(UTC) + timedelta(hours=24, seconds=1)
        restarted = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client), clock=lambda: later)
        await restarted.tick()
        await drain(restarted)

    asyncio.run(main())
    task = env.store.get(task_id)
    assert task.status == "failed" and task.plan[0]["state"] == "failed"
    assert env.calls == [] and len(client.tool_seen) == 1
    assert "timed out" in client.seen[-1]["messages"][-1]["content"]
    assert "Approval expired" in ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)[0].content


def test_attach_replays_then_follows_live_with_cursor(env):
    hub = TaskEventHub()
    client = Client(["Done"], [status(), status()])
    task_id = start(env, client)
    cursor = env.store.events(task_id)[-1]["seq"]
    replay_seq = env.store.append_event(task_id, "note", {"note": "replay me"})

    async def main():
        async with aclosing(watch_task(env.store, task_id, after_seq=cursor, hub=hub, interval=0.01)) as stream:
            assert await anext(stream) == ("task.event", {"task_id": task_id, "seq": replay_seq,
                                                         "kind": "note", "payload": {"note": "replay me"}})
            await anext(stream)  # snapshot
            worker = TaskWorker(env.store, lambda cid: task_deps(env, client), hub=hub)
            await worker.tick()
            events = await collect(stream)
            await drain(worker)
        return events

    events = asyncio.run(asyncio.wait_for(main(), 2))
    seqs = [d["seq"] for n, d in events if n == "task.event"]
    assert seqs == sorted(set(seqs)) and min(seqs) > replay_seq
    assert [d["text"] for n, d in events if n == "answer"] == ["Done"]
    assert any(n == "avatar.state" for n, d in events)
    assert not hub.attached(task_id)
    app = FastAPI()
    app.include_router(task_router)
    app.dependency_overrides[get_settings] = lambda: env.settings
    with TestClient(app) as http:
        response = http.get(f"/tasks/{task_id}/stream?after_seq={cursor}")
        assert response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
        assert '"note": "replay me"' in response.text and '"text": "Done"' in response.text
        assert http.get("/tasks/missing/stream").status_code == 404
        assert http.get(f"/tasks/{task_id}/stream?after_seq=-1").status_code == 422


def test_terminal_delivery_recovered_once_across_workers(env):
    client = Client(["Completed before crash"], [status(), status()])
    task_id = start(env, client)
    asyncio.run(collect(execute_task(task_id, task_deps(env, client))))
    assert not ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)
    delivered = []

    async def deliver(reminder):
        delivered.append(reminder.id)

    async def main():
        first = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client), deliverers=[deliver])
        second = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client), deliverers=[deliver])
        await asyncio.gather(first.tick(), second.tick())
        await asyncio.gather(drain(first), drain(second))
        await first.tick()
        await drain(first)

    asyncio.run(main())
    assert delivered == [task_id]
    assert len(ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)) == 1
    assert len(ReminderStore(env.store.db_path).unread()) == 1


def test_renew_and_release_cannot_overwrite_new_lease_owner(env):
    task_id = start(env, Client())
    now = datetime.now(UTC)
    first = env.store.claim(now, 60)[0]
    second = TaskStore(env.store.db_path).claim(now + timedelta(seconds=61), 60)[0]
    assert env.store.renew(task_id, 60, expected_lease=first.lease_until, now=now) is None
    env.store.release(task_id, expected_lease=first.lease_until)
    assert env.store.get(task_id).lease_until == second.lease_until
    for seconds in (0, -1, True):
        with pytest.raises(ValueError):
            env.store.renew(task_id, seconds)


def test_lost_lease_stops_before_tool_side_effects(env):
    now = [datetime.now(UTC)]
    client = Client([], [call()])
    task_id = start(env, client)

    async def main():
        model_entered, model_return = asyncio.Event(), asyncio.Event()
        original = client.chat_with_tools

        async def slow_model(**kwargs):
            model_entered.set()
            await model_return.wait()
            return await original(**kwargs)

        client.chat_with_tools = slow_model
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client), clock=lambda: now[0])
        await worker.tick()
        await model_entered.wait()
        now[0] += timedelta(seconds=61)
        replacement = TaskStore(env.store.db_path).claim(now[0], 60)[0]
        model_return.set()
        await drain(worker)
        assert env.store.get(task_id).lease_until == replacement.lease_until

    asyncio.run(asyncio.wait_for(main(), 2))
    assert env.calls == [] and env.store.get(task_id).status == "running"
    assert not any(e["kind"] == "answer" for e in env.store.events(task_id))


def test_budget_report_reaches_attached_stream_before_waiting_exit(env):
    client = Client(["Stopped at the step budget"], [status(), status()])
    task_id = start(env, client, budget={"max_steps": 1})

    async def main():
        reporting, release = asyncio.Event(), asyncio.Event()
        original = client.chat

        async def slow_report(**kwargs):
            reporting.set()
            await release.wait()
            return await original(**kwargs)

        client.chat = slow_report
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        watcher = asyncio.create_task(collect(watch_task(env.store, task_id, stop_when_waiting=True, interval=0.01)))
        await worker.tick()
        await reporting.wait()
        assert env.store.get(task_id).status == "waiting_user"
        assert not watcher.done()
        release.set()
        await drain(worker)
        return await watcher

    events = asyncio.run(asyncio.wait_for(main(), 2))
    assert [d["text"] for n, d in events if n == "answer"] == ["Stopped at the step budget"]
    assert env.store.get(task_id).status == "waiting_user"


def test_failed_executor_still_saves_report_and_notifies(env):
    class FailingClient(Client):
        async def chat_with_tools(self, **kwargs):
            raise RuntimeError("Provider unavailable")

    client = FailingClient()
    task_id = start(env, client)

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        await drain(worker)

    asyncio.run(main())
    assert env.store.get(task_id).status == "failed"
    assert "Provider unavailable" in ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)[0].content
    assert len(ReminderStore(env.store.db_path).unread()) == 1


def test_restored_untrusted_evidence_requires_new_approval(env):
    client = Client([], [call("write", "new-write")])
    task_id = start(env, client)
    env.store.update_step(task_id, "s1", "done")
    env.store.append_event(task_id, "tool_result", {"step_id": "s1", "call_id": "read", "ok": True,
                                                  "summary": "Untrusted data", "sources": [], "untrusted": True})

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        await drain(worker)

    asyncio.run(main())
    assert env.calls == [] and env.store.get(task_id).status == "waiting_approval"


def test_waiting_paused_and_live_leased_tasks_are_not_executed(env):
    client = Client()
    for state in ("waiting_user", "waiting_approval", "paused"):
        task_id = start(env, client)
        env.store.transition(task_id, state)
    active = start(env, client)
    env.store.claim(datetime.now(UTC), 60)

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        await worker.tick()
        assert worker.executing == {}

    asyncio.run(main())
    assert client.tool_seen == [] and env.store.get(active).status == "running"


def test_legacy_terminal_task_without_execution_note_is_not_redelivered(env):
    client = Client(["Foreground report"], [status(), status()])
    task_id = start(env, client)
    asyncio.run(collect(execute_task(task_id, task_deps(env, client))))
    with connect(env.store.db_path) as connection:
        connection.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'note' "
                           "AND json_extract(payload_json, '$.execution') IS NOT NULL", (task_id,))
    delivered = []

    async def deliver(reminder):
        delivered.append(reminder.id)

    async def main():
        worker = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client), deliverers=[deliver])
        await worker.tick()
        await drain(worker)

    asyncio.run(main())
    assert delivered == []
    assert not ChatHistory(env.store.db_path).list_messages(env.context.conversation_id)
    assert not ReminderStore(env.store.db_path).unread()


def test_disconnected_watcher_cancels_chat_run_while_task_idles(env, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    client = Client([{"needs_plan": True, "steps": steps("s1", "s2")}])

    async def gone():
        return True

    async def main():
        stream = _stream_assistant_tool_turn(
            service=SimpleNamespace(client=client), settings=env.settings, rag=None,
            request=SimpleNamespace(task="Read, verify and save"), run_id="run",
            conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
            assistant_chunks=[], assistant_citations=[], run_store=None, is_disconnected=gone,
        )
        async with aclosing(stream):
            async for _ in stream:
                pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(asyncio.wait_for(main(), 2))
    assert env.store.list()[0].status == "running"
