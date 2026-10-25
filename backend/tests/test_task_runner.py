"""Offline task execution with scripted LLM replies and real persistent state."""

import asyncio
import copy
from contextlib import aclosing
import json
from types import SimpleNamespace

import pytest

from app.agents.task_runner import plan_goal, run_task
from app.agents.tool_loop import ApprovalBroker
from app.api.agent.assistant import _stream_assistant_tool_turn
from app.core.config import Settings
from app.db.sqlite import init_db
from app.llm.openai_client import ToolCall, ToolTurn
from app.services.chat_history import ChatHistory
from app.services.tasks import TaskStore
from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult, ToolSource

MODEL = "cx/gpt-6.1-sol"


def steps(*ids):
    return [{"id": id, "title": f"Do {id}", "tools_hint": ["read"], "done_when": f"{id} verified"} for id in ids]


def status(state="done", note="verified", question=None):
    return ToolTurn(model=MODEL, content=json.dumps({"step_status": state, "note": note, "question": question}))


def call(name="read", id="c1", arguments=None):
    return ToolTurn(model=MODEL, content="", tool_calls=(ToolCall(
        id=id, name=name, arguments=json.dumps(arguments or {"path": "input.txt"}),
    ),))


class Client:
    def __init__(self, chats=(), turns=()):
        self.chats, self.turns = list(chats), list(turns)
        self.seen, self.tool_seen = [], []

    async def chat(self, **kwargs):
        assert kwargs["model"] == MODEL
        self.seen.append(copy.deepcopy(kwargs))
        response = self.chats.pop(0)
        return SimpleNamespace(message=response if isinstance(response, str) else json.dumps(response))

    async def chat_with_tools(self, **kwargs):
        assert kwargs["model"] == MODEL
        self.tool_seen.append(copy.deepcopy(kwargs))
        return self.turns.pop(0)


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path)
    init_db(settings.sqlite_db_path)
    store = TaskStore(settings.sqlite_db_path)
    conversation = ChatHistory(store.db_path).ensure_conversation(None, "Tasks")
    registry, calls = ToolRegistry(), []

    async def read(args, context):
        calls.append(args)
        return ToolResult(content="Private full output " * 1000, summary="Read input", untrusted=True,
                          sources=[ToolSource(title=args["path"], path=str(tmp_path / args["path"]))])

    async def write(args, context):
        calls.append({"write": args})
        return ToolResult(content="saved", summary="Saved report", sources=[ToolSource(title="Report", path="report.md")])

    registry.register(Tool("read", "Read a file", {"type": "object"}, read))
    registry.register(Tool("write", "Write a file", {"type": "object"}, write, side_effects=True))
    return SimpleNamespace(store=store, settings=settings, registry=registry, calls=calls,
                           context=ToolContext("run", conversation, tmp_path), history=[{"role": "user", "content": "Earlier"}])


def runner(env, client, plan=None, **kwargs):
    return run_task(client=client, model=MODEL, goal="Read, check and write a report", plan=plan or steps("s1", "s2"),
                    history=env.history, registry=env.registry, tool_names=env.registry.names(), context=env.context,
                    language="English", approvals=kwargs.pop("approvals", ApprovalBroker()), store=env.store, **kwargs)


async def collect(events):
    return [event async for event in events]


def task(env):
    return env.store.list()[0]


@pytest.mark.parametrize("reply", ["not JSON", "[]", {"needs_plan": False},
    {"needs_plan": "true", "steps": steps("a", "b")}, {"needs_plan": True, "steps": steps("a")},
    {"needs_plan": True, "steps": steps("a", "a")}, {"needs_plan": True, "steps": [None, None]},
    {"needs_plan": True, "steps": steps(*map(str, range(9)))},
])
def test_defensive_triage(reply):
    client = Client([reply])
    result = asyncio.run(plan_goal(client, MODEL, "mấy giờ rồi?", [], ["read: Read a file"], "Vietnamese"))
    assert result == {"needs_plan": False, "steps": []}
    assert "3 actions" in client.seen[0]["messages"][0]["content"]


def test_planned_steps_events_usage_report_and_sources(env):
    client = Client(["Done: input.txt and other.txt [W2]."], [call(), status(), call(id="c2", arguments={"path": "other.txt"}), status()])
    events = asyncio.run(collect(runner(env, client)))
    saved = task(env)
    assert saved.status == "done" and [s["state"] for s in saved.plan] == ["done", "done"]
    assert saved.usage["steps"] == 2 and saved.usage["tool_calls"] == 2 and saved.usage["seconds"] > 0
    records = env.store.events(saved.id)
    assert [e["payload"]["step_id"] for e in records if e["kind"] == "step_started"] == ["s1", "s2"]
    assert len([e for e in records if e["kind"] == "tool_result"]) == 2
    assert "Private full output" not in json.dumps(records)
    assert len([e for e in records if e["kind"] == "step_done" and e["payload"].get("note") == "verified"]) == 2
    streamed = [d for n, d in events if n == "task.event"]
    assert [d["seq"] for d in streamed] == [e["seq"] for e in records]
    assert streamed == [{k: e[k] for k in ("task_id", "seq", "kind", "payload")} for e in records]
    answers = [d for n, d in events if n == "answer"]
    assert len(answers) == 1 and answers[0]["text"].startswith("Done:")
    assert [s["id"] for s in answers[0]["sources"]] == ["W1", "W2"]
    assert all(item["messages"][1] == env.history[0] for item in client.tool_seen)
    last_step = json.loads(client.tool_seen[2]["messages"][-1]["content"].split("\nFinish only")[0])
    assert last_step["plan"][0]["state"] == "done" and last_step["finished_step_notes"][0]["note"] == "verified"
    report = json.loads(client.seen[-1]["messages"][-1]["content"])
    assert len(report["tool_events"]) == 4 and report["tool_events"][-1]["payload"]["sources"][0]["path"].endswith("other.txt")


def test_blocked_replans_remaining_work_preserving_finished_steps(env):
    client = Client([{"steps": steps("r1")}, "Completed revised work"], [status(), status("blocked", "Missing input"), status()])
    events = asyncio.run(collect(runner(env, client)))
    saved = task(env)
    assert saved.status == "done"
    assert [(s["id"], s["state"]) for s in saved.plan] == [("s1", "done"), ("r1", "done")]
    edits = [d for n, d in events if n == "task.event" and d["kind"] == "plan_edit"]
    assert edits[0]["payload"]["plan"][0]["state"] == "done"
    assert "Missing input" in client.seen[0]["messages"][-1]["content"]


def test_replan_cap_fails_step_and_reports_unfinished_work(env):
    client = Client([{"steps": steps("r1")}, {"steps": steps("r2")}, "Not done: missing input"],
                    [status("blocked", "missing input") for _ in range(3)])
    events = asyncio.run(collect(runner(env, client)))
    assert task(env).status == "failed" and task(env).plan[0]["state"] == "failed"
    assert len(client.seen) == 3 and task(env).usage["steps"] == 3
    report = json.loads(client.seen[-1]["messages"][-1]["content"])
    assert report["plan"][0]["state"] == "failed" and "missing input" in json.dumps(report)
    assert "Not done" in dict(events)["answer"]["text"]


@pytest.mark.parametrize("replacement", ["bad json", {"steps": []}, {"steps": steps("s1")}])
def test_invalid_replan_keeps_finished_steps_and_reports(env, replacement):
    client = Client([replacement, "Blocked"], [status(), status("blocked", "missing")])
    asyncio.run(collect(runner(env, client)))
    assert [(s["id"], s["state"]) for s in task(env).plan] == [("s1", "done"), ("s2", "failed")]
    assert task(env).status == "done"  # at least one step succeeded


def test_need_user_waits_and_streams_only_question(env):
    client = Client([], [status("need_user", "Need location", "Which folder?")])
    events = asyncio.run(collect(runner(env, client)))
    assert task(env).status == "waiting_user" and not client.seen
    assert dict(events)["answer"]["text"] == "Which folder?"
    assert [e["payload"]["question"] for e in env.store.events(task(env).id) if e["kind"] == "ask"] == ["Which folder?"]


@pytest.mark.parametrize("answer", ["Completed without status", '{"step_status":"wrong"}', '[]',
    '{"step_status":"need_user","note":"missing","question":null}', '{"step_status":"done","note":42}'])
def test_missing_or_invalid_status_becomes_note(env, answer, caplog):
    client = Client(["Report"], [ToolTurn(model=MODEL, content=answer), status()])
    asyncio.run(collect(runner(env, client)))
    assert task(env).status == "done"
    assert any(e["kind"] == "note" and e["payload"]["note"] == answer for e in env.store.events(task(env).id))
    assert "no valid status JSON" in caplog.text


@pytest.mark.parametrize("budget,turns,executed", [
    ({"max_steps": 1}, [status(), status()], 0),
    ({"max_tool_calls": 1}, [call(id="a"), call(id="b")], 1),
])
def test_budget_waits_and_reports(env, budget, turns, executed):
    client = Client(["Stopped: budget exceeded"], turns)
    events = asyncio.run(collect(runner(env, client, budget=budget)))
    assert task(env).status == "waiting_user" and len(env.calls) == executed
    assert any(n == "task.event" and d["kind"] == "budget" for n, d in events)
    assert "budget" in dict(events)["answer"]["text"]


@pytest.mark.parametrize("max_steps", [1, 2, 3])
def test_step_budget_limits_side_effects_and_preserves_completed_steps(env, max_steps):
    async def save(args, context):
        (context.workspace / args["path"]).write_text("saved")
        return ToolResult(content="saved", summary=f"Saved {args['path']}")

    env.registry.register(Tool("save", "Save a file", {"type": "object"}, save, side_effects=True))
    turns = []
    for index in range(1, 4):
        turns.extend([call("save", f"c{index}", {"path": f"output{index}.txt"}),
                      status(note=f"Saved output{index}.txt")])
    client = Client(["Final report"], turns)
    events = asyncio.run(collect(runner(env, client, plan=steps("s1", "s2", "s3"),
                                        budget={"max_steps": max_steps})))
    saved = task(env)
    assert saved.status == ("waiting_user" if max_steps < 3 else "done")
    assert saved.usage["steps"] == saved.usage["tool_calls"] == max_steps
    assert [step["state"] for step in saved.plan] == ["done"] * max_steps + ["pending"] * (3 - max_steps)
    assert [(env.context.workspace / f"output{index}.txt").exists() for index in range(1, 4)] == [
        index <= max_steps for index in range(1, 4)
    ]
    records = env.store.events(saved.id)
    assert [e["payload"]["step_id"] for e in records if e["kind"] == "step_started"] == [
        f"s{index}" for index in range(1, max_steps + 1)
    ]
    report = json.loads(client.seen[-1]["messages"][-1]["content"])
    assert report["plan"] == saved.plan
    assert [note["note"] for note in report["notes"]] == [
        f"Saved output{index}.txt" for index in range(1, max_steps + 1)
    ]
    assert len([e for e in report["tool_events"] if e["kind"] == "tool_result"]) == max_steps
    assert len([d for n, d in events if n == "task.event" and d["kind"] == "budget"]) == (max_steps < 3)
    assert dict(events)["task.updated"]["task"] == saved.to_dict()


@pytest.mark.parametrize("step_status", ["done", "blocked", "need_user"])
def test_budget_after_received_result_preserves_status_and_note(env, monkeypatch, step_status):
    from app.agents import task_runner

    now = [0.0]
    monkeypatch.setattr(task_runner.time, "monotonic", lambda: now[0])
    original_loop = task_runner.run_tool_loop

    async def delayed_cleanup(**kwargs):
        async with aclosing(original_loop(**kwargs)) as loop:
            async for name, data in loop:
                yield name, data
                if name == "answer":
                    now[0] = 2.0

    monkeypatch.setattr(task_runner, "run_tool_loop", delayed_cleanup)
    client = Client(["Stopped on elapsed budget"], [call(), status(step_status, "Step evidence", "Which folder?")])
    events = asyncio.run(collect(runner(env, client, budget={"max_seconds": 1})))
    saved = task(env)
    assert saved.status == "waiting_user" and saved.usage["steps"] == 1
    assert [step["state"] for step in saved.plan] == ["done" if step_status == "done" else "running", "pending"]
    assert len(env.calls) == 1 and len(client.seen) == 1
    report = json.loads(client.seen[-1]["messages"][-1]["content"])
    assert report["notes"] == [{"step_id": "s1", "step_status": step_status,
                                "note": "Step evidence", "question": "Which folder?"}]
    assert report["plan"] == saved.plan
    assert any(n == "task.event" and d["kind"] == "note" and d["payload"] == report["notes"][0]
               for n, d in events)


def test_cancel_between_steps_stops_without_report(env):
    client = Client([], [status()])

    async def main():
        events = []
        async for n, d in runner(env, client):
            events.append((n, d))
            if (n == "task.updated" and d["task"]["status"] == "running"
                    and d["task"]["plan"][0]["state"] == "done"):
                env.store.transition(d["task"]["id"], "cancelled")
        return events

    events = asyncio.run(main())
    assert task(env).status == "cancelled" and len(client.tool_seen) == 1
    assert any(n == "task.event" and d["payload"].get("to") == "cancelled" for n, d in events)


def test_cancel_stops_before_next_tool_in_same_model_reply(env):
    turn = call()
    turn = ToolTurn(model=MODEL, content="", tool_calls=turn.tool_calls + call(id="c2").tool_calls)
    client = Client([], [turn])

    async def main():
        async for n, d in runner(env, client):
            if n == "tool.call.completed":
                env.store.transition(task(env).id, "cancelled")

    asyncio.run(main())
    assert len(env.calls) == 1 and task(env).status == "cancelled"


def test_disconnect_closes_and_records_cancel(env):
    client = Client([], [call()])

    async def main():
        async with aclosing(runner(env, client)) as events:
            async for n, d in events:
                if n == "tool.call.started":
                    break

    asyncio.run(main())
    assert task(env).status == "cancelled" and env.calls == []


def test_poll_disconnect_before_tool(env):
    disconnected = False

    async def check():
        return disconnected

    async def main():
        nonlocal disconnected
        async for n, d in runner(env, Client([], [call()]), is_disconnected=check):
            if n == "tool.call.started":
                disconnected = True

    asyncio.run(main())
    assert task(env).status == "cancelled" and env.calls == []


def test_approval_across_steps_and_source_trust_are_preserved(env):
    broker = ApprovalBroker()
    client = Client(["Read and saved report.md"], [call(), status(), call("write", "w"), status()])

    async def main():
        events = []
        async for n, d in runner(env, client, approvals=broker):
            events.append((n, d))
            if n == "tool.call.approval_required":
                assert broker.pending("run") == ["w"]
                assert broker.decide("run", "w", True)
        return events

    events = asyncio.run(main())
    assert [d["name"] for n, d in events if n == "tool.call.approval_required"] == ["write"]
    assert len(env.calls) == 2 and broker.pending("run") == [] and task(env).status == "done"


def test_cancel_while_waiting_approval_cleans_broker(env):
    broker = ApprovalBroker()
    client = Client([], [call(), status(), call("write", "w")])

    async def main():
        async for n, d in runner(env, client, approvals=broker):
            if n == "tool.call.approval_required":
                env.store.transition(task(env).id, "cancelled")

    asyncio.run(main())
    assert len(env.calls) == 1 and broker.pending("run") == [] and task(env).status == "cancelled"


@pytest.mark.parametrize("planned", [False, True])
def test_route_single_turn_or_planned_sse(env, monkeypatch, planned):
    from app import tools
    from app.api.agent import deps

    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    monkeypatch.setattr(deps, "_retrieval_store", lambda settings: None)
    monkeypatch.setattr(deps, "_embedding_provider", lambda settings: None)
    client = Client([{"needs_plan": planned, "steps": steps("s1", "s2")}] + (["Final report [W1]"] if planned else []),
                    [call(), status(), status()] if planned else [ToolTurn(model=MODEL, content="It is noon.")])
    chunks, citations, audits = [], [], []
    events = asyncio.run(collect(_stream_assistant_tool_turn(
        service=SimpleNamespace(client=client), settings=env.settings, rag=None,
        request=SimpleNamespace(task="Create a file report" if planned else "mấy giờ rồi?"),
        run_id="run", conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
        assistant_chunks=chunks, assistant_citations=citations,
        run_store=SimpleNamespace(record_tool_call=lambda **kwargs: audits.append(kwargs)),
    )))
    text = "".join(events)
    assert "message.delta" in text and "step_status" not in "".join(chunks)
    if planned:
        assert len(env.store.list()) == 1 and task(env).status == "done"
        assert "task.created" in text and "task.updated" in text and "task.event" in text
        assert "".join(chunks) == "Final report [W1]" and audits[0]["task_id"] == task(env).id
        assert citations[0]["filename"].endswith("input.txt")
    else:
        assert env.store.list() == [] and "task.created" not in text and "".join(chunks) == "It is noon."


def test_elapsed_budget_stops_before_tool_execution(env, monkeypatch):
    from app.agents import task_runner

    now = [0.0]
    monkeypatch.setattr(task_runner.time, "monotonic", lambda: now[0])

    async def main():
        events = []
        async for n, d in runner(env, Client(["Stopped on elapsed budget"], [call()]), budget={"max_seconds": 1}):
            events.append((n, d))
            if n == "tool.call.started":
                now[0] = 2.0
        return events

    events = asyncio.run(main())
    assert task(env).status == "waiting_user" and task(env).usage["seconds"] == 2
    assert env.calls == [] and any(n == "task.event" and d["kind"] == "budget" for n, d in events)


def test_cancelling_active_approval_wait_stops_promptly(env):
    broker = ApprovalBroker()
    client = Client([], [call(), status(), call("write", "w")])

    async def main():
        started = asyncio.Event()

        async def cancel():
            await started.wait()
            await asyncio.sleep(0.02)
            env.store.transition(task(env).id, "cancelled")

        cancellation = asyncio.create_task(cancel())
        try:
            async for n, d in runner(env, client, approvals=broker):
                if n == "tool.call.approval_required":
                    started.set()
        finally:
            await cancellation

    async def bounded():
        await asyncio.wait_for(main(), timeout=2)

    asyncio.run(bounded())
    assert task(env).status == "cancelled" and len(env.calls) == 1 and broker.pending("run") == []


def test_route_persists_tool_call_task_link(env, monkeypatch):
    from app import tools
    from app.api.agent import deps
    from app.db.sqlite import connect
    from app.services.agent_run_store import AgentRunStore

    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    monkeypatch.setattr(deps, "_retrieval_store", lambda settings: None)
    monkeypatch.setattr(deps, "_embedding_provider", lambda settings: None)
    run_store = AgentRunStore(env.store.db_path)
    run = run_store.create_run(conversation_id=env.context.conversation_id, user_message_id=None, mode="agent")
    client = Client([{"needs_plan": True, "steps": steps("s1", "s2")}, "Final report"], [call(), status(), status()])
    asyncio.run(collect(_stream_assistant_tool_turn(
        service=SimpleNamespace(client=client), settings=env.settings, rag=None,
        request=SimpleNamespace(task="Create a file report"), run_id=run.id,
        conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
        assistant_chunks=[], assistant_citations=[], run_store=run_store,
    )))
    with connect(env.store.db_path) as connection:
        rows = connection.execute("SELECT task_id, run_id FROM tool_calls").fetchall()
    assert len(rows) == 1 and rows[0]["task_id"] == task(env).id and rows[0]["run_id"] == run.id


def test_api_cancellation_propagates_to_assistant_turn(env, monkeypatch):
    from app import tools
    from app.api.agent import deps

    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    monkeypatch.setattr(deps, "_retrieval_store", lambda settings: None)
    monkeypatch.setattr(deps, "_embedding_provider", lambda settings: None)
    client = Client([{"needs_plan": True, "steps": steps("s1", "s2")}])

    async def main():
        async for event in _stream_assistant_tool_turn(
            service=SimpleNamespace(client=client), settings=env.settings, rag=None,
            request=SimpleNamespace(task="Create a file report"), run_id="run",
            conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
            assistant_chunks=[], assistant_citations=[], run_store=None,
        ):
            if "event: task.created" in event:
                env.store.transition(task(env).id, "cancelled")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main())
    assert task(env).status == "cancelled" and client.tool_seen == []


def test_replan_time_counts_towards_task_budget(env, monkeypatch):
    from app.agents import task_runner

    now = [0.0]
    monkeypatch.setattr(task_runner.time, "monotonic", lambda: now[0])

    class SlowReplan(Client):
        async def chat(self, **kwargs):
            if "Replan the remaining" in kwargs["messages"][0]["content"]:
                now[0] = 3.0
            return await super().chat(**kwargs)

    client = SlowReplan([{"steps": steps("r1")}, "Stopped on elapsed budget"], [status("blocked", "missing")])
    events = asyncio.run(collect(runner(env, client, budget={"max_seconds": 2})))
    assert task(env).status == "waiting_user" and task(env).usage["seconds"] == 3
    assert len(client.tool_seen) == 1 and any(n == "task.event" and d["kind"] == "budget" for n, d in events)


@pytest.mark.parametrize("report_seconds,max_steps,expected_status", [
    (5, 20, "done"),
    (10, 20, "done"),
    (20, 20, "waiting_user"),
    (20, 1, "waiting_user"),
])
def test_report_time_counts_towards_task_budget(env, monkeypatch, report_seconds, max_steps, expected_status):
    from app.agents import task_runner

    now = [0.0]
    monkeypatch.setattr(task_runner.time, "monotonic", lambda: now[0])

    class SlowReport(Client):
        async def chat(self, **kwargs):
            now[0] += report_seconds
            return await super().chat(**kwargs)

    client = SlowReport(["Final report"], [status(), status()])
    events = asyncio.run(collect(runner(env, client, budget={"max_seconds": 10, "max_steps": max_steps})))
    saved = task(env)
    assert saved.status == expected_status and saved.usage["seconds"] == report_seconds
    assert len(client.seen) == 1 and dict(events)["answer"]["text"] == "Final report"
    assert dict(events)["task.updated"]["task"] == saved.to_dict()
    budget_events = [d for n, d in events if n == "task.event" and d["kind"] == "budget"]
    assert len(budget_events) == (1 if expected_status == "waiting_user" else 0)
    if report_seconds > 10 and max_steps == 20:
        assert budget_events[0]["payload"] == {"usage": saved.usage, "budget": saved.budget}
    if expected_status == "waiting_user":
        assert not any(n == "task.event" and d["kind"] == "status" and d["payload"].get("to") == "done"
                       for n, d in events)
