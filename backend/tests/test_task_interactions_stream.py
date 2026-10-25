"""A reply to a task waiting on a question resumes that task through the runner, offline."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.api.agent.assistant import _stream_assistant_tool_turn
from app.services.artifacts import ArtifactStore
from app.services.chat_history import ChatHistory
from app.services.task_interactions import resume_waiting_task
from app.services.task_worker import TaskWorker
from app.services.tasks import TaskStore
from app.tools.assistant_core import ASSISTANT_TOOLS
from test_agent_stream_e2e import harness, _parse_sse  # shared isolated HTTP harness
from test_task_runner import Client, MODEL, call, env, status, steps, task_deps
from test_task_worker import drain, start

ASK = {"question": "Which format?", "options": ["PDF", "Text"]}


class Triage:
    """The router model that decides whether a reply answers the waiting task."""

    def __init__(self, reply):
        self.reply, self.seen = reply, []

    async def chat(self, **kwargs):
        self.seen.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(message=json.dumps(self.reply))


def with_ask_tools(env):
    for tool in ASSISTANT_TOOLS:
        if tool.name in {"ask_user", "read_artifact"}:
            env.registry.register(tool)


def run_until_waiting_then_reply(env, client, reply, text, *, restart=False, plan=None):
    """Run the task to its question, answer it, and let a worker finish it."""
    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        task_id = start(env, client, plan or steps("s1", "s2"))
        await worker.tick()
        await drain(worker)
        assert env.store.get(task_id).status == "waiting_user"
        resumed = await resume_waiting_task(store=env.store, conversation_id=env.context.conversation_id,
                                            text=text, client=Triage(reply), model="router")
        if restart:
            # A new process: nothing in memory survives, only SQLite.
            worker = TaskWorker(TaskStore(env.store.db_path), lambda cid: task_deps(env, client))
        if resumed is not None:
            assert resumed.status == "running"
            await worker.tick()
            await drain(worker)
        return task_id, resumed

    return asyncio.run(asyncio.wait_for(main(), 5))


@pytest.mark.parametrize("restart", [False, True])
def test_reply_resumes_the_same_step_without_repeating_earlier_tools(env, restart):
    with_ask_tools(env)
    client = Client(["Final report [W1]."], [
        call(), call("ask_user", "ask", ASK),  # s1: read, then ask
        status(), status(),                     # s1 resumed, s2
    ])
    task_id, resumed = run_until_waiting_then_reply(env, client, {"resume": True}, "PDF", restart=restart)
    assert resumed is not None
    task = env.store.get(task_id)
    assert task.status == "done" and [step["state"] for step in task.plan] == ["done", "done"]
    assert task.lease_until is None
    events = env.store.events(task_id, limit=None)
    ask = next(e for e in events if e["kind"] == "ask")
    answer = next(e for e in events if e["kind"] == "answer" and "ask_seq" in e["payload"])
    assert ask["payload"]["step_id"] == "s1" and ask["payload"]["options"] == ["PDF", "Text"]
    assert answer["payload"] == {"text": "PDF", "ask_seq": ask["seq"]}
    # The read ran once; its full text is in the artifact store and the handle is on the event.
    assert env.calls == [{"path": "input.txt"}]
    saved = next(e["payload"] for e in events if e["kind"] == "tool_result" and e["payload"]["name"] == "read")
    assert saved["artifact"] and "Private full output" in ArtifactStore(env.context.workspace).read(saved["artifact"])
    # The retried step sees the question, the answer and that evidence.
    retry = client.tool_seen[2]["messages"][-1]["content"]
    assert '"user_answer": "PDF"' in retry and "Which format?" in retry and saved["artifact"] in retry
    assert len(client.tool_seen) == 4


def test_need_user_step_resumes_and_triage_failure_takes_a_free_text_answer(env):
    client = Client(["Final report."], [status("need_user", "Needs the file name", "Which file?"),
                                       status(), status()])
    task_id, resumed = run_until_waiting_then_reply(
        env, client, RuntimeError("router unavailable"), "report.md", plan=steps("s1", "s2"))
    assert resumed is not None and env.store.get(task_id).status == "done"
    assert '"user_answer": "report.md"' in client.tool_seen[1]["messages"][-1]["content"]


def test_new_request_leaves_the_waiting_task_alone(env):
    with_ask_tools(env)
    client = Client([], [call("ask_user", "ask", ASK)])
    task_id, resumed = run_until_waiting_then_reply(env, client, {"resume": False}, "Write me a poem")
    assert resumed is None
    task = env.store.get(task_id)
    assert task.status == "waiting_user" and task.plan[0]["state"] == "running"
    assert not any(e["kind"] == "answer" for e in env.store.events(task_id, limit=None))


def test_chat_stream_attaches_to_the_resumed_task_and_returns_its_report(env, monkeypatch):
    from app import tools

    with_ask_tools(env)
    monkeypatch.setattr(tools, "build_registry", lambda **kwargs: env.registry)
    client = Client(["Final report [W1]."], [call(), call("ask_user", "ask", ASK), status(), status()])

    async def main():
        worker = TaskWorker(env.store, lambda cid: task_deps(env, client))
        task_id = start(env, client)
        await worker.tick()
        await drain(worker)
        resumed = await resume_waiting_task(store=env.store, conversation_id=env.context.conversation_id,
                                            text="Text", client=Triage({"resume": True}), model="router")
        chunks, result = [], {}
        stream = _stream_assistant_tool_turn(
            service=SimpleNamespace(client=client), settings=env.settings, rag=None,
            request=SimpleNamespace(task="Text"), run_id="run-2",
            conversation_id=env.context.conversation_id, model=MODEL, previous_messages=[],
            assistant_chunks=chunks, assistant_citations=[], run_store=None, task_result=result,
            resume_task_id=resumed.id,
        )

        async def consume():
            return [event async for event in stream]

        reading = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await worker.tick()
        events = await asyncio.wait_for(reading, 3)
        await drain(worker)
        return task_id, chunks, result, events

    task_id, chunks, result, events = asyncio.run(asyncio.wait_for(main(), 6))
    assert env.store.get(task_id).status == "done"
    # The old question is not replayed into this turn; the report is.
    assert "".join(chunks) == "Final report [W1]." and result["message_id"] == f"task-report:{task_id}"
    assert not any("task.created" in event for event in events)
    assert env.calls == [{"path": "input.txt"}]


@pytest.mark.parametrize("enabled", [False, True])
def test_new_request_or_disabled_tools_do_not_resume_over_http(harness, monkeypatch, enabled):
    settings = harness.settings
    cid = ChatHistory(settings.sqlite_db_path).ensure_conversation(None, "Task conversation")
    store = TaskStore(settings.sqlite_db_path)
    task = store.create("Write a report", conversation_id=cid)
    store.set_plan(task.id, [{"id": "draft", "title": "Draft report", "state": "running"}])
    store.transition(task.id, "running")
    store.user_interaction(task.id, "ask", {"step_id": "draft", **ASK})
    triage = []
    original_chat = harness.llm.chat

    async def chat(**kwargs):
        if "pending task question" in kwargs["messages"][0]["content"]:
            triage.append(kwargs)
            return SimpleNamespace(message='{"resume": false}')
        return await original_chat(**kwargs)

    async def tools(**kwargs):
        pytest.fail("A new casual request should use the chat route")

    monkeypatch.setattr(harness.llm, "chat", chat)
    monkeypatch.setattr(harness.llm, "chat_with_tools", tools, raising=False)
    response = harness.client.post("/agent/run/stream", json={
        "task": "Write a poem about autumn", "conversation_id": cid,
        "allowed_tools": ["assistant_tools"] if enabled else [],
    })
    events = _parse_sse(response.text)
    assert "run.completed" in dict(events) and "run.failed" not in dict(events), events
    assert len(triage) == int(enabled)
    assert store.get(task.id).status == "waiting_user"
    assert not any(event["kind"] == "answer" for event in store.events(task.id))
