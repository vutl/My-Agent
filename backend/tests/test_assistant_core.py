"""Offline interaction primitives: real stores and scripted model/deliverers."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.tool_loop import ApprovalBroker, run_tool_loop
from app.core.config import Settings
from app.db.sqlite import connect, init_db
from app.llm.openai_client import ToolCall, ToolTurn
from app.services.artifacts import ArtifactStore
from app.services.chat_history import ChatHistory
from app.services.long_term_memory import MemoryItemStore
from app.services.task_interactions import resume_waiting_task
from app.services.tasks import TaskStore
from app.services.reminders import ReminderStore
from app.tools import build_registry
from app.tools import assistant_core as core
from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult, ToolSource


@pytest.fixture
def context(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", assistant_workspace=str(tmp_path / "workspace"))
    init_db(settings.sqlite_db_path)
    cid = ChatHistory(settings.sqlite_db_path).ensure_conversation(None, "Primitives")
    return ToolContext(run_id="run-1", conversation_id=cid, workspace=Path(settings.assistant_workspace),
                       settings=settings)


class Client:
    def __init__(self, turns=(), decision=None):
        self.turns = list(turns)
        self.seen = []
        self.decision = decision

    async def chat_with_tools(self, **kwargs):
        self.seen.append({**kwargs, "messages": [dict(message) for message in kwargs["messages"]]})
        return self.turns.pop(0)

    async def chat(self, **kwargs):
        self.seen.append(kwargs)
        return SimpleNamespace(message=json.dumps(self.decision))


def call(name, args, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=json.dumps(args))


def turn(*calls):
    return ToolTurn(model="m", content="", tool_calls=tuple(calls))


async def collect(context, client, registry=None, decision=None):
    registry = registry or build_registry()
    broker, events = ApprovalBroker(), []
    async for name, data in run_tool_loop(
        client=client, model="m", task="Do the task", history=[], registry=registry,
        tool_names=registry.names(), context=context, language="English", approvals=broker,
    ):
        events.append((name, data))
        if name == "tool.call.approval_required":
            broker.decide(context.run_id, data["call_id"], decision)
    return events


def task_context(context):
    store = TaskStore(context.settings.sqlite_db_path)
    task = store.create("Research and report", conversation_id=context.conversation_id)
    store.set_plan(task.id, [{"id": "research", "title": "Find evidence", "state": "running"}])
    store.transition(task.id, "running")
    return replace(context, task_id=task.id), store


def ask_task(context, options=("Text", "PDF")):
    """A running task parked on a question, as the runner leaves it."""
    context, store = task_context(context)
    store.user_interaction(context.task_id, "ask", {"step_id": "research", "question": "Which format?",
                                                    "options": list(options), "reason": None})
    return context, store


def test_ask_ends_turn_without_executing_later_calls(context):
    client = Client([turn(call("ask_user", {"question": "Which format?", "options": ["PDF", "Text"],
                                            "reason": "Choose output"}),
                          call("remember", {"fact": "Must never execute"}, "c2"))])
    events = asyncio.run(collect(context, client))
    data = dict(events)
    assert data["answer"]["text"] == "Which format?"
    assert data["ask.user"]["options"] == ["PDF", "Text"]
    assert data["ask.user"]["reason"] == "Choose output"
    assert len(client.seen) == 1
    assert not MemoryItemStore(context.settings.sqlite_db_path).relevant("execute", conversation_id=context.conversation_id)


@pytest.mark.parametrize("args", [
    {"question": ""}, {"question": "?", "options": ["a"] * 7},
    {"question": "?", "options": "a"}, {"question": "?", "options": [4]},
])
def test_invalid_ask_is_tool_error_and_does_not_end_turn(context, args):
    client = Client([turn(call("ask_user", args)), ToolTurn(model="m", content="Retry")])
    events = asyncio.run(collect(context, client))
    assert "ask.user" not in dict(events)
    assert dict(events)["tool.call.completed"]["ok"] is False


def test_memory_roundtrip_provenance_scopes_and_unicode_slug(context):
    async def run():
        await core.remember({"fact": "I prefer jasmine tea"}, context)
        await core.remember({"fact": "Jasmine report is private", "scope": "conversation", "key": "report"}, context)
        await core.remember({"fact": "日本語", "kind": "procedural"}, context)
        result = await core.recall({"query": "jasmine"}, context)
        assert "jasmine tea" in result.content and "report is private" in result.content and "updated_at=" in result.content
        other = replace(context, conversation_id="other")
        result = await core.recall({"query": "jasmine"}, other)
        assert "jasmine tea" in result.content and "report is private" not in result.content
    asyncio.run(run())
    items = MemoryItemStore(context.settings.sqlite_db_path).relevant("jasmine", conversation_id=context.conversation_id)
    assert all(core._MEMORY_OPERATION_KEY_RE.fullmatch(item.memory_key) for item in items)
    assert all(item.metadata["source"] == "tool" and item.source_conversation_id == context.conversation_id for item in items)


@pytest.mark.parametrize("facts", [
    ("日本語を2年間勉強しました", "東京に2年間住んでいました"),
    ("A shared fact " * 20 + "first", "A shared fact " * 20 + "second"),
])
def test_generated_keys_distinguish_complete_facts(context, facts):
    async def run():
        for fact in facts:
            await core.remember({"fact": fact}, context)
    asyncio.run(run())
    with connect(context.settings.sqlite_db_path) as db:
        items = db.execute("SELECT memory_key, content FROM memory_items WHERE status = 'active'").fetchall()
    assert {item["content"] for item in items} == set(facts)
    assert len({item["memory_key"] for item in items}) == 2
    assert all(core._MEMORY_OPERATION_KEY_RE.fullmatch(item["memory_key"]) for item in items)


@pytest.mark.parametrize("fact", ["My password is do-not-store", "sk-abcdefghijklmnop123456789", "API key: abc"])
def test_sensitive_memory_refused(context, fact):
    result = asyncio.run(core.remember({"fact": fact}, context))
    assert not result.ok and "sensitive" in result.content
    with connect(context.settings.sqlite_db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0] == 0


@pytest.mark.parametrize("args", [
    {"fact": "tea", "key": "../bad"}, {"fact": "tea", "scope": "all"}, {"fact": "tea", "kind": "episodic"},
])
def test_invalid_memory_arguments(context, args):
    with pytest.raises(ValueError):
        asyncio.run(core.remember(args, context))


@pytest.mark.parametrize("approve", [False, True])
def test_forget_approval_shows_exact_memory_and_scopes(context, approve):
    asyncio.run(core.remember({"fact": "Jasmine tea", "key": "tea"}, context))
    asyncio.run(core.remember({"fact": "Local tea", "scope": "conversation", "key": "tea"}, context))
    client = Client([turn(call("forget", {"key": "tea", "scope": "user"})), ToolTurn(model="m", content="Done")])
    events = asyncio.run(collect(context, client, decision=approve))
    preview = dict(events)["tool.call.approval_required"]["preview"]
    assert [item["content"] for item in preview["memories"]] == ["Jasmine tea"]
    result = asyncio.run(core.recall({"query": "tea"}, context))
    assert ("Jasmine tea" in result.content) is not approve
    assert "Local tea" in result.content


@pytest.mark.parametrize("scope", ["user", "conversation"])
@pytest.mark.parametrize("approve_updated", [False, True])
def test_forget_reapproves_changed_version_through_real_loop(context, scope, approve_updated):
    async def run():
        await core.remember({"fact": "Original tea", "key": "tea", "scope": scope}, context)
        await core.remember({"fact": "Keep procedure", "key": "tea", "scope": scope, "kind": "procedural"}, context)
        broker, previews = ApprovalBroker(), []
        client = Client([turn(call("forget", {"key": "tea", "scope": scope})), ToolTurn(model="m", content="Done")])
        async for name, data in run_tool_loop(
            client=client, model="m", task="Forget tea", history=[], registry=build_registry(),
            tool_names=["forget"], context=context, language="English", approvals=broker,
        ):
            if name == "tool.call.approval_required":
                previews.append(data["preview"])
                if len(previews) == 1:
                    # A writer in another turn supersedes one approved version.
                    await core.remember({"fact": "Updated tea", "key": "tea", "scope": scope},
                                        replace(context, run_id="other-run"))
                else:
                    assert len(previews) == 2
                    active = MemoryItemStore(context.settings.sqlite_db_path).for_key(
                        scope=scope, memory_key="tea",
                        conversation_id=context.conversation_id if scope == "conversation" else None,
                    )
                    # The stale approval must not partially forget other kinds.
                    assert {item.content for item in active} == {"Updated tea", "Keep procedure"}
                broker.decide(context.run_id, data["call_id"], True if len(previews) == 1 else approve_updated)
        assert len(previews) == 2
        assert {item["content"] for item in previews[0]["memories"]} == {"Original tea", "Keep procedure"}
        assert {item["content"] for item in previews[1]["memories"]} == {"Updated tea", "Keep procedure"}
        result = await core.recall({"query": "tea"}, context)
        assert ("Updated tea" in result.content) is not approve_updated
    asyncio.run(run())


@pytest.mark.parametrize("with_task", [False, True])
def test_notifications_use_deliverers_and_limit_across_task_turns(context, monkeypatch, with_task):
    delivered = []
    if with_task:
        context, _ = task_context(context)
    monkeypatch.setattr(core.sys, "platform", "darwin")
    settings = SimpleNamespace(sqlite_db_path=context.settings.sqlite_db_path, telegram_bot_token="fake",
                               telegram_owner_id=1, telegram_backend_url="http://unused")
    context = replace(context, settings=settings)

    async def macos(title, text):
        delivered.append(("macos", title, text))

    async def telegram(_context, title, text):
        delivered.append(("telegram", title, text))

    monkeypatch.setattr(core, "deliver_macos", macos)
    monkeypatch.setattr(core, "deliver_telegram", telegram)

    async def run():
        for i in range(6):
            ctx = replace(context, run_id=f"run-{i}", state={}) if with_task else context
            result = await core.notify({"text": "Ready", "title": "Report"}, ctx)
            assert result.ok is (i < 5)
    asyncio.run(run())
    assert delivered == [(name, "Report", "Ready") for _ in range(5) for name in ("macos", "telegram")]
    notifications = ReminderStore(context.settings.sqlite_db_path).unread()
    assert len(notifications) == 5 and all(n["conversation_id"] == context.conversation_id for n in notifications)


def test_notification_channel_is_selective_and_errors_do_not_hide_delivery(context, monkeypatch):
    async def fail(*_args):
        raise RuntimeError("delivery failed")
    monkeypatch.setattr(core.sys, "platform", "darwin")
    monkeypatch.setattr(core, "deliver_macos", fail)
    result = asyncio.run(core.notify({"text": "Ready", "channel": "app"}, context))
    assert result.ok and "macos" not in result.content
    result = asyncio.run(core.notify({"text": "Ready"}, context))
    assert not result.ok and "app" in result.content and "macos: RuntimeError" in result.content


@pytest.mark.parametrize(("returncode", "delivered"), [(0, True), (1, False)])
def test_macos_notification_reports_delivery_only_when_osascript_succeeds(context, monkeypatch, returncode, delivered):
    from app.tools import macos

    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=returncode, stderr="not allowed")

    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.subprocess, "run", run)
    result = asyncio.run(core.notify({"text": "Ready", "title": "Report", "channel": "macos"}, context))
    assert commands and commands[0][-2:] == ["Report", "Ready"]
    assert result.ok is delivered
    if delivered:
        assert "delivered to: macos" in result.content
    else:
        assert "delivered to: none" in result.content and "macos: RuntimeError" in result.content


@pytest.mark.parametrize(("decision", "options", "reply", "resumes"), [
    (None, [], "any free-text answer", True),
    (None, ["Text", "PDF"], "pdf", True),
    (None, ["Text", "PDF"], "Write me a poem instead", False),
    ({"resume": "yes"}, [], "an answer", True),
    ({"resume": False}, [], "Make a calendar event instead", False),
])
def test_unusable_triage_resumes_free_text_and_needs_an_option_match_otherwise(
        context, decision, options, reply, resumes):
    context, store = ask_task(context, options)

    class Failing(Client):
        async def chat(self, **kwargs):
            if decision is None:
                raise RuntimeError("router unavailable")
            return await super().chat(**kwargs)

    result = asyncio.run(resume_waiting_task(store=store, conversation_id=context.conversation_id, text=reply,
                                           client=Failing(decision=decision), model="router"))
    assert (result is not None) is resumes
    assert store.get(context.task_id).status == ("running" if resumes else "waiting_user")


def test_large_tool_results_saved_with_preview_and_read_by_character_offset(context):
    content = "漢字-é" * 2000
    async def large(_args, _ctx):
        return ToolResult(content=content, summary="Large analysis", untrusted=True)
    registry = ToolRegistry()
    registry.register(Tool("anything", "anything", {}, large))
    client = Client([turn(call("anything", {})), ToolTurn(model="m", content="Done")])
    events = asyncio.run(collect(context, client, registry))
    handle = dict(events)["tool.call.completed"]["artifact"]
    message = client.seen[1]["messages"][-1]["content"]
    assert 'trust="untrusted"' in message
    assert handle in message and content[:1500] in message and content[:1501] not in message
    assert "Large analysis" in message and len(message) < 2200
    path = context.workspace / ".aya_runs" / "artifacts" / handle
    assert path.read_text() == content
    result = asyncio.run(core.read_artifact({"handle": handle, "offset": 1498, "length": 51}, context))
    assert result.content == content[1498:1549] and result.untrusted


def test_large_multi_source_output_is_also_preserved(context):
    content = "x" * 6100
    async def large(_args, _ctx):
        return ToolResult(content=content, summary="Sources", sources=[ToolSource("A"), ToolSource("B")])
    registry = ToolRegistry()
    registry.register(Tool("large", "large", {}, large))
    client = Client([turn(call("large", {})), ToolTurn(model="m", content="Done")])
    events = asyncio.run(collect(context, client, registry))
    handle = dict(events)["tool.call.completed"]["artifact"]
    assert content in (context.workspace / ".aya_runs" / "artifacts" / handle).read_text()


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_artifact_failure_preserves_completed_action_and_evidence(context, monkeypatch, error_type):
    actions, writes = [], []
    content = "Receipt: action completed once.\n" + "漢字-é" * 2000
    result = ToolResult(content=content, summary="Action completed; receipt retained", untrusted=True,
                        sources=[ToolSource("Receipt", url="https://receipt.example")],
                        data={"receipt": "completed-once"})

    async def action(_args, _ctx):
        actions.append("executed")
        return result

    def fail_save(_self, run_id, call_id, text):
        writes.append((run_id, call_id, text))
        raise error_type("storage unavailable " * 1000)

    monkeypatch.setattr(ArtifactStore, "save", fail_save)
    registry = ToolRegistry()
    registry.register(Tool("action", "Action", {}, action, side_effects=True))
    client = Client([turn(call("action", {})), ToolTurn(model="m", content="Done [W1]")])
    events = asyncio.run(collect(context, client, registry))
    completed = dict(events)["tool.call.completed"]
    message = client.seen[1]["messages"][-1]["content"]
    assert actions == ["executed"] and len(writes) == 1
    assert writes[0][2].endswith(content)
    assert completed["ok"] is True and completed["summary"] == result.summary
    assert completed["artifact"] is None
    assert completed["sources"] == dict(events)["answer"]["sources"] == [
        {"id": "W1", "title": "Receipt", "url": "https://receipt.example", "path": ""},
    ]
    assert completed["artifact_error"].startswith(error_type.__name__ + ":")
    assert len(completed["artifact_error"]) <= 400
    assert "Tool execution succeeded" in message and "Artifact storage failed" in message
    assert "Do not repeat completed actions" in message
    assert 'trust="untrusted"' in message and "[W1] Receipt" in message
    assert "Receipt: action completed once." in message and result.summary in message
    assert len(message) < 2600 and "read_artifact" not in message
    assert result.ok and result.content == content and result.data["receipt"] == "completed-once"


def test_artifact_failure_preserves_ask_control_and_ends_turn(context, monkeypatch):
    question = "Which format? " * 500

    def fail_save(*_args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(ArtifactStore, "save", fail_save)
    client = Client([turn(call("ask_user", {"question": question, "options": ["Text"], "reason": "Choose"}),
                          call("remember", {"fact": "Must never execute"}, "ignored")),
                     ToolTurn(model="m", content="Should never reach this turn")])
    events = asyncio.run(collect(context, client))
    data = dict(events)
    assert data["tool.call.completed"]["ok"] is True
    assert data["tool.call.completed"]["artifact_error"] == "OSError: storage unavailable"
    assert data["ask.user"]["question"] == question.strip()
    assert data["ask.user"]["options"] == ["Text"] and data["ask.user"]["reason"] == "Choose"
    assert data["answer"]["text"] == question.strip() and len(client.seen) == 1
    assert len([name for name, _ in events if name == "tool.call.completed"]) == 1
    assert not MemoryItemStore(context.settings.sqlite_db_path).relevant("execute", conversation_id=context.conversation_id)


def test_artifact_failure_preserves_failed_execution_status(context, monkeypatch):
    async def failed(_args, _ctx):
        return ToolResult(content="Action failed. " * 600, summary="Action failed", ok=False,
                          data={"declined": True})

    def fail_save(*_args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(ArtifactStore, "save", fail_save)
    registry = ToolRegistry()
    registry.register(Tool("action", "Action", {}, failed))
    client = Client([turn(call("action", {})), ToolTurn(model="m", content="Explain failure")])
    events = asyncio.run(collect(context, client, registry))
    completed = dict(events)["tool.call.completed"]
    message = client.seen[1]["messages"][-1]["content"]
    assert completed["ok"] is False and completed["declined"] is True
    assert completed["summary"] == "Action failed" and completed["artifact_error"]
    assert "Tool execution failed" in message and "Tool execution succeeded" not in message
    assert "Action failed." in message


@pytest.mark.parametrize("handle", ["../outside.txt", "/etc/passwd", "r/../../outside.txt", "r/a/../b.txt", "r/x.txt/extra"])
def test_artifact_traversal_refused(context, handle):
    with pytest.raises(ValueError):
        asyncio.run(core.read_artifact({"handle": handle}, context))


def test_artifact_symlink_escape_and_invalid_ranges_refused(context, tmp_path):
    store = ArtifactStore(context.workspace)
    store.save("r", "c", "text")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (store.root / "r" / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError):
        store.read("r/link.txt")
    (store.root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        store.save("escape", "new", "no")
    for offset, length in [(-1, 4), (False, 4), (0, 6001), (0, -1)]:
        with pytest.raises(ValueError):
            store.read("r/c.txt", offset, length)


def test_artifact_root_symlink_escape_refused(context, tmp_path):
    context.workspace.mkdir()
    (context.workspace / ".aya_runs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        ArtifactStore(context.workspace).save("r", "c", "private")


@pytest.mark.parametrize(("decision", "reply", "resumes"), [
    ({"resume": True}, "Text please", True), ({"resume": False}, "Make a calendar event instead", False),
    ({}, "Text", True), ({"resume": "false"}, "Another request", False),
])
def test_task_resume_triage_records_answer_preserving_step(context, decision, reply, resumes):
    context, store = ask_task(context)
    result = asyncio.run(resume_waiting_task(store=store, conversation_id=context.conversation_id,
                                           text=reply, client=Client(decision=decision), model="router"))
    assert (result is not None) is resumes
    task = store.get(context.task_id)
    assert task.status == ("running" if resumes else "waiting_user")
    assert task.plan[0]["state"] == "running"
    answers = [e for e in store.events(task.id) if e["kind"] == "answer"]
    assert [e["payload"]["text"] for e in answers] == ([reply] if resumes else [])


def test_only_latest_task_in_same_conversation_can_resume(context):
    context, store = ask_task(context, ["Yes"])
    store.create("Newer task", conversation_id=context.conversation_id)
    client = Client(decision={"resume": True})
    result = asyncio.run(resume_waiting_task(store=store, conversation_id=context.conversation_id,
                                           text="Yes", client=client, model="router"))
    assert result is None and not client.seen
    result = asyncio.run(resume_waiting_task(store=store, conversation_id="different",
                                           text="Yes", client=client, model="router"))
    assert result is None and not client.seen

def test_answered_question_is_not_reused_for_a_later_budget_wait(context):
    context, store = ask_task(context, ["Yes"])
    client = Client(decision={"resume": True})
    assert asyncio.run(resume_waiting_task(store=store, conversation_id=context.conversation_id,
                                         text="Yes", client=client, model="router"))
    store.transition(context.task_id, "waiting_user", note="Budget exceeded")
    client.seen.clear()
    result = asyncio.run(resume_waiting_task(store=store, conversation_id=context.conversation_id,
                                           text="Yes", client=client, model="router"))
    assert result is None and not client.seen


def test_interaction_rolls_back_state_when_event_write_fails(context):
    import sqlite3

    context, store = task_context(context)
    with connect(store.db_path) as db:
        db.executescript("""
            CREATE TRIGGER reject_ask BEFORE INSERT ON task_events WHEN NEW.kind = 'ask' BEGIN
                SELECT RAISE(ABORT, 'event failed');
            END;
        """)
    with pytest.raises(sqlite3.IntegrityError):
        store.user_interaction(context.task_id, "ask", {"question": "Which?"})
    assert store.get(context.task_id).status == "running"
    assert not any(event["kind"] == "ask" for event in store.events(context.task_id))


def test_artifact_root_cannot_redirect_to_other_workspace_files(context):
    context.workspace.mkdir()
    elsewhere = context.workspace / "documents"
    elsewhere.mkdir()
    (context.workspace / ".aya_runs").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        ArtifactStore(context.workspace).save("r", "c", "output")
