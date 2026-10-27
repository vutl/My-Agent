"""Offline bypass/gate regressions; HTTP tests use a temporary indexed library."""

import asyncio
import json
import unicodedata
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agents.handoffs import router_handoff_tools
from app.agents.tool_loop import run_tool_loop
from app.api.agent.models import AgentRunRequest
from app.api.agent.router_gate import library_signal
from app.api.agent.scope import _safe_library_titles
from app.core.config import Settings, get_settings
from app.llm.ollama_client import OllamaError
from app.llm.openai_client import ToolCall, ToolTurn
from app.main import app
from app.services.agent_run_store import AgentRunStore
from app.services.conversation_state import ConversationWorkingState, DocumentThreadState
from app.services.document_scope_service import DocumentScopeResolution
from app.services.query_rewrite_service import QueryRewriteService
from app.services.tool_decision_service import IntentRouterService
from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult
from streaming_fakes import tool_turn_events
from test_agent_stream_e2e import harness, _first, _names, _parse_sse


def gate_inputs():
    return dict(
        request=AgentRunRequest(task="Mấy giờ rồi cậu?", allowed_tools=["assistant_tools"]),
        loaded_working_state=ConversationWorkingState([]), scope_resolution=DocumentScopeResolution(),
        recent_document_ids=[], library_titles=[("d", "Acoustic Learning (acoustic.md)")],
        named_document_ids=[], asks_library_results=False, supports_tools=True,
    )


@pytest.mark.parametrize("mode", ["chat", "file_qa", "unknown"])
def test_gate_preserves_nondefault_modes(mode):
    args = gate_inputs()
    args["request"].mode = mode
    assert library_signal(**args) == "request_mode"


@pytest.mark.parametrize("updates,signal", [
    ({"recent_document_ids": ["d"]}, "recent_retrieval"),
    ({"named_document_ids": ["d"]}, "named_document_ids"),
    ({"asks_library_results": True}, "library_results"),
    ({"supports_tools": False}, "client_without_tools"),
    ({"scope_failure": True}, "scope_failure"),
    ({"scope_resolution": DocumentScopeResolution(document_ids=("d",))}, "scope_documents"),
    ({"scope_resolution": DocumentScopeResolution(mentions=(object(),))}, "scope_mentions"),
    ({"scope_resolution": DocumentScopeResolution(ambiguous_mentions=(object(),))}, "scope_mentions"),
    ({"scope_resolution": DocumentScopeResolution(collection_removed_ids=("d",))}, "scope_failure"),
    ({"loaded_working_state": ConversationWorkingState(["d"])}, "active_documents"),
    ({"loaded_working_state": ConversationWorkingState([], active_topic="topic")}, "active_topic_or_filenames"),
    ({"loaded_working_state": ConversationWorkingState([], active_filenames=["source.md"])}, "active_topic_or_filenames"),
    ({"loaded_working_state": ConversationWorkingState([], referent_document_ids=["d"])}, "conversation_documents"),
    ({"loaded_working_state": ConversationWorkingState([], recent_document_threads=(DocumentThreadState(("d",)),))},
     "conversation_documents"),
])
def test_each_state_signal_preserves_old_path(updates, signal):
    assert library_signal(**{**gate_inputs(), **updates}) == signal


@pytest.mark.parametrize("field,value,signal", [
    ("collection_id", "c", "collection"), ("collection_id", "", "collection"),
    ("allowed_tools", [], "assistant_tools_disabled"),
])
def test_request_signals(field, value, signal):
    args = gate_inputs()
    setattr(args["request"], field, value)
    assert library_signal(**args) == signal


@pytest.mark.parametrize("task,signal", [
    ("Which papers mention training?", "library_wide"),
    ("Search in my documents", "local_retrieval_anchor"),
    ("Còn bài kia?", "vague_document_reference"),
    ("Show Figure 2", "document_visual"),
    ("Explain paper Unknown", "explicit_document_reference"),
    ("resume pdf", "document_resume"),
    ("compare methods", "document_comparison"),
])
def test_task_signals(task, signal):
    args = gate_inputs()
    args["request"].task = task
    assert library_signal(**args) == signal


@pytest.mark.parametrize("task", ["Mấy giờ rồi cậu?", "đặt nhắc 8h", "Tớ đang vui quá", "What's 2 + 2?"])
def test_plain_empty_conversation_has_no_signal(task):
    args = gate_inputs()
    args["request"].task = task
    assert library_signal(**args) is None


@pytest.mark.parametrize("task,title", [
    ("mae hoạt động ra sao?", "MAE: Masked Encoders"),
    ("WeNet toolkit", "wenet toolkit (wenet.pdf)"),
    ("Tell me about rl", "RL Methods"),
    ("am thanh?", "Âm Thanh"),
    ("dinh dang", "Định Dạng"),
    ("the pipeline", "Source (pipeline_notes.txt)"),
])
def test_distinctive_title_and_filename_tokens(task, title):
    args = gate_inputs()
    args["request"].task = task
    args["library_titles"] = [("d", title)]
    assert library_signal(**args) == "library_title_token"


@pytest.mark.parametrize("catalog_form", ["NFC", "NFD"])
@pytest.mark.parametrize("task_form", ["NFC", "NFD"])
@pytest.mark.parametrize("title,task", [
    ("Định Dạng", "dinh dang"),
    ("Định Dạng", "định dạng"),
    ("Source (định_dạng.txt)", "dinh dang"),
    ("Source (định_dạng.txt)", "định dạng"),
    ("ẤM", "ấm"),  # Two-character acronyms retain their case before folding.
])
def test_title_and_filename_tokens_are_canonically_equivalent(catalog_form, task_form, title, task):
    args = gate_inputs()
    args["library_titles"] = [("d", unicodedata.normalize(catalog_form, title))]
    args["request"].task = unicodedata.normalize(task_form, task)
    assert library_signal(**args) == "library_title_token"


def test_stopwords_and_short_nonacronyms_do_not_overlap():
    args = gate_inputs()
    args["request"].task = "của the với xy"
    args["library_titles"] = [("d", "Paper bài tài liệu the của với (xy.md)")]
    assert library_signal(**args) is None


def test_missing_catalog_is_distinguishable_from_an_empty_library():
    def unavailable(_collection_id):
        raise RuntimeError("catalog unavailable")

    rag = SimpleNamespace(library_titles=unavailable)
    assert _safe_library_titles(rag, None) == []  # Existing routing remains tolerant.
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        _safe_library_titles(rag, None, raise_on_error=True)


@pytest.fixture
def bypass(harness, monkeypatch, tmp_path):
    h = harness
    h.settings.router_bypass = True
    h.settings.agent_debug_trace_enabled = True
    h.client = TestClient(app, client=("127.0.0.1", 50000))
    h.settings.assistant_workspace = tmp_path / "workspace"
    turns, offered, routed, rewritten, streamed = [], [], [], [], []

    async def whole(**kwargs):
        offered.append(kwargs)
        turn = turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    async def stream(**kwargs):
        streamed.append(kwargs)
        async for event in tool_turn_events(await whole(**kwargs)):
            yield event

    h.llm.chat_with_tools = whole
    h.llm.chat_with_tools_stream = stream
    h.llm.router_reply = lambda _: {"route": "agent", "selected_tools": ["assistant_tools"],
                                    "confidence": "high", "needs_plan": False, "reason": "action"}
    decide, rewrite = IntentRouterService.decide, QueryRewriteService.rewrite

    async def decide_spy(self, **kwargs):
        routed.append(kwargs)
        return await decide(self, **kwargs)

    def rewrite_spy(self, **kwargs):
        rewritten.append(kwargs)
        return rewrite(self, **kwargs)

    async def no_quick(*_args, **_kwargs):
        pytest.fail("needs_plan_quick must not run on a bypass turn")

    monkeypatch.setattr(IntentRouterService, "decide", decide_spy)
    monkeypatch.setattr(QueryRewriteService, "rewrite", rewrite_spy)
    monkeypatch.setattr("app.agents.task_runner.needs_plan_quick", no_quick)

    def turn(task="Mấy giờ rồi cậu?", **kwargs):
        response = h.client.post("/agent/run/stream", json={
            "task": task, "allowed_tools": ["assistant_tools", "search_local_docs", "retrieve_visual_assets"],
            "debug_trace": True, **kwargs,
        })
        assert response.status_code == 200, response.text
        events = _parse_sse(response.text)
        run_id = _first(events, "run.started")["run_id"]
        return events, AgentRunStore(h.settings.sqlite_db_path).get_run(run_id)

    return SimpleNamespace(h=h, turns=turns, offered=offered, routed=routed, rewritten=rewritten,
                           streamed=streamed, turn=turn)


@pytest.mark.parametrize("stream_only", [False, True])
def test_bypass_streams_once_on_turn_model_without_any_router_call(bypass, stream_only):
    b = bypass
    if stream_only:
        del b.h.llm.chat_with_tools
    b.turns.append(ToolTurn("custom-turn-model", "It is noon."))
    events, run = b.turn(model="custom-turn-model")
    assert "run.completed" in _names(events) and "run.failed" not in _names(events)
    assert not b.routed and not b.rewritten and len(b.offered) == 1
    assert len(b.streamed) == 1
    assert b.offered[0]["model"] == "custom-turn-model"
    assert _first(events, "agent.route.decided")["reason"] == "router_bypass"
    assert _first(events, "timing") == {"run_id": run["id"], "conversation_id": run["conversation_id"],
                                       "stage": "router", "ms": 0, "route": "agent"}
    assert run["final_answer"] == "It is noon."
    metadata = run["metadata"]
    assert metadata["router_bypass"] is True and metadata["library_signal"] is None and metadata["handoff"] is None
    assert metadata["first_delta_ms"] >= 0
    trace = AgentRunStore(b.h.settings.sqlite_db_path).get_debug_trace(run["id"])
    assert trace["payload"]["router_bypass"]["first_delta_ms"] == metadata["first_delta_ms"]


def test_nonstreaming_client_can_bypass(bypass):
    b = bypass
    del b.h.llm.chat_with_tools_stream
    b.turns.append(ToolTurn("m", "Whole answer."))
    events, run = b.turn()
    assert not b.streamed and not b.routed and not b.rewritten and len(b.offered) == 1
    assert run["final_answer"] == "Whole answer." and "run.completed" in _names(events)


def test_catalog_lookup_failure_preserves_old_path(bypass, monkeypatch):
    from app.api.agent import stream

    original_titles = stream._safe_library_titles

    def unavailable_for_gate(rag, collection_id, **kwargs):
        if kwargs.get("raise_on_error"):
            raise RuntimeError("catalog unavailable")
        return original_titles(rag, collection_id)

    monkeypatch.setattr(stream, "_safe_library_titles", unavailable_for_gate)
    bypass.turns.append(ToolTurn("m", "Conservative route."))
    events, run = bypass.turn()
    assert len(bypass.routed) == 1 and not run["metadata"]["router_bypass"]
    assert run["metadata"]["library_signal"] == "gate_lookup_failed"
    assert "run.completed" in _names(events)


def _handoff(name, content="Let me check. "):
    return ToolTurn("m", content, (ToolCall("h1", name, '{"question":"different retrieval input", "goal":"different goal"}'),))


def _stable_routing(events):
    """Run ids/times vary; the actual routing, scope and retrieval contracts must not."""
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items()
                    if k not in {"run_id", "conversation_id", "ms", "created_at", "updated_at"}}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    return [(name, stable(data)) for name, data in events
            if name in {"agent.route.decided", "retrieval.started", "retrieval.completed", "retrieval.skipped",
                        "agent.scope.rejected", "query.rewritten"} and data.get("reason") != "router_bypass"]


@pytest.mark.parametrize("enabled", [False, True])
def test_focused_results_turn_skips_gate_library_lookups(bypass, monkeypatch, enabled):
    from app.api.agent import stream
    from app.services.rag_service import RagService

    b = bypass
    b.h.llm.router_reply = lambda _: {"route": "file_qa", "selected_tools": ["search_local_docs"],
                                      "scope": "active_document", "confidence": "high", "reason": "focused"}
    _, conversation_id, _ = b.h.turn("DualSQL dùng những tool nào?")
    b.h.settings.router_bypass = enabled
    scans, gate_catalogs = [], []
    original_scan = RagService.documents_containing_all_terms
    original_titles = stream._safe_library_titles

    def scan(self, *args, **kwargs):
        scans.append((args, kwargs))
        return original_scan(self, *args, **kwargs)

    def titles(*args, **kwargs):
        if kwargs.get("raise_on_error"):
            gate_catalogs.append((args, kwargs))
        return original_titles(*args, **kwargs)

    monkeypatch.setattr(RagService, "documents_containing_all_terms", scan)
    monkeypatch.setattr(stream, "_safe_library_titles", titles)
    question = "What are DualSQL results?"
    events, run = b.turn(question, conversation_id=conversation_id)
    assert "run.completed" in _names(events) and "run.failed" not in _names(events)
    assert run["metadata"]["library_signal"] == ("active_documents" if enabled else None)
    assert not run["metadata"]["router_bypass"]
    assert _first(events, "retrieval.started")["focus_document_ids"] == [b.h.ids["dualsql"]]
    assert not scans and not gate_catalogs
    assert b.routed[-1]["task"] == question


def test_library_handoff_resets_narration_then_replays_original_routing(bypass):
    b = bypass
    question = "Tell me about optimization"
    b.h.llm.router_reply = lambda _: {"route": "file_qa", "selected_tools": ["search_local_docs"],
                                      "scope": "library", "confidence": "high", "reason": "library_topic"}
    b.turns.append(_handoff("ask_library"))
    events, run = b.turn(question)
    assert "run.completed" in _names(events) and "run.failed" not in _names(events)
    reset = _first(events, "message.reset")
    assert reset["reason"] == "handoff" and "Let me check" not in run["final_answer"]
    assert not run["tool_calls"] and not any(n.startswith("tool.call") for n, _ in events)
    assert len(b.routed) == 1 and b.routed[0]["task"] == question
    assert b.rewritten and all(item["query"] == question for item in b.rewritten)
    assert run["metadata"]["handoff"] == "ask_library"
    b.h.settings.router_bypass = False
    baseline, _ = b.turn(question)
    assert _stable_routing(events) == _stable_routing(baseline)


def test_plan_handoff_uses_original_goal_and_task_runner_without_handoff_schemas(bypass, monkeypatch):
    b = bypass
    from app.services import task_events
    from app.services.tasks import TaskStore

    plans = []
    original_chat = b.h.llm.chat

    async def chat(**kwargs):
        if kwargs["messages"][0]["content"].startswith("Triage and plan"):
            plans.append(kwargs)
            return SimpleNamespace(message=json.dumps({"needs_plan": True, "steps": [
                {"id": name, "title": name, "tools_hint": [], "done_when": "done"} for name in ("s1", "s2")
            ]}))
        return await original_chat(**kwargs)

    async def watch(store, task_id, **_kwargs):
        # A worker stand-in completes the persisted plan, allowing the API to
        # exercise its real task creation, report attachment and finalization.
        store.transition(task_id, "done")
        yield "answer.delta", {"text": "Planned report."}
        yield "answer", {"text": "Planned report.", "sources": []}

    monkeypatch.setattr(b.h.llm, "chat", chat)
    monkeypatch.setattr(task_events, "watch_task", watch)
    b.turns.append(_handoff("plan_task"))
    events, run = b.turn("Organize tomorrow's work")
    assert "task.created" in _names(events) and "run.completed" in _names(events)
    assert not b.routed and not b.rewritten and len(plans) == 1
    assert plans[0]["model"] == b.h.settings.default_model
    assert plans[0]["messages"][-1]["content"] == "Organize tomorrow's work"
    assert "ask_library" not in plans[0]["messages"][0]["content"]
    assert "plan_task" not in plans[0]["messages"][0]["content"]
    task = TaskStore(b.h.settings.sqlite_db_path).list()[0]
    execution = next(e["payload"]["execution"] for e in TaskStore(b.h.settings.sqlite_db_path).events(task.id)
                     if "execution" in e["payload"])
    assert not {"ask_library", "plan_task"} & set(execution["tool_names"])
    assert run["metadata"]["handoff"] == "plan_task" and run["final_answer"] == "Planned report."


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("handoff", ["plan_task", "ask_library"])
def test_late_handoff_keeps_completed_actions_and_budget_in_same_loop(bypass, monkeypatch, streaming, handoff):
    from app.services.reminders import ReminderStore
    from app.services.tasks import TaskStore

    b = bypass
    if not streaming:
        del b.h.llm.chat_with_tools_stream

    async def no_plan(*_args, **_kwargs):
        pytest.fail("A handoff after execution must not restart the original goal")

    monkeypatch.setattr("app.agents.task_runner.plan_goal", no_plan)
    b.turns.extend([
        ToolTurn("m", "", (ToolCall("reminder", "create_reminder", json.dumps({
            "text": "Take a break", "when": "2099-01-01T08:00",
        })),)),
        _handoff(handoff, content=""),
        ToolTurn("m", "Your reminder is set."),
    ])
    events, run = b.turn("đặt nhắc 8h", max_steps=2)

    assert "run.completed" in _names(events) and "run.failed" not in _names(events)
    assert not b.routed and not b.rewritten and len(b.offered) == 3
    assert "task.created" not in _names(events) and "message.reset" not in _names(events)
    assert TaskStore(b.h.settings.sqlite_db_path).list() == []
    reminders = ReminderStore(b.h.settings.sqlite_db_path).pending()
    assert len(reminders) == 1 and reminders[0].text == "Take a break"
    assert [call["tool_name"] for call in run["tool_calls"]] == ["create_reminder"]
    assert [data["name"] for name, data in events if name == "tool.call.completed"] == ["create_reminder"]
    assert run["metadata"]["handoff"] is None
    assert run["final_answer"] == "Your reminder is set."
    assert not {"ask_library", "plan_task"} & {s["function"]["name"] for s in b.offered[1]["tools"]}
    assert b.offered[2]["tools"] == []  # The original loop's step budget was not reset.
    messages = b.offered[2]["messages"]
    receipts = [m for m in messages if m.get("tool_call_id") == "reminder"]
    assert len(receipts) == 1 and reminders[0].describe() in receipts[0]["content"]
    rejection = next(m for m in messages if m.get("tool_call_id") == "h1")
    assert "do not repeat completed actions" in rejection["content"].lower()
    assert "Tool budget reached" in messages[-1]["content"]


def test_gateway_failure_before_first_delta_falls_back_to_router(bypass):
    b = bypass
    b.turns.extend([OllamaError("gateway unavailable"), ToolTurn("m", "Recovered answer.")])
    events, run = b.turn()
    assert "run.failed" not in _names(events) and run["final_answer"] == "Recovered answer."
    assert len(b.routed) == 1 and len(b.offered) == 2
    assert run["metadata"]["fallback"] == "gateway_before_first_delta"
    assert not {"ask_library", "plan_task"} & {s["function"]["name"] for s in b.offered[1]["tools"]}


def test_gateway_failure_after_text_is_not_replayed(bypass, monkeypatch):
    async def broken(**_kwargs):
        yield "text", "Partial answer"
        raise OllamaError("stream broke")

    monkeypatch.setattr(bypass.h.llm, "chat_with_tools_stream", broken)
    events, run = bypass.turn()
    assert "run.failed" in _names(events) and run["status"] == "failed" and not bypass.routed


def test_flag_off_keeps_existing_route(bypass):
    b = bypass
    b.h.settings.router_bypass = False
    b.turns.append(ToolTurn("m", "Old path."))
    events, run = b.turn()
    assert len(b.routed) == 1 and b.rewritten and run["final_answer"] == "Old path."
    assert run["metadata"]["router_bypass"] is False
    assert _first(events, "agent.route.decided")["reason"] != "router_bypass"
    assert not {"ask_library", "plan_task"} & {s["function"]["name"] for s in b.offered[0]["tools"]}


@pytest.mark.parametrize("task", ["Hi Aya!", "Cảm ơn cậu", "who are you?"])
def test_deterministic_chat_shortcuts_stay_chat(bypass, task):
    events, run = bypass.turn(task)
    assert _first(events, "agent.route.decided")["route"] == "chat"
    assert not bypass.offered and run["metadata"]["router_bypass"] is False


def test_named_document_keeps_rag_path_when_flag_on(bypass):
    b = bypass
    events, run = b.turn("DualSQL dùng những tool nào?")
    assert _first(events, "retrieval.started")["focus_document_ids"] == [b.h.ids["dualsql"]]
    assert len(b.routed) == 1 and not b.offered
    assert run["metadata"]["library_signal"] and run["metadata"]["router_bypass"] is False


@pytest.mark.parametrize("streaming", [False, True])
def test_handoff_preempts_mixed_batch_without_execution_or_budget_charge(tmp_path, streaming):
    executed, registry = [], ToolRegistry()

    async def handler(args, _context):
        executed.append(args)
        return ToolResult("done", "done")

    registry.register(Tool("write", "write", {"type": "object"}, handler))
    turn = ToolTurn("m", "Narration", (ToolCall("w", "write", "{}"), ToolCall("h", "ask_library", "{}")))

    class Client:
        async def chat_with_tools(self, **kwargs):
            return turn

        async def chat_with_tools_stream(self, **kwargs):
            async for event in tool_turn_events(turn):
                yield event

    async def collect():
        return [event async for event in run_tool_loop(
            client=Client(), model="m", task="question", history=[], registry=registry, tool_names=["write"],
            context=ToolContext("run", "conv", tmp_path), language="English", stream_answer=streaming,
            handoff_tools=router_handoff_tools([]),
        )]

    events = asyncio.run(collect())
    assert not executed and events[-1][0] == "handoff"
    assert not {"tool.call.started", "tool.call.completed", "answer"} & set(_names(events))
    assert not any(n == "tool.call.pending" and d["name"] == "ask_library" for n, d in events)


def test_handoff_title_list_is_bounded_and_not_a_registry_capability():
    schemas = router_handoff_tools([(str(i), f"Title {i}") for i in range(65)])
    description = schemas[0]["function"]["description"]
    assert "Title 59" in description and "Title 60" not in description
    assert {schema["function"]["name"] for schema in schemas} == {"ask_library", "plan_task"}


def test_bypass_flag_defaults_off_and_reads_environment(monkeypatch):
    assert Settings().router_bypass is False
    get_settings.cache_clear()
    monkeypatch.setenv("AYA_ROUTER_BYPASS", "true")
    try:
        assert get_settings().router_bypass is True
    finally:
        get_settings.cache_clear()
