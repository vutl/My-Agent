"""Offline end-to-end tests that drive the real ``run_agent_stream`` SSE endpoint.

Real SQLite indexing/FTS retrieval, real routing/scope/answer code; only the
LLM clients (scripted fake) and the vector store/embeddings (hash provider,
empty LanceDB stand-in) are replaced, so nothing touches the network.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import agent as agent_api
from app.core.config import Settings, get_settings
from app.db.sqlite import init_db
from app.llm.ollama_client import StreamChunk
from app.rag.embeddings import HashEmbeddingProvider
from app.main import app
from app.services.agent_service import AgentService
from app.services.conversation_runtime import ConversationRuntimeGate
from app.services.indexing_service import IndexingService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.tool_decision_service import IntentRouterService

DOCS = {
    "dualsql.md": (
        "# DualSQL: Text-to-SQL with Multi-Agent RL\n\n"
        "## Tools\n\nDualSQL agents call the tools sqlite_query, schema_inspect and "
        "sql_verifier to execute and check candidate queries.\n"
    ),
    "retoolsql.md": (
        "# ReToolSQL: Agentic RL for Robust Text-to-SQL\n\n"
        "## Tools\n\nReToolSQL uses the tools table_lookup, column_sampler and "
        "result_checker during agentic reinforcement learning.\n"
    ),
    "thong_tu_12_2026_sang_loc.md": (
        "# THÔNG TƯ\n\nBan hành Danh mục bệnh bẩm sinh cần sàng lọc trước sinh và sơ sinh. "
        "Thông tư này có hiệu lực từ ngày 01/7/2026.\n"
    ),
    "thong_tu_83_2026_chuan_co_so.md": (
        "# THÔNG TƯ Quy định Chuẩn cơ sở giáo dục đại học\n\n"
        "Tỷ lệ người học quy đổi trên giảng viên toàn thời gian không vượt quá 40.\n"
    ),
    "vietnam.md": (
        "# Economy of Vietnam\n\n"
        "Vietnam's nominal GDP reached 430 billion US dollars in 2023.\n"
    ),
}
ANSWER = "Đây là câu trả lời thử nghiệm."


class _EmptyStore:
    async def search_document_cards(self, **_kwargs):
        return []

    async def search_text_chunks(self, **_kwargs):
        return []

    async def search_table_chunks(self, **_kwargs):
        return []

    async def search_figure_chunks(self, **_kwargs):
        return []


class ScriptedLLM:
    """One fake for chat() and stream_chat(); replies are chosen by prompt type."""

    def __init__(self) -> None:
        self.router_reply = lambda prompt: "not json"
        self.router_prompts: list[str] = []

    @staticmethod
    def _split(messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return str(system), user if isinstance(user, str) else json.dumps(user)

    async def chat(self, *, model, messages, **_kwargs):
        system, user = self._split(messages)
        if system.startswith("You are the turn-policy router"):
            self.router_prompts.append(user)
            reply = self.router_reply(user)
            reply = reply if isinstance(reply, str) else json.dumps(reply)
        elif system.startswith("You check an answer's numeric claims"):
            reply = '{"corrections": []}'
        elif "rewrite follow-up questions" in system:
            reply = json.dumps({
                "standalone_query": user.rsplit("Latest user question:", 1)[-1].split("\n")[1],
                "is_followup": False, "current_topic": None, "required_entities": [],
                "use_last_sources": False, "answer_intent": "direct_answer",
                "answer_depth": "normal",
            })
        else:
            reply = ANSWER
        return SimpleNamespace(model=model, message=reply)

    async def stream_chat(self, *, model, messages, **_kwargs):
        yield StreamChunk(content=ANSWER)
        yield StreamChunk(content="", done=True, finish_reason="stop")


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events, name, data = [], "message", []
    for line in text.splitlines() + [""]:
        if not line:
            if data:
                events.append((name, json.loads("\n".join(data))))
            name, data = "message", []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    return events


@pytest.fixture
def harness(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", lightrag_enabled=False,
                        retrieval_engine="legacy")
    init_db(settings.sqlite_db_path)
    indexer = IndexingService(settings.sqlite_db_path, artifact_root=settings.artifacts_path)
    ids = {}
    for name, body in DOCS.items():
        source = tmp_path / "library" / name
        source.parent.mkdir(exist_ok=True)
        source.write_text(body, encoding="utf-8")
        ids[name.split(".")[0]] = indexer.index_file(source_path=str(source)).id

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda _s: HashEmbeddingProvider(dimensions=16))
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda _s: _EmptyStore())

    llm = ScriptedLLM()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[agent_api.get_agent_service] = lambda: AgentService(
        client=llm, default_model=settings.default_model)
    app.dependency_overrides[agent_api.get_intent_router] = lambda: IntentRouterService(
        client=llm, default_model=settings.default_model)
    app.dependency_overrides[agent_api.get_query_rewrite_service] = lambda: QueryRewriteService(
        client=llm, default_model=settings.default_model)
    # Each TestClient request runs on its own event loop; the gate binds to one.
    app.dependency_overrides[agent_api.get_conversation_runtime_gate] = ConversationRuntimeGate
    # Background memory folding builds its own client; keep it off the network.
    monkeypatch.setattr(agent_api, "get_llm_client", lambda **_k: llm)
    monkeypatch.setattr(agent_api, "schedule_memory_fold", lambda **_k: None)

    client = TestClient(app)

    def turn(task: str, conversation_id: str | None = None):
        response = client.post("/agent/run/stream", json={
            "task": task, "conversation_id": conversation_id,
            "model": settings.default_model, "allowed_tools": [
                "search_local_docs", "retrieve_visual_assets"],
            "require_confirmation": True,
        })
        assert response.status_code == 200, response.text
        events = _parse_sse(response.text)
        started = next(d for n, d in events if n == "run.started")
        answer = "".join(d.get("delta", "") for n, d in events if n == "message.delta")
        return events, started["conversation_id"], answer

    yield SimpleNamespace(turn=turn, ids=ids, llm=llm)
    app.dependency_overrides.clear()


def _first(events, name):
    return next((d for n, d in events if n == name), None)


def _label(prompt: str, title_part: str) -> str:
    block = prompt.split("local_library:", 1)[1].split("conversation_documents:", 1)[0]
    match = re.search(rf"^(L\d+): .*{re.escape(title_part)}", block, re.MULTILINE)
    assert match, block
    return match.group(1)


def _names(events):
    return [n for n, _ in events]


def test_named_paper_turn_focuses_that_document(harness):
    events, _cid, answer = harness.turn("DualSQL dùng những tool nào?")
    retrieval = _first(events, "retrieval.started")
    assert retrieval["focus_document_ids"] == [harness.ids["dualsql"]], _names(events)
    assert "run.completed" in _names(events) and "run.failed" not in _names(events)
    assert answer.strip()
    citations = _first(events, "answer.citations")["citations"]
    assert citations and citations[0]["id"] == "S1"
    assert {item["document_id"] for item in citations} == {harness.ids["dualsql"]}


def test_pronoun_followup_uses_router_referent(harness):
    _events, cid, _ = harness.turn("DualSQL dùng những tool nào?")
    harness.llm.router_reply = lambda prompt: {
        "route": "file_qa", "selected_tools": ["search_local_docs"],
        "confidence": "high", "scope": "named_documents",
        "referent_documents": [_label(prompt, "ReToolSQL")], "reason": "other_paper",
    }
    events, _cid, answer = harness.turn("Còn bài kia thì sao?", cid)
    decided = _first(events, "agent.route.decided")
    assert decided["referent_document_ids"] == [harness.ids["retoolsql"]], decided
    assert _first(events, "retrieval.started")["focus_document_ids"] == [harness.ids["retoolsql"]]
    assert "run.completed" in _names(events) and answer.strip()


def test_malformed_router_followup_keeps_deterministic_scope(harness):
    _events, cid, _ = harness.turn("DualSQL dùng những tool nào?")
    harness.llm.router_reply = lambda prompt: "sorry, I cannot produce JSON {{"
    events, _cid, answer = harness.turn("Còn bài kia thì sao?", cid)
    decided = _first(events, "agent.route.decided")
    assert decided["route"] == "file_qa", decided
    assert _first(events, "retrieval.started") is not None, _names(events)
    assert "run.completed" in _names(events) and answer.strip()


def test_offtopic_turn_skips_retrieval(harness):
    harness.llm.router_reply = lambda prompt: {
        "route": "chat", "selected_tools": [], "confidence": "high",
        "scope": "none", "reason": "weather",
    }
    events, _cid, answer = harness.turn("thời tiết hôm nay thế nào?")
    assert _first(events, "retrieval.started") is None, _names(events)
    assert "run.completed" in _names(events) and answer.strip()


def test_turn_naming_two_papers_covers_both(harness):
    events, _cid, answer = harness.turn(
        "Bài DualSQL dùng những tool nào. Bài ReToolSQL có công cụ tương đương không?"
    )
    focus = _first(events, "retrieval.started")["focus_document_ids"]
    assert set(focus) == {harness.ids["dualsql"], harness.ids["retoolsql"]}, _names(events)
    assert "run.completed" in _names(events) and answer.strip()


def test_router_corrects_a_loose_catalog_name_match(harness):
    """A generic title ("THÔNG TƯ") must not capture every "thông tư N": the
    router, reading every title and file name, decides the loose match."""
    harness.llm.router_reply = lambda prompt: {
        "route": "file_qa", "selected_tools": ["search_local_docs"], "confidence": "high",
        "scope": "named_documents", "library_documents": [_label(prompt, "thong_tu_83")],
        "reason": "names_circular_83",
    }
    events, _cid, answer = harness.turn("Thông tư 83 quy định tỷ lệ người học trên giảng viên thế nào?")
    decided = _first(events, "agent.route.decided")
    assert harness.llm.router_prompts, "the loose catalog match skipped the router"
    assert decided["document_scope"]["document_ids"] == [harness.ids["thong_tu_83_2026_chuan_co_so"]], decided
    assert _first(events, "retrieval.started")["focus_document_ids"] == [harness.ids["thong_tu_83_2026_chuan_co_so"]]
    assert "run.completed" in _names(events) and answer.strip()


def test_router_failure_keeps_the_loose_catalog_match(harness):
    harness.llm.router_reply = lambda prompt: "not json"
    events, _cid, _answer = harness.turn("DualSQL dùng những tool nào?")
    assert _first(events, "retrieval.started")["focus_document_ids"] == [harness.ids["dualsql"]]


def test_question_about_an_unlisted_document_says_it_is_not_in_the_library(harness):
    prompts: list[str] = []
    original = harness.llm.stream_chat

    async def recording_stream(*, model, messages, **kwargs):
        prompts.append(json.dumps(messages, ensure_ascii=False))
        async for chunk in original(model=model, messages=messages, **kwargs):
            yield chunk

    harness.llm.stream_chat = recording_stream
    harness.llm.router_reply = lambda prompt: {
        "route": "chat", "selected_tools": [], "confidence": "high", "scope": "none",
        "unlisted_document": "Nghị định 81/2021", "reason": "asks_about_unlisted_regulation",
    }
    events, _cid, answer = harness.turn("Nghị định 81/2021 quy định trần học phí ngành Y năm 2023-2024 là bao nhiêu?")
    assert _first(events, "agent.route.decided")["unlisted_document"] == "Nghị định 81/2021"
    assert prompts and "Nghị định 81/2021" in prompts[-1] and "not among their indexed documents" in prompts[-1]
    # The library may still report the subject (a newer decree citing it).
    assert _first(events, "retrieval.started") is not None, _names(events)
    assert "run.completed" in _names(events) and answer.strip()



@pytest.mark.parametrize("followup", [
    "Còn bài kia thì sao, nó đánh giá trên bộ dữ liệu nào cho phân tích cảm xúc?",
    "Going back to the first paper, which tools does it call?",
])
def test_vague_reference_is_resolved_by_the_router_not_a_shortcut(harness, followup):
    _events, cid, _ = harness.turn("DualSQL dùng những tool nào?")
    if "first paper" in followup:
        _events, cid, _ = harness.turn("Bài ReToolSQL dùng những tool nào?", cid)
        target, title = "dualsql", "DualSQL"
    else:
        target, title = "retoolsql", "ReToolSQL"
    harness.llm.router_reply = lambda prompt: {
        "route": "file_qa", "selected_tools": ["search_local_docs"], "confidence": "high",
        "scope": "named_documents", "referent_documents": [_label(prompt, title)], "reason": "vague_reference",
    }
    events, _cid, answer = harness.turn(followup, cid)
    assert _first(events, "retrieval.started")["focus_document_ids"] == [harness.ids[target]], (
        _first(events, "agent.route.decided")["reason"])
    assert "run.completed" in _names(events) and answer.strip()


def test_reference_that_could_mean_several_documents_asks_which_one(harness):
    _events, cid, _ = harness.turn("DualSQL dùng những tool nào?")
    _events, cid, _ = harness.turn("Bài ReToolSQL dùng những tool nào?", cid)
    harness.llm.router_reply = lambda prompt: {
        "route": "file_qa", "selected_tools": ["search_local_docs"], "confidence": "low",
        "scope": "active_document", "referent_documents": [],
        "ambiguous_referents": [_label(prompt, "DualSQL"), _label(prompt, "ReToolSQL")],
        "reason": "ambiguous_other_paper",
    }
    events, _cid, answer = harness.turn("Còn bài kia thì sao?", cid)
    assert _first(events, "retrieval.started") is None, _names(events)
    assert "DualSQL" in answer and "ReToolSQL" in answer and "?" in answer
    assert "run.completed" in _names(events)


def test_switched_document_note_names_the_new_document_only_on_a_switch() -> None:
    from app.api.agent import _switched_document_note

    docs = [{"document_id": "b", "title": "Lecture 24: Dataset shift"}]
    note = _switched_document_note({"a": "Lecture 13: Mammography"}, ["b"], docs)
    assert "Lecture 24" in note and "Lecture 13" in note
    assert _switched_document_note({"b": "Lecture 24"}, ["b"], docs) == ""
    assert _switched_document_note({}, ["b"], docs) == ""
