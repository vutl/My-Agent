from pathlib import Path
"""Registry-derived capabilities, with chat/agent prompts isolated from RAG."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from app.agents.graph import FINAL_SYSTEM_PROMPT, _general_chat_prompt, _pick_system_prompt
from app.agents.tool_loop import agent_system_prompt, run_tool_loop
from app.llm.ollama_client import StreamChunk
from app.services.agent_service import AgentService
from app.tools import build_registry
from app.tools.capabilities import capability_groups, capability_prompt, capability_sheet
from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult


async def _searcher(query, top_k):
    raise AssertionError("Describing tools must never search the library")


async def _handler(arguments, context):
    return ToolResult(content="ok", summary="ok")


def _tool(name, group="other", label=""):
    return Tool(name, "description", {}, _handler, label=label, group=group)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("language", ["vi", "en"])
def test_sheet_lists_every_registered_group(platform, language, monkeypatch):
    monkeypatch.setattr(sys, "platform", platform)
    registry = build_registry(library_searcher=_searcher)
    expected = {"web", "files", "code", "reminders", "library", "assistant", "research"}
    if platform == "darwin":
        expected.update({"apple", "apps", "music"})
    assert set(capability_groups(registry)) == expected
    assert all(registry.get(name).group != "other" for name in registry.names())
    sheet = capability_sheet(registry, language)
    assert len(sheet) <= 1200
    assert len(sheet.splitlines()) == len(expected) + 1
    for group in expected:
        single_group = ToolRegistry()
        single_group.register(next(registry.get(name) for name in registry.names()
                                   if registry.get(name).group == group))
        assert capability_sheet(single_group, language).splitlines()[0] in sheet


def test_sheet_languages_and_approvals():
    registry = build_registry()
    vi = capability_sheet(registry)
    en = capability_sheet(registry, "en")
    assert "tìm thông tin mới" in vi and "Việc có rủi ro tôi sẽ hỏi cậu trước." in vi
    assert "current information" in en and "I will ask you before risky actions." in en
    assert vi != en
    assert capability_sheet(registry, "Vietnamese") == vi
    assert capability_sheet(registry, "English") == en


def test_dropping_last_tool_in_group_removes_it_from_sheet():
    full, reduced = ToolRegistry(), ToolRegistry()
    full.register(_tool("search", "web"))
    full.register(_tool("play", "music"))
    reduced.register(full.get("search"))
    assert "Spotify" in capability_sheet(full)
    assert capability_groups(reduced) == ["web"]
    assert "Spotify" not in capability_sheet(reduced)
    assert "Spotify" not in capability_prompt("vi", reduced)
    assert "Thư viện" not in capability_prompt("vi", reduced)


def test_unknown_group_uses_labels_and_old_constructor_still_works():
    registry = ToolRegistry()
    old_style = Tool("first", "description", {}, _handler, "auto", "Dịch văn bản")
    assert old_style.group == "other"
    registry.register(old_style)
    registry.register(_tool("second", label="Đọc chính tả"))
    registry.register(_tool("third", "custom", "Export slides"))
    registry.register(_tool("no_label", "custom"))
    sheet = capability_sheet(registry)
    assert "Dịch văn bản, Đọc chính tả" in sheet
    assert "Export slides, no_label" in sheet
    assert capability_groups(registry) == ["other", "custom"]


def test_cache_is_per_registry_and_language_and_updates_after_registration():
    registry = ToolRegistry()
    registry.register(_tool("search", "web"))
    sheet = capability_sheet(registry)
    assert capability_sheet(registry) is sheet
    assert capability_sheet(registry, "en") != sheet
    other = ToolRegistry()
    assert "Web:" not in capability_sheet(other)
    registry.register(_tool("play", "music"))
    assert "Spotify" in capability_sheet(registry)
    assert capability_sheet(registry) != sheet


def test_empty_registry_does_not_invent_groups():
    registry = ToolRegistry()
    assert capability_groups(registry) == []
    assert capability_sheet(registry) == "Việc có rủi ro tôi sẽ hỏi cậu trước."


@pytest.mark.parametrize("language, header, library", [
    ("Vietnamese", "Những việc Aya làm được", "Thư viện:"),
    ("English", "What Aya can do", "Library:"),
])
def test_chat_system_has_capabilities_and_always_present_library(language, header, library):
    prompt = _pick_system_prompt(None, route="chat", language=language)
    assert header in prompt and library in prompt
    assert capability_sheet(build_registry(), language) in prompt
    if language == "English":
        assert "never claim you cannot do a listed capability" in prompt
    else:
        assert "không nói tôi không làm được việc đã liệt kê" in prompt
    chat_prompt = _general_chat_prompt(user_task="hello", conversation_context="", answer_language=language,
                                       style_instruction="")
    assert "unless relevant to the user's request or a question about your capabilities" in chat_prompt


@pytest.mark.parametrize("language, header", [
    ("Vietnamese", "Những việc Aya làm được"), ("English", "What Aya can do"),
])
def test_agent_system_keeps_tool_lines_and_has_same_sheet(language, header):
    registry = build_registry(library_searcher=_searcher)
    tool_lines = [f"{name}: {registry.get(name).description}" for name in registry.names()]
    prompt = agent_system_prompt(language, tool_lines, registry=registry)
    assert header in prompt and capability_sheet(registry, language) in prompt
    assert all(line in prompt for line in tool_lines)


@pytest.mark.parametrize("route", ["chat", "file_qa", "research"])
@pytest.mark.parametrize("task, language, header", [
    ("Cậu có những khả năng nào?", "Vietnamese", "Những việc Aya làm được"),
    ("What are your capabilities?", "English", "What Aya can do"),
])
def test_actual_answer_system_has_sheet_only_for_chat(route, task, language, header):
    seen = []

    class Client:
        async def stream_chat(self, **kwargs):
            seen.extend(kwargs["messages"])
            yield StreamChunk("answer", done=True)

    async def run():
        service = AgentService(client=Client(), default_model="test")
        result = await service.run_graph(
            run_id="r", conversation_id="c", user_message_id="m", task=task, mode="auto",
            model=None, temperature=0, retrieved_docs=[{"filename": "document.md", "content": "evidence"}],
            tool_decision={"route": route, "selected_tools": [] if route == "chat" else ["search_local_docs"]},
        )
        from app.agents.graph import _answer_language

        assert _answer_language(task) == language
        return [chunk async for chunk in service.stream_final_answer(
            prompt=result.final_prompt, model=None, temperature=0,
            route=result.route, answer_language=_answer_language(task),
        )]

    asyncio.run(run())
    system = seen[0]["content"]
    if route == "chat":
        assert capability_prompt(language) in system and header in system
    else:
        assert system == FINAL_SYSTEM_PROMPT
        assert header not in seen[1]["content"]
        assert capability_sheet(build_registry(), language) not in seen[1]["content"]


def test_tool_loop_describes_only_enabled_tools(tmp_path):
    registry = ToolRegistry()
    registry.register(_tool("search", "web"))
    registry.register(_tool("play", "music"))
    seen = []

    class Client:
        async def chat_with_tools(self, **kwargs):
            seen.extend(kwargs["messages"])
            return SimpleNamespace(content="answer", tool_calls=[])

    async def run():
        return [event async for event in run_tool_loop(
            client=Client(), model="test", task="hello", history=[], registry=registry,
            tool_names=["search"], context=ToolContext("r", "c", tmp_path), language="English",
        )]

    asyncio.run(run())
    assert "What Aya can do" in seen[0]["content"]
    assert "Web:" in seen[0]["content"]
    assert "Spotify" not in seen[0]["content"]


def test_router_prompt_sends_live_information_to_the_agent() -> None:
    from app.services import tool_decision_service

    source = Path(tool_decision_service.__file__).read_text(encoding="utf-8")
    assert "Live or current information" in source
    assert "Never route such a question to chat" in source

