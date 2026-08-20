"""Hybrid LLM intent router — mock payloads + soft policy gates."""

from __future__ import annotations

import asyncio

import pytest

from app.api.agent import _phase4_routing_decision, _should_run_local_fallback
from app.llm.ollama_client import OllamaError
from app.services.tool_decision_service import (
    IntentRouterService,
    ToolDecision,
    answer_needs_local_fallback,
    decision_from_payload,
)


class _FakeChatClient:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    async def chat(self, **kwargs):  # noqa: ANN003
        self.calls += 1

        class _Completion:
            def __init__(self, text: str) -> None:
                self.message = text

        return _Completion(self.message)


def test_file_qa_mode_forces_tools_without_llm() -> None:
    client = _FakeChatClient('{"route":"chat","selected_tools":[]}')
    router = IntentRouterService(client=client, default_model="test")
    decision = asyncio.run(
        router.decide(
            task="anything",
            mode="file_qa",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
        )
    )
    assert decision.route == "file_qa"
    assert decision.use_local_retrieval is True
    assert "search_local_docs" in decision.selected_tools
    assert client.calls == 0


def test_pre_resolved_catalog_scope_bypasses_llm_router_and_keeps_all_documents() -> None:
    client = _FakeChatClient('malformed response that must never be called')
    router = IntentRouterService(client=client, default_model="test")
    decision = asyncio.run(
        router.decide(
            task="msf ser versus wav2small",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
            resolved_document_ids=["doc-msf", "doc-wav2small"],
        )
    )
    assert decision.route == "file_qa"
    assert decision.answer_intent == "compare"
    assert decision.reason == "catalog_document_scope"
    assert decision.use_local_retrieval is True
    assert client.calls == 0


@pytest.mark.parametrize(
    "task",
    [
        "Đưa abstract của ASPIRE và KST",
        "Cho tôi dataset của ASPIRE and KST",
        "Liệt kê bảng trong ASPIRE với KST",
    ],
)
def test_multidocument_coverage_does_not_force_comparison_intent(task: str) -> None:
    client = _FakeChatClient('malformed response that must never be called')
    router = IntentRouterService(client=client, default_model="test")
    decision = asyncio.run(
        router.decide(
            task=task,
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
            resolved_document_ids=["doc-aspire", "doc-kst"],
        )
    )

    assert decision.route == "file_qa"
    assert decision.answer_intent == "direct_answer"
    assert decision.reason == "catalog_document_scope"
    assert client.calls == 0


@pytest.mark.parametrize(
    "task",
    [
        "đối chiếu ASPIRE và KST",
        "ASPIRE và KST khác với nhau thế nào?",
        "phân biệt ASPIRE với KST",
        "differences between ASPIRE and KST",
        "similarities between ASPIRE and KST",
        "ASPIRE against KST",
        "contrast ASPIRE and KST",
        "ASPIRE hay KST tốt hơn?",
    ],
)
def test_pre_resolved_comparison_uses_one_shared_language_policy(task: str) -> None:
    client = _FakeChatClient('malformed response that must never be called')
    decision = asyncio.run(
        IntentRouterService(client=client, default_model="test").decide(
            task=task,
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
            resolved_document_ids=["doc-aspire", "doc-kst"],
        )
    )

    assert decision.answer_intent == "compare"
    assert client.calls == 0


def test_llm_router_chat_casual() -> None:
    client = _FakeChatClient(
        '{"route":"chat","selected_tools":[],"answer_intent":"direct_answer",'
        '"answer_depth":"brief","confidence":"high","needs_fallback":false,"reason":"casual_chat"}'
    )
    decision = asyncio.run(
        _phase4_routing_decision(
            task="sủa đi",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            router=IntentRouterService(client=client),
        )
    )
    assert decision["route"] == "chat"
    assert decision["use_local_retrieval"] is False
    assert decision["selected_tools"] == []
    assert decision["answer_intent"] == "direct_answer"
    assert decision["answer_depth"] == "brief"
    assert decision["reason"] == "casual_chat"


@pytest.mark.parametrize(
    "task",
    ["Thank you!", "Thanks a lot", "OK", "Okay.", "Got it", "Cảm ơn nhé!", "Ừ"],
)
def test_social_acknowledgement_bypasses_retrieval_even_with_active_focus(
    task: str,
) -> None:
    client = _FakeChatClient('not used')
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task=task,
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=True,
            allowed_tools=["search_local_docs"],
            recent_document_ids=["doc-aspire"],
            working_topic="ASPIRE",
            working_filenames=["ASPIRE.pdf"],
        )
    )
    assert decision.route == "chat"
    assert decision.selected_tools == []
    assert decision.reason == "deterministic_social_acknowledgement"
    assert client.calls == 0


def test_llm_router_paper_entity() -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"elaborate",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,"reason":"paper_entity"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="WhiSER là gì?",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
            resolved_document_ids=["doc-whiser"],
        )
    )
    assert decision.route == "file_qa"
    assert decision.use_local_retrieval is True
    assert decision.answer_intent == "direct_answer"


def test_llm_router_figure_selects_visual() -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs","retrieve_visual_assets"],'
        '"answer_intent":"infer_structure","answer_depth":"detailed","confidence":"high",'
        '"needs_fallback":false,"reason":"figure_question"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="Figure 1 architecture của ASPIRE",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=["search_local_docs", "retrieve_visual_assets"],
            resolved_document_ids=["doc-aspire"],
        )
    )
    assert decision.selected_tools == ["search_local_docs", "retrieve_visual_assets"]
    assert decision.answer_intent == "direct_answer"


def test_llm_router_followup_with_recent_retrieval() -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"elaborate",'
        '"answer_depth":"detailed","confidence":"medium","needs_fallback":false,"reason":"doc_followup"}'
    )
    decision = asyncio.run(
        _phase4_routing_decision(
            task="giải thích rõ hơn đi",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=True,
            router=IntentRouterService(client=client),
        )
    )
    assert decision["route"] == "file_qa"
    assert decision["use_local_retrieval"] is True


def test_document_resume_is_routed_to_sticky_retrieval() -> None:
    client = _FakeChatClient(
        '{"route":"chat","selected_tools":[],"confidence":"high","reason":"wrong_chat"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="Quay lại paper lúc nãy, benchmark Acc F1 CCC thế nào?",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=True,
            allowed_tools=["search_local_docs"],
            recent_document_ids=["doc-aspire"],
            working_topic="ASPIRE",
            working_filenames=["ASPIRE.pdf"],
        )
    )
    assert decision.route == "file_qa"
    assert decision.selected_tools == ["search_local_docs"]
    assert decision.reason == "resume_document_thread"
    assert client.calls == 0


def test_explicit_structured_paper_request_skips_provider_router() -> None:
    client = _FakeChatClient('not used')
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="Bài ASPIRE dùng dataset nào thế? Cho tôi bảng kết quả đi",
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=["search_local_docs", "retrieve_visual_assets"],
        )
    )

    assert decision.route == "file_qa"
    assert decision.answer_intent == "direct_answer"
    assert decision.reason == "explicit_structured_document_request"
    assert decision.selected_tools == ["search_local_docs"]
    assert client.calls == 0


def test_structured_followup_on_active_paper_skips_provider_router() -> None:
    client = _FakeChatClient('not used')
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="dataset nào, đưa bảng kết quả đi",
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=True,
            allowed_tools=["search_local_docs"],
            recent_document_ids=["doc-aspire"],
            working_topic="ASPIRE",
            working_filenames=["ASPIRE.pdf"],
        )
    )

    assert decision.route == "file_qa"
    assert decision.reason == "focused_structured_document_followup"
    assert client.calls == 0


def test_cross_paper_comparison_still_uses_provider_router() -> None:
    client = _FakeChatClient(
        '{"route":"research","selected_tools":["search_local_docs"],'
        '"answer_intent":"compare","answer_depth":"normal","confidence":"high",'
        '"needs_fallback":false,"reason":"cross_paper_compare"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="Trong thư viện của tôi, so sánh bảng kết quả bài ASPIRE với bài MSF-SER",
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=["search_local_docs"],
        )
    )

    assert decision.route == "research"
    assert decision.answer_intent == "compare"
    assert client.calls == 1


def test_allowlist_filters_visual_tool() -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs","retrieve_visual_assets"],'
        '"answer_intent":"direct_answer","answer_depth":"normal","confidence":"high",'
        '"needs_fallback":false,"reason":"local_docs"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="có hình không?",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=True,
            allowed_tools=["search_local_docs"],
            recent_document_ids=["doc-aspire"],
        )
    )
    assert decision.selected_tools == ["search_local_docs"]


def test_visual_without_local_search_becomes_chat() -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["retrieve_visual_assets"],'
        '"answer_intent":"direct_answer","answer_depth":"normal","confidence":"high",'
        '"needs_fallback":false,"reason":"visual_only"}'
    )
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="hình architecture",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=True,
            allowed_tools=["retrieve_visual_assets"],
            recent_document_ids=["doc-aspire"],
        )
    )
    assert decision.route == "chat"
    assert decision.use_local_retrieval is False
    assert decision.selected_tools == []


@pytest.mark.parametrize(
    "task",
    [
        "Use the class materials below to write thesis statements about data justice.",
        "Analyze the provided context and suggest an argument.",
        "Read the text below and help me edit it.",
        "Dựa trên tài liệu môn học bên dưới, giúp tôi lập luận về công bằng dữ liệu.",
    ],
)
def test_llm_cannot_activate_local_rag_for_unscoped_external_material(task: str) -> None:
    client = _FakeChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs"],'
        '"answer_intent":"elaborate","answer_depth":"detailed",'
        '"confidence":"high","needs_fallback":false,"reason":"document_request"}'
    )

    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task=task,
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=["search_local_docs"],
        )
    )

    assert decision.route == "chat"
    assert decision.selected_tools == []
    assert decision.reason == "unscoped_local_retrieval_rejected"
    assert client.calls == 1


def test_explicit_local_library_discovery_can_use_rag() -> None:
    client = _FakeChatClient(
        '{"route":"research","selected_tools":["search_local_docs"],'
        '"answer_intent":"compare","answer_depth":"detailed",'
        '"confidence":"high","needs_fallback":false,"reason":"library_discovery"}'
    )

    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="Which papers in my library use IEMOCAP?",
            mode="auto",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=["search_local_docs"],
        )
    )

    assert decision.route == "research"
    assert decision.selected_tools == ["search_local_docs"]
    assert decision.reason == "library_discovery"


def test_parse_failure_needs_fallback() -> None:
    client = _FakeChatClient("not json at all")
    decision = asyncio.run(
        IntentRouterService(client=client).decide(
            task="??? ",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            allowed_tools=None,
        )
    )
    assert decision.route == "chat"
    assert decision.needs_fallback is True
    assert decision.reason == "router_parse_or_llm_failure"


def test_router_provider_failure_stops_instead_of_guessing_or_switching_model() -> None:
    class _QuotaClient:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise OllamaError("LLM request failed (429): usage limit reached")

    with pytest.raises(OllamaError, match="429.*usage limit"):
        asyncio.run(
            IntentRouterService(client=_QuotaClient()).decide(
                task="ASPIRE benchmark",
                mode="research",
                previous_messages=[],
                has_recent_retrieval=False,
                allowed_tools=None,
            )
        )


def test_decision_from_payload_research_rounds() -> None:
    decision = decision_from_payload(
        {
            "route": "research",
            "selected_tools": ["search_local_docs"],
            "answer_intent": "compare",
            "answer_depth": "detailed",
            "confidence": "medium",
            "needs_fallback": False,
            "reason": "compare_papers",
        },
        allowed_tools=None,
    )
    assert decision.route == "research"
    assert decision.max_tool_rounds == 2
    assert decision.answer_intent == "compare"


def test_phase4_payload_injection() -> None:
    decision = asyncio.run(
        _phase4_routing_decision(
            task="ignored",
            mode="research",
            previous_messages=[],
            has_recent_retrieval=False,
            payload={
                "route": "chat",
                "selected_tools": [],
                "answer_intent": "example",
                "answer_depth": "brief",
                "confidence": "high",
                "needs_fallback": False,
                "reason": "injected",
            },
        )
    )
    assert decision["answer_intent"] == "example"
    assert decision["reason"] == "injected"


def test_uncertain_answer_can_trigger_local_fallback() -> None:
    assert answer_needs_local_fallback("Tôi không chắc về phần này.", "ASR model này là gì?") is True
    decision = ToolDecision(
        route="chat",
        selected_tools=[],
        reason="uncertain",
        confidence="medium",
        needs_fallback=False,
    )
    assert _should_run_local_fallback(
        tool_decision=decision,
        answer="Tôi không chắc về phần này.",
        task="ASR model này là gì?",
    )
    high_chat = ToolDecision(
        route="chat",
        selected_tools=[],
        reason="casual",
        confidence="high",
        needs_fallback=False,
    )
    assert not _should_run_local_fallback(
        tool_decision=high_chat,
        answer="Tôi không chắc.",
        task="sủa đi",
    )


def test_web_search_never_injected_when_disabled() -> None:
    decision = decision_from_payload(
        {
            "route": "chat",
            "selected_tools": ["web_search"],
            "confidence": "low",
            "needs_fallback": False,
            "reason": "current_info",
        },
        allowed_tools=[],
    )
    assert "web_search" not in decision.selected_tools
