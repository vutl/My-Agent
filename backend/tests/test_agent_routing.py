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


def test_named_but_unresolved_paper_request_asks_the_router_and_falls_back_to_file_qa() -> None:
    client = _FakeChatClient('not json')
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
    assert decision.reason == "document_turn_fallback"
    assert decision.selected_tools == ["search_local_docs"]
    assert client.calls == 1


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


def _router_returning(payload: str):
    from app.services.tool_decision_service import IntentRouterService

    class _Client:
        def __init__(self):
            self.prompts = []

        async def chat(self, **kwargs):  # noqa: ANN003
            self.prompts.append(kwargs["messages"][-1]["content"])

            class _Completion:
                message = payload

            return _Completion()

    client = _Client()
    return IntentRouterService(client=client, default_model="test"), client


LIBRARY = [("doc-a", "Graph Attention for Traffic Forecasting (traffic.pdf)"),
           ("doc-b", "A Robust Fusion Model for Speech in Tonal Languages (tonal.pdf)")]


def test_router_sees_the_library_and_maps_its_choice_to_a_document():
    router, client = _router_returning(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"direct_answer",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,'
        '"library_document":"L2","reason":"described_library_document"}'
    )
    decision = asyncio.run(router.decide(
        task="Bài về ngôn ngữ có thanh điệu đóng góp gì?", mode="auto", previous_messages=[],
        has_recent_retrieval=False, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
    ))
    assert "L2: A Robust Fusion Model for Speech in Tonal Languages" in client.prompts[0]
    # A recognised library document is a local anchor: not downgraded to chat.
    assert decision.route == "file_qa" and decision.library_document_id == "doc-b"


def test_unknown_or_missing_library_label_keeps_unscoped_requests_in_chat():
    for label in ("null", '"L9"', '"nonsense"'):
        router, _client = _router_returning(
            '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"direct_answer",'
            '"answer_depth":"normal","confidence":"high","needs_fallback":false,'
            f'"library_document":{label},"reason":"generic"}}'
        )
        decision = asyncio.run(router.decide(
            task="Summarize the class materials", mode="auto", previous_messages=[],
            has_recent_retrieval=False, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
        ))
        assert decision.library_document_id is None
        assert decision.route == "chat"


def test_artifact_requests_without_a_paper_name_stay_on_the_active_paper():
    from app.api.agent import _asks_for_active_paper_artifact

    assert _asks_for_active_paper_artifact("Cho xem bảng so sánh kết quả trên IEMOCAP và LSSED.")
    assert _asks_for_active_paper_artifact("Show the figure with the training curves on MELD")
    assert not _asks_for_active_paper_artifact("Paper nào có bảng kết quả trên IEMOCAP?")
    assert not _asks_for_active_paper_artifact("Which papers report a table on LSSED?")
    assert not _asks_for_active_paper_artifact("IEMOCAP có bao nhiêu session?")


def test_library_wide_questions_are_detected():
    from app.api.agent import _is_library_wide_question

    for text in (
        "Paper nào KHÔNG dùng IEMOCAP?",
        "Which papers use Whisper?",
        "Thư viện của tôi có bao nhiêu paper về nhận diện cảm xúc?",
        "GPU dùng để train của từng paper là gì?",
        "Liệt kê tất cả các bài thử nghiệm trên MSP-Podcast",
    ):
        assert _is_library_wide_question(text), text
    for text in (
        "Cho xem bảng so sánh kết quả trên IEMOCAP và LSSED.", "ASPIRE dùng GPU gì?",
        "Compare these two papers on IEMOCAP.", "So sánh hai bài đó về kết quả.",
    ):
        assert not _is_library_wide_question(text), text


def test_library_scan_searches_every_document_and_lists_the_catalog():
    import asyncio
    from types import SimpleNamespace
    from app.api.agent import _retrieve_library_wide
    import app.api.agent as agent_module

    class _Rag:
        def library_titles(self, collection_id):
            return [("a", "Paper A"), ("b", "Paper B"), ("c", "Paper C")]

        def get_document(self, document_id):
            return {"filename": f"{document_id}.pdf"}

        async def search_hybrid(self, *, document_ids, **kwargs):
            document_id = document_ids[0]
            if document_id == "c":
                return {"results": []}
            return {"results": [{"document_id": document_id, "chunk_id": f"{document_id}1", "filename": f"{document_id}.pdf",
                                 "content": f"{document_id} trains on an RTX 3090.", "score": 0.5}]}

    settings = SimpleNamespace(rerank_enabled=False, rerank_mode="embedding", rerank_cross_encoder_path=None,
                               rerank_max_candidates=10)
    original_store, original_embeddings = agent_module.deps._retrieval_store, agent_module.deps._embedding_provider
    agent_module.deps._retrieval_store = lambda settings: None
    agent_module.deps._embedding_provider = lambda settings: None
    try:
        result = asyncio.run(_retrieve_library_wide(rag=_Rag(), settings=settings, query="GPU của từng paper", collection_id=None))
    finally:
        agent_module.deps._retrieval_store, agent_module.deps._embedding_provider = original_store, original_embeddings
    documents = result["documents"]
    assert result["mode"] == "library_scan" and "3 documents" in documents[0]["content"]
    assert "c.pdf" in documents[0]["content"]
    assert {item.get("document_id") for item in documents[1:]} == {"a", "b"}


def test_named_query_terms_pick_names_and_acronyms():
    from app.api.agent import _named_query_terms

    assert _named_query_terms("Những paper nào nói rõ GPU dùng, là GPU gì?") == ["GPU"]
    assert _named_query_terms("Which papers report results on MSP-Podcast and LSSED?") == ["MSP-Podcast", "LSSED"]
    assert _named_query_terms("Những paper nào dùng Whisper?") == ["Whisper"]


def test_missing_information_answers_trigger_full_document_reading():
    from app.api.agent import _answer_reports_missing_information as missing

    assert missing("À cậu ơi~ phần trích xuất hiện tại chưa nêu GPU dùng để train.")
    assert missing("The excerpts do not specify the learning-rate schedule.")
    assert missing("Paper không đề cập số giờ dữ liệu.")
    assert not missing("ASPIRE đạt F1 76.31% trên IEMOCAP; CMA không có WA trong bảng.")
    assert not missing("Here you go~ Table 1 with every model's Acc and F1.")


def test_results_ask_about_a_library_term_needs_retrieval():
    from app.api.agent import _asks_library_results_for_named_terms

    class _Rag:
        def documents_containing_all_terms(self, terms, collection_id=None):
            return ["crab", "aspire"] if terms == ["WavLM"] else []

    assert _asks_library_results_for_named_terms(_Rag(), "Cho mình kết quả của baseline WavLM.", None)
    assert not _asks_library_results_for_named_terms(_Rag(), "Kể chuyện cười về WavLM đi", None)
    assert not _asks_library_results_for_named_terms(_Rag(), "Thủ đô của Úc là gì?", None)

    class _Everything:
        def documents_containing_all_terms(self, terms, collection_id=None):
            return ["plan"]

    pasted = ("The results are as follows: Pairs Frequency 1 2 5 Find the mean and Complete the Relative "
              "Frequency column and the Cumulative column.")
    assert not _asks_library_results_for_named_terms(_Everything(), pasted, None)


def test_paper_described_by_terms_pattern():
    from app.api.agent import _PAPER_BY_TERMS_RE

    assert _PAPER_BY_TERMS_RE.search("Ý mình là bài có Medusa với MemoCMT ấy.")
    assert _PAPER_BY_TERMS_RE.search("the paper with Medusa and MemoCMT")
    assert not _PAPER_BY_TERMS_RE.search("ASPIRE đạt bao nhiêu trên IEMOCAP?")


def test_router_reports_scope_and_presentation():
    router, _client = _router_returning(
        '{"route":"research","selected_tools":["search_local_docs"],"answer_intent":"compare",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,"library_document":null,'
        '"scope":"library","presentation":"answer","reason":"library_question"}'
    )
    decision = asyncio.run(router.decide(
        task="Which papers do not train on IEMOCAP?", mode="auto", previous_messages=[],
        has_recent_retrieval=True, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
    ))
    assert decision.scope == "library" and decision.presentation == "answer"
    assert decision.to_dict()["scope"] == "library"

    router, _client = _router_returning(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"direct_answer",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,"library_document":"L2",'
        '"scope":"everything","presentation":"poster","reason":"x"}'
    )
    decision = asyncio.run(router.decide(
        task="Show the table of the tonal paper", mode="auto", previous_messages=[],
        has_recent_retrieval=False, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
    ))
    assert decision.scope is None and decision.presentation is None  # unknown values ignored


def test_sentence_initial_words_are_not_named_terms():
    from app.api.agent import _named_query_terms

    text = ("Forty randomly selected students were asked. Let X = pairs owned.\\nPairs of sneakers\\n"
            "Find the mean. Complete the Relative Frequency column.")
    assert "Let" not in _named_query_terms(text) and "Find" not in _named_query_terms(text)
    assert _named_query_terms("Which papers use Whisper and the GPU?") == ["Whisper", "GPU"]


def test_router_maps_every_described_document():
    router, _client = _router_returning(
        '{"route":"research","selected_tools":["search_local_docs"],"answer_intent":"compare",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,"library_document":null,'
        '"library_documents":["L1","L2","L99"],"scope":"named_documents","presentation":"answer","reason":"compare"}'
    )
    decision = asyncio.run(router.decide(
        task="Mô hình dùng WavLM làm global feature hay mô hình Wav2Vec 2.0 + CNN pitch cao hơn?", mode="auto",
        previous_messages=[], has_recent_retrieval=False, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
    ))
    assert len(decision.library_document_ids) == 2  # unknown label L99 dropped


def test_document_ids_in_a_message_are_explicit_references():
    from app.api.agent import _document_ids_named_in

    class _Rag:
        def get_document(self, document_id):
            return {"id": document_id} if document_id == "ebf777ce-7455-463a-aed0-b7f9b06bb79e" else None

    text = "Trích xuất nguyên văn Bảng 2 trong file EBF777CE-7455-463A-AED0-B7F9B06BB79E và 00000000-0000-0000-0000-000000000000"
    assert _document_ids_named_in(_Rag(), text, None) == ["ebf777ce-7455-463a-aed0-b7f9b06bb79e"]


def test_need_context_marker_is_held_back_and_removed():
    from app.api.agent import NEED_CONTEXT_MARKER, _split_marker_tail

    shown, tail = _split_marker_tail("The table stops after row 4.\n⟦need")
    assert shown == "The table stops after row 4.\n" and tail == "⟦need"
    shown, tail = _split_marker_tail("Answer." + NEED_CONTEXT_MARKER)
    assert shown == "Answer." and tail == ""


def test_dominant_document_needs_a_clear_majority():
    from app.api.agent import _dominant_document

    docs = [{"document_id": "rag"}] * 4 + [{"document_id": "lim"}]
    assert _dominant_document(docs) == ["rag"]
    assert _dominant_document([{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}]) == []


def test_guessed_focus_yields_to_the_document_the_evidence_points_to():
    import asyncio
    from types import SimpleNamespace
    import app.api.agent as agent_module

    class _Rag:
        async def search_hybrid(self, **kwargs):
            return {"results": [{"document_id": "cpi"}] * 4 + [{"document_id": "gdp"}] * 2}

    settings = SimpleNamespace(rerank_enabled=False, rerank_mode="embedding", rerank_cross_encoder_path=None,
                               rerank_max_candidates=10)
    store, embed = agent_module.deps._retrieval_store, agent_module.deps._embedding_provider
    agent_module.deps._retrieval_store = lambda settings: None
    agent_module.deps._embedding_provider = lambda settings: None
    try:
        switch = asyncio.run(agent_module._evidence_backed_focus(_Rag(), settings, "giá nhiên liệu", "labour", None))
        keep = asyncio.run(agent_module._evidence_backed_focus(_Rag(), settings, "giá nhiên liệu", "cpi", None))
    finally:
        agent_module.deps._retrieval_store, agent_module.deps._embedding_provider = store, embed
    # Never switch to the top-ranked document (dense documents dominate);
    # a guess that retrieval does see is kept.
    assert switch == ("drop", None) and keep == ("keep", None)


def test_guessed_focus_is_dropped_when_retrieval_never_sees_it():
    import asyncio
    from types import SimpleNamespace
    import app.api.agent as agent_module

    class _Rag:
        async def search_hybrid(self, **kwargs):
            return {"results": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "a"}, {"document_id": "a"}]}

    settings = SimpleNamespace(rerank_enabled=False, rerank_mode="embedding", rerank_cross_encoder_path=None,
                               rerank_max_candidates=10)
    store, embed = agent_module.deps._retrieval_store, agent_module.deps._embedding_provider
    agent_module.deps._retrieval_store = lambda settings: None
    agent_module.deps._embedding_provider = lambda settings: None
    try:
        verdict = asyncio.run(agent_module._evidence_backed_focus(_Rag(), settings, "q", "zzz", None))
        kept = asyncio.run(agent_module._evidence_backed_focus(_Rag(), settings, "q", "b", None))
    finally:
        agent_module.deps._retrieval_store, agent_module.deps._embedding_provider = store, embed
    assert verdict == ("drop", None) and kept == ("keep", None)


def test_truncation_phrases_trigger_whole_document_reading():
    from app.api.agent import _answer_reports_missing_information as missing

    assert missing("The excerpt of Table 2 is cut off after a few rows.")
    assert missing("Phần trích Table 2 bị cắt sau vài dòng đầu.")
    assert missing("The other rows are not visible in the extracted table.")


def test_translation_variant_is_added_once():
    from app.api.agent import _with_translation_variant

    base = [("original", "Census bán lẻ tháng 8 tăng bao nhiêu?")]
    out = _with_translation_variant(base, "How much did Census retail sales rise in August?", base[0][1])
    assert out[-1] == ("translation", "How much did Census retail sales rise in August?")
    assert _with_translation_variant(base, None, base[0][1]) == base
    assert _with_translation_variant(base, base[0][1], base[0][1]) == base


def test_router_returns_a_query_translation():
    router, _client = _router_returning(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"answer_intent":"direct_answer",'
        '"answer_depth":"normal","confidence":"high","needs_fallback":false,"library_document":"L2",'
        '"query_translation":"Which retail category rose the most in August?","reason":"x"}'
    )
    decision = asyncio.run(router.decide(
        task="Ngành bán lẻ nào tăng mạnh nhất tháng 8?", mode="auto", previous_messages=[],
        has_recent_retrieval=False, allowed_tools=["search_local_docs"], library_titles=LIBRARY,
    ))
    assert decision.query_translation == "Which retail category rose the most in August?"


def test_answer_language_detects_english_and_vietnamese_without_diacritics():
    from app.agents.graph import _answer_language

    assert _answer_language("Do the English and Vietnamese articles agree on GDP for 2024?") == "English"
    assert _answer_language("What is the main idea of this paper?") == "English"
    assert _answer_language("cai do do co dung khong") == "same as user"
    assert _answer_language("Bài này nói gì?") == "Vietnamese"


def test_answer_language_ignores_substrings_and_accented_names():
    from app.agents.graph import _answer_language

    assert _answer_language("How many units did Viet Nam have, broken down into wards and communes?") == "English"
    assert _answer_language("How many communes does Tuyên Quang have? The English and Vietnamese articles don't agree.") == "English"
    assert _answer_language("According to the article, when did Đồng Nai and Bắc Ninh become cities?") == "English"
    assert _answer_language("Paper này dùng dataset nào, accuracy là bao nhiêu?") == "Vietnamese"
    assert _answer_language("ok bro giai thich lai di") == "Vietnamese"


class _RecordingChatClient(_FakeChatClient):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.prompts: list[str] = []

    async def chat(self, **kwargs):  # noqa: ANN003
        self.prompts.append(kwargs["messages"][-1]["content"])
        return await super().chat(**kwargs)


_LIBRARY = [("doc-a", "DualSQL: Text-to-SQL with Multi-Agent RL"), ("doc-b", "ReToolSQL: Agentic RL for Text-to-SQL"),
            ("doc-c", "Lost in the Middle: How Language Models Use Long Contexts")]


def test_vague_reference_is_resolved_by_the_model_from_conversation_documents() -> None:
    client = _RecordingChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"referent_documents":["L2"],'
        '"library_documents":[],"scope":"active_document"}'
    )
    decision = asyncio.run(IntentRouterService(client=client, default_model="test").decide(
        task="Còn bài kia thì sao?", mode="research", previous_messages=[], has_recent_retrieval=True,
        allowed_tools=None, recent_document_ids=["doc-a"], working_filenames=["dualsql.pdf"],
        library_titles=_LIBRARY, conversation_documents=[("doc-a", True)], inferred_document_ids=["doc-a"],
    ))
    assert client.calls == 1  # a guessed scope does not skip the model
    assert decision.referent_document_ids == ("doc-b",)
    assert "conversation_documents:\n- L1: DualSQL" in client.prompts[0] and "(active)" in client.prompts[0]


def test_inferred_scope_stands_when_the_model_router_fails() -> None:
    client = _FakeChatClient("not json at all")
    decision = asyncio.run(IntentRouterService(client=client, default_model="test").decide(
        task="Nó dùng tool gì?", mode="research", previous_messages=[], has_recent_retrieval=True,
        allowed_tools=None, library_titles=_LIBRARY, inferred_document_ids=["doc-a"],
    ))
    assert decision.route == "file_qa" and decision.reason == "inferred_document_scope"


def test_conversation_documents_lists_threads_oldest_first_with_the_active_last() -> None:
    from types import SimpleNamespace

    from app.api.agent import _conversation_documents

    state = SimpleNamespace(
        active_document_ids=["doc-b"], referent_document_ids=[],
        recent_document_threads=(SimpleNamespace(document_ids=("doc-a",)), SimpleNamespace(document_ids=("doc-c", "doc-a"))),
    )
    assert _conversation_documents(state) == [("doc-c", False), ("doc-a", False), ("doc-b", True)]


def test_structured_followup_fast_path_yields_to_the_model_when_the_scope_is_guessed() -> None:
    client = _RecordingChatClient(
        '{"route":"file_qa","selected_tools":["search_local_docs"],"referent_documents":["L2"]}'
    )
    decision = asyncio.run(IntentRouterService(client=client, default_model="test").decide(
        task="Còn bài kia thì sao, nó đánh giá trên bộ dữ liệu nào?", mode="research", previous_messages=[],
        has_recent_retrieval=True, allowed_tools=None, recent_document_ids=["doc-a"], working_filenames=["a.pdf"],
        library_titles=_LIBRARY, conversation_documents=[("doc-a", True)], inferred_document_ids=["doc-a"],
    ))
    assert client.calls == 1 and decision.referent_document_ids == ("doc-b",)


def test_escalation_adds_each_documents_closest_sibling_within_the_cap() -> None:
    from app.api.agent import _answer_reports_missing_information, _with_sibling_documents

    class _Rag:
        siblings = {"adb-slides": "adb-report", "wb-oct": "wb-may", "adb-report": "adb-slides"}

        def closest_sibling_document(self, document_id, collection_id=None):
            return self.siblings.get(document_id)

    assert _with_sibling_documents(_Rag(), ["adb-slides", "wb-oct"], None) == ["adb-slides", "wb-oct", "adb-report"]
    assert _with_sibling_documents(_Rag(), ["adb-report"], None) == ["adb-report", "adb-slides"]
    assert _answer_reports_missing_information("The excerpts don’t show the median.")
    assert not _answer_reports_missing_information("The median was 3.8% [S2].")


def test_which_files_questions_span_the_library() -> None:
    from app.api.agent import _is_library_wide_question

    assert _is_library_wide_question("Which of my files cover wheat?")
    assert _is_library_wide_question("Những file nào nhắc tới USDA?")
    assert _is_library_wide_question("Which reports mention rice?")
    assert not _is_library_wide_question("hai bài đó dùng dataset nào")
    assert not _is_library_wide_question("What file is this?")


def test_router_prompt_distinguishes_casual_actions_from_ability_questions():
    from app.services.tool_decision_service import _ROUTER_SYSTEM_PROMPT

    assert "Casual or indirect requests to DO something with a registered capability still use route=agent" in _ROUTER_SYSTEM_PROMPT
    assert 'without saying "use a tool"' in _ROUTER_SYSTEM_PROMPT
    assert "Intent to act takes precedence over casual tone" in _ROUTER_SYSTEM_PROMPT
    assert "Questions ABOUT Aya's abilities" in _ROUTER_SYSTEM_PROMPT
    assert 'stay route=chat, selected_tools=[]' in _ROUTER_SYSTEM_PROMPT
    for example in ("mở nhạc đi", "bật bài…", "nhắc tớ…", "mở Safari", "tìm trên mạng giúp tớ", "tạo file…"):
        assert example in _ROUTER_SYSTEM_PROMPT


@pytest.mark.parametrize("task", [
    "mở nhạc đi", "bật bài nào vui vui", "nhắc tớ nghỉ sau 30 phút", "mở Safari",
    "tìm trên mạng giúp tớ thời tiết", "tạo file ghi chú cho tớ", "Some music would be nice",
])
@pytest.mark.parametrize("allowed", [True, False])
def test_casual_action_agent_decision_obeys_tool_allowlist(task, allowed):
    router, client = _router_returning(
        '{"route":"agent","selected_tools":["assistant_tools"],"reason":"requested_action"}'
    )
    decision = asyncio.run(router.decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"] if allowed else ["search_local_docs"],
    ))
    assert len(client.prompts) == 1
    assert decision.route == ("agent" if allowed else "chat")
    assert decision.selected_tools == (["assistant_tools"] if allowed else [])
    if not allowed:
        assert decision.reason == "requested_action_tool_not_allowed"
        assert decision.confidence == "low"


@pytest.mark.parametrize("task", ["cậu làm được gì?", "What can you do?"])
def test_capability_questions_accept_chat_with_assistant_tools_allowed(task):
    router, client = _router_returning('{"route":"chat","selected_tools":[],"reason":"abilities"}')
    decision = asyncio.run(router.decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"],
    ))
    assert len(client.prompts) == 1
    assert decision.route == "chat" and decision.selected_tools == []


@pytest.mark.parametrize("task", [
    "hey, play some music",
    "chào cậu, mở Safari giúp tớ",
    "hello, remind me in 30 minutes",
    "Hi Aya! Search the web for today's weather",
    "good morning create a notes file for me",
    "xin chào\nmở lịch cho tớ",
])
@pytest.mark.parametrize("allowed", [True, False])
def test_greeting_prefixed_actions_reach_router_and_obey_tool_allowlist(task, allowed):
    router, client = _router_returning(
        '{"route":"agent","selected_tools":["assistant_tools"],"reason":"requested_action"}'
    )
    decision = asyncio.run(router.decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"] if allowed else ["search_local_docs"],
    ))
    assert len(client.prompts) == 1
    assert decision.route == ("agent" if allowed else "chat")
    assert decision.selected_tools == (["assistant_tools"] if allowed else [])
    if not allowed:
        assert decision.reason == "requested_action_tool_not_allowed"
        assert decision.confidence == "low"


@pytest.mark.parametrize("task", [
    "hi", " Hello! ", "hey there!", "Chào cậu!", "Xin chào Aya.", "chao ban",
    "Good morning, Aya!", "good evening", "alo?",
])
def test_complete_greetings_bypass_router(task):
    client = _FakeChatClient("not used")
    decision = asyncio.run(IntentRouterService(client=client, default_model="test").decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"],
    ))
    assert client.calls == 0
    assert decision.route == "chat" and decision.selected_tools == []
    assert decision.reason == "deterministic_casual_chat"


@pytest.mark.parametrize("task", [
    "cậu thích nhạc gì, bật cho tớ nghe đi",
    "How are you? Play some music for me.",
    "cậu khoẻ không mở Safari giúp tớ",
    "cậu muốn ăn gì, tìm trên mạng giúp tớ một quán gần đây",
    "cậu có thích chơi nhạc không? Mở nhạc cho tớ nhé",
    "hôm nay thế nào nhắc tớ nghỉ sau 30 phút",
    "kể chuyện, tạo file lưu lại giúp tớ",
    "nói phét rồi mở lịch cho tớ",
    "Tell me a joke and save it in a notes file",
    "Who are you? Open Safari.",
    "What is your name? Remind me to stretch in 30 minutes.",
    "Thanks! Play some music for me.",
    "OK mở Safari giúp tớ",
])
@pytest.mark.parametrize("allowed", [True, False])
def test_social_prefixed_actions_reach_router_and_obey_tool_allowlist(task, allowed):
    router, client = _router_returning(
        '{"route":"agent","selected_tools":["assistant_tools"],"reason":"requested_action"}'
    )
    decision = asyncio.run(router.decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"] if allowed else ["search_local_docs"],
    ))
    assert len(client.prompts) == 1
    assert decision.route == ("agent" if allowed else "chat")
    assert decision.selected_tools == (["assistant_tools"] if allowed else [])
    assert decision.reason == ("requested_action" if allowed else "requested_action_tool_not_allowed")
    if not allowed:
        assert decision.confidence == "low"


@pytest.mark.parametrize("task", [
    "Cậu khoẻ không?", "cậu là ai?", "Bạn tên là gì?", "cậu thích gì?",
    "cậu muốn ăn gì?", "hôm nay thế nào?", "kể chuyện!", "nói phét",
    "How are you?", "Who are you?", "What is your name?", "Tell me a joke.",
])
def test_complete_social_messages_bypass_router(task):
    client = _FakeChatClient("not used")
    decision = asyncio.run(IntentRouterService(client=client, default_model="test").decide(
        task=task, mode="auto", previous_messages=[], has_recent_retrieval=False,
        allowed_tools=["assistant_tools"],
    ))
    assert client.calls == 0
    assert decision.route == "chat" and decision.selected_tools == []
    assert decision.reason == "deterministic_casual_chat"
