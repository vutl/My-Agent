from app.services.query_rewrite_service import (
    QueryRewriteService,
    _visual_followup_rewrite,
    enrich_retrieval_query,
    has_result_table_intent,
    has_visual_intent,
    wants_single_figure,
)
import asyncio
from types import SimpleNamespace


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def test_visual_intent_detected() -> None:
    assert has_visual_intent("thêm các hình khác đi") is True


def test_model_phrase_is_not_mistaken_for_visual_intent() -> None:
    assert has_visual_intent("kiến trúc tổng thể của mô hình ASPIRE") is False
    assert has_visual_intent("mô hình này hoạt động ra sao?") is False
    assert has_visual_intent("cho xem hình kiến trúc của mô hình") is True
    assert has_visual_intent("mô hình có sơ đồ không?") is True


def test_single_figure_intent_handles_best_and_explicit_one_requests() -> None:
    assert wants_single_figure(
        "Cho mình figure hoặc sơ đồ kiến trúc phù hợp nhất, đừng lấy logo."
    ) is True
    assert wants_single_figure("Show exactly one architecture figure") is True
    assert wants_single_figure("Show the most relevant diagram") is True


def test_single_figure_intent_does_not_collapse_plural_requests() -> None:
    assert wants_single_figure("Cho mình top 3 figure phù hợp nhất") is False
    assert wants_single_figure("Thêm các hình khác đi") is False
    assert wants_single_figure("Show two architecture diagrams") is False


def test_enrich_retrieval_query_uses_visual_anchors() -> None:
    enriched = enrich_retrieval_query(
        "giải thích wav2small thêm cả hình nữa đi",
        topic="wav2small",
        entities=["wav2small"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-wav2small"],
    )
    assert "figure" in enriched
    assert "diagram" in enriched


def test_visual_followup_rewrite_resolves_topic_from_history() -> None:
    messages = [
        _Msg("user", "giải thích wav2small đi"),
        _Msg("assistant", "Wav2Small là model nhỏ cho SER A/D/V"),
    ]
    rewrite = _visual_followup_rewrite("thêm các hình khác đi", messages)
    assert rewrite is not None
    topic, standalone_query, entities = rewrite
    assert topic.lower() == "wav2small"
    assert "figure" in standalone_query
    assert entities == [topic]


def test_topic_from_history_prefers_most_recent_paper() -> None:
    from app.services.query_rewrite_service import _topic_from_history

    history = (
        "user: Pitch-fusion architecture là gì?\n"
        "assistant: Pitch-fusion dùng Wav2Vec 2.0 và pitch encoder.\n"
        "user: thế ASPIRE đi\n"
        "assistant: ASPIRE là multimodal SER với arousal-valence guidance."
    )
    assert _topic_from_history(history) == "ASPIRE"


def test_visual_followup_after_aspire_keeps_aspire_not_pitch() -> None:
    import asyncio
    from app.services.query_rewrite_service import QueryRewriteService

    class _NoLLM:
        async def chat(self, **kwargs):
            raise AssertionError("should not call LLM")

    messages = [
        _Msg("user", "architecture của Pitch-fusion"),
        _Msg("assistant", "Pitch-fusion fuse pitch với Wav2Vec 2.0."),
        _Msg("user", "thế ASPIRE đi, tôi tò mò về model này"),
        _Msg("assistant", "ASPIRE dùng RoBERTa + WavLM và A/V guidance."),
    ]

    async def _run():
        return await QueryRewriteService(client=_NoLLM(), default_model="x").rewrite(
            query="có hình ảnh gì về architecture hay gì không",
            previous_messages=messages,
        )

    result = asyncio.run(_run())
    assert result.current_topic == "ASPIRE"
    assert result.use_last_sources is True
    assert result.diagnostics.get("reason") == "visual_followup"



def test_enrich_retrieval_query_adds_architecture_anchors_without_visual_cue() -> None:
    enriched = enrich_retrieval_query(
        "giải thích wav2small đi",
        topic="wav2small",
        entities=["wav2small"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-wav2small"],
    )
    assert "architecture" in enriched
    assert "introduction" in enriched


def test_enrich_retrieval_query_keeps_result_ask_table_shaped() -> None:
    enriched = enrich_retrieval_query(
        "ASPIRE benchmark comparison table Acc F1 CCC",
        topic="ASPIRE",
        entities=["ASPIRE"],
        answer_intent="compare",
        focus_document_ids=["doc-aspire"],
    )

    assert has_result_table_intent(enriched) is True
    assert "table results metrics" in enriched
    assert "architecture" not in enriched
    assert "introduction" not in enriched


def test_generic_result_query_prefers_main_results_without_ablation_bias() -> None:
    enriched = enrich_retrieval_query(
        "Tôi muốn xem bảng result của bài MSF-SER",
        topic="MSF-SER",
        entities=["MSF-SER"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-msf-ser"],
    )

    assert "main experimental table results metrics" in enriched
    assert "ablation" not in enriched


def test_generic_benchmark_chat_does_not_get_paper_metric_expansion() -> None:
    class _RewriteClient:
        calls = 0

        async def chat(self, **kwargs):  # noqa: ANN003
            self.calls += 1
            return SimpleNamespace(
                message=(
                    '{"standalone_query":"IBM Cloud benchmark examples",'
                    '"is_followup":true,"current_topic":"IBM Cloud",'
                    '"required_entities":["IBM Cloud"],"use_last_sources":true,'
                    '"answer_intent":"direct_answer","answer_depth":"normal"}'
                )
            )

    client = _RewriteClient()
    result = asyncio.run(
        QueryRewriteService(client=client, default_model="test").rewrite(
            query="Can you show another IBM benchmark?",
            previous_messages=[
                _Msg("user", "How do IBM Cloud benchmarks work?"),
                _Msg("assistant", "They measure cloud workload behavior."),
            ],
        )
    )

    assert client.calls == 1
    assert result.rewrite_used is True
    assert result.standalone_query == "IBM Cloud benchmark examples"
    assert "Acc F1 CCC" not in result.standalone_query


def test_explicit_ablation_query_keeps_ablation_anchor() -> None:
    enriched = enrich_retrieval_query(
        "đưa bảng ablation của MSF-SER",
        topic="MSF-SER",
        entities=["MSF-SER"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-msf-ser"],
    )

    assert "ablation table results metrics" in enriched


def test_result_table_intent_uses_metric_boundaries() -> None:
    assert has_result_table_intent("benchmark Acc F1 CCC") is True
    assert has_result_table_intent("How do I access this API?") is False


def test_enrich_retrieval_query_skips_casual_chat() -> None:
    query = "hôm nay trời đẹp không?"
    assert (
        enrich_retrieval_query(
            query,
            topic=None,
            entities=[],
            answer_intent="direct_answer",
            focus_document_ids=None,
        )
        == query
    )
