from app.services.query_rewrite_service import (
    QueryRewriteService,
    _visual_followup_rewrite,
    enrich_retrieval_query,
    has_result_table_intent,
    has_visual_intent,
    requires_visual_pixels,
    wants_single_figure,
)
from app.rag.paper_facets import extract_query_facets
import asyncio
from types import SimpleNamespace
import pytest


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


@pytest.mark.parametrize("question,expected_terms", [
    ("Bài Alpha dùng bộ dữ liệu nào?", ("dataset", "split", "experimental setup")),
    ("What datasets does Alpha use?", ("dataset", "split", "experimental setup")),
    ("Gamma: loss function và optimizer thế nào?", ("loss", "optimizer")),
    ("What are the limitations of Delta?", ("limitations", "failure cases")),
    ("Epsilon có đóng góp gì?", ("contributions", "novelty")),
])
def test_explicit_nonarchitecture_facets_do_not_get_architecture_default(question, expected_terms):
    enriched = enrich_retrieval_query(
        question, topic="Document", entities=["Document"],
        answer_intent="direct_answer", focus_document_ids=["doc"],
    )
    assert enriched.endswith(question)
    for term in expected_terms:
        assert term in enriched
    assert "architecture" not in enriched
    assert "distillation" not in enriched


def test_dataset_and_result_request_preserves_both_explicit_facets():
    enriched = enrich_retrieval_query(
        "Alpha dùng dataset nào, cho bảng kết quả luôn?", topic="Alpha", entities=["Alpha"],
        answer_intent="elaborate", focus_document_ids=["doc"],
    )
    assert "table results metrics" in enriched
    assert "dataset split" in enriched
    assert "architecture" not in enriched


@pytest.mark.parametrize("question", [
    "Does extra training on classification tasks affect performance?",
    "Compare performance across supervised tasks.",
    "What does performance in downstream tasks indicate?",
])
def test_contextual_task_nouns_do_not_add_paper_overview_obligations(question):
    from app.rag.paper_facets import requested_paper_facets, retrieval_query_expansion
    facets = requested_paper_facets(question)
    assert "benchmark_results" in facets
    assert "task" not in facets
    assert "contributions" not in facets
    expanded, _ = retrieval_query_expansion(question)
    assert "research problem" not in expanded


@pytest.mark.parametrize("question", [
    "Compare their tasks and benchmark results.",
    "On which tasks do they report performance?",
    "What tasks and datasets does ORION support?",
    "Bài này giải quyết bài toán gì, kết quả ra sao?",
])
def test_explicit_task_requests_survive_contextual_noun_filter(question):
    from app.rag.paper_facets import requested_paper_facets
    assert "task" in requested_paper_facets(question)


def test_explicit_architecture_and_dataset_request_keeps_both_facets():
    enriched = enrich_retrieval_query(
        "Describe the architecture and datasets of Alpha.", topic="Alpha", entities=["Alpha"],
        answer_intent="direct_answer", focus_document_ids=["doc"],
    )
    assert "architecture" in enriched
    assert "dataset split" in enriched


def test_visual_intent_detected() -> None:
    assert has_visual_intent("thêm các hình khác đi") is True
    assert has_visual_intent("what is shown in the picture on slide 6?") is True
    assert has_visual_intent("How many figures contain the left arrows?") is True
    assert has_visual_intent("Which stages are shown on slides 12 and 14?") is True
    assert has_visual_intent("How many people with red shirt are on Page 14?") is True
    assert has_visual_intent("What are the top2 texts of the yellow words?") is True


def test_research_topic_terms_are_not_pixel_requests() -> None:
    # A paper *about* multimodal/visual grounding is answered from its text;
    # treating topic words as pixel requests replaced answers with a
    # visual-insufficiency fallback whenever no image was attached.
    for query in (
        "How important is visual grounding for multilingual representations?",
        "Explain the pixel information used by the model",
        "What multimodal fusion strategy does the paper use?",
        "Mô hình đa phương thức này dùng image-text như thế nào?",
        "Show me the accuracy of the model",
        "Show the results table for IEMOCAP",
    ):
        assert has_visual_intent(query) is False, query
        assert requires_visual_pixels(query) is False, query
    assert requires_visual_pixels("Show me the architecture diagram") is True


def test_ordinary_text_questions_are_not_visual() -> None:
    # Weak cues (people/text/từ/trang/document) need an explicit visual scope.
    for query in (
        "Tóm tắt nội dung từ trang 3",
        "What text does the document say about training?",
        "Who is the person that wrote this document?",
        "Trang 5 nói về điều gì?",
    ):
        assert has_visual_intent(query) is False, query
        assert requires_visual_pixels(query) is False, query
    assert has_visual_intent("How many people appear in the photo?") is True
    assert has_visual_intent("What color is the shirt on page 14?") is True


def test_explanation_questions_are_not_result_table_requests() -> None:
    for query in (
        "What is the model architecture?",
        "Giải thích các thành phần của mô hình",
        "Nêu các bước của phương pháp",
        "What is the value of this approach?",
    ):
        assert has_result_table_intent(query) is False, query
    for query in (
        "Which languages are covered?",
        "Liệt kê các bộ dữ liệu được dùng",
        "Các mô hình nào được so sánh?",
        "What accuracy does the model reach?",
    ):
        assert has_result_table_intent(query) is True, query


def test_bare_model_word_does_not_add_architecture_facet() -> None:
    assert extract_query_facets("Which dataset was the model evaluated on?") == ["dataset_setup"]
    assert "architecture" not in extract_query_facets(
        "What results does the model achieve on IEMOCAP?"
    )
    assert "architecture" not in extract_query_facets("Mô hình đạt kết quả gì?")
    assert "architecture" in extract_query_facets("Mô hình được thiết kế ra sao?")
    assert "architecture" in extract_query_facets("Giải thích các thành phần của mô hình")


def test_visual_relation_requires_pixels_for_content_inside_image() -> None:
    assert requires_visual_pixels(
        "What is the utility derived from each item in the picture on slide 6?"
    ) is True
    assert requires_visual_pixels("Giá trị nào được thể hiện trong biểu đồ?") is True


def test_visual_caption_inventory_can_use_metadata_without_pixels() -> None:
    assert requires_visual_pixels("What is the caption of Figure 2?") is False
    assert requires_visual_pixels("List the figures in this paper") is False


def test_observable_visual_questions_require_pixels_without_figure_noun() -> None:
    assert requires_visual_pixels("How many people with red shirt are on Page 14?") is True
    assert requires_visual_pixels("What are the top2 texts of the yellow words?") is True
    assert requires_visual_pixels("How many datasets are reported in Pages 49-50?") is False


def test_model_phrase_is_not_mistaken_for_visual_intent() -> None:
    assert has_visual_intent("kiến trúc tổng thể của mô hình ASPIRE") is False
    assert has_visual_intent("mô hình này hoạt động ra sao?") is False
    assert has_visual_intent("cho xem hình kiến trúc của mô hình") is True
    assert has_visual_intent("mô hình có sơ đồ không?") is True
    assert has_visual_intent("The television provision is unchanged") is False


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
    assert has_result_table_intent("Compare architecture, dataset and results") is True
    assert has_result_table_intent("How do I access this API?") is False


def test_result_table_intent_catches_natural_evaluation_wording() -> None:
    """Table preference must not depend on the literal word ``table``.

    Users frequently ask about metrics/performance/results in prose.  Those
    requests should reserve canonical table evidence in the same way as an
    explicit table ask, while unrelated words such as ``access`` remain
    outside the detector.
    """

    assert has_result_table_intent("Which metrics were used for performance evaluation?") is True
    assert has_result_table_intent("How did the scores compare with prior methods?") is True
    assert has_result_table_intent("What is the result of the proposed method?") is True
    assert has_result_table_intent("What is the R&D to asset ratio for FY 2019?") is True
    assert has_result_table_intent("What percentage was denied a promotion?") is True
    assert has_result_table_intent("The resulting architecture is described next.") is False
    assert has_result_table_intent("How should this valuation be interpreted?") is False


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
