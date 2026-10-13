import asyncio

import pytest

from app.api.agent import (
    _compose_accumulated_retrieval,
    _curate_figure_sources,
    _direct_canonical_table_answer,
    _filter_figure_sources,
    _merge_focused_table_inventory,
    _resolve_query_document_focus,
    _retrieval_cache_key,
    _retrieval_has_table_sources,
    _select_retrieval_engine,
    _should_merge_visual_sources,
    _scope_documents_to_focus,
    _has_canonical_visual_evidence,
    _visual_source_count,
    visual_page_reservation_count,
    _visual_evidence_sufficiency_context,
    _visual_answer_is_abstention,
    _visual_sufficiency_fallback,
    _VISUAL_PAGE_PROVIDERS,
    shutdown_visual_page_providers,
    _wants_table_inventory,
    _wants_result_tables,
)


def test_visual_page_reservation_preserves_all_explicit_hints() -> None:
    pages = [
        {"document_id": "doc", "page_number": 12, "visual_page": True},
        {"document_id": "doc", "page_number": 14, "visual_page": True},
        {"document_id": "doc", "page_number": 20, "visual_page": True},
    ]
    assert visual_page_reservation_count("compare slides 12 and 14", pages) == 2
    assert visual_page_reservation_count("what is shown?", pages) == 1
    assert visual_page_reservation_count("slide 99", pages) == 1
    assert visual_page_reservation_count("which sources are used?", pages) == 3
    assert visual_page_reservation_count("how important is the visual grounding?", pages) == 2
from app.services.query_rewrite_service import (
    QueryRewriteResult,
    QueryRewriteService,
    _explicit_document_target_entities,
    _looks_like_topic_switch,
    _query_named_entities,
    format_recent_conversation,
)


class _FakeRag:
    def resolve_explicit_document_ids_for_query(
        self, *, query, entities=None, collection_id=None, compare=False
    ):
        resolved = self.resolve_document_ids_for_entities(
            entities=entities or [], collection_id=collection_id, query=query
        )
        return resolved[:2] if compare else resolved[:1]

    def resolve_document_ids_for_entities(self, *, entities, collection_id=None, query=None):
        mapping = {
            "aspire": ["doc-aspire"],
            "msf-ser": ["doc-msf-ser"],
            "wav2small": ["doc-wav2small"],
            "crab": ["doc-crab"],
            "kst": ["doc-kst"],
            "mamba_fusion": ["doc-mamba"],
        }
        tokens = []
        for entity in entities or []:
            key = str(entity).lower().replace(" ", "_")
            if key in mapping:
                tokens.extend(mapping[key])
        if query and "crab" in query.lower():
            return ["doc-crab"]
        if query and "wav2small" in query.lower():
            return ["doc-wav2small"]
        if query and "msf-ser" in query.lower():
            return ["doc-msf-ser"]
        if query and "aspire" in query.lower():
            return ["doc-aspire"]
        return tokens[:1] if tokens else []


def test_scoped_comparison_keeps_paper_overview_not_only_backbone_figures(monkeypatch, tmp_path):
    from app.api import agent as agent_api
    from app.core.config import Settings

    class Rag:
        def list_document_figures(self, document_id):
            return []

        async def search_hybrid(self, **kwargs):
            figures = [{
                "document_id": "doc", "filename": "orion.pdf", "chunk_id": f"fig-{i}",
                "figure_id": f"fig-{i}", "caption": f"Figure {i}: Architecture of a baseline encoder",
                "content": "Encoder architecture " * 140,
            } for i in range(8)]
            overview = {
                "document_id": "doc", "filename": "orion.pdf", "chunk_id": "overview",
                "section_title": "Abstract",
                "content": "This paper evaluates four multimodal fusion approaches combining audio and text.",
            }
            return {"results": figures[:6], "candidate_results": [*figures, overview]}

        def expand_with_neighbor_chunks(self, results, **kwargs):
            return results

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda _: object())
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda _: object())
    result = asyncio.run(agent_api._retrieve_legacy_for_agent(
        rag=Rag(), settings=Settings(data_dir=tmp_path),
        query="ORION architecture pipeline encoder framework",
        coverage_query="How do the two papers differ?",
        collection_id=None, retrieval_mode="hybrid", focus_document_ids=["doc"],
        answer_intent="compare", include_visual_boost=True,
    ))
    assert any(item["chunk_id"] == "overview" for item in result["documents"])
    coverage = result["context_stats"]["required_facet_coverage"][0]
    assert "task" in coverage["covered_facets"]
    assert "benchmark_results" in coverage["missing_facets"]


def test_implicit_fact_query_does_not_invent_overview_reservations():
    from app.api.agent import _context_facet_requirements

    assert _context_facet_requirements("Which languages are explored?", ["doc"], "direct_answer") == {}
    assert _context_facet_requirements("What is the dataset size?", ["doc"], "direct_answer") == {
        "doc": ["dataset_setup"],
    }
    assert len(_context_facet_requirements("How do they differ?", ["doc"], "compare")["doc"]) == 5


def test_resolve_query_document_focus_prefers_entity_over_cached_focus() -> None:
    rewrite = QueryRewriteResult(
        original_query="thế CRAB model đi",
        standalone_query="thế CRAB model đi",
        is_followup=True,
        current_topic="CRAB",
        required_entities=["CRAB"],
        use_last_sources=True,
        answer_intent="direct_answer",
        answer_depth="normal",
        rewrite_used=False,
        diagnostics={},
    )
    focus = _resolve_query_document_focus(
        _FakeRag(),
        rewrite=rewrite,
        collection_id=None,
        existing_focus=["doc-wav2small"],
    )
    assert focus == ["doc-crab"]


def test_filter_figure_sources_uses_text_document_when_focus_missing() -> None:
    documents = [
        {"document_id": "doc-crab", "content": "CRAB architecture", "chunk_id": "c1"},
        {
            "document_id": "doc-crab",
            "figure_id": "fig-crab",
            "caption": "Fig. 1: Crab model architecture",
            "chunk_id": "f1",
        },
        {
            "document_id": "doc-router",
            "figure_id": "fig-router",
            "caption": "Figure extracted from page 1",
            "chunk_id": "f2",
        },
    ]
    filtered = _filter_figure_sources(
        documents,
        allowed_document_ids=None,
        answer_intent="direct_answer",
    )
    figure_ids = [doc["figure_id"] for doc in filtered if doc.get("figure_id")]
    assert figure_ids == ["fig-crab"]


def test_filter_figure_sources_keeps_compare_documents() -> None:
    documents = [
        {"document_id": "doc-a", "content": "A", "chunk_id": "a1"},
        {"document_id": "doc-b", "content": "B", "chunk_id": "b1"},
        {"document_id": "doc-a", "figure_id": "fa", "chunk_id": "fa1"},
        {"document_id": "doc-b", "figure_id": "fb", "chunk_id": "fb1"},
    ]
    filtered = _filter_figure_sources(
        documents,
        allowed_document_ids=["doc-a", "doc-b"],
        answer_intent="compare",
    )
    figure_ids = {doc["figure_id"] for doc in filtered if doc.get("figure_id")}
    assert figure_ids == {"fa", "fb"}


@pytest.mark.parametrize(
    "query",
    [
        "cho tôi xem hình 1 của bài ViSEC",
        "what does the chart in the paper show?",
        "đọc sơ đồ kiến trúc trong hình này",
    ],
)
def test_visual_evidence_sufficiency_requires_canonical_visual_artifact(query: str) -> None:
    visual_context = _visual_evidence_sufficiency_context(
        query,
        [
            {
                "document_id": "doc-a",
                "figure_id": "figure-a",
                "artifact_type": "figure",
                "image_path": "/approved/figure-a.png",
            }
        ],
    )
    assert "at least one canonical figure/page/image artifact" in visual_context
    assert "Do not infer visual content" not in visual_context


def test_visual_evidence_sufficiency_preserves_explicit_page_constraint() -> None:
    context = _visual_evidence_sufficiency_context(
        "what is shown in the picture on slide 6?",
        [{"document_id": "doc-a", "artifact_type": "page", "visual_page": True}],
    )
    assert "Exact page/slide constraint" in context
    assert "page(s) 6" in context
    assert "do not transfer visual or numeric claims from other pages" in context


def test_visual_caption_metadata_does_not_count_as_pixels_for_appearance_query() -> None:
    context = _visual_evidence_sufficiency_context(
        "what does the chart show?",
        [{"document_id": "doc-a", "figure_id": "figure-a", "artifact_type": "figure"}],
    )
    assert "no canonical figure/page/image artifact" in context


def test_visual_caption_query_can_use_figure_metadata_without_pixels() -> None:
    context = _visual_evidence_sufficiency_context(
        "what is the caption of Figure 1?",
        [{"document_id": "doc-a", "figure_id": "figure-a", "artifact_type": "figure"}],
    )
    assert "at least one canonical figure/page/image artifact" in context


def test_visual_evidence_sufficiency_rejects_text_or_table_as_visual_proof() -> None:
    context = _visual_evidence_sufficiency_context(
        "bảng này có biểu đồ gì?",
        [
            {"document_id": "doc-a", "artifact_type": "table", "content": "x y"},
            {"document_id": "doc-a", "chunk_type": "text", "content": "caption words"},
        ],
    )
    assert "no canonical figure/page/image artifact" in context
    assert "OCR or table text alone is not proof" in context


def test_visual_evidence_sufficiency_is_empty_for_nonvisual_question() -> None:
    assert _visual_evidence_sufficiency_context(
        "bảng 2 so sánh accuracy là bao nhiêu?",
        [{"document_id": "doc-a", "artifact_type": "table"}],
    ) == ""


def test_visual_source_count_includes_page_evidence_for_observability() -> None:
    assert _visual_source_count(
        [
            {"artifact_type": "page", "visual_page": True},
            {"artifact_type": "table"},
            {"figure_id": "figure-a"},
        ]
    ) == 2


def test_visual_guard_does_not_treat_plain_page_text_as_visual_proof() -> None:
    assert not _has_canonical_visual_evidence(
        [{"artifact_type": "page", "content": "OCR of the page"}]
    )
    assert _has_canonical_visual_evidence(
        [{"artifact_type": "page", "visual_page": True, "image_path": "/tmp/p.png"}]
    )


def test_visual_guard_accepts_explicit_abstention_and_produces_localized_fallback() -> None:
    assert _visual_answer_is_abstention(
        "No visual evidence was retrieved, so I cannot determine the chart."
    )
    fallback = _visual_sufficiency_fallback("hình này mô tả gì?")
    assert "canonical figure/page/image" in fallback
    assert "không thể xác định" in fallback


def test_visual_provider_cache_is_drained_on_shutdown() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    first = FakeProvider()
    second = FakeProvider()
    _VISUAL_PAGE_PROVIDERS.clear()
    _VISUAL_PAGE_PROVIDERS[("one", "", "", "", "", "", "", False, 1.0)] = first
    _VISUAL_PAGE_PROVIDERS[("two", "", "", "", "", "", "", False, 1.0)] = second

    asyncio.run(shutdown_visual_page_providers())

    assert first.closed and second.closed
    assert _VISUAL_PAGE_PROVIDERS == {}


def test_topic_switch_detected_for_crab_followup() -> None:
    assert _looks_like_topic_switch("thế CRAB model đi") is True


def test_explicit_paper_target_wins_even_when_name_appears_in_active_paper_table() -> None:
    import asyncio

    class _NoLLM:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise AssertionError("explicit paper target must be deterministic")

    messages = [
        {"role": "user", "content": "đưa bảng 2 bài ASPIRE đây"},
        {
            "role": "assistant",
            "content": "Table 2 ASPIRE contains an MSF-SER baseline row.",
        },
    ]
    query = "Tôi muốn xem bảng result của bài MSF-SER, bạn làm được không?"

    result = asyncio.run(
        QueryRewriteService(client=_NoLLM(), default_model="unused").rewrite(
            query=query,
            previous_messages=messages,
            working_topic="ASPIRE",
            working_document_hint="ASPIRE.pdf",
        )
    )

    assert _explicit_document_target_entities(query) == ["MSF-SER"]
    assert _looks_like_topic_switch(query, previous_messages=messages) is True
    assert result.current_topic == "MSF-SER"
    assert result.required_entities == ["MSF-SER"]
    assert result.use_last_sources is False
    assert result.diagnostics["reason"] == "explicit_document_target"
    assert (
        _resolve_query_document_focus(
            _FakeRag(),
            rewrite=result,
            collection_id=None,
            existing_focus=["doc-aspire"],
        )
        == ["doc-msf-ser"]
    )


@pytest.mark.parametrize(
    "query",
    [
        "đối chiếu bài ASPIRE với bài KST",
        "phân biệt paper ASPIRE với paper KST",
        "differences between paper ASPIRE and paper KST",
        "paper ASPIRE against paper KST",
    ],
)
def test_explicit_multi_document_targets_share_comparison_language_policy(
    query: str,
) -> None:
    import asyncio

    class _NoLLM:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise AssertionError("explicit comparison must be deterministic")

    result = asyncio.run(
        QueryRewriteService(client=_NoLLM(), default_model="unused").rewrite(
            query=query,
            previous_messages=[],
        )
    )

    assert _explicit_document_target_entities(query) == ["ASPIRE", "KST"]
    assert result.required_entities == ["ASPIRE", "KST"]
    assert result.answer_intent == "compare"
    assert result.diagnostics["reason"] == "explicit_document_target"


def test_table_number_followed_by_paper_alias_overrides_sticky_focus() -> None:
    import asyncio

    class _NoLLM:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise AssertionError("explicit table target must be deterministic")

    result = asyncio.run(
        QueryRewriteService(client=_NoLLM(), default_model="unused").rewrite(
            query="đưa bảng 2 aspire đây",
            previous_messages=[
                {"role": "assistant", "content": "Đang nói về bảng ViSEC."},
            ],
            working_topic="ViSEC",
            working_document_hint="ICASSP_2024___ViSEC.pdf",
        )
    )

    assert result.current_topic == "aspire"
    assert result.use_last_sources is False
    assert result.diagnostics["reason"] == "explicit_document_target"
    assert (
        _resolve_query_document_focus(
            _FakeRag(),
            rewrite=result,
            collection_id=None,
            existing_focus=["doc-visec"],
        )
        == ["doc-aspire"]
    )


def test_correction_with_old_and_new_paper_names_uses_last_explicit_target() -> None:
    query = "đây là bảng của bài ASPIRE rồi, bài MSF-SER bảng của nó cơ"

    assert _query_named_entities(query) == ["ASPIRE", "MSF-SER"]
    assert _explicit_document_target_entities(query) == ["MSF-SER"]


def test_table_inventory_intent_detects_vietnamese_count_question() -> None:
    assert _wants_table_inventory("bài ViSEC có mấy bảng ở experiment?") is True
    assert _wants_table_inventory("Bài ASPIRE có những bảng nào?") is True
    assert _wants_table_inventory("bài này có bảng gì?") is True
    assert _wants_table_inventory("What tables are in ASPIRE?") is True
    assert _wants_table_inventory("show all tables in ASPIRE") is True
    assert _wants_table_inventory("đưa dữ liệu Table 2") is False


def test_table_inventory_prepends_every_scoped_canonical_table() -> None:
    class _InventoryRag:
        def get_document(self, document_id):
            return {
                "id": document_id,
                "filename": "ViSEC.pdf",
                "source_path": "/pdf/ViSEC.pdf",
            }

        def list_document_tables(self, document_id):
            return [
                {
                    "id": "table-1",
                    "document_id": document_id,
                    "table_index": 0,
                    "page_number": 4,
                    "caption": "Table 1. Dataset distribution",
                    "markdown": "| Label | Count |\n|---|---:|\n| Happy | 1 |",
                    "metadata": {"table_type": "dataset"},
                },
                {
                    "id": "table-2",
                    "document_id": document_id,
                    "table_index": 1,
                    "page_number": 4,
                    "caption": "Table 2. Results",
                    "markdown": "| Model | UA |\n|---|---:|\n| Pitch-fusion | 72.72 |",
                    "metadata": {"table_type": "comparison"},
                },
            ]

    existing_table_2 = {
        "chunk_id": "table:table-2",
        "document_id": "visec-id",
        "table_id": "table-2",
        "artifact_type": "table",
        "content": "existing ranked Table 2",
    }
    merged, count = _merge_focused_table_inventory(
        rag=_InventoryRag(),
        sources=[
            existing_table_2,
            {"chunk_id": "text-1", "document_id": "visec-id"},
        ],
        focus_document_ids=["visec-id"],
    )

    assert count == 2
    assert [item.get("table_id") for item in merged[:2]] == ["table-1", "table-2"]
    assert merged[1] is existing_table_2


@pytest.mark.parametrize("query", [
    "Which language pairs are explored?",
    "How many layers does the encoder have?",
    "Những thành phần nào tạo nên kiến trúc?",
    "What are the results from these proposed strategies?",
])
def test_implicit_structured_ask_does_not_force_inventory_or_discard_figures(query) -> None:
    class InventoryRag:
        def list_document_tables(self, document_id):
            raise AssertionError("An implicit structured ask is not a table inventory")

    sources = [{
        "document_id": "report",
        "figure_id": "flowchart",
        "artifact_type": "figure",
        "caption": "Figure 4. Encoder stages",
        "content": "The encoder has three layers.",
    }]
    merged, count = _merge_focused_table_inventory(
        rag=InventoryRag(), sources=sources, focus_document_ids=["report"], query=query,
    )
    assert merged == sources
    assert count == 0


def test_table_inventory_keeps_companion_figure_and_revision_identity() -> None:
    class InventoryRag:
        def get_document(self, document_id):
            return {"filename": "report.pdf", "active_revision_id": "active"}

        def list_document_tables(self, document_id):
            return [{"id": "table", "revision_id": "active", "caption": "Table 1. Results",
                     "markdown": "| Model | Score |\n|---|---|\n| Encoder | 0.8 |"}]

    figure = {"document_id": "report", "figure_id": "figure", "artifact_type": "figure",
              "caption": "Figure 2. Result curves", "content": "Companion plot"}
    merged, count = _merge_focused_table_inventory(
        rag=InventoryRag(), sources=[figure], focus_document_ids=["report"],
        query="show all tables in this report",
    )
    assert count == 1
    assert merged[0]["revision_id"] == "active"
    assert figure in merged


@pytest.mark.parametrize("query", [
    "Interpret the benchmark performance of these two systems.",
    "So sánh kết quả hai bài đó.",
    "Đánh giá bảng kết quả CCC của phương pháp này.",
])
def test_analytical_result_gathering_is_independent_of_direct_table_rendering(query):
    class InventoryRag:
        def get_document(self, document_id):
            return {"filename": f"{document_id}.pdf", "active_revision_id": "active"}

        def list_document_tables(self, document_id):
            return [
                {"id": f"{document_id}-abl", "revision_id": "active", "table_index": 0,
                 "caption": "Table 1. Ablation performance on Corpus-X",
                 "markdown": "| Variant | CCC |\n|---|---|\n| No gate | 0.51 |"},
                {"id": f"{document_id}-main", "revision_id": "active", "table_index": 7,
                 "caption": "Table 8. Comparison with prior methods on Corpus-X",
                 "markdown": "| Method | CCC |\n|---|---|\n| ORION | 0.74 |"},
                {"id": f"{document_id}-stats", "revision_id": "active", "table_index": 8,
                 "caption": "Table 9. Corpus statistics for benchmark evaluation",
                 "markdown": "| Split | Count |\n|---|---|\n| Train | 200 |"},
            ]

    companion = {"document_id": "alpha", "figure_id": "curves", "artifact_type": "figure"}
    merged, count = _merge_focused_table_inventory(
        rag=InventoryRag(), sources=[companion], focus_document_ids=["alpha", "beta"], query=query,
    )
    assert count == 2
    assert {s.get("table_id") for s in merged if s.get("table_id")} == {"alpha-main", "beta-main"}
    assert all(s["revision_id"] == "active" for s in merged if s.get("table_id"))
    assert companion in merged
    # Analysis is still generated by the model; gathering a table is not
    # permission to bypass interpretation by directly rendering it.
    assert _direct_canonical_table_answer(query, merged) is None


def test_explicit_ablation_request_does_not_expand_main_result_inventory():
    class InventoryRag:
        def list_document_tables(self, document_id):
            raise AssertionError("Ablation analysis must retain its specific candidate frontier")

    source = {"table_id": "ablation", "document_id": "alpha", "artifact_type": "table"}
    merged, count = _merge_focused_table_inventory(
        rag=InventoryRag(), sources=[source], focus_document_ids=["alpha"],
        query="Interpret the ablation performance and CCC results.",
    )
    assert merged == [source]
    assert count == 0


def test_exact_table_inventory_injection_does_not_depend_on_vector_top_k() -> None:
    class _InventoryRag:
        def get_document(self, document_id):
            return {"filename": "ViSEC.pdf", "source_path": "/pdf/ViSEC.pdf"}

        def list_document_tables(self, document_id):
            return [
                {
                    "id": "table-1",
                    "table_index": 0,
                    "page_number": 3,
                    "caption": "Table 1. Emotion distributions",
                    "markdown": "| Label | Count |\n|---|---:|\n| Happy | 1 |",
                    "metadata": {"table_type": "dataset"},
                },
                {
                    "id": "table-2",
                    "table_index": 1,
                    "page_number": 4,
                    "caption": "Table 2. Experimental results on tonal languages",
                    "markdown": "| Model | UA |\n|---|---:|\n| Pitch-fusion | 72.72 |",
                    "metadata": {"table_type": "comparison"},
                },
            ]

    merged, count = _merge_focused_table_inventory(
        rag=_InventoryRag(),
        sources=[
            {
                "chunk_id": "table:table-1",
                "document_id": "visec-id",
                "table_id": "table-1",
                "artifact_type": "table",
                "caption": "Table 1. Emotion distributions",
                "content": "lexically stronger but explicitly unrequested table",
            },
            {"chunk_id": "text-only", "document_id": "visec-id"},
        ],
        focus_document_ids=["visec-id"],
        query="đưa bảng 2 bài pitch fusion",
    )

    assert count == 1
    assert merged[0]["table_id"] == "table-2"
    assert "Pitch-fusion" in merged[0]["content"]
    assert all(item.get("table_id") != "table-1" for item in merged)


def test_printed_table_number_wins_over_conflicting_positional_index() -> None:
    class _ShiftedInventoryRag:
        def get_document(self, document_id):
            return {"filename": "Target.pdf", "source_path": "/pdf/Target.pdf"}

        def list_document_tables(self, document_id):
            return [
                {
                    "id": "caption-table-3",
                    "table_index": 1,
                    "page_number": 3,
                    "caption": "Table 3. Architecture",
                    "markdown": "| Layer | Width |\n|---|---:|\n| Encoder | 128 |",
                    "metadata": {},
                }
            ]

    wrong, wrong_count = _merge_focused_table_inventory(
        rag=_ShiftedInventoryRag(),
        sources=[],
        focus_document_ids=["target"],
        query="đưa bảng 2",
    )
    correct, correct_count = _merge_focused_table_inventory(
        rag=_ShiftedInventoryRag(),
        sources=[],
        focus_document_ids=["target"],
        query="đưa bảng 3",
    )

    assert wrong_count == 0
    assert wrong == []
    assert correct_count == 1
    assert correct[0]["table_id"] == "caption-table-3"


def test_multiple_explicit_table_numbers_are_preserved_in_request_order() -> None:
    class _MultiInventoryRag:
        def get_document(self, document_id):
            return {"filename": "Target.pdf", "source_path": "/pdf/Target.pdf"}

        def list_document_tables(self, document_id):
            return [
                {
                    "id": f"table-{number}",
                    "table_index": number - 1,
                    "page_number": number,
                    "caption": f"Table {number}. Results {number}",
                    "markdown": (
                        f"| Model | Score |\n|---|---:|\n| Model-{number} | 0.{number} |"
                    ),
                    "metadata": {"table_type": "comparison"},
                }
                for number in (1, 2, 3)
            ]

    merged, count = _merge_focused_table_inventory(
        rag=_MultiInventoryRag(),
        sources=[],
        focus_document_ids=["target"],
        query="đưa bảng 2 và bảng 1",
    )

    assert count == 2
    assert [item["table_id"] for item in merged] == ["table-1", "table-2"]
    direct = _direct_canonical_table_answer(
        "đưa bảng 2 và bảng 1",
        merged,
        expected_document_ids=["target"],
    )
    assert direct is not None
    answer, selected = direct
    assert answer.index("Table 2") < answer.index("Table 1")
    assert selected["table_ids"] == ["table-2", "table-1"]

    for query in (
        "đưa bảng 2 + bảng 1",
        "đưa bảng 2 với bảng 1",
        "show Table 2 with Table 1",
    ):
        variant = _direct_canonical_table_answer(
            query,
            merged,
            expected_document_ids=["target"],
        )
        assert variant is not None, query
        assert variant[1]["table_ids"] == ["table-2", "table-1"], query


def test_discourse_marker_keeps_model_already_discussed_in_active_paper() -> None:
    query = (
        "ờ đấy thế sao không đưa dữ liệu bảng 2 so sánh "
        "pitch fusion với các model khác?"
    )
    messages = [
        {"role": "user", "content": "bài ViSEC có mấy bảng ở experiment?"},
        {
            "role": "assistant",
            "content": "Table 2 so sánh Pitch-fusion với Wav2Vec 2.0.",
        },
    ]

    assert _query_named_entities(query) == ["pitchfusion"]
    assert _looks_like_topic_switch(query, previous_messages=messages) is False


def test_result_followup_for_known_model_keeps_sticky_paper_without_llm() -> None:
    import asyncio

    class _NoLLM:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise AssertionError("known same-paper result follow-up must be deterministic")

    query = "thế đưa bảng kết quả so sánh pitch fusion với model khác đi"
    messages = [
        {"role": "user", "content": "bài ViSEC có mấy bảng ở experiment?"},
        {
            "role": "assistant",
            "content": "Table 2 so sánh Pitch-fusion với Wav2Vec 2.0.",
        },
    ]

    result = asyncio.run(
        QueryRewriteService(client=_NoLLM(), default_model="unused").rewrite(
            query=query,
            previous_messages=messages,
            working_topic="ViSEC",
            working_document_hint="ICASSP_2024___ViSEC.pdf",
        )
    )

    assert result.current_topic == "ViSEC"
    assert result.required_entities == ["ViSEC"]
    assert result.use_last_sources is True
    assert result.diagnostics["reason"] == "heuristic_deepen_followup"


def test_curate_figure_sources_picks_one_per_document_for_compare() -> None:
    documents = [
        {"document_id": "doc-kst", "content": "KST overview", "chunk_id": "k1"},
        {
            "document_id": "doc-kst",
            "figure_id": "kst-fig-4",
            "figure_index": 3,
            "caption": "Fig. 4. CCAB details",
            "score": 0.9,
            "chunk_id": "kf4",
        },
        {
            "document_id": "doc-kst",
            "figure_id": "kst-fig-2",
            "figure_index": 1,
            "caption": "Fig. 2. Overview structure of the proposed model.",
            "score": 0.7,
            "chunk_id": "kf2",
        },
        {
            "document_id": "doc-mamba",
            "figure_id": "mamba-fig-2",
            "figure_index": 1,
            "caption": "Figure 2: Architecture of the dual-branch Mamba-based fusion model.",
            "score": 0.8,
            "chunk_id": "mf2",
        },
        {
            "document_id": "doc-mamba",
            "figure_id": "mamba-fallback",
            "figure_index": 3,
            "caption": "Page 4 visual fallback",
            "score": 0.95,
            "chunk_id": "mf4",
        },
    ]
    curated = _curate_figure_sources(
        documents,
        answer_intent="compare",
        focus_document_ids=["doc-kst", "doc-mamba"],
        query="so sánh KST với Mamba fig 2",
    )
    figure_ids = [doc["figure_id"] for doc in curated if doc.get("figure_id")]
    assert figure_ids == ["kst-fig-2", "mamba-fig-2"]


def test_curate_figure_sources_scopes_to_focus_document() -> None:
    documents = [
        {"document_id": "doc-aspire", "content": "ASPIRE model", "chunk_id": "a1"},
        {
            "document_id": "doc-whisper",
            "figure_id": "whisper-fig-2",
            "figure_index": 1,
            "caption": "Figure 2: (a) Distribution of primary emotions",
            "score": 0.99,
            "chunk_id": "wf2",
        },
        {
            "document_id": "doc-aspire",
            "figure_id": "aspire-fig-1",
            "figure_index": 0,
            "caption": "Figure 1: Architecture of ASPIRE",
            "score": 0.4,
            "chunk_id": "af1",
        },
        {
            "document_id": "doc-aspire",
            "figure_id": "aspire-fig-2",
            "figure_index": 1,
            "caption": "Figure 2: Per-class F1 comparison between ASPIRE and the",
            "score": 0.95,
            "chunk_id": "af2",
        },
    ]
    curated = _curate_figure_sources(
        documents,
        answer_intent="direct_answer",
        focus_document_ids=["doc-aspire"],
        query="architecture của ASPIRE",
    )
    figure_ids = [doc["figure_id"] for doc in curated if doc.get("figure_id")]
    assert figure_ids[0] == "aspire-fig-1"
    assert "whisper-fig-2" not in figure_ids


def test_curate_figure_sources_returns_only_top_ranked_visual_when_best_requested() -> None:
    documents = [
        {"document_id": "doc-a", "content": "Paper overview", "chunk_id": "text-1"},
        {
            "document_id": "doc-a",
            "figure_id": "architecture",
            "figure_type": "architecture",
            "caption": "Figure 1: Overall model architecture and processing pipeline",
            "score": 0.4,
            "chunk_id": "figure-1",
        },
        {
            "document_id": "doc-a",
            "figure_id": "result-plot",
            "figure_type": "plot",
            "caption": "Figure 2: Per-class result comparison",
            "score": 0.95,
            "chunk_id": "figure-2",
        },
        {
            "document_id": "doc-a",
            "figure_id": "publisher-logo",
            "asset_kind": "logo",
            "caption": "Publisher logo",
            "score": 1.0,
            "chunk_id": "figure-logo",
        },
    ]

    curated = _curate_figure_sources(
        documents,
        answer_intent="direct_answer",
        focus_document_ids=["doc-a"],
        query=(
            "Cho mình figure hoặc sơ đồ kiến trúc phù hợp nhất, "
            "trả kèm hình và đừng lấy logo."
        ),
    )

    assert [doc["chunk_id"] for doc in curated if not doc.get("figure_id")] == ["text-1"]
    assert [doc["figure_id"] for doc in curated if doc.get("figure_id")] == ["architecture"]


def test_topic_switch_detects_new_paper_without_switch_words() -> None:
    messages = [
        {"role": "user", "content": "WhiSER là gì?"},
        {"role": "assistant", "content": "WHiSER là corpus SER từ White House tapes."},
    ]
    assert _looks_like_topic_switch(
        "tôi cần hiểu rõ hơn về architecture của ASPIRE",
        previous_messages=messages,
    ) is True


def test_deepen_does_not_bind_new_paper_to_history_topic() -> None:
    from app.services.query_rewrite_service import _generic_deepen_rewrite

    messages = [
        {"role": "user", "content": "WhiSER là gì?"},
        {"role": "assistant", "content": "WHiSER là corpus SER."},
    ]
    assert (
        _generic_deepen_rewrite(
            "tôi cần hiểu rõ hơn về architecture của ASPIRE",
            messages,
        )
        is None
    )


def test_deepen_pitch_topic_uses_generic_grounded_anchors_only() -> None:
    from app.services.query_rewrite_service import _generic_deepen_rewrite

    result = _generic_deepen_rewrite(
        "giải thích kỹ hơn architecture đi",
        [],
        working_topic="Pitch-fusion",
    )

    assert result is not None
    _topic, standalone, required = result
    assert "cấu trúc/thành phần" in standalone
    assert required == ["Pitch-fusion"]
    for unsupported_component in ("Kaldi", "Cross-Attention", "Pitch Encoder"):
        assert unsupported_component not in standalone


def test_auto_retrieval_engine_keeps_focused_qa_on_fast_hybrid() -> None:
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="direct_answer",
            focus_document_ids=["doc-a"],
            prefer_legacy_tables=False,
        )
        == "legacy"
    )


def test_decomposed_compare_branch_uses_scoped_fast_hybrid() -> None:
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="compare",
            focus_document_ids=["one-canonical-paper"],
            prefer_legacy_tables=False,
        )
        == "legacy"
    )


def test_auto_retrieval_engine_uses_graph_for_discovery_and_cross_document() -> None:
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="direct_answer",
            focus_document_ids=[],
            prefer_legacy_tables=False,
        )
        == "lightrag"
    )
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="compare",
            focus_document_ids=["doc-a", "doc-b"],
            prefer_legacy_tables=False,
        )
        == "lightrag"
    )


def test_explicit_fts_and_table_requests_never_route_to_graph() -> None:
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="fts",
            answer_intent="compare",
            focus_document_ids=["doc-a", "doc-b"],
            prefer_legacy_tables=False,
        )
        == "legacy"
    )
    assert (
        _select_retrieval_engine(
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="direct_answer",
            focus_document_ids=["doc-a"],
            prefer_legacy_tables=True,
        )
        == "legacy"
    )


def test_exact_table_request_can_render_one_provenanced_canonical_table() -> None:
    source = {
        "document_id": "visec",
        "table_id": "table-2",
        "table_index": 1,
        "filename": "ViSEC.pdf",
        "page_number": 4,
        "caption": "Table 2. Experimental results",
        "content": (
            "caption: Table 2. Experimental results\ncontent:\n"
            "| Model | UA (%) | WA (%) |\n"
            "|---|---:|---:|\n"
            "| Pitch-fusion | 72.72 | 71.90 |"
        ),
    }

    direct = _direct_canonical_table_answer("Đưa mình bảng 2 của paper", [source])

    assert direct is not None
    answer, selected = direct
    assert "| Pitch-fusion | 72.72 | 71.90 |" in answer
    assert "không tự tính thêm chênh lệch" in answer
    assert selected["table_id"] == "table-2"


def test_table_interpretation_declines_direct_renderer() -> None:
    source = {
        "document_id": "visec",
        "table_id": "table-2",
        "table_index": 1,
        "caption": "Table 2. Experimental results",
        "content": "| Model | UA |\n|---|---:|\n| Pitch-fusion | 72.72 |",
    }

    assert (
        _direct_canonical_table_answer("Giải thích và đánh giá bảng 2", [source])
        is None
    )


def test_vague_result_table_request_declines_direct_renderer() -> None:
    source = {
        "document_id": "doc-msf-ser",
        "table_id": "table-2",
        "table_index": 1,
        "caption": "Table 2. Ablation results",
        "content": "| Variant | CCC |\n|---|---:|\n| Full | 0.692 |",
    }

    assert (
        _direct_canonical_table_answer(
            "Tôi muốn xem bảng result của bài MSF-SER",
            [source],
            expected_document_ids=["doc-msf-ser"],
        )
        is None
    )


def test_generic_result_request_renders_one_canonical_main_comparison_table() -> None:
    source = {
        "document_id": "doc-msf-ser",
        "table_id": "table-3",
        "table_index": 2,
        "filename": "MSF-SER.pdf",
        "caption": "Table 3. Comparison of different models on IEMOCAP",
        "content": (
            "| Model | CCC V | CCC avg |\n"
            "|---|---:|---:|\n"
            "| MSF-SER | 0.632 | 0.638 |"
        ),
    }

    direct = _direct_canonical_table_answer(
        "Tôi muốn xem bảng result của bài MSF-SER",
        [source],
        expected_document_ids=["doc-msf-ser"],
    )

    assert direct is not None
    answer, selected = direct
    assert "| MSF-SER | 0.632 | 0.638 |" in answer
    assert selected["table_id"] == "table-3"


def test_dataset_plus_results_prefers_canonical_performance_coverage_table() -> None:
    performance = {
        "document_id": "doc-aspire",
        "table_id": "table-1",
        "table_index": 0,
        "filename": "ASPIRE.pdf",
        "caption": "Table 1: Performance on IEMOCAP and MSP-Podcast v2.0 (4-class)",
        "content": (
            "| Dataset | Model | Acc. | F1 |\n"
            "|---|---|---:|---:|\n"
            "| IEMOCAP | ASPIRE | 75.86 | 76.31 |\n"
            "| MSP-Podcast Test1 | ASPIRE | 72.10 | 67.58 |"
        ),
    }
    prior_methods = {
        "document_id": "doc-aspire",
        "table_id": "table-2",
        "table_index": 1,
        "filename": "ASPIRE.pdf",
        "caption": "Table 2: Comparison with prior methods on IEMOCAP",
        "content": "| Model | Acc. |\n|---|---:|\n| ASPIRE | 75.86 |",
    }

    direct = _direct_canonical_table_answer(
        "Bài ASPIRE dùng dataset nào thế? Cho tôi bảng kết quả đi",
        [prior_methods, performance],
        expected_document_ids=["doc-aspire"],
    )

    assert direct is not None
    answer, selected = direct
    assert selected["table_id"] == "table-1"
    assert "MSP-Podcast" in answer


def test_prior_model_result_ask_prefers_canonical_comparison_table() -> None:
    sources = [
        {
            "document_id": "doc-aspire",
            "table_id": "table-1",
            "table_index": 0,
            "filename": "ASPIRE.pdf",
            "caption": "Table 1: Performance on IEMOCAP and MSP-Podcast v2.0",
            "content": "| Dataset | Acc. |\n|---|---:|\n| IEMOCAP | 75.86 |",
        },
        {
            "document_id": "doc-aspire",
            "table_id": "table-2",
            "table_index": 1,
            "filename": "ASPIRE.pdf",
            "caption": "Table 2: Comparison with prior methods on IEMOCAP",
            "content": "| Model | Acc. |\n|---|---:|\n| ASPIRE | 75.86 |",
        },
    ]

    direct = _direct_canonical_table_answer(
        "cho bảng kết quả so sánh ASPIRE với prior models",
        sources,
        expected_document_ids=["doc-aspire"],
    )

    assert direct is not None
    assert direct[1]["table_id"] == "table-2"


def test_metric_only_benchmark_followup_prefers_main_result_over_ablation() -> None:
    """Metric follow-ups must not let the model choose an ablation table."""

    sources = [
        {
            "document_id": "doc-aspire",
            "table_id": "table-1",
            "table_index": 0,
            "filename": "ASPIRE.pdf",
            "caption": "Table 1: Performance on IEMOCAP and MSP-Podcast v2.0",
            "content": (
                "| Model | IEMOCAP Acc. | IEMOCAP F1 | MSP Acc. | MSP F1 |\n"
                "|---|---:|---:|---:|---:|\n"
                "| ASPIRE | 75.86 | 76.31 | 72.10 | 67.58 |"
            ),
        },
        {
            "document_id": "doc-aspire",
            "table_id": "table-4",
            "table_index": 3,
            "filename": "ASPIRE.pdf",
            "caption": "Table 4: Ablation study for RQ1 on IEMOCAP",
            "content": (
                "| Variant | Classification Acc. | Classification F1 | CCC A | CCC V |\n"
                "|---|---:|---:|---:|---:|\n"
                "| ASPIRE (Full) | 75.86 | 76.31 | 0.7140 | 0.7400 |"
            ),
        },
        {
            "document_id": "doc-aspire",
            "table_id": "table-2",
            "table_index": 1,
            "filename": "ASPIRE.pdf",
            "caption": "Table 2: Comparison with prior methods on IEMOCAP",
            "content": (
                "| Model | Acc. | F1 | CCC A | CCC V |\n"
                "|---|---:|---:|---:|---:|\n"
                "| ASPIRE | 75.86 | 76.31 | 0.7140 | 0.7400 |"
            ),
        },
    ]

    direct = _direct_canonical_table_answer(
        "Quay lại bài lúc nãy: benchmark Acc, F1 và CCC theo evidence nói gì?",
        sources,
        expected_document_ids=["doc-aspire"],
    )

    assert direct is not None
    assert direct[1]["table_id"] == "table-2"


def test_generic_result_table_tie_remains_fail_closed() -> None:
    sources = [
        {
            "document_id": "doc-gban",
            "table_id": f"table-{index}",
            "table_index": index,
            "filename": "GBAN.pdf",
            "caption": caption,
            "content": "| Model | WA |\n|---|---:|\n| Ours | 0.72 |",
        }
        for index, caption in enumerate(
            (
                "Table 1: Comparison across representations",
                "Table 2: Comparison across fusion methods",
            )
        )
    ]

    assert (
        _direct_canonical_table_answer(
            "đưa bảng kết quả bài GBAN",
            sources,
            expected_document_ids=["doc-gban"],
        )
        is None
    )


@pytest.mark.parametrize(
    "caption",
    [
        "Table 2. Comparison with prior methods on IEMOCAP",
        "Table 2. Experimental results on tonal languages",
        "Table 4. Performance on the test datasets",
        "Table 1. Benchmark results for speech emotion recognition",
        "Table 3. Comparative evaluation of competing methods",
    ],
)
def test_generic_result_renderer_accepts_varied_main_result_captions(caption: str) -> None:
    source = {
        "document_id": "target-paper",
        "table_id": "canonical-result",
        "table_index": 1,
        "filename": "Target.pdf",
        "caption": caption,
        "content": "| Model | Score |\n|---|---:|\n| Ours | 0.81 |",
    }

    direct = _direct_canonical_table_answer(
        "đưa bảng kết quả của bài Target",
        [source],
        expected_document_ids=["target-paper"],
    )

    assert direct is not None
    assert direct[1]["table_id"] == "canonical-result"


@pytest.mark.parametrize(
    "caption",
    [
        "Table 2. Ablation results",
        "Table 1. Emotion distributions of the experimented datasets",
        "Table 1. Dataset statistics",
        "Table 5. Hyperparameters",
        "Table 6. Training settings",
    ],
)
def test_generic_result_renderer_rejects_non_main_table_semantics(caption: str) -> None:
    source = {
        "document_id": "target-paper",
        "table_id": "non-main",
        "table_index": 0,
        "filename": "Target.pdf",
        "caption": caption,
        "content": "| Field | Value |\n|---|---:|\n| Example | 1 |",
    }

    assert (
        _direct_canonical_table_answer(
            "đưa bảng kết quả của bài Target",
            [source],
            expected_document_ids=["target-paper"],
        )
        is None
    )


def test_direct_table_renderer_rejects_table_from_wrong_focused_paper() -> None:
    source = {
        "document_id": "doc-aspire",
        "table_id": "table-2",
        "table_index": 1,
        "caption": "Table 2. ASPIRE benchmark",
        "content": "| Model | F1 |\n|---|---:|\n| ASPIRE | 76.31 |",
    }

    assert (
        _direct_canonical_table_answer(
            "đưa bảng 2 bài MSF-SER đây",
            [source],
            expected_document_ids=["doc-msf-ser"],
        )
        is None
    )


def test_adaptive_hop_accumulates_initial_and_new_evidence() -> None:
    merged = _compose_accumulated_retrieval(
        query="compare A and B",
        retrievals=[
            {
                "mode": "first",
                "documents": [
                    {
                        "chunk_id": "a1",
                        "document_id": "doc-a",
                        "filename": "A.pdf",
                        "content": "Initial evidence for A.",
                    }
                ],
            },
            {
                "mode": "second",
                "documents": [
                    {
                        "chunk_id": "b1",
                        "document_id": "doc-b",
                        "filename": "B.pdf",
                        "content": "Recovered evidence for B.",
                    }
                ],
            },
        ],
        answer_intent="compare",
        answer_depth="detailed",
        include_visual_boost=False,
        prefer_tables=False,
    )

    assert merged["mode"] == "adaptive_multihop"
    assert {item["document_id"] for item in merged["documents"]} == {
        "doc-a",
        "doc-b",
    }
    assert merged["diagnostics"]["hop_count"] == 2


def test_comparison_does_not_imply_visual_retrieval_without_visual_intent() -> None:
    assert (
        _should_merge_visual_sources(
            query="Compare the benchmark results of AlphaNet and BetaNet.",
            include_visual_boost=False,
        )
        is False
    )
    assert (
        _should_merge_visual_sources(
            query="Compare Figure 2 of AlphaNet and BetaNet.",
            include_visual_boost=False,
        )
        is True
    )
    assert (
        _should_merge_visual_sources(
            query="Compare AlphaNet and BetaNet.",
            include_visual_boost=True,
        )
        is True
    )


def test_benchmark_followup_keeps_sticky_topic() -> None:
    from app.services.query_rewrite_service import (
        _generic_deepen_rewrite,
        _is_named_entity,
        _query_named_entities,
        looks_like_followup,
    )

    assert looks_like_followup("benchmark với mấy model khác đi, bảng so sánh Acc F1 CCC")
    assert _query_named_entities("benchmark Acc F1 CCC") == []
    assert not _is_named_entity("Acc")
    assert not _is_named_entity("CCC")
    assert not _is_named_entity("benchmark")

    messages = [
        {"role": "user", "content": "Giải thích architecture bài ASPIRE"},
        {"role": "assistant", "content": "ASPIRE dùng audio-text fusion."},
    ]
    result = _generic_deepen_rewrite(
        "benchmark với mấy model khác đi, bảng so sánh Acc F1 CCC",
        messages,
        working_topic="aspire",
    )
    assert result is not None
    topic, standalone, required = result
    assert topic.lower() == "aspire"
    assert "aspire" in standalone.lower()
    assert required == ["aspire"]


def test_natural_resume_does_not_invent_quay_or_evidence_entities() -> None:
    import asyncio

    class _NoLLM:
        async def chat(self, **kwargs):  # noqa: ANN003
            raise AssertionError("resume rewrite must stay deterministic")

    query = "Quay lại paper lúc nãy: benchmark Acc, F1 và CCC theo evidence nói gì?"
    messages = [
        {"role": "user", "content": "Giải thích architecture bài ASPIRE"},
        {"role": "assistant", "content": "ASPIRE dùng audio-visual fusion."},
        {"role": "user", "content": "Cậu thích cà phê không?"},
        {"role": "assistant", "content": "Có."},
    ]

    result = asyncio.run(
        QueryRewriteService(client=_NoLLM(), default_model="unused").rewrite(
            query=query,
            previous_messages=messages,
            working_topic="ASPIRE",
            working_document_hint="ASPIRE.pdf",
        )
    )

    assert _query_named_entities(query) == []
    assert _looks_like_topic_switch(query, previous_messages=messages) is False
    assert result.current_topic == "ASPIRE"
    assert result.required_entities == ["ASPIRE"]
    assert result.use_last_sources is True
    assert result.diagnostics["reason"] == "resume_working_focus"
    assert "ASPIRE" in result.standalone_query


def test_recent_context_budget_always_keeps_newest_message() -> None:
    messages = [
        {"role": "user", "content": "old " * 1000},
        {"role": "assistant", "content": "ASPIRE latest grounded answer"},
    ]
    packed = format_recent_conversation(messages, max_messages=8, max_chars=100)
    assert "ASPIRE latest grounded answer" in packed


def test_result_table_intent_uses_metric_boundaries() -> None:
    assert _wants_result_tables("benchmark Acc F1 CCC") is True
    assert _wants_result_tables("How do I access this API?") is False


def test_table_intent_rejects_text_only_retrieval_cache() -> None:
    text_only = {
        "documents": [
            {
                "chunk_id": "text-1",
                "document_id": "doc-aspire",
                "artifact_type": None,
            }
        ]
    }
    with_table = {
        "documents": [
            {
                "chunk_id": "table:benchmark",
                "document_id": "doc-aspire",
                "artifact_type": "table",
            }
        ]
    }

    assert _retrieval_has_table_sources(text_only) is False
    assert _retrieval_has_table_sources(with_table) is True


def test_focused_legacy_retrieval_reserves_table_context(monkeypatch, tmp_path) -> None:
    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings

    class _RagWithLateTable:
        def get_document(self, document_id):
            return {"filename": "ASPIRE.pdf", "source_path": "/pdf/ASPIRE.pdf"}

        def list_document_tables(self, document_id):
            return []

        async def search_hybrid(self, **kwargs):  # noqa: ANN003
            assert kwargs["document_ids"] == ["doc-aspire"]
            return {
                "results": [
                    {
                        "chunk_id": "text-architecture",
                        "document_id": "doc-aspire",
                        "filename": "ASPIRE.pdf",
                        "content": "architecture " * 500,
                    },
                    {
                        "chunk_id": "table:benchmark",
                        "document_id": "doc-aspire",
                        "filename": "ASPIRE.pdf",
                        "artifact_type": "table",
                        "chunk_type": "table",
                        "table_id": "benchmark",
                        "content": (
                            "table_type: comparison\ncontent:\n"
                            "| Model | Acc | F1 | CCC |\n"
                            "|---|---:|---:|---:|\n"
                            "| Proposed | 75 | 74 | 0.70 |"
                        ),
                    },
                ],
                "selected_document_ids": ["doc-aspire"],
                "forced_document_ids": ["doc-aspire"],
                "retrieval_channels": ["lancedb_table_chunks"],
                "document_card_results": [],
            }

        def expand_with_neighbor_chunks(self, results, **kwargs):  # noqa: ANN003
            return results

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda settings: object())

    retrieval = asyncio.run(
        agent_api._retrieve_legacy_for_agent(  # noqa: SLF001
            rag=_RagWithLateTable(),
            settings=Settings(data_dir=tmp_path),
            query="ASPIRE benchmark Acc F1 CCC",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-aspire"],
            answer_intent="direct_answer",
            prefer_tables=True,
        )
    )

    assert retrieval["documents"][0]["table_id"] == "benchmark"
    assert retrieval["context_stats"]["table_source_count"] == 1


def test_visual_provider_runs_for_explicit_figure_question_outside_coverage(
    monkeypatch, tmp_path
) -> None:
    """Visual retrieval must not be limited to document-wide page queries."""

    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings

    class _Rag:
        async def search_hybrid(self, **kwargs):  # noqa: ANN003
            return {
                "results": [
                    {
                        "chunk_id": "text-figure",
                        "document_id": "doc-a",
                        "content": "Figure 1 is referenced in the method section.",
                        "score": 0.9,
                    }
                ],
                "selected_document_ids": ["doc-a"],
                "retrieval_channels": ["lancedb_text_chunks"],
            }

        def expand_with_neighbor_chunks(self, results, **kwargs):  # noqa: ANN003
            return results

        async def retrieve_visual_assets(self, **kwargs):  # noqa: ANN003
            return {"results": []}

    calls: list[str] = []

    async def fake_visual(**kwargs):  # noqa: ANN003
        calls.append(kwargs["query"])
        return (
            [
                {
                    "chunk_id": "visual-page-1",
                    "document_id": "doc-a",
                    "page_number": 1,
                    "artifact_type": "page",
                    "visual_page": True,
                    "coverage_mode": True,
                    "image_path": str(tmp_path / "page-1.png"),
                    "content": "Visual page evidence",
                    "retrieval_channels": ["visual_page", "colsmol"],
                }
            ],
            None,
        )

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieve_visual_pages_for_agent", fake_visual)

    result = asyncio.run(
        agent_api._retrieve_legacy_for_agent(  # noqa: SLF001
            rag=_Rag(),
            settings=Settings(data_dir=tmp_path, visual_retrieval_enabled=True),
            query="What does Figure 1 show?",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a"],
            answer_intent="direct_answer",
        )
    )

    assert calls == ["What does Figure 1 show?"]
    assert result["context_stats"]["visual_page_source_count"] == 1
    assert result["documents"][0]["visual_page"] is True


def test_visual_provider_does_not_run_for_plain_text_question(monkeypatch, tmp_path) -> None:
    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings

    class _Rag:
        async def search_hybrid(self, **kwargs):  # noqa: ANN003
            return {
                "results": [
                    {
                        "chunk_id": "text-date",
                        "document_id": "doc-a",
                        "content": "The complaint was filed in October.",
                        "score": 0.9,
                    }
                ]
            }

        def expand_with_neighbor_chunks(self, results, **kwargs):  # noqa: ANN003
            return results

    async def unexpected_visual(**kwargs):  # noqa: ANN003
        raise AssertionError("plain text retrieval must not invoke visual provider")

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieve_visual_pages_for_agent", unexpected_visual)

    result = asyncio.run(
        agent_api._retrieve_legacy_for_agent(  # noqa: SLF001
            rag=_Rag(),
            settings=Settings(data_dir=tmp_path, visual_retrieval_enabled=True),
            query="When was the complaint filed?",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a"],
            answer_intent="direct_answer",
        )
    )

    assert result["context_stats"].get("visual_page_source_count") == 0


def test_visual_provider_runs_for_active_image_only_readiness(monkeypatch, tmp_path) -> None:
    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings

    class _Rag:
        async def search_hybrid(self, **kwargs):  # noqa: ANN003
            return {
                "results": [
                    {
                        "chunk_id": "table-image",
                        "document_id": "doc-a",
                        "content": "Table image projection",
                        "score": 0.9,
                    }
                ],
                "selected_document_ids": ["doc-a"],
            }

        def documents_requiring_visual_channel(self, document_ids):  # noqa: ANN001
            return [document_id for document_id in document_ids if document_id == "doc-a"]

        def expand_with_neighbor_chunks(self, results, **kwargs):  # noqa: ANN003
            return results

    calls: list[str] = []

    async def fake_visual(**kwargs):  # noqa: ANN003
        calls.append(kwargs["query"])
        return (
            [
                {
                    "chunk_id": "visual-page-1",
                    "document_id": "doc-a",
                    "page_number": 1,
                    "artifact_type": "page",
                    "visual_page": True,
                    "coverage_mode": True,
                    "content": "Visual page evidence",
                    "retrieval_channels": ["visual_page", "colsmol"],
                }
            ],
            None,
        )

    monkeypatch.setattr(agent_api, "_embedding_provider", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieval_store", lambda settings: object())
    monkeypatch.setattr(agent_api, "_retrieve_visual_pages_for_agent", fake_visual)

    result = asyncio.run(
        agent_api._retrieve_legacy_for_agent(  # noqa: SLF001
            rag=_Rag(),
            settings=Settings(data_dir=tmp_path, visual_retrieval_enabled=True),
            query="How many vendors are in the table?",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a"],
            answer_intent="direct_answer",
        )
    )

    assert calls == ["How many vendors are in the table?"]
    assert result["context_stats"]["visual_page_source_count"] == 1
    assert result["diagnostics"]["visual_readiness_required"] is True


def test_scope_documents_to_focus_no_fallback_leak() -> None:
    scoped = _scope_documents_to_focus(
        [
            {"document_id": "doc-a", "filename": "A.pdf"},
            {"document_id": "doc-b", "filename": "B.pdf"},
        ],
        ["doc-missing"],
    )
    assert scoped == []



def test_curate_figure_sources_exact_figure_number_on_focus_doc() -> None:
    documents = [
        {
            "document_id": "doc-whisper",
            "figure_id": "whisper-fig-2",
            "figure_index": 1,
            "caption": "Figure 2: Distribution of primary emotions",
            "score": 0.99,
            "chunk_id": "wf2",
        },
        {
            "document_id": "doc-aspire",
            "figure_id": "aspire-fig-2",
            "figure_index": 1,
            "caption": "Figure 2: Per-class F1 comparison between ASPIRE and the",
            "score": 0.4,
            "chunk_id": "af2",
        },
    ]
    curated = _curate_figure_sources(
        documents,
        answer_intent="direct_answer",
        focus_document_ids=["doc-aspire"],
        query="Figure 2 của ASPIRE",
    )
    figure_ids = [doc["figure_id"] for doc in curated if doc.get("figure_id")]
    assert figure_ids == ["aspire-fig-2"]


def test_second_retrieval_retry_payload_exposes_bounded_parallel_plan() -> None:
    from app.api import agent as agent_api
    from app.services.retrieval_agent_service import RetrievalBranch, SecondRetrievalPlan

    branches = [
        RetrievalBranch(
            query="A architecture",
            focus_document_ids=["doc-a"],
            reason="adaptive_second_hop:missing_query_facets",
            hop=2,
            facets=["architecture"],
        ),
        RetrievalBranch(
            query="B benchmark",
            focus_document_ids=["doc-b"],
            reason="adaptive_second_hop:missing_query_facets",
            hop=2,
            facets=["benchmark_results"],
            bridge_anchors=["Cross-modal fusion"],
        ),
    ]
    plan = SecondRetrievalPlan(
        query=branches[0].query,
        reasons=["missing_query_facets"],
        queries=[branch.query for branch in branches],
        branches=branches,
        hop_count=2,
        missing_facets=["architecture", "benchmark_results"],
        bridge_anchors=["Cross-modal fusion"],
    )
    payload = agent_api._second_retrieval_retry_payload(  # noqa: SLF001
        run_id="run-1",
        conversation_id="conversation-1",
        plan=plan,
        branches=branches,
        max_hops=2,
        missing_entities=[],
        previous_focus_document_ids=["doc-a", "doc-b"],
        agent_reasoning="smart",
        smart_allowed=True,
    )

    assert payload["sub_queries"] == ["A architecture", "B benchmark"]
    assert payload["hop"] == 2
    assert payload["max_hops"] == 2
    assert payload["parallel"] is True
    assert payload["reasons"] == ["missing_query_facets"]
    assert payload["branches"][0]["focus_document_ids"] == ["doc-a"]
    assert payload["missing_facets"] == ["architecture", "benchmark_results"]
    assert payload["bridge_anchors"] == ["Cross-modal fusion"]

    single_payload = agent_api._second_retrieval_retry_payload(  # noqa: SLF001
        run_id="run-1",
        conversation_id="conversation-1",
        plan=plan,
        branches=branches[:1],
        max_hops=2,
        missing_entities=[],
        previous_focus_document_ids=["doc-a", "doc-b"],
        agent_reasoning="smart",
        smart_allowed=True,
    )
    assert single_payload["parallel"] is False

    no_progress = agent_api._second_hop_diagnostics_payload(  # noqa: SLF001
        plan=plan,
        branches=branches[:1],
        branch_diagnostics=[
            {
                "query": branches[0].query,
                "focus_document_ids": ["doc-a"],
                "timing_ms": 8.0,
            }
        ],
        max_hops=2,
        smart_retrieval=True,
        total_ms=8.0,
        new_evidence_count=0,
    )
    assert no_progress["retry_discarded"] is True
    assert no_progress["retry_discard_reason"] == "no_new_evidence"
    assert no_progress["adaptive_second_hop"]["parallel"] is False
    assert no_progress["adaptive_second_hop"]["discarded"] is True


def test_second_retrieval_branches_run_in_parallel_with_exact_scope_and_cap(
    monkeypatch,
) -> None:
    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings
    from app.services.retrieval_agent_service import RetrievalBranch

    active = 0
    max_active = 0
    calls: list[dict] = []

    async def fake_retrieve_for_agent(**kwargs):  # noqa: ANN003
        nonlocal active, max_active
        calls.append(kwargs)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        document_id = kwargs["focus_document_ids"][0]
        return {
            "mode": f"mode:{document_id}",
            "documents": [
                {
                    "chunk_id": f"chunk:{document_id}",
                    "document_id": document_id,
                    "content": "evidence",
                }
            ],
            "diagnostics": {
                "selected_engine": "legacy",
                "policy_reason": "focused_single_document_fast_path",
            },
        }

    monkeypatch.setattr(agent_api, "_retrieve_for_agent", fake_retrieve_for_agent)
    branches = [
        RetrievalBranch(
            query=f"query {suffix}",
            focus_document_ids=[f"doc-{suffix}"],
            reason="adaptive_second_hop:coverage",
            hop=2,
        )
        for suffix in ("a", "b", "c")
    ]
    results, diagnostics = asyncio.run(
        agent_api._execute_second_retrieval_branches(  # noqa: SLF001
            branches=branches,
            rag=object(),
            settings=Settings(
                agentic_retrieval_max_subqueries=2,
                agentic_retrieval_hop_timeout_seconds=5,
            ),
            collection_id=None,
            retrieval_mode="auto",
            answer_intent="compare",
            answer_depth="normal",
            include_visual_boost=False,
            prefer_legacy_tables=False,
        )
    )

    assert max_active == 2
    assert len(results) == 2
    assert [call["focus_document_ids"] for call in calls] == [
        ["doc-a"],
        ["doc-b"],
    ]
    assert all(call["allow_decomposition"] is False for call in calls)
    assert [item["focus_document_ids"] for item in diagnostics] == [
        ["doc-a"],
        ["doc-b"],
    ]
    assert all(item["timing_ms"] >= 0 for item in diagnostics)


def test_second_retrieval_branch_timeout_and_provider_error_propagate(
    monkeypatch,
) -> None:
    import asyncio

    import pytest

    from app.api import agent as agent_api
    from app.core.config import Settings
    from app.services.retrieval_agent_service import RetrievalBranch

    branch = RetrievalBranch(
        query="timed query",
        focus_document_ids=["doc-a"],
        reason="adaptive_second_hop:coverage",
        hop=2,
    )

    async def slow_retrieval(**_kwargs):
        await asyncio.sleep(0.05)
        return {"mode": "late", "documents": []}

    monkeypatch.setattr(agent_api, "_retrieve_for_agent", slow_retrieval)
    timeout_settings = Settings.model_construct(
        agentic_retrieval_max_subqueries=1,
        agentic_retrieval_hop_timeout_seconds=0.005,
    )
    with pytest.raises(TimeoutError, match="exceeded"):
        asyncio.run(
            agent_api._execute_second_retrieval_branches(  # noqa: SLF001
                branches=[branch],
                rag=object(),
                settings=timeout_settings,
                collection_id=None,
                retrieval_mode="auto",
                answer_intent="compare",
                answer_depth="normal",
                include_visual_boost=False,
                prefer_legacy_tables=False,
            )
        )

    async def provider_failure(**_kwargs):
        raise RuntimeError("9router quota exhausted")

    monkeypatch.setattr(agent_api, "_retrieve_for_agent", provider_failure)
    with pytest.raises(RuntimeError, match="9router quota exhausted"):
        asyncio.run(
            agent_api._execute_second_retrieval_branches(  # noqa: SLF001
                branches=[branch],
                rag=object(),
                settings=Settings(
                    agentic_retrieval_max_subqueries=1,
                    agentic_retrieval_hop_timeout_seconds=5,
                ),
                collection_id=None,
                retrieval_mode="auto",
                answer_intent="compare",
                answer_depth="normal",
                include_visual_boost=False,
                prefer_legacy_tables=False,
            )
        )


def test_accumulated_retrieval_preserves_engine_policy_diagnostics() -> None:
    from app.api import agent as agent_api

    merged = agent_api._compose_accumulated_retrieval(  # noqa: SLF001
        query="compare A and B",
        retrievals=[
            {
                "mode": "legacy",
                "documents": [
                    {
                        "chunk_id": "a",
                        "document_id": "doc-a",
                        "content": "A",
                    }
                ],
                "diagnostics": {
                    "selected_engine": "legacy",
                    "policy_reason": "focused_single_document_fast_path",
                },
            },
            {
                "mode": "lightrag",
                "documents": [
                    {
                        "chunk_id": "b",
                        "document_id": "doc-b",
                        "content": "B",
                    }
                ],
                "diagnostics": {
                    "selected_engine": "lightrag",
                    "policy_reason": "cross_document_reasoning",
                },
            },
        ],
        answer_intent="compare",
        answer_depth="normal",
        include_visual_boost=False,
        prefer_tables=False,
    )

    diagnostics = merged["diagnostics"]
    assert diagnostics["selected_engine"] == "legacy"
    assert diagnostics["policy_reason"] == "focused_single_document_fast_path"
    assert diagnostics["selected_engines"] == ["legacy", "lightrag"]
    assert diagnostics["policy_reasons"] == [
        "focused_single_document_fast_path",
        "cross_document_reasoning",
    ]
    assert (
        agent_api._retrieval_engine_policy_reason(  # noqa: SLF001
            configured_engine="auto",
            retrieval_mode="auto",
            answer_intent="compare",
            focus_document_ids=["doc-a", "doc-b"],
            prefer_legacy_tables=False,
        )
        == "cross_document_reasoning"
    )


def test_parallel_branches_are_one_hop_and_canonical_evidence_is_deduped() -> None:
    from app.api import agent as agent_api

    initial_documents = [
        {
            "document_id": "doc-a",
            "table_id": "table-1",
            "chunk_id": "table-old-chunk",
            "content": "initial table",
        },
        {
            "document_id": "doc-a",
            "figure_id": "figure-1",
            "chunk_id": "figure-old-chunk",
            "content": "initial figure",
        },
        {
            "document_id": "doc-a",
            "parent_chunk_id": "parent-1",
            "chunk_id": "child-1",
            "content": "initial parent evidence",
        },
    ]
    additional_documents = [
        {
            "document_id": "doc-a",
            "table_id": "table-1",
            "chunk_id": "table-new-chunk",
            "content": "same canonical table",
        },
        {
            "document_id": "doc-a",
            "figure_id": "figure-1",
            "chunk_id": "figure-new-chunk",
            "content": "same canonical figure",
        },
        {
            "document_id": "doc-a",
            "parent_chunk_id": "parent-1",
            "chunk_id": "child-2",
            "content": "same canonical parent",
        },
        {
            "document_id": "doc-b",
            "table_id": "table-1",
            "content": "same table id but a different canonical document",
        },
    ]
    new_evidence = agent_api._new_retrieval_evidence(  # noqa: SLF001
        initial_documents=initial_documents,
        additional_documents=additional_documents,
    )
    merged_documents = agent_api._merge_retrieved_documents(  # noqa: SLF001
        initial_documents,
        additional_documents,
    )

    assert [item["document_id"] for item in new_evidence] == ["doc-b"]
    assert len(merged_documents) == len(initial_documents) + 1

    accumulated = agent_api._compose_accumulated_retrieval(  # noqa: SLF001
        query="survey",
        retrievals=[
            {
                "mode": "hop-1",
                "documents": [
                    {
                        "document_id": "doc-initial",
                        "chunk_id": "initial",
                        "content": "initial",
                    }
                ],
            },
            *[
                {
                    "mode": f"hop-2-branch-{index}",
                    "documents": [
                        {
                            "document_id": f"doc-{index}",
                            "chunk_id": f"branch-{index}",
                            "content": f"branch {index}",
                        }
                    ],
                }
                for index in range(3)
            ],
        ],
        answer_intent="compare",
        answer_depth="normal",
        include_visual_boost=False,
        prefer_tables=False,
    )
    assert accumulated["diagnostics"]["hop_count"] == 2
    assert accumulated["diagnostics"]["branch_count"] == 3


def test_retrieval_cache_configuration_tracks_agentic_policy() -> None:
    from app.api import agent as agent_api
    from app.core.config import Settings

    first = agent_api._retrieval_index_configuration(  # noqa: SLF001
        Settings(
            agentic_retrieval_max_hops=1,
            agentic_retrieval_max_subqueries=1,
        )
    )
    second = agent_api._retrieval_index_configuration(  # noqa: SLF001
        Settings(
            agentic_retrieval_max_hops=2,
            agentic_retrieval_max_subqueries=3,
        )
    )

    assert first["retrieval_engine"] == second["retrieval_engine"]
    assert first["context_projection_version"] == agent_api.RETRIEVAL_CACHE_VERSION
    assert first["agentic_retrieval_max_hops"] == 1
    assert first["agentic_retrieval_max_subqueries"] == 1
    assert second["agentic_retrieval_max_hops"] == 2
    assert second["agentic_retrieval_max_subqueries"] == 3
    assert first != second


def test_retrieval_cache_key_preserves_raw_artifact_obligations() -> None:
    common = {
        "normalized_query": "aspire table results metrics",
        "collection_id": None,
        "focus_document_ids": ["doc-aspire"],
        "retrieval_mode": "auto",
        "index_fingerprint": "index-v1",
    }

    table_two = _retrieval_cache_key(
        **common,
        original_query="ASPIRE Table 2 benchmark results",
    )
    table_four = _retrieval_cache_key(
        **common,
        original_query="ASPIRE Table 4 ablation results",
    )
    generic_benchmark = _retrieval_cache_key(
        **common,
        original_query="ASPIRE benchmark Acc F1 CCC",
    )

    assert table_two != table_four
    assert table_two != generic_benchmark
    assert table_four != generic_benchmark


def test_compare_decomposition_reports_branch_engine_policy(monkeypatch) -> None:
    import asyncio

    from app.api import agent as agent_api
    from app.core.config import Settings

    async def fake_legacy_retrieval(**kwargs):  # noqa: ANN003
        document_id = kwargs["focus_document_ids"][0]
        return {
            "mode": "legacy:hybrid",
            "documents": [
                {
                    "document_id": document_id,
                    "chunk_id": f"chunk:{document_id}",
                    "content": f"evidence for {document_id}",
                }
            ],
            "diagnostics": {
                "selected_engine": "legacy",
                "policy_reason": "focused_single_document_fast_path",
            },
        }

    monkeypatch.setattr(
        agent_api,
        "_retrieve_legacy_for_agent",
        fake_legacy_retrieval,
    )
    retrieval = asyncio.run(
        agent_api._retrieve_for_agent(  # noqa: SLF001
            rag=object(),
            settings=Settings(retrieval_engine="legacy"),
            query="compare A and B",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a", "doc-b"],
            answer_intent="compare",
        )
    )

    diagnostics = retrieval["diagnostics"]
    assert diagnostics["selected_engine"] == "decomposed_compare"
    assert diagnostics["policy_reason"] == "compare_per_document"
    assert diagnostics["branch_selected_engines"] == ["legacy", "legacy"]
    assert diagnostics["branch_policy_reasons"] == [
        "configured_legacy",
        "configured_legacy",
    ]


def test_main_result_gathering_keeps_only_best_matching_tables_per_document() -> None:
    # Every reserved table displaces a retrieved source from a 3-5 source
    # context; a paper with many result tables must not flood it.
    class InventoryRag:
        def get_document(self, document_id):
            return {"filename": "report.pdf", "active_revision_id": "active"}

        def list_document_tables(self, document_id):
            return [
                {"id": f"t{index}", "revision_id": "active",
                 "caption": f"Table {index}. Results on {name}",
                 "markdown": f"| Model | Accuracy |\n|---|---|\n| Encoder | 0.{index} |"}
                for index, name in enumerate(["Corpus-A", "Corpus-B", "Corpus-C", "Corpus-D"], start=1)
            ]

    merged, count = _merge_focused_table_inventory(
        rag=InventoryRag(), sources=[], focus_document_ids=["report"],
        query="Compare the accuracy results on Corpus-C",
    )
    assert count == 2
    assert merged[0]["table_id"] == "t3"


def test_direct_table_answer_never_shows_projected_page_context():
    table = {
        "document_id": "whiser", "table_id": "t1", "artifact_type": "table", "filename": "WhiSER.pdf",
        "page_number": 3, "caption": "Table 1: CCC results for arousal",
        "content": (
            "| Model | LR |\n|---|---|\n| wavLM | 0.299 |\n\n"
            "[same-page native text context]\nWe evaluate if the results are statistically significant."
        ),
    }
    direct = _direct_canonical_table_answer("Show Table 1 from WhiSER.", [table])
    assert direct is not None
    answer, _source = direct
    assert "| wavLM | 0.299 |" in answer
    assert "same-page" not in answer and "statistically significant" not in answer


def test_described_table_wins_over_a_misremembered_number():
    from app.api.agent import _described_table_numbers

    tables = [
        {"caption": "Table 1: Passenger counts by station", "table_index": 0},
        {"caption": "Table 2: Delay statistics by line and month", "table_index": 1},
        {"caption": "Table 3: Delay forecasting results by model", "table_index": 2},
    ]
    # Number and description disagree: the described table wins.
    assert _described_table_numbers("Show Table 1, the delay forecasting results by model", tables, (1,)) == (3,)
    # Number alone, or a description that fits the number, keeps the number.
    assert _described_table_numbers("Show Table 1", tables, (1,)) == (1,)
    assert _described_table_numbers("Show Table 1 with passenger counts", tables, (1,)) == (1,)
    # A weak, one-word overlap is not enough to override the user.
    assert _described_table_numbers("Show Table 1 on delay", tables, (1,)) == (1,)
