from app.rag.context import compose_retrieval_context, select_ranked_evidence


def test_selection_trace_attributes_drops_without_exposing_content():
    rows = [{"chunk_id": "first", "document_id": "doc", "content": "Private text"},
            {"chunk_id": "first", "document_id": "doc", "content": "Private text"},
            {"chunk_id": "extra", "document_id": "doc", "content": "Private second text"},
            {"chunk_id": "empty", "document_id": "other", "content": ""},
            {"table_id": "large", "document_id": "other", "content": "X" * 500}]
    result = compose_retrieval_context(rows, max_sources=8, max_chunks_per_document=1,
                                       max_chars=350)
    trace = result.stats["selection_trace"]
    assert trace["complete"]
    assert trace["counts"] == {"selected": 1, "duplicate_canonical_identity": 1,
                                "per_document_limit": 1, "empty_content": 1, "character_budget": 1}
    assert "Private" not in str(trace)
    assert {event["evidence_id"] for event in trace["events"]} >= {"text:first", "table:large"}


def test_selection_trace_is_bounded_and_does_not_change_context():
    rows = [{"chunk_id": str(i), "document_id": str(i), "content": "Text " * 100} for i in range(5)]
    normal = compose_retrieval_context(rows, max_sources=2, max_chars_per_source=60)
    bounded = compose_retrieval_context(rows, max_sources=2, max_chars_per_source=60,
                                        selection_trace_limit=1)
    assert normal.sources == bounded.sources and normal.context_text == bounded.context_text
    assert normal.stats["selection_trace"]["counts"] == {"selected": 2, "source_limit_unexamined": 3}
    assert normal.stats["selection_trace"]["events"][0]["projected"]
    assert not bounded.stats["selection_trace"]["complete"]
    assert bounded.stats["selection_trace"]["omitted_events"] == 4


def test_context_preserves_independent_figures_on_same_page():
    figures = [{"chunk_id": f"vector-{i}", "figure_id": f"figure-{i}",
                "document_id": "doc", "artifact_type": "figure", "page_number": 7,
                "content": f"Figure {i}: independent experiment"} for i in (1, 2)]
    composed = compose_retrieval_context(figures, min_figures=2, max_sources=8,
                                          query="Compare both figures on page 7")
    assert {source["figure_id"] for source in composed.sources} == {"figure-1", "figure-2"}


def test_context_deduplicates_canonical_artifact_not_vector_alias():
    figure = {"chunk_id": "vector-a", "figure_id": "canonical", "document_id": "doc",
              "artifact_type": "figure", "page_number": 7, "content": "Caption"}
    duplicate = {**figure, "chunk_id": "vector-b"}
    assert len(compose_retrieval_context([figure, duplicate]).sources) == 1


def test_context_keeps_crop_and_full_page_as_distinct_evidence():
    figure = {"chunk_id": "crop", "figure_id": "figure", "document_id": "doc",
              "artifact_type": "figure", "page_number": 7, "content": "Cropped figure"}
    page = {"chunk_id": "page-vector", "document_id": "doc", "artifact_type": "visual_page",
            "page_number": 7, "visual_page": True, "content": "Other content across whole page"}
    assert len(compose_retrieval_context([figure, page], max_sources=8).sources) == 2
    duplicate_page = {**page, "chunk_id": "other-page-vector"}
    assert len(compose_retrieval_context([page, duplicate_page], max_sources=8).sources) == 1


def test_empty_visual_projection_does_not_suppress_valid_same_page_source():
    empty = {"chunk_id": "first", "document_id": "doc", "artifact_type": "visual_page",
             "page_number": 7, "content": "", "visual_page": True}
    valid = {**empty, "chunk_id": "second", "content": "Readable page evidence"}
    composed = compose_retrieval_context([empty, valid], max_sources=8)
    assert len(composed.sources) == 1
    assert composed.sources[0]["content"] == "Readable page evidence"


def test_compose_retrieval_context_prioritizes_figures_when_requested() -> None:
    results = [
        {"chunk_id": "t1", "document_id": "doc-a", "content": "text one"},
        {"chunk_id": "t2", "document_id": "doc-a", "content": "text two"},
        {"chunk_id": "f2", "document_id": "doc-a", "figure_id": "fig-2", "content": "Fig 2 caption"},
        {"chunk_id": "f1", "document_id": "doc-a", "figure_id": "fig-1", "content": "Fig 1 caption"},
    ]
    composed = compose_retrieval_context(results, max_sources=4, min_figures=2)
    figure_ids = [source["figure_id"] for source in composed.sources if source.get("figure_id")]
    assert figure_ids[:2] == ["fig-2", "fig-1"]


def test_compose_retrieval_context_reserves_requested_figure_label() -> None:
    """A concrete Figure N request outranks a merely similar visual hit."""

    results = [
        {
            "chunk_id": "visual-82",
            "document_id": "doc-a",
            "figure_id": "fig-82",
            "figure_index": 81,
            "artifact_type": "figure",
            "page_number": 82,
            "caption": "Figure 43: A visually similar machine panel",
            "content": "similar visual evidence",
            "score": 0.9,
        },
        {
            "chunk_id": "visual-118",
            "document_id": "doc-a",
            "figure_id": "fig-89",
            "figure_index": 105,
            "artifact_type": "figure",
            "page_number": 118,
            "caption": "Figure 89: Coffee machine control panel",
            "content": "the requested figure",
            "score": 0.7,
        },
    ]

    composed = compose_retrieval_context(
        results,
        query="In Figure 89, what is shown in the coffee machine picture?",
        max_sources=1,
        min_figures=1,
    )

    assert [source["figure_id"] for source in composed.sources] == ["fig-89"]

    composed_alias = compose_retrieval_context(
        results,
        query="What does Fig. 89 show?",
        max_sources=1,
        min_figures=1,
    )
    assert [source["figure_id"] for source in composed_alias.sources] == ["fig-89"]


def test_select_ranked_evidence_reserves_requested_table_before_text_fill() -> None:
    results = [
        {"chunk_id": "text-1", "document_id": "doc-a", "score": 1.0, "content": "text"},
        {"chunk_id": "text-2", "document_id": "doc-a", "score": 0.9, "content": "more text"},
        {"chunk_id": "table-row", "table_id": "table-1", "artifact_type": "table", "document_id": "doc-a", "score": 0.2, "content": "Table 1"},
    ]
    selected, _ = select_ranked_evidence(
        results,
        limit=2,
        query="What results does Table 1 report?",
        required_document_ids=["doc-a"],
    )
    assert any(item.get("table_id") == "table-1" for item in selected)


def test_select_ranked_evidence_keeps_two_sources_per_required_document() -> None:
    results = [
        {"chunk_id": f"a-{index}", "document_id": "doc-a", "score": 1.0 - index / 100, "content": f"a {index}"}
        for index in range(6)
    ] + [
        {"chunk_id": f"b-{index}", "document_id": "doc-b", "score": 0.5 - index / 100, "content": f"b {index}"}
        for index in range(6)
    ]
    selected, _ = select_ranked_evidence(
        results,
        limit=4,
        query="compare the purpose of both documents",
        required_document_ids=["doc-a", "doc-b"],
    )
    counts = {doc: sum(item["document_id"] == doc for item in selected) for doc in ("doc-a", "doc-b")}
    assert counts == {"doc-a": 2, "doc-b": 2}
