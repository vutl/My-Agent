from app.rag.context import compose_retrieval_context


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
