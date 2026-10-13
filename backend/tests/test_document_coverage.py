from pathlib import Path

from app.rag.document_coverage import detect_document_coverage
from app.rag.context import compose_retrieval_context
from app.api.agent import (
    _merge_page_coverage_sources,
    _exhaustive_short_document_pages,
    _enrich_visual_sources,
    _retrieved_image_data_urls,
    _visual_source_count,
    _visual_attachment_labels,
    _should_reuse_last_retrieval,
)
from app.rag.retriever import RetrievedChunk
from app.services.rag_service import RagService


def test_document_coverage_detects_english_page_aggregation() -> None:
    policy = detect_document_coverage(
        "How many pages include charts whose horizontal axis is set as year?"
    )

    assert policy.enabled is True
    assert policy.reason == "page_aggregation"
    assert policy.page_limit == 32


def test_document_coverage_detects_vietnamese_page_list() -> None:
    policy = detect_document_coverage("Liệt kê những trang có biểu đồ trong toàn bộ tài liệu")

    assert policy.enabled is True
    assert policy.reason == "page_aggregation"


def test_document_coverage_does_not_widen_normal_fact_or_table_questions() -> None:
    assert not detect_document_coverage("What does the architecture do?").enabled
    assert not detect_document_coverage("Show Table 2 results for ASPIRE").enabled
    assert not detect_document_coverage("Compare all models in the table").enabled


def test_document_coverage_handles_ranked_items_in_a_document() -> None:
    policy = detect_document_coverage(
        "What are the top2 texts of the yellow words in the document?"
    )

    assert policy.enabled is True
    assert policy.reason == "document_wide_aggregation"


def test_document_coverage_handles_page_superlatives() -> None:
    policy = detect_document_coverage(
        "What is the Top-Level Page name of the page with the slowest page processing time?"
    )

    assert policy.enabled is True
    assert policy.reason == "page_superlative"


def test_document_coverage_does_not_reuse_compact_followup_cache() -> None:
    rewrite = type(
        "Rewrite",
        (),
        {
            "is_followup": True,
            "use_last_sources": True,
            "answer_intent": "elaborate",
        },
    )()

    assert not _should_reuse_last_retrieval(
        rewrite,
        original_query="How many pages contain charts across the document?",
    )


def test_page_coverage_context_allows_multiple_pages_from_one_document() -> None:
    results = [
        {
            "chunk_id": f"page:d1:{page}",
            "document_id": "d1",
            "filename": "scan.pdf",
            "page_number": page,
            "artifact_type": "page",
            "coverage_mode": True,
            "content": f"Page {page} contains a chart with year on the horizontal axis.",
            "retrieval_channels": ["sqlite_fts5_page_coverage"],
        }
        for page in range(1, 14)
    ]

    composed = compose_retrieval_context(
        results,
        max_sources=24,
        max_chars=14_000,
        max_chars_per_source=760,
        max_chunks_per_document=3,
        coverage_mode=True,
        query="How many pages contain charts?",
    )

    assert len(composed.sources) == 13
    assert composed.stats["page_source_count"] == 13
    assert composed.stats["coverage_mode"] is True
    assert {item["page_number"] for item in composed.sources} == set(range(1, 14))


def test_visual_page_join_preserves_ocr_content_and_adds_image_provenance() -> None:
    merged = _merge_page_coverage_sources(
        [
            {
                "chunk_id": "page:d1:3",
                "document_id": "d1",
                "page_number": 3,
                "artifact_type": "page",
                "coverage_mode": True,
                "content": "OCR chart text",
                "retrieval_channels": ["sqlite_fts5_page_coverage"],
            }
        ],
        [
            {
                "chunk_id": "page:d1:r2:3",
                "document_id": "d1",
                "page_number": 3,
                "artifact_type": "page",
                "coverage_mode": True,
                "content": "Visual page evidence selected from the active document revision.",
                "image_path": "/tmp/page-3.png",
                "visual_score": 0.91,
                "retrieval_channels": ["visual_page", "colsmol"],
            }
        ],
    )

    assert len(merged) == 1
    assert merged[0]["content"] == "OCR chart text"
    assert merged[0]["image_path"] == "/tmp/page-3.png"
    assert merged[0]["visual_score"] == 0.91
    assert merged[0]["retrieval_channels"] == [
        "sqlite_fts5_page_coverage",
        "visual_page",
        "colsmol",
    ]


def test_visual_page_join_interleaves_new_visual_pages_before_context_cap() -> None:
    text_pages = [
        {
            "chunk_id": f"page:d1:{page}",
            "document_id": "d1",
            "page_number": page,
            "artifact_type": "page",
            "coverage_mode": True,
            "content": f"text page {page}",
        }
        for page in (1, 2, 3, 4)
    ]
    visual_pages = [
        {
            "chunk_id": f"page:d1:r2:{page}",
            "document_id": "d1",
            "page_number": page,
            "artifact_type": "page",
            "coverage_mode": True,
            "content": "visual page",
            "image_path": f"/tmp/page-{page}.png",
            "visual_score": 1.0 - page / 100,
        }
        for page in (5, 6, 7, 8)
    ]

    merged = _merge_page_coverage_sources(text_pages, visual_pages)

    assert [item["page_number"] for item in merged] == [5, 1, 6, 2, 7, 3, 8, 4]
    assert all(item.get("coverage_mode") and item.get("visual_page") for item in merged[:1:2])


def test_visual_page_images_are_bounded_and_feature_gated(tmp_path: Path) -> None:
    image = tmp_path / "page-1.png"
    image.write_bytes(b"fake-png")
    documents = [
        {
            "document_id": "d1",
            "page_number": 1,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(image),
        },
        {
            "document_id": "d1",
            "page_number": 1,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(image),
        },
    ]

    assert _retrieved_image_data_urls(documents, enabled=False) == []
    urls = _retrieved_image_data_urls(documents, enabled=True, max_images=4)
    assert len(urls) == 1
    assert urls[0].startswith("data:image/png;base64,")


def test_context_reserves_all_explicit_visual_pages() -> None:
    results = [
        {
            "chunk_id": f"visual-{page}",
            "document_id": "d1",
            "artifact_type": "page",
            "page_number": page,
            "visual_page": True,
            "coverage_mode": True,
            "retrieval_channels": ["visual_page", "colsmol"],
            "image_path": f"/tmp/page-{page}.png",
            "content": f"Visual page {page}",
        }
        for page in (12, 14)
    ] + [
        {
            "chunk_id": "text-noise",
            "document_id": "d1",
            "content": "ordinary text that must not evict requested pages",
            "page_number": 10,
        }
    ]

    composed = compose_retrieval_context(
        results,
        query="Which stages are shown on slides 12 and 14?",
        max_sources=2,
        max_chunks_per_document=2,
        min_visual_pages=2,
    )

    assert [item["page_number"] for item in composed.sources] == [12, 14]
    assert composed.stats["visual_page_source_count"] == 2


def test_visual_page_images_honor_explicit_page_hint(tmp_path: Path) -> None:
    page_one = tmp_path / "page-1.png"
    page_six = tmp_path / "page-6.png"
    page_one.write_bytes(b"page-one")
    page_six.write_bytes(b"page-six")
    documents = [
        {
            "document_id": "d1",
            "page_number": 1,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(page_one),
        },
        {
            "document_id": "d1",
            "page_number": 6,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(page_six),
        },
    ]

    urls = _retrieved_image_data_urls(
        documents,
        enabled=True,
        max_images=4,
        query="what is shown in the picture on slide 6?",
    )
    assert len(urls) == 1
    assert urls[0].endswith("cGFnZS1zaXg=")


def test_visual_attachment_labels_match_page_filtered_order(tmp_path: Path) -> None:
    page_one = tmp_path / "page-1.png"
    page_six = tmp_path / "page-6.png"
    page_one.write_bytes(b"page-one")
    page_six.write_bytes(b"page-six")
    documents = [
        {
            "document_id": "d1",
            "filename": "slides.pdf",
            "page_number": 1,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(page_one),
        },
        {
            "document_id": "d1",
            "filename": "slides.pdf",
            "page_number": 6,
            "artifact_type": "page",
            "visual_page": True,
            "image_path": str(page_six),
        },
    ]
    labels = _visual_attachment_labels(
        documents,
        enabled=True,
        max_images=4,
        query="what is shown in the picture on slide 6?",
    )
    assert labels == ["attachment 1: page 6; file slides.pdf"]


def test_visual_attachments_accept_canonical_figure_images(tmp_path: Path) -> None:
    figure = tmp_path / "figure-1.webp"
    figure.write_bytes(b"figure-bytes")
    urls = _retrieved_image_data_urls(
        [
            {
                "document_id": "d1",
                "page_number": 2,
                "artifact_type": "figure",
                "figure_id": "fig-1",
                "image_path": str(figure),
            }
        ],
        enabled=True,
        query="what does Figure 1 show?",
    )
    assert len(urls) == 1
    assert urls[0].startswith("data:image/webp;base64,")


def test_visual_attachments_accept_image_backed_table_without_affecting_native_table(
    tmp_path: Path,
) -> None:
    table_image = tmp_path / "table-3.png"
    table_image.write_bytes(b"table-image")
    image_table = {
        "document_id": "d1",
        "page_number": 4,
        "artifact_type": "table",
        "table_id": "table-3",
        "metadata": {"table_image_path": str(table_image)},
    }
    native_table = {
        "document_id": "d1",
        "page_number": 2,
        "artifact_type": "table",
        "table_id": "table-1",
        "metadata": {"markdown": "| metric | value |"},
    }

    enriched = _enrich_visual_sources([image_table, native_table])
    urls = _retrieved_image_data_urls(
        enriched,
        enabled=True,
        max_images=4,
        query="show the table values",
    )

    assert len(urls) == 1
    assert urls[0].startswith("data:image/png;base64,")
    assert enriched[0]["image_path"] == str(table_image)
    assert _visual_source_count(enriched) == 1


def test_visual_page_hit_is_reserved_for_multimodal_context_before_text_cap() -> None:
    results = [
        {
            "chunk_id": f"text-{index}",
            "document_id": "d1",
            "content": f"text evidence {index}",
            "score": 1.0 - index / 100,
        }
        for index in range(8)
    ] + [
        {
            "chunk_id": "visual-page-7",
            "document_id": "d1",
            "artifact_type": "page",
            "page_number": 7,
            "image_path": "/tmp/page-7.png",
            "content": "Visual page evidence selected from the active document revision.",
            "retrieval_channels": ["visual_page", "colsmol"],
            "score": 0.01,
        }
    ]

    composed = compose_retrieval_context(
        results,
        max_sources=4,
        max_chunks_per_document=2,
        min_visual_pages=1,
    )

    assert composed.sources[0]["page_number"] == 7
    assert composed.sources[0]["retrieval_channels"] == ["visual_page", "colsmol"]


def test_search_page_coverage_collapses_child_chunks_and_keeps_page_provenance() -> None:
    class FakeRag(RagService):
        def search(self, query: str, top_k: int = 5, document_ids=None):  # type: ignore[override]
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="d1",
                    source_path="scan.pdf",
                    filename="scan.pdf",
                    content="page one chart",
                    score=-10.0,
                    chunk_index=0,
                    page_number=1,
                    heading_path=[],
                    token_count=3,
                    char_count=15,
                ),
                RetrievedChunk(
                    chunk_id="c2",
                    document_id="d1",
                    source_path="scan.pdf",
                    filename="scan.pdf",
                    content="page one continuation",
                    score=-9.0,
                    chunk_index=1,
                    page_number=1,
                    heading_path=[],
                    token_count=3,
                    char_count=21,
                ),
                RetrievedChunk(
                    chunk_id="c3",
                    document_id="d1",
                    source_path="scan.pdf",
                    filename="scan.pdf",
                    content="page two chart",
                    score=-8.0,
                    chunk_index=2,
                    page_number=2,
                    heading_path=[],
                    token_count=3,
                    char_count=15,
                ),
            ]

    rows = FakeRag(None)  # type: ignore[arg-type]
    pages = rows.search_page_coverage("charts", page_limit=2, document_ids=["d1"])

    assert [item["page_number"] for item in pages] == [1, 2]
    assert pages[0]["chunk_id"] == "page:d1:1"
    assert pages[0]["coverage_chunk_ids"] == ["c1", "c2"]


def test_exhaustive_short_document_projection_is_bounded_by_active_page_count() -> None:
    class FakeRag:
        def document_page_counts(self, document_ids):
            return {document_ids[0]: 3}

        def list_page_coverage(self, *, document_ids, page_limit):
            assert page_limit == 3
            return [
                {
                    "document_id": document_ids[0],
                    "page_number": page,
                    "chunk_id": f"page:{document_ids[0]}:{page}",
                    "artifact_type": "page",
                    "content": f"page {page}",
                }
                for page in range(1, 4)
            ]

    pages, counts = _exhaustive_short_document_pages(
        rag=FakeRag(),
        focus_document_ids=["d1"],
        coverage_policy=detect_document_coverage(
            "How many pages contain charts across the document?"
        ),
    )

    assert counts == {"d1": 3}
    assert [item["page_number"] for item in pages] == [1, 2, 3]


def test_exhaustive_short_document_projection_does_not_widen_long_documents() -> None:
    class FakeRag:
        def document_page_counts(self, document_ids):
            return {document_ids[0]: 80}

        def list_page_coverage(self, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("long documents must keep bounded ranked coverage")

    pages, counts = _exhaustive_short_document_pages(
        rag=FakeRag(),
        focus_document_ids=["d1"],
        coverage_policy=detect_document_coverage(
            "How many pages contain charts across the document?"
        ),
    )

    assert pages == []
    assert counts == {"d1": 80}


def test_exhaustive_document_projection_allows_bounded_medium_documents() -> None:
    class FakeRag:
        def document_page_counts(self, document_ids):
            return {document_ids[0]: 40}

        def list_page_coverage(self, *, document_ids, page_limit):
            assert page_limit == 40
            return [{"document_id": document_ids[0], "page_number": page} for page in range(1, 41)]

    pages, counts = _exhaustive_short_document_pages(
        rag=FakeRag(),
        focus_document_ids=["d1"],
        coverage_policy=detect_document_coverage(
            "How many pages contain charts across the document?"
        ),
    )

    assert counts == {"d1": 40}
    assert len(pages) == 40
