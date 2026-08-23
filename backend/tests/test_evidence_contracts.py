from app.rag.evidence import (
    EvidenceBundle,
    FigureEvidence,
    TableEvidence,
    TextEvidence,
    evidence_candidate_from_result,
)
from app.rag.parsers import ParsedDocument, ParsedPage, ParsedFigure, ParsedTable
from app.rag.readiness import assess_document_readiness


def test_evidence_result_conversion_keeps_modality_and_provenance() -> None:
    table = evidence_candidate_from_result(
        {
            "chunk_id": "table:1",
            "document_id": "doc-a",
            "content": "Model | Acc",
            "artifact_type": "table",
            "table_id": "table:1",
            "table_index": 1,
            "caption": "Results",
            "page_number": 5,
            "raw_vector_score": 0.12,
            "rrf_score": 0.03,
            "retrieval_channels": ["lancedb_table_chunks"],
        },
        query_origins=("original", "rewrite"),
    )

    assert isinstance(table, TableEvidence)
    assert table.kind == "table"
    assert table.table_id == "table:1"
    assert table.scores.vector == 0.12
    assert table.query_origins == ("original", "rewrite")
    assert table.to_dict()["provenance"]["retrieval_channels"] == ["lancedb_table_chunks"]


def test_evidence_bundle_tracks_document_and_facet_coverage() -> None:
    results = [
        {
            "chunk_id": "text:1",
            "document_id": "doc-a",
            "content": "task and contribution",
            "chunk_type": "text",
            "evidence_facets": ["task", "contribution"],
        },
        {
            "chunk_id": "figure:1",
            "document_id": "doc-a",
            "content": "architecture diagram",
            "artifact_type": "figure",
            "figure_id": "figure:1",
            "evidence_facets": ["architecture"],
        },
    ]
    bundle = EvidenceBundle.from_results(
        query="what does the paper do?",
        results=results,
        query_origins=("original",),
        readiness_by_document={"doc-a": "partial"},
    )

    assert bundle.document_ids == ("doc-a",)
    assert bundle.facet_coverage["doc-a"] == ("architecture", "contribution", "task")
    assert bundle.readiness_by_document["doc-a"] == "partial"
    assert bundle.to_dict()["selected"][1]["kind"] == "figure"


def test_readiness_distinguishes_native_ocr_visual_and_unresolved_pages() -> None:
    parsed = ParsedDocument(
        text="native searchable text",
        parser_name="test",
        parser_version="1",
        page_count=4,
        pages=[
            ParsedPage(1, "This page has enough native searchable words for a deterministic readiness test."),
            ParsedPage(2, "OCR recovered enough searchable words for this page and its surrounding section context.", "ocr"),
            ParsedPage(3, ""),
            ParsedPage(4, "..."),
        ],
        tables=[
            ParsedTable(0, 1, "Table 1", "| A |", 1, 1, "test"),
        ],
        figures=[
            ParsedFigure(0, 3, "Figure 1", None, None, "test", metadata={"quality_status": "needs_review"}),
        ],
    )

    report = assess_document_readiness(parsed)
    states = [page.state for page in report.pages]
    assert states == ["native_text_ready", "ocr_text_ready", "visual_only", "unresolved"]
    assert report.state == "partial"
    assert report.table_count == 1
    assert report.figure_quality_counts == {"needs_review": 1}


def test_plain_result_defaults_to_typed_text_evidence() -> None:
    result = evidence_candidate_from_result({"chunk_id": "text:1", "content": "hello"})
    assert isinstance(result, TextEvidence)
    assert not isinstance(result, FigureEvidence)


def test_indexing_persists_readiness_snapshot(tmp_path) -> None:
    from app.db.sqlite import init_db
    from app.services.indexing_service import IndexingService

    db_path = tmp_path / "app.db"
    source = tmp_path / "notes.txt"
    source.write_text("A searchable document with enough content for readiness.", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))

    stored = IndexingService(db_path).get_document(document.id)
    assert stored is not None
    metadata = __import__("json").loads(stored["metadata_json"])
    assert metadata["document_readiness"]["state"] == "ready"
