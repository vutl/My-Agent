from __future__ import annotations

from pathlib import Path
import json

from app.db.sqlite import connect, init_db
from app.services.figure_enrich_service import (
    FigureEnrichService,
    _is_rate_limit_error,
    _is_unanchored_review,
)


def _seed_figure_context(db_path: Path, artifact_root: Path) -> None:
    init_db(db_path)
    figure_path = artifact_root / "doc-1" / "figures" / "figure.png"
    page_path = artifact_root / "doc-1" / "pages" / "page_009.png"
    figure_path.parent.mkdir(parents=True)
    page_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"figure")
    page_path.write_bytes(b"page")

    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO indexed_folders
               (id, folder_path, recursive, file_types, created_at, updated_at)
               VALUES ('folder-1', '/papers', 0, '["pdf"]', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO documents
               (id, folder_id, source_path, filename, file_type, title, content_hash,
                modified_at, indexed_at, chunk_count, table_count, figure_count,
                parse_status, index_status, updated_at, metadata_json)
               VALUES ('doc-1', 'folder-1', '/papers/CMDM.pdf', 'CMDM.pdf', 'pdf',
                       'CMDM', 'hash', 'now', 'now', 1, 1, 1,
                       'parsed', 'indexed', 'now', '{}')"""
        )
        connection.execute(
            """INSERT INTO document_cards
               (id, document_id, title_guess, short_summary, importance_score,
                confidence, should_deep_index, created_at, updated_at, metadata_json)
               VALUES ('card-1', 'doc-1', 'Cross-Modal Distribution Matching',
                       'PDF Download. Total Citations: 53. The paper studies multimodal emotion recognition.',
                       0.8, 0.8, 1, 'now', 'now', '{}')"""
        )
        connection.execute(
            """INSERT INTO chunks
               (id, document_id, chunk_index, content, source_path, filename,
                chunk_type, order_index, page_number, created_at, metadata_json)
               VALUES ('chunk-1', 'doc-1', 0,
                       'Experiments. As shown in Fig. 6, the confusion matrices compare four settings.',
                       '/papers/CMDM.pdf', 'CMDM.pdf', 'text', 0, 9, 'now',
                       '{"section_title":"Experiments"}')"""
        )
        connection.execute(
            """INSERT INTO document_tables
               (id, document_id, table_index, page_number, caption, markdown,
                extraction_method, created_at, metadata_json)
               VALUES ('table-1', 'doc-1', 0, 9, 'Table 2: Results',
                       '| Model | Acc |\n|---|---|\n| CMDM | 0.75 |',
                       'docling_table', 'now', '{}')"""
        )
        connection.execute(
            """INSERT INTO document_figures
               (id, document_id, figure_index, page_number, caption, image_path,
                extraction_method, bbox_json, created_at, metadata_json)
               VALUES ('figure-1', 'doc-1', 0, 9, 'Figure 6: Confusion matrices', ?,
                       'logical_composite', '{"x0":60,"y0":493,"x1":280,"y1":697}',
                       'now', '{"quality_status":"accepted","asset_kind":"figure"}')""",
            (str(figure_path),),
        )


def test_enrich_context_contains_document_page_section_and_nearby_table(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    artifact_root = tmp_path / "artifacts"
    _seed_figure_context(db_path, artifact_root)
    service = FigureEnrichService(
        db_path=db_path,
        artifact_root=artifact_root,
        ollama_host="http://localhost:11434",
        vision_model="qwen3-vl:4b",
    )

    figure = service._list_figures(document_id="doc-1", limit=None)[0]
    context = service._document_context(figure)

    assert context.title == "Cross-Modal Distribution Matching"
    assert context.section_title == "Experiments"
    assert context.page_number == 9
    assert context.reference_sentences == (
        "As shown in Fig. 6, the confusion matrices compare four settings.",
    )
    assert "Table 2: Results" in context.nearby_tables[0]
    assert "PDF Download" not in (context.summary or "")
    assert service._page_image_path(figure) == artifact_root / "doc-1" / "pages" / "page_009.png"


def test_bulk_enrich_skips_only_unanchored_review_assets() -> None:
    assert _is_unanchored_review(
        metadata={"quality_status": "needs_review"},
        caption="Visual asset extracted from page 2",
    )
    assert not _is_unanchored_review(
        metadata={"quality_status": "needs_review", "figure_number": 6},
        caption="Figure 6: Confusion matrices",
    )
    assert not _is_unanchored_review(
        metadata={"quality_status": "accepted"},
        caption="Figure 3: Architecture",
    )


def test_document_fingerprint_requires_current_vision_provenance(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    artifact_root = tmp_path / "artifacts"
    _seed_figure_context(db_path, artifact_root)
    service = FigureEnrichService(
        db_path=db_path,
        artifact_root=artifact_root,
        ollama_host="http://localhost:11434",
        vision_provider="openai_compatible",
        vision_model="cx/gpt-5.5",
    )

    assert not service._document_vision_is_complete("doc-1")
    with connect(db_path) as connection:
        metadata = {
            "quality_status": "accepted",
            "asset_kind": "figure",
            "is_content": True,
            "is_complete": True,
            "vision_provider": "9router",
            "vision_model": "cx/gpt-5.5",
        }
        connection.execute(
            "UPDATE document_figures SET visual_summary = ?, metadata_json = ? WHERE id = 'figure-1'",
            ("figure_type: architecture", json.dumps(metadata)),
        )

    assert service._document_vision_is_complete("doc-1")


def test_vision_rate_limit_is_detected_without_treating_other_errors_as_quota() -> None:
    assert _is_rate_limit_error(
        "9router vision request failed (429): usage limit reached; reset after 30m"
    )
    assert not _is_rate_limit_error("9router vision request failed (500): upstream error")
