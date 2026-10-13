from __future__ import annotations

from pathlib import Path
import json

import pytest

from app.rag.vision import FigureRetrievalContext, VisionSummaryError

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


def _seed_revision_context(db_path: Path, artifact_root: Path) -> None:
    """Deliberately retain conflicting legacy/retired/active/staged evidence."""
    _seed_figure_context(db_path, artifact_root)
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET active_revision_id = 'rev-active' WHERE id = 'doc-1'"
        )
        connection.execute(
            "UPDATE document_cards SET revision_id = 'rev-active', "
            "title_guess = 'Active title', short_summary = 'Active summary.'"
        )
        for revision in ("active", "staged", "retired"):
            revision_id = f"rev-{revision}"
            root = artifact_root / f"assets-{revision}"
            figure_path = root / "figures" / "figure.png"
            page_path = root / "pages" / "page_009.png"
            figure_path.parent.mkdir(parents=True)
            page_path.parent.mkdir(parents=True)
            figure_path.write_bytes(revision.encode())
            page_path.write_bytes(revision.encode())
            connection.execute(
                """INSERT INTO document_revisions
                   (revision_id, document_id, raw_content_hash, extracted_content_hash,
                    parser_fingerprint, chunker_fingerprint, artifact_fingerprint,
                    materialization_fingerprint, status, created_at)
                   VALUES (?, 'doc-1', ?, ?, 'parser', 'chunker', 'artifacts', ?, ?, 'now')""",
                (revision_id, revision, revision, revision_id, revision),
            )
            connection.execute(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, content, source_path, filename,
                    chunk_type, page_number, revision_id, metadata_json)
                   VALUES (?, 'doc-1', 0, ?, '/papers/CMDM.pdf', 'CMDM.pdf', 'text', 9, ?, ?)""",
                (
                    f"chunk-{revision}",
                    f"{revision} excerpt. Fig. 6 depicts the {revision} experiment.",
                    revision_id,
                    json.dumps({"section_title": f"{revision} section"}),
                ),
            )
            connection.execute(
                """INSERT INTO document_tables
                   (id, document_id, table_index, page_number, caption, markdown,
                    revision_id, metadata_json)
                   VALUES (?, 'doc-1', 0, 9, ?, ?, ?, '{}')""",
                (f"table-{revision}", f"{revision} table", f"{revision} cells", revision_id),
            )
            connection.execute(
                """INSERT INTO document_figures
                   (id, document_id, figure_index, page_number, caption, image_path,
                    extraction_method, revision_id, metadata_json)
                   VALUES (?, 'doc-1', 0, 9, 'Fig. 6: Experiment', ?,
                           'logical_composite', ?, '{"quality_status":"accepted"}')""",
                (f"figure-{revision}", str(figure_path), revision_id),
            )
            connection.execute(
                """INSERT INTO document_card_revisions
                   (id, document_id, revision_id, title_guess, short_summary,
                    created_at, updated_at)
                   VALUES (?, 'doc-1', ?, ?, ?, 'now', 'now')""",
                (f"card-{revision}", revision_id, f"{revision.title()} title", f"{revision.title()} summary."),
            )


def _revision_service(db_path: Path, artifact_root: Path, revision_id: str | None = None):
    return FigureEnrichService(
        db_path=db_path,
        artifact_root=artifact_root,
        ollama_host="http://localhost:11434",
        vision_model="cx/gpt-5.6-sol",
        revision_id=revision_id,
    )


@pytest.mark.parametrize("explicit_revision", [None, "rev-active", "rev-staged"])
def test_context_is_owned_by_selected_figure_revision(tmp_path, explicit_revision):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    service = _revision_service(db_path, root, explicit_revision)
    figure = service._list_figures(document_id="doc-1", limit=None)[0]
    owner = "staged" if explicit_revision == "rev-staged" else "active"
    context = service._document_context(figure)
    assert figure.get("revision_id") == f"rev-{owner}"
    assert context.title == f"{owner.title()} title"
    assert context.summary == f"{owner.title()} summary."
    assert context.section_title == f"{owner} section"
    assert context.nearby_text == f"{owner} excerpt. Fig. 6 depicts the {owner} experiment."
    assert context.nearby_tables == (f"{owner} table\n{owner} cells",)
    assert context.reference_sentences == (f"Fig. 6 depicts the {owner} experiment.",)


def test_context_does_not_fall_back_to_mismatched_document_card(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM document_card_revisions WHERE revision_id = 'rev-staged'")
    service = _revision_service(db_path, root, "rev-staged")
    figure = service._list_figures(document_id="doc-1", limit=None)[0]
    context = service._document_context(figure)
    assert context.title is None
    assert context.summary is None
    assert "staged excerpt" in context.nearby_text


@pytest.mark.parametrize("explicit_revision", [None, "rev-staged"])
def test_page_image_cannot_cross_revision_even_when_legacy_page_exists(tmp_path, explicit_revision):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    service = _revision_service(db_path, root, explicit_revision)
    figure = service._list_figures(document_id="doc-1", limit=None)[0]
    owner = "staged" if explicit_revision else "active"
    expected = root / f"assets-{owner}" / "pages" / "page_009.png"
    assert service._page_image_path(figure) == expected
    expected.unlink()
    assert service._page_image_path(figure) is None


def test_completion_uses_active_figures_not_legacy_null_rows(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    metadata = json.dumps({
        "quality_status": "accepted", "asset_kind": "figure", "is_content": True,
        "is_complete": True, "vision_provider": "9router", "vision_model": "cx/gpt-5.6-sol",
    })
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE document_figures SET visual_summary = 'complete', metadata_json = ? "
            "WHERE revision_id IS NULL OR revision_id = 'rev-retired'", (metadata,),
        )
    service = _revision_service(db_path, root)
    assert not service._document_vision_is_complete("doc-1")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE document_figures SET visual_summary = 'complete', metadata_json = ? "
            "WHERE revision_id = 'rev-active'", (metadata,),
        )
    assert service._document_vision_is_complete("doc-1")
    assert not _revision_service(db_path, root, "rev-staged")._document_vision_is_complete("doc-1")


def test_staged_enrichment_does_not_mark_active_document_complete(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    service = _revision_service(db_path, root, "rev-staged")
    service._mark_document_vision_fingerprint("doc-1")
    with connect(db_path) as connection:
        metadata = json.loads(connection.execute(
            "SELECT metadata_json FROM documents WHERE id = 'doc-1'"
        ).fetchone()["metadata_json"])
    assert "vision_model" not in metadata


def test_revision_switch_during_enrichment_does_not_mark_new_active_revision(tmp_path, monkeypatch):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    seen = []

    class FakeVision:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "cx/gpt-5.6-sol"

        def summarize_image_context(self, image_path, **kwargs):
            context = kwargs["document_context"]
            seen.append((image_path, context, kwargs["page_image_path"]))
            with connect(db_path) as connection:
                connection.execute(
                    "UPDATE documents SET active_revision_id = 'rev-staged' WHERE id = 'doc-1'"
                )
            return FigureRetrievalContext(
                raw_text="Observed experiment plot.", figure_type="plot", asset_kind="figure",
                is_content=True, is_complete=True,
                vision_provider="9router", vision_model="cx/gpt-5.6-sol",
            )

    monkeypatch.setattr("app.services.figure_enrich_service.OpenAICompatibleVisionSummarizer", FakeVision)
    result = _revision_service(db_path, root).enrich_document("doc-1")
    assert result["enriched"] == 1
    assert result["failed"] == 0
    assert len(seen) == 1
    assert seen[0][1].summary == "Active summary."
    assert "active excerpt" in seen[0][1].nearby_text
    assert seen[0][2] == root / "assets-active" / "pages" / "page_009.png"
    with connect(db_path) as connection:
        document = connection.execute("SELECT * FROM documents WHERE id = 'doc-1'").fetchone()
        active = connection.execute("SELECT * FROM document_figures WHERE id = 'figure-active'").fetchone()
        staged = connection.execute("SELECT * FROM document_figures WHERE id = 'figure-staged'").fetchone()
    assert "vision_model" not in json.loads(document["metadata_json"])
    assert active["visual_summary"] == "Observed experiment plot."
    assert json.loads(active["metadata_json"])["vision_revision_id"] == "rev-active"
    assert not staged["visual_summary"]


def test_explicit_revision_mismatch_is_rejected_before_context_lookup(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    figure = _revision_service(db_path, root)._list_figures(document_id="doc-1", limit=None)[0]
    with pytest.raises(VisionSummaryError, match="figure_revision_mismatch"):
        _revision_service(db_path, root, "rev-staged")._document_context(figure)


def test_legacy_result_without_revision_resolves_artifact_not_new_active_revision(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    service = _revision_service(db_path, root)
    figure = service._list_figures(document_id="doc-1", limit=None)[0]
    figure.pop("revision_id")
    with connect(db_path) as connection:
        connection.execute("UPDATE documents SET active_revision_id = 'rev-staged' WHERE id = 'doc-1'")
    assert service._page_text_excerpt(figure).startswith("active excerpt")
    assert service._section_title(figure) == "active section"


def test_unknown_document_or_revision_cannot_be_complete(tmp_path):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    assert not _revision_service(db_path, root)._document_vision_is_complete("absent")
    assert not _revision_service(db_path, root, "absent-revision")._document_vision_is_complete("doc-1")


def test_cached_description_tagged_for_another_revision_is_not_current(tmp_path, monkeypatch):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    calls = []
    stale_metadata = {
        "quality_status": "accepted", "asset_kind": "figure", "is_content": True,
        "is_complete": True, "vision_provider": "9router", "vision_model": "cx/gpt-5.6-sol",
        "vision_revision_id": "rev-retired",
    }
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE document_figures SET visual_summary = 'Old interpretation', metadata_json = ? "
            "WHERE id = 'figure-active'", (json.dumps(stale_metadata),),
        )

    class FakeVision:
        def __init__(self, **kwargs):
            pass

        def summarize_image_context(self, image_path, **kwargs):
            calls.append(kwargs["document_context"].summary)
            return FigureRetrievalContext(
                raw_text="Current interpretation", figure_type="plot", asset_kind="figure",
                is_content=True, is_complete=True,
                vision_provider="9router", vision_model="cx/gpt-5.6-sol",
            )

    monkeypatch.setattr("app.services.figure_enrich_service.OpenAICompatibleVisionSummarizer", FakeVision)
    service = _revision_service(db_path, root)
    assert not service._document_vision_is_complete("doc-1")
    result = service.enrich_document("doc-1")
    assert result["enriched"] == 1
    assert calls == ["Active summary."]
    assert service._document_vision_is_complete("doc-1")


def test_partial_batch_does_not_mark_unprocessed_active_figures_complete(tmp_path, monkeypatch):
    db_path, root = tmp_path / "app.db", tmp_path / "artifacts"
    _seed_revision_context(db_path, root)
    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO document_figures
               (id, document_id, figure_index, page_number, caption, image_path,
                extraction_method, revision_id, metadata_json)
               SELECT 'figure-active-2', document_id, 1, page_number, 'Fig. 7: Another plot',
                      image_path, extraction_method, revision_id, metadata_json
               FROM document_figures WHERE id = 'figure-active'"""
        )

    class FakeVision:
        def __init__(self, **kwargs):
            pass

        def summarize_image_context(self, image_path, **kwargs):
            return FigureRetrievalContext(
                raw_text="Current interpretation", figure_type="plot", asset_kind="figure",
                is_content=True, is_complete=True,
                vision_provider="9router", vision_model="cx/gpt-5.6-sol",
            )

    monkeypatch.setattr("app.services.figure_enrich_service.OpenAICompatibleVisionSummarizer", FakeVision)
    service = _revision_service(db_path, root)
    assert service.enrich_document("doc-1", limit=1)["enriched"] == 1
    assert not service._document_vision_is_complete("doc-1")
    with connect(db_path) as connection:
        metadata = json.loads(connection.execute(
            "SELECT metadata_json FROM documents WHERE id = 'doc-1'"
        ).fetchone()["metadata_json"])
    assert "vision_model" not in metadata
