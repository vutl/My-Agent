from __future__ import annotations

from pathlib import Path

import pytest

from app.db.sqlite import connect, init_db
from app.services.indexing_service import IndexingService


def test_rescan_unchanged_pdf_does_not_reparse(tmp_path, monkeypatch) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("stable content for a no-op rescan", encoding="utf-8")
    db_path = tmp_path / "app.db"
    init_db(db_path)
    service = IndexingService(db_path)
    first = service.index_file(source_path=str(source))

    import app.services.indexing_service as indexing_module

    original = indexing_module.parse_document

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unchanged raw bytes must not be parsed")

    monkeypatch.setattr(indexing_module, "parse_document", fail_if_called)
    second = service.index_file(source_path=str(source))
    assert second.id == first.id
    assert second.revision_id == first.revision_id
    monkeypatch.setattr(indexing_module, "parse_document", original)


def test_docx_parser_version_drift_marks_unchanged_source_for_reindex(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("A DOCX canonical paragraph")
    source = tmp_path / "report.docx"
    document.save(source)
    db_path = tmp_path / "app.db"
    init_db(db_path)
    service = IndexingService(db_path)
    indexed = service.index_file(source_path=str(source))

    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET metadata_json = ? WHERE id = ?",
            ('{"page_chunk_version":0,"artifact_extraction_version":0}', indexed.id),
        )
        assert service._needs_reindex_for_page_metadata(connection, indexed.id)


def test_exact_duplicate_paths_share_one_canonical_document_and_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("# Same\n\nThe same bytes should be indexed once.", encoding="utf-8")
    second_path.write_bytes(first_path.read_bytes())

    init_db(db_path)
    service = IndexingService(db_path)
    first = service.index_file(source_path=str(first_path))
    second = service.index_file(source_path=str(second_path))

    assert second.id == first.id
    with connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1
        aliases = connection.execute(
            "SELECT source_path FROM document_sources WHERE document_id = ? ORDER BY source_path",
            (first.id,),
        ).fetchall()
        assert [row["source_path"] for row in aliases] == sorted(
            [str(first_path), str(second_path)]
        )


@pytest.mark.parametrize("alias_first", [False, True])
def test_alias_policy_change_stages_new_revision_without_overwriting_active(
    tmp_path: Path, alias_first: bool,
) -> None:
    from dataclasses import replace

    db_path = tmp_path / "app.db"
    original = tmp_path / "old" / "document.md"
    alias = tmp_path / "new" / "document.md"
    original.parent.mkdir()
    alias.parent.mkdir()
    original.write_text("# Same source\n\nText bytes are unchanged.", encoding="utf-8")
    alias.write_bytes(original.read_bytes())
    init_db(db_path)
    service = IndexingService(db_path, canonical_artifact_fingerprint="image-bytes-v1")
    first = service.index_file(source_path=str(original))
    if alias_first:
        assert service.index_file(source_path=str(alias)).revision_id == first.revision_id
    replacement = replace(service, canonical_artifact_fingerprint="image-bytes-v2")
    staged = replacement.index_file(source_path=str(alias), defer_activation=True)

    assert staged.id == first.id
    assert staged.revision_id != first.revision_id
    with connect(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 2
        assert c.execute("SELECT active_revision_id FROM documents").fetchone()[0] == first.revision_id
        revisions = c.execute(
            "SELECT revision_id, status, metadata_json FROM document_revisions ORDER BY created_at"
        ).fetchall()
        assert revisions[0]["status"] == "active"
        assert "image-bytes-v1" in revisions[0]["metadata_json"]
        assert revisions[1]["status"] == "staged"
        assert "image-bytes-v2" in revisions[1]["metadata_json"]
        assert c.execute("SELECT COUNT(*) FROM chunks WHERE revision_id = ?",
                         (first.revision_id,)).fetchone()[0] > 0


@pytest.mark.parametrize("options", [
    {"enable_pdf_ocr": False},
    {"pdf_ocr_page_budget": 1},
    {"pdf_page_label_ocr_budget": 0},
])
def test_pdf_parse_policy_change_stages_distinct_revision_without_losing_active(
    tmp_path: Path, options: dict,
) -> None:
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "native.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((60, 60), "Stable native text for parse policy testing")
    pdf.save(source)
    pdf.close()
    db_path = tmp_path / "app.db"
    init_db(db_path)
    original = IndexingService(db_path)
    first = original.index_file(source_path=str(source))
    replacement = IndexingService(db_path, **options)
    staged = replacement.index_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["pdf"],
        defer_activation=True,
    ).documents[0]
    assert staged.id == first.id
    assert staged.revision_id != first.revision_id
    with connect(db_path) as connection:
        active = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (first.id,),
        ).fetchone()[0]
        assert active == first.revision_id
        import json

        revisions = {
            row["revision_id"]: json.loads(row["metadata_json"])["materialization_policy"]
            for row in connection.execute(
                "SELECT revision_id, metadata_json FROM document_revisions WHERE document_id = ?",
                (first.id,),
            )
        }
        assert revisions[first.revision_id] == original.materialization_policy("pdf")
        assert revisions[staged.revision_id] == replacement.materialization_policy("pdf")
    repeated = replacement.index_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["pdf"],
        defer_activation=True,
    ).documents[0]
    assert repeated.revision_id == staged.revision_id


def test_raw_change_creates_revision_and_retires_previous_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text("# Revision one\n\nold content", encoding="utf-8")

    init_db(db_path)
    service = IndexingService(db_path)
    indexed = service.index_file(source_path=str(source))
    with connect(db_path) as connection:
        old_revision = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (indexed.id,)
        ).fetchone()[0]

    source.write_bytes(b"# Revision one\n\nnew content\n")
    service.index_file(source_path=str(source))

    with connect(db_path) as connection:
        revisions = connection.execute(
            "SELECT revision_id, status FROM document_revisions WHERE document_id = ? ORDER BY created_at",
            (indexed.id,),
        ).fetchall()
        active = connection.execute(
            "SELECT active_revision_id, raw_content_hash FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()
    assert len(revisions) == 2
    assert old_revision != active["active_revision_id"]
    assert {row["status"] for row in revisions} == {"active", "retired"}


def test_staged_reindex_keeps_active_revision_until_explicit_activation(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text("# Stable\n\nfirst revision", encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)

    first = service.index_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["md"], defer_activation=True
    ).documents[0]
    assert service.activate_revision(first.id, first.revision_id)
    old_revision = first.revision_id

    source.write_text("# Stable\n\nsecond revision", encoding="utf-8")
    staged = service.index_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["md"], defer_activation=True
    ).documents[0]

    with connect(db_path) as connection:
        document = connection.execute(
            "SELECT active_revision_id, revision_status FROM documents WHERE id = ?",
            (first.id,),
        ).fetchone()
    assert staged.revision_id != old_revision
    assert document["active_revision_id"] == old_revision
    assert document["revision_status"] == "staged"


def test_unchanged_rescan_returns_unactivated_staged_revision_for_retry(tmp_path: Path) -> None:
    """A provider outage must not make an unchanged retry silently disappear."""

    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text("# Stable\n\nwaiting for local vectors", encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)

    staged = service.index_folder(
        folder_path=str(tmp_path),
        recursive=False,
        file_types=["md"],
        defer_activation=True,
    ).documents[0]
    assert staged.revision_id

    retry = service.index_folder(
        folder_path=str(tmp_path),
        recursive=False,
        file_types=["md"],
        defer_activation=True,
    ).documents[0]

    assert retry.id == staged.id
    assert retry.revision_id == staged.revision_id
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT active_revision_id, revision_status FROM documents WHERE id = ?",
            (staged.id,),
        ).fetchone()
    assert row["active_revision_id"] is None
    assert row["revision_status"] == "staged"


def test_schema_backfill_creates_revision_projection_for_legacy_document(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO indexed_folders(id, folder_path, recursive, file_types, created_at, updated_at)
            VALUES ('folder', '/tmp', 0, '[\"md\"]', 'now', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO documents(
                id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count, table_count, figure_count
            ) VALUES ('legacy-doc', 'folder', '/tmp/legacy.md', 'legacy.md', 'md',
                      'text-hash', 'now', 'now', 0, 0, 0)
            """
        )

    # init_db is the idempotent migration entrypoint.
    init_db(db_path)
    with connect(db_path) as connection:
        document = connection.execute(
            "SELECT active_revision_id, raw_content_hash FROM documents WHERE id = 'legacy-doc'"
        ).fetchone()
        revision = connection.execute(
            "SELECT status FROM document_revisions WHERE document_id = 'legacy-doc'"
        ).fetchone()
    assert document["active_revision_id"]
    assert document["raw_content_hash"] == "text-hash"
    assert revision["status"] == "active"


def test_folder_rename_rebinds_alias_and_missing_source_is_tombstoned(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    folder = tmp_path / "folder"
    folder.mkdir()
    old_path = folder / "old.md"
    old_path.write_text("# Stable\n\ncontent", encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)
    first = service.index_folder(folder_path=str(folder), recursive=False, file_types=["md"])
    document_id = first.documents[0].id

    new_path = folder / "renamed.md"
    old_path.rename(new_path)
    service.index_folder(folder_path=str(folder), recursive=False, file_types=["md"])

    with connect(db_path) as connection:
        aliases = connection.execute(
            "SELECT source_path, status FROM document_sources WHERE document_id = ? ORDER BY source_path",
            (document_id,),
        ).fetchall()
        document = connection.execute(
            "SELECT revision_status FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    assert {(row["source_path"], row["status"]) for row in aliases} == {
        (str(old_path), "deleted"),
        (str(new_path), "active"),
    }
    assert document["revision_status"] == "active"

    new_path.unlink()
    service.index_folder(folder_path=str(folder), recursive=False, file_types=["md"])
    with connect(db_path) as connection:
        document = connection.execute(
            "SELECT revision_status, index_status FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    assert document["revision_status"] == "retired"
    assert document["index_status"] == "retired"


def test_reparse_after_interrupted_stage_replaces_its_leftover_rows(tmp_path: Path, monkeypatch) -> None:
    """A process killed after writing a revision's chunks but before activating
    it leaves those rows behind; the retry derives the same revision id and must
    replace them instead of failing on the chunk primary key."""

    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text("# Stable\n\n" + "\n\n".join(f"paragraph {i} " * 40 for i in range(6)), encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)
    staged = service.index_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["md"], defer_activation=True
    ).documents[0]
    init_db(db_path)  # the backend restarts after the crash
    with connect(db_path) as connection:
        chunks_before = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE revision_id = ?", (staged.revision_id,)
        ).fetchone()[0]

    # Any reparse path (here a reindex predicate) re-derives the same revision.
    monkeypatch.setattr(IndexingService, "_needs_reindex_for_page_metadata", lambda *_args: True)
    retry = service.index_folder(folder_path=str(tmp_path), recursive=False, file_types=["md"])
    document = retry.documents[0]
    assert document.revision_id == staged.revision_id
    with connect(db_path) as connection:
        chunks_after = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE revision_id = ?", (staged.revision_id,)
        ).fetchone()[0]
        active = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (staged.id,)
        ).fetchone()[0]
    assert chunks_after == chunks_before > 0
    assert active == staged.revision_id
