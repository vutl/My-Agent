from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.db.sqlite import connect, init_db
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_processing_service import ProcessingJob
from app.services.indexing_service import IndexedDocument, IndexFolderResult, IndexingService


@pytest.mark.parametrize("new_source_alias", [False, True])
def test_authored_artifacts_are_staged_before_projection_and_failed_stage_keeps_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_source_alias: bool,
) -> None:
    pytest.importorskip("lancedb")
    from app.rag.embeddings import HashEmbeddingProvider
    import app.services.document_processing_service as module

    monkeypatch.setattr(module, "OllamaEmbeddingProvider", lambda **kwargs: HashEmbeddingProvider(dimensions=16))
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Note\n\nCanonical local evidence.", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(
        db_path=db_path, indexing=IndexingService(db_path),
        lancedb_path=tmp_path / "lance", lightrag_enabled=False,
    )
    first = coordinator.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])
    assert asyncio.run(coordinator.run_job_now(first)).status == "complete"
    with connect(db_path) as connection:
        document_id, old_revision = connection.execute(
            "SELECT id, active_revision_id FROM documents"
        ).fetchone()
        old_chunks = [tuple(row) for row in connection.execute(
            "SELECT id, content FROM chunks WHERE revision_id = ? ORDER BY chunk_index",
            (old_revision,),
        )]
    if new_source_alias:
        previous_docs = docs
        docs = tmp_path / "alias"
        docs.mkdir()
        (docs / "note.md").write_bytes((previous_docs / "note.md").read_bytes())

    def crash(document_id, revision_id):
        raise RuntimeError("authored_artifact_crash")

    coordinator.indexing = IndexingService(db_path, canonical_artifact_fingerprint="attachments-v1")
    coordinator.canonical_artifact_materializer = crash
    failed = coordinator.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])
    assert asyncio.run(coordinator.run_job_now(failed)).status == "partial"
    with connect(db_path) as connection:
        assert connection.execute("SELECT active_revision_id FROM documents").fetchone()[0] == old_revision
        assert [tuple(row) for row in connection.execute(
            "SELECT id, content FROM chunks WHERE revision_id = ? ORDER BY chunk_index",
            (old_revision,),
        )] == old_chunks

    def materialize(document_id, revision_id):
        with connect(db_path) as connection:
            assert connection.execute("SELECT active_revision_id FROM documents").fetchone()[0] == old_revision
            assert revision_id != old_revision
            connection.execute(
                """INSERT INTO document_tables
                   (id, document_id, revision_id, table_index, caption, markdown, extraction_method, created_at)
                   VALUES (?, ?, ?, 0, 'Authored table', '| Task | Score |\n|---|---|\n| Example | 42 |', 'trusted_adapter', '2026-10-05')""",
                (f"table-{revision_id}", document_id, revision_id),
            )

    coordinator.indexing = IndexingService(db_path, canonical_artifact_fingerprint="attachments-v2")
    coordinator.canonical_artifact_materializer = materialize
    completed = coordinator.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])
    job = asyncio.run(coordinator.run_job_now(completed))
    assert job.status == "complete"
    assert job.result["local_projection"][0]["table_chunks"] == 1
    with connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        new_revision = connection.execute("SELECT active_revision_id FROM documents").fetchone()[0]
        assert new_revision != old_revision
        assert connection.execute("SELECT table_count FROM documents").fetchone()[0] == 1
        assert connection.execute(
            "SELECT status FROM document_stage_receipts WHERE revision_id = ? AND stage = 'canonical_artifacts'",
            (new_revision,),
        ).fetchone()[0] == "complete"
        assert [tuple(row) for row in connection.execute(
            "SELECT id, content FROM chunks WHERE revision_id = ? ORDER BY chunk_index",
            (old_revision,),
        )] == old_chunks


def test_empty_folder_type_selection_covers_all_supported_formats(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))

    job_id = coordinator.enqueue_folder(
        folder_path=str(tmp_path), recursive=False, file_types=[]
    )

    job = coordinator.get_job(job_id)
    assert job is not None
    assert job.file_types == ["docx", "md", "pdf", "txt"]


def test_process_folder_api_request_defaults_to_all_supported_formats() -> None:
    from app.api.files import ProcessFolderRequest

    request = ProcessFolderRequest(folder_path="/tmp/corpus")
    assert request.file_types == ["docx", "md", "pdf", "txt"]


def test_processing_lanes_keep_folder_scan_ahead_of_optional_enrichment(
    tmp_path: Path,
) -> None:
    """A cold optional index must not consume the local-search worker lane."""

    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Ready\n", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))

    enrichment = coordinator._enqueue_enrichment_job(  # noqa: SLF001 - scheduler contract
        "placeholder-document", "placeholder-revision", "visual_index"
    )
    folder = coordinator.enqueue_folder(
        folder_path=str(docs), recursive=False, file_types=["md"]
    )

    claimed = coordinator._claim_next_job(lane="local")  # noqa: SLF001 - scheduler contract
    assert claimed == folder
    assert coordinator.get_job(enrichment).status == "pending"

    with pytest.raises(ValueError, match="lane must be all, local or enrichment"):
        coordinator._claim_next_job(lane="invalid")  # noqa: SLF001 - validation contract


def test_enrichment_lane_does_not_claim_a_pending_folder_scan(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Ready\n", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))

    folder = coordinator.enqueue_folder(
        folder_path=str(docs), recursive=False, file_types=["md"]
    )
    enrichment = coordinator._enqueue_enrichment_job(  # noqa: SLF001 - scheduler contract
        "placeholder-document", "placeholder-revision", "visual_index"
    )

    claimed = coordinator._claim_next_job(lane="enrichment")  # noqa: SLF001
    assert claimed == enrichment
    assert coordinator.get_job(folder).status == "pending"


def test_folder_processing_job_is_retryable_when_local_projection_is_unavailable(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Note\n\nLocal processing", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
        lancedb_path=None,
    )

    job_id = coordinator.enqueue_folder(
        folder_path=str(docs),
        recursive=False,
        file_types=["md"],
        metadata={"test": True},
    )
    assert coordinator.get_job(job_id).status == "pending"
    asyncio.run(coordinator.run_pending_once())

    job = coordinator.get_job(job_id)
    assert job is not None
    assert job.status == "pending"
    assert job.next_attempt_at is not None
    with connect(db_path) as connection:
        receipt = connection.execute(
            """
            SELECT status FROM document_stage_receipts
            WHERE stage = 'local_projection'
            """
        ).fetchone()
        document_id = connection.execute("SELECT id FROM documents").fetchone()[0]
    assert receipt["status"] == "failed"
    status = coordinator.document_status(document_id)
    assert status["local_search_state"] != "ready"
    assert status["visual_state"] == "pending"


def test_expired_processing_job_is_recovered(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))
    job_id = coordinator.enqueue_folder(
        folder_path=str(tmp_path), recursive=False, file_types=["md"]
    )
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE document_processing_jobs
            SET status = 'running', lease_expires_at = '2000-01-01T00:00:00+00:00'
            WHERE job_id = ?
            """,
            (job_id,),
        )
    assert coordinator.recover_expired_jobs() == 1
    assert coordinator.get_job(job_id).status == "pending"


def test_run_job_now_claims_requested_job_without_consuming_queue_sibling(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "one.md").write_text("one", encoding="utf-8")
    (second_dir / "two.md").write_text("two", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
        lancedb_path=None,
    )
    first = coordinator.enqueue_folder(
        folder_path=str(first_dir), recursive=False, file_types=["md"]
    )
    second = coordinator.enqueue_folder(
        folder_path=str(second_dir), recursive=False, file_types=["md"]
    )

    result = asyncio.run(coordinator.run_job_now(second))

    assert result is not None
    assert result.job_id == second
    assert result.status == "pending"  # local projection is retryable without LanceDB
    assert coordinator.get_job(first).status == "pending"
    assert result.attempt_count == 1


def test_document_status_does_not_report_stale_stage_as_current_ready(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "note.md"
    source.write_text("# Stable\n\nbytes", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO document_stage_receipts(
                revision_id, stage, fingerprint, status, started_at,
                finished_at, metadata_json
            ) VALUES (?, 'canonical', 'old-provider-fingerprint', 'complete', 'now', 'now', '{}')
            """,
            (indexed.revision_id,),
        )
        connection.execute(
            """
            INSERT INTO document_stage_receipts(
                revision_id, stage, fingerprint, status, started_at,
                finished_at, metadata_json
            ) VALUES (?, 'local_projection', 'old-provider-fingerprint', 'complete', 'now', 'now', '{}')
            """,
            (indexed.revision_id,),
        )

    status = coordinator.document_status(indexed.id)
    assert status is not None
    assert status["local_search_state"] == "pending"
    assert status["stale_stages"]["canonical"] >= 1
    assert status["stale_stages"]["local_projection"] == 1


def test_document_status_exposes_parser_readiness_separately(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "scan.md"
    source.write_text("# Scan\n\nplaceholder", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET metadata_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "document_readiness": {
                            "state": "partial",
                            "native_text_pages": 0,
                            "visual_only_pages": 1,
                            "warnings": ["1 page(s) have no text layer"],
                        }
                    }
                ),
                indexed.id,
            ),
        )
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))

    status = coordinator.document_status(indexed.id)

    assert status is not None
    assert status["readiness_state"] == "partial"
    assert status["readiness"]["visual_only_pages"] == 1
    # No local revision exists in this fixture, so local search is pending;
    # the assertion above is the important contract: readiness is not hidden.
    assert status["local_search_state"] == "pending"


def test_canonical_receipt_uses_revision_materialization_fingerprint(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "note.md"
    source.write_text("# Stable\n\nbytes", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(
        source_path=str(source),
        defer_activation=True,
    )
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
    )
    assert coordinator.indexing.activate_revision(indexed.id, indexed.revision_id)

    status = coordinator.document_status(indexed.id)

    assert status is not None
    assert status["stale_stages"].get("canonical", 0) == 0
    assert any(
        receipt["stage"] == "canonical" and receipt["status"] == "complete"
        for receipt in status["receipts"]
    )


def test_unchanged_folder_scan_reuses_idempotent_job(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "note.md"
    source.write_text("# Stable\n\nbytes", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))

    first = coordinator.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])
    source.touch()
    second = coordinator.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])

    assert first == second
    with connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_processing_jobs").fetchone()[0] == 1


def test_pdf_ocr_policy_change_does_not_reuse_completed_folder_job(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "source.pdf").write_bytes(b"job identity fixture")
    init_db(db_path)
    first = DocumentProcessingService(db_path=db_path, indexing=IndexingService(db_path))
    job_id = first.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["pdf"])
    first._finish_job(job_id, status="complete", result={})
    same = first.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["pdf"])
    assert same == job_id
    changed = DocumentProcessingService(
        db_path=db_path, indexing=IndexingService(db_path, enable_pdf_ocr=False),
    )
    assert changed.enqueue_folder(
        folder_path=str(docs), recursive=False, file_types=["pdf"],
    ) != job_id


def test_processing_contract_change_does_not_reuse_old_folder_job(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Stable\n\nbytes", encoding="utf-8")
    init_db(db_path)
    first = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
        stage_version="foundation-test-v1",
    )
    second = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
        stage_version="foundation-test-v2",
    )

    first_job = first.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])
    second_job = second.enqueue_folder(folder_path=str(docs), recursive=False, file_types=["md"])

    assert first_job != second_job
    with connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_processing_jobs").fetchone()[0] == 2


def test_partial_folder_enqueues_enrichment_only_for_local_ready_documents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ready.md").write_text("ready", encoding="utf-8")
    (docs / "failed.md").write_text("failed", encoding="utf-8")
    init_db(db_path)
    class FakeIndexing:
        materialization_policy = IndexingService(db_path).materialization_policy

        def index_folder(self, **_kwargs):
            return IndexFolderResult(
                folder_id="folder",
                folder_path=str(docs),
                scanned_files=2,
                indexed_files=2,
                skipped_files=0,
                failed_files=1,
                documents=[ready, failed],
            )

    coordinator = DocumentProcessingService(db_path=db_path, indexing=FakeIndexing())
    ready = IndexedDocument(
        id="doc-ready",
        file_id=None,
        source_path=str(docs / "ready.md"),
        filename="ready.md",
        file_type="md",
        chunk_count=1,
        revision_id="rev-ready",
    )
    failed = IndexedDocument(
        id="doc-failed",
        file_id=None,
        source_path=str(docs / "failed.md"),
        filename="failed.md",
        file_type="md",
        chunk_count=0,
        revision_id="rev-failed",
    )

    async def fake_projection(document_id: str, **_kwargs):
        if document_id == ready.id:
            return {"document_id": document_id, "ok": True}
        return {"document_id": document_id, "ok": False, "error": "parse_failed"}

    monkeypatch.setattr(coordinator, "_index_local_projection", fake_projection)
    approved: list[str] = []
    enqueued: list[str] = []
    monkeypatch.setattr(
        coordinator,
        "_set_document_enrichment_approval",
        lambda document_id: approved.append(document_id),
    )
    monkeypatch.setattr(
        coordinator,
        "_enqueue_enrichment",
        lambda document_id, revision_id: enqueued.append(f"{document_id}:{revision_id}"),
    )

    job_id = coordinator.enqueue_folder(
        folder_path=str(docs),
        recursive=False,
        file_types=["md"],
        metadata={"enrichment_approved": True},
    )
    result = asyncio.run(coordinator.run_job_now(job_id))

    assert result is not None and result.status == "partial"
    assert approved == [ready.id]
    assert enqueued == [f"{ready.id}:{ready.revision_id}"]


def test_folder_processing_persists_per_file_canonical_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("one", encoding="utf-8")
    init_db(db_path)

    document = IndexedDocument(
        id="doc-one",
        file_id=None,
        source_path=str(docs / "one.md"),
        filename="one.md",
        file_type="md",
        chunk_count=1,
        revision_id="rev-one",
    )

    class FakeIndexing:
        materialization_policy = IndexingService(db_path).materialization_policy

        def index_folder(self, *, progress_callback=None, **_kwargs):
            if progress_callback is not None:
                progress_callback(
                    {
                        "scanned_files": 1,
                        "indexed_files": 1,
                        "skipped_files": 0,
                        "failed_files": 0,
                        "current_path": str(docs / "one.md"),
                        "current_file_type": "md",
                        "document_id": document.id,
                        "revision_id": document.revision_id,
                    }
                )
            return IndexFolderResult(
                folder_id="folder",
                folder_path=str(docs),
                scanned_files=1,
                indexed_files=1,
                skipped_files=0,
                failed_files=0,
                documents=[document],
            )

    coordinator = DocumentProcessingService(db_path=db_path, indexing=FakeIndexing())
    monkeypatch.setattr(
        coordinator,
        "_index_local_projection",
        lambda *_args, **_kwargs: asyncio.sleep(0, result={"document_id": document.id, "ok": True}),
    )
    job_id = coordinator.enqueue_folder(
        folder_path=str(docs), recursive=False, file_types=["md"]
    )
    result = asyncio.run(coordinator.run_job_now(job_id))
    assert result is not None and result.status == "complete"
    assert result.progress["stage"] == "complete"
    assert result.progress["scanned_files"] == 1
    assert result.progress["current_path"] is None


def test_partial_visual_enrichment_is_not_reported_as_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
    )
    job = ProcessingJob(
        job_id="visual-job",
        kind="enrichment:visual_index",
        folder_path="doc-1",
        recursive=False,
        file_types=[],
        status="running",
        attempt_count=1,
    )
    finished: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        coordinator,
        "_job_metadata",
        lambda _job_id: {"document_id": "doc-1", "revision_id": "rev-1"},
    )
    monkeypatch.setattr(coordinator, "_active_revision", lambda _document_id: "rev-1")
    async def fake_visual_index(_document_id: str, _revision_id: str, **_kwargs) -> dict:
        return {
            "status": "partial",
            "errors": [{"page_id": "p1", "error": "visual worker unavailable"}],
        }

    monkeypatch.setattr(coordinator, "_build_visual_index", fake_visual_index)
    monkeypatch.setattr(coordinator, "_mark_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        coordinator,
        "_reschedule_job",
        lambda *_args, **_kwargs: finished.append(("rescheduled", {})),
    )
    monkeypatch.setattr(
        coordinator,
        "_finish_job",
        lambda _job_id, *, status, result, **_kwargs: finished.append((status, result)),
    )

    asyncio.run(coordinator._run_enrichment_job(job))

    assert finished and finished[0][0] == "rescheduled"


def test_visual_progress_persists_only_for_current_job_lease(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.md").write_text("progress", encoding="utf-8")
    init_db(db_path)
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=IndexingService(db_path),
        lease_seconds=30,
    )
    job_id = coordinator.enqueue_folder(
        folder_path=str(folder), recursive=False, file_types=["md"]
    )
    assert coordinator._claim_job(job_id) is True  # noqa: SLF001
    coordinator._update_job_progress(  # noqa: SLF001
        job_id,
        {
            "stage": "visual_index",
            "requested_pages": 8,
            "processed_pages": 2,
            "remaining_pages": 6,
        },
    )
    job = coordinator.get_job(job_id)
    assert job is not None
    assert job.progress["processed_pages"] == 2
    assert job.progress["remaining_pages"] == 6
