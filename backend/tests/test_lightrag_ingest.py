from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json

import pytest

from app.db.sqlite import connect, init_db
from app.lightrag import ingest


@dataclass
class _Deletion:
    status: str


class _DocStatus:
    def __init__(self, statuses):
        self.statuses = statuses
        self.direct_deleted = []
        self.flushed = 0

    async def get_by_id(self, document_id):
        return self.statuses.get(document_id)

    async def get_docs_paginated(self, *, page, page_size):
        items = list(self.statuses.items())
        start = (page - 1) * page_size
        return items[start : start + page_size], len(items)

    async def delete(self, document_ids):
        self.direct_deleted.extend(document_ids)
        for document_id in document_ids:
            self.statuses.pop(document_id, None)

    async def index_done_callback(self):
        self.flushed += 1


class _TextChunks:
    def __init__(self, records=None):
        self.records = records or {}

    async def get_by_ids(self, chunk_ids):
        return [
            {"_id": chunk_id, **self.records[chunk_id]}
            if chunk_id in self.records
            else None
            for chunk_id in chunk_ids
        ]


class _FakeLightRAG:
    def __init__(self, statuses):
        self.doc_status = _DocStatus(statuses)
        records = {}
        for document_id, status in statuses.items():
            if status.get("status") != "processed":
                continue
            chunk_id = f"{document_id}-chunk-000"
            status.setdefault("chunks_list", [chunk_id])
            status.setdefault("chunks_count", 1)
            records[chunk_id] = {
                "content": "paper content",
                "full_doc_id": document_id,
                "file_path": status.get("file_path") or f"/papers/{document_id}.pdf",
                "chunk_order_index": 0,
            }
        self.text_chunks = _TextChunks(records)
        self.inserted = []
        self.deleted = []

    async def ainsert(self, text, *, ids, file_paths):
        self.inserted.append((ids, file_paths, text))
        chunk_id = f"{ids}-chunk-000"
        self.text_chunks.records[chunk_id] = {
            "content": text,
            "full_doc_id": ids,
            "file_path": file_paths,
            "chunk_order_index": 0,
        }
        self.doc_status.statuses[ids] = {
            "status": "processed",
            "file_path": file_paths,
            "chunks_list": [chunk_id],
            "chunks_count": 1,
        }
        return f"track:{ids}"

    async def adelete_by_doc_id(self, document_id):
        self.deleted.append(document_id)
        self.doc_status.statuses.pop(document_id, None)
        return _Deletion("success")

    def _build_global_config(self):
        async def llm_model_func(*_args, **_kwargs):
            raise AssertionError("test LLM must not be called")

        return {
            "summary_context_size": 1,
            "summary_max_tokens": 1,
            "force_llm_summary_on_merge": 1,
            "role_llm_funcs": {"extract": llm_model_func},
            "llm_model_func": llm_model_func,
        }


def _seed_document(db_path, document_id="doc-1") -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """INSERT OR IGNORE INTO indexed_folders
               (id, folder_path, recursive, file_types, created_at, updated_at)
               VALUES ('folder-1', '/papers', 0, '["pdf"]', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO documents
               (id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count)
               VALUES (?, 'folder-1', ?, 'paper.pdf', 'pdf', 'hash', 'now', 'now', 1)""",
            (document_id, f"/papers/{document_id}.pdf"),
        )
        connection.execute(
            """INSERT INTO chunks
               (id, document_id, chunk_index, content, source_path, filename,
                chunk_type, parent_chunk_id, created_at, metadata_json)
               VALUES (?, ?, 0, 'paper content', ?, 'paper.pdf', 'text', ?,
                       'now', ?)""",
            (
                f"chunk-{document_id}",
                document_id,
                f"/papers/{document_id}.pdf",
                f"parent-{document_id}",
                json.dumps({"parent_content": "paper content"}),
            ),
        )


def test_processed_document_is_idempotently_skipped(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG({"doc-1": {"status": "processed", "track_id": "old"}})
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_document(db_path, "doc-1"))

    assert result.skipped is True
    assert result.reason == "already_processed"
    assert rag.inserted == []
    with connect(db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM lightrag_chunk_parent_provenance"
            ).fetchone()[0]
            == 1
        )


def test_active_pipeline_document_is_never_deleted(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)

    for status in ("pending", "parsing", "analyzing", "preprocessed", "processing"):
        rag = _FakeLightRAG({"doc-1": {"status": status}})
        monkeypatch.setattr(ingest, "get_lightrag", lambda rag=rag: rag)

        result = asyncio.run(ingest.ingest_document(db_path, "doc-1"))

        assert result.skipped is True
        assert result.reason == f"already_{status}"
        assert rag.deleted == []
        assert rag.inserted == []


def test_failed_document_is_deleted_before_retry(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG({"doc-1": {"status": "failed"}})
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_document(db_path, "doc-1"))

    assert result.skipped is False
    assert rag.deleted == ["doc-1"]
    assert [item[0] for item in rag.inserted] == ["doc-1"]
    with connect(db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM lightrag_chunk_parent_provenance"
            ).fetchone()[0]
            == 1
        )


def test_insert_fails_when_lightrag_returns_track_id_but_status_is_failed(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)

    class StatusFailingLightRAG(_FakeLightRAG):
        async def ainsert(self, text, *, ids, file_paths):
            self.inserted.append((ids, file_paths, text))
            self.doc_status.statuses[ids] = {
                "status": "failed",
                "file_path": file_paths,
                "error_msg": "provider quota exhausted",
            }
            return f"track:{ids}"

    rag = StatusFailingLightRAG({})
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    with pytest.raises(RuntimeError, match="provider quota exhausted"):
        asyncio.run(ingest.ingest_document(db_path, "doc-1"))


def test_full_sync_clears_interrupted_canonical_queue_before_first_insert(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path, "doc-1")
    _seed_document(db_path, "doc-2")

    class QueueObservingLightRAG(_FakeLightRAG):
        async def ainsert(self, text, *, ids, file_paths):
            active = {
                document_id
                for document_id, value in self.doc_status.statuses.items()
                if value.get("status") != "processed"
            }
            assert active == set()
            return await super().ainsert(
                text,
                ids=ids,
                file_paths=file_paths,
            )

    rag = QueueObservingLightRAG(
        {
            "doc-1": {"status": "failed", "chunks_count": 2},
            "doc-2": {"status": "analyzing", "chunks_count": 3},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(
        ingest.ingest_all_documents(db_path, prune_stale=False)
    )

    assert result.inserted == 2
    assert result.failed == 0
    assert rag.deleted == ["doc-1", "doc-2"]
    assert [item[0] for item in rag.inserted] == ["doc-1", "doc-2"]


def test_full_sync_stops_after_first_runtime_failure(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path, "doc-1")
    _seed_document(db_path, "doc-2")

    class FirstFailureLightRAG(_FakeLightRAG):
        async def ainsert(self, text, *, ids, file_paths):
            self.inserted.append((ids, file_paths, text))
            self.doc_status.statuses[ids] = {
                "status": "failed",
                "file_path": file_paths,
                "error_msg": "provider unavailable",
            }
            return f"track:{ids}"

    rag = FirstFailureLightRAG({})
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_all_documents(db_path))

    assert result.inserted == 0
    assert result.failed == 1
    assert result.prune_skipped_reason == "batch_aborted_after_error"
    assert [item[0] for item in rag.inserted] == ["doc-1"]
    assert result.results[0]["document_id"] == "doc-1"
    assert "provider unavailable" in result.results[0]["error"]


def test_legacy_same_filename_record_is_deleted_before_canonical_insert(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG(
        {
            "legacy-id": {
                "status": "processed",
                "file_path": "/old/location/doc-1.pdf",
            },
            "dup-id": {
                "status": "failed",
                "file_path": "doc-1.pdf",
                "chunks_count": 0,
            },
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_document(db_path, "doc-1"))

    assert result.skipped is False
    assert rag.doc_status.direct_deleted == ["dup-id"]
    assert rag.doc_status.flushed == 1
    assert rag.deleted == ["legacy-id"]
    assert [item[0] for item in rag.inserted] == ["doc-1"]


def test_full_sync_prunes_statuses_missing_from_canonical_sqlite(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG(
        {
            "doc-1": {"status": "processed"},
            "stale-doc": {"status": "failed"},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_all_documents(db_path))

    assert result.skipped == 1
    assert result.deleted_stale == 1
    assert rag.deleted == ["stale-doc"]


def test_prune_stale_does_not_reingest_canonical_documents(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG(
        {
            "doc-1": {"status": "processed"},
            "stale-failed": {"status": "failed"},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.prune_stale_documents(db_path))

    assert result.deleted == 1
    assert result.failed == 0
    assert rag.deleted == ["stale-failed"]
    assert rag.inserted == []


def test_prune_stale_hard_disables_lightrag_rebuild_llm(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)

    class ConfigAwareLightRAG(_FakeLightRAG):
        def __init__(self, statuses):
            super().__init__(statuses)
            self.observed_delete_config = None
            self.llm_calls = 0

        def _build_global_config(self):
            async def external_llm(*_args, **_kwargs):
                self.llm_calls += 1
                return "unexpected"

            return {
                "summary_context_size": 1,
                "summary_max_tokens": 1,
                "force_llm_summary_on_merge": 1,
                "role_llm_funcs": {"extract": external_llm},
                "llm_model_func": external_llm,
            }

        async def adelete_by_doc_id(self, document_id):
            config = self._build_global_config()
            self.observed_delete_config = config
            # Mirrors the installed LightRAG summary decision: the regular
            # config above would call external_llm for a multi-description
            # rebuild, whereas the prune wrapper must force a direct join.
            if (
                2 >= config["force_llm_summary_on_merge"]
                or 2 >= config["summary_max_tokens"]
            ):
                await config["role_llm_funcs"]["extract"]("summarize")
            self.deleted.append(document_id)
            self.doc_status.statuses.pop(document_id, None)
            return _Deletion("success")

    rag = ConfigAwareLightRAG(
        {
            "doc-1": {"status": "processed"},
            "stale-processed": {"status": "processed"},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.prune_stale_documents(db_path))

    assert result.deleted == 1
    assert result.failed == 0
    assert rag.deleted == ["stale-processed"]
    assert rag.llm_calls == 0
    assert (
        rag.observed_delete_config["force_llm_summary_on_merge"]
        == ingest._NO_LLM_SUMMARY_LIMIT
    )
    assert (
        rag.observed_delete_config["role_llm_funcs"]["extract"]
        is ingest._reject_lightrag_llm_call
    )


def test_prune_stale_fails_closed_if_delete_still_attempts_llm(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)

    class UnexpectedLLMLightRAG(_FakeLightRAG):
        async def adelete_by_doc_id(self, document_id):
            config = self._build_global_config()
            await config["role_llm_funcs"]["extract"]("unexpected path")
            return _Deletion("success")

    rag = UnexpectedLLMLightRAG(
        {
            "doc-1": {"status": "processed"},
            "stale-processed": {"status": "processed"},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.prune_stale_documents(db_path))

    assert result.deleted == 0
    assert result.failed == 1
    assert rag.deleted == []
    assert "lightrag_no_llm_delete_guard:llm_call_blocked" in result.results[0]["error"]


def test_prune_stale_waits_for_every_canonical_document_to_be_processed(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)
    rag = _FakeLightRAG(
        {
            "doc-1": {"status": "processing"},
            "stale-processed": {"status": "processed"},
        }
    )
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.prune_stale_documents(db_path))

    assert result.deleted == 0
    assert result.skipped_reason == "canonical_documents_not_processed"
    assert result.unready_document_ids == ["doc-1"]
    assert rag.deleted == []


def test_failed_canonical_ingest_never_prunes_previous_graph(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(db_path)

    class FailingLightRAG(_FakeLightRAG):
        async def ainsert(self, text, *, ids, file_paths):
            raise RuntimeError("synthetic ingest failure")

    rag = FailingLightRAG({"stale-processed": {"status": "processed"}})
    monkeypatch.setattr(ingest, "get_lightrag", lambda: rag)

    result = asyncio.run(ingest.ingest_all_documents(db_path))

    assert result.failed == 1
    assert result.deleted_stale == 0
    assert result.prune_skipped_reason == "batch_aborted_after_error"
    assert result.unready_document_ids == ["doc-1"]
    assert rag.deleted == []


def test_status_scan_reads_every_paginated_document() -> None:
    statuses = {f"doc-{index}": {"status": "processed"} for index in range(205)}
    rag = _FakeLightRAG(statuses)

    document_ids = asyncio.run(ingest._lightrag_document_ids(rag))

    assert document_ids == set(statuses)
