from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import unicodedata

from app.db.sqlite import connect
from app.lightrag.client import get_lightrag
from app.lightrag.provenance import sync_document_provenance


_NO_LLM_SUMMARY_LIMIT = (1 << 63) - 1


async def _reject_lightrag_llm_call(*_args: Any, **_kwargs: Any) -> str:
    """Fail closed if a library deletion path unexpectedly reaches an LLM."""
    raise RuntimeError("lightrag_no_llm_delete_guard:llm_call_blocked")


class _NoLLMDeleteProxy:
    """Delegate LightRAG deletion while hard-disabling its rebuild LLM roles.

    LightRAG's ``adelete_by_doc_id`` rebuilds shared graph nodes from cached
    extraction rows.  Despite that wording, its merge step can call the
    extraction LLM again to summarize multiple descriptions.  Calling the
    method on this proxy preserves the library's storage-consistency logic but
    gives only that invocation a no-LLM global config; the live LightRAG
    instance is never mutated for concurrent queries.

    Entity/relation vector rebuilds may still use the configured embedding
    function.  The remote/chat LLM roles are both avoided by the high merge
    thresholds and blocked fail-closed in case a future LightRAG version adds
    another path.
    """

    def __init__(self, rag: Any) -> None:
        object.__setattr__(self, "_rag", rag)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rag, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._rag, name, value)

    def _build_global_config(self) -> dict[str, Any]:
        builder = getattr(self._rag, "_build_global_config", None)
        if not callable(builder):
            raise RuntimeError(
                "lightrag_no_llm_delete_unsupported:missing_global_config_builder"
            )

        config = dict(builder())
        # In the installed LightRAG version these three conditions force the
        # cached descriptions to be joined directly instead of summarized.
        config["summary_context_size"] = _NO_LLM_SUMMARY_LIMIT
        config["summary_max_tokens"] = _NO_LLM_SUMMARY_LIMIT
        config["force_llm_summary_on_merge"] = _NO_LLM_SUMMARY_LIMIT

        role_llm_funcs = dict(config.get("role_llm_funcs") or {})
        for role in set(role_llm_funcs) | {"extract", "query"}:
            role_llm_funcs[role] = _reject_lightrag_llm_call
        config["role_llm_funcs"] = role_llm_funcs
        # Older LightRAG releases read the legacy function directly.
        config["llm_model_func"] = _reject_lightrag_llm_call
        return config


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    track_id: str
    source_path: str
    char_count: int
    skipped: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class IngestAllResult:
    total: int
    inserted: int
    skipped: int
    failed: int
    deleted_stale: int
    prune_skipped_reason: str | None
    unready_document_ids: list[str]
    results: list[dict]


@dataclass(frozen=True)
class PruneStaleResult:
    deleted: int
    failed: int
    skipped_reason: str | None
    unready_document_ids: list[str]
    results: list[dict]


def _document_text(db_path: Path, document_id: str) -> tuple[str, str] | None:
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT source_path, filename
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        chunks = connection.execute(
            """
            SELECT content, parent_chunk_id, metadata_json
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        ).fetchall()
    if not chunks:
        return None
    passages: list[str] = []
    seen_parent_ids: set[str] = set()
    for chunk in chunks:
        parent_id = str(chunk["parent_chunk_id"] or "").strip()
        if parent_id:
            if parent_id in seen_parent_ids:
                continue
            metadata = json.loads(chunk["metadata_json"] or "{}")
            parent_content = str(metadata.get("parent_content") or "").strip()
            if parent_content:
                passages.append(parent_content)
                seen_parent_ids.add(parent_id)
                continue
        content = str(chunk["content"] or "").strip()
        if content:
            passages.append(content)
    text = "\n\n".join(passages)
    if not text.strip():
        return None
    return text, str(row["source_path"])


def list_document_ids(db_path: Path, *, limit: int | None = None) -> list[str]:
    query = "SELECT id FROM documents ORDER BY indexed_at ASC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with connect(db_path) as connection:
        rows = connection.execute(query).fetchall()
    return [str(row["id"]) for row in rows]


async def ingest_document(db_path: Path, document_id: str) -> IngestResult:
    loaded = _document_text(db_path, document_id)
    if loaded is None:
        raise ValueError(f"document_not_found_or_empty:{document_id}")
    text, source_path = loaded
    rag = get_lightrag()
    existing = await rag.doc_status.get_by_id(document_id)
    existing_status = _status_value(existing)
    if existing_status == "processed":
        await sync_document_provenance(db_path, rag, document_id)
        return IngestResult(
            document_id=document_id,
            track_id=str(_field(existing, "track_id") or "existing"),
            source_path=source_path,
            char_count=len(text),
            skipped=True,
            reason="already_processed",
        )
    if existing_status in {
        "pending",
        "parsing",
        "analyzing",
        "preprocessed",
        "processing",
    }:
        return IngestResult(
            document_id=document_id,
            track_id=str(_field(existing, "track_id") or "in_progress"),
            source_path=source_path,
            char_count=len(text),
            skipped=True,
            reason=f"already_{existing_status}",
        )
    if existing:
        # Retrying a failed document must not unexpectedly spend chat-model
        # quota merely to clean its partial graph rows.
        deletion = await _delete_by_doc_id_without_llm(rag, document_id)
        deletion_status = _field(deletion, "status")
        if str(deletion_status) not in {"success", "not_found"}:
            raise RuntimeError(
                f"lightrag_existing_document_cleanup_failed:{document_id}:{deletion_status}"
            )
    await _remove_conflicting_filename_records(
        rag,
        document_id=document_id,
        source_path=source_path,
    )
    track_id = await rag.ainsert(
        text,
        ids=document_id,
        file_paths=source_path,
    )
    # LightRAG records per-document extraction failures in doc_status but its
    # SDK ainsert entry point still returns the track ID. Treat anything other
    # than PROCESSED as a failed ingest so callers never report a false
    # "inserted" result or continue a costly corpus batch after provider/quota
    # failure.
    final_status_record = await rag.doc_status.get_by_id(document_id)
    final_status = _status_value(final_status_record)
    if final_status != "processed":
        error = str(_field(final_status_record, "error_msg") or "unknown_error")
        raise RuntimeError(
            "lightrag_ingest_not_processed:"
            f"{document_id}:{final_status or 'missing'}:{error}"
        )
    await sync_document_provenance(db_path, rag, document_id)
    return IngestResult(
        document_id=document_id,
        track_id=track_id,
        source_path=source_path,
        char_count=len(text),
    )


async def ingest_all_documents(
    db_path: Path,
    *,
    limit: int | None = None,
    prune_stale: bool | None = None,
) -> IngestAllResult:
    document_ids = list_document_ids(db_path, limit=limit)
    rag = get_lightrag()
    await _require_pipeline_idle(rag)
    # LightRAG's SDK pipeline resumes every FAILED/PROCESSING document in the
    # workspace whenever a single new document is inserted. A process crash or
    # provider outage can therefore turn the next one-document retry into an
    # unbounded concurrent retry of the entire backlog. Remove only incomplete
    # canonical rows up front; processed canonical graph data remains intact,
    # and each subsequent ainsert has exactly one canonical document eligible
    # for processing.
    await _reset_incomplete_canonical_queue(rag, set(document_ids))

    results: list[dict] = []
    inserted = 0
    skipped = 0
    failed = 0
    batch_aborted = False
    deleted_stale = 0
    prune_skipped_reason: str | None = None
    unready_document_ids: list[str] = []
    for document_id in document_ids:
        try:
            result = await ingest_document(db_path, document_id)
            if result.skipped:
                skipped += 1
            else:
                inserted += 1
            results.append(
                {
                    "document_id": result.document_id,
                    "track_id": result.track_id,
                    "source_path": result.source_path,
                    "char_count": result.char_count,
                    "skipped": result.skipped,
                    "reason": result.reason,
                    "ok": True,
                }
            )
        except ValueError as exc:
            skipped += 1
            results.append({"document_id": document_id, "ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — collect per-doc failures for batch ingest
            failed += 1
            results.append({"document_id": document_id, "ok": False, "error": str(exc)})
            # Fail closed on the first runtime/provider failure. The remaining
            # documents stay absent from the queue and can be resumed later
            # with the exact same configured model.
            batch_aborted = True
            break

    should_prune = limit is None if prune_stale is None else prune_stale
    if batch_aborted:
        prune_skipped_reason = "batch_aborted_after_error"
        unready_document_ids = await _unready_document_ids(rag, set(document_ids))
    elif should_prune:
        prune_result = await prune_stale_documents(db_path)
        deleted_stale = prune_result.deleted
        failed += prune_result.failed
        prune_skipped_reason = prune_result.skipped_reason
        unready_document_ids = prune_result.unready_document_ids
        results.extend(prune_result.results)
    return IngestAllResult(
        total=len(document_ids),
        inserted=inserted,
        skipped=skipped,
        failed=failed,
        deleted_stale=deleted_stale,
        prune_skipped_reason=prune_skipped_reason,
        unready_document_ids=unready_document_ids,
        results=results,
    )


async def _require_pipeline_idle(rag: Any) -> None:
    """Refuse queue cleanup while another LightRAG pipeline owns the workspace."""
    workspace = getattr(rag, "workspace", None)
    if workspace is None:
        return
    from lightrag.kg.shared_storage import get_namespace_data

    pipeline_status = await get_namespace_data(
        "pipeline_status",
        workspace=workspace,
    )
    if pipeline_status.get("busy", False):
        raise RuntimeError(
            "lightrag_batch_busy:"
            f"{pipeline_status.get('job_name') or 'unknown_job'}"
        )


async def _reset_incomplete_canonical_queue(
    rag: Any,
    canonical_ids: set[str],
) -> None:
    """Remove interrupted canonical work so the upstream resume queue is empty."""
    for document_id in sorted(canonical_ids):
        existing = await rag.doc_status.get_by_id(document_id)
        status = _status_value(existing)
        if status is None or status == "processed":
            continue
        deletion = await _delete_by_doc_id_without_llm(rag, document_id)
        deletion_status = str(_field(deletion, "status") or "")
        if deletion_status not in {"success", "not_found"}:
            raise RuntimeError(
                "lightrag_canonical_queue_cleanup_failed:"
                f"{document_id}:{deletion_status}"
            )


async def prune_stale_documents(db_path: Path) -> PruneStaleResult:
    """Delete graph/status rows absent from canonical SQLite without re-ingesting."""
    canonical_ids = set(list_document_ids(db_path))
    rag = get_lightrag()
    if not canonical_ids:
        return PruneStaleResult(
            deleted=0,
            failed=0,
            skipped_reason="canonical_index_empty",
            unready_document_ids=[],
            results=[
                {
                    "ok": False,
                    "deleted": False,
                    "reason": "prune_skipped:canonical_index_empty",
                }
            ],
        )
    unready_document_ids = await _unready_document_ids(rag, canonical_ids)
    if unready_document_ids:
        return PruneStaleResult(
            deleted=0,
            failed=0,
            skipped_reason="canonical_documents_not_processed",
            unready_document_ids=unready_document_ids,
            results=[
                {
                    "ok": False,
                    "deleted": False,
                    "reason": "prune_skipped:canonical_documents_not_processed",
                    "unready_document_ids": unready_document_ids,
                }
            ],
        )
    deleted = 0
    failed = 0
    results: list[dict] = []
    for stale_id in sorted(await _lightrag_document_ids(rag) - canonical_ids):
        try:
            deletion = await _delete_by_doc_id_without_llm(rag, stale_id)
            status = str(_field(deletion, "status") or "")
            if status in {"success", "not_found"}:
                deleted += 1
                results.append(
                    {
                        "document_id": stale_id,
                        "ok": True,
                        "deleted": True,
                        "reason": "not_in_canonical_sqlite",
                    }
                )
            else:
                failed += 1
                results.append(
                    {
                        "document_id": stale_id,
                        "ok": False,
                        "error": f"lightrag_stale_delete_failed:{status}",
                    }
                )
        except Exception as exc:  # noqa: BLE001 — retain per-doc sync diagnostics
            failed += 1
            results.append(
                {
                    "document_id": stale_id,
                    "ok": False,
                    "error": f"lightrag_stale_delete_failed:{exc}",
                }
            )
    return PruneStaleResult(
        deleted=deleted,
        failed=failed,
        skipped_reason=None,
        unready_document_ids=[],
        results=results,
    )


async def _delete_by_doc_id_without_llm(rag: Any, document_id: str) -> Any:
    """Run LightRAG's graph-aware delete with a per-call no-LLM config."""
    delete_impl = getattr(type(rag), "adelete_by_doc_id", None)
    if not callable(delete_impl):
        raise RuntimeError(
            "lightrag_no_llm_delete_unsupported:missing_adelete_by_doc_id"
        )
    proxy = _NoLLMDeleteProxy(rag)
    return await delete_impl(proxy, document_id)


async def _unready_document_ids(rag: Any, document_ids: set[str]) -> list[str]:
    unready: list[str] = []
    for document_id in sorted(document_ids):
        if _status_value(await rag.doc_status.get_by_id(document_id)) != "processed":
            unready.append(document_id)
    return unready


async def _lightrag_document_ids(rag: Any) -> set[str]:
    if not hasattr(rag.doc_status, "get_docs_paginated"):
        return set()
    page = 1
    # LightRAG clamps this API to at most 200 rows. Keeping our own page size
    # aligned avoids skipping pages in a 201-500 document corpus.
    page_size = 200
    document_ids: set[str] = set()
    while True:
        rows, total = await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
        )
        if isinstance(rows, dict):
            document_ids.update(str(document_id) for document_id in rows)
        else:
            document_ids.update(str(document_id) for document_id, _status in rows)
        if page * page_size >= int(total or 0) or not rows:
            break
        page += 1
    return document_ids


async def _remove_conflicting_filename_records(
    rag: Any,
    *,
    document_id: str,
    source_path: str,
) -> None:
    """Remove legacy status/graph rows that block a canonical-ID migration.

    LightRAG deduplicates by basename across every status, including old FAILED
    duplicate-attempt rows. Therefore retrying with a new canonical ID can be
    reported as "inserted" while the background worker silently rejects it.
    Remove only records with the same normalized basename, preserving unrelated
    graph rows. The graph-aware no-LLM delete also safely handles the small
    number of old PROCESSED IDs before their canonical replacement is queued.
    """

    target = _normalized_basename(source_path)
    if not target or target == "unknown_source":
        return
    conflicts: dict[str, Any] = {}
    page = 1
    page_size = 200
    while True:
        rows, total = await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
        )
        items = rows.items() if isinstance(rows, dict) else rows
        for candidate_id, status in items:
            candidate_id = str(candidate_id)
            if candidate_id == document_id:
                continue
            if _normalized_basename(str(_field(status, "file_path") or "")) == target:
                conflicts[candidate_id] = status
        if page * page_size >= int(total or 0) or not rows:
            break
        page += 1

    direct_status_deletes = sorted(
        conflict_id
        for conflict_id, status in conflicts.items()
        if _status_value(status) == "failed"
        and int(_field(status, "chunks_count") or 0) == 0
    )
    if direct_status_deletes:
        # Duplicate-attempt rows have no full_docs/chunks/graph ownership.
        # Deleting them through graph-aware adelete rewrites the entire graph
        # once per row, so remove and flush these status-only tombstones in one
        # operation.
        await rag.doc_status.delete(direct_status_deletes)
        callback = getattr(rag.doc_status, "index_done_callback", None)
        if callable(callback):
            await callback()

    graph_conflicts = sorted(set(conflicts) - set(direct_status_deletes))
    for conflict_id in graph_conflicts:
        deletion = await _delete_by_doc_id_without_llm(rag, conflict_id)
        status = str(_field(deletion, "status") or "")
        if status not in {"success", "not_found"}:
            raise RuntimeError(
                "lightrag_conflicting_filename_cleanup_failed:"
                f"{conflict_id}:{status}"
            )


def _normalized_basename(value: str) -> str:
    name = Path(str(value or "")).name.strip()
    return unicodedata.normalize("NFC", name).casefold()


def _status_value(value: Any) -> str | None:
    status = _field(value, "status")
    if status is None:
        return None
    raw = getattr(status, "value", status)
    return str(raw).lower()


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
