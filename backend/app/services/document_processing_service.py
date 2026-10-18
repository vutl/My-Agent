"""Durable, local-first folder processing coordinator.

The coordinator deliberately keeps provider-backed enrichment out of the
local-search critical path. A folder job parses/indexes canonical data and
then attempts the local LanceDB projection; VLM/cards/LightRAG remain separate
policy-gated jobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import socket
import uuid
from typing import Any, Callable

from app.db.sqlite import connect
from app.rag.embeddings import OllamaEmbeddingProvider
from app.rag.parsers import SUPPORTED_TEXT_TYPES
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.indexing_service import IndexFolderResult, IndexingService
from app.services.vector_index_service import VectorIndexService


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _lease_expiry(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _projection_counts_match(result: dict[str, Any]) -> bool:
    counts = result.get("projection_counts")
    if counts is None:
        return True
    try:
        return all(
            int(counts.get(key, -1)) == int(result.get(key, -1))
            for key in ("document_cards", "text_chunks", "table_chunks", "figure_chunks")
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _folder_snapshot_fingerprint(root: Path, recursive: bool, file_types: list[str]) -> str:
    """Hash the discoverable folder snapshot without touching SQLite/indexes."""

    allowed = set(file_types)
    paths = root.rglob("*") if recursive else root.iterdir()
    entries: list[tuple[str, int, str]] = []
    for path in sorted((item for item in paths if item.is_file()), key=lambda item: str(item)):
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in allowed:
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            stat = path.stat()
        except OSError:
            # A file changing or disappearing during discovery invalidates this
            # snapshot deterministically; the worker will rescan later.
            entries.append((str(path.resolve()), -1, "unreadable"))
            continue
        # mtime is intentionally excluded: touching an unchanged file must
        # reuse the same durable job and must not manufacture a revision.
        entries.append((str(path.resolve()), int(stat.st_size), digest.hexdigest()))
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProcessingJob:
    job_id: str
    kind: str
    folder_path: str
    recursive: bool
    file_types: list[str]
    status: str
    attempt_count: int
    idempotency_key: str | None = None
    snapshot_fingerprint: str | None = None
    next_attempt_at: str | None = None
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    provider: str | None = None
    model: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error_class: str | None = None
    error_message: str | None = None
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentProcessingService:
    db_path: Path
    indexing: IndexingService
    lancedb_path: Path | None = None
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "embeddinggemma:300m"
    embedding_timeout_seconds: float = 120.0
    embedding_batch_size: int = 16
    embedding_transient_retries: int = 2
    embedding_query_prefix: str = "task: search result | query: "
    embedding_document_prefix: str = "title: none | text: "
    vision_provider: str = "openai_compatible"
    vision_model: str = "cx/gpt-6.1-sol"
    visual_retrieval_enabled: bool = False
    visual_retrieval_provider: str = "disabled"
    visual_retrieval_index_root: Path | None = None
    visual_retrieval_model: str = "vidore/colSmol-256M"
    visual_retrieval_model_revision: str = "c79b633e17e060cc109b11aa2bf92a1517d3bd6f"
    visual_retrieval_base_model: str = "vidore/ColSmolVLM-Instruct-256M-base"
    visual_retrieval_base_revision: str = "99ca96f1f6b95b3a69e6abef74a2416cb738fed0"
    visual_retrieval_device: str = "auto"
    visual_retrieval_local_files_only: bool = False
    visual_retrieval_render_scale: float = 1.25
    visual_retrieval_render_max_pixels: int = 2_000_000
    visual_retrieval_batch_max_pixels: int = 4_000_000
    visual_retrieval_worker_python: Path | None = None
    openai_api_base: str = "http://localhost:20128/v1"
    openai_api_key: str = "any"
    request_timeout_seconds: float = 120.0
    paper_evidence_card_build_enabled: bool = False
    paper_evidence_card_model: str = "cx/gpt-6.1-sol"
    paper_evidence_card_schema_version: str = "v1"
    paper_evidence_card_prompt_version: str = "v2"
    paper_evidence_card_max_concurrency: int = 2
    lightrag_enabled: bool = True
    lease_seconds: int = 300
    worker_id: str = ""
    max_attempts: int = 3
    # Bump this contract whenever parser/chunker/projection semantics change;
    # it participates in folder-job idempotency and stage receipts, allowing
    # an unchanged byte snapshot to be safely rebuilt once under the new
    # implementation instead of reusing an obsolete job.
    stage_version: str = "foundation-v5-table-rows"
    canonical_artifact_materializer: Callable[[str, str], None] | None = field(
        default=None, repr=False
    )
    _visual_provider_instance: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Any) -> "DocumentProcessingService":
        indexing = IndexingService(
            db_path=settings.sqlite_db_path,
            artifact_root=settings.artifacts_path,
            ollama_host=settings.ollama_host,
            vision_model=settings.vision_model,
            vision_provider=settings.vision_provider,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            request_timeout_seconds=settings.request_timeout_seconds,
            pdf_ocr_page_budget=settings.pdf_ocr_page_budget,
            pdf_page_label_ocr_budget=settings.pdf_page_label_ocr_budget,
            paper_evidence_card_build_enabled=settings.paper_evidence_card_build_enabled,
            paper_evidence_card_model=settings.paper_evidence_card_model,
            paper_evidence_card_schema_version=settings.paper_evidence_card_schema_version,
            paper_evidence_card_prompt_version=settings.paper_evidence_card_prompt_version,
        )
        return cls(
            db_path=settings.sqlite_db_path,
            indexing=indexing,
            lancedb_path=settings.lancedb_path,
            ollama_host=settings.ollama_host,
            embedding_model=settings.embedding_model,
            embedding_timeout_seconds=settings.request_timeout_seconds,
            embedding_batch_size=settings.embedding_batch_size,
            embedding_transient_retries=settings.embedding_transient_retries,
            embedding_query_prefix=settings.embedding_query_prefix,
            embedding_document_prefix=settings.embedding_document_prefix,
            vision_provider=settings.vision_provider,
            vision_model=settings.vision_model,
            visual_retrieval_enabled=settings.visual_retrieval_enabled,
            visual_retrieval_provider=settings.visual_retrieval_provider,
            visual_retrieval_index_root=settings.visual_retrieval_path,
            visual_retrieval_model=settings.visual_retrieval_model,
            visual_retrieval_model_revision=settings.visual_retrieval_model_revision,
            visual_retrieval_base_model=settings.visual_retrieval_base_model,
            visual_retrieval_base_revision=settings.visual_retrieval_base_revision,
            visual_retrieval_device=settings.visual_retrieval_device,
            visual_retrieval_local_files_only=settings.visual_retrieval_local_files_only,
            visual_retrieval_render_scale=settings.visual_retrieval_render_scale,
            visual_retrieval_render_max_pixels=settings.visual_retrieval_render_max_pixels,
            visual_retrieval_batch_max_pixels=settings.visual_retrieval_batch_max_pixels,
            visual_retrieval_worker_python=settings.visual_retrieval_worker_python,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            request_timeout_seconds=settings.request_timeout_seconds,
            paper_evidence_card_build_enabled=settings.paper_evidence_card_build_enabled,
            paper_evidence_card_model=settings.paper_evidence_card_model,
            paper_evidence_card_schema_version=settings.paper_evidence_card_schema_version,
            paper_evidence_card_prompt_version=settings.paper_evidence_card_prompt_version,
            paper_evidence_card_max_concurrency=settings.paper_evidence_card_max_concurrency,
            lightrag_enabled=settings.lightrag_enabled,
            stage_version=getattr(settings, "stage_version", "foundation-v5-table-rows"),
        )

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = f"{socket.gethostname()}:{uuid.uuid4().hex[:12]}"

    def enqueue_folder(
        self,
        *,
        folder_path: str,
        recursive: bool,
        file_types: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")
        normalized_types = sorted({item.lower().lstrip(".") for item in file_types if item.strip()})
        if not normalized_types:
            normalized_types = sorted(SUPPORTED_TEXT_TYPES)
        snapshot = _folder_snapshot_fingerprint(root, recursive, normalized_types)
        idempotency_key = hashlib.sha256(
            json.dumps(
                {
                    "kind": "folder_scan",
                    "folder": str(root),
                    "recursive": bool(recursive),
                    "file_types": normalized_types,
                    "snapshot": snapshot,
                    # A stable source snapshot alone is not enough to decide
                    # that an old job is reusable: parser/chunker/projection
                    # semantics can change while file bytes stay identical.
                    # Version the coordinator contract so a rescan naturally
                    # creates one new staged revision after an implementation
                    # upgrade, while unchanged rescans within the same
                    # contract remain idempotent.
                    "processor_contract": self.stage_version,
                    "materialization_policy": {
                        file_type: self.indexing.materialization_policy(file_type)
                        for file_type in normalized_types
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        now = _now()
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT job_id, status FROM document_processing_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["status"] in {"failed", "partial"}:
                    connection.execute(
                        """
                        UPDATE document_processing_jobs
                        SET status = 'pending', attempt_count = 0,
                            next_attempt_at = NULL, error_class = NULL,
                            error_message = NULL, finished_at = NULL
                        WHERE job_id = ?
                        """,
                        (existing["job_id"],),
                    )
                return str(existing["job_id"])
            job_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO document_processing_jobs(
                    job_id, kind, folder_path, recursive, file_types_json,
                    idempotency_key, snapshot_fingerprint, status, requested_at,
                    provider, model, progress_json, metadata_json
                ) VALUES (?, 'folder_scan', ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(root),
                    int(recursive),
                    json.dumps(normalized_types),
                    idempotency_key,
                    snapshot,
                    now,
                    "local",
                    self.embedding_model,
                    json.dumps({"stage": "queued", "files": 0}),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return job_id

    def get_job(self, job_id: str) -> ProcessingJob | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM document_processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(self, *, limit: int = 50) -> list[ProcessingJob]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_processing_jobs
                ORDER BY requested_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def document_status(self, document_id: str) -> dict[str, Any] | None:
        """Expose capability-level readiness without implying enrichment."""
        with connect(self.db_path) as connection:
            document = connection.execute(
                """
                SELECT id, filename, source_path, file_type, active_revision_id,
                       revision_status, parse_status, index_status, metadata_json
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if document is None:
                return None
            try:
                document_metadata = json.loads(document["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                document_metadata = {}
            if not isinstance(document_metadata, dict):
                document_metadata = {}
            readiness = document_metadata.get("document_readiness")
            if not isinstance(readiness, dict):
                readiness = {}
            revision_id = document["active_revision_id"]
            receipts = []
            if revision_id:
                receipts = connection.execute(
                    """
                    SELECT stage, status, error_class, error_message, metadata_json,
                           fingerprint, attempt_count, lease_owner, lease_expires_at
                    FROM document_stage_receipts
                    WHERE revision_id = ?
                    ORDER BY stage
                    """,
                    (revision_id,),
                ).fetchall()
        by_stage: dict[str, dict[str, Any]] = {}
        stale_stages: dict[str, list[dict[str, Any]]] = {}
        for row in receipts:
            stage_name = str(row["stage"])
            candidate = dict(row)
            # A changed provider/model creates a new receipt fingerprint. The
            # current configuration must win over an older complete receipt;
            # otherwise status could report stale enrichment as ready.
            if revision_id and stage_name in {
                "canonical", "local_projection", "figure_enrichment",
                "evidence_cards", "lightrag", "visual_index",
            }:
                expected = self._stage_fingerprint(revision_id, stage_name)
                if str(candidate.get("fingerprint") or "") == expected:
                    by_stage[stage_name] = candidate
                    continue
                # A receipt for an older provider/model/config is historical,
                # not evidence that the current stage is complete. Keep it in
                # the response for auditability, but make the current stage
                # pending until a receipt with the active fingerprint exists.
                stale_stages.setdefault(stage_name, []).append(candidate)
                continue
            by_stage.setdefault(stage_name, candidate)

        def stage_state(name: str) -> str:
            row = by_stage.get(name)
            return str(row["status"]) if row else "pending"

        if revision_id and stage_state("canonical") == "complete" and stage_state("local_projection") == "complete":
            local_state = "ready"
        elif str(document["revision_status"] or "") == "staged":
            local_state = "staged"
        elif str(document["revision_status"] or "") == "retired":
            local_state = "failed"
        else:
            local_state = "pending"
        enrichment_names = ("figure_enrichment", "evidence_cards", "lightrag")
        enrichment_values = [stage_state(name) for name in enrichment_names]
        if all(value == "complete" for value in enrichment_values):
            enrichment_state = "ready"
        elif all(value == "skipped" for value in enrichment_values):
            enrichment_state = "not_requested"
        else:
            enrichment_state = "pending"
        return {
            "document_id": document_id,
            "filename": document["filename"],
            "source_path": document["source_path"],
            "file_type": document["file_type"],
            "active_revision_id": revision_id,
            # Local projection readiness and parser/readability readiness are
            # intentionally separate capabilities.  A scan can have valid
            # page/figure artifacts and a healthy Lance projection while its
            # text layer remains partial or visual-only; exposing the parser
            # report here prevents a generic ``ready`` label from hiding that
            # limitation in operational status clients.
            "readiness_state": str(readiness.get("state") or "unknown"),
            "readiness": readiness,
            "local_search_state": local_state,
            "visual_state": stage_state("visual_index"),
            "figure_state": stage_state("figure_enrichment"),
            "card_state": stage_state("evidence_cards"),
            "graph_state": stage_state("lightrag"),
            "enrichment_state": enrichment_state,
            "partial_reasons": [
                row["error_message"] or row["error_class"]
                for row in receipts
                if row["status"] in {"partial", "failed"} and (row["error_message"] or row["error_class"])
            ],
            "stale_stages": {
                stage: len(values) for stage, values in sorted(stale_stages.items())
            },
            "receipts": [dict(row) for row in receipts],
        }

    async def run_pending_once(self, *, lane: str = "all") -> bool:
        """Run one durable job from a bounded processing lane.

        Folder scans and canonical local projections are latency-critical.
        Provider-backed work (visual/VLM/cards/graph) is not.  Keeping their
        claims separate ensures a slow optional model cannot starve a newly
        submitted folder scan, while SQLite leases still prevent duplicates.
        """

        if lane not in {"all", "local", "enrichment"}:
            raise ValueError("lane must be all, local or enrichment")
        self.recover_expired_jobs()
        job_id = self._claim_next_job(lane=lane)
        if job_id is None:
            return False
        await self.run_job(job_id)
        return True

    async def run_job_now(self, job_id: str) -> ProcessingJob | None:
        """Claim and execute one durable job without relying on queue order.

        Folder APIs normally use ``run_pending_once`` in the background
        worker. Isolated evaluators and operational probes sometimes need to
        run the exact job they just enqueued; claiming by id avoids accidentally
        consuming an unrelated pending job while preserving the same lease,
        heartbeat, retry and activation semantics.
        """

        self.recover_expired_jobs()
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status in {"complete", "failed", "partial"}:
            return job
        if job.status != "pending" or not self._claim_job(job_id):
            return self.get_job(job_id)
        await self.run_job(job_id)
        return self.get_job(job_id)

    async def run_job(self, job_id: str) -> ProcessingJob | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(job_id))
        try:
            if job.kind.startswith("enrichment:"):
                await self._run_enrichment_job(job)
                return self.get_job(job_id)
            result: IndexFolderResult = await asyncio.to_thread(
                self.indexing.index_folder,
                folder_path=job.folder_path,
                recursive=job.recursive,
                file_types=job.file_types,
                defer_activation=True,
                progress_callback=lambda progress: self._update_job_progress(
                    job_id,
                    {
                        "stage": "canonical",
                        "status": "running",
                        **progress,
                    },
                ),
            )
            self._update_job_progress(
                job_id,
                {
                    "stage": "canonical",
                    "status": "complete" if result.failed_files == 0 else "partial",
                    "scanned_files": result.scanned_files,
                    "indexed_files": result.indexed_files,
                    "skipped_files": result.skipped_files,
                    "failed_files": result.failed_files,
                    "errors": result.errors,
                    "current_path": None,
                    "current_file_type": None,
                },
            )
            projection_results: list[dict[str, Any]] = []
            projection_errors = 0
            for document in result.documents:
                try:
                    if self.canonical_artifact_materializer is not None:
                        if not self.indexing.canonical_artifact_fingerprint:
                            raise RuntimeError("canonical_artifact_fingerprint_missing")
                        self._mark_stage(document.revision_id, "canonical_artifacts", "pending")
                        try:
                            await asyncio.to_thread(
                                self.canonical_artifact_materializer,
                                document.id,
                                document.revision_id,
                            )
                        except Exception as exc:
                            self._mark_stage(
                                document.revision_id, "canonical_artifacts", "failed",
                                error_class=type(exc).__name__,
                            )
                            raise
                        self._mark_stage(document.revision_id, "canonical_artifacts", "complete")
                    projection_results.append(
                        await self._index_local_projection(
                            document.id,
                            revision_id=document.revision_id,
                            # Activation is deliberately inside the validated
                            # projection path.  A staged revision must never
                            # become visible merely because parsing succeeded.
                            activate_revision=True,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - persisted as a retryable job error
                    projection_errors += 1
                    projection_results.append(
                        {
                            "document_id": document.id,
                            "ok": False,
                            "error_class": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            status = "complete" if result.failed_files == 0 and projection_errors == 0 else "partial"
            payload = {
                "folder_id": result.folder_id,
                "folder_path": result.folder_path,
                "scanned_files": result.scanned_files,
                "indexed_files": result.indexed_files,
                "skipped_files": result.skipped_files,
                "failed_files": result.failed_files,
                "errors": result.errors,
                "local_projection": projection_results,
            }
            # A folder job can be partial because one source failed while
            # another document reached a validated local-ready revision. Do
            # not hold optional enrichment hostage to the failed sibling:
            # enqueue only the documents whose canonical/local projection
            # actually succeeded. Idempotency keys keep retries harmless.
            ready_document_ids = {
                str(item.get("document_id"))
                for item in projection_results
                if item.get("ok") and item.get("document_id")
            }
            folder_metadata = self._job_metadata(job_id)
            if folder_metadata.get("enrichment_approved"):
                for document in result.documents:
                    if document.id in ready_document_ids:
                        self._set_document_enrichment_approval(document.id)
            for document in result.documents:
                if document.id in ready_document_ids:
                    self._enqueue_enrichment(document.id, document.revision_id)
            if status == "complete":
                self._finish_job(job_id, status=status, result=payload)
            elif self._is_retryable_result(payload) and job.attempt_count < self.max_attempts:
                self._reschedule_job(job_id, payload, "projection_retryable")
            else:
                self._finish_job(job_id, status=status, result=payload)
        except Exception as exc:  # noqa: BLE001 - fail closed and leave durable evidence
            if self._retryable_error(exc) and job.attempt_count < self.max_attempts:
                self._reschedule_job(job_id, {}, type(exc).__name__, str(exc))
            else:
                self._finish_job(
                    job_id,
                    status="failed",
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                    result={},
                )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        return self.get_job(job_id)

    async def close(self) -> None:
        """Stop optional child providers owned by this coordinator.

        The visual worker is lazy and normally disabled, but when enabled it
        is a real child process. Closing it with the application lifecycle
        prevents an orphaned model process after a restart or deploy.
        """

        provider = self._visual_provider_instance
        self._visual_provider_instance = None
        if provider is None:
            return
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
            return
        close_sync = getattr(provider, "close", None)
        if close_sync is not None:
            await asyncio.to_thread(close_sync)

    def recover_expired_jobs(self) -> int:
        now = _now()
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE document_processing_jobs
                SET status = 'pending', lease_owner = NULL, lease_expires_at = NULL,
                    next_attempt_at = ?, error_class = 'worker_recovered'
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (now, now),
            )
        return int(cursor.rowcount)

    async def worker_loop(
        self,
        stop_event: asyncio.Event,
        *,
        idle_seconds: float = 2.0,
        lane: str = "all",
    ) -> None:
        while not stop_event.is_set():
            ran = await self.run_pending_once(lane=lane)
            if not ran:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=idle_seconds)
                except asyncio.TimeoutError:
                    pass

    async def _index_local_projection(
        self,
        document_id: str,
        *,
        revision_id: str | None = None,
        activate_revision: bool = False,
    ) -> dict[str, Any]:
        revision = revision_id or self._active_revision(document_id)
        if revision is None:
            raise RuntimeError(f"active_revision_missing:{document_id}")
        self._mark_stage(revision, "local_projection", "pending")
        if self.lancedb_path is None:
            self._mark_stage(revision, "local_projection", "failed", error_class="provider_not_configured")
            raise RuntimeError("local_projection_provider_not_configured")
        try:
            store = LanceDBRetrievalStore(self.lancedb_path)
            embeddings = OllamaEmbeddingProvider(
                host=self.ollama_host,
                model=self.embedding_model,
                timeout_seconds=self.embedding_timeout_seconds,
                batch_size=self.embedding_batch_size,
                transient_retries=self.embedding_transient_retries,
                query_prefix=self.embedding_query_prefix,
                document_prefix=self.embedding_document_prefix,
            )
            vector_index = VectorIndexService(self.db_path, store, embeddings)
            result = await vector_index.index_document(
                document_id,
                revision_id=revision,
                activate_revision=False,
            )
            if not result.get("ok"):
                raise RuntimeError(str(result.get("error") or "vector_projection_failed"))
            # Validate against rows actually persisted in Lance, not just the
            # number of records assembled before the write. Legacy tables may
            # lack a physical revision column; in that migration window the
            # store returns None and the SQLite/FTS checks below remain the
            # conservative compatibility validation.
            physical_counts = getattr(store, "count_document_revision", None)
            if callable(physical_counts):
                counts = physical_counts(document_id, revision)
                if counts is not None:
                    result["projection_counts"] = counts
            self._mark_stage(revision, "local_projection", "complete")
            if not self._validate_local_projection(document_id, revision, result):
                raise RuntimeError(f"revision_activation_failed:{document_id}:{revision}")
            if callable(getattr(store, "document_revision_ids", None)) and not vector_index.projection_identity_matches(document_id, revision):
                raise RuntimeError(f"revision_projection_identity_mismatch:{document_id}:{revision}")
            if activate_revision and not self.indexing.activate_revision(document_id, revision):
                raise RuntimeError(f"revision_activation_failed:{document_id}:{revision}")
            return result
        except LanceDBUnavailable:
            self._mark_stage(revision, "local_projection", "failed", error_class="lancedb_unavailable")
            raise
        except Exception as exc:
            self._mark_stage(
                revision,
                "local_projection",
                "failed",
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            raise

    async def _run_enrichment_job(self, job: ProcessingJob) -> None:
        """Run one optional derived projection without blocking local search."""

        stage = job.kind.split(":", 1)[1]
        metadata = self._job_metadata(job.job_id)
        document_id = str(metadata.get("document_id") or job.folder_path).strip()
        revision_id = str(metadata.get("revision_id") or "").strip() or None
        if not document_id or not revision_id:
            self._finish_job(
                job.job_id,
                status="failed",
                error_class="invalid_enrichment_job",
                error_message="document_id_and_revision_id_required",
                result={},
            )
            return
        if self._active_revision(document_id) != revision_id:
            self._mark_stage(
                revision_id,
                stage,
                "failed",
                error_class="stale_revision",
                error_message="enrichment_revision_is_not_active",
            )
            self._finish_job(
                job.job_id,
                status="failed",
                error_class="stale_revision",
                error_message="enrichment_revision_is_not_active",
                result={"document_id": document_id, "revision_id": revision_id},
            )
            return

        try:
            if stage == "figure_enrichment":
                from app.services.figure_enrich_service import FigureEnrichService

                result = await asyncio.to_thread(
                    FigureEnrichService(
                        db_path=self.db_path,
                        ollama_host=self.ollama_host,
                        vision_model=self.vision_model,
                        vision_provider=self.vision_provider,
                        openai_api_base=self.openai_api_base,
                        openai_api_key=self.openai_api_key,
                        request_timeout_seconds=self.request_timeout_seconds,
                        revision_id=revision_id,
                    ).enrich_document,
                    document_id,
                )
                status = "partial" if int(result.get("failed", 0)) else "complete"
            elif stage == "evidence_cards":
                if not self.paper_evidence_card_build_enabled:
                    self._mark_stage(
                        revision_id,
                        stage,
                        "skipped",
                        error_class="privacy_gate_not_approved",
                    )
                    self._finish_job(
                        job.job_id,
                        status="complete",
                        result={"document_id": document_id, "skipped": True},
                    )
                    return
                from app.llm.openai_client import get_llm_client
                from app.services.paper_evidence_builder import PaperEvidenceBuilder
                from app.services.paper_evidence_service import PaperEvidenceService

                client = get_llm_client(
                    provider="openai_compatible",
                    ollama_host=self.ollama_host,
                    openai_api_base=self.openai_api_base,
                    openai_api_key=self.openai_api_key,
                    timeout_seconds=self.request_timeout_seconds,
                )
                builder = PaperEvidenceBuilder(
                    service=PaperEvidenceService(
                        self.db_path,
                        schema_version=self.paper_evidence_card_schema_version,
                        prompt_version=self.paper_evidence_card_prompt_version,
                    ),
                    client=client,
                    model=self.paper_evidence_card_model,
                    max_concurrency=self.paper_evidence_card_max_concurrency,
                )
                result = await builder.build_document(document_id)
                status = "complete"
            elif stage == "lightrag":
                if not self.lightrag_enabled:
                    self._mark_stage(
                        revision_id,
                        stage,
                        "skipped",
                        error_class="provider_not_configured",
                    )
                    self._finish_job(
                        job.job_id,
                        status="complete",
                        result={"document_id": document_id, "skipped": True},
                    )
                    return
                from app.lightrag.ingest import ingest_document

                ingest_result = await ingest_document(self.db_path, document_id)
                result = {
                    "document_id": ingest_result.document_id,
                    "track_id": ingest_result.track_id,
                    "skipped": ingest_result.skipped,
                }
                status = "complete"
            elif stage == "visual_index":
                result = await self._build_visual_index(
                    document_id,
                    revision_id,
                    job_id=job.job_id,
                )
                status = "complete" if result.get("status") == "complete" else "partial"
            else:
                raise RuntimeError(f"unsupported_enrichment_stage:{stage}")
            self._mark_stage(
                revision_id,
                stage,
                status,
                error_class=(
                    "visual_index_partial"
                    if stage == "visual_index" and status == "partial"
                    else None
                ),
                error_message=(
                    str(result.get("reason") or "visual page indexing incomplete")
                    if stage == "visual_index" and status == "partial"
                    else None
                ),
                metadata={
                    "provider": self._stage_provider(stage),
                    "model": self._stage_model(stage),
                    **(
                        {"model_fingerprint": result.get("model_fingerprint")}
                        if stage == "visual_index" and result.get("model_fingerprint")
                        else {}
                    ),
                },
            )
            # A derived stage may complete only a prefix of its requested
            # work (for example, one visual page batch failed while others
            # were indexed). Keep that distinction in the durable job instead
            # of reporting success and losing retry/monitoring information.
            if status == "partial":
                errors = result.get("errors") or []
                retryable = any(
                    self._retryable_error(RuntimeError(str(item.get("error") or item)))
                    for item in errors
                    if isinstance(item, dict)
                )
                if retryable and job.attempt_count < self.max_attempts:
                    self._reschedule_job(
                        job.job_id,
                        result,
                        "derived_stage_partial",
                        str(result.get("reason") or "derived stage returned partial results"),
                    )
                else:
                    self._finish_job(
                        job.job_id,
                        status="partial",
                        result=result,
                        error_class="derived_stage_partial",
                        error_message=str(
                            result.get("reason") or "derived stage returned partial results"
                        ),
                    )
            else:
                self._finish_job(job.job_id, status="complete", result=result)
        except Exception as exc:  # provider failures remain durable/retryable
            self._mark_stage(
                revision_id,
                stage,
                "failed",
                error_class=type(exc).__name__,
                error_message=str(exc),
                metadata={
                    "provider": self._stage_provider(stage),
                    "model": self._stage_model(stage),
                },
            )
            if self._retryable_error(exc) and job.attempt_count < self.max_attempts:
                self._reschedule_job(job.job_id, {}, type(exc).__name__, str(exc))
            else:
                self._finish_job(
                    job.job_id,
                    status="failed",
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                    result={},
                )

    def _job_metadata(self, job_id: str) -> dict[str, Any]:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM document_processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        try:
            value = json.loads(row["metadata_json"] or "{}") if row else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
        return value if isinstance(value, dict) else {}
    def _active_revision(self, document_id: str) -> str | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT active_revision_id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return str(row["active_revision_id"]) if row and row["active_revision_id"] else None

    def _mark_stage(
        self,
        revision_id: str,
        stage: str,
        status: str,
        *,
        error_class: str | None = None,
        error_message: str | None = None,
        fingerprint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO document_stage_receipts(
                    revision_id, stage, fingerprint, status, started_at,
                    finished_at, attempt_count, error_class, error_message,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(revision_id, stage, fingerprint) DO UPDATE SET
                    status = excluded.status,
                    finished_at = excluded.finished_at,
                    attempt_count = document_stage_receipts.attempt_count + 1,
                    error_class = excluded.error_class,
                    error_message = excluded.error_message,
                    metadata_json = excluded.metadata_json
                """,
                (
                    revision_id,
                    stage,
                    fingerprint or self._stage_fingerprint(revision_id, stage),
                    status,
                    now,
                    now if status in {"complete", "partial", "failed", "skipped"} else None,
                    error_class,
                    error_message,
                    json.dumps(metadata or {"provider": "local"}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def _stage_fingerprint(self, revision_id: str, stage: str) -> str:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT materialization_fingerprint FROM document_revisions WHERE revision_id = ?",
                (revision_id,),
            ).fetchone()
        materialization = str(row["materialization_fingerprint"] if row else revision_id)
        # Canonical parsing is committed by IndexingService before the
        # coordinator starts local projection.  Its receipt intentionally uses
        # the revision materialization fingerprint itself (raw bytes + parser,
        # chunker and artifact fingerprints), not provider configuration.  Do
        # not compare it with the projection/enrichment fingerprint below:
        # doing so makes every freshly parsed revision look stale and prevents
        # status from ever reporting a healthy canonical stage.
        if stage == "canonical":
            return materialization
        config = {
            "embedding_model": self.embedding_model,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_document_prefix": self.embedding_document_prefix,
        }
        if stage == "figure_enrichment":
            config.update({"provider": self.vision_provider, "model": self.vision_model})
        elif stage == "evidence_cards":
            config.update({
                "provider": "openai_compatible",
                "model": self.paper_evidence_card_model,
                "schema": self.paper_evidence_card_schema_version,
                "prompt": self.paper_evidence_card_prompt_version,
            })
        elif stage == "lightrag":
            config.update({"provider": "lightrag"})
        elif stage == "visual_index":
            config.update(
                {
                    "provider": self.visual_retrieval_provider,
                    "model": self.visual_retrieval_model,
                    "model_revision": self.visual_retrieval_model_revision,
                    "base_model": self.visual_retrieval_base_model,
                    "base_revision": self.visual_retrieval_base_revision,
                    "render_scale": self.visual_retrieval_render_scale,
                    "render_max_pixels": self.visual_retrieval_render_max_pixels,
                    "batch_max_pixels": self.visual_retrieval_batch_max_pixels,
                    "worker_protocol": "ndjson-v1",
                }
            )
        payload = json.dumps(
            {"stage_version": self.stage_version, "stage": stage, "materialization": materialization, "config": config},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _enqueue_enrichment(self, document_id: str, revision_id: str | None) -> None:
        if not revision_id:
            return
        # Visual page indexing is fully local.  It has its own feature gate but
        # never inherits the 9router privacy gate used by VLM/cards/LightRAG.
        if self.visual_retrieval_enabled and self.visual_retrieval_provider == "colsmol":
            self._mark_stage(
                revision_id,
                "visual_index",
                "pending",
                fingerprint=self._stage_fingerprint(revision_id, "visual_index"),
            )
            self._enqueue_enrichment_job(document_id, revision_id, "visual_index")
        else:
            self._mark_stage(
                revision_id,
                "visual_index",
                "skipped",
                error_class="visual_retrieval_disabled",
            )
        # Derived providers are opt-in per folder.  Local canonical search is
        # already active and remains usable when this privacy gate is absent.
        approved = self._document_enrichment_approved(document_id)
        for stage in ("figure_enrichment", "evidence_cards", "lightrag"):
            if not approved:
                self._mark_stage(
                    revision_id,
                    stage,
                    "skipped",
                    error_class="privacy_gate_not_approved",
                )
                continue
            self._mark_stage(
                revision_id,
                stage,
                "pending",
                fingerprint=self._stage_fingerprint(revision_id, stage),
            )
            self._enqueue_enrichment_job(document_id, revision_id, stage)

    def _document_enrichment_approved(self, document_id: str) -> bool:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        try:
            metadata = json.loads(row["metadata_json"] or "{}") if row else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        return bool(isinstance(metadata, dict) and metadata.get("enrichment_approved"))

    def _set_document_enrichment_approval(self, document_id: str) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["enrichment_approved"] = True
            connection.execute(
                "UPDATE documents SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), document_id),
            )

    def _enqueue_enrichment_job(self, document_id: str, revision_id: str, stage: str) -> str:
        fingerprint = self._stage_fingerprint(revision_id, stage)
        idempotency_key = hashlib.sha256(
            f"enrichment:{stage}:{document_id}:{revision_id}:{fingerprint}".encode("utf-8")
        ).hexdigest()
        now = _now()
        job_id = str(uuid.uuid4())
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT job_id FROM document_processing_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return str(existing["job_id"])
            connection.execute(
                """
                INSERT INTO document_processing_jobs(
                    job_id, kind, folder_path, recursive, file_types_json,
                    idempotency_key, snapshot_fingerprint, status, requested_at,
                    provider, model, progress_json, metadata_json
                ) VALUES (?, ?, ?, 0, '[]', ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    f"enrichment:{stage}",
                    document_id,
                    idempotency_key,
                    fingerprint,
                    now,
                    self._stage_provider(stage),
                    self._stage_model(stage),
                    json.dumps({"stage": stage, "document_id": document_id, "revision_id": revision_id}),
                    json.dumps({"document_id": document_id, "revision_id": revision_id, "stage": stage}),
                ),
            )
        return job_id

    def _stage_provider(self, stage: str) -> str:
        if stage == "visual_index":
            return self.visual_retrieval_provider
        if stage == "lightrag":
            return "lightrag"
        return "9router"

    def _stage_model(self, stage: str) -> str | None:
        if stage == "visual_index":
            return self.visual_retrieval_model
        if stage == "figure_enrichment":
            return self.vision_model
        if stage == "evidence_cards":
            return self.paper_evidence_card_model
        return None

    async def _build_visual_index(
        self,
        document_id: str,
        revision_id: str,
        *,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Render/index active-revision visuals without changing canonical RAG."""

        if not self.visual_retrieval_enabled or self.visual_retrieval_provider != "colsmol":
            return {
                "ok": False,
                "status": "disabled",
                "reason": "visual_retrieval_disabled",
            }
        if self.visual_retrieval_index_root is None:
            raise RuntimeError("visual_retrieval_provider_not_configured")

        from app.rag.visual_worker_client import ColSmolWorkerClient
        from app.rag.visual_retrieval import (
            page_records_from_document_rows,
            render_pdf_page_records,
        )

        with connect(self.db_path) as connection:
            document = connection.execute(
                """
                SELECT id, source_path, file_type, active_revision_id
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if document is None or str(document["active_revision_id"] or "") != revision_id:
                raise RuntimeError("visual_index_revision_is_not_active")
            source_path = Path(str(document["source_path"])).expanduser().resolve()
            file_type = str(document["file_type"] or "").casefold()
            figure_rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id, document_id, revision_id, figure_index, page_number,
                           image_path, caption, metadata_json
                    FROM document_figures
                    WHERE document_id = ? AND revision_id = ?
                    ORDER BY page_number, figure_index
                    """,
                    (document_id, revision_id),
                ).fetchall()
            ]

        if file_type == "pdf":
            pages = await asyncio.to_thread(
                render_pdf_page_records,
                source_path=source_path,
                document_id=document_id,
                revision_id=revision_id,
                output_root=self.visual_retrieval_index_root / "rendered",
                scale=self.visual_retrieval_render_scale,
                max_render_pixels=self.visual_retrieval_render_max_pixels,
            )
        else:
            # DOCX embedded images are still useful visual artifacts, whereas
            # Markdown/TXT without local images correctly remain unsupported.
            pages = page_records_from_document_rows(figure_rows)
        if not pages:
            return {
                "ok": False,
                "status": "partial",
                "requested_pages": 0,
                "reason": f"visual_pages_unavailable_for_{file_type or 'unknown'}",
            }

        if self._visual_provider_instance is None:
            if self.visual_retrieval_worker_python is None:
                raise RuntimeError("visual_retrieval_worker_python_not_configured")
            self._visual_provider_instance = ColSmolWorkerClient(
                python_executable=self.visual_retrieval_worker_python,
                index_root=self.visual_retrieval_index_root / "colsmol",
                model_name=self.visual_retrieval_model,
                model_revision=self.visual_retrieval_model_revision,
                base_model_name=self.visual_retrieval_base_model,
                base_model_revision=self.visual_retrieval_base_revision,
                device=self.visual_retrieval_device,
                local_files_only=self.visual_retrieval_local_files_only,
                max_image_pixels_per_request=self.visual_retrieval_batch_max_pixels,
            )
        self._update_job_progress(
            job_id,
            {
                "stage": "visual_index",
                "status": "running",
                "requested_pages": len(pages),
                "processed_pages": 0,
                "remaining_pages": len(pages),
                "document_id": document_id,
                "revision_id": revision_id,
            },
        )
        result = await self._visual_provider_instance.index_pages(
            pages,
            progress_callback=(
                lambda progress: self._update_job_progress(
                    job_id,
                    {
                        "stage": "visual_index",
                        "status": "running",
                        "document_id": document_id,
                        "revision_id": revision_id,
                        **progress,
                    },
                )
            )
            if job_id
            else None,
        )
        return {
            **result,
            "document_id": document_id,
            "revision_id": revision_id,
            "source_path": str(source_path),
        }

    def _update_job_progress(
        self,
        job_id: str | None,
        progress: dict[str, Any],
    ) -> None:
        """Persist bounded derived-stage progress without changing job state.

        Visual page batches commit independently in the worker. Persisting
        their counters lets status clients show forward motion and lets a
        restarted job resume from the worker manifest while local text/table
        search remains active. The update is conditional on the current lease
        owner so a recovered worker cannot overwrite a newer attempt.
        """

        if not job_id:
            return
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE document_processing_jobs
                SET progress_json = ?, heartbeat_at = ?, lease_expires_at = ?
                WHERE job_id = ? AND status = 'running' AND lease_owner = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=False, default=str),
                    _now(),
                    _lease_expiry(self.lease_seconds),
                    job_id,
                    self.worker_id,
                ),
            )

    def _validate_local_projection(
        self,
        document_id: str,
        revision_id: str,
        result: dict[str, Any],
    ) -> bool:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM chunks WHERE document_id = ? AND revision_id = ? AND chunk_type = 'text') AS chunks,
                    (SELECT COUNT(*) FROM document_tables WHERE document_id = ? AND revision_id = ?) AS tables,
                    (SELECT COUNT(*) FROM document_figures WHERE document_id = ? AND revision_id = ?) AS figures,
                    (SELECT COUNT(*) FROM rag_chunks_fts_v2 WHERE document_id = ? AND revision_id = ?) AS fts
                """,
                (
                    document_id, revision_id,
                    document_id, revision_id,
                    document_id, revision_id,
                    document_id, revision_id,
                ),
            ).fetchone()
        if row is None:
            return False
        return (
            int(row["chunks"]) == int(result.get("text_chunks", -1))
            and 0 <= int(result.get("table_chunks", -1)) <= int(row["tables"])
            and 0 <= int(result.get("figure_chunks", -1)) <= int(row["figures"])
            and int(row["fts"]) == int(result.get("text_chunks", -1))
            and _projection_counts_match(result)
        )
    async def _heartbeat_loop(self, job_id: str) -> None:
        interval = max(1.0, min(30.0, self.lease_seconds / 3))
        while True:
            await asyncio.sleep(interval)
            now = _now()
            with connect(self.db_path) as connection:
                connection.execute(
                    """
                    UPDATE document_processing_jobs
                    SET heartbeat_at = ?, lease_expires_at = ?
                    WHERE job_id = ? AND status = 'running' AND lease_owner = ?
                    """,
                    (now, _lease_expiry(self.lease_seconds), job_id, self.worker_id),
                )

    def _retryable_error(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__} {exc}".lower()
        return any(
            token in text
            for token in ("timeout", "connection", "unavailable", "embedding", "provider", "lancedb", "temporarily")
        )

    def _is_retryable_result(self, payload: dict[str, Any]) -> bool:
        return any(
            "error" in item and self._retryable_error(RuntimeError(str(item.get("error"))))
            for item in payload.get("local_projection", [])
        )

    def _reschedule_job(
        self,
        job_id: str,
        result: dict[str, Any],
        error_class: str,
        error_message: str | None = None,
    ) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT attempt_count FROM document_processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            attempt = int(row["attempt_count"] if row else 1)
            delay = min(900, 2 ** max(0, attempt - 1) * 5)
            retry_at = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
            connection.execute(
                """
                UPDATE document_processing_jobs
                SET status = 'pending', next_attempt_at = ?, finished_at = NULL,
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_class = ?, error_message = ?, result_json = ?,
                    progress_json = ?
                WHERE job_id = ?
                """,
                (
                    retry_at,
                    error_class,
                    error_message,
                    json.dumps(result, ensure_ascii=False, default=str),
                    json.dumps({"stage": "retry_scheduled", "delay_seconds": delay}),
                    job_id,
                ),
            )

    def _claim_next_job(self, *, lane: str = "all") -> str | None:
        if lane not in {"all", "local", "enrichment"}:
            raise ValueError("lane must be all, local or enrichment")
        now = _now()
        lane_clause = ""
        if lane == "local":
            lane_clause = " AND kind NOT LIKE 'enrichment:%'"
        elif lane == "enrichment":
            lane_clause = " AND kind LIKE 'enrichment:%'"
        with connect(self.db_path) as connection:
            row = connection.execute(
                f"""
                SELECT job_id FROM document_processing_jobs
                WHERE status = 'pending'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                  {lane_clause}
                ORDER BY
                  CASE WHEN kind LIKE 'enrichment:%' THEN 1 ELSE 0 END,
                  requested_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            cursor = connection.execute(
                """
                UPDATE document_processing_jobs
                SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                    heartbeat_at = ?, started_at = ?, attempt_count = attempt_count + 1,
                    progress_json = ?
                WHERE job_id = ? AND status = 'pending'
                """,
                (
                    self.worker_id,
                    _lease_expiry(self.lease_seconds),
                    now,
                    now,
                    json.dumps({"stage": "running"}),
                    job_id,
                ),
            )
            return job_id if cursor.rowcount == 1 else None

    def _claim_job(self, job_id: str) -> bool:
        now = _now()
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE document_processing_jobs
                SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                    heartbeat_at = ?, started_at = ?, attempt_count = attempt_count + 1,
                    progress_json = ?
                WHERE job_id = ? AND status = 'pending'
                """,
                (
                    self.worker_id,
                    _lease_expiry(self.lease_seconds),
                    now,
                    now,
                    json.dumps({"stage": "running"}),
                    job_id,
                ),
            )
        return cursor.rowcount == 1

    def _finish_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any],
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with connect(self.db_path) as connection:
            prior = connection.execute(
                "SELECT progress_json FROM document_processing_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            try:
                progress = json.loads(prior["progress_json"] or "{}") if prior else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                progress = {}
            if not isinstance(progress, dict):
                progress = {}
            # Keep the last per-file counters/path when a job becomes terminal.
            # This makes a completed/partial job auditable after the worker
            # releases its lease instead of collapsing useful progress into
            # only ``{"stage": "complete"}``.
            progress = {**progress, "stage": status, "status": status}
            connection.execute(
                """
                UPDATE document_processing_jobs
                SET status = ?, finished_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL,
                    error_class = ?, error_message = ?, result_json = ?,
                    progress_json = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    _now(),
                    error_class,
                    error_message,
                    json.dumps(result, ensure_ascii=False, default=str),
                    json.dumps(progress, ensure_ascii=False, default=str),
                    job_id,
                ),
            )

    @staticmethod
    def _job_from_row(row: Any) -> ProcessingJob:
        return ProcessingJob(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            folder_path=str(row["folder_path"]),
            recursive=bool(row["recursive"]),
            file_types=list(json.loads(row["file_types_json"] or "[]")),
            status=str(row["status"]),
            attempt_count=int(row["attempt_count"] or 0),
            idempotency_key=row["idempotency_key"],
            snapshot_fingerprint=row["snapshot_fingerprint"],
            next_attempt_at=row["next_attempt_at"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            provider=row["provider"],
            model=row["model"],
            progress=dict(json.loads(row["progress_json"] or "{}")),
            error_class=row["error_class"],
            error_message=row["error_message"],
            result=dict(json.loads(row["result_json"] or "{}")),
        )
