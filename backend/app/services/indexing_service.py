from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import mimetypes
from pathlib import Path
import shutil
import uuid
from typing import Callable

from app.catalog.document_card import DocumentCardDraft, build_document_card
from app.db.sqlite import connect
from app.rag.chunking import chunk_parsed_document
from app.rag.text_normalization import searchable_text
from app.rag.document_identity import revision_fingerprint
from app.rag.figure_quality import figure_is_indexable
from app.rag.parsers import (
    SUPPORTED_TEXT_TYPES,
    ParsedDocument,
    PAGE_OCR_SCHEMA,
    ParsedPage,
    parse_document,
    supported_file_type,
)
from app.rag.readiness import assess_document_readiness
from app.rag.vision import (
    FigureDocumentContext,
    OllamaVisionSummarizer,
    OpenAICompatibleVisionSummarizer,
)


CHUNKING_VERSION = 4
ARTIFACT_EXTRACTION_VERSION = 11


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class IndexedDocument:
    id: str
    file_id: str | None
    source_path: str
    filename: str
    file_type: str
    chunk_count: int
    revision_id: str | None = None
    previous_revision_id: str | None = None


@dataclass(frozen=True)
class IndexFolderResult:
    folder_id: str
    folder_path: str
    scanned_files: int
    indexed_files: int
    skipped_files: int
    failed_files: int
    documents: list[IndexedDocument] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class IndexingService:
    db_path: Path
    artifact_root: Path | None = None
    ollama_host: str = "http://localhost:11434"
    vision_model: str | None = None
    vision_provider: str = "openai_compatible"
    openai_api_base: str = "http://localhost:20128/v1"
    openai_api_key: str = "any"
    request_timeout_seconds: float = 120.0
    # OCR is part of the normal folder lifecycle.  Isolated visual retrieval
    # experiments may disable it explicitly so cold OCR cost cannot mask the
    # visual provider's own recall; production/default behavior stays enabled.
    enable_pdf_ocr: bool = True
    # Bound foreground OCR for image-heavy sources. Pages outside this budget
    # remain visual-only and are eligible for a later durable OCR retry.
    pdf_ocr_page_budget: int | None = 12
    # Printed-page labels are cheap footer metadata and are OCR'd during the
    # ingestion stage, never synchronously inside retrieval. Large documents
    # above this budget remain honest physical-page-only until enrichment.
    pdf_page_label_ocr_budget: int | None = 16
    paper_evidence_card_build_enabled: bool = False
    paper_evidence_card_model: str = "cx/gpt-6.1-sol"
    paper_evidence_card_schema_version: str = "v1"
    paper_evidence_card_prompt_version: str = "v2"
    # Trusted local source adapters may contribute authored attachments that
    # are not represented by the extracted text bytes (e.g. a table image).
    # Their content/policy identity must participate in revision identity.
    canonical_artifact_fingerprint: str | None = None

    def index_file(
        self,
        *,
        source_path: str,
        collection_name: str | None = None,
        collection_type: str = "manual",
        scope_type: str = "global",
        scope_id: str | None = None,
        defer_activation: bool = False,
    ) -> IndexedDocument:
        path = Path(source_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {source_path}")

        file_type = supported_file_type(path)
        if file_type is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        now = utc_now()
        folder_id = self._upsert_folder(path.parent, False, [file_type], now)
        self._upsert_approved_folder(path.parent, "read_index", False, now)
        collection_id = None
        if collection_name:
            collection_id = self._upsert_collection(
                name=collection_name,
                collection_type=collection_type,
                scope_type=scope_type,
                scope_id=scope_id,
                metadata={"source": "index_file"},
                now=now,
            )

        document = self._index_file(
            path,
            file_type,
            folder_id,
            collection_id=collection_id,
            return_existing=not defer_activation,
        )
        if document is None:
            existing = self._get_indexed_document_by_path(str(path))
            if existing is None:
                raise ValueError(f"Could not index file: {source_path}")
            return existing
        return document

    def index_selected_files(
        self,
        *,
        source_paths: list[str],
        collection_name: str,
        collection_type: str = "manual",
        scope_type: str = "global",
        scope_id: str | None = None,
        defer_activation: bool = False,
    ) -> dict:
        if not source_paths:
            raise ValueError("No files were provided")

        now = utc_now()
        collection_id = self._upsert_collection(
            name=collection_name,
            collection_type=collection_type,
            scope_type=scope_type,
            scope_id=scope_id,
            metadata={"source": "index_selected_files"},
            now=now,
        )
        indexed: list[IndexedDocument] = []
        skipped = failed = 0
        errors: list[dict] = []

        for source_path in source_paths:
            path = Path(source_path).expanduser().resolve()
            file_type = supported_file_type(path)
            if not path.exists() or not path.is_file() or file_type is None:
                failed += 1
                errors.append({"source_path": source_path, "error": "missing_or_unsupported"})
                continue

            folder_id = self._upsert_folder(path.parent, False, [file_type], now)
            self._upsert_approved_folder(path.parent, "read_index", False, now)
            try:
                document = self._index_file(
                    path,
                    file_type,
                    folder_id,
                    collection_id=collection_id,
                    return_existing=not defer_activation,
                )
            except Exception as exc:
                failed += 1
                errors.append({"source_path": source_path, "error": str(exc)})
                continue

            if document is None:
                skipped += 1
                existing = self._get_indexed_document_by_path(str(path))
                if existing is not None:
                    indexed.append(existing)
            else:
                indexed.append(document)

        return {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "indexed_files": len(indexed),
            "skipped_files": skipped,
            "failed_files": failed,
            "documents": indexed,
            "errors": errors,
        }

    def index_folder(
        self,
        *,
        folder_path: str,
        recursive: bool,
        file_types: list[str],
        defer_activation: bool = False,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> IndexFolderResult:
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        selected_types = {item.lower().lstrip(".") for item in file_types if item.strip()}
        if not selected_types:
            selected_types = set(SUPPORTED_TEXT_TYPES)

        now = utc_now()
        folder_id = self._upsert_folder(root, recursive, sorted(selected_types), now)
        self._upsert_approved_folder(root, "read_index", recursive, now)
        collection_id = self._upsert_collection(
            name=f"folder:{root.name or str(root)}",
            collection_type="folder",
            scope_type="folder",
            scope_id=folder_id,
            metadata={"folder_path": str(root), "recursive": recursive},
            now=now,
        )
        scanned = indexed = skipped = failed = 0
        errors: list[dict[str, object]] = []
        documents: list[IndexedDocument] = []
        seen_paths: set[str] = set()

        paths = root.rglob("*") if recursive else root.iterdir()
        for path in paths:
            if not path.is_file():
                continue
            file_type = supported_file_type(path)
            if file_type is None or file_type not in selected_types:
                continue

            scanned += 1
            seen_paths.add(str(path.resolve()))
            try:
                document = self._index_file(
                    path,
                    file_type,
                    folder_id,
                    collection_id=collection_id,
                    # A durable folder job must stage every new/changed
                    # revision and let DocumentProcessingService activate it
                    # only after the local FTS/Lance projection validates.
                    # The previous call omitted this propagation, so the
                    # supposedly deferred folder path could activate a
                    # revision during parsing.
                    return_existing=not defer_activation,
                )
            except Exception as exc:
                failed += 1
                # Keep the folder result diagnostic and retryable. Previously
                # the coordinator only returned ``failed_files`` so an
                # arbitrary document failure became an opaque partial job.
                errors.append(
                    {
                        "source_path": str(path.resolve()),
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "stage": "canonical",
                                "status": "running",
                                "scanned_files": scanned,
                                "indexed_files": indexed,
                                "skipped_files": skipped,
                                "failed_files": failed,
                                "current_path": str(path.resolve()),
                                "current_file_type": file_type,
                            }
                        )
                    except Exception:
                        pass
                continue

            if document is None:
                skipped += 1
            else:
                indexed += 1
                documents.append(document)
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "stage": "canonical",
                            "status": "running",
                            "scanned_files": scanned,
                            "indexed_files": indexed,
                            "skipped_files": skipped,
                            "failed_files": failed,
                            "current_path": str(path.resolve()),
                            "current_file_type": file_type,
                            "document_id": document.id if document is not None else None,
                            "revision_id": document.revision_id if document is not None else None,
                        }
                    )
                except Exception:
                    pass

        # A failed/partial scan must never tombstone documents that were not
        # observed. Only a complete scan can establish that a source vanished.
        if failed == 0:
            self._reconcile_folder_sources(
                folder_id=folder_id,
                seen_paths=seen_paths,
                selected_types=selected_types,
                now=now,
            )
        if not defer_activation:
            for document in documents:
                if document.revision_id:
                    self.activate_revision(document.id, document.revision_id)

        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "stage": "canonical",
                        "status": "complete" if failed == 0 else "partial",
                        "scanned_files": scanned,
                        "indexed_files": indexed,
                        "skipped_files": skipped,
                        "failed_files": failed,
                        "current_path": None,
                        "current_file_type": None,
                    }
                )
            except Exception:
                pass

        return IndexFolderResult(
            folder_id=folder_id,
            folder_path=str(root),
            scanned_files=scanned,
            indexed_files=indexed,
            skipped_files=skipped,
            failed_files=failed,
            documents=documents,
            errors=errors,
        )

    def _reconcile_folder_sources(
        self,
        *,
        folder_id: str,
        seen_paths: set[str],
        selected_types: set[str],
        now: str,
    ) -> None:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, source_path
                FROM document_sources
                WHERE folder_id = ? AND status = 'active'
                """,
                (folder_id,),
            ).fetchall()
            for row in rows:
                source_path = str(row["source_path"])
                suffix = Path(source_path).suffix.lower().lstrip(".")
                if suffix not in selected_types or source_path in seen_paths:
                    continue
                connection.execute(
                    """
                    UPDATE document_sources
                    SET status = 'deleted', last_seen_at = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (now, '{"reason":"folder_scan_missing"}', row["id"]),
                )
                active_alias = connection.execute(
                    """
                    SELECT 1 FROM document_sources
                    WHERE document_id = ? AND status = 'active'
                    LIMIT 1
                    """,
                    (row["document_id"],),
                ).fetchone()
                if active_alias is None:
                    connection.execute(
                        """
                        UPDATE documents
                        SET revision_status = 'retired', index_status = 'retired', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, row["document_id"]),
                    )

    def activate_revision(self, document_id: str, revision_id: str) -> bool:
        """Atomically switch a logical document to a validated revision.

        Canonical rows and vectors are written before this pointer changes.
        Keeping the switch in SQLite makes retrieval either see the prior
        complete revision or the new complete revision, never a half-built
        mixture.
        """
        now = utc_now()
        with connect(self.db_path) as connection:
            document = connection.execute(
                "SELECT active_revision_id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            revision = connection.execute(
                """
                SELECT revision_id, status
                FROM document_revisions
                WHERE revision_id = ? AND document_id = ?
                """,
                (revision_id, document_id),
            ).fetchone()
            if document is None or revision is None:
                return False
            if revision["status"] not in {"staged", "active"}:
                return False
            previous = document["active_revision_id"]
            self._promote_revision_card(connection, document_id, revision_id, now)
            connection.execute(
                """
                UPDATE documents
                SET active_revision_id = ?, revision_status = 'active',
                    index_status = 'indexed', updated_at = ?,
                    table_count = (SELECT COUNT(*) FROM document_tables WHERE document_id = ? AND revision_id = ?),
                    figure_count = (SELECT COUNT(*) FROM document_figures WHERE document_id = ? AND revision_id = ?)
                WHERE id = ?
                """,
                (revision_id, now, document_id, revision_id, document_id, revision_id, document_id),
            )
            connection.execute(
                """
                UPDATE document_revisions
                SET status = 'active', activated_at = ?
                WHERE revision_id = ?
                """,
                (now, revision_id),
            )
            if previous and previous != revision_id:
                connection.execute(
                    """
                    UPDATE document_revisions
                    SET status = 'retired', retired_at = ?
                    WHERE revision_id = ? AND status = 'active'
                    """,
                    (now, previous),
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO document_stage_receipts(
                    revision_id, stage, fingerprint, status, started_at,
                    finished_at, metadata_json
                ) VALUES (?, 'activate_local', ?, 'complete', ?, ?, ?)
                """,
                (
                    revision_id,
                    f"activation:{revision_id}",
                    now,
                    now,
                    json.dumps({"previous_revision_id": previous}),
                ),
            )
        return True

    def list_folders(self) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, folder_path, recursive, file_types, created_at, updated_at
                FROM indexed_folders
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                **dict(row),
                "recursive": bool(row["recursive"]),
                "file_types": json.loads(row["file_types"]),
            }
            for row in rows
        ]

    def list_documents(self, limit: int = 100) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, file_id, folder_id, source_path, filename, file_type,
                       title, doc_type, language, modified_at, indexed_at,
                       chunk_count, table_count, figure_count, parse_status,
                       index_status, parser_name, page_count
                FROM documents
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, file_id, folder_id, source_path, filename, file_type,
                       title, doc_type, language, content_hash, modified_at,
                       indexed_at, chunk_count, table_count, figure_count,
                       parse_status, index_status, parser_name, parser_version,
                       page_count, error_message, updated_at, metadata_json
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_chunks(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, file_id, chunk_index, content, source_path, filename,
                       chunk_type, heading_path_json, token_count, char_count, order_index,
                       parent_chunk_id, page_number, lance_table, lance_id, created_at, metadata_json
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        chunks: list[dict] = []
        for row in rows:
            item = dict(row)
            item["heading_path"] = json.loads(item.pop("heading_path_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            chunks.append(item)
        return chunks

    def rechunk_document(self, document_id: str) -> dict:
        """Upgrade canonical text chunks without touching tables or figures."""

        with connect(self.db_path) as connection:
            document = connection.execute(
                """
                SELECT id, file_id, source_path, filename, file_type, content_hash,
                       parser_name, parser_version, page_count, metadata_json
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if document is None:
                raise ValueError(f"Document not found: {document_id}")
            rows = connection.execute(
                """
                SELECT content, page_number, chunk_index, metadata_json
                FROM chunks
                WHERE document_id = ? AND chunk_type = 'text'
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
            card = connection.execute(
                """
                SELECT title_guess, short_summary
                FROM document_cards
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if not rows:
            raise ValueError(f"Document has no canonical text chunks: {document_id}")

        parsed = _reconstruct_parsed_document(document, rows)
        chunks = chunk_parsed_document(parsed)
        now = utc_now()
        metadata = json.loads(document["metadata_json"] or "{}")
        metadata.update(
            {
                "page_chunk_version": CHUNKING_VERSION,
                "chunking_strategy": "token_structural_parent_child_v1",
                "chunk_tokenizer": "unicode_lexical_v1",
            }
        )

        with connect(self.db_path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM rag_chunks_fts WHERE document_id = ?",
                    (document_id,),
                )
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?",
                    (document_id,),
                )
                self._insert_text_chunks(
                    connection,
                    chunks=chunks,
                    document_id=document_id,
                    file_id=document["file_id"],
                    source_path=str(document["source_path"]),
                    filename=str(document["filename"]),
                    file_type=str(document["file_type"]),
                    content_hash=str(document["content_hash"]),
                    document_title=str(card["title_guess"] or "") if card else "",
                    document_summary=str(card["short_summary"] or "") if card else "",
                    now=now,
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET chunk_count = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (len(chunks), json.dumps(metadata), now, document_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "document_id": document_id,
            "filename": document["filename"],
            "old_chunk_count": len(rows),
            "new_chunk_count": len(chunks),
            "chunking_version": CHUNKING_VERSION,
        }

    def rechunk_all_documents(self, limit: int | None = None) -> dict:
        with connect(self.db_path) as connection:
            query = "SELECT id FROM documents ORDER BY indexed_at ASC"
            params: tuple = ()
            if limit is not None:
                query += " LIMIT ?"
                params = (limit,)
            document_ids = [
                str(row["id"]) for row in connection.execute(query, params).fetchall()
            ]
        results = [self.rechunk_document(document_id) for document_id in document_ids]
        return {
            "documents": len(document_ids),
            "chunking_version": CHUNKING_VERSION,
            "results": results,
        }

    def delete_document(self, document_id: str) -> bool:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if existing is None:
                return False
            self._delete_document(connection, document_id)
        self._delete_artifacts(document_id)
        return True

    def list_collections(self) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, name, type, scope_type, scope_id, created_at, updated_at, metadata_json
                FROM collections
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [_decode_metadata(dict(row)) for row in rows]

    def list_document_cards(self, limit: int = 100) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_cards.*, documents.filename, documents.source_path
                FROM document_cards
                JOIN documents ON documents.id = document_cards.document_id
                ORDER BY importance_score DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_decode_card(dict(row)) for row in rows]

    def backfill_missing_document_cards(
        self,
        *,
        document_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        """Create deterministic cards for legacy documents which have text.

        Modern ingestion creates a card before it persists chunks. Older
        imports or interrupted migrations can therefore contain canonical text
        but no card; vector indexing correctly skips them, leaving a document
        visible in SQLite yet unreachable through hybrid retrieval. This local,
        idempotent repair derives a card only from the stored filename and text
        chunks. It never reparses a source, calls a model, or replaces a card
        that already exists.
        """

        requested_ids = list(dict.fromkeys(
            str(document_id).strip()
            for document_id in (document_ids or [])
            if str(document_id).strip()
        ))
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when provided")

        clauses = ["dc.document_id IS NULL"]
        params: list[object] = []
        if requested_ids:
            placeholders = ", ".join("?" for _ in requested_ids)
            clauses.append(f"d.id IN ({placeholders})")
            params.extend(requested_ids)
        sql = """
            SELECT d.id, d.source_path, d.filename
            FROM documents d
            LEFT JOIN document_cards dc ON dc.document_id = d.id
            WHERE """ + " AND ".join(clauses) + " ORDER BY d.indexed_at ASC, d.id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        now = utc_now()
        backfilled: list[str] = []
        skipped: list[dict[str, str]] = []
        with connect(self.db_path) as connection:
            documents = connection.execute(sql, params).fetchall()
            for document in documents:
                document_id = str(document["id"])
                chunk_rows = connection.execute(
                    """
                    SELECT content
                    FROM chunks
                    WHERE document_id = ? AND chunk_type = 'text'
                    ORDER BY chunk_index ASC
                    """,
                    (document_id,),
                ).fetchall()
                text = "\n\n".join(
                    str(row["content"] or "").strip()
                    for row in chunk_rows
                    if str(row["content"] or "").strip()
                )
                if not text:
                    skipped.append(
                        {
                            "document_id": document_id,
                            "reason": "no_canonical_text_chunks",
                        }
                    )
                    continue
                source_path = str(document["source_path"] or document["filename"] or "document")
                card = build_document_card(Path(source_path), text)
                self._upsert_document_card(connection, document_id, card, now)
                connection.execute(
                    """
                    UPDATE documents
                    SET title = COALESCE(NULLIF(title, ''), ?),
                        doc_type = COALESCE(NULLIF(doc_type, ''), ?),
                        language = COALESCE(NULLIF(language, ''), ?),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        card.title_guess,
                        card.doc_type,
                        card.language,
                        now,
                        document_id,
                    ),
                )
                backfilled.append(document_id)

        return {
            "requested_document_ids": requested_ids,
            "backfilled_document_ids": backfilled,
            "skipped": skipped,
            "backfilled_count": len(backfilled),
        }

    def _upsert_folder(self, root: Path, recursive: bool, file_types: list[str], now: str) -> str:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM indexed_folders WHERE folder_path = ?",
                (str(root),),
            ).fetchone()
            if existing:
                folder_id = existing["id"]
                connection.execute(
                    """
                    UPDATE indexed_folders
                    SET recursive = ?, file_types = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (int(recursive), json.dumps(file_types), now, folder_id),
                )
                return folder_id

            folder_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO indexed_folders (id, folder_path, recursive, file_types, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (folder_id, str(root), int(recursive), json.dumps(file_types), now, now),
            )
            return folder_id

    def _index_file(
        self,
        path: Path,
        file_type: str,
        folder_id: str,
        *,
        collection_id: str | None = None,
        return_existing: bool = False,
    ) -> IndexedDocument | None:
        now = utc_now()
        file_id = self._upsert_file(path, file_type, now)
        raw_content_hash = _hash_file(path)
        source_path = str(path)
        revision_rebuild_document_id: str | None = None

        # Raw bytes are the authoritative change detector.  Do this lookup
        # before parsing: unchanged PDFs (especially image-only PDFs) used to
        # pay the full Docling + OCR cost on every rescan, and duplicate bytes
        # at a second path were parsed before being recognized as aliases.
        # Re-parse only when a parser/artifact version explicitly requires a
        # replacement revision.
        with connect(self.db_path) as connection:
            existing_fast = connection.execute(
                """
                SELECT id, file_id, source_path, filename, file_type, chunk_count,
                       active_revision_id, raw_content_hash, revision_status,
                       index_status
                FROM documents
                WHERE source_path = ?
                """,
                (source_path,),
            ).fetchone()
            alias_fast = None
            if existing_fast is None:
                alias_fast = connection.execute(
                    """
                    SELECT id, file_id, source_path, filename, file_type, chunk_count,
                           active_revision_id, raw_content_hash, revision_status,
                           index_status
                    FROM documents
                    WHERE raw_content_hash = ?
                    ORDER BY indexed_at ASC
                    LIMIT 1
                    """,
                    (raw_content_hash,),
                ).fetchone()

            fast_document = existing_fast or alias_fast
            if fast_document is not None and str(fast_document["raw_content_hash"] or "") == raw_content_hash:
                needs_reindex = self._needs_reindex_for_page_metadata(
                    connection, str(fast_document["id"])
                )
                if needs_reindex:
                    # A byte-identical alias can still need a new parser or
                    # authored-attachment revision. Carry its logical identity
                    # into the rebuild lookup below; otherwise the later raw
                    # dedup branch returns the old active revision and an
                    # adapter may overwrite its supposedly immutable assets.
                    revision_rebuild_document_id = str(fast_document["id"])
                if not needs_reindex:
                    document_id = str(fast_document["id"])
                    self._ensure_document_source(
                        connection,
                        document_id=document_id,
                        folder_id=folder_id,
                        source_path=source_path,
                        raw_content_hash=raw_content_hash,
                        now=now,
                    )
                    if collection_id:
                        self._link_document_to_collection(connection, collection_id, document_id, now)
                    # A previous run may have completed canonical parsing but
                    # stopped before local FTS/Lance validation (for example
                    # while Ollama was offline). Treat that staged revision as
                    # work to resume; otherwise an unchanged rescan would
                    # return ``None`` and the durable retry could never
                    # project or activate it.
                    staged_revision = connection.execute(
                        """
                        SELECT revision_id
                        FROM document_revisions
                        WHERE document_id = ? AND status = 'staged'
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (document_id,),
                    ).fetchone()
                    active_revision = str(fast_document["active_revision_id"] or "")
                    revision_status = str(fast_document["revision_status"] or "")
                    pending_revision_id = (
                        str(staged_revision["revision_id"])
                        if staged_revision is not None
                        else ""
                    )
                    if pending_revision_id or not active_revision or revision_status != "active":
                        return IndexedDocument(
                            id=document_id,
                            file_id=fast_document["file_id"],
                            source_path=str(fast_document["source_path"] or source_path),
                            filename=str(fast_document["filename"] or path.name),
                            file_type=str(fast_document["file_type"] or file_type),
                            chunk_count=int(fast_document["chunk_count"] or 0),
                            revision_id=pending_revision_id or active_revision or None,
                        )
                    if return_existing:
                        return IndexedDocument(
                            id=document_id,
                            file_id=file_id,
                            source_path=str(fast_document["source_path"] or source_path),
                            filename=str(fast_document["filename"] or path.name),
                            file_type=str(fast_document["file_type"] or file_type),
                            chunk_count=int(fast_document["chunk_count"] or 0),
                            revision_id=fast_document["active_revision_id"],
                        )
                    return None

        # New/changed content (or an explicit parser/artifact rebuild) is
        # parsed once below.  The raw hash remains stable even if extracted
        # text changes because Docling/OCR versions change.
        # This first pass is only for the extracted-text integrity hash and
        # card metadata.  Full page OCR belongs to the canonical artifact pass
        # below; running it here as well doubled cold indexing time for scanned
        # PDFs while contributing no new identity information.
        parsed_for_hash = parse_document(path, enable_pdf_ocr=False)
        text = parsed_for_hash.text
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        card = build_document_card(path, text)
        existing_document_id: str | None = None
        existing_revision_id: str | None = None
        existing_collection_ids: list[str] = []

        with connect(self.db_path) as connection:
            existing = connection.execute(
                """
                SELECT id, file_id, source_path, filename, file_type, chunk_count,
                       content_hash, raw_content_hash, active_revision_id
                FROM documents
                WHERE source_path = ? OR id = ?
                """,
                (source_path, revision_rebuild_document_id),
            ).fetchone()
            if existing:
                existing_document_id = str(existing["id"])
                existing_revision_id = str(existing["active_revision_id"] or "") or None
                existing_collection_ids = [
                    str(row["collection_id"])
                    for row in connection.execute(
                        "SELECT collection_id FROM collection_documents WHERE document_id = ?",
                        (existing_document_id,),
                    ).fetchall()
                ]
            raw_matches = bool(
                existing
                and str(existing["raw_content_hash"] or "") == raw_content_hash
            )
            if existing and raw_matches and existing["content_hash"] == content_hash:
                if self._needs_reindex_for_page_metadata(connection, existing["id"]):
                    # Keep the current canonical document live until parsing,
                    # chunking and artifact staging for its replacement have
                    # all succeeded. Deleting here causes data loss whenever a
                    # parser/chunker fails during a version-triggered rebuild.
                    pass
                else:
                    if existing["file_id"] is None:
                        connection.execute(
                            "UPDATE documents SET file_id = ?, updated_at = ? WHERE id = ?",
                            (file_id, now, existing["id"]),
                        )
                    self._upsert_document_card(connection, existing["id"], card, now)
                    connection.execute(
                        "UPDATE document_cards SET revision_id = COALESCE(revision_id, ?) WHERE document_id = ?",
                        (existing["active_revision_id"], existing["id"]),
                    )
                    self._ensure_document_source(
                        connection,
                        document_id=str(existing["id"]),
                        folder_id=folder_id,
                        source_path=source_path,
                        raw_content_hash=raw_content_hash,
                        now=now,
                    )
                    if collection_id:
                        self._link_document_to_collection(connection, collection_id, existing["id"], now)
                    if return_existing:
                        return IndexedDocument(
                            id=existing["id"],
                            file_id=file_id,
                            source_path=existing["source_path"],
                            filename=existing["filename"],
                            file_type=existing["file_type"],
                            chunk_count=existing["chunk_count"],
                            revision_id=existing["active_revision_id"],
                        )
                    return None

            # An exact duplicate at another path is an alias of the existing
            # logical document. Do not parse, chunk, embed, or create a second
            # canonical document for it.
            if existing is None:
                duplicate = connection.execute(
                    """
                    SELECT d.id, d.file_id, d.source_path, d.filename,
                           d.file_type, d.chunk_count, d.active_revision_id
                    FROM documents d
                    LEFT JOIN files f ON f.id = d.file_id
                    WHERE COALESCE(d.raw_content_hash, f.content_hash) = ?
                    ORDER BY d.indexed_at ASC
                    LIMIT 1
                    """,
                    (raw_content_hash,),
                ).fetchone()
                if duplicate:
                    duplicate_id = str(duplicate["id"])
                    self._ensure_document_source(
                        connection,
                        document_id=duplicate_id,
                        folder_id=folder_id,
                        source_path=source_path,
                        raw_content_hash=raw_content_hash,
                        now=now,
                    )
                    if collection_id:
                        self._link_document_to_collection(connection, collection_id, duplicate_id, now)
                    if return_existing:
                        return IndexedDocument(
                            id=duplicate_id,
                            file_id=file_id,
                            source_path=str(duplicate["source_path"]),
                            filename=str(duplicate["filename"]),
                            file_type=str(duplicate["file_type"]),
                            chunk_count=int(duplicate["chunk_count"] or 0),
                            revision_id=duplicate["active_revision_id"],
                        )
                    return None

        # A source path owns a stable canonical document identity. Derived
        # LightRAG/Lance rows, sticky L1 state and persisted citations can then
        # survive parser/chunking version rebuilds.
        document_id = existing_document_id or str(uuid.uuid4())
        artifact_staging_id = (
            str(uuid.uuid4())
            if existing_document_id and file_type == "pdf" and self.artifact_root is not None
            else document_id
        )
        parsed = self._parse_for_index(
            path,
            file_type,
            artifact_staging_id,
            parsed_for_hash,
            card,
        )
        chunks = chunk_parsed_document(parsed)
        tables = parsed.tables or []
        figures = parsed.figures or []
        readiness = assess_document_readiness(parsed)
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        parser_fingerprint = f"{parsed.parser_name}:{parsed.parser_version}"
        chunker_fingerprint = f"token_structural_parent_child:{CHUNKING_VERSION}"
        policy = self.materialization_policy(file_type)
        policy_digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode("utf-8")
        ).hexdigest()
        artifact_fingerprint = (
            f"artifacts:{ARTIFACT_EXTRACTION_VERSION}:"
            f"vision={self.vision_provider}:{self.vision_model or 'off'}:policy={policy_digest}"
        )
        materialization_fingerprint = revision_fingerprint(
            raw_content_hash=raw_content_hash,
            parser_fingerprint=parser_fingerprint,
            chunker_fingerprint=chunker_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
        )
        revision_id = f"rev:{document_id}:{materialization_fingerprint[:24]}"

        with connect(self.db_path) as connection:
            existing_row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            metadata_json = json.dumps(
                {
                    "folder_id": folder_id,
                    "file_id": file_id,
                    "page_chunk_version": CHUNKING_VERSION,
                    "artifact_extraction_version": ARTIFACT_EXTRACTION_VERSION,
                    "materialization_policy": policy,
                    "vision_provider": self.vision_provider if self.vision_model else None,
                    "vision_model": self.vision_model,
                    "document_readiness": readiness.to_dict(),
                    "active_revision_before_stage": (
                        existing_row["active_revision_id"] if existing_row else None
                    ),
                }
            )
            if existing_row:
                # Do not delete the old canonical rows. They remain queryable
                # through the old active pointer until the local projections
                # for this revision are ready and activate_revision() commits.
                connection.execute(
                    """
                    UPDATE documents SET
                        file_id = ?, folder_id = ?, source_path = ?, filename = ?,
                        file_type = ?, title = ?, doc_type = ?, language = ?,
                        content_hash = ?, modified_at = ?, indexed_at = ?,
                        chunk_count = ?, table_count = ?, figure_count = ?,
                        parse_status = 'parsed', index_status = 'staged',
                        parser_name = ?, parser_version = ?, page_count = ?,
                        updated_at = ?, raw_content_hash = ?, revision_status = 'staged',
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        file_id, folder_id, source_path, path.name, file_type,
                        card.title_guess, card.doc_type, card.language,
                        content_hash, modified_at, now, len(chunks), len(tables),
                        len(figures), parsed.parser_name, parsed.parser_version,
                        parsed.page_count, now, raw_content_hash, metadata_json,
                        document_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, file_id, folder_id, source_path, filename, file_type,
                        title, doc_type, language, content_hash, modified_at, indexed_at,
                        chunk_count, table_count, figure_count, parse_status, index_status,
                        parser_name, parser_version, page_count, updated_at,
                        raw_content_hash, active_revision_id, revision_status, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id, file_id, folder_id, source_path, path.name, file_type,
                        card.title_guess, card.doc_type, card.language, content_hash,
                        modified_at, now, len(chunks), len(tables), len(figures),
                        "parsed", "staged", parsed.parser_name, parsed.parser_version,
                        parsed.page_count, now, raw_content_hash, None, "staged", metadata_json,
                    ),
                )
            # A staged revision owns its own navigation card.  The active
            # document card is promoted atomically with activate_revision()
            # after the local vector projection validates.
            self._upsert_revision_card(connection, document_id, revision_id, card, now)
            connection.execute(
                """
                INSERT OR IGNORE INTO document_revisions(
                    revision_id, document_id, raw_content_hash, extracted_content_hash,
                    parser_fingerprint, chunker_fingerprint, artifact_fingerprint,
                    materialization_fingerprint, status, created_at, activated_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'staged', ?, NULL, ?)
                """,
                (
                    revision_id,
                    document_id,
                    raw_content_hash,
                    content_hash,
                    parser_fingerprint,
                    chunker_fingerprint,
                    artifact_fingerprint,
                    f"{document_id}:{materialization_fingerprint}",
                    now,
                    json.dumps({"source": "indexing_service", "materialization_policy": policy}),
                ),
            )
            if existing_revision_id == revision_id:
                # A forced parser/chunker rebuild with identical materialized
                # identity replaces that revision in place. This preserves
                # citation IDs while avoiding duplicate primary keys; a real
                # byte/fingerprint change gets a distinct revision and keeps
                # the prior rows intact.
                connection.execute(
                    """
                    DELETE FROM rag_chunks_fts
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE document_id = ? AND revision_id = ?
                    )
                    """,
                    (document_id, revision_id),
                )
                connection.execute(
                    "DELETE FROM rag_chunks_fts_v2 WHERE document_id = ? AND revision_id = ?",
                    (document_id, revision_id),
                )
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ? AND revision_id = ?",
                    (document_id, revision_id),
                )
                connection.execute(
                    "DELETE FROM document_tables WHERE document_id = ? AND revision_id = ?",
                    (document_id, revision_id),
                )
                connection.execute(
                    "DELETE FROM document_figures WHERE document_id = ? AND revision_id = ?",
                    (document_id, revision_id),
                )
            # The prior revision is retired only by activate_revision(), after
            # its replacement has passed local projection validation.
            self._ensure_document_source(
                connection,
                document_id=document_id,
                folder_id=folder_id,
                source_path=source_path,
                raw_content_hash=raw_content_hash,
                now=now,
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO document_stage_receipts(
                    revision_id, stage, fingerprint, status, finished_at,
                    metadata_json
                ) VALUES (?, 'canonical', ?, 'complete', ?, ?)
                """,
                # Keep the receipt byte-for-byte identical to the revision's
                # stored materialization fingerprint.  That fingerprint is
                # document-scoped (``document_id:<content fingerprint>``),
                # while the local variable above is the content hash alone.
                # Mixing the two makes a healthy canonical stage appear stale.
                (
                    revision_id,
                    f"{document_id}:{materialization_fingerprint}",
                    now,
                    '{"provider":null}',
                ),
            )
            for linked_collection_id in dict.fromkeys(
                [*existing_collection_ids, *([collection_id] if collection_id else [])]
            ):
                self._link_document_to_collection(
                    connection,
                    linked_collection_id,
                    document_id,
                    now,
                )

            self._insert_text_chunks(
                connection,
                chunks=chunks,
                document_id=document_id,
                file_id=file_id,
                source_path=source_path,
                filename=path.name,
                file_type=file_type,
                content_hash=content_hash,
                document_title=card.title_guess,
                document_summary=card.short_summary,
                revision_id=revision_id,
                now=now,
            )
            self._insert_page_label_rows(
                connection,
                pages=parsed.pages or [],
                existing_chunks=chunks,
                document_id=document_id,
                file_id=file_id,
                source_path=source_path,
                filename=path.name,
                revision_id=revision_id,
                now=now,
            )

            for table in tables:
                connection.execute(
                    """
                    INSERT INTO document_tables (
                        id, document_id, file_id, table_index, page_number, caption,
                        markdown, row_count, column_count, extraction_method,
                        bbox_json, created_at, revision_id, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_table_id(document_id, table, revision_id=revision_id),
                        document_id,
                        file_id,
                        table.table_index,
                        table.page_number,
                        table.caption,
                        table.markdown,
                        table.row_count,
                        table.column_count,
                        table.extraction_method,
                        None,
                        now,
                        revision_id,
                        json.dumps({"source": "parser"}),
                    ),
                )

            for figure in figures:
                figure_metadata = {"source": "parser", **(figure.metadata or {})}
                connection.execute(
                    """
                    INSERT INTO document_figures (
                        id, document_id, file_id, figure_index, page_number, caption,
                        image_path, visual_summary, extraction_method, bbox_json,
                        created_at, revision_id, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_figure_id(document_id, figure, revision_id=revision_id),
                        document_id,
                        file_id,
                        figure.figure_index,
                        figure.page_number,
                        figure.caption,
                        figure.image_path,
                        figure.visual_summary,
                        figure.extraction_method,
                        json.dumps(figure.bbox) if figure.bbox else None,
                        now,
                        revision_id,
                        json.dumps(figure_metadata),
                    ),
                )

            if self.paper_evidence_card_build_enabled:
                # Indexing itself stays provider-free/foreground-safe. The
                # durable pending row is consumed only by an explicit build or
                # backfill call after the canonical transaction commits.
                connection.execute(
                    """
                    INSERT INTO paper_evidence_build_jobs (
                        document_id, status, attempt_count, requested_at,
                        schema_version, prompt_version, generator_model,
                        metadata_json
                    ) VALUES (?, 'pending', 0, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        status = 'pending',
                        requested_at = excluded.requested_at,
                        error_message = NULL,
                        schema_version = excluded.schema_version,
                        prompt_version = excluded.prompt_version,
                        generator_model = excluded.generator_model,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        document_id,
                        now,
                        self.paper_evidence_card_schema_version,
                        self.paper_evidence_card_prompt_version,
                        self.paper_evidence_card_model,
                        json.dumps({"source": "canonical_index"}),
                    ),
                )

        if artifact_staging_id != document_id:
            artifact_backup = self._promote_staged_artifacts(
                staging_document_id=artifact_staging_id,
                document_id=document_id,
            )
            try:
                self._rewrite_figure_artifact_paths(
                    document_id=document_id,
                    old_root=self._artifact_dir(artifact_staging_id),
                    new_root=self._artifact_dir(document_id),
                )
            except Exception:
                self._rollback_artifact_promotion(
                    staging_document_id=artifact_staging_id,
                    document_id=document_id,
                    backup_path=artifact_backup,
                )
                raise
            if artifact_backup is not None:
                shutil.rmtree(artifact_backup, ignore_errors=True)
        self._write_artifact_manifest(
            document_id=document_id,
            path=path,
            parsed=parsed,
            chunks_count=len(chunks),
            tables_count=len(tables),
            figures_count=len(figures),
            indexed_at=now,
        )
        if return_existing:
            self.activate_revision(document_id, revision_id)
        return IndexedDocument(
            id=document_id,
            file_id=file_id,
            source_path=source_path,
            filename=path.name,
            file_type=file_type,
            chunk_count=len(chunks),
            revision_id=revision_id,
            previous_revision_id=existing_revision_id,
        )

    def _insert_text_chunks(
        self,
        connection,
        *,
        chunks,
        document_id: str,
        file_id: str | None,
        source_path: str,
        filename: str,
        file_type: str,
        content_hash: str,
        document_title: str,
        document_summary: str,
        revision_id: str | None = None,
        now: str,
    ) -> None:
        for index, chunk in enumerate(chunks):
            chunk_id = _stable_chunk_id(document_id, chunk, index, revision_id=revision_id)
            parent_chunk_id = _stable_parent_chunk_id(document_id, chunk, revision_id=revision_id)
            heading_path = _heading_path_for_chunk(
                file_type,
                chunk.text,
                chunk.section_title,
            )
            context_prefix = _chunk_context_prefix(
                filename=filename,
                page_number=chunk.page_number,
                page_label=getattr(chunk, "page_label", None),
                section_title=chunk.section_title,
                document_title=document_title,
                document_summary=document_summary,
            )
            metadata = {
                "content_hash": content_hash,
                "indexed_at": now,
                "file_type": file_type,
                "page_number": chunk.page_number,
                "page_label": getattr(chunk, "page_label", None),
                "section_title": chunk.section_title,
                "context_prefix": context_prefix,
                "embedding_context_prefix": context_prefix,
                "chunking_strategy": "token_structural_parent_child_v1",
                "tokenizer": "unicode_lexical_v1",
                "parent_index": chunk.parent_index,
                "parent_content": chunk.parent_text,
                "parent_token_count": chunk.parent_token_count,
                "child_index": chunk.child_index,
            }
            lexical_content = searchable_text(chunk.text)
            page_label = getattr(chunk, "page_label", None)
            if page_label and str(page_label) != str(chunk.page_number):
                lexical_content = f"{lexical_content}\nPrinted page: {page_label}."
            connection.execute(
                """
                INSERT INTO chunks (
                    id, document_id, file_id, chunk_index, content, source_path, filename,
                    chunk_type, heading_path_json, token_count, char_count,
                    order_index, parent_chunk_id, page_number, created_at, revision_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    document_id,
                    file_id,
                    index,
                    chunk.text,
                    source_path,
                    filename,
                    "text",
                    json.dumps(heading_path),
                    (
                        chunk.token_count
                        if chunk.token_count is not None
                        else len(chunk.text.split())
                    ),
                    len(chunk.text),
                    index,
                    parent_chunk_id,
                    chunk.page_number,
                    now,
                    revision_id,
                    json.dumps(metadata),
                ),
            )
            connection.execute(
                """
                INSERT INTO rag_chunks_fts (
                    chunk_id, document_id, source_path, filename, content
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, source_path, filename, lexical_content),
            )
            connection.execute(
                """
                INSERT INTO rag_chunks_fts_v2(
                    chunk_id, document_id, revision_id, source_path, filename, content
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, revision_id or "", source_path, filename, lexical_content),
            )

    def _insert_page_label_rows(
        self,
        connection,
        *,
        pages,
        existing_chunks,
        document_id: str,
        file_id: str | None,
        source_path: str,
        filename: str,
        revision_id: str,
        now: str,
    ) -> None:
        """Persist labels for pages that have no text chunk.

        Image-only pages are deliberately not embedded or added to FTS.  A
        lightweight ``page`` row is nevertheless needed so explicit printed
        page hints can resolve through the same active-revision contract as
        text pages.  This row is canonical provenance, not searchable text.
        """

        represented_pages = {
            int(getattr(chunk, "page_number"))
            for chunk in existing_chunks
            if getattr(chunk, "page_number", None) is not None
        }
        next_index = len(existing_chunks)
        for page in pages:
            page_number = getattr(page, "page_number", None)
            page_label = getattr(page, "page_label", None)
            if page_number is None or not page_label:
                continue
            page_number = int(page_number)
            if page_number in represented_pages:
                continue
            page_id = f"page:{document_id}:{revision_id}:{page_number}"
            metadata = json.dumps(
                {
                    "source": "parser_page_label",
                    "page_number": page_number,
                    "page_label": str(page_label),
                    "text_origin": getattr(page, "text_origin", "native"),
                    "canonical_page_provenance": True,
                },
                ensure_ascii=False,
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    id, document_id, file_id, chunk_index, content, source_path,
                    filename, chunk_type, heading_path_json, token_count,
                    char_count, order_index, parent_chunk_id, page_number,
                    created_at, revision_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'page', ?, 0, 0, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    document_id,
                    file_id,
                    next_index,
                    "",
                    source_path,
                    filename,
                    "[]",
                    next_index,
                    page_number,
                    now,
                    revision_id,
                    metadata,
                ),
            )
            represented_pages.add(page_number)
            next_index += 1

    def _delete_document(self, connection, document_id: str) -> None:
        connection.execute("DELETE FROM fts_document_cards WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_cards WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_card_revisions WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM collection_documents WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_tables WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_figures WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM rag_chunks_fts WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM rag_chunks_fts_v2 WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def materialization_policy(self, file_type: str) -> dict:
        """Describe extraction choices that can change canonical evidence.

        Model/provider addresses and request timeouts are operational settings;
        OCR enablement and budgets change which pages have searchable text and
        must create a distinct revision even when source bytes are unchanged.
        """

        policy = {
            "chunking_version": CHUNKING_VERSION,
            "artifact_extraction_version": ARTIFACT_EXTRACTION_VERSION,
            "vision_provider": self.vision_provider if self.vision_model else None,
            "vision_model": self.vision_model,
        }
        if self.canonical_artifact_fingerprint is not None:
            policy["canonical_artifact_fingerprint"] = self.canonical_artifact_fingerprint
        if file_type == "pdf":
            policy.update({
                "enable_pdf_ocr": self.enable_pdf_ocr,
                "pdf_ocr_page_budget": self.pdf_ocr_page_budget,
                "pdf_page_label_ocr_budget": self.pdf_page_label_ocr_budget,
            })
        return policy

    def _page_ocr_is_outdated(self, raw_content_hash: str) -> bool:
        """The document's scanned pages were OCRed by an older page-OCR
        implementation (its OCR cache carries another schema). Only documents
        that needed page OCR are rebuilt; native PDFs are untouched."""
        if not raw_content_hash or self.artifact_root is None:
            return False
        cache = Path(self.artifact_root) / "_ocr_cache" / f"{raw_content_hash}.json"
        if not cache.is_file():
            return False
        try:
            schema = json.loads(cache.read_text(encoding="utf-8")).get("schema_version")
        except (OSError, ValueError):
            return True
        return schema != PAGE_OCR_SCHEMA

    def _needs_reindex_for_page_metadata(self, connection, document_id: str) -> bool:
        row = connection.execute(
            """
            SELECT file_type, figure_count, metadata_json, raw_content_hash
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        # DOCX has ordered table/image extraction and therefore participates
        # in the same parser/chunker/artifact version contract as PDF/MD/TXT.
        # Excluding it makes a parser upgrade silently reuse stale canonical
        # projections when the raw bytes are unchanged.
        if row is None or row["file_type"] not in {"txt", "md", "pdf", "docx"}:
            return False

        metadata = json.loads(row["metadata_json"] or "{}")
        if metadata.get("materialization_policy") != self.materialization_policy(row["file_type"]):
            return True
        if row["file_type"] == "pdf" and self._page_ocr_is_outdated(str(row["raw_content_hash"] or "")):
            return True
        if metadata.get("page_chunk_version") != CHUNKING_VERSION:
            return True
        if metadata.get("artifact_extraction_version") != ARTIFACT_EXTRACTION_VERSION:
            return True
        if (
            row["file_type"] == "pdf"
            and int(row["figure_count"] or 0) > 0
            and self.vision_model
            and (
                metadata.get("vision_provider") != self.vision_provider
                or metadata.get("vision_model") != self.vision_model
            )
        ):
            return True

        missing = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM chunks
            WHERE document_id = ? AND page_number IS NULL
            """,
            (document_id,),
        ).fetchone()
        return bool(missing and missing["count"])

    def _parse_for_index(
        self,
        path: Path,
        file_type: str,
        document_id: str,
        parsed_for_hash,
        card: DocumentCardDraft,
    ):
        if file_type not in {"pdf", "docx"} or self.artifact_root is None:
            return parsed_for_hash

        try:
            return parse_document(
                path,
                artifact_dir=self._artifact_dir(document_id),
                vision_summarizer=self._vision_summarizer(path=path, card=card),
                enable_pdf_ocr=self.enable_pdf_ocr,
                pdf_ocr_page_budget=self.pdf_ocr_page_budget,
                pdf_page_label_ocr_budget=self.pdf_page_label_ocr_budget,
            )
        except Exception:
            self._delete_artifacts(document_id)
            raise

    def _artifact_dir(self, document_id: str) -> Path:
        root = self.artifact_root or self.db_path.parent / "artifacts"
        return root / document_id

    def _delete_artifacts(self, document_id: str) -> None:
        if self.artifact_root is None:
            return
        shutil.rmtree(self._artifact_dir(document_id), ignore_errors=True)

    def _promote_staged_artifacts(
        self,
        *,
        staging_document_id: str,
        document_id: str,
    ) -> Path | None:
        staging = self._artifact_dir(staging_document_id)
        target = self._artifact_dir(document_id)
        staging.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4()}"
            target.replace(backup)
        try:
            staging.replace(target)
        except Exception:
            if backup is not None and backup.exists():
                backup.replace(target)
            raise
        return backup

    def _rollback_artifact_promotion(
        self,
        *,
        staging_document_id: str,
        document_id: str,
        backup_path: Path | None,
    ) -> None:
        staging = self._artifact_dir(staging_document_id)
        target = self._artifact_dir(document_id)
        if target.exists() and not staging.exists():
            target.replace(staging)
        if backup_path is not None and backup_path.exists():
            backup_path.replace(target)

    def _rewrite_figure_artifact_paths(
        self,
        *,
        document_id: str,
        old_root: Path,
        new_root: Path,
    ) -> None:
        old_prefix = str(old_root)
        new_prefix = str(new_root)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT id, image_path, metadata_json FROM document_figures WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            for row in rows:
                metadata = json.loads(row["metadata_json"] or "{}")
                rewritten_metadata = _replace_artifact_root(
                    metadata,
                    old_prefix=old_prefix,
                    new_prefix=new_prefix,
                )
                image_path = _replace_artifact_root(
                    row["image_path"],
                    old_prefix=old_prefix,
                    new_prefix=new_prefix,
                )
                connection.execute(
                    "UPDATE document_figures SET image_path = ?, metadata_json = ? WHERE id = ?",
                    (image_path, json.dumps(rewritten_metadata), row["id"]),
                )

    def _write_artifact_manifest(
        self,
        *,
        document_id: str,
        path: Path,
        parsed,
        chunks_count: int,
        tables_count: int,
        figures_count: int,
        indexed_at: str,
    ) -> None:
        if self.artifact_root is None:
            return
        artifact_dir = self._artifact_dir(document_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "document_id": document_id,
            "filename": path.name,
            "source_path": str(path),
            "parser_name": parsed.parser_name,
            "parser_version": parsed.parser_version,
            "page_count": parsed.page_count,
            "chunk_count": chunks_count,
            "table_count": tables_count,
            "figure_count": figures_count,
            "chunking_version": CHUNKING_VERSION,
            "artifact_extraction_version": ARTIFACT_EXTRACTION_VERSION,
            "indexed_at": indexed_at,
            "visual_quality": _visual_quality_report(parsed.figures or []),
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _vision_summarizer(self, *, path: Path, card: DocumentCardDraft):
        if not self.vision_model:
            return None
        if self.vision_provider == "openai_compatible":
            client = OpenAICompatibleVisionSummarizer(
                base_url=self.openai_api_base,
                api_key=self.openai_api_key or "any",
                model=self.vision_model,
                timeout_seconds=max(self.request_timeout_seconds, 300.0),
            )
        elif self.vision_provider == "ollama":
            client = OllamaVisionSummarizer(
                host=self.ollama_host,
                model=self.vision_model,
                timeout_seconds=self.request_timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported VISION_PROVIDER: {self.vision_provider}")

        def summarize(
            image_path,
            caption,
            page_text,
            *,
            page_number=None,
            page_image_path=None,
            extraction_method=None,
        ):
            return client.summarize_image_context(
                image_path,
                caption=caption,
                page_text=page_text,
                extraction_method=extraction_method,
                document_context=FigureDocumentContext(
                    filename=path.name,
                    title=card.title_guess,
                    summary=card.short_summary,
                    page_number=page_number,
                    nearby_text=page_text,
                ),
                page_image_path=page_image_path,
            )

        return summarize

    def _get_indexed_document_by_path(self, source_path: str) -> IndexedDocument | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, file_id, source_path, filename, file_type, chunk_count
                FROM documents
                WHERE source_path = ?
                """,
                (source_path,),
            ).fetchone()
        if row is None:
            return None
        return IndexedDocument(**dict(row))

    def _upsert_approved_folder(
        self,
        folder: Path,
        access_level: str,
        recursive: bool,
        now: str,
    ) -> str:
        folder_path = str(folder)
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM approved_folders WHERE folder_path = ?",
                (folder_path,),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE approved_folders
                    SET access_level = ?, recursive = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (access_level, int(recursive), now, existing["id"]),
                )
                return existing["id"]

            folder_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO approved_folders (
                    id, folder_path, access_level, recursive, watch_enabled,
                    created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    folder_id,
                    folder_path,
                    access_level,
                    int(recursive),
                    now,
                    now,
                    json.dumps({"source": "indexing_service"}),
                ),
            )
            return folder_id

    def _upsert_file(self, path: Path, file_type: str, now: str) -> str:
        stat = path.stat()
        source_path = str(path)
        file_hash = _hash_file(path)
        mime_type, _ = mimetypes.guess_type(source_path)
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM files WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if existing:
                file_id = existing["id"]
                connection.execute(
                    """
                    UPDATE files
                    SET filename = ?, extension = ?, mime_type = ?, size_bytes = ?,
                        modified_at = ?, content_hash = ?, parent_folder = ?,
                        approved = 1, last_seen_at = ?, is_supported = 1, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        path.name,
                        file_type,
                        mime_type,
                        stat.st_size,
                        datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                        file_hash,
                        str(path.parent),
                        now,
                        json.dumps({"source": "indexing_service"}),
                        file_id,
                    ),
                )
            else:
                file_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO files (
                        id, source_path, filename, extension, mime_type, size_bytes,
                        created_at, modified_at, content_hash, parent_folder, approved,
                        last_seen_at, is_supported, risk_level, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 1, 'normal', ?)
                    """,
                    (
                        file_id,
                        source_path,
                        path.name,
                        file_type,
                        mime_type,
                        stat.st_size,
                        datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                        datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                        file_hash,
                        str(path.parent),
                        now,
                        json.dumps({"source": "indexing_service"}),
                    ),
                )
            self._sync_file_fts(connection, file_id, path, file_type)
        return file_id

    def _upsert_collection(
        self,
        *,
        name: str,
        collection_type: str,
        scope_type: str,
        scope_id: str | None,
        metadata: dict,
        now: str,
    ) -> str:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                """
                SELECT id FROM collections
                WHERE name = ? AND type = ? AND scope_type = ? AND COALESCE(scope_id, '') = COALESCE(?, '')
                """,
                (name, collection_type, scope_type, scope_id),
            ).fetchone()
            if existing:
                connection.execute(
                    "UPDATE collections SET updated_at = ?, metadata_json = ? WHERE id = ?",
                    (now, json.dumps(metadata), existing["id"]),
                )
                return existing["id"]

            collection_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO collections (
                    id, name, type, scope_type, scope_id, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection_id,
                    name,
                    collection_type,
                    scope_type,
                    scope_id,
                    now,
                    now,
                    json.dumps(metadata),
                ),
            )
            return collection_id

    def _link_document_to_collection(
        self,
        connection,
        collection_id: str,
        document_id: str,
        now: str,
        role: str = "reference",
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO collection_documents (
                collection_id, document_id, added_at, role, metadata_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                collection_id,
                document_id,
                now,
                role,
                json.dumps({"source": "indexing_service"}),
            ),
        )

    def _ensure_document_source(
        self,
        connection,
        *,
        document_id: str,
        folder_id: str | None,
        source_path: str,
        raw_content_hash: str,
        now: str,
    ) -> None:
        """Upsert a path alias without creating a second canonical document."""

        file_row = connection.execute(
            "SELECT size_bytes, mime_type FROM files WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        size_bytes = int(file_row["size_bytes"] or 0) if file_row else 0
        mime_type = file_row["mime_type"] if file_row else None
        connection.execute(
            """
            INSERT INTO content_blobs(
                raw_content_hash, size_bytes, mime_type, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(raw_content_hash) DO UPDATE SET
                size_bytes = MAX(content_blobs.size_bytes, excluded.size_bytes),
                mime_type = COALESCE(content_blobs.mime_type, excluded.mime_type)
            """,
            (raw_content_hash, size_bytes, mime_type, now, '{"source":"indexing_service"}'),
        )
        source_id = f"source:{hashlib.sha256(source_path.encode('utf-8')).hexdigest()[:32]}"
        connection.execute(
            """
            INSERT INTO document_sources(
                id, document_id, folder_id, source_path, raw_content_hash,
                status, first_seen_at, last_seen_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                document_id = excluded.document_id,
                folder_id = excluded.folder_id,
                raw_content_hash = excluded.raw_content_hash,
                status = 'active',
                last_seen_at = excluded.last_seen_at,
                metadata_json = excluded.metadata_json
            """,
            (
                source_id,
                document_id,
                folder_id,
                source_path,
                raw_content_hash,
                now,
                now,
                '{"source":"indexing_service"}',
            ),
        )

    def _upsert_document_card(
        self,
        connection,
        document_id: str,
        card: DocumentCardDraft,
        now: str,
    ) -> None:
        card_id = str(uuid.uuid4())
        topic_tags = json.dumps(card.topic_tags)
        project_tags = json.dumps(card.project_tags)
        keywords = json.dumps(card.keywords)
        connection.execute(
            """
            INSERT INTO document_cards (
                id, document_id, title_guess, doc_type, language, short_summary,
                topic_tags_json, project_tags_json, people_json, orgs_json,
                dates_json, keywords_json, importance_score, confidence,
                should_deep_index, created_at, updated_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', '[]', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title_guess = excluded.title_guess,
                doc_type = excluded.doc_type,
                language = excluded.language,
                short_summary = excluded.short_summary,
                topic_tags_json = excluded.topic_tags_json,
                project_tags_json = excluded.project_tags_json,
                keywords_json = excluded.keywords_json,
                importance_score = excluded.importance_score,
                confidence = excluded.confidence,
                should_deep_index = excluded.should_deep_index,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                card_id,
                document_id,
                card.title_guess,
                card.doc_type,
                card.language,
                card.short_summary,
                topic_tags,
                project_tags,
                keywords,
                card.importance_score,
                card.confidence,
                int(card.should_deep_index),
                now,
                now,
                json.dumps({"source": "document_card_builder"}),
            ),
        )
        connection.execute("DELETE FROM fts_document_cards WHERE document_id = ?", (document_id,))
        connection.execute(
            """
            INSERT INTO fts_document_cards (
                document_id, title, short_summary, topic_tags, project_tags, keywords
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                card.title_guess,
                card.short_summary,
                " ".join(card.topic_tags),
                " ".join(card.project_tags),
                " ".join(card.keywords),
            ),
        )

    def _upsert_revision_card(
        self,
        connection,
        document_id: str,
        revision_id: str,
        card: DocumentCardDraft,
        now: str,
    ) -> None:
        """Persist a card for a staged revision without touching active routing."""

        connection.execute(
            """
            INSERT INTO document_card_revisions(
                id, document_id, revision_id, title_guess, doc_type, language,
                short_summary, topic_tags_json, project_tags_json, people_json,
                orgs_json, dates_json, keywords_json, importance_score,
                confidence, should_deep_index, created_at, updated_at,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', '[]', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, revision_id) DO UPDATE SET
                title_guess = excluded.title_guess,
                doc_type = excluded.doc_type,
                language = excluded.language,
                short_summary = excluded.short_summary,
                topic_tags_json = excluded.topic_tags_json,
                project_tags_json = excluded.project_tags_json,
                keywords_json = excluded.keywords_json,
                importance_score = excluded.importance_score,
                confidence = excluded.confidence,
                should_deep_index = excluded.should_deep_index,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                f"card-rev:{document_id}:{revision_id}",
                document_id,
                revision_id,
                card.title_guess,
                card.doc_type,
                card.language,
                card.short_summary,
                json.dumps(card.topic_tags, ensure_ascii=False),
                json.dumps(card.project_tags, ensure_ascii=False),
                json.dumps(card.keywords, ensure_ascii=False),
                card.importance_score,
                card.confidence,
                int(card.should_deep_index),
                now,
                now,
                json.dumps({"source": "document_card_builder", "revision_id": revision_id}),
            ),
        )

    def _promote_revision_card(
        self,
        connection,
        document_id: str,
        revision_id: str,
        now: str,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM document_card_revisions WHERE document_id = ? AND revision_id = ?",
            (document_id, revision_id),
        ).fetchone()
        if row is None:
            return
        connection.execute(
            """
            INSERT INTO document_cards(
                id, document_id, title_guess, doc_type, language, short_summary,
                topic_tags_json, project_tags_json, people_json, orgs_json,
                dates_json, keywords_json, importance_score, confidence,
                should_deep_index, created_at, updated_at, revision_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title_guess = excluded.title_guess,
                doc_type = excluded.doc_type,
                language = excluded.language,
                short_summary = excluded.short_summary,
                topic_tags_json = excluded.topic_tags_json,
                project_tags_json = excluded.project_tags_json,
                people_json = excluded.people_json,
                orgs_json = excluded.orgs_json,
                dates_json = excluded.dates_json,
                keywords_json = excluded.keywords_json,
                importance_score = excluded.importance_score,
                confidence = excluded.confidence,
                should_deep_index = excluded.should_deep_index,
                updated_at = excluded.updated_at,
                revision_id = excluded.revision_id,
                metadata_json = excluded.metadata_json
            """,
            (
                f"card:{document_id}",
                document_id,
                row["title_guess"],
                row["doc_type"],
                row["language"],
                row["short_summary"],
                row["topic_tags_json"],
                row["project_tags_json"],
                row["people_json"],
                row["orgs_json"],
                row["dates_json"],
                row["keywords_json"],
                row["importance_score"],
                row["confidence"],
                row["should_deep_index"],
                row["created_at"] or now,
                now,
                revision_id,
                row["metadata_json"],
            ),
        )
        connection.execute("DELETE FROM fts_document_cards WHERE document_id = ?", (document_id,))
        connection.execute(
            """
            INSERT INTO fts_document_cards(
                document_id, title, short_summary, topic_tags, project_tags, keywords
            )
            SELECT document_id, title_guess, short_summary, topic_tags_json,
                   project_tags_json, keywords_json
            FROM document_cards WHERE document_id = ?
            """,
            (document_id,),
        )
        connection.execute(
            "DELETE FROM document_card_revisions WHERE document_id = ? AND revision_id <> ?",
            (document_id, revision_id),
        )
    def _sync_file_fts(self, connection, file_id: str, path: Path, file_type: str) -> None:
        connection.execute("DELETE FROM fts_files WHERE file_id = ?", (file_id,))
        connection.execute(
            """
            INSERT INTO fts_files (file_id, filename, source_path, extension)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, path.name, str(path), file_type),
        )


def _reconstruct_parsed_document(document, rows) -> ParsedDocument:
    page_parts: dict[int | None, list[str]] = {}
    page_order: list[int | None] = []
    page_sections: dict[int | None, str | None] = {}
    page_labels: dict[int, str] = {}
    for row in rows:
        page_number = row["page_number"]
        if page_number not in page_parts:
            page_parts[page_number] = []
            page_order.append(page_number)
            page_sections[page_number] = None
        metadata = json.loads(row["metadata_json"] or "{}")
        if page_number is not None and metadata.get("page_label"):
            page_labels[int(page_number)] = str(metadata["page_label"])
        section = str(metadata.get("section_title") or "").strip() or None
        content = str(row["content"] or "").strip()
        if not content:
            continue
        previous_section = page_sections[page_number]
        if (
            section
            and section != previous_section
            and section.casefold() not in content[:200].casefold()
        ):
            page_parts[page_number].append(section)
        if page_parts[page_number]:
            page_parts[page_number][-1] = _merge_overlapping_text(
                page_parts[page_number][-1],
                content,
            )
        else:
            page_parts[page_number].append(content)
        page_sections[page_number] = section

    page_texts = {
        page_number: "\n\n".join(part for part in parts if part).strip()
        for page_number, parts in page_parts.items()
    }
    numbered_pages = [
        ParsedPage(
            page_number=int(page_number),
            text=page_texts[page_number],
            page_label=page_labels.get(int(page_number)),
        )
        for page_number in page_order
        if page_number is not None and page_texts.get(page_number)
    ]
    full_text = "\n\n".join(
        page_texts[page_number]
        for page_number in page_order
        if page_texts.get(page_number)
    )
    return ParsedDocument(
        text=full_text,
        parser_name=str(document["parser_name"] or "canonical_reconstruction"),
        parser_version=str(document["parser_version"] or "1"),
        page_count=document["page_count"],
        pages=numbered_pages or None,
        tables=[],
        figures=[],
    )


def _merge_overlapping_text(left: str, right: str, *, max_overlap: int = 600) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    limit = min(max_overlap, len(left), len(right))
    for size in range(limit, 19, -1):
        if left[-size:] == right[:size]:
            return f"{left}{right[size:]}"
    return f"{left}\n\n{right}"


def _heading_path_for_chunk(file_type: str, chunk: str, section_title: str | None = None) -> list[str]:
    if section_title:
        return [section_title]
    if file_type != "md":
        return []

    headings: list[str] = []
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        marker, _, title = stripped.partition(" ")
        if marker and set(marker) == {"#"} and title.strip():
            level = len(marker)
            headings = headings[: level - 1]
            headings.append(title.strip())
    return headings


def _chunk_context_prefix(
    *,
    filename: str,
    page_number: int | None,
    page_label: str | None = None,
    section_title: str | None,
    document_title: str | None = None,
    document_summary: str | None = None,
) -> str:
    parts = [f"Document: {filename}."]
    clean_title = " ".join((document_title or "").split())
    if clean_title and clean_title.casefold() != filename.casefold():
        parts.append(f"Title: {clean_title[:200]}.")
    if page_number is not None:
        parts.append(f"Page: {page_number}.")
    if page_label and str(page_label) != str(page_number):
        parts.append(f"Printed page: {page_label}.")
    if section_title:
        parts.append(f"Section: {section_title}.")
    clean_summary = " ".join((document_summary or "").split())
    if clean_summary:
        parts.append(f"Paper context: {clean_summary[:280]}.")
    return " ".join(parts)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_parent_chunk_id(document_id: str, chunk, *, revision_id: str | None = None) -> str | None:
    parent_text = getattr(chunk, "parent_text", None)
    if not parent_text:
        return None
    identity = {
        "page_number": getattr(chunk, "page_number", None),
        "section_title": getattr(chunk, "section_title", None),
        "parent_index": getattr(chunk, "parent_index", None),
        "content_hash": hashlib.sha256(parent_text.encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    revision_key = revision_id or "legacy"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"my-agent:chunk-parent:{document_id}:{revision_key}:{raw}"))


def _stable_chunk_id(document_id: str, chunk, chunk_index: int, *, revision_id: str | None = None) -> str:
    identity = {
        "page_number": getattr(chunk, "page_number", None),
        "section_title": getattr(chunk, "section_title", None),
        "parent_index": getattr(chunk, "parent_index", None),
        "child_index": getattr(chunk, "child_index", None),
        "chunk_index": chunk_index,
        "content_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    revision_key = revision_id or "legacy"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"my-agent:chunk:{document_id}:{revision_key}:{raw}"))


def _stable_figure_id(document_id: str, figure, *, revision_id: str | None = None) -> str:
    metadata = getattr(figure, "metadata", None) or {}
    # Labels/group IDs are useful semantic anchors but are not guaranteed to
    # be unique: OCR/caption fallback can emit ``Figure 1`` on more than one
    # crop/page. Keep the parser's stable ordinal in the identity envelope so
    # two valid visual artifacts can never collide on the SQLite primary key.
    logical_identity = {
        "anchor": (
            metadata.get("logical_group_id")
            or metadata.get("figure_label")
            or metadata.get("image_hash")
            or {
                "page_number": getattr(figure, "page_number", None),
                "bbox": getattr(figure, "bbox", None),
                "caption": getattr(figure, "caption", None),
                "extraction_method": getattr(figure, "extraction_method", None),
            }
        ),
        "figure_index": getattr(figure, "figure_index", None),
    }
    raw = json.dumps(logical_identity, ensure_ascii=False, sort_keys=True, default=str)
    revision_key = revision_id or "legacy"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"my-agent:figure:{document_id}:{revision_key}:{raw}"))


def _stable_table_id(document_id: str, table, *, revision_id: str | None = None) -> str:
    """Keep persisted table citations valid across deterministic re-indexes."""
    logical_identity = {
        "page_number": getattr(table, "page_number", None),
        "caption": getattr(table, "caption", None),
        "markdown": getattr(table, "markdown", None),
        "extraction_method": getattr(table, "extraction_method", None),
        "table_index": getattr(table, "table_index", None),
    }
    raw = json.dumps(logical_identity, ensure_ascii=False, sort_keys=True, default=str)
    revision_key = revision_id or "legacy"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"my-agent:table:{document_id}:{revision_key}:{raw}"))


def _replace_artifact_root(value, *, old_prefix: str, new_prefix: str):
    if isinstance(value, str):
        return new_prefix + value[len(old_prefix) :] if value.startswith(old_prefix) else value
    if isinstance(value, list):
        return [
            _replace_artifact_root(item, old_prefix=old_prefix, new_prefix=new_prefix)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_artifact_root(item, old_prefix=old_prefix, new_prefix=new_prefix)
            for key, item in value.items()
        }
    return value


def _visual_quality_report(figures: list) -> dict:
    by_method: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    accepted_figures = 0
    page_fallbacks = 0
    for figure in figures:
        method = getattr(figure, "extraction_method", None) or "unknown"
        by_method[method] = by_method.get(method, 0) + 1
        metadata = getattr(figure, "metadata", None) or {}
        status = str(metadata.get("quality_status") or "legacy")
        kind = str(metadata.get("asset_kind") or metadata.get("asset_type") or "figure")
        by_status[status] = by_status.get(status, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if kind == "page":
            page_fallbacks += 1
        elif getattr(figure, "image_path", None) and figure_is_indexable(
            caption=getattr(figure, "caption", None),
            extraction_method=getattr(figure, "extraction_method", None),
            metadata=metadata,
        ):
            accepted_figures += 1
    return {
        "accepted_figures": accepted_figures,
        "page_fallbacks": page_fallbacks,
        "by_method": by_method,
        "by_status": by_status,
        "by_kind": by_kind,
    }


def _decode_metadata(row: dict) -> dict:
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    return row


def _decode_card(row: dict) -> dict:
    row["topic_tags"] = json.loads(row.pop("topic_tags_json") or "[]")
    row["project_tags"] = json.loads(row.pop("project_tags_json") or "[]")
    row["people"] = json.loads(row.pop("people_json") or "[]")
    row["orgs"] = json.loads(row.pop("orgs_json") or "[]")
    row["dates"] = json.loads(row.pop("dates_json") or "[]")
    row["keywords"] = json.loads(row.pop("keywords_json") or "[]")
    row["should_deep_index"] = bool(row["should_deep_index"])
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    return row
