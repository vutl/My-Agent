import json
from pathlib import Path
from typing import Any

from app.retrieval_store.base import RetrievalFilter, RetrievalResult, VectorRecord


class LanceDBUnavailable(RuntimeError):
    """Raised when LanceDB is requested but the optional package is not installed."""


class LanceDBRetrievalStore:
    """Optional local LanceDB retrieval store for vector-backed RAG."""

    def __init__(self, db_dir: Path) -> None:
        try:
            import lancedb  # type: ignore
        except ImportError as exc:
            raise LanceDBUnavailable(
                "LanceDB is not installed. Install lancedb to enable vector retrieval."
            ) from exc

        self._db = lancedb.connect(str(db_dir))

    async def add_document_cards(self, records: list[VectorRecord]) -> None:
        await self._add("document_cards", records)

    async def add_text_chunks(self, records: list[VectorRecord]) -> None:
        await self._add("text_chunks", records)

    async def add_table_chunks(self, records: list[VectorRecord]) -> None:
        await self._add("table_chunks", records)

    async def add_figure_chunks(self, records: list[VectorRecord]) -> None:
        await self._add("figure_chunks", records)

    async def add_memory_chunks(self, records: list[VectorRecord]) -> None:
        await self._add("memory_chunks", records)

    def vector_dimensions(self, table_name: str = "document_cards") -> int | None:
        table = self._open_table(table_name)
        if table is None:
            return None
        vector_type = table.schema.field("vector").type
        dimensions = getattr(vector_type, "list_size", None)
        return int(dimensions) if dimensions is not None else None

    def count_document_revision(self, document_id: str, revision_id: str) -> dict[str, int] | None:
        """Return physical vector counts for one document revision.

        ``None`` is returned when an existing legacy table has no physical
        ``revision_id`` column.  Callers can then retain the migration-window
        compatibility path instead of claiming a stronger validation than the
        store can provide.  Current isolated/prod-v3 tables all carry this
        column, so activation can verify the actual rows written rather than
        trusting the in-memory insertion counts alone.
        """

        counts: dict[str, int] = {}
        for table_name, key in (
            ("document_cards", "document_cards"),
            ("text_chunks", "text_chunks"),
            ("table_chunks", "table_chunks"),
            ("figure_chunks", "figure_chunks"),
        ):
            table = self._open_table(table_name)
            if table is None:
                counts[key] = 0
                continue
            if not _has_column(table, "revision_id"):
                return None
            where = (
                f"document_id = {_quote(document_id)} AND "
                + ("(revision_id IN ('legacy', '') OR revision_id IS NULL)" if revision_id == "legacy"
                   else f"revision_id = {_quote(revision_id)}")
            )
            try:
                counts[key] = int(table.count_rows(where))
            except TypeError:
                # A few LanceDB releases expose the argument as a keyword.
                counts[key] = int(table.count_rows(filter=where))
        return counts

    def document_revision_ids(self, document_id: str, revision_id: str) -> dict[str, set[str]] | None:
        """Read physical identities, not SQLite insertion receipts, for reuse."""
        identities: dict[str, set[str]] = {}
        for table_name in ("document_cards", "text_chunks", "table_chunks", "figure_chunks"):
            table = self._open_table(table_name)
            if table is None:
                identities[table_name] = set()
                continue
            if not _has_column(table, "revision_id"):
                return None
            where = (
                f"document_id = {_quote(document_id)} AND "
                + ("(revision_id IN ('legacy', '') OR revision_id IS NULL)" if revision_id == "legacy"
                   else f"revision_id = {_quote(revision_id)}")
            )
            rows = table.search().where(where).select(["id"]).limit(
                max(1, table.count_rows(where))
            ).to_list()
            identities[table_name] = {str(row["id"]) for row in rows}
        return identities

    async def search_document_cards(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        return await self._search("document_cards", query_embedding, filters, top_k, "document_card")

    async def search_text_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        return await self._search("text_chunks", query_embedding, filters, top_k, "text_chunk")

    async def search_table_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        return await self._search("table_chunks", query_embedding, filters, top_k, "table")

    async def search_figure_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        return await self._search("figure_chunks", query_embedding, filters, top_k, "figure")

    async def delete_document(self, document_id: str) -> None:
        for table_name in ("document_cards", "text_chunks", "table_chunks", "figure_chunks"):
            table = self._open_table(table_name)
            if table is not None:
                table.delete(f"document_id = {_quote(document_id)}")

    async def delete_document_revision(self, document_id: str, revision_id: str) -> None:
        """Delete only one revision's vectors, preserving the active fallback."""
        for table_name in ("document_cards", "text_chunks", "table_chunks", "figure_chunks"):
            table = self._open_table(table_name)
            if table is not None:
                if _has_column(table, "revision_id"):
                    table.delete(
                        f"document_id = {_quote(document_id)} AND "
                        + ("(revision_id IN ('legacy', '') OR revision_id IS NULL)" if revision_id == "legacy"
                           else f"revision_id = {_quote(revision_id)}")
                    )
                else:
                    table.delete(
                        f"document_id = {_quote(document_id)} AND "
                        f"metadata_json LIKE {_quote('%\\\"revision_id\\\": \\\"' + revision_id + '\\\"%')}"
                    )

    async def prune_document_revisions(self, active_revisions: dict[str, str]) -> None:
        """Remove stale revision rows while preserving each active document slice."""
        for table_name in ("document_cards", "text_chunks", "table_chunks", "figure_chunks"):
            table = self._open_table(table_name)
            if table is None:
                continue
            if _has_column(table, "revision_id"):
                for document_id, revision_id in active_revisions.items():
                    table.delete(
                        f"document_id = {_quote(document_id)} AND "
                        f"revision_id IS NOT NULL AND revision_id <> {_quote(revision_id)}"
                    )

    async def prune_documents(self, document_ids: list[str]) -> None:
        for table_name in ("document_cards", "text_chunks", "table_chunks", "figure_chunks"):
            table = self._open_table(table_name)
            if table is not None:
                if document_ids:
                    keep_filter = _in("document_id", document_ids)
                    table.delete(f"NOT ({keep_filter})")
                else:
                    # A full sweep over an empty canonical SQLite corpus means
                    # every derived vector is stale.
                    table.delete("document_id IS NOT NULL")

    async def _add(self, table_name: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        rows = [_record_to_row(record) for record in records]
        table = self._open_table(table_name)
        if table is None:
            self._db.create_table(table_name, rows)
        else:
            if not _has_column(table, "revision_id"):
                # Existing production tables are kept readable during the
                # migration window.  New isolated tables get the physical
                # column; legacy rows retain metadata-based filtering.
                rows = [{key: value for key, value in row.items() if key != "revision_id"} for row in rows]
            self._delete_existing_ids(table, [row["id"] for row in rows])
            table.add(rows)

    async def _search(
        self,
        table_name: str,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
        source: str,
    ) -> list[RetrievalResult]:
        table = self._open_table(table_name)
        if table is None:
            return []
        physical_revision = _has_column(table, "revision_id")
        query_limit = top_k if physical_revision or not filters.active_revisions else max(top_k * 4, 50)
        query = table.search(query_embedding).limit(query_limit)
        where = _where(filters, include_revision=physical_revision)
        if where:
            query = query.where(where)
        rows = query.to_list()
        results = [
            RetrievalResult(
                id=row["id"],
                text=row["text"],
                score=float(row.get("_distance", 0.0)),
                source=source,  # type: ignore[arg-type]
                metadata=_row_metadata(row),
            )
            for row in rows
        ]
        if filters.active_revisions and not physical_revision:
            results = [
                item
                for item in results
                if _legacy_revision_matches(
                    filters.active_revisions.get(str(item.metadata.get("document_id"))),
                    str(item.metadata.get("revision_id") or ""),
                )
            ]
        return results[:top_k]

    def _open_table(self, table_name: str):
        table_names = _table_names(self._db)
        if table_name not in table_names:
            return None
        return self._db.open_table(table_name)

    def _delete_existing_ids(self, table, ids: list[str]) -> None:
        if not ids:
            return
        quoted = ", ".join(_quote(item) for item in ids)
        table.delete(f"id IN ({quoted})")


def _record_to_row(record: VectorRecord) -> dict[str, Any]:
    metadata = record.metadata or {}
    return {
        "id": record.id,
        "text": record.text,
        "vector": record.embedding,
        "document_id": metadata.get("document_id") or "",
        "revision_id": metadata.get("revision_id") or "",
        "file_id": metadata.get("file_id") or "",
        "source_path": metadata.get("source_path") or "",
        "filename": metadata.get("filename") or "",
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _legacy_revision_matches(expected: str | None, actual: str) -> bool:
    """Match pre-revision Lance rows to the migration-created initial slice.

    Existing tables have no physical ``revision_id`` column and their JSON
    metadata predates revision lifecycle.  The schema migration assigns those
    rows ``rev:<document>:initial``; accepting an empty actual value only for
    that immutable initial revision preserves search during the migration
    window while still rejecting missing/stale metadata for later revisions.
    """

    if expected is None:
        return True
    if actual == expected:
        return True
    return not actual and expected.endswith(":initial")


def _where(filters: RetrievalFilter, *, include_revision: bool = True) -> str:
    clauses: list[str] = []
    if filters.document_ids:
        clauses.append(_in("document_id", filters.document_ids))
    if filters.file_ids:
          clauses.append(_in("file_id", filters.file_ids))
    if filters.folder_path:
        clauses.append(f"source_path LIKE {_quote(filters.folder_path + '%')}")
    if include_revision and filters.active_revisions:
        revision_clauses = [
            f"(document_id = {_quote(document_id)} AND revision_id = {_quote(revision_id)})"
            for document_id, revision_id in filters.active_revisions.items()
        ]
        if revision_clauses:
            clauses.append("(" + " OR ".join(revision_clauses) + ")")
    return " AND ".join(clauses)


def _in(column: str, values: list[str]) -> str:
    quoted = ", ".join(_quote(value) for value in values)
    return f"{column} IN ({quoted})"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json")
    if isinstance(raw, str) and raw:
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}
    metadata.setdefault("document_id", row.get("document_id"))
    # Revision is a physical Lance column in current projections. Preserve it
    # in the shared metadata contract so RagService's post-query compatibility
    # guard cannot discard valid table/figure rows after the pushed filter.
    metadata.setdefault("revision_id", row.get("revision_id") or None)
    metadata.setdefault("file_id", row.get("file_id") or None)
    metadata.setdefault("source_path", row.get("source_path"))
    metadata.setdefault("filename", row.get("filename"))
    return metadata


def _table_names(db) -> list[str]:
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        return list(getattr(response, "tables", response))
    return list(db.table_names())


def _has_column(table: Any, name: str) -> bool:
    try:
        return name in set(table.schema.names)
    except Exception:
        try:
            return name in {field.name for field in table.schema}
        except Exception:
            return False
