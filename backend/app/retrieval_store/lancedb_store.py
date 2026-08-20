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
        query = table.search(query_embedding).limit(top_k)
        where = _where(filters)
        if where:
            query = query.where(where)
        rows = query.to_list()
        return [
            RetrievalResult(
                id=row["id"],
                text=row["text"],
                score=float(row.get("_distance", 0.0)),
                source=source,  # type: ignore[arg-type]
                metadata=_row_metadata(row),
            )
            for row in rows
        ]

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
        "file_id": metadata.get("file_id") or "",
        "source_path": metadata.get("source_path") or "",
        "filename": metadata.get("filename") or "",
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _where(filters: RetrievalFilter) -> str:
    clauses: list[str] = []
    if filters.document_ids:
        clauses.append(_in("document_id", filters.document_ids))
    if filters.file_ids:
          clauses.append(_in("file_id", filters.file_ids))
    if filters.folder_path:
        clauses.append(f"source_path LIKE {_quote(filters.folder_path + '%')}")
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
    metadata.setdefault("file_id", row.get("file_id") or None)
    metadata.setdefault("source_path", row.get("source_path"))
    metadata.setdefault("filename", row.get("filename"))
    return metadata


def _table_names(db) -> list[str]:
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        return list(getattr(response, "tables", response))
    return list(db.table_names())
