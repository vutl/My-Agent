from dataclasses import dataclass
import json
import sqlite3

from app.rag.vietnamese_text import build_fts_query as _build_vn_fts_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_path: str
    filename: str
    content: str
    score: float
    chunk_index: int
    page_number: int | None
    heading_path: list[str]
    token_count: int | None
    char_count: int | None


def build_fts_query(query: str) -> str:
    return _build_vn_fts_query(query, max_tokens=10)


def search_chunks(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
    document_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    document_filter = ""
    params: list = [fts_query]
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        document_filter = f"AND rag_chunks_fts.document_id IN ({placeholders})"
        params.extend(document_ids)
    params.append(limit)

    rows = connection.execute(
        f"""
        SELECT
            rag_chunks_fts.chunk_id AS chunk_id,
            rag_chunks_fts.document_id AS document_id,
            rag_chunks_fts.source_path AS source_path,
            rag_chunks_fts.filename AS filename,
            rag_chunks_fts.content AS content,
            chunks.chunk_index AS chunk_index,
            chunks.page_number AS page_number,
            chunks.heading_path_json AS heading_path_json,
            chunks.token_count AS token_count,
            chunks.char_count AS char_count,
            bm25(rag_chunks_fts) AS score
        FROM rag_chunks_fts
        JOIN chunks ON chunks.id = rag_chunks_fts.chunk_id
        WHERE rag_chunks_fts MATCH ?
        {document_filter}
        ORDER BY score
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    results: list[RetrievedChunk] = []
    for row in rows:
        data = dict(row)
        data["heading_path"] = json.loads(data.pop("heading_path_json") or "[]")
        results.append(RetrievedChunk(**data))
    return results
