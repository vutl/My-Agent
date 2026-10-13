from dataclasses import dataclass
import json
import re
import sqlite3

from app.rag.vietnamese_text import (
    build_fts_query as _build_vn_fts_query,
    tokenize_for_fts,
)


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
    page_label: str | None = None


def build_fts_query(query: str) -> str:
    return _build_vn_fts_query(query, max_tokens=10)


def build_fts_queries(query: str, *, max_variants: int = 3) -> list[str]:
    """Build a small lexical query frontier without changing the index.

    The old single query kept only the first ten tokens.  That is unsafe for
    natural requests containing a long filename or several document names:
    useful identifiers can occur after the first window.  Keep the original
    query for compatibility, then add bounded identifier/tail views.  These
    are lexical fallbacks only; callers still perform one semantic embedding.
    """

    normalized = " ".join(str(query or "").split())
    if not normalized or max_variants < 1:
        return []

    variants: list[str] = []

    def add(value: str, *, max_tokens: int = 10) -> None:
        built = _build_vn_fts_query(value, max_tokens=max_tokens)
        if built and built not in variants:
            variants.append(built)

    add(normalized)
    tokens = tokenize_for_fts(normalized, max_tokens=64)

    # Preserve filename-like and exact technical anchors regardless of their
    # position in the user query.  The tokenizer keeps underscores, digits and
    # internal hyphens, which are useful in FTS5 content/filename fields.
    identifier_tokens = [
        token
        for token in tokens
        if any(char.isdigit() for char in token)
        or "_" in token
        or "-" in token
        or len(token) <= 6 and token.isascii() and token.isalnum()
    ]
    if identifier_tokens:
        add(" ".join(identifier_tokens), max_tokens=min(16, len(identifier_tokens)))

    # A tail view catches the question/facet after long catalog filenames.
    if len(tokens) > 10:
        add(" ".join(tokens[10:]), max_tokens=12)

    # If punctuation split a filename into generic pieces, add a normalized
    # stem view as a final bounded fallback (e.g. ``foo_bar.pdf`` → ``foo bar``).
    stems: list[str] = []
    for raw in re.findall(r"[\w.-]+", normalized, flags=re.UNICODE):
        if any(mark in raw for mark in ("_", "-", ".")):
            stem = re.sub(r"[_\-.]+", " ", raw).strip()
            if stem:
                stems.append(stem)
    if stems:
        add(" ".join(stems), max_tokens=16)

    return variants[:max_variants]


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
        document_filter = f"AND rag_chunks_fts_v2.document_id IN ({placeholders})"
        params.extend(document_ids)
    params.append(limit)

    rows = connection.execute(
        f"""
        SELECT
            rag_chunks_fts_v2.chunk_id AS chunk_id,
            rag_chunks_fts_v2.document_id AS document_id,
            rag_chunks_fts_v2.source_path AS source_path,
            rag_chunks_fts_v2.filename AS filename,
            rag_chunks_fts_v2.content AS content,
            chunks.chunk_index AS chunk_index,
            chunks.page_number AS page_number,
            chunks.heading_path_json AS heading_path_json,
            chunks.token_count AS token_count,
            chunks.char_count AS char_count,
            chunks.metadata_json AS metadata_json,
            bm25(rag_chunks_fts_v2) AS score
        FROM rag_chunks_fts_v2
        JOIN chunks ON chunks.id = rag_chunks_fts_v2.chunk_id
        JOIN documents ON documents.id = rag_chunks_fts_v2.document_id
        WHERE rag_chunks_fts_v2 MATCH ?
          AND (
              (
                  documents.active_revision_id IS NOT NULL
                  AND rag_chunks_fts_v2.revision_id = documents.active_revision_id
              )
              OR (
                  documents.active_revision_id IS NULL
                  AND documents.revision_status = 'active'
                  AND rag_chunks_fts_v2.revision_id = ''
                  AND chunks.revision_id IS NULL
              )
          )
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
        try:
            metadata = json.loads(data.pop("metadata_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        data["page_label"] = metadata.get("page_label")
        results.append(RetrievedChunk(**data))
    return results
