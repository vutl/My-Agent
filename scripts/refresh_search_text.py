#!/usr/bin/env python3
"""Refresh search text for documents containing Unicode math letters.

Indexes built before ``app.rag.text_normalization`` store "L𝑡𝑜𝑡𝑎𝑙"-style
glyphs in FTS and embeddings, so formula passages rarely rank. This updates,
for the active revision of affected documents only:

1. FTS rows (``rag_chunks_fts``/``rag_chunks_fts_v2``) with normalized text;
2. the LanceDB projection via ``VectorIndexService.index_document`` (local
   Ollama embeddings; no LLM calls; chunk ids and revisions unchanged).

Dry run by default; pass --apply to write. Back up the DB/LanceDB first.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from app.rag.text_normalization import needs_search_normalization, searchable_text  # noqa: E402
from app.services.vector_index_service import create_lancedb_vector_index_service  # noqa: E402


def _affected(connection: sqlite3.Connection) -> dict[str, list[str]]:
    rows = connection.execute(
        """
        SELECT c.id, c.document_id, c.content
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE COALESCE(c.revision_id, '') = COALESCE(d.active_revision_id, '')
        """
    ).fetchall()
    affected: dict[str, list[str]] = {}
    for chunk_id, document_id, content in rows:
        if needs_search_normalization(str(content or "")):
            affected.setdefault(str(document_id), []).append(str(chunk_id))
    return affected


async def _reindex(db_path: Path, lancedb_path: Path, document_ids: list[str]) -> list[dict]:
    settings = get_settings()
    embeddings = OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        batch_size=settings.embedding_batch_size,
        transient_retries=settings.embedding_transient_retries,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )
    service = create_lancedb_vector_index_service(
        db_path=db_path, lancedb_path=lancedb_path, embeddings=embeddings,
    )
    results = []
    for document_id in document_ids:
        results.append(await service.index_document(document_id))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--lancedb-path", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    connection = sqlite3.connect(args.db_path)
    affected = _affected(connection)
    names = dict(connection.execute("SELECT id, filename FROM documents").fetchall())
    for document_id, chunk_ids in affected.items():
        print(f"{'fix ' if args.apply else 'would fix'} {names.get(document_id, document_id)}: {len(chunk_ids)} chunks")
    if not args.apply or not affected:
        print(f"{len(affected)} documents affected; dry run" if not args.apply else "nothing to do")
        return 0

    for chunk_ids in affected.values():
        for chunk_id in chunk_ids:
            for table in ("rag_chunks_fts", "rag_chunks_fts_v2"):
                row = connection.execute(f"SELECT content FROM {table} WHERE chunk_id = ?", (chunk_id,)).fetchone()
                if row is not None:
                    connection.execute(
                        f"UPDATE {table} SET content = ? WHERE chunk_id = ?",
                        (searchable_text(str(row[0] or "")), chunk_id),
                    )
    connection.commit()
    connection.close()

    results = asyncio.run(_reindex(args.db_path, args.lancedb_path, list(affected)))
    failed = [result for result in results if not result.get("ok", True)]
    print(f"FTS refreshed; vectors re-indexed for {len(results)} documents; failures: {len(failed)}")
    for result in failed:
        print("  failed:", result)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
