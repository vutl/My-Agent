#!/usr/bin/env python3
"""Re-enrich figure retrieval context with the configured VLM and optionally re-vector."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.db.sqlite import connect
from app.rag.embeddings import OllamaEmbeddingProvider
from app.services.figure_enrich_service import FigureEnrichService
from app.services.vector_index_service import create_lancedb_vector_index_service


def resolve_document_id(db_path: Path, document_id: str | None, filename_like: str | None) -> str | None:
    if document_id:
        return document_id
    if not filename_like:
        return None
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, filename
            FROM documents
            WHERE filename LIKE ?
            ORDER BY indexed_at DESC
            LIMIT 1
            """,
            (f"%{filename_like}%",),
        ).fetchone()
    if row is None:
        return None
    print(f"Resolved document: {row['filename']} ({row['id']})")
    return str(row["id"])


async def maybe_revector(document_id: str | None, *, revector: bool) -> dict | None:
    if not revector:
        return None
    settings = get_settings()
    embeddings = OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )
    service = create_lancedb_vector_index_service(
        db_path=settings.sqlite_db_path,
        lancedb_path=settings.lancedb_path,
        embeddings=embeddings,
    )
    if document_id:
        return await service.index_document(document_id)
    return await service.index_all_documents()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich figure VLM retrieval context")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--filename", default=None, help="Substring match, e.g. ASPIRE")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-revector", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    document_id = resolve_document_id(
        settings.sqlite_db_path,
        args.document_id,
        args.filename,
    )
    if args.filename and document_id is None:
        print(f"No document matched filename like {args.filename!r}")
        raise SystemExit(1)

    enricher = FigureEnrichService(
        db_path=settings.sqlite_db_path,
        ollama_host=settings.ollama_host,
        vision_model=settings.vision_model,
        vision_provider=settings.vision_provider,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        request_timeout_seconds=max(settings.request_timeout_seconds, 180.0),
        artifact_root=settings.artifacts_path,
    )
    result = enricher.enrich_document(
        document_id,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    vector_result = asyncio.run(
        maybe_revector(document_id, revector=not args.no_revector and result.get("enriched", 0) > 0)
    )
    if vector_result is not None:
        print(json.dumps({"vector_index": vector_result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
