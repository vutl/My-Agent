#!/usr/bin/env python3
"""Repair legacy document-card/vector gaps without rebuilding the corpus.

Modern imports create a deterministic document card before vector indexing.
This utility is intentionally explicit for old SQLite snapshots: it fills cards
only where they are missing, then reindexes exactly those repaired documents.
It is local-only and calls Ollama solely for the normal EmbeddingGemma vectors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402
from app.services.vector_index_service import VectorIndexService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Repair one canonical document ID (repeatable). Default: all missing cards.",
    )
    parser.add_argument("--limit", type=int, help="Maximum missing cards to repair.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Required: persist cards and vectorize repaired documents.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if not args.apply:
        raise SystemExit("Refusing to mutate data without --apply")

    settings = get_settings()
    indexing = IndexingService(db_path=settings.sqlite_db_path)
    repair = indexing.backfill_missing_document_cards(
        document_ids=args.document_ids,
        limit=args.limit,
    )
    repaired_ids = list(repair["backfilled_document_ids"])
    vectors: list[dict] = []
    if repaired_ids:
        embeddings = OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
        vector_index = VectorIndexService(
            db_path=settings.sqlite_db_path,
            retrieval_store=LanceDBRetrievalStore(settings.lancedb_path),
            embeddings=embeddings,
        )
        for document_id in repaired_ids:
            vectors.append(await vector_index.index_document(document_id))

    print(json.dumps({**repair, "vector_index": vectors}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
