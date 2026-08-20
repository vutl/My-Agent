#!/usr/bin/env python3
"""Rechunk canonical text and build a staged LanceDB v3 index.

The migration deliberately preserves SQLite document/table/figure identities,
backs up SQLite first, and writes vectors to a new directory. It never invokes
the figure VLM or mutates the live LanceDB directory.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.rag.embeddings import OllamaEmbeddingProvider
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore
from app.services.indexing_service import IndexingService
from app.services.vector_index_service import VectorIndexService


async def _run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    db_path = settings.sqlite_db_path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db_path.with_name(f"{db_path.name}.pre-chunk-v3-{stamp}.bak")
    shutil.copy2(db_path, backup)

    indexing = IndexingService(
        db_path=db_path,
        artifact_root=settings.artifacts_path,
        vision_model=None,
    )
    rechunk = indexing.rechunk_all_documents(limit=args.limit)

    staging_path = args.staging_lancedb.expanduser().resolve()
    store = LanceDBRetrievalStore(staging_path)
    embeddings = OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )
    vector = await VectorIndexService(
        db_path=db_path,
        retrieval_store=store,
        embeddings=embeddings,
    ).index_all_documents(limit=args.limit)
    return {
        "sqlite_backup": str(backup),
        "staging_lancedb": str(staging_path),
        "rechunk": rechunk,
        "vector_index": vector,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--staging-lancedb",
        type=Path,
        default=PROJECT_ROOT / "data" / "lancedb-v3-staging",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
