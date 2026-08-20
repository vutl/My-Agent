#!/usr/bin/env python3
"""Build an isolated LanceDB index for embedding-model evaluation.

The canonical SQLite rows and production LanceDB directory are read-only. This
script reuses stored cards/chunks/tables/figures and writes vectors into an
explicit staging directory; it does not rerun parsing, chunking, vision, or any
LLM operation.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.rag.embeddings import HuggingFaceEmbeddingProvider, OllamaEmbeddingProvider
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore
from app.services.vector_index_service import VectorIndexService


async def _run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    staging_path = args.staging_lancedb.expanduser().resolve()
    production_path = settings.lancedb_path.expanduser().resolve()
    if staging_path == production_path:
        raise ValueError("Refusing to use the production LanceDB path as staging")

    started = time.perf_counter()
    if args.provider == "huggingface":
        embeddings = HuggingFaceEmbeddingProvider(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
            query_task=args.query_task,
            document_task=args.document_task,
            trust_remote_code=args.trust_remote_code,
            truncate_dim=args.truncate_dim,
            revision=args.revision,
            code_revision=args.code_revision,
            native_model=args.native_model,
        )
    else:
        embeddings = OllamaEmbeddingProvider(
            host=args.ollama_host or settings.ollama_host,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
        )
    result = await VectorIndexService(
        db_path=settings.sqlite_db_path,
        retrieval_store=LanceDBRetrievalStore(staging_path),
        embeddings=embeddings,
    ).index_all_documents(limit=args.limit)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "provider": args.provider,
        "revision": args.revision,
        "code_revision": args.code_revision,
        "query_prefix": args.query_prefix,
        "document_prefix": args.document_prefix,
        "sqlite_db": str(settings.sqlite_db_path),
        "production_lancedb_untouched": str(production_path),
        "staging_lancedb": str(staging_path),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "vector_index": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("ollama", "huggingface"), default="ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--staging-lancedb", type=Path, required=True)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-task", default=None)
    parser.add_argument("--document-task", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--truncate-dim", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--native-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
