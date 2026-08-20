#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.db.sqlite import init_db  # noqa: E402
from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402
from app.services.vector_index_service import VectorIndexService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local RAG index and optional LanceDB vectors.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "pdf"),
        help="File or folder to index. Defaults to the repo-local pdf/ folder.",
    )
    parser.add_argument("--collection", help="Collection name for a single file.")
    parser.add_argument("--recursive", action="store_true", help="Recursively index folders.")
    parser.add_argument("--no-vectors", action="store_true", help="Skip LanceDB vector indexing.")
    parser.add_argument("--vectors-only", action="store_true", help="Skip indexing and rebuild vectors from SQLite.")
    parser.add_argument("--skip-vision", action="store_true", help="Extract image artifacts without VLM summaries.")
    parser.add_argument("--limit", type=int, default=1000, help="Max documents for vector index-all.")
    args = parser.parse_args()

    settings = get_settings()
    init_db(settings.sqlite_db_path)
    if args.vectors_only:
        asyncio.run(_build_vectors(settings, [], args.limit))
        return

    target = Path(args.path).expanduser().resolve()
    service = IndexingService(
        db_path=settings.sqlite_db_path,
        artifact_root=settings.artifacts_path,
        ollama_host=settings.ollama_host,
        vision_model=None if args.skip_vision else settings.vision_model,
        vision_provider=settings.vision_provider,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        request_timeout_seconds=settings.request_timeout_seconds,
    )

    if target.is_file():
        collection = args.collection or target.stem
        print(f"[index] file {target}", flush=True)
        document = service.index_file(source_path=str(target), collection_name=collection)
        print(f"[index] done {document.filename} ({document.id})", flush=True)
        print(f"[index] collection {collection}", flush=True)
        document_ids = [document.id]
    elif target.is_dir():
        collection = args.collection or f"folder:{target.name or 'root'}"
        document_ids = []
        indexed = skipped = failed = 0
        paths = sorted(target.rglob("*") if args.recursive else target.iterdir())
        files = [
            path
            for path in paths
            if path.is_file() and path.suffix.lower().lstrip(".") in {"txt", "md", "pdf", "docx"}
        ]
        print(f"[index] folder {target} files={len(files)} collection={collection}", flush=True)
        for index, path in enumerate(files, start=1):
            print(f"[index] {index}/{len(files)} {path.name}", flush=True)
            try:
                document = service.index_file(source_path=str(path), collection_name=collection)
            except Exception as exc:
                failed += 1
                print(f"[index] failed {path.name}: {exc}", flush=True)
                continue
            document_ids.append(document.id)
            indexed += 1
            print(
                f"[index] done {path.name} document_id={document.id} chunks={document.chunk_count}",
                flush=True,
            )
        print(f"[index] summary indexed={indexed}, skipped={skipped}, failed={failed}", flush=True)
    else:
        raise SystemExit(f"Path does not exist: {target}")

    _prune_orphan_artifacts(settings)

    if args.no_vectors:
        return

    asyncio.run(_build_vectors(settings, document_ids, args.limit))


async def _build_vectors(settings, document_ids: list[str], limit: int) -> None:
    store = LanceDBRetrievalStore(settings.lancedb_path)
    embeddings = OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )
    vector_index = VectorIndexService(
        db_path=settings.sqlite_db_path,
        retrieval_store=store,
        embeddings=embeddings,
    )
    if document_ids:
        current_document_ids = vector_index._document_ids(None)
        await store.prune_documents(current_document_ids)

        total_text = total_tables = total_figures = failed = 0
        for index, document_id in enumerate(document_ids, start=1):
            print(f"[vectors] {index}/{len(document_ids)} document_id={document_id}", flush=True)
            try:
                result = await vector_index.index_document(document_id)
            except Exception as exc:
                failed += 1
                print(f"[vectors] failed {document_id}: {exc}", flush=True)
                continue
            total_text += int(result.get("text_chunks", 0) or 0)
            total_tables += int(result.get("table_chunks", 0) or 0)
            total_figures += int(result.get("figure_chunks", 0) or 0)
            print(
                "[vectors] done "
                f"text={result.get('text_chunks', 0)}, "
                f"tables={result.get('table_chunks', 0)}, "
                f"figures={result.get('figure_chunks', 0)}",
                flush=True,
            )
        print(
            f"[vectors] summary text={total_text}, tables={total_tables}, figures={total_figures}, failed={failed}",
            flush=True,
        )
        return

    result = await vector_index.index_all_documents(limit=limit)
    print(
        "built vectors: "
        f"indexed={result['indexed_documents']}/{result['documents']}, "
        f"skipped={result.get('skipped_documents', 0)}, "
        f"failed={result['failed_documents']}"
    )


def _prune_orphan_artifacts(settings) -> None:
    artifacts_path = settings.artifacts_path
    if not artifacts_path.exists():
        return

    connection = sqlite3.connect(settings.sqlite_db_path)
    try:
        document_ids = {
            row[0]
            for row in connection.execute("SELECT id FROM documents").fetchall()
        }
    finally:
        connection.close()

    removed = 0
    for path in artifacts_path.iterdir():
        if not path.is_dir() or path.name in document_ids:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        print(f"[artifacts] pruned orphan folders={removed}", flush=True)
    _write_root_artifact_manifest(settings, document_ids)


def _write_root_artifact_manifest(settings, document_ids: set[str]) -> None:
    artifacts_path = settings.artifacts_path
    artifacts_path.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.sqlite_db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, filename, source_path, parser_name, parser_version,
                   page_count, table_count, figure_count, indexed_at
            FROM documents
            ORDER BY filename COLLATE NOCASE
            """
        ).fetchall()
    finally:
        connection.close()

    documents = [dict(row) for row in rows]
    (artifacts_path / "_manifest.json").write_text(
        json.dumps(
            {
                "artifact_root": str(artifacts_path),
                "document_count": len(documents),
                "document_ids": sorted(document_ids),
                "documents": documents,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
