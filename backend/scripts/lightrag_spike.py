#!/usr/bin/env python3
"""Spike: LightRAG + Codex/9router — insert KST.pdf and query locally."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.db.sqlite import connect
from app.lightrag.client import init_lightrag, shutdown_lightrag
from app.lightrag.ingest import ingest_document
from app.lightrag.query import query_lightrag


def find_kst_document_id(db_path: Path) -> str | None:
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, filename, source_path
            FROM documents
            WHERE filename LIKE '%KST%' OR source_path LIKE '%KST%'
            ORDER BY indexed_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    print(f"Found document: {row['filename']} ({row['source_path']})")
    return str(row["id"])


async def check_nine_router(api_base: str) -> bool:
    models_url = api_base.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(models_url)
            response.raise_for_status()
            payload = response.json()
            model_ids = [item.get("id") for item in payload.get("data", [])]
            print(f"9router models ({len(model_ids)}): {model_ids[:8]}")
            return True
    except httpx.HTTPError as exc:
        print(f"9router check failed: {exc}")
        return False


async def run_spike(*, query: str, skip_insert: bool) -> int:
    settings = get_settings()
    print(
        json.dumps(
            {
                "lightrag_llm_model": settings.lightrag_llm_model,
                "lightrag_llm_api_base": settings.lightrag_llm_api_base,
                "embedding_model": settings.embedding_model,
                "ollama_host": settings.ollama_host,
                "working_dir": str(settings.lightrag_working_dir),
            },
            indent=2,
        )
    )

    if not await check_nine_router(settings.lightrag_llm_api_base):
        return 1

    document_id = find_kst_document_id(settings.sqlite_db_path)
    if document_id is None:
        print("KST document not found in SQLite. Index KST.pdf first.")
        return 1

    await init_lightrag(settings)
    try:
        if not skip_insert:
            print(f"Inserting document {document_id} into LightRAG graph...")
            result = await ingest_document(settings.sqlite_db_path, document_id)
            print(
                json.dumps(
                    {
                        "track_id": result.track_id,
                        "char_count": result.char_count,
                        "source_path": result.source_path,
                    },
                    indent=2,
                )
            )

        print(f"Querying: {query!r} (mode=local)")
        raw = await query_lightrag(query, mode="local", enable_rerank=False)
        print(json.dumps(raw, indent=2, ensure_ascii=False, default=str))

        blob = json.dumps(raw, ensure_ascii=False).lower()
        hits = [term for term in ("kst", "key-sparse", "sparse transformer") if term in blob]
        if hits:
            print(f"PASS: found terms {hits}")
            return 0
        print("WARN: expected KST-related terms not found in response — check insert or query.")
        return 2
    finally:
        await shutdown_lightrag()


def main() -> None:
    parser = argparse.ArgumentParser(description="LightRAG Codex spike on KST.pdf")
    parser.add_argument("--query", default="KST là gì")
    parser.add_argument("--skip-insert", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_spike(query=args.query, skip_insert=args.skip_insert)))


if __name__ == "__main__":
    main()
