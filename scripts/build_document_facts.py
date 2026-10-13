#!/usr/bin/env python3
"""Resumable offline structured-fact backfill through an approved 9router model.

Sends prose chunks that contain numbers to the configured model. Run it only on
a corpus whose text is approved to go through 9router (see docs/handoff/06-OPS).
Use APP_DATA_DIR to target an isolated clone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings, validate_runtime_model_policy  # noqa: E402
from app.db.sqlite import connect, init_db  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402
from app.services.fact_extraction_service import build_document_facts  # noqa: E402


async def _run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    validate_runtime_model_policy(settings)
    runtime_root = settings.data_dir.resolve()
    if runtime_root == (ROOT / "data").resolve() or not (runtime_root / "evaluation-runtime.json").is_file():
        raise RuntimeError("Fact backfill requires APP_DATA_DIR pointing to a marked isolated evaluation runtime")
    init_db(settings.sqlite_db_path)
    client = get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=max(settings.request_timeout_seconds, args.timeout),
    )
    health = await client.health()
    if not health.get("reachable"):
        raise RuntimeError(f"9router unavailable: {health.get('error')}")
    model = args.model or settings.fact_extraction_model
    if args.document_id:
        document_ids = [args.document_id]
    else:
        with connect(settings.sqlite_db_path) as connection:
            document_ids = [row[0] for row in connection.execute("SELECT id FROM documents ORDER BY filename")]
    results = []
    for document_id in document_ids[: args.limit or None]:
        # Provider errors stop the run: no fallback model, partial work stays.
        results.append(
            await build_document_facts(
                settings.sqlite_db_path, document_id, client=client, model=model, force=args.force,
            )
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return {"db": str(settings.sqlite_db_path), "model": model, "documents": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    summary = asyncio.run(_run(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "documents"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
