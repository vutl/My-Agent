#!/usr/bin/env python3
"""Resumable offline paper-evidence backfill through the approved 9router model."""

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
from app.db.sqlite import init_db  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402
from app.services.paper_evidence_builder import PaperEvidenceBuilder  # noqa: E402
from app.services.paper_evidence_service import PaperEvidenceService  # noqa: E402


async def _run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    validate_runtime_model_policy(settings)
    init_db(settings.sqlite_db_path)
    service = PaperEvidenceService(
        settings.sqlite_db_path,
        schema_version=settings.paper_evidence_card_schema_version,
        prompt_version=settings.paper_evidence_card_prompt_version,
    )
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
    model = args.model or settings.paper_evidence_card_model
    advertised = set(health.get("models") or [])
    if advertised and model not in advertised:
        # Some gateways advertise de-namespaced IDs. The request/response client
        # still enforces exact/de-namespaced model identity on completion.
        _, _, upstream = model.partition("/")
        if upstream not in advertised:
            raise RuntimeError(f"Configured evidence model {model!r} is not advertised by 9router")
    builder = PaperEvidenceBuilder(
        service=service,
        client=client,
        model=model,
        max_concurrency=args.max_concurrency,
    )
    if args.document_id:
        return await builder.build_document(args.document_id, force=args.force)
    return await builder.build_all(limit=args.limit, force=args.force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--max-concurrency", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--approve-external-corpus-upload",
        action="store_true",
        help="Required because canonical paper excerpts are sent through local 9router.",
    )
    args = parser.parse_args()
    if not args.approve_external_corpus_upload:
        parser.error("Refusing corpus upload without --approve-external-corpus-upload")
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": " ".join(str(exc).split())[:1000],
                    "fallback_model_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
