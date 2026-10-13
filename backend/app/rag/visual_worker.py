"""Long-lived NDJSON worker for the dependency-isolated ColSmol runtime."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

from app.rag.colsmol_retrieval import ColSmolVisualRetrievalProvider
from app.rag.visual_retrieval import VisualPageRecord


def _response(request_id: str, **payload: Any) -> None:
    sys.stdout.write(json.dumps({"id": request_id, **payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def _handle(provider: ColSmolVisualRetrievalProvider, request: dict[str, Any]) -> Any:
    operation = str(request.get("op") or "")
    if operation == "health":
        return {
            "ok": True,
            "provider": provider.name,
            "model_fingerprint": provider.model_fingerprint,
            "pid": os.getpid(),
            "requested_device": getattr(provider.encoder, "requested_device", None),
            "effective_device": getattr(provider.encoder, "_device", None),
            **provider.health(),
        }
    if operation == "index":
        pages = [VisualPageRecord(**value) for value in request.get("pages") or []]
        # This process is already the dependency-isolated worker.  Calling
        # the provider's async facade here would create an executor thread
        # for the CPU/MPS-heavy operation; when the parent aborts a timed-out
        # request, asyncio.run() then waits forever in
        # ``shutdown_default_executor``.  Run the provider's synchronous
        # transaction directly so terminating the child is a hard boundary.
        return provider._index_pages_sync(pages)  # noqa: SLF001
    if operation == "search":
        hits = provider._search_sync(  # noqa: SLF001
            str(request.get("query") or ""),
            document_ids=request.get("document_ids"),
            revision_ids=request.get("revision_ids"),
            top_k=int(request.get("top_k") or 10),
        )
        return [asdict(hit) for hit in hits]
    if operation == "get_pages":
        hits = provider._get_pages_sync(  # noqa: SLF001
            page_numbers=[int(value) for value in request.get("page_numbers") or []],
            document_ids=request.get("document_ids"),
            revision_ids=request.get("revision_ids"),
        )
        return [asdict(hit) for hit in hits]
    if operation == "delete":
        provider._delete_document_sync(  # noqa: SLF001
            str(request.get("document_id") or ""), request.get("revision_id")
        )
        return {"ok": True}
    if operation == "shutdown":
        return {"ok": True, "shutdown": True}
    raise ValueError(f"unsupported visual worker operation: {operation}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    provider = ColSmolVisualRetrievalProvider(
        index_root=args.index_root,
        model_name=args.model,
        model_revision=args.model_revision,
        base_model_name=args.base_model,
        base_model_revision=args.base_revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id") or "")
            if not request_id:
                raise ValueError("request id is required")
            # Third-party progress/log output must never corrupt stdout's NDJSON
            # protocol.  Preserve it on stderr for diagnostics.
            with redirect_stdout(sys.stderr):
                result = asyncio.run(_handle(provider, request))
            _response(request_id, ok=True, result=result)
            if isinstance(result, dict) and result.get("shutdown"):
                return 0
        except Exception as exc:
            _response(
                str(locals().get("request_id") or "unknown"),
                ok=False,
                error_type=type(exc).__name__,
                error=str(exc),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
