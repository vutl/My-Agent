#!/usr/bin/env python3
"""Backfill LightRAG chunk -> canonical parent provenance without providers.

The script reads LightRAG's local text-chunk KV JSON and only updates the
SQLite provenance table. It never initializes LightRAG, calls an embedding
model, or sends paper content to an LLM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.db.sqlite import connect, init_db
from app.lightrag.provenance import sync_document_chunk_records


def _load_records(kv_path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(kv_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"LightRAG chunk store not found: {kv_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid LightRAG chunk JSON: {kv_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {kv_path}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for storage_key, raw_record in payload.items():
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        record["_id"] = str(record.get("_id") or storage_key).strip()
        full_doc_id = str(record.get("full_doc_id") or "").strip()
        if full_doc_id:
            grouped[full_doc_id].append(record)
    return dict(grouped)


def _canonical_document_ids(db_path: Path) -> set[str]:
    with connect(db_path) as connection:
        return {
            str(row["id"])
            for row in connection.execute("SELECT id FROM documents").fetchall()
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    db_path = args.db_path.expanduser().resolve()
    kv_path = args.kv_path.expanduser().resolve()
    init_db(db_path)

    grouped = _load_records(kv_path)
    canonical_ids = _canonical_document_ids(db_path)
    requested = set(args.document_id or [])
    if requested:
        unknown_requested = sorted(requested - canonical_ids)
        if unknown_requested:
            raise SystemExit(
                "Unknown canonical document ID(s): " + ", ".join(unknown_requested)
            )
        document_ids = sorted(requested)
    else:
        # A full backfill is a rebuild. Canonical documents missing from the KV
        # are synced with an empty record set so stale mappings are removed and
        # runtime resolution fails closed.
        document_ids = sorted(canonical_ids)

    results: list[dict[str, Any]] = []
    totals = {
        "documents_synced": 0,
        "records_seen": 0,
        "records_mapped": 0,
        "mappings_written": 0,
        "records_rejected": 0,
    }
    for document_id in document_ids:
        result = sync_document_chunk_records(
            db_path,
            document_id,
            grouped.get(document_id, []),
        )
        item = {
            "document_id": result.document_id,
            "records_seen": result.records_seen,
            "records_mapped": result.records_mapped,
            "mappings_written": result.mappings_written,
            "records_rejected": result.records_rejected,
            "rejection_reasons": result.rejection_reasons,
        }
        results.append(item)
        totals["documents_synced"] += 1
        for key in (
            "records_seen",
            "records_mapped",
            "mappings_written",
            "records_rejected",
        ):
            totals[key] += item[key]

    return {
        "mode": "provider_free_local_backfill",
        "sqlite_db": str(db_path),
        "lightrag_chunk_store": str(kv_path),
        "canonical_documents": len(canonical_ids),
        "kv_documents": len(grouped),
        "unknown_kv_document_ids": sorted(set(grouped) - canonical_ids),
        "canonical_documents_without_kv_chunks": sorted(canonical_ids - set(grouped)),
        "totals": totals,
        "documents": results,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Backfill local LightRAG-to-parent provenance in SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=settings.sqlite_db_path,
    )
    parser.add_argument(
        "--kv-path",
        type=Path,
        default=settings.lightrag_working_dir / "kv_store_text_chunks.json",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Restrict to one canonical document ID; repeat for multiple IDs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
