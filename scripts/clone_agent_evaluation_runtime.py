#!/usr/bin/env python3
"""Create a durable isolated evaluation clone; never overwrite an existing root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3


def clone_runtime(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if target == source or source in target.parents and not (
        source / "retrieval_eval" in target.parents
    ):
        raise ValueError("Target must be an isolated evaluation root")
    if target.exists():
        raise FileExistsError("Refusing to replace an existing evaluation runtime")
    source_db = source / "sqlite" / "app.db"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    (target / "sqlite").mkdir(parents=True)
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(target / "sqlite" / "app.db") as dst:
            src.backup(dst)
            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite snapshot integrity check failed")
            existing = {row[0] for row in dst.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            # Copied pending work must not trigger unrequested enrichment or
            # follow a production folder path when this clone starts.
            for table in ("document_processing_jobs", "conversation_memory_jobs",
                          "conversation_memory_l3_outbox", "paper_evidence_build_jobs"):
                if table in existing:
                    dst.execute(f'DELETE FROM "{table}"')
            documents = dst.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    for name in ("lancedb", "artifacts"):
        if not (source / name).is_dir():
            raise FileNotFoundError(source / name)
        shutil.copytree(source / name, target / name)
    report = {"source": str(source), "runtime_root": str(target),
              "documents": documents, "created_at": datetime.now(timezone.utc).isoformat(),
              "production_mutated": False, "pending_jobs_suppressed": True,
              "required_server_flags": {"LIGHTRAG_ENABLED": "false",
                                         "VISUAL_RETRIEVAL_ENABLED": "false",
                                         "DUAL_QUERY_RRF_ENABLED": "false"}}
    from evaluation_checkpoint import write_checkpoint
    write_checkpoint(target / "evaluation-runtime.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(clone_runtime(args.source, args.runtime_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
