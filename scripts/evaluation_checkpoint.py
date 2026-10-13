"""Atomic evaluation checkpoints with fail-closed, ordered-suite contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Commit a complete JSON snapshot, preserving the previous one on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_completed_checkpoint(
    path: Path, *, evaluator_fingerprint: str, contract: dict[str, Any],
    selected_ids: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Reuse only successful rows from the identical code/options/source suite."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], "missing"
    except (OSError, ValueError):
        return [], "unreadable"
    if not isinstance(payload, dict):
        return [], "invalid_payload"
    if payload.get("evaluator_fingerprint") != evaluator_fingerprint:
        return [], "evaluator_changed"
    if payload.get("checkpoint_contract") != contract:
        return [], "contract_changed"
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return [], "invalid_rows"
    ids = [row.get("case_id") for row in rows if isinstance(row, dict)]
    if (
        len(ids) != len(rows)
        or any(not isinstance(case_id, str) or not case_id for case_id in ids)
        or len(set(ids)) != len(ids)
    ):
        return [], "invalid_or_duplicate_rows"
    if any(case_id not in selected_ids for case_id in ids):
        return [], "unexpected_case"
    return [row for row in rows if row.get("status") == "ok"], "matched"
