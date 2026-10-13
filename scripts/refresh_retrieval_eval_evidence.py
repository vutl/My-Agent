#!/usr/bin/env python3
"""Refresh proposal qrels from canonical local artifacts without changing rewrites.

The Gate-C proposal builder intentionally keeps question generation separate
from evidence selection.  This utility repairs *evidence packets* when an
older proposal contains a title-only text chunk or an artifact caption without
its canonical table contents.  It is source-backed and fail-closed: it never
marks a suite audited and never copies evidence into a query/rewrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha_text(value: str) -> str:
    return hashlib.sha256(_norm(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _document_id(connection: sqlite3.Connection, document: dict[str, Any]) -> str:
    rows = connection.execute(
        "SELECT id FROM documents WHERE filename = ? AND content_hash = ?",
        (str(document.get("filename") or ""), str(document.get("content_sha256") or "")),
    ).fetchall()
    if len(rows) != 1:
        raise RuntimeError(
            f"document identity did not resolve uniquely: {document.get('filename')!r} ({len(rows)})"
        )
    return str(rows[0]["id"])


def _section(value: object) -> str:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    return str(parsed.get("section_title") or "")


def _text_score(row: sqlite3.Row, role: str) -> tuple[int, int, int, int]:
    content = str(row["content"] or "")
    section = _section(row["metadata_json"])
    lowered = _norm(f"{section} {content[:1600]}")
    preferred = {
        "task": (
            ("abstract", 12), ("purpose", 11), ("objective", 11),
            ("introduction", 9), ("problem", 8), ("we propose", 8),
            ("this paper", 7), ("this work", 7), ("contribution", 7),
            ("one-line vision", 16), ("why this addendum exists", 15),
            ("what we are building first", 12), ("vision", 8),
            ("project", 6), ("goal", 6), ("why 9router", 10),
            ("solves this", 16), ("why", 5), ("challenge", 5),
        ),
        "architecture": (
            ("architecture", 12), ("proposed model", 12), ("proposed framework", 12),
            ("method", 9), ("model", 8), ("pipeline", 8), ("fusion", 8),
            ("retrieval store", 11), ("storage", 9), ("components", 8),
            ("framework", 8), ("proposed", 7), ("distillation", 10),
            ("student", 8), ("teacher", 8), ("encoder", 7), ("module", 7),
            ("listing", 14), ("feature extractor", 12), ("learnable pooling", 11),
            ("non contiguous", 10), ("transformer layers", 9),
            ("design", 7), ("approach", 7),
        ),
    }[role]
    score = sum(weight for term, weight in preferred if term in lowered)
    # Title/author/HTML chrome is not a substantive qrel even if it contains
    # the paper name.  This rule is length/structure based, not paper-specific.
    alphabetic = sum(char.isalpha() for char in content)
    substantive = 1 if len(content) >= 240 and alphabetic >= 120 else 0
    chrome_markers = ("author contributions", "acknowledgment", "references", "copyright", "terms & policies")
    chrome = 20 if len(content) < 180 or alphabetic < 80 else 0
    if any(marker in _norm(section) for marker in chrome_markers):
        chrome += 90
    return (score * 4 + substantive * 30 - chrome, alphabetic, min(len(content), 3000), -int(row["chunk_index"] or 0))


def _refresh_text(
    connection: sqlite3.Connection,
    *,
    locator: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], str]:
    document = locator.get("document") or {}
    document_id = _document_id(connection, document)
    rows = connection.execute(
        """
        SELECT chunk_index, page_number, content, metadata_json
        FROM chunks
        WHERE document_id = ? AND chunk_type = 'text'
        ORDER BY chunk_index
        """,
        (document_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"no text chunks for {document.get('filename')!r}")

    # Re-select from the complete document rather than preserving a stale
    # title-only locator.  This is intentionally role-based (task versus
    # architecture), not paper-name based.  A qrel may move to another page
    # when the old parser selected a weak HTML/title chunk; portable identity
    # remains the document hash + canonical quote.
    substantive = [
        row for row in rows
        if len(str(row["content"] or "")) >= 240
        and sum(c.isalpha() for c in str(row["content"] or "")) >= 120
    ]
    filename_tokens = {
        token for token in re.findall(r"[a-z0-9]{4,}", _norm(str(document.get("filename") or "")))
        if token not in {"paper", "pdf", "project", "plan", "addendum"}
    }
    def source_score(row: sqlite3.Row) -> tuple[int, int, int, int]:
        base = _text_score(row, role)
        content = _norm(row["content"])
        title_bonus = sum(5 for token in filename_tokens if token in content)
        return (base[0] + title_bonus, base[1], base[2], base[3])
    chosen = max(substantive or rows, key=source_score)
    content = re.sub(r"\s+", " ", str(chosen["content"] or "")).strip()
    # Short Markdown/TXT notes can be perfectly valid evidence; only reject
    # an empty/near-empty projection, not a document because it is compact.
    if len(content) < 40:
        raise RuntimeError(f"selected text evidence is too short: {document.get('filename')!r}")
    refreshed = dict(locator)
    refreshed["page_number"] = chosen["page_number"]
    refreshed["section"] = _section(chosen["metadata_json"]) or locator.get("section")
    refreshed["quote"] = content[:3000]
    refreshed["quote_sha256"] = _sha_text(refreshed["quote"])
    refreshed["evidence_refresh"] = "canonical_text_chunk"
    return refreshed, "text"


def _refresh_artifact(connection: sqlite3.Connection, *, locator: dict[str, Any]) -> tuple[dict[str, Any], str]:
    document = locator.get("document") or {}
    document_id = _document_id(connection, document)
    kind = str(locator.get("kind") or "")
    if kind == "table":
        row = connection.execute(
            """
            SELECT page_number, table_index, caption, markdown
            FROM document_tables WHERE document_id = ? AND table_index = ?
            """,
            (document_id, int(locator.get("artifact_index") or 0)),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"table locator did not resolve: {document.get('filename')!r}")
        if locator.get("caption_sha256") and _sha_text(str(row["caption"] or "")) != locator["caption_sha256"]:
            raise RuntimeError(f"table caption fingerprint mismatch: {document.get('filename')!r}")
        content = "\n".join(part for part in (str(row["caption"] or "").strip(), str(row["markdown"] or "").strip()) if part)
        refreshed = dict(locator)
        refreshed["page_number"] = row["page_number"]
        refreshed["caption"] = str(row["caption"] or "")
        refreshed["quote"] = content
        refreshed["quote_sha256"] = _sha_text(content)
        refreshed["evidence_refresh"] = "canonical_table_markdown"
        return refreshed, "table"
    if kind == "figure":
        row = connection.execute(
            """
            SELECT page_number, figure_index, caption
            FROM document_figures WHERE document_id = ? AND figure_index = ?
            """,
            (document_id, int(locator.get("artifact_index") or 0)),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"figure locator did not resolve: {document.get('filename')!r}")
        if locator.get("caption_sha256") and _sha_text(str(row["caption"] or "")) != locator["caption_sha256"]:
            raise RuntimeError(f"figure caption fingerprint mismatch: {document.get('filename')!r}")
        refreshed = dict(locator)
        refreshed["page_number"] = row["page_number"]
        refreshed["caption"] = str(row["caption"] or "")
        refreshed["quote"] = str(row["caption"] or "")
        refreshed["quote_sha256"] = _sha_text(refreshed["quote"])
        refreshed["evidence_refresh"] = "canonical_figure_caption"
        return refreshed, "figure"
    return dict(locator), "unchanged"


def refresh(rows: list[dict[str, Any]], db_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    stats = {"text": 0, "table": 0, "figure": 0, "unchanged": 0, "failed": 0}
    try:
        output: list[dict[str, Any]] = []
        for row in rows:
            current = json.loads(json.dumps(row))
            refreshed_groups = []
            for locator in current.get("evidence_groups") or []:
                try:
                    role = "architecture" if str(locator.get("group_id") or "").startswith("architecture") else "task"
                    if current.get("category") == "intra_document_multihop" and str(locator.get("group_id") or "").startswith("architecture"):
                        role = "architecture"
                    if locator.get("kind") == "text":
                        refreshed, kind = _refresh_text(connection, locator=locator, role=role)
                    elif locator.get("kind") in {"table", "figure"}:
                        refreshed, kind = _refresh_artifact(connection, locator=locator)
                    else:
                        refreshed, kind = dict(locator), "unchanged"
                    stats[kind] += 1
                    refreshed_groups.append(refreshed)
                except Exception:
                    stats["failed"] += 1
                    refreshed_groups.append(locator)
            current["evidence_groups"] = refreshed_groups
            current["curation_status"] = "review_required"
            current["language_reviewed"] = False
            current["qrels_reviewed"] = False
            current["audit"] = {"status": "pending", "reviewer": None, "reviewed_at": None, "notes": "Evidence refreshed from canonical local projections; independent semantic audit required."}
            output.append(current)
        return output, stats
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; use --force")
    db_path = args.db or get_settings().sqlite_db_path
    rows, stats = refresh(_read_jsonl(args.suite), Path(db_path))
    _write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "db": str(db_path), "stats": stats}, ensure_ascii=False, indent=2))
    if stats["failed"]:
        return 2
    return 0


if __name__ == "__main__":
    main()
