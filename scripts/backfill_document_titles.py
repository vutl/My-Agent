#!/usr/bin/env python3
"""Recompute document titles for an existing catalog (no model calls).

Titles drive catalog resolution of misspelled names and descriptions such as
"the paper about key-sparse attention". Older indexes stored a wrong
``title``/``title_guess`` (author lines, body sentences). This script rebuilds
them with ``app.catalog.titles``: from the source PDF when it still exists,
otherwise from the first chunks of the active revision.

Dry run by default; pass --apply to write.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.catalog.titles import pdf_title, text_title  # noqa: E402


def _title_for(connection: sqlite3.Connection, row: sqlite3.Row) -> tuple[str | None, str]:
    path = Path(str(row["source_path"] or ""))
    if path.suffix.lower() == ".pdf" and path.exists():
        title = pdf_title(path)
        if title:
            return title, "pdf"
    if path.exists() and path.suffix.lower() in {".md", ".markdown", ".txt"}:
        title = text_title(path.read_text(encoding="utf-8", errors="ignore"), allow_plain_lines=False)
        if title:
            return title, "file_text"
        return path.stem.replace("_", " ").strip(), "filename"
    chunks = connection.execute(
        """
        SELECT content FROM chunks
        WHERE document_id = ? AND COALESCE(revision_id, '') = COALESCE(?, '')
        ORDER BY chunk_index LIMIT 3
        """,
        (row["id"], row["active_revision_id"]),
    ).fetchall()
    title = text_title(
        "\n".join(str(chunk["content"] or "") for chunk in chunks),
        allow_plain_lines=str(row["filename"] or "").lower().endswith(".pdf"),
    )
    if title:
        return title, "chunks"
    stem = Path(str(row["filename"] or "")).stem.replace("_", " ").strip()
    return (stem, "filename") if stem else (None, "none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    connection = sqlite3.connect(args.db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT d.id, d.filename, d.source_path, d.title, d.active_revision_id, c.title_guess
        FROM documents d LEFT JOIN document_cards c ON c.document_id = d.id
        ORDER BY d.filename
        """
    ).fetchall()
    changed = 0
    for row in rows:
        title, origin = _title_for(connection, row)
        if not title:
            print(f"keep  {row['filename']}: no better title found")
            continue
        if title == (row["title"] or "") and title == (row["title_guess"] or ""):
            continue
        changed += 1
        print(f"{'set ' if args.apply else 'would'} {row['filename']}: {title!r} ({origin}; was {str(row['title'] or '')[:50]!r})")
        if args.apply:
            connection.execute("UPDATE documents SET title = ? WHERE id = ?", (title, row["id"]))
            connection.execute("UPDATE document_cards SET title_guess = ? WHERE document_id = ?", (title, row["id"]))
    if args.apply:
        connection.commit()
    print(f"{'updated' if args.apply else 'would update'} {changed} of {len(rows)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
