"""Re-derive catalog titles when the title rules change.

Titles are computed at ingest; a library indexed under older rules keeps
names such as "Thông Tư" or a legacy-encoded metadata title until its files
are re-parsed. This pass re-reads only what the rules need (page 1 of a PDF,
the DOCX paragraphs, the stored text) and never re-parses or re-embeds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.catalog.titles import TITLE_RULES_VERSION, guess_title
from app.db.sqlite import connect

logger = logging.getLogger(__name__)


def refresh_document_titles(db_path: Path) -> int:
    """Update documents whose title predates ``TITLE_RULES_VERSION``; returns
    how many titles changed. Idempotent: each document is marked once."""
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, source_path, title, metadata_json, active_revision_id FROM documents"
        ).fetchall()
    changed = 0
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict) or metadata.get("title_rules") == TITLE_RULES_VERSION:
            continue
        title = None
        path = Path(str(row["source_path"] or ""))
        try:
            if path.is_file():
                with connect(db_path) as connection:
                    text = "\n".join(
                        str(chunk["content"] or "")
                        for chunk in connection.execute(
                            "SELECT content FROM chunks WHERE document_id = ? AND coalesce(revision_id, '') = ? "
                            "ORDER BY order_index LIMIT 40",
                            (row["id"], str(row["active_revision_id"] or "")),
                        ).fetchall()
                    )
                title = guess_title(path, text)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the pass
            logger.warning("title_refresh_failed document=%s error_type=%s", row["id"], type(exc).__name__)
        metadata["title_rules"] = TITLE_RULES_VERSION
        with connect(db_path) as connection:
            if title and title != row["title"]:
                connection.execute(
                    "UPDATE documents SET title = ?, metadata_json = ? WHERE id = ?",
                    (title, json.dumps(metadata, ensure_ascii=False), row["id"]),
                )
                connection.execute(
                    "UPDATE document_cards SET title_guess = ? WHERE document_id = ?", (title, row["id"])
                )
                changed += 1
            else:
                connection.execute(
                    "UPDATE documents SET metadata_json = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), row["id"]),
                )
    return changed
