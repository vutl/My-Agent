"""Read-only permission and canonical ownership checks for answer pixels.

Visual retrieval is a search channel, not permission to upload an image.
Canonical figure/table images can be used without loading a visual retriever,
but only for explicitly approved documents at their current active revision.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any


def _metadata(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        result = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def _image_path(source: dict) -> str:
    metadata = _metadata(source.get("metadata"))
    return str(source.get("image_path") or metadata.get("image_path")
               or metadata.get("table_image_path") or "")


def approved_answer_image_sources(
    sources: list[dict], *, db_path: Path, visual_root: Path,
    visual_pages_enabled: bool,
) -> list[dict]:
    """Resolve approved active canonical pixels; never trust source permissions.

    A specific false answer-images approval overrides broad enrichment approval.
    No inferred approval from a public filename, visual flag, conversation or
    question. Returns new dictionaries and never mutates SQLite or retrieval.
    """
    approved = []
    if not db_path.is_file():
        return approved
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for source in sources:
            document_id = str(source.get("document_id") or "")
            document = connection.execute(
                "SELECT active_revision_id, metadata_json FROM documents WHERE id=?",
                (document_id,),
            ).fetchone()
            if document is None:
                continue
            policy = _metadata(document["metadata_json"])
            permission = policy.get("answer_images_approved", policy.get("enrichment_approved"))
            if permission is not True:
                continue
            active = document["active_revision_id"]
            metadata = _metadata(source.get("metadata"))
            declared = source.get("revision_id") or metadata.get("revision_id")
            if declared is not None and declared != active:
                continue
            kind = str(source.get("artifact_type") or source.get("chunk_type") or "").casefold()
            figure_id = source.get("figure_id")
            table_id = source.get("table_id")
            canonical = None
            if table_id:
                canonical = connection.execute(
                    "SELECT revision_id, metadata_json FROM document_tables "
                    "WHERE id=? AND document_id=? AND revision_id IS ?",
                    (table_id, document_id, active),
                ).fetchone()
            elif figure_id:
                canonical = connection.execute(
                    "SELECT revision_id, metadata_json, image_path FROM document_figures "
                    "WHERE id=? AND document_id=? AND revision_id IS ?",
                    (figure_id, document_id, active),
                ).fetchone()
            if canonical is not None:
                stored = _metadata(canonical["metadata_json"])
                path = (canonical["image_path"] if "image_path" in canonical.keys() else None)
                path = path or stored.get("image_path") or stored.get("table_image_path")
                digest = stored.get("image_sha256")
            elif kind in {"page", "visual_page"} and visual_pages_enabled and declared == active:
                # The renderer's committed manifest is authority for full-page
                # pixels. A random image in the visual directory is insufficient.
                try:
                    image = Path(_image_path(source)).resolve(strict=True)
                    if not image.is_relative_to(visual_root.resolve()):
                        continue
                    manifest = _metadata((image.parent / "manifest.json").read_text())
                    if manifest.get("document_id") != document_id or manifest.get("revision_id") != active:
                        continue
                    matches = [entry for entry in manifest.get("pages", [])
                               if entry.get("page_number") == source.get("page_number")
                               and Path(str(entry.get("image_path") or "")).resolve() == image]
                    if len(matches) != 1:
                        continue
                    path, digest = str(image), matches[0].get("image_sha256")
                    if not digest:
                        continue
                except (OSError, ValueError, TypeError, RuntimeError):
                    continue
            else:
                continue
            try:
                actual_path = Path(_image_path(source)).resolve(strict=True)
                canonical_path = Path(str(path or "")).resolve(strict=True)
                if actual_path != canonical_path or not actual_path.is_file():
                    continue
            except (OSError, ValueError, RuntimeError):
                continue
            approved.append({**source, "revision_id": active, "image_path": str(actual_path),
                             "answer_image_sha256": digest})
    return approved
