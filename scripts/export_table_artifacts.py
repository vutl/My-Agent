#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from io import BytesIO
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.db.sqlite import init_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export table image artifacts for already-indexed PDFs.")
    parser.add_argument("--document-id", action="append", help="Limit export to one document id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum documents to scan.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing table image metadata.")
    args = parser.parse_args()

    settings = get_settings()
    init_db(settings.sqlite_db_path)
    documents = _documents(settings.sqlite_db_path, document_ids=args.document_id, limit=args.limit)

    total_tables = total_exported = total_failed = 0
    for index, document in enumerate(documents, start=1):
        print(f"[tables] {index}/{len(documents)} {document['filename']}", flush=True)
        result = _export_document_tables(
            db_path=settings.sqlite_db_path,
            artifact_root=settings.artifacts_path,
            document=document,
            force=args.force,
        )
        total_tables += result["tables"]
        total_exported += result["exported"]
        total_failed += result["failed"]
        print(
            f"[tables] done exported={result['exported']}/{result['tables']} failed={result['failed']}",
            flush=True,
        )

    _write_root_artifact_manifest(settings.sqlite_db_path, settings.artifacts_path)
    print(
        f"[tables] summary documents={len(documents)} tables={total_tables} exported={total_exported} failed={total_failed}",
        flush=True,
    )


def _documents(db_path: Path, *, document_ids: list[str] | None, limit: int) -> list[dict]:
    sql = """
        SELECT id, source_path, filename
        FROM documents
        WHERE file_type = 'pdf'
    """
    params: list[object] = []
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        sql += f" AND id IN ({placeholders})"
        params.extend(document_ids)
    sql += " ORDER BY indexed_at DESC LIMIT ?"
    params.append(limit)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    finally:
        connection.close()


def _export_document_tables(
    *,
    db_path: Path,
    artifact_root: Path,
    document: dict,
    force: bool,
) -> dict[str, int]:
    rows = _document_table_rows(db_path, document["id"])
    if not rows:
        return {"tables": 0, "exported": 0, "failed": 0}

    pending = [
        row
        for row in rows
        if force or not (json.loads(row["metadata_json"] or "{}").get("table_image_path"))
    ]
    if not pending:
        return {"tables": len(rows), "exported": 0, "failed": 0}

    try:
        docling_document = _convert_with_docling(Path(document["source_path"]))
    except Exception as exc:
        print(f"[tables] failed convert {document['filename']}: {exc}", flush=True)
        return {"tables": len(rows), "exported": 0, "failed": len(pending)}

    docling_tables = list(getattr(docling_document, "tables", []) or [])
    tables_dir = artifact_root / document["id"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    exported = failed = 0
    for row in pending:
        table_index = int(row["table_index"])
        if table_index >= len(docling_tables):
            failed += 1
            continue
        table = docling_tables[table_index]
        try:
            image = table.get_image(docling_document)
            image_bytes = _png_bytes(image)
        except Exception as exc:
            failed += 1
            print(f"[tables] failed table {table_index} {document['filename']}: {exc}", flush=True)
            continue

        digest = hashlib.sha256(image_bytes).hexdigest()
        page_number = row["page_number"] or _docling_page_number(table) or 0
        image_path = tables_dir / f"page_{int(page_number):03d}_table_{table_index + 1:03d}_{digest[:12]}.png"
        image_path.write_bytes(image_bytes)

        metadata = json.loads(row["metadata_json"] or "{}")
        metadata.update(
            {
                "table_image_path": str(image_path),
                "table_image_hash": digest,
                "table_image_width": int(getattr(image, "width", 0) or 0),
                "table_image_height": int(getattr(image, "height", 0) or 0),
                "table_image_source": "docling_table_get_image",
                "table_image_exported_at": datetime.now(UTC).isoformat(),
            }
        )
        bbox = _docling_bbox(table)
        _update_table_metadata(db_path, table_id=row["id"], metadata=metadata, bbox=bbox)
        exported += 1

    _update_document_manifest(db_path, artifact_root, document["id"])
    return {"tables": len(rows), "exported": exported, "failed": failed}


def _document_table_rows(db_path: Path, document_id: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT *
            FROM document_tables
            WHERE document_id = ?
            ORDER BY table_index ASC
            """,
            (document_id,),
        ).fetchall()
    finally:
        connection.close()


def _convert_with_docling(path: Path):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.images_scale = 2.0
    options.generate_page_images = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )
    return converter.convert(path).document


def _png_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _docling_page_number(item) -> int | None:
    prov = getattr(item, "prov", None)
    if isinstance(prov, list) and prov:
        page_no = getattr(prov[0], "page_no", None)
        if isinstance(page_no, int):
            return page_no
    return None


def _docling_bbox(item) -> dict[str, float] | None:
    prov = getattr(item, "prov", None)
    if not isinstance(prov, list) or not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return None
    values: dict[str, float] = {}
    for attr in ("l", "t", "r", "b", "x0", "y0", "x1", "y1"):
        value = getattr(bbox, attr, None)
        if isinstance(value, int | float):
            values[attr] = float(value)
    if {"l", "t", "r", "b"}.issubset(values):
        return {"x0": values["l"], "y0": values["t"], "x1": values["r"], "y1": values["b"]}
    if {"x0", "y0", "x1", "y1"}.issubset(values):
        return {"x0": values["x0"], "y0": values["y0"], "x1": values["x1"], "y1": values["y1"]}
    return None


def _update_table_metadata(
    db_path: Path,
    *,
    table_id: str,
    metadata: dict,
    bbox: dict[str, float] | None,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            UPDATE document_tables
            SET metadata_json = ?, bbox_json = COALESCE(?, bbox_json)
            WHERE id = ?
            """,
            (json.dumps(metadata), json.dumps(bbox) if bbox else None, table_id),
        )
        connection.commit()
    finally:
        connection.close()


def _update_document_manifest(db_path: Path, artifact_root: Path, document_id: str) -> None:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        document = connection.execute(
            """
            SELECT id, filename, source_path, parser_name, parser_version,
                   page_count, chunk_count, table_count, figure_count,
                   metadata_json, indexed_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if document is None:
            return
        table_image_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM document_tables
            WHERE document_id = ?
              AND json_extract(metadata_json, '$.table_image_path') IS NOT NULL
            """,
            (document_id,),
        ).fetchone()["count"]
    finally:
        connection.close()

    metadata = json.loads(document["metadata_json"] or "{}")
    manifest = {
        "document_id": document["id"],
        "filename": document["filename"],
        "source_path": document["source_path"],
        "parser_name": document["parser_name"],
        "parser_version": document["parser_version"],
        "page_count": document["page_count"],
        "chunk_count": document["chunk_count"],
        "table_count": document["table_count"],
        "figure_count": document["figure_count"],
        "artifact_extraction_version": metadata.get("artifact_extraction_version"),
        "indexed_at": document["indexed_at"],
        "table_image_count": table_image_count,
    }
    manifest_path = artifact_root / document_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_root_artifact_manifest(db_path: Path, artifact_root: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, filename, source_path, parser_name, parser_version,
                   page_count, table_count, figure_count, indexed_at
            FROM documents
            ORDER BY filename COLLATE NOCASE
            """
        ).fetchall()
    finally:
        connection.close()

    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "_manifest.json").write_text(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "document_count": len(rows),
                "documents": [dict(row) for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
