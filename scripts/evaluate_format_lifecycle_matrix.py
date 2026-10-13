#!/usr/bin/env python3
"""Run an isolated, no-LLM lifecycle matrix for arbitrary local formats.

This is deliberately broader than the four-file smoke.  It drives the same
durable ``DocumentProcessingService`` used by folder ingestion and checks
canonical parse/readiness, revision-scoped Lance projection, retrieval,
duplicate/rescan/change/rename semantics, and a failed source in one scan.
The runtime is disposable and never touches the production database.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.sqlite import connect, init_db  # noqa: E402
from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore  # noqa: E402
from app.services.document_processing_service import DocumentProcessingService  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


def _write_image(path: Path, *, kind: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (760, 420), "white")
    draw = ImageDraw.Draw(image)
    if kind == "table":
        rows = [("Model", "UA", "WA"), ("Baseline", "61.74", "62.37"), ("Proposed", "72.72", "71.90")]
        draw.text((28, 20), "Table 1. Image table retrieval fixture", fill="black")
        y = 100
        for row in rows:
            draw.rectangle((28, y, 700, y + 72), outline="black", width=2)
            draw.line((250, y, 250, y + 72), fill="black", width=2)
            draw.line((470, y, 470, y + 72), fill="black", width=2)
            draw.text((42, y + 25), row[0], fill="black")
            draw.text((270, y + 25), row[1], fill="black")
            draw.text((490, y + 25), row[2], fill="black")
            y += 72
    else:
        draw.text((28, 20), "Figure 1. Chart diagram retrieval fixture", fill="black")
        draw.rectangle((80, 100, 150, 330), fill="#315f9d")
        draw.rectangle((230, 160, 300, 330), fill="#4f9d69")
        draw.rectangle((380, 210, 450, 330), fill="#d88a2d")
        draw.text((80, 350), "text-ocr-77", fill="black")
        draw.text((505, 190), "A chart", fill="black")
    image.save(path)


def _make_fixtures(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text(
        "TXT lifecycle fixture. The retrieval token is cobalt-42 and the source is plain text.",
        encoding="utf-8",
    )
    (root / "research.md").write_text(
        "# Markdown lifecycle fixture\n\n"
        "A local table and image are part of the same document.\n\n"
        "| Metric | Value |\n|---|---:|\n| Recall | 0.91 |\n\n"
        "![diagram](diagram.png)\n\n```python\nprint('md-code-17')\n```\n",
        encoding="utf-8",
    )
    _write_image(root / "diagram.png", kind="chart")

    from docx import Document

    document = Document()
    document.add_paragraph("DOCX lifecycle fixture ordered paragraph docx-ordered-9.")
    table = document.add_table(rows=3, cols=2)
    for row, values in enumerate((("Metric", "Value"), ("Recall", "0.91"), ("Token", "docx-table-23"))):
        for col, value in enumerate(values):
            table.cell(row, col).text = value
    document.add_picture(str(root / "diagram.png"))
    document.save(root / "report.docx")

    import fitz

    native = fitz.open()
    page = native.new_page(width=600, height=400)
    page.insert_text((50, 80), "Native PDF lifecycle fixture native-pdf-11.")
    page.insert_text((50, 120), "Table 1: UA 72.72 and WA 71.90.")
    native.save(root / "native.pdf")
    native.close()

    _write_image(root / "table.png", kind="table")
    table_pdf = fitz.open()
    page = table_pdf.new_page(width=760, height=420)
    page.insert_image(fitz.Rect(0, 0, 760, 420), filename=str(root / "table.png"))
    table_pdf.save(root / "table-image.pdf")
    table_pdf.close()

    _write_image(root / "chart.png", kind="chart")
    mixed = fitz.open()
    page = mixed.new_page(width=600, height=400)
    page.insert_text((50, 80), "Mixed PDF native page mixed-native-31.")
    page = mixed.new_page(width=760, height=420)
    page.insert_image(fitz.Rect(0, 0, 760, 420), filename=str(root / "chart.png"))
    mixed.save(root / "mixed.pdf")
    mixed.close()

    # This file must be reported as a failed/partial source without deleting
    # the other documents or their active revisions.
    (root / "broken.pdf").write_bytes(b"not a pdf; lifecycle failure fixture")
    return {
        "txt": root / "notes.txt",
        "md": root / "research.md",
        "docx": root / "report.docx",
        "native_pdf": root / "native.pdf",
        "table_image_pdf": root / "table-image.pdf",
        "mixed_pdf": root / "mixed.pdf",
        "broken_pdf": root / "broken.pdf",
    }


async def _run_job(coordinator: DocumentProcessingService, job_id: str) -> dict[str, Any]:
    # A previous partial scan is deliberately retryable.  The durable queue
    # can therefore contain an older attempt before a newly enqueued snapshot;
    # drain until this specific job reaches a terminal state instead of
    # accidentally reporting the newly requested job as still pending.
    for _ in range(64):
        job = coordinator.get_job(job_id)
        if job is None:
            raise RuntimeError(f"job_missing:{job_id}")
        if job.status not in {"pending", "running"}:
            break
        ran = await coordinator.run_pending_once()
        if not ran:
            # A retryable projection uses exponential backoff.  Spinning a
            # fixed number of times can therefore report ``job_did_not_finish``
            # while the durable job is healthy and merely waiting for its
            # ``next_attempt_at``.  Follow the persisted schedule rather than
            # treating backoff as an ingestion failure.
            delay = 0.05
            current = coordinator.get_job(job_id)
            if current is not None and current.next_attempt_at:
                try:
                    retry_at = datetime.fromisoformat(current.next_attempt_at)
                    delay = max(
                        0.05,
                        min(
                            1.0,
                            (retry_at - datetime.now(UTC)).total_seconds(),
                        ),
                    )
                except (TypeError, ValueError):
                    delay = 0.05
            await asyncio.sleep(delay)
    else:
        raise RuntimeError(f"job_did_not_finish:{job_id}")
    job = coordinator.get_job(job_id)
    if job is None:
        raise RuntimeError(f"job_missing:{job_id}")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "result": job.result,
        "error_class": job.error_class,
        "error_message": job.error_message,
    }


def _documents(db_path: Path) -> list[dict[str, Any]]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, filename, file_type, active_revision_id, revision_status,
                   parse_status, index_status, chunk_count, table_count,
                   figure_count, raw_content_hash, metadata_json
            FROM documents ORDER BY filename
            """
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        metadata = json.loads(item.pop("metadata_json") or "{}")
        item["readiness"] = metadata.get("document_readiness", {})
        result.append(item)
    return result


def _retrieval_hit(result: dict[str, Any], expected: str) -> bool:
    needle = expected.casefold()
    return any(needle in str(item.get("content") or "").casefold() for item in result.get("results", []))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = args.runtime_root.expanduser().resolve()
    if runtime.exists() and args.clean:
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    fixture_root = runtime / "fixtures"
    fixtures = _make_fixtures(fixture_root)
    db_path = runtime / "sqlite" / "app.db"
    init_db(db_path)
    indexing = IndexingService(
        db_path=db_path,
        artifact_root=runtime / "artifacts",
        vision_model=None,
        paper_evidence_card_build_enabled=False,
    )
    embeddings = OllamaEmbeddingProvider(
        host=args.ollama_host,
        model=args.embedding_model,
        timeout_seconds=120,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    coordinator = DocumentProcessingService(
        db_path=db_path,
        indexing=indexing,
        lancedb_path=runtime / "lancedb",
        ollama_host=args.ollama_host,
        embedding_model=args.embedding_model,
        embedding_query_prefix=args.query_prefix,
        embedding_document_prefix=args.document_prefix,
        lightrag_enabled=False,
        lease_seconds=60,
    )

    # Warm the configured local provider before timing lifecycle work. Ollama
    # can take tens of seconds to materialize an unloaded embedding model; if
    # the first request races that load, a valid job is misclassified as a
    # projection failure. This preflight is observability only and never
    # selects a fallback provider.
    embedding_preflight_started = datetime.now(UTC)
    await embeddings.embed_query("format lifecycle embedding preflight")
    embedding_preflight_ms = round(
        (datetime.now(UTC) - embedding_preflight_started).total_seconds() * 1000,
        3,
    )

    file_types = ["pdf", "docx", "md", "txt"]
    first_job = coordinator.enqueue_folder(
        folder_path=str(fixture_root), recursive=False, file_types=file_types
    )
    first_run = await _run_job(coordinator, first_job)
    first_docs = _documents(db_path)
    first_revisions = {row["filename"]: row["active_revision_id"] for row in first_docs}

    # Unchanged rescan should reuse the durable job and all revision IDs.
    same_job = coordinator.enqueue_folder(
        folder_path=str(fixture_root), recursive=False, file_types=file_types
    )
    same_run = await _run_job(coordinator, same_job)
    same_docs = _documents(db_path)

    # Exact duplicate bytes create a source alias, not a second logical doc.
    duplicate_path = fixture_root / "notes-copy.txt"
    duplicate_path.write_bytes(fixtures["txt"].read_bytes())
    duplicate_job = coordinator.enqueue_folder(
        folder_path=str(fixture_root), recursive=False, file_types=file_types
    )
    duplicate_run = await _run_job(coordinator, duplicate_job)
    duplicate_docs = _documents(db_path)

    # Content change creates a new revision while retaining the logical ID.
    fixtures["txt"].write_text(
        "TXT lifecycle fixture changed. The retrieval token is cobalt-43.", encoding="utf-8"
    )
    changed_job = coordinator.enqueue_folder(
        folder_path=str(fixture_root), recursive=False, file_types=file_types
    )
    changed_run = await _run_job(coordinator, changed_job)
    changed_docs = _documents(db_path)
    changed_txt = next(row for row in changed_docs if row["filename"] == "notes.txt")

    # Rename the changed file; the raw hash should resolve the same logical
    # document and retain the canonical vector/revision.
    renamed = fixture_root / "notes-renamed.txt"
    fixtures["txt"].rename(renamed)
    renamed_job = coordinator.enqueue_folder(
        folder_path=str(fixture_root), recursive=False, file_types=file_types
    )
    renamed_run = await _run_job(coordinator, renamed_job)
    renamed_docs = _documents(db_path)

    store = LanceDBRetrievalStore(runtime / "lancedb")
    rag = RagService(db_path, artifact_root=runtime / "artifacts")
    queries = {
        "research.md": ("What is the Recall value in the markdown table?", "0.91"),
        "report.docx": ("What token is in the DOCX table?", "docx-table-23"),
        "native.pdf": ("What is the native PDF token?", "native-pdf-11"),
        "mixed.pdf": ("What token is visible in the mixed PDF image?", "text-ocr-77"),
        "table-image.pdf": ("What UA value appears in the image table?", "72.72"),
        "notes-renamed.txt": ("What is the changed TXT token?", "cobalt-43"),
    }
    retrieval_rows: list[dict[str, Any]] = []
    for filename, (query, expected) in queries.items():
        document = next((row for row in renamed_docs if row["filename"] == filename), None)
        if document is None:
            # A renamed source keeps the original logical document row in the
            # current schema, so resolve by the source alias below.
            with connect(db_path) as connection:
                row = connection.execute(
                    "SELECT d.id, d.filename FROM documents d JOIN document_sources s ON s.document_id = d.id WHERE s.source_path = ? ORDER BY s.last_seen_at DESC LIMIT 1",
                    (str(fixture_root / filename),),
                ).fetchone()
            document_id = str(row["id"]) if row else None
        else:
            document_id = str(document["id"])
        if not document_id:
            retrieval_rows.append({"filename": filename, "status": "missing_document"})
            continue
        result = await rag.search_hybrid(
            query=query,
            top_k=5,
            document_ids=[document_id],
            retrieval_store=store,
            embeddings=embeddings,
            rerank=False,
        )
        retrieval_rows.append(
            {
                "filename": filename,
                "document_id": document_id,
                "expected": expected,
                "hit": _retrieval_hit(result, expected),
                "channels": result.get("retrieval_channels"),
            }
        )

    with connect(db_path) as connection:
        source_aliases = connection.execute(
            "SELECT COUNT(*) FROM document_sources WHERE source_path LIKE ?",
            (str(fixture_root) + "%",),
        ).fetchone()[0]
        logical_documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        revision_counts = {
            row["filename"]: int(row["count"])
            for row in connection.execute(
                "SELECT d.filename, COUNT(r.revision_id) AS count FROM documents d LEFT JOIN document_revisions r ON r.document_id = d.id GROUP BY d.id"
            ).fetchall()
        }

    rows_by_file = {row["filename"]: row for row in renamed_docs}
    report = {
        "schema_version": "format-lifecycle-matrix-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_root": str(runtime),
        "embedding_model": args.embedding_model,
        "embedding_preflight_ms": embedding_preflight_ms,
        "jobs": {
            "initial": first_run,
            "unchanged_rescan": same_run,
            "duplicate_bytes": duplicate_run,
            "content_change": changed_run,
            "rename_move": renamed_run,
        },
        "invariants": {
            "initial_scan_partial_only_due_to_broken_source": first_run["status"] == "partial",
            "unchanged_revision_stable": all(
                first_revisions.get(name) == next((row["active_revision_id"] for row in same_docs if row["filename"] == name), None)
                for name in first_revisions
            ),
            "duplicate_did_not_create_logical_document": len(duplicate_docs) == len(first_docs),
            "changed_txt_kept_logical_document": changed_txt["id"] == next(row["id"] for row in first_docs if row["filename"] == "notes.txt"),
            "changed_txt_new_revision": revision_counts.get("notes.txt", 0) >= 2,
            "rename_kept_logical_document": any(row["id"] == changed_txt["id"] for row in renamed_docs),
            "all_active_rows_have_revision": all(bool(row["active_revision_id"]) for row in renamed_docs),
            "source_aliases_recorded": source_aliases >= len(first_docs),
        },
        "documents": list(rows_by_file.values()),
        "retrieval": retrieval_rows,
        "retrieval_all_hit": all(row.get("hit") for row in retrieval_rows),
        "logical_documents": logical_documents,
        "revision_counts": revision_counts,
        "notes": [
            "The broken PDF is intentionally included to verify partial scan isolation.",
            "OCR/visual provenance is reported by parser readiness; no VLM or 9router call is made.",
            "This is a bounded format/lifecycle gate, not evidence for arbitrary-document production readiness.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("/private/tmp/aya-format-lifecycle-matrix"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/retrieval_eval/internal/format-lifecycle-matrix.json")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model", default="embeddinggemma:300m")
    parser.add_argument("--query-prefix", default="task: search result | query: ")
    parser.add_argument("--document-prefix", default="title: none | text: ")
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.set_defaults(clean=True)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "jobs": report["jobs"],
        "invariants": report["invariants"],
        "retrieval_all_hit": report["retrieval_all_hit"],
        "retrieval": report["retrieval"],
    }, ensure_ascii=False, indent=2))
    required = (
        "unchanged_revision_stable",
        "duplicate_did_not_create_logical_document",
        "changed_txt_kept_logical_document",
        "changed_txt_new_revision",
        "rename_kept_logical_document",
        "all_active_rows_have_revision",
        "source_aliases_recorded",
    )
    return 0 if all(report["invariants"].get(key) for key in required) and report["retrieval_all_hit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
