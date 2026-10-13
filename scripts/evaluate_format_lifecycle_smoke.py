#!/usr/bin/env python3
"""Exercise the isolated local-search lifecycle on PDF/DOCX/MD/TXT fixtures.

The fixtures are generated locally and contain no external corpus.  This is a
cheap contract smoke for parser provenance, canonical artifacts, Lance
projection and hybrid retrieval; it deliberately makes no LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.db.sqlite import connect, init_db  # noqa: E402
from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.vector_index_service import VectorIndexService  # noqa: E402


def _retrieval_contract(
    *,
    retrieval_hit: bool,
    readiness: dict,
    projection_ok: bool,
) -> tuple[bool, bool]:
    """Separate searchable-text success from an honest non-searchable state.

    A scan/image-only page may be correctly classified as ``partial`` or
    ``visual_only`` while native/OCR text retrieval cannot find an arbitrary
    token. That is not a contract failure: the document must remain visibly
    non-ready and the visual channel is optional. Conversely, a page with
    native/OCR-ready text must produce the expected local hit.
    """

    state = str(readiness.get("state") or "").casefold()
    searchable_pages = int(readiness.get("native_text_pages") or 0) + int(
        readiness.get("ocr_text_pages") or 0
    )
    retrieval_expected = searchable_pages > 0
    if retrieval_expected:
        return retrieval_expected, bool(retrieval_hit)
    non_searchable_state = state in {"partial", "visual_only", "unresolved", "failed"}
    return retrieval_expected, bool(projection_ok and non_searchable_state)


def _make_fixtures(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text(
        "EmbeddingGemma local retrieval fixture. The answer token is cobalt-42.",
        encoding="utf-8",
    )
    (root / "research.md").write_text(
        "# Markdown fixture\n\nA table and local image belong to this document.\n\n"
        "| Metric | Value |\n|---|---:|\n| Recall | 0.91 |\n\n"
        "![pipeline diagram](diagram.png)\n\n```python\nprint('cobalt-42')\n```\n",
        encoding="utf-8",
    )
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (180, 90), "white")
    ImageDraw.Draw(image).text((12, 35), "cobalt-42", fill="black")
    image.save(root / "diagram.png")

    # A one-page image-only PDF exercises the selective OCR/readiness path
    # without making this cheap smoke depend on a large external corpus.
    import fitz

    pdf_path = root / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=260, height=140)
    page.insert_image(fitz.Rect(0, 0, 260, 140), filename=str(root / "diagram.png"))
    pdf.save(pdf_path)
    pdf.close()

    from docx import Document

    document = Document()
    document.add_paragraph("DOCX fixture with an ordered paragraph and table.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Recall"
    table.cell(1, 1).text = "0.91"
    document.add_picture(str(root / "diagram.png"))
    docx_path = root / "report.docx"
    document.save(docx_path)
    return [root / "notes.txt", root / "research.md", docx_path, pdf_path]


async def run(args: argparse.Namespace) -> dict:
    runtime = args.runtime_root.resolve()
    fixtures = _make_fixtures(runtime / "fixtures")
    db_path = runtime / "sqlite" / "app.db"
    artifact_root = runtime / "artifacts"
    lancedb_path = runtime / "lancedb"
    init_db(db_path)
    indexing = IndexingService(db_path, artifact_root=artifact_root)
    embeddings = OllamaEmbeddingProvider(
        host=args.ollama_host,
        model=args.embedding_model,
        timeout_seconds=120,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    store = LanceDBRetrievalStore(lancedb_path)
    vector = VectorIndexService(db_path, store, embeddings)
    rag = RagService(db_path, artifact_root=artifact_root)
    rows = []
    for path in fixtures:
        indexed = indexing.index_file(source_path=str(path))
        projection = await vector.index_document(indexed.id, revision_id=indexed.revision_id)
        query = "What does the ordered paragraph say?" if path.suffix == ".docx" else "What is the cobalt-42 fixture token?"
        result = await rag.search_hybrid(
            query=query,
            top_k=5,
            document_ids=[indexed.id],
            retrieval_store=store,
            embeddings=embeddings,
            rerank=False,
        )
        with connect(db_path) as connection:
            doc = connection.execute(
                "SELECT filename, file_type, chunk_count, table_count, figure_count, active_revision_id, metadata_json FROM documents WHERE id = ?",
                (indexed.id,),
            ).fetchone()
        metadata = json.loads(doc["metadata_json"] or "{}")
        readiness = metadata.get("document_readiness") or {}
        retrieval_hit = any(
            ("ordered paragraph" if path.suffix == ".docx" else "cobalt-42")
            in str(item.get("content") or "").casefold()
            for item in result.get("results", [])
        )
        retrieval_expected, retrieval_contract_ok = _retrieval_contract(
            retrieval_hit=retrieval_hit,
            readiness=readiness,
            projection_ok=bool(projection.get("ok")),
        )
        rows.append(
            {
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "revision_id": doc["active_revision_id"],
                "chunk_count": doc["chunk_count"],
                "table_count": doc["table_count"],
                "figure_count": doc["figure_count"],
                "readiness": readiness.get("state"),
                "projection_ok": bool(projection.get("ok")),
                "retrieval_hit": retrieval_hit,
                "retrieval_expected": retrieval_expected,
                "retrieval_contract_ok": retrieval_contract_ok,
            }
        )
    return {
        "schema_version": "format-lifecycle-smoke-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": args.embedding_model,
        "cases": len(rows),
        "all_projection_ok": all(row["projection_ok"] for row in rows),
        # Keep raw text-hit visibility for diagnostics. A non-searchable
        # partial/visual-only page is accepted through its readiness contract,
        # not silently counted as a hit.
        "all_retrieval_hits": all(row["retrieval_hit"] for row in rows),
        "all_expected_retrieval_hits": all(
            row["retrieval_hit"]
            for row in rows
            if row["retrieval_expected"]
        ),
        "all_retrieval_contracts_ok": all(row["retrieval_contract_ok"] for row in rows),
        "rows": rows,
        "note": "Generated local fixtures only; no 9router/LLM calls and no production paths.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("/private/tmp/aya-format-lifecycle-smoke"))
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model", default="embeddinggemma:300m")
    parser.add_argument("--query-prefix", default="task: search result | query: ")
    parser.add_argument("--document-prefix", default="title: none | text: ")
    parser.add_argument("--output", type=Path, default=Path("data/retrieval_eval/internal/format-lifecycle-smoke.json"))
    args = parser.parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0
        if report["all_projection_ok"] and report["all_retrieval_contracts_ok"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
