#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable  # noqa: E402


def main() -> None:
    settings = get_settings()
    db_path = settings.sqlite_db_path
    print(f"sqlite: {db_path}")
    print(f"lancedb: {settings.lancedb_path}")
    print(f"artifacts: {settings.artifacts_path}")
    print()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        summary = connection.execute(
            """
            SELECT
              COUNT(*) AS documents,
              SUM(file_type = 'pdf') AS pdf_documents,
              SUM(parser_name = 'pypdf+pymupdf') AS visual_pdf_documents,
              SUM(figure_count) AS figures,
              SUM(table_count) AS tables
            FROM documents
            """
        ).fetchone()
        print(
            "documents: "
            f"total={summary['documents']}, pdf={summary['pdf_documents']}, "
            f"visual_pdf={summary['visual_pdf_documents']}, "
            f"figures={summary['figures']}, tables={summary['tables']}"
        )

        artifact_summary = connection.execute(
            """
            SELECT
              COUNT(*) AS figure_rows,
              SUM(image_path IS NOT NULL AND image_path != '') AS figure_images,
              SUM(visual_summary IS NOT NULL AND visual_summary != '') AS visual_summaries
            FROM document_figures
            """
        ).fetchone()
        print(
            "figure artifacts: "
            f"rows={artifact_summary['figure_rows']}, "
            f"images={artifact_summary['figure_images']}, "
            f"visual_summaries={artifact_summary['visual_summaries']}"
        )

        print("\nrecent documents:")
        for row in connection.execute(
            """
            SELECT filename, parser_name, chunk_count, table_count, figure_count, indexed_at
            FROM documents
            ORDER BY indexed_at DESC
            LIMIT 20
            """
        ):
            print(
                f"- {row['filename']} | {row['parser_name']} | "
                f"chunks={row['chunk_count']} tables={row['table_count']} "
                f"figures={row['figure_count']} | {row['indexed_at']}"
            )

        print("\ncollections:")
        for row in connection.execute(
            """
            SELECT collections.name, COUNT(collection_documents.document_id) AS documents
            FROM collections
            LEFT JOIN collection_documents ON collection_documents.collection_id = collections.id
            GROUP BY collections.id, collections.name
            ORDER BY collections.updated_at DESC
            """
        ):
            print(f"- {row['name']} | documents={row['documents']}")

    artifact_files = list(settings.artifacts_path.rglob("*")) if settings.artifacts_path.exists() else []
    print(f"\nartifact files: {sum(1 for path in artifact_files if path.is_file())}")
    print("\nvector tables:")
    try:
        store = LanceDBRetrievalStore(settings.lancedb_path)
        table_names = list(getattr(store._db.list_tables(), "tables", store._db.list_tables()))
        for name in ["document_cards", "text_chunks", "table_chunks", "figure_chunks"]:
            if name not in table_names:
                print(f"- {name}: missing")
                continue
            table = store._db.open_table(name)
            print(f"- {name}: rows={table.count_rows()}")
    except LanceDBUnavailable as exc:
        print(f"- unavailable: {exc}")


if __name__ == "__main__":
    main()
