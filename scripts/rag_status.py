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


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})")
    }


def main() -> None:
    settings = get_settings()
    db_path = settings.sqlite_db_path
    print(f"sqlite: {db_path}")
    print(f"lancedb: {settings.lancedb_path}")
    print(f"artifacts: {settings.artifacts_path}")
    print(
        "llm: "
        f"provider={settings.llm_provider}, base={settings.openai_api_base}, "
        f"model={settings.default_model}"
    )
    print(
        "embedding: "
        f"provider=ollama, host={settings.ollama_host}, "
        f"model={settings.embedding_model}, dim={settings.embedding_dim}"
    )
    print(
        "flags: "
        f"dual_query_rrf={settings.dual_query_rrf_enabled}, "
        f"visual_retrieval={settings.visual_retrieval_enabled}, "
        f"lightrag={settings.lightrag_enabled}"
    )
    print()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        projection_tables = ("chunks", "document_tables", "document_figures")
        documents_has_active_revision = "active_revision_id" in _table_columns(
            connection, "documents"
        )
        sqlite_revision_schema = all(
            "revision_id" in _table_columns(connection, table)
            for table in projection_tables
        ) and documents_has_active_revision
        print(
            "revision schema: "
            + ("physical-active filtering available" if sqlite_revision_schema else
               "legacy projection (migration pending; counts are compatibility totals)")
        )
        summary = connection.execute(
            """
            SELECT
              COUNT(*) AS documents,
              SUM(file_type = 'pdf') AS pdf_documents,
              SUM(file_type = 'pdf' AND figure_count > 0) AS pdf_with_figures,
              SUM(figure_count) AS figures,
              SUM(table_count) AS tables
            FROM documents
            """
        ).fetchone()
        print(
            "documents: "
            f"total={summary['documents']}, pdf={summary['pdf_documents']}, "
            f"pdf_with_figures={summary['pdf_with_figures']}, "
            f"figures={summary['figures']}, tables={summary['tables']}"
        )

        # Counts used to be read from the legacy tables without checking the
        # document's active revision.  That made a status dump look healthy
        # after a staged/retired rebuild even though retrieval correctly
        # ignored those rows.  Keep both views visible, but make active rows
        # the operational default and explicitly label the legacy aggregate.
        if sqlite_revision_schema:
            active_summary = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM chunks c JOIN documents d
                     ON d.id = c.document_id
                    WHERE (d.active_revision_id IS NULL AND c.revision_id IS NULL)
                       OR c.revision_id = d.active_revision_id) AS chunks,
                  (SELECT COUNT(*) FROM document_tables t JOIN documents d
                     ON d.id = t.document_id
                    WHERE (d.active_revision_id IS NULL AND t.revision_id IS NULL)
                       OR t.revision_id = d.active_revision_id) AS tables,
                  (SELECT COUNT(*) FROM document_figures f JOIN documents d
                     ON d.id = f.document_id
                    WHERE (d.active_revision_id IS NULL AND f.revision_id IS NULL)
                       OR f.revision_id = d.active_revision_id) AS figures,
                  (SELECT COUNT(*) FROM document_figures f JOIN documents d
                     ON d.id = f.document_id
                    WHERE ((d.active_revision_id IS NULL AND f.revision_id IS NULL)
                        OR f.revision_id = d.active_revision_id)
                      AND f.visual_summary IS NOT NULL
                      AND f.visual_summary != '') AS visual_summaries
                """
            ).fetchone()
        else:
            # The production DB is intentionally not migrated until an
            # isolated RC passes.  Keep this command useful on the legacy
            # schema while making the weaker guarantee explicit above.
            active_summary = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM chunks) AS chunks,
                  (SELECT COUNT(*) FROM document_tables) AS tables,
                  (SELECT COUNT(*) FROM document_figures) AS figures,
                  (SELECT COUNT(*) FROM document_figures
                    WHERE visual_summary IS NOT NULL AND visual_summary != '')
                    AS visual_summaries
                """
            ).fetchone()
        print(
            "canonical projections (active when revision schema exists): "
            f"chunks={active_summary['chunks']}, "
            f"tables={active_summary['tables']}, "
            f"figures={active_summary['figures']}, "
            f"visual_summaries={active_summary['visual_summaries']}"
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

        readiness_rows = connection.execute(
            """
            SELECT
              COALESCE(json_extract(d.metadata_json, '$.document_readiness.state'),
                       'unknown') AS readiness_state,
              COUNT(*) AS documents
            FROM documents d
            GROUP BY readiness_state
            ORDER BY readiness_state
            """
        ).fetchall()
        print("readiness:")
        for row in readiness_rows:
            print(f"- {row['readiness_state']}: documents={row['documents']}")

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
            total_rows = table.count_rows()
            print(f"- {name}: rows_total={total_rows}")

        # A physical Lance table can contain retired revisions during the
        # migration window.  Report the active count separately so this
        # operational command agrees with retrieval's revision filter.
        active_revisions: list[tuple[str, str]] = []
        if documents_has_active_revision:
            with sqlite3.connect(db_path) as connection:
                active_revisions = [
                    (str(row[0]), str(row[1]))
                    for row in connection.execute(
                        """
                        SELECT id, active_revision_id
                        FROM documents
                        WHERE active_revision_id IS NOT NULL
                        """
                    )
                    if row[1]
                ]
        active_counts: dict[str, int] = {}
        # SQLite and LanceDB evolve independently during the migration window.
        # Do not infer Lance capability from the SQLite schema: an existing
        # Lance table may still lack the physical revision_id column even when
        # SQLite has already been backfilled.  In that case retrieval can use
        # the documented legacy metadata guard, but status must not claim that
        # a physical active-revision count was validated.
        lance_revision_schema: dict[str, bool] = {}
        for name in ["document_cards", "text_chunks", "table_chunks", "figure_chunks"]:
            if name in table_names:
                try:
                    lance_revision_schema[name] = "revision_id" in set(
                        store._db.open_table(name).schema.names
                    )
                except Exception:
                    lance_revision_schema[name] = False
        if sqlite_revision_schema and active_revisions and lance_revision_schema and all(
            lance_revision_schema.values()
        ):
            for name in ["document_cards", "text_chunks", "table_chunks", "figure_chunks"]:
                if name not in table_names:
                    continue
                count = 0
                for document_id, revision_id in active_revisions:
                    try:
                        per_revision = store.count_document_revision(document_id, revision_id)
                    except Exception:
                        per_revision = None
                    if not per_revision:
                        continue
                    count += int(per_revision.get(name, 0) or 0)
                active_counts[name] = count
        if active_counts:
            print("active revision rows:")
            for name, count in active_counts.items():
                print(f"- {name}: rows_active={count}")
        else:
            legacy_tables = [name for name, physical in lance_revision_schema.items() if not physical]
            if legacy_tables:
                print(
                    "active revision rows: unavailable "
                    "(legacy Lance projection; physical revision filter pending for "
                    + ", ".join(legacy_tables)
                    + ")"
                )
            elif not sqlite_revision_schema:
                print("active revision rows: unavailable (legacy SQLite schema)")
            else:
                print("active revision rows: unavailable (no active revisions or vector tables)")
    except LanceDBUnavailable as exc:
        print(f"- unavailable: {exc}")


if __name__ == "__main__":
    main()
