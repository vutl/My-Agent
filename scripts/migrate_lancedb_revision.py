#!/usr/bin/env python3
"""Build a revision-isolated LanceDB projection without mutating production.

The existing index is deliberately opened read-only from the caller's point of
view.  A new directory is built, validated against SQLite's active revisions,
and can be switched by the deployment layer after an external smoke test.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Iterator

TABLES = ("document_cards", "text_chunks", "table_chunks", "figure_chunks")

@contextmanager
def _read_only_sqlite(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the authority DB without running migrations or allowing writes."""

    resolved = db_path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"sqlite_db_missing:{resolved}")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _table_names(db) -> list[str]:
    response = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    return list(getattr(response, "tables", response))


def _active_revisions(db_path: Path) -> dict[str, str]:
    with _read_only_sqlite(db_path) as connection:
        return {
            str(row["id"]): str(row["active_revision_id"])
            for row in connection.execute(
                "SELECT id, active_revision_id FROM documents WHERE active_revision_id IS NOT NULL"
            ).fetchall()
        }


def _validate_active_revision_db(
    db_path: Path,
    *,
    source_has_rows: bool,
) -> dict[str, str]:
    """Fail closed when the migration is pointed at the wrong SQLite DB.

    A Lance source with rows but no active SQLite revisions is almost always a
    path/configuration mistake (for example, the unrelated legacy
    ``data/rag.db``).  Copying those rows with blank revision IDs would create
    a projection that looks healthy but cannot satisfy active-revision
    isolation.  Empty source and empty database are valid for a no-op audit;
    a non-empty source requires at least one authoritative active revision.
    """

    active = _active_revisions(db_path)
    if source_has_rows and not active:
        raise RuntimeError(
            "active_revision_db_empty: source Lance tables contain rows but "
            f"{db_path} has no active document revisions"
        )
    return active


def _canonical_ids(db_path: Path, table_name: str) -> dict[str, tuple[str, str]]:
    """Map legacy vector IDs to the active canonical revision.

    Legacy Lance rows predate the physical revision column.  They are only
    attributable during migration when their artifact ID still exists in the
    active SQLite projection; otherwise the row is dropped fail-closed.
    """

    table_map = {
        "document_cards": "document_cards",
        "text_chunks": "chunks",
        "table_chunks": "document_tables",
        "figure_chunks": "document_figures",
    }
    source = table_map.get(table_name)
    if source is None:
        return {}
    with _read_only_sqlite(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT {source}.id, {source}.document_id, documents.active_revision_id
            FROM {source}
            JOIN documents ON documents.id = {source}.document_id
            WHERE documents.active_revision_id IS NOT NULL
              AND ({source}.revision_id = documents.active_revision_id
                   OR {source}.revision_id IS NULL)
            """
        ).fetchall()
    prefix = {
        "document_cards": "card",
        "text_chunks": "text",
        "table_chunks": "table",
        "figure_chunks": "figure",
    }[table_name]
    result: dict[str, tuple[str, str]] = {}
    for row in rows:
        identity = (str(row["document_id"]), str(row["active_revision_id"]))
        artifact_id = str(row["id"])
        result[artifact_id] = identity
        result[f"{prefix}:{artifact_id}"] = identity
        if table_name == "document_cards":
            result[str(row["document_id"])] = identity
    return result


def _validate_staged_projection(db, active: dict[str, str]) -> dict[str, object]:
    """Validate physical revision isolation before the staging directory moves.

    Migration is intentionally fail-closed: a copied row without a physical
    ``revision_id`` or without an active SQLite owner is not a usable RC,
    even if it can be filtered through legacy metadata later.  This check
    reads only the staging LanceDB and the already-authoritative ``active``
    mapping; it never modifies either store.
    """

    table_names = set(_table_names(db))
    counts: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    invalid: list[dict[str, str]] = []
    for name in TABLES:
        if name not in table_names:
            counts[name] = 0
            continue
        table = db.open_table(name)
        try:
            schema_names = set(table.schema.names)
        except Exception as exc:  # pragma: no cover - Lance version dependent
            raise RuntimeError(f"staged_schema_unreadable:{name}:{exc}") from exc
        if "revision_id" not in schema_names:
            raise RuntimeError(f"staged_revision_column_missing:{name}")
        try:
            vector_type = table.schema.field("vector").type
            list_size = getattr(vector_type, "list_size", None)
            if list_size is not None:
                dimensions[name] = int(list_size)
        except Exception:
            # A malformed vector schema is reported by the row/readability
            # checks below; do not manufacture a dimension value.
            pass
        rows = table.to_arrow().to_pylist()
        counts[name] = len(rows)
        for row in rows:
            document_id = str(row.get("document_id") or "").strip()
            revision_id = str(row.get("revision_id") or "").strip()
            expected = active.get(document_id)
            if not document_id or not revision_id:
                invalid.append(
                    {
                        "table": name,
                        "id": str(row.get("id") or ""),
                        "reason": "missing_document_or_revision",
                    }
                )
            elif expected is None:
                invalid.append(
                    {
                        "table": name,
                        "id": str(row.get("id") or ""),
                        "reason": "document_not_active",
                    }
                )
            elif revision_id != expected:
                invalid.append(
                    {
                        "table": name,
                        "id": str(row.get("id") or ""),
                        "reason": "revision_not_active",
                    }
                )
    if invalid:
        sample = ",".join(
            f"{item['table']}:{item['id']}:{item['reason']}"
            for item in invalid[:5]
        )
        raise RuntimeError(
            f"staged_revision_validation_failed:{len(invalid)}:{sample}"
        )
    nonzero_dimensions = set(dimensions.values())
    if len(nonzero_dimensions) > 1:
        raise RuntimeError(
            "staged_vector_dimension_mismatch:"
            + ",".join(f"{name}={value}" for name, value in sorted(dimensions.items()))
        )
    return {
        "physical_revision_tables": sorted(
            name for name in TABLES if name in table_names
        ),
        "rows": counts,
        "vector_dimensions": dimensions,
        "invalid_rows": 0,
    }


def migrate(source: Path, output: Path, db_path: Path) -> dict:
    try:
        import lancedb  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("LanceDB is not installed") from exc
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging: Path | None = None
    copied: dict[str, int] = {}
    skipped: dict[str, int] = {}
    try:
        src = lancedb.connect(str(source))
        source_tables = set(_table_names(src))
        source_has_rows = any(
            name in source_tables and int(src.open_table(name).count_rows()) > 0
            for name in TABLES
        )
        active = _validate_active_revision_db(db_path, source_has_rows=source_has_rows)
        staging = Path(tempfile.mkdtemp(prefix=f"{output.name}.", dir=str(output.parent)))
        dst = lancedb.connect(str(staging))
        for name in TABLES:
            if name not in source_tables:
                continue
            table = src.open_table(name)
            rows = table.to_arrow().to_pylist()
            canonical_ids = _canonical_ids(db_path, name)
            keep: list[dict] = []
            dropped = 0
            for row in rows:
                metadata = {}
                raw = row.get("metadata_json")
                if isinstance(raw, str) and raw:
                    try:
                        metadata = json.loads(raw)
                    except json.JSONDecodeError:
                        metadata = {}
                document_id = str(row.get("document_id") or metadata.get("document_id") or "")
                revision_id = str(row.get("revision_id") or metadata.get("revision_id") or "")
                expected = active.get(document_id)
                if expected and revision_id and revision_id != expected:
                    dropped += 1
                    continue
                if expected and not revision_id:
                    # Legacy rows have no revision column. Attribute only rows
                    # whose artifact ID is present in the active SQLite
                    # projection; arbitrary same-document vectors are stale-
                    # ambiguous and must not enter the new index.
                    canonical = canonical_ids.get(str(row.get("id") or ""))
                    if canonical is None and name == "document_cards":
                        canonical = canonical_ids.get(document_id)
                    if canonical is None or canonical != (document_id, expected):
                        dropped += 1
                        continue
                    revision_id = expected
                row["revision_id"] = revision_id
                keep.append(row)
            if keep:
                dst.create_table(name, keep)
            copied[name] = len(keep)
            skipped[name] = dropped
        validation = _validate_staged_projection(dst, active)
        staging.rename(output)
        staging = None
        return {
            "source": str(source),
            "output": str(output),
            "active_documents": len(active),
            "copied": copied,
            "skipped_stale_or_unattributed": skipped,
            "validation": validation,
        }
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.output, args.db), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
