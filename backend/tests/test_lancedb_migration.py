from __future__ import annotations

import pytest
import sqlite3
from types import SimpleNamespace

from app.db.sqlite import init_db
from app.services.indexing_service import IndexingService
from scripts.migrate_lancedb_revision import (
    _read_only_sqlite,
    _validate_staged_projection,
    _validate_active_revision_db,
)


class _FakeArrow:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class _FakeTable:
    def __init__(self, rows, *, revision_column=True):
        names = {"id", "document_id", "vector"}
        if revision_column:
            names.add("revision_id")
        self.schema = SimpleNamespace(
            names=names,
            field=lambda _name: SimpleNamespace(
                type=SimpleNamespace(list_size=768)
            ),
        )
        self._rows = rows

    def to_arrow(self):
        return _FakeArrow(self._rows)


class _FakeDb:
    def __init__(self, tables):
        self._tables = tables

    def list_tables(self):
        return list(self._tables)

    def open_table(self, name):
        return self._tables[name]


def test_migration_fails_closed_for_nonempty_source_and_wrong_empty_db(tmp_path) -> None:
    db_path = tmp_path / "wrong.db"
    init_db(db_path)

    with pytest.raises(RuntimeError, match="active_revision_db_empty"):
        _validate_active_revision_db(db_path, source_has_rows=True)


def test_migration_allows_empty_source_without_active_documents(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    init_db(db_path)

    assert _validate_active_revision_db(db_path, source_has_rows=False) == {}


def test_migration_accepts_authoritative_active_revision(tmp_path) -> None:
    db_path = tmp_path / "active.db"
    source = tmp_path / "note.md"
    source.write_text("authoritative active revision", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))

    active = _validate_active_revision_db(db_path, source_has_rows=True)

    assert active[indexed.id] == indexed.revision_id


def test_migration_authority_connection_is_read_only(tmp_path) -> None:
    db_path = tmp_path / "authority.db"
    init_db(db_path)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with _read_only_sqlite(db_path) as connection:
            connection.execute("CREATE TABLE should_not_exist(id INTEGER)")


def test_staged_projection_validation_requires_physical_revision_column() -> None:
    db = _FakeDb(
        {
            "text_chunks": _FakeTable(
                [{"id": "chunk", "document_id": "doc", "vector": [0.0]}],
                revision_column=False,
            )
        }
    )

    with pytest.raises(RuntimeError, match="staged_revision_column_missing:text_chunks"):
        _validate_staged_projection(db, {"doc": "rev-1"})


def test_staged_projection_validation_rejects_non_active_rows() -> None:
    db = _FakeDb(
        {
            "text_chunks": _FakeTable(
                [
                    {
                        "id": "chunk",
                        "document_id": "doc",
                        "revision_id": "retired",
                        "vector": [0.0],
                    }
                ]
            )
        }
    )

    with pytest.raises(RuntimeError, match="staged_revision_validation_failed"):
        _validate_staged_projection(db, {"doc": "active"})


def test_staged_projection_validation_reports_physical_counts_and_dimensions() -> None:
    db = _FakeDb(
        {
            "text_chunks": _FakeTable(
                [
                    {
                        "id": "chunk",
                        "document_id": "doc",
                        "revision_id": "active",
                        "vector": [0.0],
                    }
                ]
            )
        }
    )

    result = _validate_staged_projection(db, {"doc": "active"})

    assert result["invalid_rows"] == 0
    assert result["rows"]["text_chunks"] == 1
    assert result["vector_dimensions"]["text_chunks"] == 768
