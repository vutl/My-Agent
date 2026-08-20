from __future__ import annotations

import asyncio
import json

from app.db.sqlite import connect, init_db
from app.lightrag.provenance import (
    resolve_lightrag_chunk_parents,
    sync_document_chunk_records,
    sync_document_provenance,
)


def _seed_document(
    db_path,
    *,
    document_id: str = "doc-1",
    source_path: str = "/papers/paper.pdf",
    filename: str = "paper.pdf",
    parents: tuple[str, ...] = (
        "First parent contains architecture encoder attention details.",
        "Second parent contains benchmark accuracy and F1 results.",
    ),
) -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO indexed_folders
                (id, folder_path, recursive, file_types, created_at, updated_at)
            VALUES ('folder-1', '/papers', 0, '["pdf"]', 'now', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO documents (
                id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count
            ) VALUES (?, 'folder-1', ?, ?, 'pdf', ?, 'now', 'now', ?)
            """,
            (document_id, source_path, filename, f"hash-{document_id}", len(parents)),
        )
        for index, parent_content in enumerate(parents):
            parent_id = f"{document_id}-parent-{index}"
            connection.execute(
                """
                INSERT INTO chunks (
                    id, document_id, chunk_index, content, source_path,
                    filename, chunk_type, parent_chunk_id, created_at,
                    page_number, heading_path_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'text', ?, 'now', ?, ?, ?)
                """,
                (
                    f"{document_id}-child-{index}",
                    document_id,
                    index,
                    parent_content[:25],
                    source_path,
                    filename,
                    parent_id,
                    index + 1,
                    json.dumps([f"Section {index + 1}"]),
                    json.dumps(
                        {
                            "parent_content": parent_content,
                            "page_number": index + 1,
                            "section_title": f"Section {index + 1}",
                        }
                    ),
                ),
            )


def _record(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "doc-1",
    file_path: str = "paper.pdf",
    order: int = 0,
) -> dict:
    return {
        "_id": chunk_id,
        "content": content,
        "full_doc_id": document_id,
        "file_path": file_path,
        "chunk_order_index": order,
    }


def test_exact_cross_parent_mapping_and_resolver_scope(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    parents = (
        "Alpha architecture encoder context.",
        "Beta benchmark accuracy evidence.",
    )
    _seed_document(db_path, parents=parents)
    content = f"{parents[0]}\n\n{parents[1]}"

    result = sync_document_chunk_records(
        db_path,
        "doc-1",
        [_record("lr-chunk-1", content)],
    )

    assert result.records_mapped == 1
    assert result.mappings_written == 2
    resolved = resolve_lightrag_chunk_parents(db_path, ["lr-chunk-1"])
    assert [item.parent_content for item in resolved["lr-chunk-1"]] == list(parents)
    assert {item.mapping_method for item in resolved["lr-chunk-1"]} == {
        "exact_offset"
    }
    assert [
        item.overlap_chars for item in resolved["lr-chunk-1"]
    ] == sorted(
        [len(parent) for parent in parents],
        reverse=True,
    )
    assert {
        (
            item.page_number,
            item.section_title,
            item.heading_path,
            item.chunk_type,
        )
        for item in resolved["lr-chunk-1"]
    } == {
        (1, "Section 1", ("Section 1",), "text"),
        (2, "Section 2", ("Section 2",), "text"),
    }
    assert resolve_lightrag_chunk_parents(
        db_path,
        ["lr-chunk-1"],
        allowed_document_ids=["another-doc"],
    ) == {"lr-chunk-1": []}


def test_normalized_whitespace_match_must_be_unique(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    parents = (
        "Unique phrase with several words for normalized matching.",
        "Other parent has unrelated benchmark evidence.",
    )
    _seed_document(db_path, parents=parents)

    result = sync_document_chunk_records(
        db_path,
        "doc-1",
        [
            _record(
                "normalized",
                "Unique   phrase\nwith several words for normalized matching.",
            )
        ],
    )

    assert result.records_mapped == 1
    resolved = resolve_lightrag_chunk_parents(db_path, ["normalized"])
    assert resolved["normalized"][0].mapping_method == "normalized_whitespace_unique"

    repeated = (
        "Repeated five token phrase lives here.",
        "Repeated five token phrase lives here.",
    )
    other_db = tmp_path / "ambiguous.db"
    _seed_document(other_db, parents=repeated)
    rejected = sync_document_chunk_records(
        other_db,
        "doc-1",
        [_record("ambiguous", "Repeated five token phrase lives here.")],
    )
    assert rejected.records_mapped == 0
    assert rejected.rejection_reasons == {
        "content_not_unique_or_shingle_ambiguous": 1
    }


def test_legacy_shingle_mapping_requires_strong_clear_top_parent(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    parents = (
        "Architecture encoder combines acoustic features with attention layers "
        "before the final classifier predicts emotion labels.",
        "Benchmark evaluation reports accuracy precision recall and F1 across "
        "several speech emotion datasets.",
    )
    _seed_document(db_path, parents=parents)
    legacy_chunk = (
        "Earlier extraction said the architecture encoder combines acoustic "
        "features with attention layers before the final classifier."
    )

    result = sync_document_chunk_records(
        db_path,
        "doc-1",
        [_record("legacy", legacy_chunk)],
    )

    assert result.records_mapped == 1
    assert result.mappings_written == 1
    resolved = resolve_lightrag_chunk_parents(db_path, ["legacy"])["legacy"]
    assert len(resolved) == 1
    assert resolved[0].parent_chunk_id == "doc-1-parent-0"
    assert resolved[0].mapping_method == "token_5_shingle_top1"
    assert resolved[0].mapping_score >= 0.55
    assert resolved[0].document_char_start == -1
    assert resolved[0].document_char_end == -1
    assert resolved[0].overlap_chars == 0


def test_shingle_near_tie_is_rejected_fail_closed(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    parents = (
        "Shared architecture encoder attention layer predicts emotion labels alpha.",
        "Shared architecture encoder attention layer predicts emotion labels beta.",
    )
    _seed_document(db_path, parents=parents)

    result = sync_document_chunk_records(
        db_path,
        "doc-1",
        [
            _record(
                "near-tie",
                "Old text: shared architecture encoder attention layer predicts "
                "emotion labels in speech.",
            )
        ],
    )

    assert result.records_mapped == 0
    assert result.rejection_reasons == {
        "content_not_unique_or_shingle_ambiguous": 1
    }


def test_canonical_resolution_never_uses_focus_or_ambiguous_filename(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    _seed_document(
        db_path,
        document_id="doc-1",
        source_path="/one/paper.pdf",
        filename="paper.pdf",
    )
    _seed_document(
        db_path,
        document_id="doc-2",
        source_path="/two/paper.pdf",
        filename="paper.pdf",
    )
    content = "First parent contains architecture encoder attention details."

    ambiguous = sync_document_chunk_records(
        db_path,
        "doc-1",
        [
            _record(
                "ambiguous-path",
                content,
                document_id="legacy-id",
                file_path="paper.pdf",
            )
        ],
    )
    assert ambiguous.records_mapped == 0
    assert ambiguous.rejection_reasons == {"ambiguous_exact_path": 1}

    exact_path = sync_document_chunk_records(
        db_path,
        "doc-1",
        [
            _record(
                "exact-path",
                content,
                document_id="legacy-id",
                file_path="/one/paper.pdf",
            )
        ],
    )
    assert exact_path.records_mapped == 1
    resolved = resolve_lightrag_chunk_parents(db_path, ["exact-path"])
    assert resolved["exact-path"][0].document_id == "doc-1"
    assert resolved["exact-path"][0].canonical_method == "exact_source_path"

    foreign = sync_document_chunk_records(
        db_path,
        "doc-1",
        [
            _record(
                "foreign",
                content,
                document_id="doc-2",
                file_path="/two/paper.pdf",
            )
        ],
    )
    assert foreign.records_mapped == 0
    assert foreign.rejection_reasons == {"foreign_document": 1}


def test_rebuild_invalidates_old_rows_and_resolver_rejects_stale_parent(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    content = "First parent contains architecture encoder attention details."
    _seed_document(db_path)
    sync_document_chunk_records(
        db_path,
        "doc-1",
        [_record("old", content)],
    )

    replacement = sync_document_chunk_records(
        db_path,
        "doc-1",
        [_record("new", content)],
    )
    assert replacement.records_mapped == 1
    assert resolve_lightrag_chunk_parents(db_path, ["old"]) == {"old": []}

    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT metadata_json
            FROM chunks
            WHERE parent_chunk_id = 'doc-1-parent-0'
            LIMIT 1
            """
        ).fetchone()
        metadata = json.loads(row["metadata_json"])
        metadata["parent_content"] = "Changed parent invalidates durable hash."
        connection.execute(
            """
            UPDATE chunks
            SET metadata_json = ?
            WHERE parent_chunk_id = 'doc-1-parent-0'
            """,
            (json.dumps(metadata),),
        )
    assert resolve_lightrag_chunk_parents(db_path, ["new"]) == {"new": []}


def test_async_storage_sync_reads_kv_only_and_schema_cascades(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    content = "First parent contains architecture encoder attention details."
    _seed_document(db_path)

    class StatusStorage:
        async def get_by_id(self, document_id):
            assert document_id == "doc-1"
            return {
                "status": "processed",
                "chunks_list": ["lr-1"],
                "chunks_count": 1,
            }

    class TextStorage:
        async def get_by_ids(self, chunk_ids):
            assert chunk_ids == ["lr-1"]
            return [_record("lr-1", content)]

    class Rag:
        doc_status = StatusStorage()
        text_chunks = TextStorage()

    result = asyncio.run(sync_document_provenance(db_path, Rag(), "doc-1"))
    assert result.records_mapped == 1
    with connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM lightrag_chunk_parent_provenance"
        ).fetchone()[0]
        assert count == 1
        connection.execute("DELETE FROM documents WHERE id = 'doc-1'")
        count = connection.execute(
            "SELECT COUNT(*) FROM lightrag_chunk_parent_provenance"
        ).fetchone()[0]
        assert count == 0
