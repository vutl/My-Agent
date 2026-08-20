import asyncio
import sqlite3

import pytest

from app.db.sqlite import init_db
from app.rag.embeddings import HashEmbeddingProvider
from app.retrieval_store.base import RetrievalFilter
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService
from app.services.vector_index_service import VectorIndexService


class FakeRetrievalStore:
    def __init__(self) -> None:
        self.document_cards = []
        self.text_chunks = []
        self.table_chunks = []
        self.figure_chunks = []
        self.deleted_documents = []
        self.pruned_document_sets = []

    async def add_document_cards(self, records) -> None:
        self.document_cards.extend(records)

    async def add_text_chunks(self, records) -> None:
        self.text_chunks.extend(records)

    async def add_table_chunks(self, records) -> None:
        self.table_chunks.extend(records)

    async def add_figure_chunks(self, records) -> None:
        self.figure_chunks.extend(records)

    async def delete_document(self, document_id) -> None:
        self.deleted_documents.append(document_id)
        for records in (
            self.document_cards,
            self.text_chunks,
            self.table_chunks,
            self.figure_chunks,
        ):
            records[:] = [
                record
                for record in records
                if record.metadata.get("document_id") != document_id
            ]

    async def prune_documents(self, document_ids) -> None:
        self.pruned_document_sets.append(list(document_ids))
        allowed = set(document_ids)
        for records in (
            self.document_cards,
            self.text_chunks,
            self.table_chunks,
            self.figure_chunks,
        ):
            records[:] = [
                record
                for record in records
                if record.metadata.get("document_id") in allowed
            ]


def test_lancedb_vector_index_and_hybrid_search(tmp_path) -> None:
    pytest.importorskip("lancedb")
    db_path = tmp_path / "app.db"
    lancedb_path = tmp_path / "lancedb"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    ser = docs_dir / "ser.md"
    notes = docs_dir / "notes.md"
    ser.write_text(
        "# Audio Visual SER\n\nSpeech emotion recognition uses audio visual fusion and multimodal retrieval.",
        encoding="utf-8",
    )
    notes.write_text(
        "# Planning\n\nThe desktop agent keeps a local catalog and project plan notes.",
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    indexed = indexing.index_selected_files(
        source_paths=[str(ser), str(notes)],
        collection_name="vector_seed",
        collection_type="project",
        scope_type="project",
        scope_id="local_ai_agent",
    )
    store = LanceDBRetrievalStore(lancedb_path)
    embeddings = HashEmbeddingProvider(dimensions=64)
    vector_index = VectorIndexService(db_path, store, embeddings)

    first_index = asyncio.run(vector_index.index_all_documents())
    second_index = asyncio.run(vector_index.index_all_documents())
    hybrid = asyncio.run(
        RagService(db_path).search_hybrid(
            query="audio visual speech emotion fusion",
            top_k=3,
            collection_id=indexed["collection_id"],
            retrieval_store=store,
            embeddings=embeddings,
        )
    )

    assert first_index["documents"] == 2
    assert all(item["ok"] for item in first_index["results"])
    assert second_index["documents"] == 2
    assert hybrid["results"]
    assert hybrid["results"][0]["filename"] == "ser.md"
    assert hybrid["results"][0]["page_number"] == 1
    assert any("lancedb" in item["retrieval_channels"] for item in hybrid["results"])


def test_vector_index_includes_table_and_figure_chunks_without_lancedb(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "visual.md"
    source.write_text(
        "\n".join(
            [
                "# Visual Report",
                "",
                "Table 1: Model quality",
                "| metric | value |",
                "| --- | --- |",
                "| recall | 0.88 |",
                "",
                "Figure 1: Retrieval architecture diagram",
            ]
        ),
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    indexed = indexing.index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    store = FakeRetrievalStore()
    embeddings = HashEmbeddingProvider(dimensions=64)
    vector_index = VectorIndexService(db_path, store, embeddings)

    result = asyncio.run(vector_index.index_document(indexed.documents[0].id))

    assert result["table_chunks"] == 1
    assert result["figure_chunks"] == 1
    assert store.table_chunks[0].metadata["caption"] == "Table 1: Model quality"
    assert store.figure_chunks[0].metadata["caption"] == "Figure 1: Retrieval architecture diagram"


def test_vector_index_only_embeds_quality_accepted_logical_figures(tmp_path) -> None:
    service = VectorIndexService(
        tmp_path / "app.db",
        FakeRetrievalStore(),
        HashEmbeddingProvider(dimensions=64),
    )
    common = {
        "document_id": "doc-1",
        "file_id": "file-1",
        "source_path": "/papers/CMDM.pdf",
        "filename": "CMDM.pdf",
        "page_number": 9,
        "visual_summary": "confusion matrices",
    }

    records = asyncio.run(
        service._figure_records(  # noqa: SLF001 - quality-gate regression
            [
                {
                    **common,
                    "id": "accepted",
                    "figure_index": 0,
                    "caption": "Figure 6: Confusion matrices",
                    "extraction_method": "docling_logical_composite",
                    "metadata_json": '{"quality_status":"accepted","asset_kind":"figure","is_complete":true,"figure_number":6}',
                },
                {
                    **common,
                    "id": "panel",
                    "figure_index": 1,
                    "caption": "Figure 6 panel",
                    "extraction_method": "docling_picture",
                    "metadata_json": '{"quality_status":"needs_review","asset_kind":"panel","is_complete":false}',
                },
                {
                    **common,
                    "id": "logo",
                    "figure_index": 2,
                    "caption": "Conference logo",
                    "extraction_method": "docling_picture",
                    "metadata_json": '{"quality_status":"rejected","asset_kind":"branding","is_content":false}',
                },
            ]
        )
    )

    assert [record.id for record in records] == ["figure:accepted"]
    assert records[0].metadata["figure_number"] == 6


def test_vector_reindex_replaces_complete_document_slice(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "visual.md"
    source.write_text(
        "\n".join(
            [
                "# Visual Report",
                "",
                "Table 1: Model quality",
                "| metric | value |",
                "| --- | --- |",
                "| recall | 0.88 |",
                "",
                "Figure 1: Retrieval architecture diagram",
            ]
        ),
        encoding="utf-8",
    )

    init_db(db_path)
    indexed = IndexingService(db_path).index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    document_id = indexed.documents[0].id
    store = FakeRetrievalStore()
    service = VectorIndexService(db_path, store, HashEmbeddingProvider(dimensions=64))
    asyncio.run(service.index_document(document_id))

    assert store.table_chunks
    assert store.figure_chunks

    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM document_tables WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_figures WHERE document_id = ?", (document_id,))

    asyncio.run(service.index_document(document_id))

    assert store.deleted_documents == [document_id, document_id]
    assert store.table_chunks == []
    assert store.figure_chunks == []
    assert len(store.document_cards) == 1


def test_limited_vector_index_does_not_prune_unprocessed_documents(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "first.md").write_text("# First\n\nAlpha retrieval.", encoding="utf-8")
    (docs_dir / "second.md").write_text("# Second\n\nBeta retrieval.", encoding="utf-8")

    init_db(db_path)
    indexed = IndexingService(db_path).index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    document_ids = {document.id for document in indexed.documents}
    store = FakeRetrievalStore()
    service = VectorIndexService(db_path, store, HashEmbeddingProvider(dimensions=64))

    asyncio.run(service.index_all_documents())
    asyncio.run(service.index_all_documents(limit=1))

    stored_document_ids = {
        record.metadata["document_id"] for record in store.document_cards
    }
    assert stored_document_ids == document_ids
    assert store.pruned_document_sets == [list(service._document_ids(None))]  # noqa: SLF001


def test_vector_index_all_skips_legacy_documents_without_cards(tmp_path) -> None:
    pytest.importorskip("lancedb")
    db_path = tmp_path / "app.db"
    lancedb_path = tmp_path / "lancedb"
    init_db(db_path)
    store = LanceDBRetrievalStore(lancedb_path)
    embeddings = HashEmbeddingProvider(dimensions=64)
    vector_index = VectorIndexService(db_path, store, embeddings)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO indexed_folders (id, folder_path, recursive, file_types, created_at, updated_at)
            VALUES ('folder-1', '/tmp', 0, '["md"]', 'now', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO documents (
                id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count
            )
            VALUES ('legacy-doc', 'folder-1', '/tmp/demo.md', 'demo.md', 'md', 'hash', 'now', 'now', 0)
            """
        )

    result = asyncio.run(vector_index.index_all_documents())

    assert result["documents"] == 1
    assert result["indexed_documents"] == 0
    assert result["skipped_documents"] == 1
    assert result["failed_documents"] == 0


def test_full_vector_sweep_over_empty_corpus_requests_complete_prune(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    store = FakeRetrievalStore()
    service = VectorIndexService(db_path, store, HashEmbeddingProvider(dimensions=64))

    result = asyncio.run(service.index_all_documents())

    assert result["documents"] == 0
    assert store.pruned_document_sets == [[]]


def test_vector_index_all_replaces_reindexed_document_vectors_with_stable_id(tmp_path) -> None:
    pytest.importorskip("lancedb")
    db_path = tmp_path / "app.db"
    lancedb_path = tmp_path / "lancedb"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "ser.md"
    source.write_text(
        "# Audio Visual SER\n\nSpeech emotion recognition uses multimodal fusion.",
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    indexed = indexing.index_selected_files(
        source_paths=[str(source)],
        collection_name="vector_seed",
        collection_type="project",
        scope_type="project",
        scope_id="local_ai_agent",
    )
    old_document_id = indexed["documents"][0].id
    store = LanceDBRetrievalStore(lancedb_path)
    embeddings = HashEmbeddingProvider(dimensions=64)
    vector_index = VectorIndexService(db_path, store, embeddings)
    asyncio.run(vector_index.index_all_documents())

    source.write_text(
        "# Audio Visual SER Updated\n\nSpeech emotion recognition now includes visual diagrams and table references.",
        encoding="utf-8",
    )
    reindexed = indexing.index_selected_files(
        source_paths=[str(source)],
        collection_name="vector_seed",
        collection_type="project",
        scope_type="project",
        scope_id="local_ai_agent",
    )
    new_document_id = reindexed["documents"][0].id

    assert new_document_id == old_document_id

    asyncio.run(vector_index.index_all_documents())
    results = asyncio.run(
        store.search_document_cards(
            query_embedding=asyncio.run(embeddings.embed_query("audio visual ser")),
            filters=RetrievalFilter(),
            top_k=20,
        )
    )
    document_ids = {result.metadata["document_id"] for result in results}

    assert new_document_id in document_ids
