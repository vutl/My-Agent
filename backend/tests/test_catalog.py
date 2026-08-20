from app.db.sqlite import init_db
from app.services.catalog_service import CatalogService
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService


def test_scan_folder_catalogs_files_without_deep_index(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "paper.md").write_text("Speech emotion recognition fusion model.", encoding="utf-8")
    (docs_dir / "image.bin").write_bytes(b"binary")

    init_db(db_path)
    catalog = CatalogService(db_path)

    result = catalog.scan_folder(folder_path=str(docs_dir))
    search = catalog.search(query="paper speech", folder_path=str(docs_dir))

    assert result["file_count"] == 2
    assert result["supported_files"] == 1
    assert result["unsupported_files"] == 1
    assert search["results"]
    assert search["results"][0]["source"] == "file"

    resolved = catalog.resolve_file(
        filename_or_query="paper.md",
        base_folder=str(docs_dir),
    )
    direct = catalog.read_file_direct(
        source_path=resolved["candidates"][0]["source_path"],
        max_tokens=20,
    )

    assert resolved["status"] == "single_match"
    assert direct["filename"] == "paper.md"
    assert "Speech emotion recognition" in direct["content"]


def test_read_file_direct_requires_approved_folder(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    note = docs_dir / "private.md"
    note.write_text("This file should not be read before approval.", encoding="utf-8")

    init_db(db_path)
    catalog = CatalogService(db_path)

    try:
        catalog.read_file_direct(source_path=str(note), max_tokens=20)
    except ValueError as exc:
        assert "approved folder" in str(exc)
    else:
        raise AssertionError("read_file_direct should reject unapproved paths")


def test_index_selected_files_creates_collection_card_and_collection_search(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    first = docs_dir / "fusion.md"
    second = docs_dir / "rag.md"
    first.write_text(
        "# Fusion\n\nSpeech emotion recognition uses multimodal fusion.",
        encoding="utf-8",
    )
    second.write_text(
        "# RAG\n\nLocal AI agent uses SQLite FTS5 and LanceDB retrieval.",
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    rag = RagService(db_path)
    catalog = CatalogService(db_path)

    result = indexing.index_selected_files(
        source_paths=[str(first), str(second)],
        collection_name="seed_docs",
        collection_type="project",
        scope_type="project",
        scope_id="local_ai_agent",
    )
    collection_id = result["collection_id"]
    chunk_results = rag.search_in_collection(
        collection_id=collection_id,
        query="LanceDB retrieval",
        top_k=5,
    )
    catalog_results = catalog.search(query="speech emotion fusion", top_k=5)

    assert result["indexed_files"] == 2
    assert catalog.list_collections()[0]["name"] == "seed_docs"
    assert chunk_results["results"]
    assert chunk_results["results"][0]["filename"] == "rag.md"
    assert any(item.get("source") == "document_card" for item in catalog_results["results"])
