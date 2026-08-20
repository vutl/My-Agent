import asyncio
import sqlite3

from app.db.sqlite import init_db
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService


class _UnexpectedEmbeddingProvider:
    async def embed_query(self, _text):
        raise AssertionError("empty scopes must return before embedding")


class _UnexpectedRetrievalStore:
    def __getattr__(self, name):
        raise AssertionError(f"empty scopes must not call retrieval store: {name}")


def test_empty_or_unknown_collection_never_searches_whole_corpus(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "paper.md"
    source.write_text("# Paper\n\nUnique scoped evidence token.", encoding="utf-8")
    init_db(db_path)
    IndexingService(db_path).index_file(source_path=str(source))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO collections (id, name, type, scope_type, created_at, updated_at)
            VALUES ('empty-collection', 'Empty', 'manual', 'global', 'now', 'now')
            """
        )

    rag = RagService(db_path)
    unknown = rag.search_in_collection(
        collection_id="missing-collection",
        query="unique scoped evidence",
    )
    empty = rag.search_in_collection(
        collection_id="empty-collection",
        query="unique scoped evidence",
    )

    assert unknown["document_ids"] == []
    assert unknown["results"] == []
    assert empty["document_ids"] == []
    assert empty["results"] == []
    assert rag.search("unique scoped evidence", document_ids=[]) == []

    hybrid = asyncio.run(
        rag.search_hybrid(
            query="unique scoped evidence",
            top_k=5,
            collection_id="missing-collection",
            retrieval_store=_UnexpectedRetrievalStore(),
            embeddings=_UnexpectedEmbeddingProvider(),
        )
    )
    visuals = asyncio.run(
        rag.retrieve_visual_assets(
            query="unique scoped evidence",
            top_k=5,
            collection_id="missing-collection",
            retrieval_store=_UnexpectedRetrievalStore(),
            embeddings=_UnexpectedEmbeddingProvider(),
        )
    )
    figures = asyncio.run(
        rag.retrieve_figures(
            query="unique scoped evidence",
            top_k=5,
            collection_id="missing-collection",
            retrieval_store=_UnexpectedRetrievalStore(),
            embeddings=_UnexpectedEmbeddingProvider(),
        )
    )

    assert hybrid["results"] == []
    assert visuals["results"] == []
    assert figures["results"] == []


def test_listing_figures_keeps_caption_lookup_connection_open(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "figures.md"
    source.write_text(
        "# Figures\n\nFigure 1: Retrieval architecture diagram",
        encoding="utf-8",
    )
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))

    figures = RagService(db_path).list_document_figures(document.id)

    assert len(figures) == 1
    assert figures[0]["caption"] == "Figure 1: Retrieval architecture diagram"
    assert figures[0]["image_url"] is None
