from dataclasses import dataclass
import json
from pathlib import Path

from app.db.sqlite import connect
from app.rag.embeddings import EmbeddingProvider
from app.retrieval_store.base import VectorRecord
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable


@dataclass(frozen=True)
class VectorIndexService:
    db_path: Path
    retrieval_store: LanceDBRetrievalStore
    embeddings: EmbeddingProvider

    async def index_document(self, document_id: str) -> dict:
        card = self._document_card(document_id)
        chunks = self._document_chunks(document_id)
        tables = self._document_tables(document_id)
        figures = self._document_figures(document_id)
        if card is None:
            # A card-less document is not indexable, but any vectors left from a
            # prior version of that document are stale and must be removed.
            await self.retrieval_store.delete_document(document_id)
            return {
                "document_id": document_id,
                "ok": False,
                "skipped": True,
                "error": "document_card_not_found",
            }

        card_record = await self._card_record(card)
        chunk_records = await self._chunk_records(chunks)
        table_records = await self._table_records(tables)
        figure_records = await self._figure_records(figures)
        # Replace the complete document slice. Upserting only the current ids
        # leaves deleted chunks/tables/figures searchable forever.
        await self.retrieval_store.delete_document(document_id)
        await self.retrieval_store.add_document_cards([card_record])
        await self.retrieval_store.add_text_chunks(chunk_records)
        await self.retrieval_store.add_table_chunks(table_records)
        await self.retrieval_store.add_figure_chunks(figure_records)

        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE document_cards
                SET lance_table = 'document_cards', lance_id = ?
                WHERE document_id = ?
                """,
                (card_record.id, document_id),
            )
            for record in chunk_records:
                connection.execute(
                    """
                    UPDATE chunks
                    SET lance_table = 'text_chunks', lance_id = ?
                    WHERE id = ?
                    """,
                    (record.id, record.id),
                )

        return {
            "document_id": document_id,
            "ok": True,
            "document_cards": 1,
            "text_chunks": len(chunk_records),
            "table_chunks": len(table_records),
            "figure_chunks": len(figure_records),
        }

    async def index_all_documents(self, limit: int | None = None) -> dict:
        document_ids = self._document_ids(limit)
        # A limited run is a partial refresh, not a declaration that every
        # unprocessed document was deleted. Global pruning is safe only for a
        # complete corpus sweep.
        if limit is None:
            await self.retrieval_store.prune_documents(document_ids)
        results = []
        for document_id in document_ids:
            results.append(await self.index_document(document_id))
        ok_count = sum(1 for result in results if result.get("ok"))
        skipped_count = sum(1 for result in results if result.get("skipped"))
        return {
            "documents": len(document_ids),
            "indexed_documents": ok_count,
            "skipped_documents": skipped_count,
            "failed_documents": len(results) - ok_count - skipped_count,
            "results": results,
        }

    def _document_ids(self, limit: int | None) -> list[str]:
        sql = "SELECT id FROM documents ORDER BY indexed_at DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with connect(self.db_path) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [row["id"] for row in rows]

    def _document_card(self, document_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT document_cards.*, documents.file_id, documents.source_path, documents.filename
                FROM document_cards
                JOIN documents ON documents.id = document_cards.document_id
                WHERE document_cards.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def _document_chunks(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, file_id, content, source_path, filename,
                       chunk_type, heading_path_json, order_index, parent_chunk_id,
                       page_number, metadata_json
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _document_tables(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_tables.*, documents.source_path, documents.filename
                FROM document_tables
                JOIN documents ON documents.id = document_tables.document_id
                WHERE document_tables.document_id = ?
                ORDER BY document_tables.table_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _document_figures(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_figures.*, documents.source_path, documents.filename
                FROM document_figures
                JOIN documents ON documents.id = document_figures.document_id
                WHERE document_figures.document_id = ?
                ORDER BY document_figures.figure_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def _card_record(self, card: dict) -> VectorRecord:
        topic_tags = json.loads(card.get("topic_tags_json") or "[]")
        project_tags = json.loads(card.get("project_tags_json") or "[]")
        keywords = json.loads(card.get("keywords_json") or "[]")
        text = "\n".join(
            item
            for item in [
                card.get("title_guess") or card.get("filename") or "",
                card.get("short_summary") or "",
                " ".join(topic_tags),
                " ".join(project_tags),
                " ".join(keywords),
            ]
            if item
        )
        # A document card is indexed content, not a retrieval query. Providers
        # with asymmetric query/document instructions must receive it through
        # the document embedding path.
        embedding = (await self.embeddings.embed_texts([text]))[0]
        return VectorRecord(
            id=f"card:{card['document_id']}",
            text=text,
            embedding=embedding,
            metadata={
                "document_id": card["document_id"],
                "file_id": card.get("file_id"),
                "source_path": card.get("source_path"),
                "filename": card.get("filename"),
                "doc_type": card.get("doc_type"),
                "topic_tags": topic_tags,
                "project_tags": project_tags,
            },
        )

    async def _chunk_records(self, chunks: list[dict]) -> list[VectorRecord]:
        embedding_texts = [_contextual_chunk_text(chunk) for chunk in chunks]
        embeddings = await self.embeddings.embed_texts(embedding_texts)
        records: list[VectorRecord] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            metadata = json.loads(chunk.get("metadata_json") or "{}")
            records.append(
                VectorRecord(
                    id=chunk["id"],
                    text=chunk["content"],
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "document_id": chunk["document_id"],
                        "file_id": chunk.get("file_id"),
                        "source_path": chunk.get("source_path"),
                        "filename": chunk.get("filename"),
                        "chunk_type": chunk.get("chunk_type"),
                        "heading_path": json.loads(chunk.get("heading_path_json") or "[]"),
                        "chunk_index": chunk.get("order_index"),
                        "order_index": chunk.get("order_index"),
                        "parent_chunk_id": chunk.get("parent_chunk_id"),
                        "page_number": chunk.get("page_number"),
                        "section_title": metadata.get("section_title"),
                    },
                )
            )
        return records

    async def _table_records(self, tables: list[dict]) -> list[VectorRecord]:
        items = [
            (table, text)
            for table in tables
            if (text := _table_search_text(table))
        ]
        if not items:
            return []

        embeddings = await self.embeddings.embed_texts([text for _, text in items])
        records: list[VectorRecord] = []
        for (table, text), embedding in zip(items, embeddings, strict=True):
            metadata = json.loads(table.get("metadata_json") or "{}")
            records.append(
                VectorRecord(
                    id=f"table:{table['id']}",
                    text=text,
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "document_id": table["document_id"],
                        "file_id": table.get("file_id"),
                        "source_path": table.get("source_path"),
                        "filename": table.get("filename"),
                        "artifact_type": "table",
                        "chunk_type": "table",
                        "table_id": table["id"],
                        "table_index": table.get("table_index"),
                        "page_number": table.get("page_number"),
                        "caption": table.get("caption"),
                    },
                )
            )
        return records

    async def _figure_records(self, figures: list[dict]) -> list[VectorRecord]:
        from app.rag.figure_quality import figure_is_indexable

        items: list[tuple[dict, str]] = []
        for figure in figures:
            metadata = json.loads(figure.get("metadata_json") or "{}")
            if not figure_is_indexable(
                caption=figure.get("caption"),
                extraction_method=figure.get("extraction_method"),
                metadata=metadata,
            ):
                continue
            text = _figure_search_text(figure)
            if text:
                items.append((figure, text))
        if not items:
            return []

        embeddings = await self.embeddings.embed_texts([text for _, text in items])
        records: list[VectorRecord] = []
        for (figure, text), embedding in zip(items, embeddings, strict=True):
            metadata = json.loads(figure.get("metadata_json") or "{}")
            asset_type = metadata.get("asset_type") or "figure"
            records.append(
                VectorRecord(
                    id=f"figure:{figure['id']}",
                    text=text,
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "document_id": figure["document_id"],
                        "file_id": figure.get("file_id"),
                        "source_path": figure.get("source_path"),
                        "filename": figure.get("filename"),
                        "artifact_type": asset_type,
                        "chunk_type": "visual_page" if asset_type == "page" else "figure",
                        "figure_id": figure["id"],
                        "figure_index": figure.get("figure_index"),
                        "page_number": figure.get("page_number"),
                        "caption": figure.get("caption"),
                        "image_path": figure.get("image_path"),
                    },
                )
            )
        return records


def create_lancedb_vector_index_service(
    *,
    db_path: Path,
    lancedb_path: Path,
    embeddings: EmbeddingProvider,
) -> VectorIndexService:
    try:
        store = LanceDBRetrievalStore(lancedb_path)
    except LanceDBUnavailable:
        raise
    return VectorIndexService(db_path=db_path, retrieval_store=store, embeddings=embeddings)


def _table_search_text(table: dict) -> str:
    from app.rag.vision import build_table_retrieval_context

    return build_table_retrieval_context(
        caption=table.get("caption"),
        markdown=table.get("markdown"),
        filename=table.get("filename"),
    )


def _contextual_chunk_text(chunk: dict) -> str:
    metadata = json.loads(chunk.get("metadata_json") or "{}")
    context_prefix = metadata.get("embedding_context_prefix") or metadata.get("context_prefix")
    if not context_prefix:
        return chunk["content"]
    return f"{context_prefix}\n\n{chunk['content']}"


def _figure_search_text(figure: dict) -> str:
    metadata = json.loads(figure.get("metadata_json") or "{}")
    detected_captions = metadata.get("detected_captions") or []
    if isinstance(detected_captions, list):
        caption_text = "\n".join(str(caption) for caption in detected_captions)
    else:
        caption_text = str(detected_captions)

    search_phrases = metadata.get("search_phrases") or []
    if isinstance(search_phrases, list):
        phrases_text = ", ".join(str(phrase) for phrase in search_phrases if phrase)
    else:
        phrases_text = str(search_phrases)

    figure_type = metadata.get("figure_type")
    figure_label = metadata.get("figure_label")
    return "\n".join(
        part
        for part in [
            f"filename: {figure.get('filename')}" if figure.get("filename") else None,
            f"figure_label: {figure_label}" if figure_label else None,
            f"figure_type: {figure_type}" if figure_type else None,
            figure.get("caption"),
            figure.get("visual_summary"),
            f"search_phrases: {phrases_text}" if phrases_text else None,
            caption_text,
            metadata.get("page_text_excerpt"),
        ]
        if part
    ).strip()
