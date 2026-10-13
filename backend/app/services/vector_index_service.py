from dataclasses import dataclass
import json
from pathlib import Path

from app.db.sqlite import connect
from app.rag.embeddings import EmbeddingProvider
from app.retrieval_store.base import VectorRecord
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.rag.text_normalization import searchable_text


@dataclass(frozen=True)
class VectorIndexService:
    db_path: Path
    retrieval_store: LanceDBRetrievalStore
    embeddings: EmbeddingProvider

    async def index_document(
        self,
        document_id: str,
        revision_id: str | None = None,
        *,
        activate_revision: bool = True,
    ) -> dict:
        if revision_id is None:
            revision_id = self._target_revision_id(document_id)
        if revision_id is None:
            return {"document_id": document_id, "ok": False, "error": "revision_not_found"}
        card = self._document_card(document_id, revision_id)
        chunks = self._document_chunks(document_id, revision_id)
        tables = self._document_tables(document_id, revision_id)
        figures = self._document_figures(document_id, revision_id)
        if card is None:
            # A card-less document is not indexable, but any vectors left from a
            # prior version of that document are stale and must be removed.
            await self._delete_projection(document_id, revision_id)
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
        await self._delete_projection(document_id, revision_id)
        await self.retrieval_store.add_document_cards([card_record])
        await self.retrieval_store.add_text_chunks(chunk_records)
        await self.retrieval_store.add_table_chunks(table_records)
        await self.retrieval_store.add_figure_chunks(figure_records)

        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE document_cards
                SET lance_table = 'document_cards', lance_id = ?, revision_id = ?
                WHERE document_id = ? AND (revision_id IS NULL OR revision_id = ?)
                """,
                (card_record.id, revision_id, document_id, revision_id),
            )
            # ``document_card_revisions`` is the revision-authoritative card
            # table. Updating only the compatibility row makes readiness
            # probes see a NULL Lance identity and re-embed the entire
            # document on every warm run even though the vector exists.
            connection.execute(
                """
                UPDATE document_card_revisions
                SET lance_table = 'document_cards', lance_id = ?
                WHERE document_id = ? AND revision_id = ?
                """,
                (card_record.id, document_id, revision_id),
            )
            for record in chunk_records:
                connection.execute(
                    """
                    UPDATE chunks
                    SET lance_table = 'text_chunks', lance_id = ?
                        WHERE id = ? AND (revision_id IS NULL OR revision_id = ?)
                        """,
                        (record.id, record.id, revision_id),
                )

        result = {
            "document_id": document_id,
            "revision_id": revision_id,
            "ok": True,
            "document_cards": 1,
            "text_chunks": len(chunk_records),
            "table_chunks": len(table_records),
            "figure_chunks": len(figure_records),
        }
        # The manual vector endpoint also uses this service. Do not let that
        # path activate a replacement merely because the insertion calls
        # returned; verify the physical projection just like the coordinator.
        if callable(getattr(self.retrieval_store, "document_revision_ids", None)):
            if not self.projection_identity_matches(document_id, revision_id):
                return {**result, "ok": False, "error": "revision_projection_identity_mismatch"}
        if activate_revision and revision_id != "legacy":
            from app.services.indexing_service import IndexingService

            if not IndexingService(self.db_path).activate_revision(document_id, revision_id):
                return {**result, "ok": False, "error": "revision_activation_failed"}
            result["activated"] = True
        return result

    async def _embed_documents(self, texts: list[str]) -> list:
        """Embed indexed text after search normalization (math letters → plain)."""
        return await self.embeddings.embed_texts([searchable_text(text) for text in texts])

    async def _delete_projection(self, document_id: str, revision_id: str | None) -> None:
        if revision_id and hasattr(self.retrieval_store, "delete_document_revision"):
            await self.retrieval_store.delete_document_revision(document_id, revision_id)
        else:
            await self.retrieval_store.delete_document(document_id)

    def expected_projection_ids(self, document_id: str, revision_id: str) -> dict[str, set[str]]:
        """Apply exactly the indexing eligibility policy without embedding."""
        card = self._document_card(document_id, revision_id)
        return {
            "document_cards": {f"card:{document_id}:{revision_id}"} if card else set(),
            "text_chunks": {str(row["id"]) for row in self._document_chunks(document_id, revision_id)},
            "table_chunks": {
                f"table:{row['id']}" for row in self._document_tables(document_id, revision_id)
                if _table_search_text(row)
            },
            "figure_chunks": {
                f"figure:{row['id']}" for row in self._document_figures(document_id, revision_id)
                if _indexable_figure_text(row)
            },
        }

    def projection_identity_matches(self, document_id: str, revision_id: str) -> bool:
        read_ids = getattr(self.retrieval_store, "document_revision_ids", None)
        if not callable(read_ids):
            return False
        physical = read_ids(document_id, revision_id)
        expected = self.expected_projection_ids(document_id, revision_id)
        if physical is None or physical != expected:
            return False
        read_counts = getattr(self.retrieval_store, "count_document_revision", None)
        if callable(read_counts):
            counts = read_counts(document_id, revision_id)
            return counts is not None and counts == {key: len(ids) for key, ids in expected.items()}
        return True

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

    def _document_card(self, document_id: str, revision_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT cards.*, d.file_id, d.source_path, d.filename
                FROM document_card_revisions cards
                JOIN documents d ON d.id = cards.document_id
                WHERE cards.document_id = ? AND cards.revision_id = ?
                """,
                (document_id, revision_id),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT cards.*, d.file_id, d.source_path, d.filename
                    FROM document_cards cards
                    JOIN documents d ON d.id = cards.document_id
                    WHERE cards.document_id = ?
                      AND (cards.revision_id = ? OR cards.revision_id IS NULL)
                    """,
                    (document_id, revision_id),
                ).fetchone()
        return dict(row) if row else None

    def _target_revision_id(self, document_id: str) -> str | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT d.active_revision_id,
                       (SELECT revision_id FROM document_revisions r
                        WHERE r.document_id = d.id AND r.status = 'staged'
                        ORDER BY r.created_at DESC LIMIT 1) AS staged_revision_id
                FROM documents d
                WHERE d.id = ?
                """,
                (document_id,),
            ).fetchone()
        if not row:
            return None
        return str(row["staged_revision_id"] or row["active_revision_id"] or "legacy")

    def _document_chunks(self, document_id: str, revision_id: str | None = None) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, file_id, content, source_path, filename,
                       chunk_type, heading_path_json, order_index, parent_chunk_id,
                       page_number, revision_id, metadata_json
                FROM chunks
                WHERE document_id = ?
                  AND chunk_type = 'text'
                  AND (revision_id = ? OR (revision_id IS NULL AND ? = 'legacy'))
                ORDER BY chunk_index ASC
                """,
                (document_id, revision_id, revision_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def _document_tables(self, document_id: str, revision_id: str | None = None) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_tables.*, documents.source_path, documents.filename
                FROM document_tables
                JOIN documents ON documents.id = document_tables.document_id
                WHERE document_tables.document_id = ?
                  AND (document_tables.revision_id = ? OR (document_tables.revision_id IS NULL AND ? = 'legacy'))
                ORDER BY document_tables.table_index ASC
                """,
                (document_id, revision_id, revision_id),
            ).fetchall()
            # Captions on image-backed tables are frequently terse.  Attach a
            # bounded same-page text excerpt to the indexing payload so terms
            # introduced in the surrounding section (metric names, task
            # names, dataset labels) can retrieve the table without changing
            # the canonical table content or leaking another revision.
            def _needs_page_context(row) -> bool:
                extraction = str(row["extraction_method"] or "").casefold()
                metadata = json.loads(row["metadata_json"] or "{}")
                # Structured native tables already contain their headers/rows;
                # appending an entire page can dilute exact-value retrieval.
                # Image-only/adapter tables need surrounding prose because
                # OCR/captions alone are often too sparse.
                return (
                    not str(row["markdown"] or "").strip()
                    or extraction.endswith("image")
                    or extraction.endswith("image_table")
                    or extraction.startswith("spiqa_")
                    or metadata.get("asset_type") == "table"
                )

            context_rows = [row for row in rows if _needs_page_context(row)]
            page_numbers = {
                int(row["page_number"])
                for row in context_rows
                if row["page_number"] is not None
            }
            page_text: dict[int, list[str]] = {page: [] for page in page_numbers}
            if page_numbers:
                placeholders = ",".join("?" for _ in page_numbers)
                page_rows = connection.execute(
                    f"""
                    SELECT page_number, content
                    FROM chunks
                    WHERE document_id = ?
                      AND (revision_id = ? OR (revision_id IS NULL AND ? = 'legacy'))
                      AND chunk_type = 'text'
                      AND page_number IN ({placeholders})
                    ORDER BY chunk_index ASC
                    """,
                    (document_id, revision_id, revision_id, *sorted(page_numbers)),
                ).fetchall()
                for page_row in page_rows:
                    text = " ".join(str(page_row["content"] or "").split())
                    if text:
                        page_text[int(page_row["page_number"])].append(text)
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            page = item.get("page_number")
            if page is not None and _needs_page_context(row):
                excerpt = " ".join(page_text.get(int(page), []))[:1800]
                if excerpt:
                    item["page_text_excerpt"] = excerpt
            result.append(item)
        return result

    def _document_figures(self, document_id: str, revision_id: str | None = None) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_figures.*, documents.source_path, documents.filename
                FROM document_figures
                JOIN documents ON documents.id = document_figures.document_id
                WHERE document_figures.document_id = ?
                  AND (document_figures.revision_id = ? OR (document_figures.revision_id IS NULL AND ? = 'legacy'))
                ORDER BY document_figures.figure_index ASC
                """,
                (document_id, revision_id, revision_id),
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
        embedding = (await self._embed_documents([text]))[0]
        revision_id = card.get("revision_id") or "legacy"
        return VectorRecord(
            id=f"card:{card['document_id']}:{revision_id}",
            text=text,
            embedding=embedding,
            metadata={
                "document_id": card["document_id"],
                "file_id": card.get("file_id"),
                "source_path": card.get("source_path"),
                "filename": card.get("filename"),
                "revision_id": revision_id if revision_id != "legacy" else None,
                "doc_type": card.get("doc_type"),
                "topic_tags": topic_tags,
                "project_tags": project_tags,
            },
        )

    async def _chunk_records(self, chunks: list[dict]) -> list[VectorRecord]:
        embedding_texts = [_contextual_chunk_text(chunk) for chunk in chunks]
        embeddings = await self._embed_documents(embedding_texts)
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
                        "revision_id": chunk.get("revision_id"),
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

        embeddings = await self._embed_documents([text for _, text in items])
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
                        "revision_id": table.get("revision_id"),
                        "artifact_type": "table",
                        "chunk_type": "table",
                        "table_id": table["id"],
                        "table_index": table.get("table_index"),
                        "page_number": table.get("page_number"),
                        "caption": table.get("caption"),
                        # Most parsed tables are textual, but an ingestion
                        # adapter may retain an authoritative table image in
                        # metadata. Preserve it through LanceDB so a selected
                        # TableEvidence can be shown to a multimodal answerer
                        # instead of being reduced to an opaque caption.
                        "image_path": metadata.get("image_path"),
                    },
                )
            )
        return records

    async def _figure_records(self, figures: list[dict]) -> list[VectorRecord]:
        items: list[tuple[dict, str]] = []
        for figure in figures:
            text = _indexable_figure_text(figure)
            if text:
                items.append((figure, text))
        if not items:
            return []

        embeddings = await self._embed_documents([text for _, text in items])
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
                        "revision_id": figure.get("revision_id"),
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


def _indexable_figure_text(figure: dict) -> str:
    from app.rag.figure_quality import figure_is_indexable

    if not figure_is_indexable(
        caption=figure.get("caption"),
        extraction_method=figure.get("extraction_method"),
        metadata=json.loads(figure.get("metadata_json") or "{}"),
    ):
        return ""
    return _figure_search_text(figure)


def _table_search_text(table: dict) -> str:
    from app.rag.vision import build_table_retrieval_context

    metadata = json.loads(table.get("metadata_json") or "{}")
    context = "\n".join(
        str(metadata.get(key) or "").strip()
        for key in ("page_text_excerpt", "artifact_context", "surrounding_text")
        if str(metadata.get(key) or "").strip()
    )
    base = build_table_retrieval_context(
        caption=table.get("caption"),
        markdown=table.get("markdown"),
        filename=table.get("filename"),
        document_title=metadata.get("document_title") or metadata.get("title"),
        page_number=table.get("page_number"),
        table_id=table.get("id"),
    )
    # Artifact rows can carry a bounded, provenance-preserving excerpt from
    # the same page/section.  It is useful when an image table's caption is
    # terse (e.g. ``Table 1``) but the prose explains the metric or task.  Do
    # not invent context here: only metadata produced by the parser/adapter is
    # accepted, and the excerpt remains bounded before embedding.
    return f"{base}\ncontext:\n{context[:1800]}" if context else base


def _contextual_chunk_text(chunk: dict) -> str:
    metadata = json.loads(chunk.get("metadata_json") or "{}")
    context_prefix = metadata.get("embedding_context_prefix") or metadata.get("context_prefix")
    content = searchable_text(chunk["content"])
    if not context_prefix:
        return content
    return f"{searchable_text(context_prefix)}\n\n{content}"


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
            (
                f"asset_filename: {metadata.get('external_artifact_file')}"
                if metadata.get("external_artifact_file")
                else None
            ),
            f"figure_label: {figure_label}" if figure_label else None,
            f"figure_type: {figure_type}" if figure_type else None,
            figure.get("caption"),
            figure.get("visual_summary"),
            metadata.get("ocr_text"),
            f"search_phrases: {phrases_text}" if phrases_text else None,
            caption_text,
            metadata.get("page_text_excerpt"),
            metadata.get("artifact_context"),
            metadata.get("surrounding_text"),
        ]
        if part
    ).strip()
