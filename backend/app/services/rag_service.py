from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import unicodedata

from app.db.sqlite import connect
from app.rag.embeddings import EmbeddingProvider
from app.rag.figure_caption import best_figure_caption, caption_looks_truncated
from app.rag.figure_quality import extract_figure_label
from app.rag.reranker import rerank_candidates
from app.rag.retriever import RetrievedChunk, build_fts_query, search_chunks
from app.retrieval_store.base import RetrievalFilter
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore


@dataclass(frozen=True)
class CatalogDocumentMention:
    surface: str
    start_word: int
    end_word: int
    document_id: str | None
    candidate_ids: tuple[str, ...]
    alias_source: str
    strength: int

    @property
    def ambiguous(self) -> bool:
        return self.document_id is None and len(self.candidate_ids) > 1


@dataclass(frozen=True)
class CatalogMentionResolution:
    mentions: tuple[CatalogDocumentMention, ...]

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                mention.document_id
                for mention in self.mentions
                if mention.document_id is not None
            )
        )

    @property
    def ambiguous_mentions(self) -> tuple[CatalogDocumentMention, ...]:
        return tuple(mention for mention in self.mentions if mention.ambiguous)


@dataclass(frozen=True)
class RagService:
    db_path: Path
    artifact_root: Path | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        # None means unscoped; an explicitly empty scope must never widen to the
        # whole corpus inside the FTS helper.
        if document_ids is not None and not document_ids:
            return []
        with connect(self.db_path) as connection:
            return search_chunks(connection, query, top_k, document_ids=document_ids)

    def search_in_collection(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int = 8,
    ) -> dict:
        document_ids = self._collection_document_ids(collection_id)
        results = self.search(query, top_k=top_k, document_ids=document_ids) if document_ids else []
        return {
            "collection_id": collection_id,
            "query": query,
            "document_ids": document_ids,
            "retrieval_channels": ["sqlite_fts5_collection_chunks"],
            "results": [result.__dict__ for result in results],
        }

    def search_debug(self, query: str, top_k: int = 5) -> dict:
        results = self.search(query, top_k)
        return {
            "query": query,
            "fts_query": build_fts_query(query),
            "retrieval_channels": ["sqlite_fts5"],
            "results": [result.__dict__ for result in results],
        }

    def expand_with_neighbor_chunks(
        self,
        results: list[dict],
        *,
        window: int = 1,
        max_neighbor_chars: int = 450,
        query: str | None = None,
    ) -> list[dict]:
        expanded: list[dict] = []
        for result in results:
            if not _is_text_chunk_result(result):
                expanded.append(result)
                continue
            document_id = result.get("document_id")
            chunk_index = result.get("chunk_index")
            if document_id is None or chunk_index is None:
                expanded.append(result)
                continue
            parent = self._parent_context(str(result.get("chunk_id") or result.get("id") or ""))
            if parent is not None:
                neighbors = self._neighbor_chunks(
                    document_id=str(document_id),
                    chunk_index=int(chunk_index),
                    page_number=result.get("page_number"),
                    section_title=_result_section_title(result),
                    window=window,
                )
                expanded.append(
                    {
                        **result,
                        "expanded_content": _expanded_parent_context_text(
                            result,
                            parent,
                            max_parent_chars=max(max_neighbor_chars * 4, 1_800),
                        ),
                        "parent_chunk_id": parent["parent_chunk_id"],
                        "parent_expanded": True,
                        "neighbor_chunk_ids": [
                            neighbor["chunk_id"] for neighbor in neighbors
                        ],
                    }
                )
                continue
            neighbors = self._neighbor_chunks(
                document_id=str(document_id),
                chunk_index=int(chunk_index),
                page_number=result.get("page_number"),
                section_title=_result_section_title(result),
                window=window,
            )
            if not neighbors:
                expanded.append(result)
                continue
            expanded.append(
                {
                    **result,
                    "expanded_content": _expanded_context_text(
                        result,
                        neighbors,
                        max_neighbor_chars=max_neighbor_chars,
                    ),
                    "neighbor_chunk_ids": [neighbor["chunk_id"] for neighbor in neighbors],
                }
            )
        return expanded

    def _parent_context(self, chunk_id: str) -> dict | None:
        if not chunk_id:
            return None
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT parent_chunk_id, metadata_json
                FROM chunks
                WHERE id = ? AND chunk_type = 'text'
                """,
                (chunk_id,),
            ).fetchone()
        if row is None or not row["parent_chunk_id"]:
            return None
        metadata = json.loads(row["metadata_json"] or "{}")
        parent_content = str(metadata.get("parent_content") or "").strip()
        if not parent_content:
            return None
        return {
            "parent_chunk_id": str(row["parent_chunk_id"]),
            "content": parent_content,
            "parent_token_count": metadata.get("parent_token_count"),
        }

    def _neighbor_chunks(
        self,
        *,
        document_id: str,
        chunk_index: int,
        page_number: int | None,
        section_title: str | None,
        window: int,
    ) -> list[dict]:
        lower = chunk_index - window
        upper = chunk_index + window
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id AS chunk_id, document_id, source_path, filename, content,
                       chunk_index, page_number, heading_path_json, token_count, char_count,
                       metadata_json
                FROM chunks
                WHERE document_id = ?
                  AND chunk_type = 'text'
                  AND chunk_index BETWEEN ? AND ?
                  AND chunk_index != ?
                ORDER BY chunk_index ASC
                """,
                (document_id, lower, upper, chunk_index),
            ).fetchall()
        neighbors: list[dict] = []
        for row in rows:
            item = dict(row)
            if page_number is not None and item.get("page_number") != page_number:
                continue
            item["heading_path"] = json.loads(item.pop("heading_path_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            if not _neighbor_matches_section(item, section_title):
                continue
            neighbors.append(item)
        return neighbors

    async def search_hybrid(
        self,
        *,
        query: str,
        top_k: int,
        retrieval_store: LanceDBRetrievalStore,
        embeddings: EmbeddingProvider,
        collection_id: str | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = True,
        rerank_mode: str = "embedding",
        cross_encoder_model_path: str | Path | None = None,
        rerank_max_candidates: int = 20,
        visual_boost: bool = False,
    ) -> dict:
        collection_document_ids = self._collection_document_ids(collection_id) if collection_id else None
        forced_document_ids = list(document_ids) if document_ids is not None else None
        if forced_document_ids is not None and collection_document_ids is not None:
            allowed = set(collection_document_ids)
            forced_document_ids = [document_id for document_id in forced_document_ids if document_id in allowed]
        search_document_ids = (
            forced_document_ids if forced_document_ids is not None else collection_document_ids
        )
        scope_requested = collection_id is not None or document_ids is not None
        retrieval_channels = [
            "lancedb_text_chunks",
            "lancedb_table_chunks",
            "lancedb_figure_chunks",
            "sqlite_fts5_chunks",
        ]
        if scope_requested and not search_document_ids:
            return {
                "query": query,
                "collection_id": collection_id,
                "forced_document_ids": forced_document_ids or [],
                "selected_document_ids": [],
                "retrieval_channels": retrieval_channels,
                "document_card_results": [],
                "results": [],
            }

        query_embedding = await embeddings.embed_query(query)
        card_top_k = max(top_k * 4, 20)
        max_selected_documents = max(top_k * 2, 8)
        card_results = await retrieval_store.search_document_cards(
            query_embedding=query_embedding,
            filters=RetrievalFilter(document_ids=search_document_ids),
            top_k=card_top_k,
        )
        selected_document_ids = search_document_ids
        if selected_document_ids is None:
            selected_document_ids = _document_ids_from_vector_results(
                _filter_card_results_for_query(query, card_results),
                max_documents=max_selected_documents,
            )
        if scope_requested and not selected_document_ids:
            return {
                "query": query,
                "collection_id": collection_id,
                "forced_document_ids": forced_document_ids or [],
                "selected_document_ids": [],
                "retrieval_channels": retrieval_channels,
                "document_card_results": [],
                "results": [],
            }
        vector_text_chunks = await retrieval_store.search_text_chunks(
            query_embedding=query_embedding,
            filters=RetrievalFilter(document_ids=selected_document_ids or None),
            top_k=max(top_k * 4, 20),
        )
        vector_table_chunks = await retrieval_store.search_table_chunks(
            query_embedding=query_embedding,
            filters=RetrievalFilter(document_ids=selected_document_ids or None),
            top_k=max(top_k * (4 if visual_boost else 2), 12 if visual_boost else 10),
        )
        vector_figure_chunks = await retrieval_store.search_figure_chunks(
            query_embedding=query_embedding,
            filters=RetrievalFilter(document_ids=selected_document_ids or None),
            top_k=max(top_k * (4 if visual_boost else 2), 12 if visual_boost else 10),
        )
        fts_chunks = self.search(
            query,
            top_k=max(top_k * 4, 20),
            document_ids=search_document_ids,
        )
        vector_chunks = sorted(
            [*vector_text_chunks, *vector_table_chunks, *vector_figure_chunks],
            key=lambda result: result.score,
        )
        candidate_k = max(top_k * 4, 24) if rerank else top_k
        merged = _merge_hybrid_chunks(
            query,
            vector_chunks,
            fts_chunks,
            candidate_k,
            visual_boost=visual_boost,
        )
        if rerank and merged:
            merged = await rerank_candidates(
                query=query,
                results=merged,
                embeddings=embeddings,
                top_k=top_k,
                mode=rerank_mode,
                cross_encoder_model_path=cross_encoder_model_path,
                max_candidates=rerank_max_candidates,
            )
        else:
            merged = merged[:top_k]
        return {
            "query": query,
            "collection_id": collection_id,
            "forced_document_ids": forced_document_ids or [],
            "selected_document_ids": selected_document_ids,
            "retrieval_channels": retrieval_channels,
            "document_card_results": [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.text,
                    "metadata": result.metadata,
                }
                for result in card_results
            ],
            "results": merged,
        }

    async def retrieve_visual_assets(
        self,
        *,
        query: str,
        top_k: int,
        retrieval_store: LanceDBRetrievalStore,
        embeddings: EmbeddingProvider,
        collection_id: str | None = None,
        document_ids: list[str] | None = None,
        rerank: bool = True,
        rerank_mode: str = "embedding",
        cross_encoder_model_path: str | Path | None = None,
        rerank_max_candidates: int = 20,
    ) -> dict:
        collection_document_ids = self._collection_document_ids(collection_id) if collection_id else None
        forced_document_ids = list(document_ids) if document_ids is not None else None
        if forced_document_ids is not None and collection_document_ids is not None:
            allowed = set(collection_document_ids)
            forced_document_ids = [document_id for document_id in forced_document_ids if document_id in allowed]
        search_document_ids = (
            forced_document_ids if forced_document_ids is not None else collection_document_ids
        )
        scope_requested = collection_id is not None or document_ids is not None
        if scope_requested and not search_document_ids:
            return {
                "query": query,
                "collection_id": collection_id,
                "retrieval_channels": ["lancedb_figure_chunks", "lancedb_table_chunks"],
                "results": [],
            }

        query_embedding = await embeddings.embed_query(query)
        filters = RetrievalFilter(document_ids=search_document_ids or None)
        figure_chunks = await retrieval_store.search_figure_chunks(
            query_embedding=query_embedding,
            filters=filters,
            top_k=max(top_k * 3, 12),
        )
        table_chunks = await retrieval_store.search_table_chunks(
            query_embedding=query_embedding,
            filters=filters,
            top_k=max(top_k * 3, 12),
        )
        # LanceDB returns _distance (lower is better). Keep that order.
        figure_payloads = [
            _vector_payload(result)
            for result in sorted(figure_chunks, key=lambda item: item.score)
            if not _is_low_signal_visual_payload(_vector_payload(result))
        ]
        table_payloads = [
            _vector_payload(result)
            for result in sorted(table_chunks, key=lambda item: item.score)
        ]
        # Prefer figures for visual/figure-oriented queries; still allow tables.
        if _query_prefers_figures(query):
            merged = [*figure_payloads, *table_payloads]
        else:
            merged = sorted(
                [*figure_payloads, *table_payloads],
                key=lambda item: float(item.get("raw_vector_score") or 0.0),
            )
        if rerank and merged:
            merged = await rerank_candidates(
                query=query,
                results=merged,
                embeddings=embeddings,
                top_k=top_k,
                mode=rerank_mode,
                cross_encoder_model_path=cross_encoder_model_path,
                max_candidates=rerank_max_candidates,
            )
        else:
            merged = merged[:top_k]

        return {
            "query": query,
            "collection_id": collection_id,
            "retrieval_channels": ["lancedb_figure_chunks", "lancedb_table_chunks"],
            "results": merged,
        }

    async def retrieve_figures(
        self,
        *,
        query: str,
        top_k: int,
        retrieval_store: LanceDBRetrievalStore,
        embeddings: EmbeddingProvider,
        collection_id: str | None = None,
    ) -> dict:
        collection_document_ids = self._collection_document_ids(collection_id) if collection_id else None
        if collection_id and not collection_document_ids:
            return {
                "query": query,
                "collection_id": collection_id,
                "retrieval_channels": ["lancedb_figure_chunks"],
                "results": [],
            }

        query_embedding = await embeddings.embed_query(query)
        results = await retrieval_store.search_figure_chunks(
            query_embedding=query_embedding,
            filters=RetrievalFilter(document_ids=collection_document_ids or None),
            top_k=top_k,
        )
        return {
            "query": query,
            "collection_id": collection_id,
            "retrieval_channels": ["lancedb_figure_chunks"],
            "results": [_vector_payload(result) for result in results],
        }

    async def search_figures(
        self,
        *,
        query: str,
        top_k: int,
        retrieval_store: LanceDBRetrievalStore,
        embeddings: EmbeddingProvider,
        collection_id: str | None = None,
    ) -> dict:
        return await self.retrieve_figures(
            query=query,
            top_k=top_k,
            retrieval_store=retrieval_store,
            embeddings=embeddings,
            collection_id=collection_id,
        )

    def get_document(self, document_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, folder_id, source_path, filename, file_type, content_hash,
                       modified_at, indexed_at, chunk_count, index_status, parser_name,
                       parser_version, page_count, error_message, metadata_json
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def enrich_source_identities(self, sources: list[dict]) -> list[dict]:
        """Attach catalog-owned aliases to retrieval evidence in one batch.

        Chunk rows only carry filenames, while users often name a paper by a
        title acronym or proposed-model keyword.  Reuse the catalog resolver's
        alias grammar and expose only aliases uniquely owned in the corpus, so
        answer validation can recognize those identities without inventing a
        second, weaker resolver.
        """

        requested_ids = {
            str(source.get("document_id") or "").strip()
            for source in sources
            if source.get("document_id")
        }
        if not requested_ids:
            return sources
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.filename, d.title, c.title_guess, c.keywords_json
                FROM documents d
                LEFT JOIN document_cards c ON c.document_id = d.id
                """
            ).fetchall()

        aliases_by_document: dict[str, dict[tuple[str, ...], tuple[int, str]]] = {}
        alias_owners: dict[tuple[str, ...], set[str]] = {}
        rows_by_id: dict[str, dict] = {}
        for row in rows:
            item = dict(row)
            document_id = str(item["id"])
            rows_by_id[document_id] = item
            aliases = _catalog_document_aliases(
                filename=str(item.get("filename") or ""),
                title=str(item.get("title") or ""),
                title_guess=str(item.get("title_guess") or ""),
                keywords_json=item.get("keywords_json"),
            )
            aliases_by_document[document_id] = aliases
            for alias in aliases:
                alias_owners.setdefault(alias, set()).add(document_id)

        enriched: list[dict] = []
        for source in sources:
            document_id = str(source.get("document_id") or "").strip()
            catalog_row = rows_by_id.get(document_id, {})
            owned_aliases = [
                (alias, strength)
                for alias, (strength, _provenance) in aliases_by_document.get(
                    document_id,
                    {},
                ).items()
                if alias_owners.get(alias) == {document_id}
            ]
            owned_aliases.sort(
                key=lambda item: (item[1], len(item[0]), item[0]),
                reverse=True,
            )
            metadata = (
                dict(source.get("metadata"))
                if isinstance(source.get("metadata"), dict)
                else {}
            )
            metadata["catalog_aliases"] = [
                " ".join(alias) for alias, _strength in owned_aliases[:24]
            ]
            enriched.append(
                {
                    **source,
                    "document_title": source.get("document_title")
                    or catalog_row.get("title"),
                    "title_guess": source.get("title_guess")
                    or catalog_row.get("title_guess"),
                    "metadata": metadata,
                }
            )
        return enriched

    def list_document_chunks(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, chunk_index, content, source_path, filename,
                       chunk_type, heading_path_json, token_count, char_count,
                       order_index, parent_chunk_id, page_number, metadata_json
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        chunks: list[dict] = []
        for row in rows:
            item = dict(row)
            item["heading_path"] = json.loads(item.pop("heading_path_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            chunks.append(item)
        return chunks

    def list_document_tables(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, file_id, table_index, page_number, caption,
                       markdown, row_count, column_count, extraction_method,
                       bbox_json, created_at, metadata_json
                FROM document_tables
                WHERE document_id = ?
                ORDER BY page_number ASC, table_index ASC
                """,
                (document_id,),
            ).fetchall()
            tables = [_decode_artifact(dict(row)) for row in rows]
            _reconcile_mislabeled_table_captions(
                connection,
                document_id=document_id,
                tables=tables,
            )
        tables = [
            table
            for table in tables
            if not caption_identifies_figure(str(table.get("caption") or ""))
        ]
        for table in tables:
            table["image_url"] = (
                f"/rag/figures/{table['id']}/image"
                if table.get("image_path")
                else None
            )
        return tables

    def get_figure(self, figure_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, document_id, file_id, figure_index, page_number, caption,
                       image_path, visual_summary, extraction_method, bbox_json,
                       created_at, metadata_json
                FROM document_figures
                WHERE id = ?
                """,
                (figure_id,),
            ).fetchone()
            if row is None:
                return None
            figure = _decode_artifact(dict(row))
            figure["image_url"] = f"/rag/figures/{figure['id']}/image" if figure.get("image_path") else None
            self._apply_figure_caption(figure, connection=connection)
            return figure

    def list_document_figures(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, file_id, figure_index, page_number, caption,
                       image_path, visual_summary, extraction_method, bbox_json,
                       created_at, metadata_json
                FROM document_figures
                WHERE document_id = ?
                ORDER BY page_number ASC, figure_index ASC
                """,
                (document_id,),
            ).fetchall()
            figures = [_decode_artifact(dict(row)) for row in rows]
            for figure in figures:
                figure["image_url"] = (
                    f"/rag/figures/{figure['id']}/image" if figure.get("image_path") else None
                )
                self._apply_figure_caption(figure, connection=connection)
        return figures

    def repair_figure_captions(self, document_id: str | None = None) -> int:
        updated = 0
        with connect(self.db_path) as connection:
            query = """
                SELECT id, document_id, figure_index, page_number, caption, visual_summary,
                       metadata_json
                FROM document_figures
            """
            params: tuple = ()
            if document_id:
                query += " WHERE document_id = ?"
                params = (document_id,)
            rows = connection.execute(query, params).fetchall()
            for row in rows:
                figure = dict(row)
                if not caption_looks_truncated(figure.get("caption")):
                    continue
                content = self._lookup_figure_chunk_content(
                    connection,
                    document_id=figure["document_id"],
                    figure_number=_figure_number(figure),
                )
                repaired = best_figure_caption(
                    caption=figure.get("caption"),
                    content=content,
                    visual_summary=figure.get("visual_summary"),
                    figure_number=_figure_number(figure),
                )
                if repaired == (figure.get("caption") or "").strip():
                    continue
                connection.execute(
                    "UPDATE document_figures SET caption = ? WHERE id = ?",
                    (repaired, figure["id"]),
                )
                updated += 1
        return updated

    def _apply_figure_caption(self, figure: dict, *, connection=None) -> None:
        content = figure.get("content")
        if not content and connection is not None:
            content = self._lookup_figure_chunk_content(
                connection,
                document_id=figure.get("document_id"),
                figure_number=_figure_number(figure),
            )
        figure["caption"] = best_figure_caption(
            caption=figure.get("caption"),
            content=content,
            visual_summary=figure.get("visual_summary"),
            figure_number=_figure_number(figure),
        )

    def _lookup_figure_chunk_content(
        self,
        connection,
        *,
        document_id: str | None,
        figure_number: int | None,
    ) -> str | None:
        if not document_id or figure_number is None:
            return None
        number = int(figure_number)
        patterns = (
            f"Figure {number}%",
            f"Fig. {number}%",
            f"Fig {number}%",
        )
        for pattern in patterns:
            row = connection.execute(
                """
                SELECT content
                FROM chunks
                WHERE document_id = ? AND content LIKE ?
                ORDER BY chunk_index ASC
                LIMIT 1
                """,
                (document_id, pattern),
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        return None

    def delete_document(self, document_id: str) -> bool:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if existing is None:
                return False
            connection.execute("DELETE FROM fts_document_cards WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM document_cards WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM collection_documents WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM document_tables WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM document_figures WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM rag_chunks_fts WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._delete_artifacts(document_id)
        return True

    def _collection_document_ids(self, collection_id: str | None) -> list[str]:
        if not collection_id:
            return []
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_id
                FROM collection_documents
                WHERE collection_id = ?
                ORDER BY added_at ASC
                """,
                (collection_id,),
            ).fetchall()
        return [row["document_id"] for row in rows]

    def collection_document_ids(self, collection_id: str | None) -> list[str]:
        """Return the canonical document scope for an API collection."""
        return self._collection_document_ids(collection_id)

    def resolve_document_ids_for_entities(
        self,
        *,
        entities: list[str],
        collection_id: str | None = None,
        query: str | None = None,
    ) -> list[str]:
        tokens = document_match_tokens(entities=entities, query=query)
        if not tokens:
            return []

        with connect(self.db_path) as connection:
            if collection_id:
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.source_path
                    FROM documents d
                    JOIN collection_documents cd ON cd.document_id = d.id
                    WHERE cd.collection_id = ?
                    ORDER BY d.indexed_at DESC
                    """,
                    (collection_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, filename, source_path
                    FROM documents
                    ORDER BY indexed_at DESC
                    """
                ).fetchall()

        scored: list[tuple[float, str]] = []
        for row in rows:
            score = _score_document_for_tokens(tokens, row["filename"], row["source_path"] or "")
            if score > 0:
                scored.append((score, row["id"]))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [document_id for _, document_id in scored]

    def resolve_document_mentions_for_query(
        self,
        *,
        query: str,
        collection_id: str | None = None,
        limit: int = 8,
    ) -> list[str]:
        """Resolve document names/aliases mentioned naturally in ``query``.

        Unlike the legacy entity scorer, this is a catalog-backed identity
        resolver.  It derives aliases from every indexed filename, title and
        document-card keyword, treats case and common separators as equivalent,
        and only accepts aliases that identify exactly one document in the
        selected corpus.  Ambiguous short names therefore fail closed instead
        of selecting whichever document happens to rank first.

        The returned IDs follow mention order, which preserves the user's A/B
        ordering for comparison questions without encoding any paper-specific
        mapping in routing code.
        """

        resolution = self.resolve_catalog_mentions(
            query=query,
            collection_id=collection_id,
        )
        return list(resolution.document_ids[: max(0, limit)])

    def resolve_catalog_mentions(
        self,
        *,
        query: str,
        collection_id: str | None = None,
    ) -> CatalogMentionResolution:
        """Return provenance-bearing unique and ambiguous catalog mentions."""

        query_words = _search_words(query)
        if not query_words:
            return CatalogMentionResolution(mentions=())

        with connect(self.db_path) as connection:
            if collection_id:
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.title, c.title_guess, c.keywords_json
                    FROM documents d
                    JOIN collection_documents cd ON cd.document_id = d.id
                    LEFT JOIN document_cards c ON c.document_id = d.id
                    WHERE cd.collection_id = ?
                    ORDER BY d.indexed_at DESC
                    """,
                    (collection_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.title, c.title_guess, c.keywords_json
                    FROM documents d
                    LEFT JOIN document_cards c ON c.document_id = d.id
                    ORDER BY d.indexed_at DESC
                    """
                ).fetchall()

        # One normalized alias may be emitted by several catalog fields, but it
        # is safe only when all occurrences point to the same canonical doc.
        alias_owners: dict[tuple[str, ...], set[str]] = {}
        aliases_by_document: dict[str, dict[tuple[str, ...], tuple[int, str]]] = {}
        for row in rows:
            document_id = str(row["id"])
            aliases = _catalog_document_aliases(
                filename=str(row["filename"] or ""),
                title=str(row["title"] or ""),
                title_guess=str(row["title_guess"] or ""),
                keywords_json=row["keywords_json"],
            )
            aliases_by_document[document_id] = aliases
            for alias in aliases:
                alias_owners.setdefault(alias, set()).add(document_id)

        # (start, end, strength, alias width, owner IDs, source, surface).
        # Strength is catalog provenance: filename > full title > keyword.
        raw_matches: list[
            tuple[int, int, int, int, tuple[str, ...], str, str]
        ] = []
        seen_alias_matches: set[tuple[int, tuple[str, ...]]] = set()
        for document_id, aliases in aliases_by_document.items():
            for alias, (strength, source) in aliases.items():
                position = _word_sequence_position(
                    query_words,
                    list(alias),
                    allow_terminal_prefix=(
                        source in {"filename", "filename_phrase_tail"}
                    ),
                )
                if position < 0:
                    continue
                match_key = (position, alias)
                if match_key in seen_alias_matches:
                    continue
                seen_alias_matches.add(match_key)
                owners = tuple(sorted(alias_owners.get(alias, ())))
                raw_matches.append(
                    (
                        position,
                        position + len(alias),
                        strength,
                        len(alias),
                        owners,
                        source,
                        " ".join(query_words[position : position + len(alias)]),
                    )
                )

        # A long canonical title can contain a word that happens to be another
        # paper's unique card keyword.  Prefer the enclosing, stronger identity
        # span instead of returning an unrelated extra document.
        matches: list[tuple[int, int, int, int, tuple[str, ...], str, str]] = []
        for candidate in raw_matches:
            start, end, strength, width, owner_ids, _, _ = candidate
            shadowed = any(
                (
                    set(other_owner_ids).isdisjoint(owner_ids)
                    or set(other_owner_ids).issubset(set(owner_ids))
                )
                and other_start <= start
                and other_end >= end
                and (
                    other_strength > strength
                    or (other_strength == strength and other_width > width)
                )
                for (
                    other_start,
                    other_end,
                    other_strength,
                    other_width,
                    other_owner_ids,
                    _,
                    _,
                ) in raw_matches
            )
            if not shadowed:
                matches.append(candidate)

        matches.sort(key=lambda item: (item[0], -item[2], -item[3], item[4]))
        mentions: list[CatalogDocumentMention] = []
        seen_document_ids: set[str] = set()
        seen_ambiguities: set[tuple[int, int, tuple[str, ...]]] = set()
        for start, end, strength, _, owner_ids, source, surface in matches:
            if len(owner_ids) == 1:
                document_id = owner_ids[0]
                if document_id in seen_document_ids:
                    continue
                seen_document_ids.add(document_id)
                mentions.append(
                    CatalogDocumentMention(
                        surface=surface,
                        start_word=start,
                        end_word=end,
                        document_id=document_id,
                        candidate_ids=owner_ids,
                        alias_source=source,
                        strength=strength,
                    )
                )
                continue
            ambiguity_key = (start, end, owner_ids)
            if len(owner_ids) < 2 or ambiguity_key in seen_ambiguities:
                continue
            seen_ambiguities.add(ambiguity_key)
            mentions.append(
                CatalogDocumentMention(
                    surface=surface,
                    start_word=start,
                    end_word=end,
                    document_id=None,
                    candidate_ids=owner_ids,
                    alias_source=source,
                    strength=strength,
                )
            )
        mentions.sort(key=lambda item: (item.start_word, -item.strength, item.surface))
        return CatalogMentionResolution(mentions=tuple(mentions))

    def resolve_explicit_document_ids_for_query(
        self,
        *,
        query: str,
        entities: list[str] | None = None,
        collection_id: str | None = None,
        compare: bool = False,
    ) -> list[str]:
        """Resolve an explicit current-turn paper/file reference against the catalog.

        This resolver is deliberately stricter than ordinary entity search.  It
        only runs for document-language queries (``bài/paper/file/...``), an
        explicit filename, or entities already extracted from ``Table N X``.
        Exact multi-word filenames win; otherwise each named alias must have a
        unique best catalog match.  Ties fail closed instead of picking an
        arbitrary paper.
        """

        text = str(query or "").strip()
        target_entities = [str(item).strip() for item in entities or [] if str(item).strip()]
        has_document_marker = bool(
            re.search(
                r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
                text,
                flags=re.IGNORECASE,
            )
        )
        has_explicit_filename = bool(re.search(r"\.(?:pdf|md|txt)(?!\w)", text, re.I))
        if not target_entities and not has_document_marker and not has_explicit_filename:
            return []

        with connect(self.db_path) as connection:
            if collection_id:
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.source_path, c.keywords_json
                    FROM documents d
                    JOIN collection_documents cd ON cd.document_id = d.id
                    LEFT JOIN document_cards c ON c.document_id = d.id
                    WHERE cd.collection_id = ?
                    ORDER BY d.indexed_at DESC
                    """,
                    (collection_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.source_path, c.keywords_json
                    FROM documents d
                    LEFT JOIN document_cards c ON c.document_id = d.id
                    ORDER BY d.indexed_at DESC
                    """
                ).fetchall()

        # Full multi-word filenames/titles are the strongest corpus-generic
        # signal and cover natural paper names that are not acronym-shaped.
        query_words = _search_words(text)
        exact_mentions: list[tuple[int, str]] = []
        for row in rows:
            filename = str(row["filename"] or "")
            stem = Path(filename).stem
            stem_words = _search_words(stem)
            filename_words = _search_words(filename)
            if not stem_words:
                continue
            # Single-token stems such as ASPIRE/MSF-SER are handled by the
            # target-entity path below so corrections can select the last paper
            # instead of matching every old name in the sentence.
            explicit_suffix = bool(filename_words and _contains_word_sequence(query_words, filename_words))
            multiword_stem = len(stem_words) >= 2 and _contains_word_sequence(query_words, stem_words)
            if explicit_suffix or multiword_stem:
                sequence = filename_words if explicit_suffix else stem_words
                exact_mentions.append((_word_sequence_position(query_words, sequence), str(row["id"])))

        # If the caller extracted explicit short targets, resolve those first.
        # This matters for corrections and comparisons: an older long filename
        # or a hyphenated acronym elsewhere in the sentence must not eclipse
        # the current target list.  Natural long titles normally have no named
        # entity extraction and therefore still take this exact path.
        if exact_mentions and not target_entities:
            ordered = _unique_ids_by_position(exact_mentions)
            return ordered[:2] if compare else ordered[-1:]

        if not target_entities:
            return []

        resolved: list[str] = []
        for entity in target_entities:
            tokens = document_match_tokens(entities=[entity], query=entity)
            if not tokens:
                continue
            scored: list[tuple[float, str]] = []
            for row in rows:
                filename = str(row["filename"] or "")
                source_path = str(row["source_path"] or "")
                score = _score_document_for_tokens(tokens, filename, source_path)
                filename_key = _compact_token(f"{filename} {source_path}")
                keyword_keys = _document_keyword_keys(row["keywords_json"])
                for token in tokens:
                    if token and token in filename_key:
                        score += 20.0
                    if token in keyword_keys:
                        score += 12.0
                if score > 0:
                    scored.append((score, str(row["id"])))

            scored.sort(key=lambda item: (-item[0], item[1]))
            if not scored:
                continue
            # Ambiguous short aliases (for example two different 9router docs)
            # must not silently select whichever row happened to sort first.
            if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
                continue
            document_id = scored[0][1]
            if document_id not in resolved:
                resolved.append(document_id)

        if resolved:
            return resolved[:2] if compare else resolved[-1:]
        if exact_mentions:
            ordered = _unique_ids_by_position(exact_mentions)
            return ordered[:2] if compare else ordered[-1:]
        return []

    def _delete_artifacts(self, document_id: str) -> None:
        if self.artifact_root is None:
            return
        shutil.rmtree(self.artifact_root / document_id, ignore_errors=True)


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def caption_identifies_figure(caption: str) -> bool:
    """Reject artifacts whose own printed label says they are a figure."""

    return bool(
        re.match(
            r"^\s*(?:fig(?:ure)?\.?|hình|hinh)\s*(?:no\.?\s*)?[#:]?\s*"
            r"(?:\d+|[ivxlcdm]+)(?!\w)",
            str(caption or ""),
            flags=re.IGNORECASE,
        )
    )


_PRINTED_TABLE_CAPTION_LINE_RE = re.compile(
    r"^\s*(?P<label>table\s*(?:no\.?\s*)?[#:]?\s*(?P<number>\d+)\s*[:.]\s*[^\r\n]*)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _reconcile_mislabeled_table_captions(
    connection,
    *,
    document_id: str,
    tables: list[dict],
) -> None:
    """Repair a table caption only when page-local provenance is unambiguous.

    Some PDF layouts place a nearby figure caption closer to a table than the
    table's own printed caption. A valid Docling table must not be discarded in
    that case, but neither its positional index nor a caption from another page
    is safe evidence. Reconciliation therefore requires all of the following:

    * the artifact has structured Markdown and is currently labelled Figure;
    * its own document/page text contains an explicit ``Table N:`` caption;
    * that number is not already owned by another table on the same page; and
    * exactly one suspect artifact and one unmatched caption remain on the page.

    Ambiguous pages stay untouched and are filtered by the existing figure
    guard. The repair is runtime metadata only; canonical storage is unchanged.
    """

    suspects_by_page: dict[int, list[dict]] = {}
    assigned_by_page: dict[int, set[int]] = {}
    for table in tables:
        page_number = table.get("page_number")
        if not isinstance(page_number, int):
            continue
        caption = str(table.get("caption") or "")
        printed_number = _printed_table_caption_number(caption)
        if printed_number is not None:
            assigned_by_page.setdefault(page_number, set()).add(printed_number)
        if caption_identifies_figure(caption) and _has_structured_table_markdown(
            str(table.get("markdown") or "")
        ):
            suspects_by_page.setdefault(page_number, []).append(table)

    if not suspects_by_page:
        return
    page_numbers = sorted(suspects_by_page)
    placeholders = ",".join("?" for _ in page_numbers)
    chunk_rows = connection.execute(
        f"""
        SELECT page_number, content
        FROM chunks
        WHERE document_id = ?
          AND page_number IN ({placeholders})
        ORDER BY chunk_index ASC
        """,
        (document_id, *page_numbers),
    ).fetchall()

    captions_by_page: dict[int, dict[int, str]] = {}
    for row in chunk_rows:
        page_number = row["page_number"]
        if not isinstance(page_number, int):
            continue
        for match in _PRINTED_TABLE_CAPTION_LINE_RE.finditer(
            str(row["content"] or "")
        ):
            number = int(match.group("number"))
            label = " ".join(match.group("label").split()).strip()
            previous = captions_by_page.setdefault(page_number, {}).get(number)
            if previous is None or len(label) > len(previous):
                captions_by_page[page_number][number] = label

    for page_number, suspects in suspects_by_page.items():
        unmatched = {
            number: label
            for number, label in captions_by_page.get(page_number, {}).items()
            if number not in assigned_by_page.get(page_number, set())
        }
        if len(suspects) != 1 or len(unmatched) != 1:
            continue
        number, recovered_caption = next(iter(unmatched.items()))
        suspect = suspects[0]
        original_caption = str(suspect.get("caption") or "")
        suspect["caption"] = recovered_caption
        metadata = suspect.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            suspect["metadata"] = metadata
        metadata.update(
            {
                "caption_reconciled": True,
                "caption_reconciled_number": number,
                "original_caption": original_caption,
                "caption_reconciliation_source": "same_document_page_text",
            }
        )


def _printed_table_caption_number(caption: str) -> int | None:
    match = _PRINTED_TABLE_CAPTION_LINE_RE.match(str(caption or ""))
    return int(match.group("number")) if match is not None else None


def _has_structured_table_markdown(markdown: str) -> bool:
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip()]
    if len(lines) < 3 or "|" not in lines[0]:
        return False
    return bool(
        re.match(
            r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$",
            lines[1],
        )
    )


def _search_words(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_text.casefold())


def _contains_word_sequence(words: list[str], sequence: list[str]) -> bool:
    return _word_sequence_position(words, sequence) >= 0


def _word_sequence_position(
    words: list[str],
    sequence: list[str],
    *,
    allow_terminal_prefix: bool = False,
) -> int:
    if not words or not sequence or len(sequence) > len(words):
        return -1
    width = len(sequence)
    for index in range(len(words) - width + 1):
        candidate = words[index : index + width]
        if candidate == sequence:
            return index
        if not allow_terminal_prefix or candidate[:-1] != sequence[:-1]:
            continue
        stored_tail = sequence[-1]
        query_tail = candidate[-1]
        minimum_prefix = 12 if width == 1 else 4
        if (
            len(stored_tail) >= minimum_prefix
            and len(query_tail) > len(stored_tail)
            and query_tail.startswith(stored_tail)
        ):
            return index
    return -1


def _unique_ids_by_position(items: list[tuple[int, str]]) -> list[str]:
    ordered: list[str] = []
    for _, document_id in sorted(items, key=lambda item: (item[0], item[1])):
        if document_id not in ordered:
            ordered.append(document_id)
    return ordered


def _document_keyword_keys(raw: str | None) -> set[str]:
    try:
        values = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return set()
    if not isinstance(values, list):
        return set()
    return {_compact_token(str(item)) for item in values if _compact_token(str(item))}


def _catalog_document_aliases(
    *,
    filename: str,
    title: str,
    title_guess: str,
    keywords_json: str | None,
) -> dict[tuple[str, ...], tuple[int, str]]:
    """Build conservative, separator-insensitive aliases for one catalog row."""

    canonical_aliases: list[tuple[str, int, bool, str]] = []
    stem = Path(filename).stem.strip()
    if stem:
        canonical_aliases.append((stem, 3, True, "filename"))
        stem_tokens = re.findall(r"[A-Za-z0-9]+", stem)
        for token in stem_tokens:
            uppercase_count = sum(char.isupper() for char in token)
            alpha_count = sum(char.isalpha() for char in token)
            if (
                len(token) >= 3
                and alpha_count >= 2
                and (
                    any(char.isdigit() for char in token)
                    or uppercase_count >= 2
                )
                and (not token.isupper() or len(token) <= 6)
                and token.casefold() not in _CATALOG_KEYWORD_STOPWORDS
                and token.casefold() not in _GENERIC_ENTITY_TOKENS
            ):
                canonical_aliases.append((token, 2, True, "filename_token"))
        # Generate identifier-led compound aliases from the catalog itself.
        # This covers separator/optional-modifier forms such as
        # ``ACRONYM-based fusion`` without encoding any concrete paper name.
        compound_modifiers = {
            "architecture",
            "based",
            "framework",
            "fusion",
            "method",
            "model",
            "network",
            "system",
            "transformer",
        }
        removable_modifiers = {"based", "method", "model", "system"}
        normalized_stem_tokens = [
            (_search_words(token) or [""])[0] for token in stem_tokens
        ]
        for index, (raw_token, anchor) in enumerate(
            zip(stem_tokens, normalized_stem_tokens, strict=True)
        ):
            uppercase_count = sum(char.isupper() for char in raw_token)
            if (
                len(anchor) < 3
                or uppercase_count < 2
                or anchor in _CATALOG_KEYWORD_STOPWORDS
                or anchor in _GENERIC_ENTITY_TOKENS
            ):
                continue
            tail: list[str] = []
            for modifier in normalized_stem_tokens[index + 1 : index + 5]:
                if modifier not in compound_modifiers:
                    break
                tail.append(modifier)
                canonical_aliases.append(
                    (" ".join([anchor, *tail]), 2, True, "filename_compound")
                )
            simplified = [
                anchor,
                *(item for item in tail if item not in removable_modifiers),
            ]
            if len(simplified) >= 2:
                canonical_aliases.append(
                    (" ".join(simplified), 2, True, "filename_compound")
                )
        # A user often remembers a distinctive title phrase rather than the
        # complete filename. Generate only contiguous catalog-derived phrases:
        # three tokens when at least one is informative, otherwise four. Short
        # generic topics (``speech emotion recognition``) remain ineligible.
        # Ownership is still checked corpus-wide by ``resolve_catalog_mentions``.
        normalized_filename_words = _search_words(stem)
        for start in range(len(normalized_filename_words)):
            for end in range(start + 3, len(normalized_filename_words) + 1):
                phrase = tuple(normalized_filename_words[start:end])
                if not _is_distinctive_filename_phrase(phrase):
                    continue
                source = (
                    "filename_phrase_tail"
                    if end == len(normalized_filename_words)
                    else "filename_phrase"
                )
                canonical_aliases.append((" ".join(phrase), 2, False, source))
    for value in (title, title_guess):
        if str(value or "").strip():
            canonical_aliases.append((str(value).strip(), 2, False, "title"))
    try:
        keywords = json.loads(keywords_json or "[]")
    except (TypeError, json.JSONDecodeError):
        keywords = []
    if isinstance(keywords, list):
        for item in keywords:
            keyword = str(item).strip()
            if keyword and _is_identifier_like_keyword(
                keyword,
                identity_context=" ".join((stem, title, title_guess)),
            ):
                canonical_aliases.append((keyword, 1, False, "keyword"))

    aliases: dict[tuple[str, ...], tuple[int, str]] = {}
    for raw_alias, strength, allow_short, source in canonical_aliases:
        words = tuple(_search_words(raw_alias))
        phrase_safe = source.startswith("filename_phrase") and (
            _is_distinctive_filename_phrase(words)
        )
        if not phrase_safe and not _is_safe_catalog_alias(
            words,
            allow_short=allow_short,
        ):
            continue
        previous = aliases.get(words)
        if previous is None or strength > previous[0]:
            aliases[words] = (strength, source)
        # Users routinely collapse MSF-SER / KS-Transformer / filename
        # separators.  Add the exact collapsed form, never a substring match.
        if len(words) > 1:
            compact = "".join(words)
            if phrase_safe or _is_safe_catalog_alias(
                (compact,),
                allow_short=allow_short,
            ):
                previous = aliases.get((compact,))
                if previous is None or strength > previous[0]:
                    aliases[(compact,)] = (strength, source)
    return aliases


def _is_safe_catalog_alias(
    words: tuple[str, ...],
    *,
    allow_short: bool = False,
) -> bool:
    if not words:
        return False
    compact = "".join(words)
    minimum = 3 if allow_short else 4
    if len(compact) < minimum or compact in _GENERIC_ENTITY_TOKENS:
        return False
    if len(words) == 1:
        token = words[0]
        return len(token) >= minimum and token not in _GENERIC_ENTITY_TOKENS
    if allow_short and len(compact) >= 6:
        # The complete multi-token filename stem is canonical even when every
        # component is a generic word (for example an internal project-plan
        # filename). Generic filtering applies to derived partial aliases, not
        # to the full identity supplied by the catalog itself.
        return True
    informative = [
        word
        for word in words
        if len(word) >= 3 and word not in _GENERIC_ENTITY_TOKENS
    ]
    return bool(informative)


def _is_distinctive_filename_phrase(words: tuple[str, ...]) -> bool:
    if len(words) < 3 or len("".join(words)) < 12:
        return False
    grammatical = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "using",
        "with",
    }
    if words[0] in grammatical or words[-1] in grammatical:
        return False
    informative = [
        word
        for word in words
        if word not in grammatical and word not in _GENERIC_ENTITY_TOKENS
    ]
    return bool(informative) or len(words) >= 4


_CATALOG_KEYWORD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "based",
        "categorical",
        "classification",
        "dataset",
        "detection",
        "dimensional",
        "effective",
        "emotion",
        "for",
        "from",
        "fusion",
        "in",
        "language",
        "local",
        "method",
        "model",
        "multimodal",
        "network",
        "of",
        "on",
        "proposed",
        "recognition",
        "speech",
        "the",
        "to",
        "using",
        "visual",
        "while",
        "with",
    }
)


def _is_identifier_like_keyword(keyword: str, *, identity_context: str) -> bool:
    """Keep aliases that look like names, not arbitrary topical keywords."""

    words = tuple(_search_words(keyword))
    if not words:
        return False
    compact = "".join(words)
    if len(compact) < 3:
        return False
    if len(words) >= 2:
        # A separator does not make a topical phrase a document identity.
        # Keep a compound only when the catalog's filename/title actually owns
        # it (Pitch-fusion, FM-MOE), or its identifier-led initials reproduce a
        # canonical identity token (KS-Transformer -> KST). This rejects weak
        # retrieval keywords such as ``frame-level`` and ``f1-score`` without
        # encoding any paper-specific alias.
        identity_words = set(_search_words(identity_context))
        compact_identity = _compact_token(identity_context)
        compact_keyword = "".join(words)
        identifier_initials = words[0] + "".join(word[0] for word in words[1:])
        return (
            compact_keyword in compact_identity
            or (
                len(identifier_initials) >= 3
                and identifier_initials in identity_words
            )
        )
    token = words[0]
    if (
        token in _CATALOG_KEYWORD_STOPWORDS
        or token in _GENERIC_ENTITY_TOKENS
        or len(token) > 12
    ):
        return False
    if any(char.isdigit() for char in token):
        return token in _compact_token(identity_context)
    # Lower-cased acronyms in generated cards (for example LPMN) remain useful
    # when their letters are the initials of informative identity words.
    initials = "".join(
        word[0]
        for word in _search_words(identity_context)
        if word
        and word
        not in {
            "a",
            "an",
            "and",
            "for",
            "from",
            "in",
            "of",
            "on",
            "the",
            "to",
            "using",
            "with",
        }
    )
    return 3 <= len(token) <= 10 and token in initials


_GENERIC_ENTITY_TOKENS = frozenset(
    {
        "agent",
        "addendum",
        "fusion",
        "model",
        "architecture",
        "figure",
        "emotion",
        "recognition",
        "audio",
        "visual",
        "speech",
        "text",
        "dataset",
        "paper",
        "proposed",
        "method",
        "based",
        "multi",
        "label",
        "single",
        "ser",
        "robust",
        "overview",
        "structure",
        "diagram",
        "chart",
        "plot",
        "table",
        "benchmark",
        "chat",
        "code",
        "context",
        "database",
        "embedding",
        "graph",
        "history",
        "index",
        "lancedb",
        "llm",
        "memory",
        "plan",
        "project",
        "rag",
        "retrieval",
        "router",
        "tool",
        "vector",
        "results",
        "introduction",
        "abstract",
        "pipeline",
        "components",
        "training",
        "distillation",
        "definition",
        "visualization",
    }
)

_COMPOUND_ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"from_single_to_multi_label_ser", "mamba_fusion"),
    (r"mamba[- ]?based[- ]?(?:fusion)?", "mamba_fusion"),
    (r"mamba[- ]?fusion", "mamba_fusion"),
    (r"pitch[- ]?fusion", "pitch_fusion"),
    (r"ks[- ]?transformer", "kst"),
    (r"\bkst\b", "kst"),
    (r"wav2small", "wav2small"),
    (r"wav2vec", "wav2vec"),
    (r"\bcrab\b", "crab"),
    (r"\bwhiser\b", "whiser"),
    (r"\bvisec\b", "visec"),
    (r"\baspire\b", "aspire"),
]

_SHORT_MODEL_NAMES = frozenset({"mamba", "kst", "crab", "visec", "whiser", "aspire"})


def document_match_tokens(*, entities: list[str] | None, query: str | None = None) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        normalized = _compact_token(token)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tokens.append(normalized)

    normalized_query = " ".join(str(query or "").lower().split())
    for pattern, label in _COMPOUND_ENTITY_PATTERNS:
        if normalized_query and re.search(pattern, normalized_query, flags=re.IGNORECASE):
            add(label)

    for entity in entities or []:
        text = str(entity).strip()
        if not text:
            continue
        lowered = text.lower()
        matched_compound = False
        for pattern, label in _COMPOUND_ENTITY_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                add(label)
                matched_compound = True
        if matched_compound:
            continue
        compact = _compact_token(text)
        if compact in _GENERIC_ENTITY_TOKENS:
            continue
        if compact in _SHORT_MODEL_NAMES or len(compact) >= 6 or any(char.isupper() for char in text):
            add(compact)

    return tokens


def _score_document_for_tokens(tokens: list[str], filename: str, source_path: str) -> float:
    if not tokens:
        return 0.0

    haystack_compact = _compact_token(f"{filename} {source_path}")
    score = 0.0
    specific_hits = 0

    for token in tokens:
        if token in {"mamba_fusion", "mambafusion"}:
            has_mamba = "mamba" in haystack_compact
            has_fusion = "fusion" in haystack_compact
            if has_mamba and has_fusion:
                score += 24.0
                specific_hits += 1
            elif has_mamba:
                score += 10.0
                specific_hits += 1
            continue

        if token in {"pitch_fusion", "pitchfusion"}:
            has_pitch = "pitch" in haystack_compact
            has_fusion = "fusion" in haystack_compact
            if has_pitch and has_fusion:
                score += 20.0
                specific_hits += 1
            continue

        if token in _GENERIC_ENTITY_TOKENS:
            if token in haystack_compact:
                score += 0.15
            continue

        if token in haystack_compact:
            weight = 12.0 if token in _SHORT_MODEL_NAMES else max(5.0, len(token) * 0.45)
            score += weight
            specific_hits += 1

    if specific_hits == 0:
        return 0.0
    if score < 2.0 and not any(
        token in {"mamba_fusion", "mambafusion", "pitch_fusion", "pitchfusion"}
        or token in _SHORT_MODEL_NAMES
        or len(token) >= 6
        for token in tokens
    ):
        return 0.0
    return score


def _document_ids_from_vector_results(results, max_documents: int | None = None) -> list[str]:
    seen: set[str] = set()
    document_ids: list[str] = []
    for result in results:
        document_id = result.metadata.get("document_id")
        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
            if max_documents is not None and len(document_ids) >= max_documents:
                break
    return document_ids


def _filter_card_results_for_query(query: str, results) -> list:
    constraints = _query_topic_constraints(query)
    if not constraints:
        return list(results)
    constrained = []
    for result in results:
        tags = set(result.metadata.get("topic_tags") or [])
        text = f"{result.text} {result.metadata.get('filename') or ''}".lower()
        if all(tag in tags or _constraint_text_match(tag, text) for tag in constraints):
            constrained.append(result)
    return constrained or list(results)


def _query_topic_constraints(query: str) -> list[str]:
    normalized = query.lower().replace("-", " ")
    constraints: list[str] = []
    if "audio visual" in normalized or "audiovisual" in normalized or "visual audio" in normalized:
        constraints.append("audio_visual")
    if "speech emotion" in normalized or "ser" in normalized:
        constraints.append("speech_emotion_recognition")
    if "multimodal" in normalized or "fusion" in normalized or "cross modal" in normalized:
        constraints.append("multimodal_fusion")
    return constraints


def _constraint_text_match(tag: str, text: str) -> bool:
    if tag == "audio_visual":
        return "audio visual" in text or "audiovisual" in text or "visual audio" in text
    if tag == "speech_emotion_recognition":
        return "speech emotion" in text or "emotion recognition" in text
    if tag == "multimodal_fusion":
        return "multimodal" in text or "fusion" in text or "cross modal" in text
    return False


def _merge_hybrid_chunks(
    query: str,
    vector_chunks,
    fts_chunks: list[RetrievedChunk],
    top_k: int,
    *,
    visual_boost: bool = False,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for rank, result in enumerate(vector_chunks, start=1):
        chunk_id = result.id
        merged[chunk_id] = {
            **_vector_payload(result),
            "score": _rrf(rank),
            "vector_rank": rank,
            "fts_rank": None,
        }

    for rank, chunk in enumerate(fts_chunks, start=1):
        existing = merged.get(chunk.chunk_id)
        if existing:
            existing["score"] += _rrf(rank)
            existing["fts_rank"] = rank
            existing["retrieval_channels"].append("sqlite_fts5")
        else:
            merged[chunk.chunk_id] = {
                **chunk.__dict__,
                "score": _rrf(rank),
                "vector_rank": None,
                "fts_rank": rank,
                "retrieval_channels": ["sqlite_fts5"],
            }

    for item in merged.values():
        item["rrf_score"] = item["score"]
        item["rerank_boost"] = _lexical_rerank_boost(query, item, visual_boost=visual_boost)
        item["score"] = item["rrf_score"] + item["rerank_boost"]

    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


_EXPLAIN_MARKERS = (
    "giải thích",
    "explain",
    "là gì",
    "what is",
    "architecture",
    "introduction",
    "abstract",
    "overview",
    "mô tả",
    "tổng quan",
)


def _vector_payload(result) -> dict:
    metadata = result.metadata or {}
    return {
        "chunk_id": result.id,
        "document_id": metadata.get("document_id"),
        "source_path": metadata.get("source_path"),
        "filename": metadata.get("filename"),
        "content": result.text,
        "raw_vector_score": result.score,
        "retrieval_channels": ["lancedb", _lancedb_channel(result.source)],
        "heading_path": metadata.get("heading_path") or [],
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index") if metadata.get("chunk_index") is not None else metadata.get("order_index"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "section_title": metadata.get("section_title"),
        "chunk_type": metadata.get("chunk_type") or result.source,
        "artifact_type": metadata.get("artifact_type"),
        "caption": metadata.get("caption"),
        "image_path": metadata.get("image_path"),
        "image_url": f"/rag/figures/{metadata.get('figure_id')}/image" if metadata.get("figure_id") else None,
        "table_id": metadata.get("table_id"),
        "figure_id": metadata.get("figure_id"),
        "table_index": metadata.get("table_index"),
        "figure_index": metadata.get("figure_index"),
        "figure_label": metadata.get("figure_label"),
        "figure_number": metadata.get("figure_number"),
        "logical_group_id": metadata.get("logical_group_id"),
        "child_count": metadata.get("child_count"),
        "caption_source": metadata.get("caption_source"),
        "quality_status": metadata.get("quality_status"),
        "asset_kind": metadata.get("asset_kind"),
        "is_content": metadata.get("is_content"),
        "is_complete": metadata.get("is_complete"),
        "figure_type": metadata.get("figure_type"),
    }


def _lancedb_channel(source: str) -> str:
    if source == "table":
        return "lancedb_table_chunks"
    if source == "figure":
        return "lancedb_figure_chunks"
    if source == "document_card":
        return "lancedb_document_cards"
    return "lancedb_text_chunks"


def _is_text_chunk_result(result: dict) -> bool:
    chunk_type = result.get("chunk_type")
    artifact_type = result.get("artifact_type")
    chunk_id = str(result.get("chunk_id") or "")
    return (
        artifact_type is None
        and chunk_type in {None, "text", "text_chunk"}
        and not chunk_id.startswith(("table:", "figure:"))
    )


def _result_section_title(result: dict) -> str | None:
    section = result.get("section_title")
    if isinstance(section, str) and section.strip():
        return section.strip()
    heading_path = result.get("heading_path") or []
    if heading_path:
        return str(heading_path[-1]).strip() or None
    return None


def _neighbor_matches_section(neighbor: dict, section_title: str | None) -> bool:
    neighbor_section = _result_section_title(neighbor) or (neighbor.get("metadata") or {}).get("section_title")
    if section_title and neighbor_section:
        return _normalize_section(section_title) == _normalize_section(str(neighbor_section))
    if section_title and _starts_with_heading(neighbor.get("content") or ""):
        return False
    return True


def _normalize_section(section: str) -> str:
    return " ".join(section.lower().replace("#", " ").split())


def _starts_with_heading(content: str) -> bool:
    first_line = str(content or "").strip().splitlines()[0].strip() if str(content or "").strip() else ""
    if not first_line:
        return False
    if first_line.startswith("#"):
        return True
    return bool(re.match(r"^(?:[IVX]+\.|\d+(?:\.\d+)*\.?)\s+[A-ZÀ-ỸA-Za-z][^\n.]{2,120}$", first_line))


def _expanded_context_text(result: dict, neighbors: list[dict], *, max_neighbor_chars: int) -> str:
    previous = [neighbor for neighbor in neighbors if int(neighbor["chunk_index"]) < int(result["chunk_index"])]
    following = [neighbor for neighbor in neighbors if int(neighbor["chunk_index"]) > int(result["chunk_index"])]
    parts: list[str] = []
    for neighbor in previous:
        parts.append("[previous context]\n" + _truncate_neighbor(neighbor["content"], max_neighbor_chars))
    parts.append("[retrieved chunk]\n" + str(result.get("content") or result.get("text") or ""))
    for neighbor in following:
        parts.append("[next context]\n" + _truncate_neighbor(neighbor["content"], max_neighbor_chars))
    return "\n\n".join(parts)


def _expanded_parent_context_text(
    result: dict,
    parent: dict,
    *,
    max_parent_chars: int,
) -> str:
    retrieved = str(result.get("content") or result.get("text") or "").strip()
    parent_content = str(parent.get("content") or "").strip()
    if not parent_content or parent_content == retrieved:
        return retrieved
    return (
        "[retrieved chunk / child]\n"
        f"{retrieved}\n\n"
        "[parent section context]\n"
        f"{_truncate_neighbor(parent_content, max_parent_chars)}"
    )


def _truncate_neighbor(content: str, max_chars: int) -> str:
    content = " ".join(content.split())
    if len(content) <= max_chars:
        return content
    boundary = content.rfind(". ", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = content.rfind(" ", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = max_chars
    return content[:boundary].rstrip() + "..."


def _decode_artifact(row: dict) -> dict:
    row["bbox"] = json.loads(row.pop("bbox_json") or "null")
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    return row


def _figure_number(figure: dict) -> int | None:
    metadata = figure.get("metadata")
    if not isinstance(metadata, dict):
        try:
            metadata = json.loads(figure.get("metadata_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
    value = metadata.get("figure_number")
    if isinstance(value, int):
        return value
    label = extract_figure_label(figure.get("caption"))
    return label.number if label else None


def _is_low_signal_visual_payload(item: dict) -> bool:
    caption = str(item.get("caption") or "").strip().lower()
    content = str(item.get("content") or "").strip().lower()
    if caption.startswith("page ") and "visual fallback" in caption:
        return True
    if "visual fallback" in caption or "visual fallback" in content:
        return True
    return caption.startswith("figure extracted from page") or content.startswith(
        "figure extracted from page"
    )


def _query_prefers_figures(query: str) -> bool:
    lowered = " ".join(query.lower().split())
    markers = (
        "figure",
        "fig.",
        "fig ",
        "hình",
        "architecture",
        "diagram",
        "sơ đồ",
        "chart",
        "plot",
        "biểu đồ",
    )
    return any(marker in lowered for marker in markers)


def _lexical_rerank_boost(query: str, result: dict, *, visual_boost: bool = False) -> float:
    query_lower = query.lower()
    query_tokens = _rerank_tokens(query)
    boost = 0.0

    haystack = " ".join(
        str(part or "")
        for part in [
            result.get("filename"),
            " ".join(result.get("heading_path") or []),
            result.get("caption"),
            result.get("image_path"),
            result.get("content"),
        ]
    )
    haystack_tokens = set(_rerank_tokens(haystack))
    if query_tokens and haystack_tokens:
        overlap_ratio = len(set(query_tokens) & haystack_tokens) / len(set(query_tokens))
        boost += min(overlap_ratio * 0.012, 0.012)

    content_lower = str(result.get("content") or "").lower()
    if any(marker in query_lower for marker in _EXPLAIN_MARKERS):
        if any(
            marker in content_lower
            for marker in (
                "introduction",
                "abstract",
                "we propose",
                "architecture",
                "listing",
                "def __init__",
                "class ",
                "distillation",
                "parameters",
            )
        ):
            boost += 0.028
        if any(marker in content_lower for marker in ("conclusion", "reviewer #", "acknowledgment")):
            boost -= 0.012

    if visual_boost:
        artifact_type = result.get("artifact_type") or result.get("chunk_type")
        if artifact_type in {"figure", "table"}:
            boost += 0.02

    artifact_type = result.get("artifact_type")
    if artifact_type == "figure" and any(
        term in query_lower
        for term in [
            "figure",
            "fig",
            "diagram",
            "architecture",
            "model",
            "pipeline",
            "component",
            "components",
            "thành phần",
            "mô hình",
            "sơ đồ",
        ]
    ):
        boost += 0.018
    if artifact_type == "table" and any(
        term in query_lower
        for term in ["table", "metric", "result", "score", "bảng", "kết quả", "điểm"]
    ):
        boost += 0.014

    channels = set(result.get("retrieval_channels") or [])
    if {"lancedb", "sqlite_fts5"}.issubset(channels):
        boost += 0.004
    if result.get("fts_rank") == 1:
        boost += 0.002
    return boost


def _rerank_tokens(text: str) -> list[str]:
    from app.rag.vietnamese_text import tokenize_for_fts

    return tokenize_for_fts(text, max_tokens=24)
