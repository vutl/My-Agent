from dataclasses import dataclass, field
import asyncio
import json
import math
from pathlib import Path
import re
import shutil
import time
import unicodedata
from typing import Any, Callable

from app.db.sqlite import connect
from app.rag.artifact_references import extract_artifact_references
from app.rag.embeddings import EmbeddingProvider
from app.rag.figure_caption import best_figure_caption, caption_looks_truncated
from app.rag.figure_quality import extract_figure_label
from app.rag.evidence import EvidenceBundle
from app.rag.evidence_identity import canonical_artifact_type, has_physical_pages
from app.rag.evidence_obligations import (
    EvidenceObligationPlan,
    obligation_search_query,
    plan_evidence_obligations,
    reserve_obligation_coverage,
    tag_obligation_results,
)
from app.rag.fusion import fuse_ranked_results
from app.rag.paper_facets import requested_paper_facets, retrieval_query_expansion
from app.rag.page_labels import page_label_number, page_label_values
from app.rag.reranker import rerank_candidates
from app.rag.retriever import RetrievedChunk, build_fts_query, build_fts_queries, search_chunks
from app.retrieval_store.base import RetrievalFilter
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore
from app.rag.visual_retrieval import (
    VisualRetrievalProvider,
    visual_search_results_as_evidence,
)
from app.services.query_rewrite_service import has_enumerated_evidence_intent


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


_REFERENCES_PER_ANCHOR = 2
_REFERENCE_STOPWORDS = frozenset(
    "the a an of in on for to and or is are was were be by with what which how "
    "does do did this that these those paper table figure fig show shows result "
    "results về của và các những là được trong cho với này đó bảng hình nào gì "
    "như thế bài".split()
)


def _reference_query_terms(query: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w-]{3,}", str(query or "").casefold())
        if token not in _REFERENCE_STOPWORDS and not token.isdigit()
    }


def _reference_is_relevant(query: str | None, terms: set[str], *, kind: str, text: str) -> bool:
    """Whether a referenced artifact visibly serves the question: it shares a
    content term, or it is the kind of evidence asked for (results/list →
    table, visual → figure). Used to order a paragraph's references so the
    per-anchor cap keeps the useful ones first."""
    if not query:
        return True
    from app.services.query_rewrite_service import has_result_table_intent, has_visual_intent

    folded = text.casefold()
    if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded) for term in terms):
        return True
    if kind == "table" and has_result_table_intent(query):
        return True
    return kind == "figure" and has_visual_intent(query)


@dataclass(frozen=True)
class RagService:
    db_path: Path
    artifact_root: Path | None = None

    def resolve_artifact_path(self, raw_path: str | Path | None) -> Path | None:
        """Resolve a persisted artifact path within the current artifact root.

        Older revisions persisted absolute paths.  Those paths are valid only in
        the process that created the index (and break isolated clones, backups,
        or a moved workspace).  Treat the configured artifact root as the
        authority and, when an absolute path points at a previous ``artifacts``
        root, remap the document-relative suffix into the current root.  Never
        return a path outside the configured root.
        """

        if self.artifact_root is None or not raw_path:
            return None
        root = self.artifact_root.expanduser().resolve()
        candidate = Path(str(raw_path)).expanduser()

        def _safe_existing(path: Path) -> Path | None:
            try:
                resolved = path.resolve()
                if resolved.is_relative_to(root) and resolved.is_file():
                    return resolved
            except (OSError, RuntimeError, ValueError):
                return None
            return None

        direct = _safe_existing(candidate)
        if direct is not None:
            return direct

        # Portable remapping for old absolute paths such as
        # ``.../data/artifacts/<document-id>/figures/...``.  Use the final
        # ``artifacts`` component so arbitrary parent directories are ignored.
        parts = candidate.parts
        try:
            marker = max(index for index, part in enumerate(parts) if part == "artifacts")
        except ValueError:
            marker = -1
        if marker >= 0 and marker + 1 < len(parts):
            remapped = root.joinpath(*parts[marker + 1 :])
            return _safe_existing(remapped)

        if not candidate.is_absolute():
            return _safe_existing(root / candidate)
        return None

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

    def search_active_artifacts_lexical(
        self,
        query: str,
        *,
        top_k: int = 12,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return active table/figure artifacts with deterministic lexical signal.

        Canonical artifacts are stored separately from text chunks, so FTS5
        over ``rag_chunks_fts_v2`` cannot recover a table whose caption/OCR is
        the only textual representation.  This bounded projection scans only
        the active revision rows in the requested document scope and returns
        artifacts that share content-bearing query terms.  It is a retrieval
        fallback, not a second semantic index: no answer/qrel data is read and
        no staged or retired revision can participate.
        """

        limit = max(1, min(int(top_k), 64))
        terms = tuple(
            term for term in _page_context_terms(query)
            if len(term) >= 2
        )
        if not terms or document_ids is not None and not document_ids:
            return []
        where_scope = ""
        params: list[Any] = []
        if document_ids is not None:
            placeholders = ",".join("?" for _ in document_ids)
            where_scope = f"AND a.document_id IN ({placeholders})"
            params.extend(document_ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT a.id AS artifact_id, a.document_id, a.page_number,
                       a.caption, a.markdown, NULL AS visual_summary,
                       NULL AS image_path, a.metadata_json, a.table_index,
                       NULL AS figure_index, a.extraction_method AS extraction_method,
                       'table' AS artifact_type,
                       d.filename, d.source_path, d.active_revision_id,
                       a.revision_id
                FROM document_tables AS a
                JOIN documents AS d ON d.id = a.document_id
                WHERE a.revision_id = d.active_revision_id
                  {where_scope}
                UNION ALL
                SELECT a.id AS artifact_id, a.document_id, a.page_number,
                       a.caption, NULL AS markdown, a.visual_summary,
                       a.image_path, a.metadata_json, NULL AS table_index,
                       a.figure_index, a.extraction_method AS extraction_method,
                       'figure' AS artifact_type,
                       d.filename, d.source_path, d.active_revision_id,
                       a.revision_id
                FROM document_figures AS a
                JOIN documents AS d ON d.id = a.document_id
                WHERE a.revision_id = d.active_revision_id
                  {where_scope}
                """,
                tuple(params + params),
            ).fetchall()

        candidates: list[tuple[tuple[float, int, str], dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            try:
                metadata = json.loads(item.pop("metadata_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            # Some parsers keep the artifact label/index only in metadata
            # (for example a table-image with no reliable caption).  Include
            # those canonical identifiers in the lexical projection so an
            # explicit ``Table 3``/``Figure 12`` query can route correctly.
            item["metadata"] = metadata
            artifact_type = canonical_artifact_type(item, default="artifact")
            artifact_number = (
                metadata.get("figure_number")
                if artifact_type == "figure"
                else metadata.get("table_number")
            )
            artifact_label = (
                metadata.get("figure_label")
                if artifact_type == "figure"
                else metadata.get("table_label")
            )
            primary_searchable = " ".join(
                str(value or "")
                for value in (
                    item.get("caption"),
                    item.get("markdown"),
                    item.get("visual_summary"),
                    metadata.get("ocr_text"),
                    artifact_label,
                    artifact_number,
                    # Indexes are zero-based internally but users generally
                    # refer to the one-based printed label.  Keep both forms
                    # only when the parser did not provide a stronger label.
                    _one_based_index(
                        item.get("figure_index")
                        if artifact_type == "figure" and not artifact_label
                        else None
                    ),
                    _one_based_index(
                        item.get("table_index")
                        if artifact_type == "table" and not artifact_label
                        else None
                    ),
                )
            )
            # Adapter context can contain a document-wide list of every
            # figure/table label.  Treat it as a weak routing hint rather than
            # allowing it to outrank a caption/OCR that names the requested
            # artifact itself.  This prevents e.g. a hyperparameter table
            # from beating the nearby "external data" table merely because
            # both inherit the same surrounding figure inventory.
            contextual_searchable = " ".join(
                str(value or "")
                for value in (
                    metadata.get("artifact_context"),
                    metadata.get("page_text_excerpt"),
                )
            )
            primary_terms = set(_page_context_terms(primary_searchable))
            context_terms = set(_page_context_terms(contextual_searchable))
            primary_overlap = sum(term in primary_terms for term in terms)
            context_overlap = sum(term in context_terms for term in terms)
            overlap = (primary_overlap * 3) + min(context_overlap, 2)
            if overlap <= 0:
                continue
            searchable = " ".join(
                part for part in (primary_searchable, contextual_searchable) if part
            )
            artifact_id = str(item.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            item.update(
                {
                    "id": f"{artifact_type}:{artifact_id}",
                    "chunk_id": f"{artifact_type}:{artifact_id}",
                    "evidence_id": f"{artifact_type}:{artifact_id}",
                    "table_id": artifact_id if artifact_type == "table" else None,
                    "figure_id": artifact_id if artifact_type == "figure" else None,
                    "artifact_type": artifact_type,
                    "chunk_type": artifact_type,
                    "revision_id": item.get("revision_id") or item.get("active_revision_id"),
                    "content": searchable[:4_000],
                    "metadata": metadata,
                    "image_path": (
                        item.get("image_path")
                        or metadata.get("image_path")
                        or metadata.get("table_image_path")
                    ),
                    "retrieval_channels": ["sqlite_active_artifact_lexical"],
                    "raw_lexical_score": float(overlap),
                }
            )
            candidates.append(((-float(overlap), int(item.get("page_number") or 0), artifact_id), item))
        candidates.sort(key=lambda pair: pair[0])
        # Image-backed tables can exist in both canonical projections: a
        # ``document_tables`` row for structured citation and a mirrored
        # ``document_figures`` row for the pixels. They share the adapter's
        # external filename/hash, so returning both consumes two artifact
        # ranks and can hide the next genuinely distinct artifact. Collapse
        # only these explicit aliases and prefer the table identity when the
        # metadata says it is a table; ordinary table/figure rows without a
        # shared immutable asset key remain independent.
        canonical_aliases: dict[str, tuple[tuple[float, int, str], dict[str, Any]]] = {}
        deduped_candidates: list[tuple[tuple[float, int, str], dict[str, Any]]] = []
        for sort_key, item in candidates:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            alias = str(
                metadata.get("external_artifact_file")
                or metadata.get("image_sha256")
                or ""
            ).strip()
            if not alias:
                deduped_candidates.append((sort_key, item))
                continue
            alias_key = f"{item.get('document_id')}:{alias}"
            existing = canonical_aliases.get(alias_key)
            if existing is None:
                canonical_aliases[alias_key] = (sort_key, item)
                continue
            _existing_key, existing_item = existing
            existing_type = str(existing_item.get("artifact_type") or "").casefold()
            current_type = str(item.get("artifact_type") or "").casefold()
            existing_metadata = (
                existing_item.get("metadata")
                if isinstance(existing_item.get("metadata"), dict)
                else {}
            )
            existing_is_table = (
                existing_type == "table"
                or existing_metadata.get("asset_type") == "table"
            )
            current_is_table = current_type == "table" or metadata.get("asset_type") == "table"
            if current_is_table and not existing_is_table:
                canonical_aliases[alias_key] = (sort_key, item)
            elif current_is_table == existing_is_table and sort_key < _existing_key:
                canonical_aliases[alias_key] = (sort_key, item)
        deduped_candidates.extend(canonical_aliases.values())
        deduped_candidates.sort(key=lambda pair: pair[0])
        results: list[dict[str, Any]] = []
        for rank, (_sort_key, item) in enumerate(deduped_candidates[:limit], start=1):
            item["artifact_lexical_rank"] = rank
            item["score"] = _rrf(rank)
            item["rrf_score"] = item["score"]
            results.append(item)
        return results

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

    def search_page_coverage(
        self,
        query: str,
        *,
        page_limit: int = 24,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Return at most one lexical evidence unit per active document page.

        This is a bounded coverage projection, not a second semantic index.
        It deliberately reuses the revision-filtered FTS path so OCR/native
        page text can participate without exposing staged or retired chunks.
        The ordinary top-k path keeps its compact behaviour; callers opt into
        this projection only for document-wide aggregation questions.
        """

        limit = max(1, min(int(page_limit), 64))
        if document_ids is not None and not document_ids:
            return []
        # A page may contain several child chunks.  Search a deeper lexical
        # frontier, then collapse by page so repeated chunks cannot consume the
        # coverage budget.  The cap is independent of corpus size.
        lexical = self.search(
            query,
            top_k=max(limit * 8, 96),
            document_ids=document_ids,
        )
        document_ids_for_counts = list(
            dict.fromkeys(
                str(chunk.document_id)
                for chunk in lexical
                if str(chunk.document_id or "").strip()
            )
        )
        page_counts: dict[str, int] = {}
        if document_ids_for_counts and self.db_path is not None:
            placeholders = ",".join("?" for _ in document_ids_for_counts)
            with connect(self.db_path) as connection:
                rows = connection.execute(
                    f"SELECT id, page_count FROM documents WHERE id IN ({placeholders})",
                    tuple(document_ids_for_counts),
                ).fetchall()
            page_counts = {
                str(row["id"]): int(row["page_count"] or 0)
                for row in rows
                if int(row["page_count"] or 0) > 0
            }
        by_page: dict[tuple[str, int], dict] = {}
        for rank, chunk in enumerate(lexical, start=1):
            page = chunk.page_number
            document_id = str(chunk.document_id or "").strip()
            if not document_id or page is None or int(page) < 1:
                continue
            key = (document_id, int(page))
            existing = by_page.get(key)
            if existing is not None:
                # Keep the first (best FTS) chunk, but retain a short list of
                # matching chunks for audit/debugging without bloating prompt
                # content.
                matching = list(existing.get("coverage_chunk_ids") or [])
                chunk_id = str(chunk.chunk_id or "")
                if chunk_id and chunk_id not in matching and len(matching) < 4:
                    matching.append(chunk_id)
                existing["coverage_chunk_ids"] = matching
                continue
            content = str(chunk.content or "").strip()
            if not content:
                continue
            by_page[key] = {
                **chunk.__dict__,
                "chunk_id": f"page:{document_id}:{int(page)}",
                "evidence_id": f"page:{document_id}:{int(page)}",
                "artifact_type": "page",
                "chunk_type": "page",
                "page_label": chunk.page_label,
                "coverage_mode": True,
                "coverage_rank": rank,
                "coverage_chunk_ids": [str(chunk.chunk_id)] if chunk.chunk_id else [],
                "retrieval_channels": ["sqlite_fts5_page_coverage"],
                "score": float(-rank),
                "raw_fts_score": chunk.score,
                "document_page_count": page_counts.get(document_id),
            }
        ordered = sorted(
            by_page.values(),
            key=lambda item: (int(item.get("coverage_rank") or 0), str(item.get("document_id") or ""), int(item.get("page_number") or 0)),
        )
        return ordered[:limit]

    def document_page_counts(self, document_ids: list[str] | None = None) -> dict[str, int]:
        """Return active logical-document page counts for coverage planning.

        Page counts are metadata, not evidence.  They let an aggregation
        caller distinguish an exhaustive short-document frontier from a
        bounded top-k sample without widening ordinary retrieval.
        """

        if document_ids is not None and not document_ids:
            return {}
        with connect(self.db_path) as connection:
            if document_ids is None:
                rows = connection.execute(
                    "SELECT id, page_count FROM documents WHERE active_revision_id IS NOT NULL"
                ).fetchall()
            else:
                ids = list(dict.fromkeys(str(item) for item in document_ids if str(item).strip()))
                if not ids:
                    return {}
                placeholders = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"SELECT id, page_count FROM documents WHERE id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
        return {
            str(row["id"]): int(row["page_count"] or 0)
            for row in rows
            if int(row["page_count"] or 0) > 0
        }

    def list_page_coverage(
        self,
        *,
        document_ids: list[str],
        page_limit: int = 32,
    ) -> list[dict]:
        """Build an exhaustive active-revision page projection when bounded.

        This is deliberately separate from lexical ``search_page_coverage``:
        a document-wide count needs every known text page for short documents,
        while ordinary retrieval must remain query-ranked and compact.
        Image-only pages are supplied by the optional visual channel.
        """

        limit = max(1, min(int(page_limit), 64))
        if not document_ids:
            return []
        page_counts = self.document_page_counts(document_ids)
        pages: list[dict] = []
        for document_id in dict.fromkeys(str(item) for item in document_ids if str(item).strip()):
            chunks = self.list_document_chunks(document_id)
            seen_pages: set[int] = set()
            for chunk in chunks:
                raw_page = chunk.get("page_number")
                try:
                    page = int(raw_page)
                except (TypeError, ValueError):
                    continue
                if page < 1 or page in seen_pages:
                    continue
                content = str(chunk.get("content") or "").strip()
                if not content:
                    continue
                seen_pages.add(page)
                pages.append(
                    {
                        **chunk,
                        "chunk_id": f"page:{document_id}:{page}",
                        "evidence_id": f"page:{document_id}:{page}",
                        "artifact_type": "page",
                        "chunk_type": "page",
                        "coverage_mode": True,
                        "coverage_rank": len(pages) + 1,
                        "coverage_chunk_ids": [str(chunk.get("chunk_id") or "")],
                        "retrieval_channels": ["sqlite_active_page_projection"],
                        "score": 0.0,
                        "raw_fts_score": None,
                        "document_page_count": page_counts.get(document_id),
                    }
                )
                if len(seen_pages) >= limit:
                    break
        return pages

    async def search_visual_pages(
        self,
        query: str,
        *,
        provider: VisualRetrievalProvider,
        top_k: int = 24,
        collection_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Search a revision-scoped optional visual page provider.

        Visual retrieval is a derived navigation channel.  It is converted to
        the same canonical evidence shape as text retrieval and is always
        filtered by the current active revision before it reaches the agent.
        The provider owns its isolated index; this method never renders or
        mutates it during a query.
        """

        collection_document_ids = (
            self._collection_document_ids(collection_id) if collection_id else None
        )
        scoped_ids = list(document_ids) if document_ids is not None else collection_document_ids
        if scoped_ids is not None and not scoped_ids:
            return []
        if scoped_ids is not None and collection_document_ids is not None:
            allowed = set(collection_document_ids)
            scoped_ids = [item for item in scoped_ids if item in allowed]
        # An unscoped visual query must still be revision-scoped *before* the
        # provider ranks its candidates.  Passing ``None`` here lets stale
        # visual pages occupy top-k slots and filtering them afterwards cannot
        # recover active pages that were pushed out.  Resolve the complete
        # active corpus once, then use the same document/revision allow-list
        # for semantic search and explicit page-hint lookup.
        if scoped_ids is None:
            revisions = self._all_active_revision_ids()
            scoped_ids = list(revisions)
        else:
            revisions = self._active_revision_ids(scoped_ids)
            # A document can be deleted/retired between scope resolution and
            # retrieval.  Never pass an ID without an active revision to the
            # provider; an empty allow-list is a safe no-result outcome.
            scoped_ids = [document_id for document_id in scoped_ids if document_id in revisions]
        if not scoped_ids or not revisions:
            return []
        # Explicit page/slide/trang references are lexical constraints. A
        # semantic visual ranker may prefer a visually similar page elsewhere
        # in the document, so resolve requested page numbers from the provider
        # manifest before semantic search. This is both the correct scope
        # contract and important operationally: a ColVision MaxSim call over a
        # long PDF must not run when an exact active-revision page is already
        # requested. Providers that do not implement this optional capability
        # keep the normal ranked search unchanged.
        hinted_pages = requested_visual_page_numbers(query)
        get_pages = getattr(provider, "get_pages", None)
        hits: list[Any] = []
        if hinted_pages and callable(get_pages):
            # A user-facing printed page label can differ from the physical
            # PDF index. Resolve the hint per active document before querying
            # the visual provider; otherwise ``page 13`` can select physical
            # page 13 when the printed page 13 is physical page 14.
            hinted_pages_by_document = self._visual_page_hints_by_document(
                document_ids=scoped_ids,
                revisions=revisions,
                hinted_pages=hinted_pages,
            )
            hinted: list[Any] = []
            try:
                for document_id, physical_pages in hinted_pages_by_document.items():
                    if not physical_pages:
                        continue
                    hinted.extend(
                        await get_pages(
                            page_numbers=physical_pages,
                            document_ids=[document_id],
                            revision_ids=[revisions[document_id]],
                        )
                    )
            except Exception:
                hinted = []
            if hinted:
                # An explicit page/slide reference is a scope constraint, not
                # merely a ranking hint. Returning semantic hits from other
                # pages can leak adjacent visual evidence into the prompt and
                # make a correct page answer look contradictory. If an exact
                # page exists in the provider manifest, keep only those pages;
                # semantic search remains the fallback when the hint cannot
                # be resolved.
                hits = hinted
        if not hits:
            hits = await provider.search(
                query,
                document_ids=scoped_ids,
                revision_ids=list(revisions.values()) or None,
                top_k=max(1, min(int(top_k), 64)),
            )
        if not hits:
            return []
        if not revisions:
            revisions = self._active_revision_ids(
                list(dict.fromkeys(str(hit.document_id) for hit in hits if hit.document_id))
            )
        filenames: dict[str, str] = {}
        source_paths: dict[str, str] = {}
        page_counts: dict[str, int] = {}
        with connect(self.db_path) as connection:
            ids = list(dict.fromkeys(str(hit.document_id) for hit in hits if hit.document_id))
            if ids:
                placeholders = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"SELECT id, filename, source_path, active_revision_id, page_count FROM documents WHERE id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
                for row in rows:
                    document_id = str(row["id"])
                    expected_revision = str(row["active_revision_id"] or "")
                    if revisions.get(document_id) != expected_revision:
                        continue
                    filenames[document_id] = str(row["filename"] or "")
                    source_paths[document_id] = str(row["source_path"] or "")
                    count = int(row["page_count"] or 0)
                    if count > 0:
                        page_counts[document_id] = count
        evidence = visual_search_results_as_evidence(
            [
                hit
                for hit in hits
                if revisions.get(str(hit.document_id)) == str(hit.revision_id)
            ],
            filename_by_document=filenames,
            source_path_by_document=source_paths,
        )
        for item in evidence:
            item["coverage_mode"] = True
            item["visual_page"] = True
            document_page_count = page_counts.get(str(item.get("document_id") or ""))
            if document_page_count:
                item["document_page_count"] = document_page_count
        return evidence

    def _visual_page_hints_by_document(
        self,
        *,
        document_ids: list[str],
        revisions: dict[str, str],
        hinted_pages: list[int],
    ) -> dict[str, list[int]]:
        """Resolve printed page labels to physical pages for active revisions.

        ``chunks.metadata_json`` is the canonical persisted provenance source.
        Resolution is per document because a multi-document query can contain
        the same printed page number with different offsets. If no matching
        label exists, physical-page semantics remain the safe fallback.
        """

        ids = [
            str(document_id)
            for document_id in document_ids
            if str(document_id) in revisions
        ]
        if not ids or not hinted_pages:
            return {document_id: list(hinted_pages) for document_id in ids}
        revision_scope = " OR ".join(
            "(document_id = ? AND revision_id = ?)" for _ in ids
        )
        params: list[str] = [
            value
            for document_id in ids
            for value in (document_id, revisions[document_id])
        ]
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT document_id, page_number, metadata_json
                FROM chunks
                WHERE {revision_scope}
                  AND page_number IS NOT NULL
                ORDER BY document_id, page_number
                """,
                tuple(params),
            ).fetchall()
        labels_by_document: dict[str, dict[int, set[int]]] = {}
        for row in rows:
            document_id = str(row["document_id"])
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            label_values = page_label_values(metadata.get("page_label"))
            if not label_values:
                continue
            labels = labels_by_document.setdefault(document_id, {})
            for label_key in label_values:
                labels.setdefault(label_key, set()).add(int(row["page_number"]))
        resolved: dict[str, list[int]] = {}
        for document_id in ids:
            labels = labels_by_document.get(document_id, {})
            pages: list[int] = []
            for hinted_page in hinted_pages:
                # Prefer a printed-label match. Only fall back to the
                # physical number when this revision has no such label.
                candidates = labels.get(int(hinted_page)) or {int(hinted_page)}
                for page in sorted(candidates):
                    if page not in pages:
                        pages.append(page)
            resolved[document_id] = pages
        return resolved

    def explicit_page_hint_candidates(
        self,
        query: str,
        *,
        document_ids: list[str] | None,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        """Return active page evidence for user-supplied page/slide hints.

        An explicit page reference is a deterministic retrieval channel, not
        a gold-label shortcut: the page number comes from the current query
        and is resolved through the active revision's printed-label map.  It
        must therefore be visible in the same candidate schema as text/table/
        figure results, especially for image-only pages with no OCR chunk.
        """

        hinted_pages = requested_visual_page_numbers(query)
        ids = [
            str(document_id)
            for document_id in (document_ids or [])
            if str(document_id).strip()
        ]
        if not hinted_pages or not ids:
            return []
        ids = list(dict.fromkeys(ids))
        revisions = self._active_revision_ids(ids)
        if not revisions:
            return []
        resolved = self._visual_page_hints_by_document(
            document_ids=ids,
            revisions=revisions,
            hinted_pages=hinted_pages,
        )
        page_counts = self.document_page_counts(ids)
        chunks_by_page: dict[tuple[str, int], dict[str, Any]] = {}
        for document_id in ids:
            for chunk in self.list_document_chunks(document_id):
                try:
                    page = int(chunk.get("page_number"))
                except (TypeError, ValueError):
                    continue
                chunks_by_page.setdefault((document_id, page), chunk)

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for document_id in ids:
            for page in resolved.get(document_id, []):
                key = (document_id, int(page))
                if key in seen:
                    continue
                seen.add(key)
                base = dict(chunks_by_page.get(key) or {})
                if not base and int(page) > int(page_counts.get(document_id, 0) or 0):
                    continue
                # Empty page chunks are intentional: the page image renderer
                # can still provide canonical visual evidence downstream.
                candidates.append(
                    {
                        **base,
                        "document_id": document_id,
                        "revision_id": revisions[document_id],
                        "page_number": int(page),
                        "chunk_id": f"page:{document_id}:{int(page)}",
                        "evidence_id": f"page:{document_id}:{int(page)}",
                        "artifact_type": "page",
                        "chunk_type": "page",
                        "coverage_mode": True,
                        "explicit_page_hint": True,
                        "coverage_rank": len(candidates) + 1,
                        "coverage_chunk_ids": (
                            [str(base.get("chunk_id"))]
                            if base.get("chunk_id")
                            else []
                        ),
                        "retrieval_channels": ["sqlite_explicit_page_hint"],
                        "score": 0.0,
                        "raw_fts_score": None,
                        "document_page_count": page_counts.get(document_id),
                    }
                )
                if len(candidates) >= max(1, min(int(limit), 32)):
                    return candidates
        return candidates

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

    def expand_with_page_context(
        self,
        results: list[dict],
        *,
        query: str | None = None,
        max_chars: int = 1_200,
        max_chunks: int = 2,
    ) -> list[dict]:
        """Attach bounded native text from the same page as an artifact hit.

        A figure/table/page result is often the right navigation hit while its
        caption or visual summary is not enough to answer a question.  PDF
        parsers keep the surrounding native text as ordinary chunks, so merge
        a small, query-ranked page companion into the selected evidence
        instead of spending another retrieval slot on an unrelated chunk.
        The lookup is revision-scoped and never consults evaluation labels or
        answer text.  Results without an artifact/visual-page signal are
        returned byte-for-byte compatible with the previous path.
        """

        if not results or max_chars < 1 or max_chunks < 1:
            return list(results)

        targets: dict[tuple[str, int], list[int]] = {}
        excluded_chunk_ids: dict[tuple[str, int], set[str]] = {}
        detail_query = _query_requests_page_detail(query)
        query_terms = _page_context_terms(query)
        for index, result in enumerate(results):
            document_id = str(result.get("document_id") or "").strip()
            raw_page = result.get("page_number")
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                continue
            if not document_id or page_number < 1:
                continue
            channels = result.get("retrieval_channels") or []
            if isinstance(channels, str):
                channels = [channels]
            artifact_type = str(
                result.get("artifact_type") or result.get("chunk_type") or ""
            ).casefold()
            is_visual_page = bool(
                result.get("visual_page")
                or "visual_page" in {str(value).casefold() for value in channels}
                or artifact_type == "visual_page"
            )
            is_artifact = bool(
                result.get("figure_id")
                or result.get("table_id")
                or artifact_type in {"figure", "table", "image"}
            )
            # Exact-value/date/identifier questions can land on the right page
            # while the top text chunk contains only the page heading.  Reserve
            # a bounded same-page companion in that situation so a low-signal
            # child cannot hide the actual value in a sibling chunk.  This is
            # driven by query shape and lexical coverage, never by a paper or
            # benchmark locator.
            is_text_detail = False
            if detail_query and not is_visual_page and not is_artifact:
                content = str(result.get("content") or result.get("text") or "").casefold()
                term_hits = sum(1 for term in query_terms if term in content)
                has_numeric_signal = bool(re.search(r"(?<!\w)\d[\d/_.:-]{1,}(?!\w)", content))
                is_text_detail = bool(
                    query_terms
                    and (term_hits < max(2, (len(query_terms) + 1) // 2) or not has_numeric_signal)
                )
            if is_visual_page or is_artifact or is_text_detail:
                targets.setdefault((document_id, page_number), []).append(index)
                if is_text_detail:
                    excluded_chunk_ids.setdefault((document_id, page_number), set()).add(
                        str(result.get("chunk_id") or result.get("id") or "")
                    )
        if not targets:
            return list(results)

        companions: dict[tuple[str, int], list[dict]] = {}
        with connect(self.db_path) as connection:
            for document_id, page_number in targets:
                rows = connection.execute(
                    """
                    SELECT chunks.id, chunks.content, chunks.chunk_index,
                           chunks.heading_path_json, chunks.metadata_json
                    FROM chunks
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE chunks.document_id = ?
                      AND chunks.page_number = ?
                      AND chunks.chunk_type = 'text'
                      AND (
                          chunks.revision_id = documents.active_revision_id
                          OR (documents.active_revision_id IS NULL AND chunks.revision_id IS NULL)
                      )
                    ORDER BY chunks.chunk_index ASC
                    """,
                    (document_id, page_number),
                ).fetchall()
                ranked: list[tuple[tuple[int, int, int], dict]] = []
                for row in rows:
                    content = str(row["content"] or "").strip()
                    if not content:
                        continue
                    lowered = content.casefold()
                    term_hits = sum(1 for term in query_terms if term in lowered)
                    chunk_id = str(row["id"] or "")
                    if chunk_id in excluded_chunk_ids.get((document_id, page_number), set()):
                        continue
                    numeric_signal = int(
                        bool(re.search(r"(?<!\w)\d[\d/_.:-]{1,}(?!\w)", content))
                    )
                    try:
                        chunk_index = int(row["chunk_index"])
                    except (TypeError, ValueError):
                        chunk_index = 0
                    ranked.append(((-term_hits, -numeric_signal, chunk_index), dict(row)))
                ranked.sort(key=lambda item: item[0])
                selected_rows: list[dict] = []
                # For exact-value queries, keep one lexical companion and one
                # numeric-bearing companion when both exist.  This preserves
                # surrounding semantics without flooding the prompt with the
                # whole page.
                if detail_query:
                    lexical = next((item for _key, item in ranked if any(term in str(item["content"] or "").casefold() for term in query_terms)), None)
                    numeric_candidates = [
                        item
                        for _key, item in ranked
                        if re.search(
                            r"(?<!\w)\d[\d/_.:-]{1,}(?!\w)",
                            str(item["content"] or ""),
                        )
                    ]
                    numeric = max(
                        numeric_candidates,
                        key=lambda item: (
                            max(
                                (
                                    len(match.group(0))
                                    for match in re.finditer(
                                        r"(?<!\w)\d[\d/_.:-]{1,}(?!\w)",
                                        str(item["content"] or ""),
                                    )
                                ),
                                default=0,
                            ),
                            -int(item.get("chunk_index") or 0),
                        ),
                        default=None,
                    )
                    for item in (lexical, numeric):
                        if item is not None and item not in selected_rows and len(selected_rows) < max_chunks:
                            selected_rows.append(item)
                for _key, item in ranked:
                    if len(selected_rows) >= max_chunks:
                        break
                    if item not in selected_rows:
                        selected_rows.append(item)
                companions[(document_id, page_number)] = selected_rows

        if not companions:
            return list(results)
        enriched = list(results)
        for page_key, indexes in targets.items():
            rows = companions.get(page_key) or []
            if not rows:
                continue
            remaining = max_chars
            snippets: list[str] = []
            chunk_ids: list[str] = []
            for row in rows:
                content = " ".join(str(row["content"] or "").split())
                if not content or remaining <= 0:
                    continue
                snippet = content[:remaining].rstrip()
                if snippet:
                    snippets.append(snippet)
                    remaining -= len(snippet)
                    if row.get("id"):
                        chunk_ids.append(str(row["id"]))
            if not snippets:
                continue
            marker = "[same-page native text context]"
            page_context = f"{marker}\n" + "\n".join(snippets)
            for index in indexes:
                result = enriched[index]
                existing = str(
                    result.get("expanded_content")
                    or result.get("content")
                    or result.get("text")
                    or ""
                ).strip()
                if marker in existing:
                    continue
                enriched[index] = {
                    **result,
                    "expanded_content": (
                        f"{existing}\n\n{page_context}" if existing else page_context
                    ).strip(),
                    "page_context_chunk_ids": list(chunk_ids),
                }
        return enriched

    def expand_with_page_artifacts(
        self,
        results: list[dict],
        *,
        query: str | None = None,
        max_chars: int = 2_400,
        max_artifacts: int = 3,
    ) -> list[dict]:
        """Attach bounded canonical table/figure evidence to page hits.

        Visual retrieval returns a page identity, while the answer model often
        needs the table OCR/markdown or figure caption stored on that same
        active revision.  Projecting those artifacts onto the page result
        keeps page-image retrieval and structured evidence in one prompt
        without spending another retrieval slot or consulting benchmark
        annotations.  Every lookup is active-revision scoped.
        """

        if not results or max_chars < 1 or max_artifacts < 1:
            return list(results)

        targets: dict[tuple[str, int], list[int]] = {}
        for index, result in enumerate(results):
            document_id = str(result.get("document_id") or "").strip()
            if not has_physical_pages(result):
                # Markdown/DOCX artifacts all report page 1; a "same page"
                # projection would attach unrelated tables from the whole file.
                continue
            try:
                page_number = int(result.get("page_number"))
            except (TypeError, ValueError):
                continue
            channels = result.get("retrieval_channels") or []
            if isinstance(channels, str):
                channels = [channels]
            channel_set = {str(item).casefold() for item in channels}
            artifact_type = str(
                result.get("artifact_type") or result.get("chunk_type") or ""
            ).casefold()
            is_page_or_visual = bool(
                result.get("visual_page")
                or artifact_type in {"page", "visual_page"}
                or "visual_page" in channel_set
                or "colsmol" in channel_set
            )
            is_artifact = bool(
                result.get("table_id")
                or result.get("figure_id")
                or artifact_type in {"table", "figure", "image"}
            )
            if document_id and page_number > 0 and (is_page_or_visual or is_artifact):
                targets.setdefault((document_id, page_number), []).append(index)
        if not targets:
            return list(results)

        query_terms = _page_context_terms(query)
        artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
        with connect(self.db_path) as connection:
            for document_id, page_number in targets:
                table_rows = connection.execute(
                    """
                    SELECT document_tables.id AS artifact_id,
                           document_tables.page_number,
                           document_tables.revision_id,
                           document_tables.table_index,
                           document_tables.caption,
                           document_tables.markdown,
                           document_tables.metadata_json,
                           'table' AS artifact_type
                    FROM document_tables
                    JOIN documents ON documents.id = document_tables.document_id
                    WHERE document_tables.document_id = ?
                      AND document_tables.page_number = ?
                      AND (
                          document_tables.revision_id = documents.active_revision_id
                          OR (documents.active_revision_id IS NULL AND document_tables.revision_id IS NULL)
                      )
                    ORDER BY document_tables.table_index ASC
                    """,
                    (document_id, page_number),
                ).fetchall()
                figure_rows = connection.execute(
                    """
                    SELECT document_figures.id AS artifact_id,
                           document_figures.page_number,
                           document_figures.revision_id,
                           document_figures.figure_index,
                           document_figures.caption,
                           document_figures.image_path,
                           document_figures.visual_summary,
                           document_figures.metadata_json,
                           'figure' AS artifact_type
                    FROM document_figures
                    JOIN documents ON documents.id = document_figures.document_id
                    WHERE document_figures.document_id = ?
                      AND document_figures.page_number = ?
                      AND (
                          document_figures.revision_id = documents.active_revision_id
                          OR (documents.active_revision_id IS NULL AND document_figures.revision_id IS NULL)
                      )
                    ORDER BY document_figures.figure_index ASC
                    """,
                    (document_id, page_number),
                ).fetchall()
                ranked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
                for ordinal, row in enumerate([*table_rows, *figure_rows]):
                    item = dict(row)
                    try:
                        metadata = json.loads(item.pop("metadata_json") or "{}")
                    except (TypeError, ValueError):
                        metadata = {}
                    item["metadata"] = metadata
                    artifact_type = canonical_artifact_type(item, default="artifact")
                    caption = " ".join(str(item.get("caption") or "").split())
                    primary_structured = " ".join(
                        str(value or "")
                        for value in (
                            item.get("markdown"),
                            item.get("visual_summary"),
                            metadata.get("ocr_text"),
                        )
                    ).strip()
                    # Parsers retain a bounded surrounding-text projection for
                    # each artifact. Keep a query-relevant slice with the
                    # canonical artifact instead of discarding it: captions and
                    # OCR often identify a table/figure but nearby prose holds
                    # metric semantics or comparison scope. The slice is
                    # selected from query terms and bounded before prompting.
                    artifact_context = _artifact_context_snippet(
                        metadata.get("artifact_context"),
                        query_terms,
                        max_chars=min(1_000, max_chars),
                    )
                    structured = primary_structured
                    if artifact_context:
                        structured = (
                            f"{structured}\nSurrounding context: {artifact_context}"
                        ).strip()
                    if not caption and not structured:
                        continue
                    primary_haystack = f"{caption} {primary_structured}".casefold()
                    contextual_haystack = str(
                        metadata.get("artifact_context")
                        or metadata.get("page_text_excerpt")
                        or ""
                    ).casefold()
                    primary_hits = sum(
                        1 for term in query_terms if term.casefold() in primary_haystack
                    )
                    context_hits = sum(
                        1 for term in query_terms if term.casefold() in contextual_haystack
                    )
                    term_hits = (primary_hits * 3) + min(context_hits, 2)
                    # Prefer real OCR/markdown over an empty parser placeholder,
                    # then preserve artifact order for deterministic ties.
                    signal = int(bool(structured))
                    item["caption"] = caption
                    item["content"] = structured
                    item["artifact_type"] = artifact_type
                    item["chunk_type"] = artifact_type
                    if artifact_type == "table":
                        item["table_id"] = item.get("artifact_id")
                        item["figure_id"] = None
                    elif artifact_type == "figure":
                        item["figure_id"] = item.get("artifact_id")
                    ranked.append(((-term_hits, -signal, ordinal), item))
                ranked.sort(key=lambda pair: pair[0])
                artifacts_by_page[(document_id, page_number)] = [
                    item for _key, item in ranked[:max_artifacts]
                ]

        if not artifacts_by_page:
            return list(results)
        enriched = list(results)
        marker = "[same-page canonical artifact context]"
        # Keep canonical artifacts as first-class evidence rows.  If they are
        # left nested only inside a page string, context composition cannot
        # preserve table_id/figure_id or apply its modality reservations.  The
        # injected rows are derived solely from the active revision and retain
        # the originating page's filename/path for citation continuity.
        existing_artifact_ids = {
            str(result.get("table_id") or result.get("figure_id") or "").strip()
            for result in results
            if result.get("table_id") or result.get("figure_id")
        }
        injections: dict[int, list[dict[str, Any]]] = {}
        for page_key, indexes in targets.items():
            artifacts = artifacts_by_page.get(page_key) or []
            if not artifacts or not indexes:
                continue
            anchor = min(indexes)
            template = results[anchor]
            injected: list[dict[str, Any]] = []
            for artifact in artifacts:
                artifact_id = str(artifact.get("artifact_id") or "").strip()
                if not artifact_id or artifact_id in existing_artifact_ids:
                    continue
                metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
                artifact_type = canonical_artifact_type(
                    {**artifact, "metadata": metadata}, default="artifact"
                ).casefold()
                row = {
                    # Only document/page provenance transfers from the anchor.
                    # Expanded content, facet labels, indices and parent IDs
                    # belong to that anchor, not its same-page sibling.
                    **{key: template[key] for key in (
                        "document_id", "filename", "source_path", "page_number", "score",
                    ) if key in template},
                    "revision_id": artifact.get("revision_id"),
                    **({"table_index": artifact.get("table_index")} if artifact_type == "table"
                       else {"figure_index": artifact.get("figure_index")}),
                    "chunk_id": f"{artifact_type}:{artifact_id}",
                    "id": f"{artifact_type}:{artifact_id}",
                    "artifact_type": artifact_type,
                    "chunk_type": artifact_type,
                    "table_id": artifact_id if artifact_type == "table" else None,
                    "figure_id": artifact_id if artifact_type == "figure" else None,
                    "caption": artifact.get("caption"),
                    "content": artifact.get("content") or artifact.get("caption") or "",
                    "image_path": artifact.get("image_path") or metadata.get("image_path"),
                    "retrieval_channels": ["same_page_artifact"],
                    "visual_page": False,
                    "coverage_mode": False,
                    "same_page_as": str(template.get("chunk_id") or template.get("id") or ""),
                    "metadata": metadata,
                }
                injected.append(row)
                existing_artifact_ids.add(artifact_id)
            if injected:
                injections[anchor] = injected

        original_positions = {index: index for index in range(len(results))}
        if injections:
            ordered_with_artifacts: list[dict[str, Any]] = []
            for index, result in enumerate(enriched):
                injected = injections.get(index) or []
                # If the page hit is already a canonical artifact, preserve
                # that selected artifact's rank and append same-page siblings
                # after it.  Prepending a lower-signal sibling (for example a
                # hyperparameter table) used to hide the actually retrieved
                # external-data table in the context composer.
                result_is_artifact = bool(
                    result.get("table_id")
                    or result.get("figure_id")
                    or str(result.get("artifact_type") or result.get("chunk_type") or "").casefold()
                    in {"table", "figure", "image"}
                )
                if result_is_artifact:
                    original_positions[index] = len(ordered_with_artifacts)
                    ordered_with_artifacts.append(result)
                    ordered_with_artifacts.extend(injected)
                else:
                    ordered_with_artifacts.extend(injected)
                    original_positions[index] = len(ordered_with_artifacts)
                    ordered_with_artifacts.append(result)
            enriched = ordered_with_artifacts

        for page_key, indexes in targets.items():
            artifacts = artifacts_by_page.get(page_key) or []
            if not artifacts:
                continue
            remaining = max_chars
            snippets: list[str] = []
            artifact_ids: list[str] = []
            for artifact in artifacts:
                if remaining <= 0:
                    break
                label = str(artifact.get("artifact_type") or "artifact").title()
                caption = str(artifact.get("caption") or "").strip()
                content = " ".join(str(artifact.get("content") or "").split())
                text = f"{label}: {caption}" if caption else f"{label}:"
                if content:
                    text += f"\n{content}"
                snippet = text[:remaining].rstrip()
                if snippet:
                    snippets.append(snippet)
                    remaining -= len(snippet)
                if artifact.get("artifact_id"):
                    artifact_ids.append(str(artifact["artifact_id"]))
            if not snippets:
                continue
            page_context = f"{marker}\n" + "\n".join(snippets)
            for index in indexes:
                # Siblings follow artifact anchors but precede page anchors.
                # Use the actual output position, not a universal insertion
                # offset, or page context gets written onto another artifact.
                enriched_index = original_positions[index]
                result = enriched[enriched_index]
                existing = str(
                    result.get("expanded_content")
                    or result.get("content")
                    or result.get("text")
                    or ""
                ).strip()
                if marker in existing:
                    continue
                enriched[enriched_index] = {
                    **result,
                    "expanded_content": (
                        f"{existing}\n\n{page_context}" if existing else page_context
                    ).strip(),
                    "page_artifact_ids": list(artifact_ids),
                }
        return enriched

    def expand_with_referenced_artifacts(
        self,
        results: list[dict],
        *,
        query: str | None = None,
        max_artifacts: int = 8,
    ) -> list[dict]:
        """Follow bounded text/parent -> canonical artifact references.

        A useful paragraph can point to a table/figure on another page even
        when the user never asks for an image. Printed caption identity must
        resolve uniquely in the same active document revision. Unknown labels
        and stale anchors fail closed; no index+1, filename or qrel mapping.
        """
        if max_artifacts < 1 or not results:
            return list(results)
        existing_ids = {
            (str(item.get("document_id") or ""), str(item.get("table_id") or item.get("figure_id") or ""))
            for item in results if item.get("table_id") or item.get("figure_id")
        }
        catalogs: dict[str, tuple[dict, dict]] = {}
        output: list[dict] = []
        injected = 0
        query_terms = _reference_query_terms(query)
        for anchor in results:
            output.append(anchor)
            if injected >= max_artifacts or anchor.get("table_id") or anchor.get("figure_id"):
                continue
            kind = canonical_artifact_type(anchor)
            if kind not in {"text", "page"}:
                continue
            # A generated list of captions is not authored reference prose.
            anchor_metadata = anchor.get("metadata") or {}
            headings = anchor.get("heading_path") or anchor_metadata.get("heading_path") or []
            if isinstance(headings, str):
                headings = [headings]
            labels = [str(anchor.get("section_title") or anchor_metadata.get("section_title") or ""),
                      *(str(label) for label in headings)]
            if any(re.sub(r"\s+", " ", label).strip().casefold()
                   in {"figures and tables", "bibliography", "references"} for label in labels):
                continue
            text = str(anchor.get("expanded_content") or anchor.get("content") or anchor.get("text") or "")
            text = text.split("[same-page canonical artifact context]", 1)[0]
            references = extract_artifact_references(text[:16_000])
            document_id = str(anchor.get("document_id") or "")
            if not references or not document_id:
                continue
            if document_id not in catalogs:
                document = self.get_document(document_id) or {}
                by_label: dict[tuple[str, str], list[dict]] = {}
                if document:
                    for artifact_kind, artifacts in (
                        ("table", self.list_document_tables(document_id)),
                        ("figure", self.list_document_figures(document_id)),
                    ):
                        for artifact in artifacts:
                            printed = extract_artifact_references(str(artifact.get("caption") or ""))
                            if not printed or printed[0].kind != artifact_kind:
                                continue
                            key = (printed[0].kind, printed[0].number)
                            by_label.setdefault(key, []).append({**artifact, "artifact_type": artifact_kind})
                catalogs[document_id] = document, by_label
            document, by_label = catalogs[document_id]
            if not document:
                continue
            active = document.get("active_revision_id")
            revision = anchor.get("revision_id")
            if revision != active:
                # Older chunk shapes may omit the physical revision field.
                # Verify their actual row instead of assuming it is active.
                if revision is not None or not active:
                    continue
                with connect(self.db_path) as connection:
                    row = connection.execute(
                        "SELECT revision_id FROM chunks WHERE id=? AND document_id=?",
                        (anchor.get("chunk_id") or anchor.get("id"), document_id),
                    ).fetchone()
                if row is None or row["revision_id"] != active:
                    continue
            anchor_injected = 0

            def relevance(reference) -> int:
                candidates = by_label.get((reference.kind, reference.number), [])
                if len(candidates) != 1:
                    return 0
                candidate = candidates[0]
                text = f"{candidate.get('caption') or ''}\n{str(candidate.get('markdown') or '')[:2_000]}"
                return int(_reference_is_relevant(query, query_terms, kind=reference.kind, text=text))

            references = sorted(references, key=relevance, reverse=True)
            for reference in references:
                if anchor_injected >= _REFERENCES_PER_ANCHOR:
                    break
                candidates = by_label.get((reference.kind, reference.number), [])
                if len(candidates) != 1:
                    continue
                artifact = candidates[0]
                artifact_id = str(artifact.get("id") or "")
                identity = document_id, artifact_id
                if not artifact_id or identity in existing_ids:
                    continue
                metadata = dict(artifact.get("metadata") or {})
                if metadata.get("quality_status") == "rejected" or metadata.get("is_content") is False:
                    continue
                content = str(artifact.get("markdown") or artifact.get("visual_summary")
                              or metadata.get("ocr_text") or artifact.get("caption") or "")

                row = {
                    "document_id": document_id,
                    "revision_id": artifact.get("revision_id"),
                    "filename": document.get("filename"),
                    "source_path": document.get("source_path"),
                    "page_number": artifact.get("page_number"),
                    "chunk_id": f"{reference.kind}:{artifact_id}",
                    "artifact_type": reference.kind,
                    "chunk_type": reference.kind,
                    f"{reference.kind}_id": artifact_id,
                    f"{reference.kind}_index": artifact.get(f"{reference.kind}_index"),
                    "caption": artifact.get("caption"),
                    "content": content,
                    "image_path": artifact.get("image_path") or metadata.get("image_path") or metadata.get("table_image_path"),
                    "metadata": metadata,
                    "retrieval_channels": ["artifact_reference"],
                    "artifact_reference_from": str(anchor.get("chunk_id") or anchor.get("id") or ""),
                    "artifact_reference_label": f"{reference.kind} {reference.number}",
                }
                output.append(row)
                existing_ids.add(identity)
                injected += 1
                anchor_injected += 1
                if injected >= max_artifacts:
                    break
        return output

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
                SELECT chunks.id AS chunk_id, chunks.document_id, chunks.source_path,
                       chunks.filename, chunks.content, chunks.chunk_index,
                       chunks.page_number, chunks.heading_path_json, chunks.token_count,
                       chunks.char_count, chunks.metadata_json
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.document_id = ?
                  AND (
                      chunks.revision_id = documents.active_revision_id
                      OR (documents.active_revision_id IS NULL AND chunks.revision_id IS NULL)
                  )
                  AND chunks.chunk_type = 'text'
                  AND chunks.chunk_index BETWEEN ? AND ?
                  AND chunks.chunk_index != ?
                ORDER BY chunks.chunk_index ASC
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
        candidate_document_union: bool = False,
        coverage_mode: bool = False,
        coverage_page_limit: int = 24,
        evidence_obligations: bool = True,
        evidence_obligation_plan: EvidenceObligationPlan | None = None,
    ) -> dict:
        started_at = time.perf_counter()
        timings_ms: dict[str, float] = {}
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
            "sqlite_active_artifact_lexical",
        ]
        if scope_requested and not search_document_ids:
            timings_ms["total"] = _elapsed_ms(started_at)
            return {
                "query": query,
                "collection_id": collection_id,
                "forced_document_ids": forced_document_ids or [],
                "selected_document_ids": [],
                "retrieval_channels": retrieval_channels,
                "document_card_results": [],
                "candidate_results": [],
                "channel_candidates": {"fts": [], "vector": [], "artifact": []},
                "timings_ms": timings_ms,
                "results": [],
            }

        semantic_query, semantic_facets = retrieval_query_expansion(
            query,
            focused_document_count=len(search_document_ids or []),
        )
        embedding_started_at = time.perf_counter()
        query_embedding = await embeddings.embed_query(semantic_query)
        timings_ms["embedding"] = _elapsed_ms(embedding_started_at)
        effective_coverage_limit = max(1, min(int(coverage_page_limit), 64))
        # Coverage mode needs a deeper first-stage frontier.  It is only used
        # for explicitly classified document-wide aggregation requests; the
        # ordinary production top-k remains unchanged.
        retrieval_top_k = max(top_k, effective_coverage_limit) if coverage_mode else top_k
        card_top_k = max(retrieval_top_k * 4, 20)
        max_selected_documents = max(top_k * 2, 8)
        scoped_active_revisions = self._active_revision_ids(search_document_ids)
        card_started_at = time.perf_counter()
        card_results = await retrieval_store.search_document_cards(
            query_embedding=query_embedding,
            filters=RetrievalFilter(
                document_ids=search_document_ids,
                active_revisions=scoped_active_revisions or None,
            ),
            top_k=card_top_k,
        )
        timings_ms["card_lookup"] = _elapsed_ms(card_started_at)
        selected_document_ids = search_document_ids
        fts_started_at = time.perf_counter()
        fts_limit = max(retrieval_top_k * 4, 20)
        fts_queries = build_fts_queries(query)
        if semantic_query != query:
            for variant in build_fts_queries(semantic_query):
                if variant not in fts_queries:
                    fts_queries.append(variant)
        # Keep lexical work bounded.  The first views preserve the exact raw
        # query; later views are only recovery for long identifiers/tails.
        fts_queries = fts_queries[:6]
        fts_chunks = _merge_fts_chunks(
            [
                self.search(
                    fts_query,
                    top_k=fts_limit,
                    document_ids=search_document_ids,
                )
                for fts_query in fts_queries
            ],
            limit=fts_limit,
        )
        timings_ms["fts"] = _elapsed_ms(fts_started_at)
        artifact_lexical_started_at = time.perf_counter()
        # A scoped query already has a bounded document frontier.  Recover
        # canonical table/figure rows only inside that frontier; an unscoped
        # query waits for card/FTS document routing so an artifact scan cannot
        # become a corpus-wide hot path.
        # Canonical artifacts normally wait until card/FTS routing has chosen a
        # document scope.  An explicit artifact locator (``Figure 12``,
        # ``Table 3``, etc.) is a stronger routing signal than a document-card
        # similarity score, however, and must be allowed to discover its owner
        # before vector search.  This remains bounded by the canonical
        # artifact tables and is activated only for an actual label/number,
        # not for every generic visual question.
        # Artifact lexical routing must see the same bounded semantic facet
        # expansion as the vector/FTS branches.  Otherwise a natural query
        # such as "how important is visual grounding" can retrieve a useful
        # table/figure in the vector candidate pool while the canonical
        # artifact channel only sees the literal word ``visual``.  Combining
        # the raw and expanded forms is still local and deterministic; it
        # does not read answers or evaluation annotations.
        artifact_query = " ".join(
            value for value in (query, semantic_query) if str(value or "").strip()
        )
        artifact_candidates = (
            self.search_active_artifacts_lexical(
                artifact_query,
                top_k=max(retrieval_top_k * 2, 12),
                document_ids=search_document_ids,
            )
            if search_document_ids is not None
            else (
                self.search_active_artifacts_lexical(
                    artifact_query,
                    top_k=max(retrieval_top_k * 2, 12),
                    document_ids=None,
                )
                if _query_has_explicit_artifact_locator(query)
                else []
            )
        )
        timings_ms["artifact_lexical"] = _elapsed_ms(artifact_lexical_started_at)
        if selected_document_ids is None:
            card_document_ids = _document_ids_from_vector_results(
                _filter_card_results_for_query(query, card_results),
                max_documents=max_selected_documents,
            )
            if candidate_document_union:
                fts_document_ids = _document_ids_from_fts_chunks(
                    fts_chunks,
                    max_documents=4,
                )
                selected_document_ids = list(dict.fromkeys(
                    [
                        *card_document_ids[:8],
                        *fts_document_ids,
                        *_document_ids_from_artifact_results(artifact_candidates, max_documents=4),
                    ]
                ))[: max(12, max_selected_documents)]
            else:
                # A numbered/labelled artifact is an exact document-routing
                # signal.  Preserve its owner even on the production control
                # arm; otherwise a semantically similar card can hide the
                # document that actually contains ``Figure N``/``Table N``.
                artifact_document_ids = (
                    _document_ids_from_artifact_results(
                        artifact_candidates,
                        max_documents=max_selected_documents,
                    )
                    if _query_has_explicit_artifact_locator(query)
                    else []
                )
                selected_document_ids = list(
                    dict.fromkeys([*artifact_document_ids, *card_document_ids])
                )
        # For generic table/figure requests, now that the normal routing stage
        # has selected a bounded document frontier, recover canonical artifact
        # rows inside that frontier.  This avoids a corpus-wide scan on ordinary
        # queries while still making ``show the tables in this paper`` work
        # when the document was identified by cards/FTS rather than its
        # caption.
        if (
            selected_document_ids
            and _query_requests_artifact_recovery(query)
            and not artifact_candidates
        ):
            artifact_candidates = self.search_active_artifacts_lexical(
                artifact_query,
                top_k=max(retrieval_top_k * 2, 12),
                document_ids=list(selected_document_ids),
            )
        if scope_requested and not selected_document_ids:
            timings_ms["total"] = _elapsed_ms(started_at)
            return {
                "query": query,
                "semantic_query": semantic_query,
                "semantic_facets": semantic_facets,
                "collection_id": collection_id,
                "forced_document_ids": forced_document_ids or [],
                "selected_document_ids": [],
                "retrieval_channels": retrieval_channels,
                "document_card_results": [],
                "candidate_results": [],
                "channel_candidates": {"fts": [], "vector": [], "artifact": []},
                "timings_ms": timings_ms,
                "results": [],
            }
        vector_started_at = time.perf_counter()
        active_revisions = self._active_revision_ids(selected_document_ids)
        revision_filter = RetrievalFilter(
            document_ids=selected_document_ids or None,
            active_revisions=active_revisions or None,
        )
        vector_text_chunks = await retrieval_store.search_text_chunks(
            query_embedding=query_embedding,
            filters=revision_filter,
            top_k=max(retrieval_top_k * 4, 20),
        )
        vector_table_chunks = await retrieval_store.search_table_chunks(
            query_embedding=query_embedding,
            filters=revision_filter,
            top_k=max(retrieval_top_k * (4 if visual_boost else 2), 12 if visual_boost else 10),
        )
        vector_figure_chunks = await retrieval_store.search_figure_chunks(
            query_embedding=query_embedding,
            filters=revision_filter,
            top_k=max(retrieval_top_k * (4 if visual_boost else 2), 12 if visual_boost else 10),
        )
        vector_text_chunks = _filter_active_revision_results(vector_text_chunks, active_revisions)
        vector_table_chunks = _filter_active_revision_results(vector_table_chunks, active_revisions)
        vector_figure_chunks = _filter_active_revision_results(vector_figure_chunks, active_revisions)
        timings_ms["vector"] = _elapsed_ms(vector_started_at)
        vector_chunks = sorted(
            [*vector_text_chunks, *vector_table_chunks, *vector_figure_chunks],
            key=lambda result: result.score,
        )
        # Keep modality-local vector ranks for fusion.  Text, table and figure
        # projections are separate Lance tables with different population
        # sizes; concatenating them and assigning one global rank systematically
        # penalises sparse artifact channels (a table rank 1 could become rank
        # 21 simply because text returned 20 rows).  The flattened list remains
        # available for diagnostics, while RRF receives the three independent
        # channel rankings below.
        vector_ranked_groups = (
            vector_text_chunks,
            vector_table_chunks,
            vector_figure_chunks,
        )
        channel_candidates = {
            "fts": [dict(chunk.__dict__) for chunk in fts_chunks],
            "vector": [_vector_payload(chunk) for chunk in vector_chunks],
            "artifact": [dict(item) for item in artifact_candidates],
        }
        # Keep a deeper first-stage frontier even when the expensive reranker
        # is disabled.  The caller's context selector still emits at most the
        # requested top_k, but it can now recover a second facet/document that
        # would otherwise be lost to an early top-k cutoff.
        candidate_k = (
            max(retrieval_top_k * 4, 64)
            if coverage_mode
            else (max(top_k * 4, 24) if rerank else max(top_k * 3, 30))
        )
        fusion_started_at = time.perf_counter()
        merged = _merge_hybrid_chunks(
            query,
            vector_chunks,
            fts_chunks,
            candidate_k,
            visual_boost=visual_boost,
            artifact_candidates=artifact_candidates,
            vector_ranked_groups=vector_ranked_groups,
        )
        timings_ms["fusion"] = _elapsed_ms(fusion_started_at)
        explicit_page_candidates = self.explicit_page_hint_candidates(
            query,
            document_ids=list(selected_document_ids or []),
            limit=candidate_k,
        )
        if explicit_page_candidates:
            explicit_ids = {
                str(item.get("evidence_id") or item.get("chunk_id") or "")
                for item in explicit_page_candidates
            }
            merged = [
                *explicit_page_candidates,
                *[
                    item
                    for item in merged
                    if str(item.get("evidence_id") or item.get("chunk_id") or "")
                    not in explicit_ids
                ],
            ][:candidate_k]
        obligation_plan = None
        if evidence_obligations and selected_document_ids:
            obligation_plan = (
                evidence_obligation_plan or plan_evidence_obligations(query)
            )
        obligation_branches: list[dict[str, Any]] = []
        if obligation_plan is not None and obligation_plan.enabled:
            obligation_started_at = time.perf_counter()
            # Multi-evidence questions need a deeper *candidate* frontier than
            # the final answer context.  On long documents the complementary
            # operand can be semantically relevant but rank below the first
            # dozen hits (for example, a denominator in a distant balance
            # sheet).  Keep the public contract bounded at 30 and let the
            # obligation-aware selector reduce it back to ``top_k``.  This is
            # query-structure driven and never consults gold pages/answers.
            branch_top_k = min(max(top_k * 3, 30), 30)
            obligation_branches = await asyncio.gather(
                *[
                    self.search_hybrid(
                        query=obligation_search_query(obligation),
                        top_k=branch_top_k,
                        retrieval_store=retrieval_store,
                        embeddings=embeddings,
                        collection_id=collection_id,
                        document_ids=list(selected_document_ids or []),
                        rerank=False,
                        rerank_mode=rerank_mode,
                        cross_encoder_model_path=cross_encoder_model_path,
                        rerank_max_candidates=rerank_max_candidates,
                        visual_boost=visual_boost,
                        candidate_document_union=True,
                        coverage_mode=False,
                        coverage_page_limit=coverage_page_limit,
                        evidence_obligations=False,
                        evidence_obligation_plan=None,
                    )
                    for obligation in obligation_plan.obligations
                ]
            )
            ranked_for_fusion: list[tuple[str, dict[str, Any]]] = [
                ("original", item) for item in merged
            ]
            tagged_candidates: list[dict[str, Any]] = []
            for obligation, branch in zip(
                obligation_plan.obligations,
                obligation_branches,
                strict=True,
            ):
                tagged = tag_obligation_results(
                    branch.get("results") or [],
                    obligation,
                )
                tagged_candidates.extend(tagged)
                ranked_for_fusion.extend(
                    (obligation.obligation_id, item) for item in tagged
                )
            fused = fuse_ranked_results(
                ranked_for_fusion,
                top_k=candidate_k,
                rrf_k=60,
            )
            merged = reserve_obligation_coverage(
                [*fused, *tagged_candidates],
                plan=obligation_plan,
                limit=candidate_k,
            )
            timings_ms["obligation_retrieval"] = _elapsed_ms(
                obligation_started_at
            )
        else:
            timings_ms["obligation_retrieval"] = 0.0
        candidate_results = list(merged)
        if coverage_mode:
            merged = _collapse_page_coverage_results(
                merged,
                max_pages=effective_coverage_limit,
            )
        if rerank and merged:
            rerank_started_at = time.perf_counter()
            merged = await rerank_candidates(
                query=query,
                results=merged,
                embeddings=embeddings,
                top_k=retrieval_top_k if coverage_mode else top_k,
                mode=rerank_mode,
                cross_encoder_model_path=cross_encoder_model_path,
                max_candidates=rerank_max_candidates,
            )
            merged = reserve_obligation_coverage(
                [*merged, *candidate_results],
                plan=obligation_plan,
                limit=retrieval_top_k if coverage_mode else top_k,
            )
            # Reranking scores the complete candidate frontier.  Re-apply the
            # generic structured-evidence projection only after rerank so a
            # list/table question cannot lose its best artifact at the final
            # cutoff, while still allowing the production reranker to order
            # the candidate pool first.
            merged = _promote_structured_evidence(
                [*merged, *candidate_results],
                query=query,
                top_k=retrieval_top_k if coverage_mode else top_k,
            )
            timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
        else:
            timings_ms["rerank"] = 0.0
            # ``_merge_hybrid_chunks`` may receive a deep candidate frontier
            # (often <=30 rows).  Apply structured promotion again at the
            # visible cutoff; promoting only inside the first stage is a no-op
            # whenever the frontier is already smaller than that depth.
            merged = _promote_structured_evidence(
                candidate_results,
                query=query,
                top_k=retrieval_top_k if coverage_mode else top_k,
            )
        # The obligation selector operates on the candidate frontier, while
        # the final projection is bounded by ``top_k``.  Re-assert the
        # complementary-operand invariant after every final-stage operation;
        # this is deliberately query-structure driven and prevents a
        # reranker (or a future projection) from silently dropping the only
        # evidence for one side of a ratio/comparison.
        if obligation_plan is not None and obligation_plan.enabled:
            merged = reserve_obligation_coverage(
                [*merged, *candidate_results],
                plan=obligation_plan,
                limit=retrieval_top_k if coverage_mode else top_k,
            )
        timings_ms["total"] = _elapsed_ms(started_at)
        return {
            "query": query,
            "semantic_query": semantic_query,
            "semantic_facets": semantic_facets,
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
            "candidate_document_union": bool(candidate_document_union),
            "explicit_page_candidates": [
                {
                    "document_id": item.get("document_id"),
                    "revision_id": item.get("revision_id"),
                    "page_number": item.get("page_number"),
                    "evidence_id": item.get("evidence_id"),
                }
                for item in explicit_page_candidates
            ],
            "coverage_mode": bool(coverage_mode),
            "coverage_page_limit": effective_coverage_limit if coverage_mode else None,
            "coverage_page_count": sum(
                1
                for item in merged
                if item.get("page_number") is not None and item.get("document_id")
            ) if coverage_mode else 0,
            "candidate_results": candidate_results,
            "evidence_obligation_plan": (
                {
                    "kind": obligation_plan.kind,
                    "reason": obligation_plan.reason,
                    "queries": [
                        obligation.query
                        for obligation in obligation_plan.obligations
                    ],
                    "obligation_ids": [
                        obligation.obligation_id
                        for obligation in obligation_plan.obligations
                    ],
                    "branch_count": len(obligation_branches),
                    "branch_top_candidates": [
                        {
                            "obligation_id": obligation.obligation_id,
                            "candidates": [
                                {
                                    "evidence_id": (
                                        item.get("chunk_id")
                                        or item.get("table_id")
                                        or item.get("figure_id")
                                    ),
                                    "artifact_type": item.get("artifact_type")
                                    or item.get("chunk_type"),
                                    "page_number": item.get("page_number"),
                                    "score": item.get("score"),
                                }
                                for item in list(branch.get("results") or [])[:8]
                            ],
                        }
                        for obligation, branch in zip(
                            obligation_plan.obligations,
                            obligation_branches,
                            strict=True,
                        )
                    ],
                    "selected_candidates": [
                        {
                            "evidence_id": (
                                item.get("chunk_id")
                                or item.get("table_id")
                                or item.get("figure_id")
                            ),
                            "page_number": item.get("page_number"),
                            "evidence_obligation_ids": item.get(
                                "evidence_obligation_ids", []
                            ),
                        }
                        for item in merged[: min(len(merged), 12)]
                    ],
                }
                if obligation_plan is not None and obligation_plan.enabled
                else None
            ),
            "channel_candidates": channel_candidates,
            "timings_ms": timings_ms,
            "results": merged,
        }

    async def search_hybrid_queries(
        self,
        *,
        queries: list[tuple[str, str]],
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
        rrf_k: int = 60,
        coverage_mode: bool = False,
        coverage_page_limit: int = 24,
    ) -> dict:
        """Run bounded original/rewrite retrieval and fuse local candidates.

        This method is deliberately separate from ``search_hybrid`` so the
        production path remains byte-compatible while the feature flag is
        evaluated.  Reranking, when enabled, happens once after fusion rather
        than once per query variant.
        """

        started_at = time.perf_counter()
        unique_queries: list[tuple[str, str]] = []
        seen_queries: set[str] = set()
        for origin, value in queries:
            normalized = " ".join(str(value or "").split())
            key = normalized.casefold()
            if not normalized or key in seen_queries:
                continue
            seen_queries.add(key)
            unique_queries.append((str(origin), normalized))
        if not unique_queries:
            return {
                "query": "",
                "queries": [],
                "query_origins": [],
                "results": [],
                "retrieval_channels": [],
                "selected_document_ids": [],
                "document_card_results": [],
                "candidate_results": [],
                "channel_candidates": {"fts": [], "vector": []},
                "timings_ms": {"total": _elapsed_ms(started_at)},
                "dual_query_rrf": True,
            }

        # ``top_k`` is the externally visible candidate depth.  Preserve it
        # across the single-query and RRF paths: each branch may consult a
        # broader lexical/vector pool internally, but neither arm gets to
        # retain a deeper ranked candidate list before the evaluator's
        # top-30 cutoff.  This keeps the A0/A1/B comparison an ablation,
        # rather than an accidental depth advantage for RRF.
        candidate_top_k = top_k
        branch_started_at = time.perf_counter()
        branch_results = await asyncio.gather(
            *[
                self.search_hybrid(
                    query=query,
                    top_k=candidate_top_k,
                    retrieval_store=retrieval_store,
                    embeddings=embeddings,
                    collection_id=collection_id,
                    document_ids=document_ids,
                    rerank=False,
                    rerank_mode=rerank_mode,
                    cross_encoder_model_path=cross_encoder_model_path,
                    rerank_max_candidates=rerank_max_candidates,
                    visual_boost=visual_boost,
                    candidate_document_union=True,
                    coverage_mode=coverage_mode,
                    coverage_page_limit=coverage_page_limit,
                    evidence_obligations=False,
                )
                for _origin, query in unique_queries
            ]
        )
        branch_timings = {
            origin: dict(payload.get("timings_ms") or {})
            for (origin, _query), payload in zip(unique_queries, branch_results)
        }
        ranked_results = [
            (origin, payload)
            for (origin, _query), branch in zip(unique_queries, branch_results)
            for payload in branch.get("results") or []
        ]
        fusion_started_at = time.perf_counter()
        candidate_pool = fuse_ranked_results(
            ranked_results,
            top_k=candidate_top_k,
            rrf_k=rrf_k,
        )
        fusion_ms = _elapsed_ms(fusion_started_at)
        merged = candidate_pool
        if coverage_mode:
            merged = _collapse_page_coverage_results(
                merged,
                max_pages=max(1, min(int(coverage_page_limit), 64)),
            )
        if rerank and merged:
            rerank_started_at = time.perf_counter()
            merged = await rerank_candidates(
                query=unique_queries[0][1],
                results=merged,
                embeddings=embeddings,
                top_k=(max(top_k, min(int(coverage_page_limit), 64)) if coverage_mode else top_k),
                mode=rerank_mode,
                cross_encoder_model_path=cross_encoder_model_path,
                max_candidates=rerank_max_candidates,
            )
            rerank_ms = _elapsed_ms(rerank_started_at)
        else:
            rerank_ms = 0.0
            merged = merged[: (max(top_k, min(int(coverage_page_limit), 64)) if coverage_mode else top_k)]
        selected_document_ids = list(dict.fromkeys(
            document_id
            for branch in branch_results
            for document_id in branch.get("selected_document_ids") or []
        ))
        retrieval_channels = list(dict.fromkeys(
            channel
            for branch in branch_results
            for channel in branch.get("retrieval_channels") or []
        ))
        card_results = [
            card
            for branch in branch_results
            for card in branch.get("document_card_results") or []
        ]
        bundle = EvidenceBundle.from_results(
            query=unique_queries[0][1],
            results=candidate_pool,
            selected=merged,
            query_origins=tuple(origin for origin, _query in unique_queries),
            diagnostics={
                "dual_query_rrf": True,
                "rrf_k": rrf_k,
                "candidate_count": len(ranked_results),
                "branch_timings_ms": branch_timings,
            },
        )
        return {
            "query": unique_queries[0][1],
            "queries": [query for _origin, query in unique_queries],
            "query_origins": [origin for origin, _query in unique_queries],
            "collection_id": collection_id,
            "forced_document_ids": list(document_ids or []),
            "selected_document_ids": selected_document_ids,
            "retrieval_channels": retrieval_channels,
            "document_card_results": card_results,
            "candidate_document_union": True,
            "dual_query_rrf": True,
            "candidate_results": candidate_pool,
            "timings_ms": {
                "branches": _elapsed_ms(branch_started_at),
                "fusion": fusion_ms,
                "rerank": rerank_ms,
                "total": _elapsed_ms(started_at),
            },
            "branch_results": [
                {
                    "origin": origin,
                    "results": list(payload.get("candidate_results") or payload.get("results") or []),
                    "selected_document_ids": list(payload.get("selected_document_ids") or []),
                    "channel_candidates": dict(payload.get("channel_candidates") or {}),
                    "timings_ms": dict(payload.get("timings_ms") or {}),
                }
                for (origin, _query), payload in zip(unique_queries, branch_results)
            ],
            "evidence_bundle": bundle.to_dict(),
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
        active_revisions = self._active_revision_ids(search_document_ids)
        filters = RetrievalFilter(
            document_ids=search_document_ids or None,
            active_revisions=active_revisions or None,
        )
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
        figure_chunks = _filter_active_revision_results(figure_chunks, active_revisions)
        table_chunks = _filter_active_revision_results(table_chunks, active_revisions)
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
        active_revisions = self._active_revision_ids(collection_document_ids)
        results = await retrieval_store.search_figure_chunks(
            query_embedding=query_embedding,
            filters=RetrievalFilter(
                document_ids=collection_document_ids or None,
                active_revisions=active_revisions or None,
            ),
            top_k=top_k,
        )
        results = _filter_active_revision_results(
            results,
            active_revisions,
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

    def _active_revision_ids(self, document_ids: list[str] | None) -> dict[str, str]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"SELECT id, active_revision_id FROM documents WHERE id IN ({placeholders})",
                tuple(document_ids),
            ).fetchall()
        return {
            str(row["id"]): str(row["active_revision_id"])
            for row in rows
            if row["active_revision_id"]
        }

    def _all_active_revision_ids(self) -> dict[str, str]:
        """Return the active revision allow-list for an unscoped query.

        This is intentionally separate from ``_active_revision_ids`` whose
        ``None``/empty input historically means “no requested scope”.  Visual
        providers need an explicit corpus allow-list so stale revisions are
        excluded before top-k ranking rather than discarded afterwards.
        """

        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, active_revision_id
                FROM documents
                WHERE active_revision_id IS NOT NULL
                ORDER BY indexed_at ASC, id ASC
                """
            ).fetchall()
        return {
            str(row["id"]): str(row["active_revision_id"])
            for row in rows
            if row["active_revision_id"]
        }

    def get_document(self, document_id: str) -> dict | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, folder_id, source_path, filename, file_type, content_hash,
                       modified_at, indexed_at, chunk_count, index_status, parser_name,
                       parser_version, page_count, error_message, metadata_json,
                       active_revision_id
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        try:
            metadata = json.loads(payload.get("metadata_json") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        # Keep the historical metadata_json field for compatibility while
        # exposing the typed readiness snapshot directly to API consumers.
        payload["readiness"] = metadata.get("document_readiness")
        return payload

    def documents_requiring_visual_channel(
        self,
        document_ids: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        """Return active documents whose canonical content is visual-only.

        This is a readiness signal, not a query classifier.  A PDF/embedded
        image document with no active text chunks cannot be answered through
        OCR/FTS/vector text alone; when the optional visual provider is
        enabled, its page/image channel must be considered even if the user
        asks a fact-shaped question.  Results are scoped to the active
        revision so staged or retired projections never activate the signal.
        """

        ids = list(dict.fromkeys(str(item).strip() for item in (document_ids or ()) if str(item).strip()))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT d.id, d.file_type, d.page_count, d.figure_count,
                       d.table_count, d.metadata_json,
                       COUNT(c.id) AS text_chunks
                FROM documents AS d
                LEFT JOIN chunks AS c
                  ON c.document_id = d.id
                 AND c.revision_id = d.active_revision_id
                 AND c.chunk_type = 'text'
                WHERE d.id IN ({placeholders})
                  AND d.active_revision_id IS NOT NULL
                  AND COALESCE(d.revision_status, 'active') = 'active'
                GROUP BY d.id
                """,
                tuple(ids),
            ).fetchall()
        required: list[str] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{{}}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            readiness = metadata.get("document_readiness")
            readiness_state = str((readiness or {}).get("state") or "").casefold()
            page_states = {
                str(item.get("state") or "").casefold()
                for item in ((readiness or {}).get("pages") or [])
                if isinstance(item, dict)
            }
            visual_only = readiness_state == "visual_only" or (
                "visual_only" in page_states
                and not ({"native_text_ready", "ocr_text_ready"} & page_states)
            )
            # Legacy rows may lack the readiness snapshot.  A zero-text PDF
            # with canonical page/artifact evidence is still unambiguously
            # visual-only; do not classify an empty/failed document as ready.
            if not visual_only:
                visual_only = (
                    str(row["file_type"] or "").casefold() == "pdf"
                    and int(row["text_chunks"] or 0) == 0
                    and int(row["page_count"] or 0) > 0
                    and (int(row["figure_count"] or 0) + int(row["table_count"] or 0)) > 0
                )
            if visual_only:
                required.append(str(row["id"]))
        order = {value: index for index, value in enumerate(ids)}
        return sorted(required, key=lambda value: order.get(value, len(order)))

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
                body_text=self._document_name_sample(document_id),
                phrase_document_count=self._phrase_document_count,
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
                SELECT chunks.id, chunks.document_id, chunks.chunk_index, chunks.content,
                       chunks.source_path, chunks.filename, chunks.chunk_type,
                       chunks.heading_path_json, chunks.token_count, chunks.char_count,
                       chunks.order_index, chunks.parent_chunk_id, chunks.page_number,
                       chunks.metadata_json
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.document_id = ?
                  AND (
                      chunks.revision_id = documents.active_revision_id
                      OR (documents.active_revision_id IS NULL AND chunks.revision_id IS NULL)
                  )
                ORDER BY chunks.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        chunks: list[dict] = []
        for row in rows:
            item = dict(row)
            item["heading_path"] = json.loads(item.pop("heading_path_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            item["page_label"] = item["metadata"].get("page_label")
            chunks.append(item)
        return chunks

    def list_document_tables(self, document_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_tables.id, document_tables.document_id,
                       document_tables.file_id, document_tables.table_index,
                       document_tables.page_number, document_tables.caption,
                       document_tables.markdown, document_tables.row_count,
                       document_tables.column_count, document_tables.extraction_method,
                       document_tables.bbox_json, document_tables.created_at,
                       document_tables.metadata_json, document_tables.revision_id
                FROM document_tables
                JOIN documents ON documents.id = document_tables.document_id
                WHERE document_tables.document_id = ?
                  AND (
                      document_tables.revision_id = documents.active_revision_id
                      OR (documents.active_revision_id IS NULL AND document_tables.revision_id IS NULL)
                  )
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
                SELECT document_figures.id, document_figures.document_id,
                       document_figures.file_id, document_figures.figure_index,
                       document_figures.page_number, document_figures.caption,
                       document_figures.image_path, document_figures.visual_summary,
                       document_figures.extraction_method, document_figures.bbox_json,
                       document_figures.created_at, document_figures.metadata_json,
                       document_figures.revision_id
                FROM document_figures
                JOIN documents ON documents.id = document_figures.document_id
                WHERE document_figures.id = ?
                  AND (
                      document_figures.revision_id = documents.active_revision_id
                      OR (documents.active_revision_id IS NULL AND document_figures.revision_id IS NULL)
                  )
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
                SELECT document_figures.id, document_figures.document_id,
                       document_figures.file_id, document_figures.figure_index,
                       document_figures.page_number, document_figures.caption,
                       document_figures.image_path, document_figures.visual_summary,
                       document_figures.extraction_method, document_figures.bbox_json,
                       document_figures.created_at, document_figures.metadata_json,
                       document_figures.revision_id
                FROM document_figures
                JOIN documents ON documents.id = document_figures.document_id
                WHERE document_figures.document_id = ?
                  AND (
                      document_figures.revision_id = documents.active_revision_id
                      OR (documents.active_revision_id IS NULL AND document_figures.revision_id IS NULL)
                  )
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

        vocabulary = _catalog_vocabulary(rows)
        tokens = document_match_tokens(entities=entities, query=query, vocabulary=vocabulary)
        if not tokens:
            return []
        scored: list[tuple[float, str]] = []
        for row in rows:
            score = _score_document_for_tokens(
                tokens, row["filename"], row["source_path"] or "",
                aliases=_document_keyword_keys(row["keywords_json"]),
            )
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
                body_text=self._document_name_sample(document_id),
                phrase_document_count=self._phrase_document_count,
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
        if not any(mention.document_id for mention in mentions):
            inferred = _infer_catalog_mention(
                query_words, rows, aliases_by_document, alias_owners,
                word_in_library=self._word_in_library,
            )
            if inferred is not None:
                mentions = [inferred, *mentions]
        return CatalogMentionResolution(mentions=tuple(mentions))

    def library_titles(self, collection_id: str | None = None) -> list[tuple[str, str]]:
        """(document_id, readable title) of the library, newest first.

        Lets the router recognise a document named by title, translation,
        misspelling or description.
        """
        with connect(self.db_path) as connection:
            if collection_id:
                rows = connection.execute(
                    """
                    SELECT d.id, d.title, c.title_guess, d.filename
                    FROM documents d
                    JOIN collection_documents cd ON cd.document_id = d.id
                    LEFT JOIN document_cards c ON c.document_id = d.id
                    WHERE cd.collection_id = ? ORDER BY d.indexed_at DESC
                    """,
                    (collection_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT d.id, d.title, c.title_guess, d.filename
                    FROM documents d LEFT JOIN document_cards c ON c.document_id = d.id
                    ORDER BY d.indexed_at DESC
                    """
                ).fetchall()
        titles: list[tuple[str, str]] = []
        for row in rows:
            title = str(row["title"] or row["title_guess"] or "").strip()
            filename = str(row["filename"] or "").strip()
            label = f"{title} ({filename})" if title and filename else title or filename
            if label:
                titles.append((str(row["id"]), label))
        return titles

    def _document_name_sample(self, document_id: str) -> str | None:
        """First pages of the document's active text (cached per revision),
        used to tell names from topical keywords."""
        cache = self.__dict__.setdefault("_name_sample_cache", {})
        try:
            with connect(self.db_path) as connection:
                revision = connection.execute(
                    "SELECT active_revision_id FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                key = (document_id, revision[0] if revision else None)
                if key in cache:
                    return cache[key]
                rows = connection.execute(
                    """
                    SELECT content FROM chunks
                    WHERE document_id = ? AND COALESCE(revision_id, '') = COALESCE(?, '')
                    ORDER BY chunk_index LIMIT 40
                    """,
                    (document_id, key[1]),
                ).fetchall()
        except Exception:  # noqa: BLE001 - no sample keeps previous behaviour
            return None
        sample = "\n".join(str(row[0] or "") for row in rows) or None
        cache[key] = sample
        return sample

    def _phrase_document_count(self, phrase: str) -> int:
        """Number of documents whose text uses ``phrase`` (separators free)."""
        parts = re.findall(r"[A-Za-z0-9]+", str(phrase or "").lower())
        if not parts:
            return 0
        cache = self.__dict__.setdefault("_phrase_count_cache", {})
        key = " ".join(parts)
        if key in cache:
            return cache[key]
        pattern = re.compile(r"(?<![a-z0-9])" + r"[\s\-_]*".join(map(re.escape, parts)) + r"(?![a-z0-9])")
        documents: set[str] = set()
        try:
            with connect(self.db_path) as connection:
                rows = connection.execute(
                    "SELECT document_id, content FROM chunks WHERE lower(content) LIKE ?",
                    ("%" + "%".join(parts) + "%",),
                ).fetchall()
        except Exception:  # noqa: BLE001 - unknown counts as shared (fail closed)
            cache[key] = 2
            return 2
        for row in rows:
            if row[0] not in documents and pattern.search(str(row[1] or "").lower()):
                documents.add(str(row[0]))
        cache[key] = len(documents)
        return cache[key]

    def _word_in_library(self, word: str) -> bool:
        """Whether ``word`` occurs in indexed text, i.e. it is a real term of
        this library ("whisper"), not a typo of a paper name ("asprie")."""
        if not word or len(word) < 3:
            return True
        cache = self.__dict__.setdefault("_library_word_cache", {})
        if word in cache:
            return cache[word]
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(word.lower())}(?![a-z0-9])")
        try:
            with connect(self.db_path) as connection:
                rows = connection.execute(
                    "SELECT content FROM chunks WHERE lower(content) LIKE ? LIMIT 200",
                    (f"%{word.lower()}%",),
                ).fetchall()
        except Exception:  # noqa: BLE001 - unknown is treated as a real word
            cache[word] = True
            return True
        # Whole words only: "wav2smal" occurs inside "wav2small" but is a typo.
        cache[word] = any(pattern.search(" ".join(_search_words(str(row[0] or "")))) for row in rows)
        return cache[word]

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
        vocabulary = _catalog_vocabulary(rows)
        for entity in target_entities:
            tokens = document_match_tokens(entities=[entity], query=entity, vocabulary=vocabulary)
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


# Ordinary words that must never be "corrected" into a paper name.
_FUZZY_STOPWORDS = frozenset(
    "about above after again against also what which where when there their these those "
    "paper papers model models method methods table figure show give tell explain data "
    "dataset result results score scores value values using based other first second third "
    "nhung cua cho bang hinh mo hinh bai bao tai lieu nghien cuu ket qua".split()
)
_DOCUMENT_MARKER_WORDS = frozenset(
    "paper papers article study report document bai bao tai lieu nghien cuu cong trinh".split()
)


def _edit_distance(first: str, second: str, limit: int) -> int:
    """Optimal string alignment distance, early-exiting above ``limit``."""
    if abs(len(first) - len(second)) > limit:
        return limit + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, start=1):
        current = [i] + [0] * len(second)
        for j, char_b in enumerate(second, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (
                previous_previous is not None and i > 1 and j > 1
                and char_a == second[j - 2] and first[i - 2] == char_b
            ):
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        if min(current) > limit:
            return limit + 1
        previous_previous, previous = previous, current
    return previous[-1]


def _fuzzy_limit(length: int) -> int:
    return 0 if length < 4 else 1 if length <= 6 else 2


def _title_head(title: str) -> list[str]:
    """"ASPIRE: Attentive ..." is named ASPIRE; "Crab: Multi Layer ..." Crab."""
    head = str(title or "").split(":", 1)[0] if ":" in str(title or "") else ""
    words = _search_words(head)
    return words if 1 <= len(words) <= 3 else []


def _infer_catalog_mention(
    query_words: list[str],
    rows: list[Any],
    aliases_by_document: dict[str, dict[tuple[str, ...], tuple[int, str]]],
    alias_owners: dict[tuple[str, ...], set[str]],
    *,
    word_in_library: Callable[[str], bool] | None = None,
) -> CatalogDocumentMention | None:
    """Resolve a misspelled name or a description to one catalog document.

    Runs only when no alias matched exactly. Typos: every word of a
    single-owner alias (filename, title head, distinctive keyword) must be
    within a small edit distance of a query window, and exactly one document
    may reach the best distance. Descriptions: with a document word in the
    query ("paper", "bài"), query words found in titles are weighted by how
    rare they are across titles; the best document must clearly lead.
    """
    # --- misspelled names -------------------------------------------------
    candidates: dict[str, tuple[str, ...]] = {}  # alias text -> owners
    for document_id, aliases in aliases_by_document.items():
        for alias, (_strength, source) in aliases.items():
            owners = alias_owners.get(alias, set())
            if len(owners) != 1 or len(alias) > 4:
                continue
            if source in {"title", "title_phrase"}:
                continue
            if sum(len(word) for word in alias) < 4:
                continue
            candidates[" ".join(alias)] = (document_id,)
    for row in rows:
        head = _title_head(str(row["title"] or row["title_guess"] or ""))
        if head and sum(len(word) for word in head) >= 4:
            key = " ".join(head)
            owners = {str(row["id"])}
            if key in candidates and candidates[key] != (str(row["id"]),):
                candidates.pop(key)
                continue
            candidates.setdefault(key, tuple(owners))
    best: dict[str, tuple[int, int, int, str]] = {}  # doc -> (distance, start, end, surface)
    windows: list[tuple[str, tuple[str, ...], list[str], int, int]] = []
    for alias_text, owners in candidates.items():
        alias_words = alias_text.split()
        # A one-word name may be typed split in two ("Tide Nett" for TideNet).
        widths = {len(alias_words)} | ({2} if len(alias_words) == 1 and len(alias_text) >= 6 else set())
        for width in widths:
            for start in range(0, len(query_words) - width + 1):
                windows.append((alias_text, owners, query_words[start:start + width], start, width))
    for alias_text, owners, window, start, width in windows:
        alias_words = alias_text.split()
        joined_alias = "".join(alias_words)
        if any(word in _FUZZY_STOPWORDS for word in window):
            continue
        # A word the library itself uses ("whisper", "fusion") is a real
        # term, never a misspelled paper name.
        if word_in_library is not None and any(
            word not in alias_words and word_in_library(word) for word in window
        ):
            continue
        total = 0 if len(window) == len(alias_words) else -1
        for query_word, alias_word in (zip(window, alias_words) if total == 0 else ()):
            limit = _fuzzy_limit(len(alias_word))
            distance = _edit_distance(query_word, alias_word, limit)
            if distance > limit:
                total = -1
                break
            total += distance
        if total < 0:
            # A name typed as one word ("keysparse") or split ("ks t").
            joined = "".join(window)
            limit = _fuzzy_limit(len(joined_alias))
            distance = _edit_distance(joined, joined_alias, limit) if limit else limit + 1
            if distance > limit:
                continue
            total = distance
        if total == 0:
            continue  # exact matches were handled by the alias pass
        document_id = owners[0]
        surface = " ".join(window)
        current = best.get(document_id)
        if current is None or total < current[0]:
            best[document_id] = (total, start, start + width, surface)
    if best:
        minimum = min(item[0] for item in best.values())
        winners = [doc for doc, item in best.items() if item[0] == minimum]
        if len(winners) == 1:
            distance, start, end, surface = best[winners[0]]
            return CatalogDocumentMention(
                surface=surface, start_word=start, end_word=end, document_id=winners[0],
                candidate_ids=(winners[0],), alias_source="fuzzy_name", strength=0,
            )

    # --- descriptions -------------------------------------------------------
    if not _DOCUMENT_MARKER_WORDS.intersection(query_words):
        return None
    title_words: dict[str, set[str]] = {}
    for row in rows:
        words = {
            word for word in _search_words(str(row["title"] or row["title_guess"] or ""))
            if len(word) >= 3 and word not in _FUZZY_STOPWORDS
        }
        title_words[str(row["id"])] = words
    if not title_words:
        return None
    frequency: dict[str, int] = {}
    for words in title_words.values():
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
    total_documents = len(title_words)
    query_set = {word for word in query_words if len(word) >= 3 and word not in _FUZZY_STOPWORDS}
    scored: list[tuple[float, int, str]] = []
    for document_id, words in title_words.items():
        shared = query_set & words
        if not shared:
            continue
        score = sum(math.log(1 + total_documents / frequency[word]) for word in shared)
        scored.append((score, len(shared), document_id))
    if not scored:
        return None
    scored.sort(reverse=True)
    top_score, top_shared, top_document = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if top_shared >= 2 and top_score >= 1.5 * runner_up and top_score >= 2.0:
        return CatalogDocumentMention(
            surface=" ".join(sorted(query_set & title_words[top_document])),
            start_word=0, end_word=0, document_id=top_document,
            candidate_ids=(top_document,), alias_source="title_description", strength=0,
        )
    return None


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
    body_text: str | None = None,
    phrase_document_count: Callable[[str], int] | None = None,
) -> dict[tuple[str, ...], tuple[int, str]]:
    """Build conservative, separator-insensitive aliases for one catalog row.

    ``body_text`` (a sample of the document's own text) lets a multi-word
    keyword count as a name only when the paper itself writes it like one.
    """

    canonical_aliases: list[tuple[str, int, bool, str]] = []
    stem = Path(filename).stem.strip()
    if stem:
        canonical_aliases.append((stem, 3, True, "filename"))
        # Public paper corpora often normalize an arXiv identifier into the
        # filename (for example ``spiqa-1907_04152``) while users mention it as
        # ``1907.04152``.  Treat the identifier as a catalog-derived alias,
        # never as a paper-specific mapping.  Corpus-wide ownership checks
        # below still make duplicate IDs fail closed instead of guessing.
        for match in re.finditer(
            r"(?<!\d)(\d{4})[._-](\d{4,5})(?!\d)",
            stem,
        ):
            canonical_aliases.append(
                (
                    f"{match.group(1)} {match.group(2)}",
                    3,
                    True,
                    "filename_identifier",
                )
            )
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
            words = _search_words(keyword)
            compound = len(words) >= 2
            identity = " ".join((stem, title, title_guess))
            if not keyword:
                continue
            if not compound or body_text is None:
                accepted = _is_identifier_like_keyword(keyword, identity_context=identity)
            else:
                initials = words[0] + "".join(word[0] for word in words[1:])
                written_as_name = _written_as_name(keyword, body_text, title or title_guess)
                accepted = (
                    # KS-Transformer -> KST.pdf: initials reproduce the filename.
                    (len(initials) >= 3 and initials in set(_search_words(stem)))
                    # FM-MOE: the paper writes it as a name and nobody else uses it.
                    or (
                        written_as_name
                        and (phrase_document_count is None or phrase_document_count(keyword) <= 1)
                    )
                    # Pitch-fusion: in the title and written as a name; a title
                    # word written in lower case (semi-supervised) is a technique.
                    or (written_as_name and "".join(words) in _compact_token(identity))
                )
            if accepted:
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


def _written_as_name(keyword: str, body_text: str, title: str = "") -> bool:
    """Whether the paper itself writes ``keyword`` as a proper name.

    Only mid-sentence occurrences count (a lower-case word right before it on
    the same line), so titles, headings and keyword lists are ignored. A name
    is capitalised there ("introduces Pitch-fusion", "employ FM-MOE"); a
    technique is not ("a novel semi-supervised method", "named key-sparse").
    """
    parts = [re.escape(part) for part in re.findall(r"[A-Za-z0-9]+", keyword)]
    if len(parts) < 2:
        return True
    pattern = re.compile(r"[ \t\-_]*".join(parts), re.IGNORECASE)
    mid_sentence = 0
    capitalised = 0
    for match in pattern.finditer(body_text):
        before = body_text[max(0, match.start() - 30):match.start()]
        if not re.search(r"[a-z]{2,}[ \t]+$", before):
            continue
        mid_sentence += 1
        if match.group(0)[:1].isupper():
            capitalised += 1
    return mid_sentence > 0 and capitalised * 2 >= mid_sentence


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

@dataclass(frozen=True)
class CatalogVocabulary:
    """Name vocabulary derived from the indexed catalog, never a paper list.

    ``document_frequency`` counts in how many document names a word occurs;
    a word unique to one name ("mamba", "whiser") identifies that document,
    while shared words ("fusion") only help inside a compound phrase.
    """

    document_frequency: dict[str, int]
    acronyms: frozenset[str]
    # Hyphenated content keywords ("pitch-fusion", "ks-transformer") name a
    # model/method inside a document; their parts only count together.
    aliases: frozenset[str] = frozenset()
    alias_words: frozenset[str] = frozenset()
    # In how many documents' content keywords a word occurs: topic words
    # ("arousal", "multimodal") recur across papers, names ("mamba") do not.
    keyword_frequency: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_names(
        cls, names: list[str], keywords: list[list[str]] | None = None,
    ) -> "CatalogVocabulary":
        frequency: dict[str, int] = {}
        acronyms: set[str] = set()
        for name in names:
            words = set(_name_words(name))
            for word in words:
                frequency[word] = frequency.get(word, 0) + 1
            stem = re.sub(r"\.[a-z0-9]{2,5}$", "", str(name or ""), flags=re.I)
            # An all-caps multi-word name ("FROM_SINGLE_..._FUSION_MODEL") is
            # shouting, not a list of acronyms.
            if stem.upper() == stem and len(re.findall(r"[A-Za-z]{2,}", stem)) >= 3:
                continue
            acronyms.update(
                _compact_token(token)
                for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,15}", stem)
                if sum(char.isupper() for char in token) >= 2
            )
        aliases: set[str] = set()
        alias_words: set[str] = set()
        keyword_frequency: dict[str, int] = {}
        for document_keywords in keywords or []:
            for word in {
                part
                for keyword in document_keywords
                for part in re.split(r"[^a-z0-9]+", str(keyword or "").lower())
                if part
            }:
                keyword_frequency[word] = keyword_frequency.get(word, 0) + 1
            for keyword in document_keywords:
                text = str(keyword or "").strip().lower()
                if "-" not in text or not 4 <= len(text) <= 40:
                    continue
                aliases.add(_compact_token(text))
                alias_words.update(part for part in re.split(r"[^a-z0-9]+", text) if part)
        return cls(
            document_frequency=frequency,
            acronyms=frozenset(a for a in acronyms if len(a) >= 3),
            aliases=frozenset(aliases),
            alias_words=frozenset(alias_words),
            keyword_frequency=keyword_frequency,
        )

    def is_distinctive(self, word: str) -> bool:
        return (
            self.document_frequency.get(word, 0) == 1
            and self.keyword_frequency.get(word, 0) <= 1
            and word not in _GENERIC_ENTITY_TOKENS
        )

    def knows(self, word: str) -> bool:
        return word in self.document_frequency or word in self.acronyms or word in self.aliases

    def joins(self, word: str, other: str) -> bool:
        """Whether two adjacent query words form one catalog name."""
        if word + other in self.aliases:
            return True
        return word in self.document_frequency and other in self.document_frequency


# Function words never identify a document ("…Gateway cho AI…", "…for SER").
_NAME_STOPWORDS = frozenset(
    "the and for from with into using via that this are was not you your our its "
    "cho của và với các những được trong về như này một khi thì gì là có không "
    "cua voi cac nhung duoc trong nhu nay mot khi thi khong".split()
)


def _name_words(name: str) -> list[str]:
    stem = re.sub(r"\.[a-z0-9]{2,5}$", "", str(name or "").lower())
    return [
        word
        for word in re.split(r"[^0-9a-zà-ỹđ]+", stem)
        if len(word) >= 3 and not word.isdigit() and word not in _NAME_STOPWORDS
    ]


# Fillers inside one name ("mamba based fusion"); conjunctions such as
# "và"/"and"/"với" join two different entities and never form a compound.
_MATCH_FILLER_WORDS = frozenset({"based", "the", "of"})
_CONJUNCTION_WORDS = frozenset({"and", "or", "vs", "versus", "with", "và", "với", "hoặc"})


def _query_alias_tokens(query: str, vocabulary: CatalogVocabulary | None) -> list[str]:
    """Catalog-relative aliases a free-text query names.

    * a word that occurs in exactly one catalog name ("whiser", "mamba");
    * a compound of two catalog words, optionally joined by a filler
      ("mamba fusion", "mamba based fusion") — shared words such as "fusion"
      only count inside such a compound;
    * initials of a hyphenated name ("KS-Transformer" → "kst") or an
      upper-case acronym ("MSF-SER") present in the catalog.
    Without a vocabulary only acronym-like tokens are returned.
    """
    text = str(query or "")
    tokens: list[str] = []
    word_text = text.lower()
    if vocabulary is not None:
        # A known hyphenated name ("MSF-SER", "pitch-fusion") is one word;
        # its parts must not be matched separately.
        word_text = re.sub(
            r"[0-9a-zà-ỹđ]+(?:-[0-9a-zà-ỹđ]+)+",
            lambda match: (
                _compact_token(match.group(0))
                if _compact_token(match.group(0)) in vocabulary.aliases | vocabulary.acronyms
                else match.group(0)
            ),
            word_text,
        )
    words = [word for word in re.split(r"[^0-9a-zà-ỹđ]+", word_text) if word]
    if vocabulary is not None:
        for index, word in enumerate(words):
            if len(word) >= 3 and (
                vocabulary.is_distinctive(word) or word in vocabulary.aliases or word in vocabulary.acronyms
            ):
                tokens.append(word)
            for gap in (1, 2):
                if index + gap >= len(words):
                    continue
                other = words[index + gap]
                if gap == 2 and words[index + 1] not in _MATCH_FILLER_WORDS:
                    continue
                if gap == 1 and (word in _CONJUNCTION_WORDS or other in _CONJUNCTION_WORDS):
                    continue
                if (
                    len(word) >= 2 and len(other) >= 3
                    and word not in _MATCH_FILLER_WORDS and other not in _MATCH_FILLER_WORDS
                    and vocabulary.joins(word, other)
                ):
                    tokens.append(word + other)
    for term in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+", text):
        parts = term.split("-")
        initials = _compact_token(parts[0] if parts[0].isupper() and len(parts[0]) <= 3 else parts[0][0])
        initials += "".join(part[0].lower() for part in parts[1:] if part)
        if vocabulary is None or initials in vocabulary.acronyms or vocabulary.knows(initials):
            tokens.append(initials)
        compact = _compact_token(term)
        if vocabulary is None or compact in vocabulary.acronyms or compact in vocabulary.aliases:
            tokens.append(compact)
    # Stand-alone acronyms only; parts of "MSF-SER" were handled above.
    for term in re.findall(r"(?<![\w-])[A-Za-z]*[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?![\w-])", text):
        compact = _compact_token(term)
        if len(compact) >= 3 and compact not in _GENERIC_ENTITY_TOKENS and (
            vocabulary is None or vocabulary.knows(compact)
        ):
            tokens.append(compact)
    return tokens


def document_match_tokens(
    *,
    entities: list[str] | None,
    query: str | None = None,
    vocabulary: CatalogVocabulary | None = None,
) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        normalized = _compact_token(token)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tokens.append(normalized)

    for token in _query_alias_tokens(str(query or ""), vocabulary):
        add(token)

    for entity in entities or []:
        text = str(entity).strip()
        if not text:
            continue
        aliases = _query_alias_tokens(text, vocabulary)
        for token in aliases:
            add(token)
        compact = _compact_token(text)
        if compact in _GENERIC_ENTITY_TOKENS:
            continue
        distinctive = vocabulary is not None and vocabulary.is_distinctive(compact)
        if distinctive or len(compact) >= 6 or any(char.isupper() for char in text):
            add(compact)

    return tokens


_DEFAULT_VOCABULARY_CACHE: dict[str, tuple[float, CatalogVocabulary]] = {}


def default_catalog_vocabulary(db_path: str | Path | None = None) -> CatalogVocabulary | None:
    """Vocabulary of the configured catalog, cached until the DB changes.

    Used by query rewriting, which has no RagService handle. Returns None when
    no catalog is available, so callers fall back to acronym-only matching.
    """
    try:
        if db_path is None:
            from app.core.config import get_settings

            db_path = get_settings().sqlite_db_path
        path = Path(db_path)
        mtime = path.stat().st_mtime
    except (OSError, TypeError, ValueError):
        return None
    key = str(path)
    cached = _DEFAULT_VOCABULARY_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with connect(path) as connection:
            rows = connection.execute(
                """
                SELECT d.filename, c.keywords_json
                FROM documents d LEFT JOIN document_cards c ON c.document_id = d.id
                """
            ).fetchall()
    except Exception:  # noqa: BLE001 - a missing/old schema means no vocabulary
        return None
    vocabulary = _catalog_vocabulary(rows)
    _DEFAULT_VOCABULARY_CACHE[key] = (mtime, vocabulary)
    return vocabulary


def _catalog_vocabulary(rows) -> CatalogVocabulary:
    keywords: list[list[str]] = []
    for row in rows:
        try:
            raw = json.loads(row["keywords_json"] or "[]") if "keywords_json" in row.keys() else []
        except (TypeError, ValueError):
            raw = []
        keywords.append([str(item) for item in raw] if isinstance(raw, list) else [])
    return CatalogVocabulary.from_names([str(row["filename"] or "") for row in rows], keywords)


def _score_document_for_tokens(
    tokens: list[str], filename: str, source_path: str, *, aliases: set[str] | frozenset[str] = frozenset(),
) -> float:
    if not tokens:
        return 0.0

    haystack_compact = _compact_token(f"{filename} {source_path}")
    name_words = set(_name_words(filename))
    score = 0.0
    specific_hits = 0

    for token in tokens:
        if token in _GENERIC_ENTITY_TOKENS:
            if token in haystack_compact:
                score += 0.15
            continue

        if token in haystack_compact:
            # A whole name word ("crab", "kst") is as strong as a long token.
            weight = 12.0 if token in name_words else max(5.0, len(token) * 0.45)
            score += weight
            specific_hits += 1
            continue

        if token in aliases and len(token) >= 6:
            # A model/method name taken from the document's own keywords.
            score += 12.0
            specific_hits += 1
            continue

        # Compound alias whose parts are separated in the name
        # ("mambafusion" vs "..._MAMBA_BASED_FUSION_MODEL").
        for split in range(3, len(token) - 2):
            head, tail = token[:split], token[split:]
            if head in name_words and tail in name_words:
                score += 20.0
                specific_hits += 1
                break

    if specific_hits == 0:
        return 0.0
    if score < 2.0 and not any(len(token) >= 3 for token in tokens):
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


def _document_ids_from_fts_chunks(
    results: list[RetrievedChunk],
    max_documents: int | None = None,
) -> list[str]:
    """Return document owners in lexical rank order for candidate recovery."""

    seen: set[str] = set()
    document_ids: list[str] = []
    for result in results:
        document_id = str(result.document_id or "").strip()
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
        if max_documents is not None and len(document_ids) >= max_documents:
            break
    return document_ids


def _document_ids_from_artifact_results(
    results: list[dict[str, Any]],
    max_documents: int | None = None,
) -> list[str]:
    """Return artifact owners in lexical rank order for scope recovery."""

    seen: set[str] = set()
    document_ids: list[str] = []
    for result in results:
        document_id = str(result.get("document_id") or "").strip()
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
        if max_documents is not None and len(document_ids) >= max_documents:
            break
    return document_ids


def _merge_fts_chunks(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Fuse lexical candidates from the raw and structural query views.

    The original query preserves exact user terms; the structural view adds
    facet anchors such as ``abstract`` or ``architecture`` for underspecified
    paper questions.  Reciprocal-rank fusion keeps either view from replacing
    the other and retains the canonical chunk payload from the best lexical
    occurrence.
    """

    if limit < 1:
        return []
    merged: dict[str, tuple[RetrievedChunk, float, int]] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            identity = str(chunk.chunk_id or "")
            if not identity:
                continue
            contribution = _rrf(rank)
            current = merged.get(identity)
            if current is None:
                merged[identity] = (chunk, contribution, rank)
                continue
            existing, score, best_rank = current
            preferred = chunk if rank < best_rank else existing
            merged[identity] = (preferred, score + contribution, min(best_rank, rank))
    ordered = sorted(
        merged.values(),
        key=lambda item: (item[1], -item[2], -float(item[0].score or 0.0), item[0].chunk_id),
        reverse=True,
    )
    return [chunk for chunk, _score, _rank in ordered[:limit]]


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
    artifact_candidates: list[dict[str, Any]] | None = None,
    vector_ranked_groups: tuple[list[Any], ...] | None = None,
) -> list[dict]:
    merged: dict[str, dict] = {}
    if vector_ranked_groups is None:
        vector_entries = tuple(
            (rank, result)
            for rank, result in enumerate(vector_chunks, start=1)
        )
    else:
        # Each Lance projection is a separate retrieval channel.  Rank within
        # that channel, then let the common RRF score merge duplicate evidence
        # identities across modalities.
        vector_entries = tuple(
            (rank, result)
            for group in vector_ranked_groups
            for rank, result in enumerate(
                sorted(group, key=lambda item: float(item.score or 0.0)),
                start=1,
            )
        )
    for rank, result in vector_entries:
        chunk_id = result.id
        contribution = _rrf(rank)
        existing = merged.get(chunk_id)
        if existing is not None:
            existing["score"] += contribution
            existing["vector_rank"] = min(
                int(existing.get("vector_rank") or rank), rank
            )
            existing["retrieval_channels"] = list(
                dict.fromkeys(
                    [
                        *list(existing.get("retrieval_channels") or []),
                        *_vector_payload(result).get("retrieval_channels", []),
                    ]
                )
            )
            continue
        merged[chunk_id] = {
            **_vector_payload(result),
            "score": contribution,
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

    # Canonical table/figure rows are not child text chunks and therefore do
    # not appear in FTS5.  Merge their bounded lexical frontier as a third
    # ranked channel while preserving the same evidence identity and scoring
    # contract used by text/vector candidates.
    # Canonical artifact recovery is intentionally opt-in by query shape.  A
    # table/figure row should rescue an explicit artifact question, but must
    # not consume a visible slot for ordinary architecture/method questions
    # where the text frontier is the stronger evidence channel.
    artifact_frontier = (
        artifact_candidates or []
        if _query_requests_artifact_recovery(query)
        else []
    )
    for rank, artifact in enumerate(artifact_frontier, start=1):
        identity = str(
            artifact.get("chunk_id")
            or artifact.get("id")
            or artifact.get("evidence_id")
            or ""
        ).strip()
        if not identity:
            continue
        existing = merged.get(identity)
        if existing is not None:
            existing["score"] += _rrf(rank)
            existing["retrieval_channels"] = list(
                dict.fromkeys(
                    [
                        *list(existing.get("retrieval_channels") or []),
                        *list(artifact.get("retrieval_channels") or []),
                    ]
                )
            )
            existing.setdefault("artifact_lexical_rank", rank)
            continue
        contribution = _rrf(rank)
        merged[identity] = {
            **artifact,
            "score": contribution,
            "rrf_score": contribution,
            "vector_rank": None,
            "fts_rank": None,
            "artifact_lexical_rank": rank,
            "retrieval_channels": list(
                artifact.get("retrieval_channels") or ["sqlite_active_artifact_lexical"]
            ),
        }

    for item in merged.values():
        item["rrf_score"] = item["score"]
        item["rerank_boost"] = _lexical_rerank_boost(query, item, visual_boost=visual_boost)
        item["score"] = item["rrf_score"] + item["rerank_boost"]

    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return _promote_structured_evidence(ranked, query=query, top_k=top_k)


_STRUCTURED_EVIDENCE_MARKERS = (
    "table", "tables", "bảng", "metric", "metrics", "result", "results",
    "kết quả", "benchmark", "performance", "accuracy", "f1", "ccc",
    "ratio", "proportion", "percentage", "percent", "value", "values",
    "compare", "comparison", "so sánh", "baseline", "ablation",
    "figure", "fig.", "diagram", "chart", "plot", "image", "picture",
    "hình", "biểu đồ", "sơ đồ",
    "visual", "grounding", "image", "images", "pixel", "multimodal",
    "visual grounding", "image-text", "image text",
)


def _query_requests_artifact_recovery(query: str) -> bool:
    """Return whether canonical table/figure rows may join the text frontier."""

    lowered = " ".join(str(query or "").casefold().split())
    return any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered)
        for marker in _STRUCTURED_EVIDENCE_MARKERS
    ) or _query_requests_enumerated_evidence(lowered)


def _query_requests_enumerated_evidence(query: str | None) -> bool:
    """Detect list/count questions where a table may carry the exact items.

    This is a query-shape signal rather than a benchmark or paper rule.  It
    catches questions such as "which languages/classes/models" while avoiding
    a table reservation for ordinary singular prose questions.
    """

    return has_enumerated_evidence_intent(query)


def _query_has_explicit_artifact_locator(query: str | None) -> bool:
    """Detect a concrete table/figure label used as a routing constraint.

    A generic visual request should still be routed by document cards and text
    signals.  A request naming ``Figure 89`` or ``Table No. 2`` is different:
    the label is an exact identifier and can safely recover the owning
    document from the bounded canonical-artifact projection before vector
    search.  The pattern is language/format agnostic and intentionally does
    not inspect answers or evaluation metadata.
    """

    lowered = " ".join(str(query or "").casefold().split())
    if not lowered:
        return False
    # Keep labels separate from ordinary prose such as "figure in the paper".
    # Optional separators cover ``Fig. 2``, ``Table #3`` and Vietnamese forms.
    return bool(
        re.search(
            r"(?<!\w)(?:figure|fig\.?|table|bảng|hình)\s*"
            r"(?:#|no\.?|number|số)?\s*\d+(?!\w)",
            lowered,
            flags=re.IGNORECASE,
        )
    )


def _one_based_index(value: object) -> int | None:
    """Return a parser index in the human-facing one-based form."""

    try:
        return int(value) + 1 if value is not None else None
    except (TypeError, ValueError):
        return None


def _promote_structured_evidence(
    ranked: list[dict[str, Any]],
    *,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Reserve a high-signal artifact when the query asks for structure.

    Hybrid RRF can place several near-duplicate title/intro chunks ahead of a
    table or figure whose caption contains the requested metric.  The
    candidate is already in the same retrieved frontier; this function only
    moves one artifact into the visible top-k when the query has a generic
    structured-evidence cue and the artifact shares a content-bearing query
    term.  It never creates evidence, broadens the search, or names a paper.
    """

    if top_k < 1 or len(ranked) <= top_k:
        return ranked[:max(0, top_k)]
    lowered_query = " ".join(str(query or "").casefold().split())
    if not any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered_query)
        for marker in _STRUCTURED_EVIDENCE_MARKERS
    ) and not _query_requests_enumerated_evidence(lowered_query):
        return ranked[:top_k]

    query_terms = {
        token.casefold()
        for token in _rerank_tokens(query)
        if len(token) >= 2
    }
    artifact_rows: list[tuple[tuple[int, float, int], dict[str, Any]]] = []
    structured_request = _query_requests_artifact_recovery(lowered_query)
    for index, item in enumerate(ranked):
        artifact_type = str(
            item.get("artifact_type") or item.get("chunk_type") or ""
        ).casefold()
        if artifact_type not in {"table", "figure", "image"} and not (
            item.get("table_id") or item.get("figure_id")
        ):
            continue
        # ``content`` for a vector artifact may include a long inherited
        # page/document context. Do not let that context make an unrelated
        # artifact look lexically relevant; caption/summary/OCR are the
        # artifact's own evidence, while only a bounded content prefix is
        # considered as a fallback for table cells.
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if artifact_type == "figure":
            artifact_text_parts = (
                item.get("caption"),
                item.get("visual_summary"),
                metadata.get("ocr_text"),
                metadata.get("search_phrases"),
            )
        else:
            artifact_text_parts = (
                item.get("caption"),
                item.get("visual_summary"),
                metadata.get("ocr_text"),
                str(item.get("content") or "")[:1_200],
                item.get("section_title"),
            )
        text = " ".join(
            str(value or "")
            for value in artifact_text_parts
        ).casefold()
        overlap = sum(
            _query_term_matches_text(term, text)
            for term in query_terms
        )
        if overlap <= 0:
            # A structured question can be semantically relevant to an
            # artifact even when its wording does not repeat the caption's
            # exact nouns (for example ``visual grounding`` versus a caption
            # that says ``image-text data``). Keep only rows that came through
            # a canonical artifact/vector channel; ordinary text rows never
            # enter this fallback. The lexical/vector rank remains the
            # deterministic tie-breaker, so this cannot manufacture evidence.
            channels = {
                str(value).casefold()
                for value in (item.get("retrieval_channels") or [])
            }
            if not structured_request or not (
                channels.intersection(
                    {
                        "sqlite_active_artifact_lexical",
                        "lancedb_table_chunks",
                        "lancedb_figure_chunks",
                        "same_page_artifact",
                    }
                )
            ):
                continue
        artifact_type = str(
            item.get("artifact_type") or item.get("chunk_type") or ""
        ).casefold()
        if _query_requests_enumerated_evidence(query):
            # For list/count questions, the best modality-local vector hit is
            # usually the artifact carrying the exact item set; broad context
            # words such as ``experiment`` can otherwise make a comparison
            # table outrank the dataset table.  Prefer a canonical table over
            # a figure wrapper, then vector distance, with lexical overlap as
            # a deterministic tie-breaker.  Missing distances are kept last.
            try:
                distance = float(item.get("raw_vector_score"))
            except (TypeError, ValueError):
                distance = float("inf")
            artifact_rows.append(
                (
                    (
                        # Prefer the canonical table identity over a mirrored
                        # figure wrapper for image tables.  Both may share a
                        # page and OCR, but only the table row carries the
                        # table citation contract used by the composer.
                        int(bool(item.get("table_id")))
                        if artifact_type == "table"
                        else int(bool(item.get("figure_id"))),
                        -distance,
                        overlap,
                        float(item.get("score") or 0.0),
                        -index,
                    ),
                    item,
                )
            )
        else:
            try:
                lexical_rank = int(item.get("artifact_lexical_rank") or 10_000)
            except (TypeError, ValueError):
                lexical_rank = 10_000
            # Prefer a stronger canonical lexical match when overlap ties;
            # this prevents a semantically similar but unrelated artifact from
            # beating an exact caption match merely because its vector rank is
            # a little higher.
            artifact_rows.append(
                (
                    (
                        overlap,
                        -lexical_rank,
                        float(item.get("score") or 0.0),
                        -index,
                    ),
                    item,
                )
            )
    if not artifact_rows:
        chosen = None
    else:
        _score, chosen = max(artifact_rows, key=lambda pair: pair[0])
    visible = ranked[:top_k]
    if chosen is not None and chosen not in visible:
        # Keep the strongest top-k-1 results and reserve the artifact at the
        # end; subsequent context projection may still reorder explicit
        # modality reservations, but the artifact cannot disappear at stage 1.
        visible = [*visible[: max(0, top_k - 1)], chosen]

    return visible


def _collapse_page_coverage_results(
    results: list[dict],
    *,
    max_pages: int,
) -> list[dict]:
    """Collapse a ranked chunk frontier to one result per document page.

    A page-level aggregation query should not spend all of its context on
    adjacent child chunks from one page.  Artifact rows without a page remain
    distinct and are retained after the page frontier.  This function only
    changes the projection returned by coverage mode; regular retrieval keeps
    its existing chunk ranking.
    """

    page_limit = max(1, min(int(max_pages), 64))
    pages: dict[tuple[str, int], dict] = {}
    remainder: list[dict] = []
    for item in results:
        document_id = str(item.get("document_id") or "").strip()
        page = item.get("page_number")
        artifact_type = str(item.get("artifact_type") or item.get("chunk_type") or "").lower()
        # Canonical tables/figures are already page-addressable but must keep
        # their artifact identity for citations and exact lookup.  Only text
        # chunks collapse into a synthetic page evidence unit.
        if (
            item.get("table_id")
            or item.get("figure_id")
            or artifact_type in {"table", "figure"}
            or not document_id
            or page is None
        ):
            remainder.append(item)
            continue
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            remainder.append(item)
            continue
        key = (document_id, page_number)
        existing = pages.get(key)
        if existing is None:
            pages[key] = {
                **item,
                "chunk_id": f"page:{document_id}:{page_number}",
                "evidence_id": f"page:{document_id}:{page_number}",
                "artifact_type": "page",
                "chunk_type": "page",
                "coverage_mode": True,
                "coverage_chunk_ids": [
                    value
                    for value in (str(item.get("chunk_id") or ""),)
                    if value
                ],
            }
            continue
        existing_ids = list(existing.get("coverage_chunk_ids") or [])
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id not in existing_ids and len(existing_ids) < 4:
            existing_ids.append(chunk_id)
        existing["coverage_chunk_ids"] = existing_ids

    ordered_pages = sorted(
        pages.values(),
        key=lambda item: (
            -float(item.get("score") or item.get("rrf_score") or 0.0),
            str(item.get("document_id") or ""),
            int(item.get("page_number") or 0),
        ),
    )[:page_limit]
    return [*ordered_pages, *remainder]


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
    artifact_type = canonical_artifact_type(
        {
            **metadata,
            "artifact_type": metadata.get("artifact_type"),
            "chunk_type": metadata.get("chunk_type") or result.source,
        },
        default=result.source,
    )
    is_table = artifact_type == "table"
    table_id = metadata.get("table_id") or (metadata.get("figure_id") if is_table else None)
    figure_id = metadata.get("figure_id")
    return {
        "chunk_id": result.id,
        "document_id": metadata.get("document_id"),
        "revision_id": metadata.get("revision_id"),
        "source_path": metadata.get("source_path"),
        "filename": metadata.get("filename"),
        "content": result.text,
        "raw_vector_score": result.score,
        "retrieval_channels": ["lancedb", _lancedb_channel(result.source)],
        "heading_path": metadata.get("heading_path") or [],
        "page_number": metadata.get("page_number"),
        "page_label": metadata.get("page_label"),
        "chunk_index": metadata.get("chunk_index") if metadata.get("chunk_index") is not None else metadata.get("order_index"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "section_title": metadata.get("section_title"),
        "chunk_type": "table" if is_table else (metadata.get("chunk_type") or result.source),
        "artifact_type": artifact_type if artifact_type != "text" else metadata.get("artifact_type"),
        "caption": metadata.get("caption"),
        "image_path": metadata.get("image_path"),
        "image_url": f"/rag/figures/{figure_id}/image" if figure_id else None,
        "table_id": table_id,
        "figure_id": figure_id,
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


def _filter_active_revision_results(results, active_revisions: dict[str, str]):
    """Drop stale revision vectors while preserving legacy initial projections.

    The first revision migration is metadata-only: existing Lance rows cannot
    be rewritten atomically just to add ``revision_id``.  Such rows are
    canonically owned by the backfilled ``rev:<document>:initial`` revision
    until the isolated projection rebuild runs.  Missing revision metadata is
    *not* accepted for any later revision, so staged/retired vectors cannot
    leak into an active search.
    """

    if not active_revisions:
        return list(results)
    filtered = []
    for result in results:
        metadata = result.metadata or {}
        document_id = str(metadata.get("document_id") or "")
        expected = active_revisions.get(document_id)
        revision_id = str(metadata.get("revision_id") or "")
        legacy_initial = not revision_id and str(expected or "").endswith(":initial")
        if expected is None or revision_id == expected or legacy_initial:
            filtered.append(result)
    return filtered


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
    # Table images are intentionally stored as canonical table rows.  Older
    # adapters persisted the path in metadata rather than a physical column;
    # normalize both layouts at this boundary so every downstream consumer
    # sees the same lossless evidence shape.
    if not row.get("image_path"):
        metadata = row["metadata"]
        if isinstance(metadata, dict):
            image_path = metadata.get("image_path") or metadata.get("table_image_path")
            if image_path:
                row["image_path"] = image_path
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

    # Structural priors are deliberately section/content based rather than
    # paper-name rules. They keep an abstract/introduction or method section
    # from losing to a dense reference/listing chunk when the user asks a
    # generic purpose/architecture question.
    requested_facets = requested_paper_facets(
        query,
        focused_document_count=2 if any(marker in query_lower for marker in ("compare", "so sánh", "đối chiếu")) else 1,
    )
    structural_text = " ".join(
        [
            str(result.get("section_title") or ""),
            " ".join(str(item) for item in result.get("heading_path") or []),
            str(result.get("caption") or ""),
            str(result.get("content") or "")[:700],
        ]
    ).casefold()
    facet_markers = {
        "task": ("abstract", "introduction", "purpose", "vision", "problem", "in this work", "we explore", "to address"),
        "architecture": ("method", "architecture", "proposed method", "framework", "pipeline", "model", "fusion", "design"),
        "contributions": ("contribution", "novel", "we propose", "proposed"),
        "dataset_setup": ("dataset", "data", "corpus", "protocol", "experimental setup"),
        "benchmark_results": ("result", "experiment", "evaluation", "benchmark", "table"),
        "training_method": ("training", "loss", "optimizer", "objective"),
    }
    structural_hits = sum(
        1
        for facet in requested_facets
        if any(marker in structural_text for marker in facet_markers.get(facet, ()))
    )
    boost += min(structural_hits * 0.004, 0.016)

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

    # A caller may enable the visual channel because a document contains
    # artifacts, but that alone should not make every table/figure outrank
    # prose for a semantic question.  Promote artifact rows when the query
    # itself asks for a visual artifact; otherwise keep the native lexical /
    # semantic ranking in charge.  This avoids a table-image caption such as
    # ``Table 3: Results`` displacing the paragraph that explains the metric.
    artifact_request = any(
        marker in query_lower
        for marker in (
            "table", "tables", "figure", "fig.", "diagram", "chart", "plot",
            "image", "picture", "bảng", "hình", "biểu đồ", "sơ đồ",
        )
    )
    if visual_boost and artifact_request:
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
        for term in ["table", "tables", "bảng"]
    ):
        boost += 0.014

    channels = set(result.get("retrieval_channels") or [])
    if {"lancedb", "sqlite_fts5"}.issubset(channels):
        boost += 0.004
    # A small exact-term tie-breaker preserves hybrid behavior without making
    # FTS-only captions dominate semantically stronger vector results. Larger
    # weights must be justified by paired retrieval evidence, not assumed from
    # a single benchmark family.
    if result.get("fts_rank") == 1:
        boost += 0.002
    return boost


def _rerank_tokens(text: str) -> list[str]:
    from app.rag.vietnamese_text import tokenize_for_fts

    return tokenize_for_fts(text, max_tokens=24)


def _query_term_matches_text(term: str, text: str) -> bool:
    """Match a query anchor with conservative English plural variants.

    OCR/table headers frequently use ``Language`` while a natural question
    asks for ``languages`` (and similarly ``model``/``models``).  A tiny
    suffix normalization improves artifact reservation without introducing a
    stemmer, fuzzy match, or paper-specific vocabulary.
    """

    normalized_term = str(term or "").casefold().strip()
    normalized_text = str(text or "").casefold()
    if not normalized_term:
        return False
    variants = {normalized_term}
    if len(normalized_term) > 3 and normalized_term.endswith("s"):
        variants.add(normalized_term[:-1])
    if len(normalized_term) > 4 and normalized_term.endswith("ies"):
        variants.add(normalized_term[:-3] + "y")
    # Common query/caption derivations that matter for structured evidence
    # (``compare`` vs ``comparison``, ``evaluate`` vs ``evaluation``). Keep
    # this finite and transparent rather than introducing fuzzy matching that
    # could make unrelated figures look relevant.
    variants.update(
        {
            "comparison" if normalized_term in {"compare", "comparing", "comparative"} else normalized_term,
            "compare" if normalized_term == "comparison" else normalized_term,
            "evaluation" if normalized_term in {"evaluate", "evaluated", "evaluating"} else normalized_term,
            "evaluate" if normalized_term == "evaluation" else normalized_term,
        }
    )
    return any(
        re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized_text)
        for variant in variants
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _page_context_terms(query: str | None) -> tuple[str, ...]:
    """Extract content-bearing terms for deterministic same-page ranking."""

    if not query:
        return ()
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(query)).casefold(), flags=re.UNICODE)
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
        "do", "does", "for", "from", "how", "much", "in", "is", "it", "its", "of",
        "on", "or", "other", "that", "the", "their", "they", "this", "to", "was",
        "were", "what", "when", "where", "which", "who", "why", "with", "đã", "cho", "có",
        "của", "gì", "là", "nào", "như", "và", "về", "trên", "thế", "sao",
    }
    # Preserve numbers/acronyms and terms long enough to carry meaning.  Keep
    # order stable so ties resolve by the source chunk order.
    return tuple(dict.fromkeys(token for token in tokens if token not in stopwords and (len(token) > 1 or token.isdigit())))


def _artifact_context_snippet(
    raw_context: object,
    query_terms: tuple[str, ...],
    *,
    max_chars: int,
) -> str:
    """Select a small query-relevant excerpt from parser artifact context.

    ``artifact_context`` is derived parser text, not benchmark data. It can
    contain a whole nearby section, so injecting it wholesale would make a
    table/figure row noisy. Prefer a window around a query-term hit; when no
    term occurs, omit it and keep the caption/OCR projection.
    """

    if max_chars < 1 or not raw_context:
        return ""
    context = " ".join(str(raw_context).split()).strip()
    if not context:
        return ""
    if len(context) <= max_chars:
        return context
    lowered = context.casefold()
    positions = [lowered.find(term.casefold()) for term in query_terms if term]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return ""
    center = min(positions)
    before = max_chars // 3
    start = max(0, center - before)
    end = min(len(context), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    snippet = context[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(context):
        snippet = snippet.rstrip() + "…"
    return snippet


def _query_requests_page_detail(query: str | None) -> bool:
    """Detect queries where a sibling chunk may carry the exact answer value.

    A page heading can match a semantic query while the requested number/date/
    identifier lives several child chunks later.  Keep this contract based on
    the *shape* of the request rather than a list of papers or answer strings.
    The signal is deliberately conservative: format placeholders, numeric
    literals, units and explicit exact-value wording are the only triggers.
    """

    normalized = " ".join(str(query or "").casefold().split())
    if not normalized:
        return False
    if re.search(r"\b(?:y{2,4}|m{1,2}|d{1,2})(?:[-/. ]+(?:y{2,4}|m{1,2}|d{1,2})){1,2}\b", normalized):
        return True
    if re.search(r"(?<!\w)\d[\d/_.:-]{1,}(?!\w)", normalized):
        return True
    return bool(
        re.search(
            r"\b(?:exact(?:ly)?|value|values|amount|number|date|dated|when|how many|percentage|percent|metric|score|accuracy|f1|ccc|ua|wa)\b",
            normalized,
        )
    )


_VISUAL_PAGE_HINT_RE = re.compile(
    # Accept the natural singular/plural forms (``page 3``, ``pages 3 and
    # 5``, ``slide 12`` and ``slides 12 and 14``).  This is only a lexical
    # transport constraint for explicit page requests; it never substitutes
    # a page for semantic retrieval or reads benchmark annotations.
    r"(?<!\w)(?:pages?|p\.?|slides?|trang(?:s)?)\s*"
    r"(?:number|no\.?|số)?\s*([1-9]\d{0,3})(?!\w)",
    re.IGNORECASE,
)
_VISUAL_PAGE_LIST_RE = re.compile(
    r"(?<!\w)(?:pages?|p\.?|slides?|trang(?:s)?)\s*"
    r"(?:number|no\.?|số)?\s*([1-9]\d{0,3})"
    r"(?:(?:\s*(?:,|and|&|và)\s*)([1-9]\d{0,3})){1,15}",
    re.IGNORECASE,
)
_VISUAL_ROMAN_PAGE_LIST_RE = re.compile(
    r"(?<!\w)(?:pages?|p\.?|slides?|trang(?:s)?)\s*"
    r"(?:number|no\.?|số)?\s*([MDCLXVI]{1,12})"
    r"(?:(?:\s*(?:,|and|&|và)\s*)([MDCLXVI]{1,12})){1,15}",
    re.IGNORECASE,
)
_VISUAL_ROMAN_PAGE_HINT_RE = re.compile(
    r"(?<!\w)(?:pages?|p\.?|slides?|trang(?:s)?)\s*"
    r"(?:number|no\.?|số)?\s*([MDCLXVI]{1,12})(?!\w)",
    re.IGNORECASE,
)

_ROMAN_PAGE_RE = re.compile(r"^(?=[MDCLXVI]+$)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$", re.IGNORECASE)


def _page_label_number(value: object) -> int | None:
    """Normalize numeric or Roman printed-page labels to an integer key."""
    return page_label_number(value)


def requested_visual_page_numbers(query: str | None) -> list[int]:
    """Extract explicit page/slide references without treating artifact IDs as pages."""

    if not query:
        return []
    values: list[int] = []
    # Handle natural lists such as ``slides 12 and 14`` in one lexical pass;
    # a second marker is not required for every member of the list.
    normalized_query = unicodedata.normalize("NFKC", str(query))
    for match in _VISUAL_PAGE_LIST_RE.finditer(normalized_query):
        for number_match in re.finditer(r"(?<!\w)([1-9]\d{0,3})(?!\w)", match.group(0)):
            raw = number_match.group(1)
            try:
                page = int(raw)
            except (TypeError, ValueError):
                continue
            if page > 0 and page not in values:
                values.append(page)
    for match in _VISUAL_PAGE_HINT_RE.finditer(normalized_query):
        try:
            page = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in values:
            values.append(page)
    for match in _VISUAL_ROMAN_PAGE_LIST_RE.finditer(normalized_query):
        for raw in match.groups():
            page = _page_label_number(raw)
            if page and page not in values:
                values.append(page)
    for match in _VISUAL_ROMAN_PAGE_HINT_RE.finditer(normalized_query):
        page = _page_label_number(match.group(1))
        if page and page not in values:
            values.append(page)
    return values[:16]


def visual_page_reservation_count(
    query: str | None,
    visual_pages: list[dict[str, Any]],
) -> int:
    """Return the number of retrieved visual pages to reserve in context.

    Explicit page/slide hints are scope constraints. Every matching unique
    document/page identity returned by the visual retriever must survive the
    bounded context projection; otherwise a multi-page question can lose
    evidence before the multimodal prompt is built. Unconstrained visual
    requests retain a one-page minimum, while unresolved hints never create
    synthetic evidence.
    """

    if not visual_pages:
        return 0
    hinted_pages = set(requested_visual_page_numbers(query))
    identities: set[tuple[str, int]] = set()
    for item in visual_pages:
        page = item.get("page_number")
        if page is None:
            continue
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number < 1:
            continue
        identities.add((str(item.get("document_id") or ""), page_number))
    if not identities:
        return 0
    if hinted_pages:
        hinted_identities: set[tuple[str, int]] = set()
        for item in visual_pages:
            try:
                page_number = int(item.get("page_number"))
            except (TypeError, ValueError):
                continue
            metadata = item.get("metadata")
            page_label = item.get("page_label")
            if page_label is None and isinstance(metadata, dict):
                page_label = metadata.get("page_label")
            if page_number in hinted_pages or any(
                label in hinted_pages for label in page_label_values(page_label)
            ):
                hinted_identities.add((str(item.get("document_id") or ""), page_number))
        hinted_count = len(hinted_identities)
        if hinted_count:
            return hinted_count
    # A generic list/count request can require more than one visual artifact
    # even when it never says "figure" or "compare" (for example, "which
    # sources are used?"). Preserve a small ranked frontier for those
    # questions, but never reserve more pages than the retriever returned.
    if has_enumerated_evidence_intent(query):
        return min(4, len(identities))
    # Questions that ask for importance/effect/comparison often need both the
    # explanatory figure and the result table.  Reserving a two-page frontier
    # here is content-driven (the retriever must already have returned both
    # pages), bounded, and applies to arbitrary documents rather than a named
    # benchmark or artifact.
    relation_markers = (
        "important",
        "importance",
        "effect",
        "impact",
        "degrad",
        "without",
        "compare",
        "so sánh",
        "ảnh hưởng",
        "quan trọng",
        "tác động",
    )
    lowered = " ".join(str(query or "").casefold().split())
    if any(marker in lowered for marker in relation_markers):
        return min(2, len(identities))
    return 1
