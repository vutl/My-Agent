from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from app.core.config import Settings
from app.db.sqlite import connect
from app.lightrag.provenance import (
    ResolvedParentPassage,
    resolve_lightrag_chunk_parent_status,
)
from app.lightrag.query import query_lightrag, resolve_query_mode
from app.rag.context import compose_retrieval_context


_STRUCTURAL_GRAPH_LABEL = re.compile(
    r"^(?:fig(?:ure)?\.?|table|section|appendix|equation|eq\.?|page)\s*"
    r"(?:[ivxlcdm]+|\d+(?:[.\-]\d+)*|[a-z])$",
    flags=re.IGNORECASE,
)
_GRAPH_FIELD_SEPARATOR = "<SEP>"
_HIGH_CARDINALITY_GENERIC_SOURCE_COUNT = 8
_DEFAULT_PARENT_CANDIDATES_PER_DOCUMENT = 4
_MAX_BRIDGE_RESULTS = 32


@dataclass(frozen=True)
class LightRAGBridge:
    settings: Settings

    async def retrieve(
        self,
        query: str,
        *,
        answer_intent: str | None = None,
        retrieval_mode: str | None = None,
        focus_document_ids: list[str] | None = None,
        answer_depth: str | None = None,
        include_visual_boost: bool = False,
    ) -> dict[str, Any]:
        mode = resolve_query_mode(answer_intent=answer_intent, retrieval_mode=retrieval_mode)
        raw = await query_lightrag(query, mode=mode, enable_rerank=False)
        context_budget = _context_budget(answer_intent, answer_depth)
        results, bridge_diagnostics = self._materialize_retrieval_results(
            raw,
            focus_document_ids=focus_document_ids,
            max_parent_candidates_per_document=context_budget[
                "max_chunks_per_document"
            ],
        )
        composed = compose_retrieval_context(
            results,
            query=query,
            max_sources=context_budget["max_sources"],
            max_chars=context_budget["max_chars"],
            max_chars_per_source=context_budget["max_chars_per_source"],
            max_chunks_per_document=context_budget["max_chunks_per_document"],
        )
        metadata = raw.get("metadata") or {}
        processing = metadata.get("processing_info") or {}
        return {
            "mode": f"lightrag:{mode}",
            "documents": composed.sources,
            "context_text": composed.context_text,
            "context_stats": composed.stats,
            "diagnostics": {
                "lightrag_mode": mode,
                "lightrag_status": raw.get("status"),
                "keywords": metadata.get("keywords"),
                "processing_info": processing,
                "focus_document_ids": focus_document_ids or [],
                "include_visual_boost": include_visual_boost,
                "bridge": bridge_diagnostics,
                "graph_bridge_metadata": bridge_diagnostics.get(
                    "graph_bridge_metadata",
                    [],
                ),
            },
            "lightrag_raw": raw,
        }

    def _to_retrieval_results(
        self,
        raw: dict[str, Any],
        *,
        focus_document_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Backward-compatible list-only projection used by the debug route."""

        results, _diagnostics = self._materialize_retrieval_results(
            raw,
            focus_document_ids=focus_document_ids,
        )
        return results

    def _materialize_retrieval_results(
        self,
        raw: dict[str, Any],
        *,
        focus_document_ids: list[str] | None,
        max_parent_candidates_per_document: int = (
            _DEFAULT_PARENT_CANDIDATES_PER_DOCUMENT
        ),
        max_results: int = _MAX_BRIDGE_RESULTS,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Use graph hits as navigation and return canonical parent evidence.

        Every eligible LightRAG source chunk is resolved in one SQLite call.
        Entity/relation descriptions are never answer evidence themselves.
        Unmapped raw chunks survive only when their own ID/path proves one
        canonical document inside the explicit focus, when supplied.
        """

        data = raw.get("data") or {}
        document_index = self._document_path_index()
        raw_chunks = [
            item for item in (data.get("chunks") or []) if isinstance(item, dict)
        ]
        raw_entities = [
            item for item in (data.get("entities") or []) if isinstance(item, dict)
        ]
        raw_relations = [
            item
            for item in (data.get("relationships") or [])
            if isinstance(item, dict)
        ]
        suppressions: Counter[str] = Counter()
        mapping_methods: Counter[str] = Counter()
        canonical_methods: Counter[str] = Counter()
        graph_cardinalities: list[int] = []

        graph_candidates: list[tuple[str, dict[str, Any], list[str]]] = []
        for kind, candidates in (
            ("entity", raw_entities),
            ("relation", raw_relations),
        ):
            for item in candidates:
                source_chunk_ids = _source_chunk_ids(item)
                graph_cardinalities.append(len(source_chunk_ids))
                if _is_generic_graph_candidate(
                    item,
                    relation=kind == "relation",
                ):
                    suppressions[f"generic_{kind}"] += 1
                    continue
                if not source_chunk_ids:
                    suppressions[f"{kind}_missing_source_id"] += 1
                    continue
                graph_candidates.append((kind, item, source_chunk_ids))

        requested_chunk_ids = _dedupe_strings(
            [
                chunk_id
                for chunk in raw_chunks
                for chunk_id in _source_chunk_ids(chunk)
            ]
            + [
                chunk_id
                for _kind, _item, source_ids in graph_candidates
                for chunk_id in source_ids
            ]
        )
        provenance_resolution = resolve_lightrag_chunk_parent_status(
            self.settings.sqlite_db_path,
            requested_chunk_ids,
            allowed_document_ids=focus_document_ids,
        )
        provenance = provenance_resolution.parents_by_chunk_id
        mapped_chunk_ids = {
            chunk_id
            for chunk_id, parents in provenance.items()
            if parents
        }
        for parents in provenance.values():
            for parent in parents:
                mapping_methods[parent.mapping_method] += 1
                canonical_methods[parent.canonical_method] += 1

        results: list[dict[str, Any]] = []
        parent_result_by_id: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}
        suppressed_parent_identities: set[tuple[str, str]] = set()
        parent_counts_by_document: Counter[str] = Counter()
        raw_chunk_ids_seen: set[str] = set()
        parent_deduplicated_count = 0
        raw_fallback_count = 0

        def emit_parent(
            parent: ResolvedParentPassage,
            *,
            source_chunk_id: str,
            channel: str,
            graph_item: dict[str, Any] | None = None,
            graph_kind: str | None = None,
        ) -> None:
            nonlocal parent_deduplicated_count
            parent_identity = (
                parent.document_id,
                parent.parent_chunk_id,
            )
            existing = parent_result_by_id.get(parent_identity)
            if existing is not None:
                parent_deduplicated_count += 1
                _merge_parent_navigation(
                    existing,
                    source_chunk_id=source_chunk_id,
                    channel=channel,
                    graph_item=graph_item,
                    graph_kind=graph_kind,
                )
                return
            if parent_identity in suppressed_parent_identities:
                return
            if (
                parent_counts_by_document[parent.document_id]
                >= max(1, max_parent_candidates_per_document)
            ):
                suppressions["parent_document_quota"] += 1
                suppressed_parent_identities.add(parent_identity)
                return
            if len(results) >= max(1, max_results):
                suppressions["result_cap"] += 1
                suppressed_parent_identities.add(parent_identity)
                return

            result = _parent_retrieval_result(
                parent,
                source_chunk_id=source_chunk_id,
                channel=channel,
            )
            _merge_parent_navigation(
                result,
                source_chunk_id=source_chunk_id,
                channel=channel,
                graph_item=graph_item,
                graph_kind=graph_kind,
            )
            results.append(result)
            parent_result_by_id[parent_identity] = result
            parent_counts_by_document[parent.document_id] += 1

        for chunk in raw_chunks:
            source_chunk_ids = _source_chunk_ids(chunk)
            mapped_parents = [
                (chunk_id, parent)
                for chunk_id in source_chunk_ids
                for parent in provenance.get(chunk_id, [])
            ]
            if mapped_parents:
                for chunk_id, parent in mapped_parents:
                    emit_parent(
                        parent,
                        source_chunk_id=chunk_id,
                        channel="lightrag_chunk",
                    )
                continue
            if any(
                chunk_id in provenance_resolution.known_chunk_ids
                for chunk_id in source_chunk_ids
            ):
                suppressions["raw_chunk_known_mapping_unresolved"] += 1
                continue

            # The raw chunk is a compatibility fallback only after durable
            # provenance misses.  It still needs one canonical document; an
            # unscoped ambiguous basename is not sufficient.
            file_path = str(chunk.get("file_path") or "")
            document = document_index.resolve(chunk, focus_document_ids=focus_document_ids)
            if document is None:
                suppressions["raw_chunk_unsafe_provenance"] += 1
                continue
            content = str(chunk.get("content") or "").strip()
            if not content:
                suppressions["raw_chunk_empty"] += 1
                continue
            chunk_id = str(
                chunk.get("chunk_id")
                or chunk.get("reference_id")
                or ""
            ).strip()
            if chunk_id and chunk_id in raw_chunk_ids_seen:
                suppressions["raw_chunk_duplicate"] += 1
                continue
            if len(results) >= max(1, max_results):
                suppressions["result_cap"] += 1
                continue
            if chunk_id:
                raw_chunk_ids_seen.add(chunk_id)
            raw_fallback_count += 1
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document.document_id,
                    "content": content,
                    "source_path": document.source_path,
                    "filename": document.filename,
                    "lightrag_source_path": file_path,
                    "retrieval_channels": ["lightrag_chunk"],
                    "provenance_fallback": "canonical_raw_chunk",
                }
            )

        for graph_kind, graph_item, source_chunk_ids in graph_candidates:
            candidate_mapped = False
            for source_chunk_id in source_chunk_ids:
                parents = provenance.get(source_chunk_id, [])
                if not parents:
                    continue
                candidate_mapped = True
                for parent in parents:
                    emit_parent(
                        parent,
                        source_chunk_id=source_chunk_id,
                        channel="graph_bridge",
                        graph_item=graph_item,
                        graph_kind=graph_kind,
                    )
            if not candidate_mapped:
                suppressions[f"unmapped_{graph_kind}"] += 1

        graph_bridge_metadata = _dedupe_graph_bridge_metadata(
            [
                record
                for result in results
                for record in result.get("graph_bridge_metadata", [])
                if isinstance(record, dict)
            ]
        )
        diagnostics = {
            "raw_chunk_candidate_count": len(raw_chunks),
            "graph_entity_candidate_count": len(raw_entities),
            "graph_relation_candidate_count": len(raw_relations),
            "graph_navigation_candidate_count": len(graph_candidates),
            "graph_source_cardinality_total": sum(graph_cardinalities),
            "graph_source_cardinality_max": max(graph_cardinalities, default=0),
            "provenance_requested_chunk_count": len(requested_chunk_ids),
            "provenance_mapped_chunk_count": len(mapped_chunk_ids),
            "provenance_unmapped_chunk_count": (
                len(requested_chunk_ids) - len(mapped_chunk_ids)
            ),
            "provenance_known_chunk_count": len(
                provenance_resolution.known_chunk_ids
            ),
            "provenance_stale_chunk_count": len(
                provenance_resolution.stale_chunk_ids
            ),
            "provenance_scoped_out_chunk_count": len(
                provenance_resolution.scoped_out_chunk_ids
            ),
            "provenance_parent_link_count": sum(
                len(parents) for parents in provenance.values()
            ),
            "provenance_mapping_methods": dict(sorted(mapping_methods.items())),
            "provenance_canonical_methods": dict(
                sorted(canonical_methods.items())
            ),
            "parent_passage_count": len(parent_result_by_id),
            "parent_deduplicated_count": parent_deduplicated_count,
            "parent_suppressed_count": len(
                suppressed_parent_identities
            ),
            "parent_counts_by_document": dict(
                sorted(parent_counts_by_document.items())
            ),
            "raw_fallback_count": raw_fallback_count,
            "suppression_reasons": dict(sorted(suppressions.items())),
            "graph_bridge_metadata": graph_bridge_metadata,
            "result_count": len(results),
            "result_cap": max(1, max_results),
            "parent_document_quota": max(
                1,
                max_parent_candidates_per_document,
            ),
        }
        return results, diagnostics

    def _document_path_index(self) -> _DocumentPathIndex:
        with connect(self.settings.sqlite_db_path) as connection:
            rows = connection.execute("SELECT id, source_path, filename FROM documents").fetchall()
        return _DocumentPathIndex.from_rows(rows)


@dataclass(frozen=True)
class _DocumentRef:
    document_id: str
    source_path: str
    filename: str


@dataclass(frozen=True)
class _DocumentPathIndex:
    """Resolve LightRAG provenance without guessing across duplicate filenames."""

    by_id: dict[str, _DocumentRef]
    by_path: dict[str, frozenset[str]]
    by_basename: dict[str, frozenset[str]]

    @classmethod
    def from_rows(cls, rows: list[Any]) -> _DocumentPathIndex:
        by_id: dict[str, _DocumentRef] = {}
        path_sets: dict[str, set[str]] = {}
        basename_sets: dict[str, set[str]] = {}
        for row in rows:
            document_id = str(row["id"])
            source_path = str(row["source_path"] or "")
            filename = str(row["filename"] or Path(source_path).name or "document")
            by_id[document_id] = _DocumentRef(
                document_id=document_id,
                source_path=source_path,
                filename=filename,
            )
            for alias in _path_aliases(source_path):
                path_sets.setdefault(alias, set()).add(document_id)
            basename_sets.setdefault(filename.casefold(), set()).add(document_id)
            if source_path:
                basename_sets.setdefault(Path(source_path).name.casefold(), set()).add(document_id)
        return cls(
            by_id=by_id,
            by_path={key: frozenset(value) for key, value in path_sets.items()},
            by_basename={key: frozenset(value) for key, value in basename_sets.items()},
        )

    def resolve(
        self,
        item: dict[str, Any],
        *,
        focus_document_ids: list[str] | None,
    ) -> _DocumentRef | None:
        allowed = set(focus_document_ids) if focus_document_ids is not None else None

        # Some LightRAG versions preserve the caller-provided document id.
        for key in ("document_id", "doc_id", "full_doc_id"):
            candidate = str(item.get(key) or "").strip()
            if candidate in self.by_id:
                return self.by_id[candidate] if allowed is None or candidate in allowed else None

        document_ids: set[str] = set()
        for file_path in _source_paths(item):
            path_candidates: set[str] = set()
            for alias in _path_aliases(file_path):
                path_candidates.update(self.by_path.get(alias, ()))
            if path_candidates:
                # Do not reinterpret an exact path as another same-named file.
                document_ids.update(path_candidates)
                continue

            # LightRAG canonicalizes document paths to basenames. Entity and
            # relationship provenance can join several basenames with <SEP>.
            basename = Path(file_path).name.casefold()
            document_ids.update(self.by_basename.get(basename, ()))
        return self._unique_allowed(document_ids, allowed)

    def _unique_allowed(
        self,
        document_ids: set[str],
        allowed: set[str] | None,
    ) -> _DocumentRef | None:
        # Focus is a filter, never provenance. If LightRAG only preserved an
        # ambiguous basename or a multi-paper <SEP> aggregate, intersecting it
        # with the active paper would silently relabel foreign evidence as that
        # paper. Drop it instead and let scoped Lance/FTS provide evidence.
        if len(document_ids) != 1:
            return None
        document_id = next(iter(document_ids))
        if allowed is not None and document_id not in allowed:
            return None
        return self.by_id[document_id]


def _path_aliases(value: str) -> set[str]:
    raw = value.strip()
    if not raw:
        return set()
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path)
    aliases = {raw}
    try:
        path = Path(raw).expanduser()
        aliases.add(str(path))
        aliases.add(str(path.resolve(strict=False)))
    except (OSError, RuntimeError, ValueError):
        pass
    return aliases


def _source_paths(item: dict[str, Any]) -> list[str]:
    raw = item.get("file_path") or item.get("source_path") or ""
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    paths: list[str] = []
    for value in values:
        paths.extend(
            part.strip()
            for part in str(value).split("<SEP>")
            if part.strip() and part.strip() not in {"unknown_source", "truncated"}
        )
    return paths


def _source_chunk_ids(item: dict[str, Any]) -> list[str]:
    raw = item.get("source_id") or item.get("chunk_id") or ""
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value).split(_GRAPH_FIELD_SEPARATOR):
            chunk_id = part.strip()
            if (
                not chunk_id
                or chunk_id in {"unknown_source", "truncated"}
                or chunk_id in seen
            ):
                continue
            seen.add(chunk_id)
            result.append(chunk_id)
    return result


def _is_generic_graph_candidate(item: dict[str, Any], *, relation: bool = False) -> bool:
    """Reject graph navigation hubs that are labels, not shared concepts.

    LightRAG merges equal entity names corpus-wide.  Labels such as "Figure 2"
    consequently join unrelated papers, while very short names backed by many
    source chunks behave like low-IDF stop nodes.  This filter uses only graph
    structure/cardinality; it does not encode paper-specific mappings.
    """

    if relation:
        labels = [
            str(item.get("src_id") or "").strip(),
            str(item.get("tgt_id") or "").strip(),
        ]
        meaningful = [label for label in labels if label]
        if not meaningful:
            return True
        if meaningful and all(_STRUCTURAL_GRAPH_LABEL.fullmatch(label) for label in meaningful):
            return True
        token_count = sum(len(re.findall(r"\w+", label, flags=re.UNICODE)) for label in meaningful)
        cardinality_limit = _HIGH_CARDINALITY_GENERIC_SOURCE_COUNT + 4
    else:
        label = str(item.get("entity_name") or item.get("name") or "").strip()
        if not label:
            return True
        if _STRUCTURAL_GRAPH_LABEL.fullmatch(label):
            return True
        token_count = len(re.findall(r"\w+", label, flags=re.UNICODE))
        cardinality_limit = _HIGH_CARDINALITY_GENERIC_SOURCE_COUNT

    return (
        len(_source_chunk_ids(item)) >= cardinality_limit
        and token_count <= 2
    )


def _parent_retrieval_result(
    parent: ResolvedParentPassage,
    *,
    source_chunk_id: str,
    channel: str,
) -> dict[str, Any]:
    return {
        "chunk_id": parent.parent_chunk_id,
        "parent_chunk_id": parent.parent_chunk_id,
        "document_id": parent.document_id,
        "content": parent.parent_content,
        "expanded_content": parent.parent_content,
        "source_path": parent.source_path,
        "filename": parent.filename,
        "lightrag_source_path": parent.lightrag_file_path,
        "lightrag_source_chunk_ids": [source_chunk_id],
        "lightrag_chunk_order_index": parent.lightrag_chunk_order_index,
        "parent_order_index": parent.parent_order_index,
        "page_number": parent.page_number,
        "section_title": parent.section_title,
        "heading_path": list(parent.heading_path),
        "chunk_type": parent.chunk_type,
        "provenance_mapping_method": parent.mapping_method,
        "provenance_mapping_score": parent.mapping_score,
        "provenance_overlap_chars": parent.overlap_chars,
        "provenance_content_hash": parent.content_hash,
        "parent_content_hash": parent.parent_content_hash,
        "retrieval_channels": _dedupe_strings(
            ["lightrag_parent", channel]
        ),
    }


def _merge_parent_navigation(
    result: dict[str, Any],
    *,
    source_chunk_id: str,
    channel: str,
    graph_item: dict[str, Any] | None,
    graph_kind: str | None,
) -> None:
    result["lightrag_source_chunk_ids"] = _dedupe_strings(
        list(result.get("lightrag_source_chunk_ids") or [])
        + [source_chunk_id]
    )
    channels = list(result.get("retrieval_channels") or []) + [channel]
    if graph_item is not None and graph_kind in {"entity", "relation"}:
        channels.extend(["graph_bridge", f"lightrag_{graph_kind}"])
    result["retrieval_channels"] = _dedupe_strings(channels)
    if graph_item is None or graph_kind not in {"entity", "relation"}:
        return

    anchors = _graph_anchors(graph_item, relation=graph_kind == "relation")
    if not anchors:
        return
    result["anchors"] = _dedupe_strings(
        list(result.get("anchors") or []) + anchors
    )
    record = {
        "document_id": str(result.get("document_id") or ""),
        "parent_chunk_id": str(result.get("parent_chunk_id") or ""),
        "source_chunk_ids": [source_chunk_id],
        "anchors": anchors,
        "graph_kind": graph_kind,
        "source_cardinality": len(_source_chunk_ids(graph_item)),
        "mapping_method": result.get("provenance_mapping_method"),
        "mapping_score": result.get("provenance_mapping_score"),
        "resolved": True,
        "covered": True,
        "requires_followup": False,
        "coverage_status": "resolved",
    }
    records = list(result.get("graph_bridge_metadata") or [])
    records.append(record)
    records = _dedupe_graph_bridge_metadata(records)
    result["graph_bridge_metadata"] = records
    result["graph_bridge"] = {
        "document_id": record["document_id"],
        "parent_chunk_id": record["parent_chunk_id"],
        "anchors": result["anchors"],
        "source_chunk_ids": list(result["lightrag_source_chunk_ids"]),
        "resolved": True,
        "coverage_status": "resolved",
    }
    metadata = result.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["graph_bridge_metadata"] = records
    result["metadata"] = metadata


def _graph_anchors(
    item: dict[str, Any],
    *,
    relation: bool,
) -> list[str]:
    if relation:
        candidates = [
            str(item.get("src_id") or "").strip(),
            str(item.get("tgt_id") or "").strip(),
        ]
        keywords = str(item.get("keywords") or "")
        candidates.extend(
            value.strip()
            for value in re.split(r"(?:<SEP>|[;|\n,])", keywords)
            if value.strip()
        )
    else:
        candidates = [
            str(item.get("entity_name") or item.get("name") or "").strip()
        ]
    return _dedupe_strings(
        [
            candidate
            for candidate in candidates
            if candidate
            and not _STRUCTURAL_GRAPH_LABEL.fullmatch(candidate)
        ]
    )


def _dedupe_graph_bridge_metadata(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        identity = (
            str(record.get("document_id") or ""),
            str(record.get("parent_chunk_id") or ""),
            str(record.get("graph_kind") or ""),
            tuple(_dedupe_strings(list(record.get("source_chunk_ids") or []))),
            tuple(_dedupe_strings(list(record.get("anchors") or []))),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(record)
    return deduped


def _dedupe_strings(values: list[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value or "").strip()
        )
    )


def _context_budget(answer_intent: str | None, answer_depth: str | None) -> dict[str, int]:
    detailed = answer_depth == "detailed" or answer_intent in {"elaborate", "compare", "infer_structure"}
    if detailed:
        return {
            "max_sources": 8,
            "max_chars": 6_500,
            "max_chars_per_source": 1_400,
            "max_chunks_per_document": 4,
        }
    return {
        "max_sources": 6,
        "max_chars": 5_200,
        "max_chars_per_source": 1_200,
        "max_chunks_per_document": 3,
    }
