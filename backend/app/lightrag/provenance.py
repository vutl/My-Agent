from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from app.db.sqlite import connect


_PARENT_SEPARATOR = "\n\n"
_SHINGLE_SIZE = 5
_SHINGLE_MIN_SCORE = 0.55
_SHINGLE_MIN_MARGIN = 0.05
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class ProvenanceSyncError(RuntimeError):
    """Raised when LightRAG storage cannot be read safely for a sync."""


@dataclass(frozen=True)
class ProvenanceSyncResult:
    document_id: str
    records_seen: int
    records_mapped: int
    mappings_written: int
    records_rejected: int
    rejection_reasons: dict[str, int]


@dataclass(frozen=True)
class ResolvedParentPassage:
    lightrag_chunk_id: str
    document_id: str
    parent_chunk_id: str
    parent_content: str
    source_path: str
    filename: str
    lightrag_full_doc_id: str
    lightrag_file_path: str
    lightrag_chunk_order_index: int
    parent_order_index: int
    content_hash: str
    parent_content_hash: str
    overlap_chars: int
    canonical_method: str
    mapping_method: str
    mapping_score: float
    document_char_start: int
    document_char_end: int
    mapped_at: str
    page_number: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()
    chunk_type: str = "text"


@dataclass(frozen=True)
class ProvenanceResolution:
    parents_by_chunk_id: dict[str, list[ResolvedParentPassage]]
    known_chunk_ids: frozenset[str]
    stale_chunk_ids: frozenset[str]
    scoped_out_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class _CanonicalDocument:
    document_id: str
    source_path: str
    filename: str


@dataclass(frozen=True)
class _ParentSpan:
    parent_chunk_id: str
    content: str
    order_index: int
    start: int
    end: int
    content_hash: str
    page_number: int | None
    section_title: str | None
    heading_path: tuple[str, ...]
    chunk_type: str


@dataclass(frozen=True)
class _CanonicalResolution:
    document: _CanonicalDocument
    method: str
    score: float


@dataclass(frozen=True)
class _ContentResolution:
    start: int
    end: int
    method: str
    score: float


@dataclass(frozen=True)
class _MappingRow:
    lightrag_chunk_id: str
    parent_chunk_id: str
    document_id: str
    lightrag_full_doc_id: str
    lightrag_file_path: str
    lightrag_chunk_order_index: int
    parent_order_index: int
    content_hash: str
    parent_content_hash: str
    overlap_chars: int
    document_char_start: int
    document_char_end: int
    canonical_method: str
    mapping_method: str
    mapping_score: float
    mapped_at: str


ChunkRecordInput = (
    Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
)


def sync_document_chunk_records(
    db_path: Path,
    document_id: str,
    records: ChunkRecordInput,
) -> ProvenanceSyncResult:
    """Atomically rebuild one document's LightRAG-to-parent provenance.

    This entry point is deliberately provider-free.  It accepts records already
    loaded from ``kv_store_text_chunks.json`` or LightRAG's ``text_chunks`` KV
    storage.  Invalid, ambiguous, foreign-document, or non-unique content
    matches are omitted rather than relabeled from conversation focus.
    """

    record_list = _coerce_records(records)
    rejection_reasons: Counter[str] = Counter()
    mapped_chunk_ids: set[str] = set()
    mapping_rows: list[_MappingRow] = []
    mapped_at = datetime.now(timezone.utc).isoformat()

    with connect(db_path) as connection:
        # Serialize document rebuilds and take a consistent view of documents
        # plus child metadata before replacing the old mapping rows.
        connection.execute("BEGIN IMMEDIATE")
        document_rows = connection.execute(
            "SELECT id, source_path, filename FROM documents"
        ).fetchall()
        documents = {
            str(row["id"]): _CanonicalDocument(
                document_id=str(row["id"]),
                source_path=str(row["source_path"] or ""),
                filename=str(row["filename"] or ""),
            )
            for row in document_rows
        }
        target_document = documents.get(document_id)
        if target_document is None:
            raise ValueError(f"canonical_document_not_found:{document_id}")

        parents, document_text = _load_parent_sequence(connection, document_id)
        parent_by_id = {parent.parent_chunk_id: parent for parent in parents}
        seen_records: dict[str, tuple[Any, ...]] = {}
        conflicted_ids: set[str] = set()

        for record in record_list:
            chunk_id = str(record.get("_id") or "").strip()
            if not chunk_id:
                rejection_reasons["missing_chunk_id"] += 1
                continue
            fingerprint = (
                record.get("content"),
                record.get("full_doc_id"),
                record.get("file_path"),
                record.get("chunk_order_index"),
            )
            previous = seen_records.get(chunk_id)
            if previous is not None:
                if previous != fingerprint and chunk_id not in conflicted_ids:
                    rejection_reasons["conflicting_duplicate_chunk_id"] += 1
                    conflicted_ids.add(chunk_id)
                continue
            seen_records[chunk_id] = fingerprint

        for record in record_list:
            chunk_id = str(record.get("_id") or "").strip()
            if not chunk_id or chunk_id in conflicted_ids:
                continue
            fingerprint = (
                record.get("content"),
                record.get("full_doc_id"),
                record.get("file_path"),
                record.get("chunk_order_index"),
            )
            if seen_records.get(chunk_id) != fingerprint:
                continue
            # Consume each identical duplicate once.
            seen_records.pop(chunk_id, None)

            content = record.get("content")
            if not isinstance(content, str) or not content.strip():
                rejection_reasons["missing_content"] += 1
                continue
            order_index = record.get("chunk_order_index")
            if isinstance(order_index, bool) or not isinstance(order_index, int):
                rejection_reasons["invalid_chunk_order_index"] += 1
                continue
            full_doc_id = str(record.get("full_doc_id") or "").strip()
            file_path = str(record.get("file_path") or "").strip()
            canonical, canonical_error = _resolve_canonical_document(
                documents,
                full_doc_id=full_doc_id,
                file_path=file_path,
            )
            if canonical is None:
                rejection_reasons[canonical_error or "unresolved_document"] += 1
                continue
            if canonical.document.document_id != document_id:
                rejection_reasons["foreign_document"] += 1
                continue
            if not parents or not document_text:
                rejection_reasons["no_parent_passages"] += 1
                continue

            content_match = _resolve_content_offset(document_text, content)
            if content_match is not None:
                overlapping_parents = [
                    parent
                    for parent in parents
                    if max(parent.start, content_match.start)
                    < min(parent.end, content_match.end)
                ]
                if not overlapping_parents:
                    rejection_reasons["no_overlapping_parent"] += 1
                    continue
                mapping_method = content_match.method
                mapping_score = content_match.score
                document_char_start = content_match.start
                document_char_end = content_match.end
            else:
                # The current graph may predate a canonical parent-child
                # migration.  Permit a provider-free local backfill only when
                # one parent in the already-resolved canonical document is a
                # strong, clearly separated top-1 shingle match.  Ties and
                # near-ties remain unmapped rather than guessing a page.
                shingle_match = _resolve_parent_by_shingles(parents, content)
                if shingle_match is None:
                    rejection_reasons[
                        "content_not_unique_or_shingle_ambiguous"
                    ] += 1
                    continue
                shingle_parent, shingle_score = shingle_match
                overlapping_parents = [shingle_parent]
                mapping_method = "token_5_shingle_top1"
                mapping_score = shingle_score
                # A fuzzy match proves parent ownership, not a trustworthy
                # document character span.
                document_char_start = -1
                document_char_end = -1

            content_hash = _sha256(content)
            score = canonical.score * mapping_score
            for parent in overlapping_parents:
                overlap_chars = (
                    max(
                        0,
                        min(parent.end, document_char_end)
                        - max(parent.start, document_char_start),
                    )
                    if document_char_start >= 0
                    else 0
                )
                mapping_rows.append(
                    _MappingRow(
                        lightrag_chunk_id=chunk_id,
                        parent_chunk_id=parent.parent_chunk_id,
                        document_id=document_id,
                        lightrag_full_doc_id=full_doc_id,
                        lightrag_file_path=file_path,
                        lightrag_chunk_order_index=order_index,
                        parent_order_index=parent.order_index,
                        content_hash=content_hash,
                        parent_content_hash=parent.content_hash,
                        overlap_chars=overlap_chars,
                        document_char_start=document_char_start,
                        document_char_end=document_char_end,
                        canonical_method=canonical.method,
                        mapping_method=mapping_method,
                        mapping_score=score,
                        mapped_at=mapped_at,
                    )
                )
            mapped_chunk_ids.add(chunk_id)

        # A sync is a replacement, not an append.  Even a completely invalid
        # record set clears stale mappings so subsequent retrieval fails closed.
        connection.execute(
            """
            DELETE FROM lightrag_chunk_parent_provenance
            WHERE document_id = ?
            """,
            (document_id,),
        )
        if mapping_rows:
            connection.executemany(
                """
                INSERT INTO lightrag_chunk_parent_provenance (
                    lightrag_chunk_id,
                    parent_chunk_id,
                    document_id,
                    lightrag_full_doc_id,
                    lightrag_file_path,
                    lightrag_chunk_order_index,
                    parent_order_index,
                    content_hash,
                    parent_content_hash,
                    overlap_chars,
                    document_char_start,
                    document_char_end,
                    canonical_method,
                    mapping_method,
                    mapping_score,
                    mapped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.lightrag_chunk_id,
                        row.parent_chunk_id,
                        row.document_id,
                        row.lightrag_full_doc_id,
                        row.lightrag_file_path,
                        row.lightrag_chunk_order_index,
                        row.parent_order_index,
                        row.content_hash,
                        row.parent_content_hash,
                        row.overlap_chars,
                        row.document_char_start,
                        row.document_char_end,
                        row.canonical_method,
                        row.mapping_method,
                        row.mapping_score,
                        row.mapped_at,
                    )
                    for row in mapping_rows
                ],
            )

    rejected = sum(rejection_reasons.values())
    return ProvenanceSyncResult(
        document_id=document_id,
        records_seen=len(record_list),
        records_mapped=len(mapped_chunk_ids),
        mappings_written=len(mapping_rows),
        records_rejected=rejected,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )


async def sync_document_provenance(
    db_path: Path,
    rag: Any,
    document_id: str,
) -> ProvenanceSyncResult:
    """Read one processed LightRAG document's chunks and rebuild locally.

    Only KV/status reads are made.  This function never invokes an LLM,
    embedding function, query pipeline, or other provider-backed operation.
    """

    status_storage = getattr(rag, "doc_status", None)
    text_chunks = getattr(rag, "text_chunks", None)
    if status_storage is None or text_chunks is None:
        sync_document_chunk_records(db_path, document_id, [])
        raise ProvenanceSyncError(
            f"lightrag_provenance_storage_unavailable:{document_id}"
        )

    status = await status_storage.get_by_id(document_id)
    status_value = _status_value(status)
    if status_value != "processed":
        sync_document_chunk_records(db_path, document_id, [])
        raise ProvenanceSyncError(
            f"lightrag_provenance_document_not_processed:{document_id}:{status_value}"
        )
    chunk_ids_raw = _field(status, "chunks_list")
    chunk_ids = _unique_nonempty_strings(chunk_ids_raw)
    if not chunk_ids:
        sync_document_chunk_records(db_path, document_id, [])
        raise ProvenanceSyncError(
            f"lightrag_provenance_chunks_missing:{document_id}"
        )

    loaded = await text_chunks.get_by_ids(chunk_ids)
    records = _records_for_chunk_ids(chunk_ids, loaded)
    return sync_document_chunk_records(db_path, document_id, records)


def resolve_lightrag_chunk_parents(
    db_path: Path,
    lightrag_chunk_ids: Sequence[str],
    *,
    allowed_document_ids: Sequence[str] | None = None,
) -> dict[str, list[ResolvedParentPassage]]:
    """Resolve durable mappings and verify current parent hashes.

    Unknown, stale, or out-of-scope mappings return no passage.  The allowed
    document set is only a filter; it is never used to infer provenance.
    """

    return resolve_lightrag_chunk_parent_status(
        db_path,
        lightrag_chunk_ids,
        allowed_document_ids=allowed_document_ids,
    ).parents_by_chunk_id


def resolve_lightrag_chunk_parent_status(
    db_path: Path,
    lightrag_chunk_ids: Sequence[str],
    *,
    allowed_document_ids: Sequence[str] | None = None,
) -> ProvenanceResolution:
    """Resolve parents while distinguishing unknown, stale, and scoped rows."""

    chunk_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in lightrag_chunk_ids
            if str(value).strip()
        )
    )
    resolved: dict[str, list[ResolvedParentPassage]] = {
        chunk_id: [] for chunk_id in chunk_ids
    }
    if not chunk_ids:
        return ProvenanceResolution(
            parents_by_chunk_id=resolved,
            known_chunk_ids=frozenset(),
            stale_chunk_ids=frozenset(),
            scoped_out_chunk_ids=frozenset(),
        )
    allowed = (
        {str(value) for value in allowed_document_ids}
        if allowed_document_ids is not None
        else None
    )

    rows: list[Any] = []
    with connect(db_path) as connection:
        for batch_start in range(0, len(chunk_ids), 400):
            batch = chunk_ids[batch_start : batch_start + 400]
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    f"""
                    SELECT
                        p.*,
                        d.source_path,
                        d.filename
                    FROM lightrag_chunk_parent_provenance AS p
                    JOIN documents AS d ON d.id = p.document_id
                    WHERE p.lightrag_chunk_id IN ({placeholders})
                    ORDER BY
                        p.lightrag_chunk_id,
                        p.parent_order_index,
                        p.parent_chunk_id
                    """,
                    list(batch),
                ).fetchall()
            )

        parent_cache: dict[str, dict[str, _ParentSpan]] = {}
        known_chunk_ids = {
            str(row["lightrag_chunk_id"]) for row in rows
        }
        valid_unscoped_chunk_ids: set[str] = set()
        for row in rows:
            canonical_document_id = str(row["document_id"])
            if canonical_document_id not in parent_cache:
                current_parents, _document_text = _load_parent_sequence(
                    connection,
                    canonical_document_id,
                )
                parent_cache[canonical_document_id] = {
                    parent.parent_chunk_id: parent for parent in current_parents
                }
            current_parent = parent_cache[canonical_document_id].get(
                str(row["parent_chunk_id"])
            )
            if current_parent is None:
                continue
            if current_parent.content_hash != str(row["parent_content_hash"]):
                continue
            chunk_id = str(row["lightrag_chunk_id"])
            valid_unscoped_chunk_ids.add(chunk_id)
            if allowed is not None and canonical_document_id not in allowed:
                continue
            resolved[chunk_id].append(
                ResolvedParentPassage(
                    lightrag_chunk_id=chunk_id,
                    document_id=canonical_document_id,
                    parent_chunk_id=current_parent.parent_chunk_id,
                    parent_content=current_parent.content,
                    source_path=str(row["source_path"] or ""),
                    filename=str(row["filename"] or ""),
                    lightrag_full_doc_id=str(row["lightrag_full_doc_id"] or ""),
                    lightrag_file_path=str(row["lightrag_file_path"] or ""),
                    lightrag_chunk_order_index=int(
                        row["lightrag_chunk_order_index"]
                    ),
                    parent_order_index=current_parent.order_index,
                    content_hash=str(row["content_hash"]),
                    parent_content_hash=current_parent.content_hash,
                    overlap_chars=int(row["overlap_chars"]),
                    canonical_method=str(row["canonical_method"]),
                    mapping_method=str(row["mapping_method"]),
                    mapping_score=float(row["mapping_score"]),
                    document_char_start=int(row["document_char_start"]),
                    document_char_end=int(row["document_char_end"]),
                    mapped_at=str(row["mapped_at"]),
                    page_number=current_parent.page_number,
                    section_title=current_parent.section_title,
                    heading_path=current_parent.heading_path,
                    chunk_type=current_parent.chunk_type,
                )
            )
        for parents in resolved.values():
            parents.sort(
                key=lambda parent: (
                    -parent.mapping_score,
                    -parent.overlap_chars,
                    parent.parent_order_index,
                    parent.parent_chunk_id,
                )
            )
    returned_chunk_ids = {
        chunk_id for chunk_id, parents in resolved.items() if parents
    }
    return ProvenanceResolution(
        parents_by_chunk_id=resolved,
        known_chunk_ids=frozenset(known_chunk_ids),
        stale_chunk_ids=frozenset(
            known_chunk_ids - valid_unscoped_chunk_ids
        ),
        scoped_out_chunk_ids=frozenset(
            valid_unscoped_chunk_ids - returned_chunk_ids
        ),
    )


def resolve_lightrag_chunk_provenance(
    db_path: Path,
    lightrag_chunk_ids: Sequence[str],
    *,
    allowed_document_ids: Sequence[str] | None = None,
) -> dict[str, list[ResolvedParentPassage]]:
    """Compatibility name for bridge callers."""

    return resolve_lightrag_chunk_parents(
        db_path,
        lightrag_chunk_ids,
        allowed_document_ids=allowed_document_ids,
    )


def _load_parent_sequence(
    connection: Any,
    document_id: str,
) -> tuple[list[_ParentSpan], str]:
    rows = connection.execute(
        """
        SELECT
            parent_chunk_id,
            metadata_json,
            page_number,
            heading_path_json,
            chunk_type
        FROM chunks
        WHERE document_id = ?
        ORDER BY chunk_index ASC
        """,
        (document_id,),
    ).fetchall()
    ordered_ids: list[str] = []
    contents: dict[str, str] = {}
    attributes: dict[str, dict[str, Any]] = {}
    invalid_ids: set[str] = set()
    for row in rows:
        parent_chunk_id = str(row["parent_chunk_id"] or "").strip()
        if not parent_chunk_id:
            continue
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            invalid_ids.add(parent_chunk_id)
            continue
        if not isinstance(metadata, Mapping):
            invalid_ids.add(parent_chunk_id)
            continue
        parent_content = metadata.get("parent_content")
        if not isinstance(parent_content, str) or not parent_content.strip():
            invalid_ids.add(parent_chunk_id)
            continue
        parent_content = parent_content.strip()
        section_title = str(metadata.get("section_title") or "").strip() or None
        page_number = row["page_number"]
        if page_number is None:
            page_number = metadata.get("page_number")
        try:
            page_number = int(page_number) if page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        try:
            heading_value = json.loads(row["heading_path_json"] or "[]")
        except (TypeError, ValueError):
            heading_value = []
        heading_path = tuple(
            str(value).strip()
            for value in heading_value
            if str(value or "").strip()
        ) if isinstance(heading_value, list) else ()
        chunk_type = str(row["chunk_type"] or "text")
        if parent_chunk_id not in contents:
            ordered_ids.append(parent_chunk_id)
            contents[parent_chunk_id] = parent_content
            attributes[parent_chunk_id] = {
                "page_number": page_number,
                "section_title": section_title,
                "heading_path": heading_path,
                "chunk_type": chunk_type,
            }
        elif contents[parent_chunk_id] != parent_content:
            invalid_ids.add(parent_chunk_id)
        else:
            parent_attributes = attributes[parent_chunk_id]
            if parent_attributes["page_number"] is None and page_number is not None:
                parent_attributes["page_number"] = page_number
            if (
                parent_attributes["section_title"] is None
                and section_title is not None
            ):
                parent_attributes["section_title"] = section_title
            if not parent_attributes["heading_path"] and heading_path:
                parent_attributes["heading_path"] = heading_path

    ordered_ids = [value for value in ordered_ids if value not in invalid_ids]
    parents: list[_ParentSpan] = []
    parts: list[str] = []
    cursor = 0
    for order_index, parent_chunk_id in enumerate(ordered_ids):
        content = contents[parent_chunk_id]
        parent_attributes = attributes[parent_chunk_id]
        if parts:
            cursor += len(_PARENT_SEPARATOR)
        start = cursor
        end = start + len(content)
        parents.append(
            _ParentSpan(
                parent_chunk_id=parent_chunk_id,
                content=content,
                order_index=order_index,
                start=start,
                end=end,
                content_hash=_sha256(content),
                page_number=parent_attributes["page_number"],
                section_title=parent_attributes["section_title"],
                heading_path=parent_attributes["heading_path"],
                chunk_type=parent_attributes["chunk_type"],
            )
        )
        parts.append(content)
        cursor = end
    return parents, _PARENT_SEPARATOR.join(parts)


def _resolve_canonical_document(
    documents: Mapping[str, _CanonicalDocument],
    *,
    full_doc_id: str,
    file_path: str,
) -> tuple[_CanonicalResolution | None, str | None]:
    by_id = documents.get(full_doc_id) if full_doc_id else None
    path_candidates: set[str] = set()
    path_method = ""
    if file_path and "<SEP>" not in file_path:
        exact_source = {
            document.document_id
            for document in documents.values()
            if document.source_path == file_path
        }
        if exact_source:
            path_candidates = exact_source
            path_method = "exact_source_path"
        elif Path(file_path).name == file_path:
            exact_filename = {
                document.document_id
                for document in documents.values()
                if document.filename == file_path
                or Path(document.source_path).name == file_path
            }
            if exact_filename:
                path_candidates = exact_filename
                path_method = "exact_filename"

    if by_id is not None:
        # A canonical storage ID is authoritative.  A uniquely resolved,
        # contradictory path indicates corruption; an ambiguous basename does
        # not invalidate the exact ID and is never used to relabel it.
        if path_candidates and by_id.document_id not in path_candidates:
            return None, "document_provenance_conflict"
        return _CanonicalResolution(by_id, "full_doc_id", 1.0), None
    if len(path_candidates) == 1:
        document = documents[next(iter(path_candidates))]
        score = 0.99 if path_method == "exact_source_path" else 0.97
        return _CanonicalResolution(document, path_method, score), None
    if len(path_candidates) > 1:
        return None, "ambiguous_exact_path"
    return None, "unresolved_document"


def _resolve_content_offset(
    document_text: str,
    chunk_content: str,
) -> _ContentResolution | None:
    content = chunk_content.strip()
    exact_positions = _unique_occurrence(document_text, content)
    if exact_positions is not None:
        start, end = exact_positions
        return _ContentResolution(start, end, "exact_offset", 1.0)

    normalized_document, positions = _normalize_whitespace_with_positions(
        document_text
    )
    normalized_content, _unused = _normalize_whitespace_with_positions(content)
    if not normalized_content:
        return None
    normalized_match = _unique_occurrence(
        normalized_document,
        normalized_content,
    )
    if normalized_match is None:
        return None
    normalized_start, normalized_end = normalized_match
    original_start = positions[normalized_start]
    original_end = positions[normalized_end - 1] + 1
    return _ContentResolution(
        original_start,
        original_end,
        "normalized_whitespace_unique",
        0.95,
    )


def _resolve_parent_by_shingles(
    parents: Sequence[_ParentSpan],
    chunk_content: str,
) -> tuple[_ParentSpan, float] | None:
    source_shingles = _token_shingles(chunk_content)
    if not source_shingles:
        return None
    scored: list[tuple[float, int, _ParentSpan]] = []
    for parent in parents:
        parent_shingles = _token_shingles(parent.content)
        denominator = min(len(source_shingles), len(parent_shingles))
        score = (
            len(source_shingles & parent_shingles) / denominator
            if denominator
            else 0.0
        )
        scored.append((score, -parent.order_index, parent))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    top_score, _order, top_parent = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if top_score < _SHINGLE_MIN_SCORE:
        return None
    if top_score - second_score < _SHINGLE_MIN_MARGIN:
        return None
    return top_parent, top_score


def _token_shingles(value: str) -> set[tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = _WORD_PATTERN.findall(normalized)
    if len(tokens) < _SHINGLE_SIZE:
        return set()
    return {
        tuple(tokens[index : index + _SHINGLE_SIZE])
        for index in range(len(tokens) - _SHINGLE_SIZE + 1)
    }


def _unique_occurrence(haystack: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    first = haystack.find(needle)
    if first < 0:
        return None
    if haystack.find(needle, first + 1) >= 0:
        return None
    return first, first + len(needle)


def _normalize_whitespace_with_positions(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    whitespace_pending = False
    whitespace_position = 0
    for index, character in enumerate(value):
        if character.isspace():
            if normalized and not whitespace_pending:
                whitespace_position = index
            whitespace_pending = bool(normalized)
            continue
        if whitespace_pending:
            normalized.append(" ")
            positions.append(whitespace_position)
            whitespace_pending = False
        normalized.append(character)
        positions.append(index)
    return "".join(normalized), positions


def _coerce_records(records: ChunkRecordInput) -> list[dict[str, Any]]:
    if isinstance(records, Mapping):
        output: list[dict[str, Any]] = []
        for chunk_id, value in records.items():
            record = dict(value)
            record.setdefault("_id", str(chunk_id))
            output.append(record)
        return output
    return [dict(record) for record in records]


def _records_for_chunk_ids(
    chunk_ids: list[str],
    loaded: Any,
) -> list[dict[str, Any]]:
    if isinstance(loaded, Mapping):
        records: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            record = dict(loaded.get(chunk_id) or {})
            record["_id"] = chunk_id
            records.append(record)
        return records
    values = list(loaded or [])
    records: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(chunk_ids):
        value = values[index] if index < len(values) else None
        record = dict(value or {})
        record.setdefault("_id", chunk_id)
        records.append(record)
    return records


def _unique_nonempty_strings(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value or "").strip()
        )
    )


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _status_value(value: Any) -> str:
    status = _field(value, "status")
    raw = getattr(status, "value", status)
    return str(raw or "").lower()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
