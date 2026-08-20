"""Canonical, provenance-backed evidence cards for research papers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
import uuid

from app.db.sqlite import connect
from app.rag.paper_facets import ALL_PAPER_FACETS, CORE_PAPER_FACETS, normalize_facets
from app.services.evidence_validator import validate_answer_claims


EVIDENCE_CARD_SCHEMA_VERSION = "v1"
EVIDENCE_CARD_PROMPT_VERSION = "v2"
EVIDENCE_CARD_PROVIDER = "openai_compatible"


@dataclass(frozen=True)
class EvidenceRefDraft:
    source_kind: str
    source_id: str
    quote: str
    page: int | None = None
    section_title: str | None = None
    source_content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItemDraft:
    claim_text: str
    evidence_refs: list[EvidenceRefDraft]


@dataclass(frozen=True)
class EvidenceFacetDraft:
    facet: str
    synopsis: str
    items: list[EvidenceItemDraft]
    status: str = "complete"
    confidence: float = 0.0


@dataclass(frozen=True)
class PaperEvidenceDraft:
    document_id: str
    facets: list[EvidenceFacetDraft]
    generator_model: str
    generator_provider: str = EVIDENCE_CARD_PROVIDER
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CardCoverage:
    document_id: str
    requested_facets: list[str]
    covered_facets: list[str]
    missing_facets: list[str]
    stale: bool
    status: str


class PaperEvidenceService:
    def __init__(
        self,
        db_path: Path,
        *,
        schema_version: str = EVIDENCE_CARD_SCHEMA_VERSION,
        prompt_version: str = EVIDENCE_CARD_PROMPT_VERSION,
    ) -> None:
        self.db_path = db_path
        self.schema_version = schema_version
        self.prompt_version = prompt_version

    def list_document_ids(self, *, limit: int | None = None) -> list[str]:
        sql = "SELECT id FROM documents ORDER BY indexed_at, id"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(0, int(limit)),)
        with connect(self.db_path) as connection:
            return [str(row["id"]) for row in connection.execute(sql, params).fetchall()]

    def get_status(self) -> dict[str, Any]:
        with connect(self.db_path) as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            rows = connection.execute(
                "SELECT status, COUNT(*) count FROM paper_evidence_cards GROUP BY status"
            ).fetchall()
            jobs = connection.execute(
                "SELECT status, COUNT(*) count FROM paper_evidence_build_jobs GROUP BY status"
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "documents": total,
            "cards": sum(counts.values()),
            "card_status": counts,
            "job_status": {str(row["status"]): int(row["count"]) for row in jobs},
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
        }

    def card_for_document(self, document_id: str) -> dict[str, Any] | None:
        cards = self.load_cards([document_id])
        return cards[0] if cards else None

    def load_cards(
        self,
        document_ids: Iterable[str],
        *,
        requested_facets: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Batch-load cards while preserving canonical scope order."""

        ordered_ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        requested = normalize_facets(requested_facets or ALL_PAPER_FACETS)
        with connect(self.db_path) as connection:
            document_rows = connection.execute(
                f"""
                SELECT id, filename, title, content_hash, parser_name, parser_version
                FROM documents WHERE id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()
            card_rows = connection.execute(
                f"SELECT * FROM paper_evidence_cards WHERE document_id IN ({placeholders})",
                ordered_ids,
            ).fetchall()
            card_by_doc = {str(row["document_id"]): dict(row) for row in card_rows}
            docs = {str(row["id"]): dict(row) for row in document_rows}
            card_ids = [str(row["id"]) for row in card_rows]
            facet_rows: list[Any] = []
            item_rows: list[Any] = []
            if card_ids:
                card_placeholders = ",".join("?" for _ in card_ids)
                facet_rows = connection.execute(
                    f"SELECT * FROM paper_evidence_facets WHERE card_id IN ({card_placeholders}) ORDER BY facet",
                    card_ids,
                ).fetchall()
                facet_ids = [str(row["id"]) for row in facet_rows]
                if facet_ids:
                    facet_placeholders = ",".join("?" for _ in facet_ids)
                    item_rows = connection.execute(
                        f"SELECT * FROM paper_evidence_items WHERE facet_id IN ({facet_placeholders}) ORDER BY ordinal",
                        facet_ids,
                    ).fetchall()

        items_by_facet: dict[str, list[dict[str, Any]]] = {}
        for row in item_rows:
            decoded = dict(row)
            decoded["evidence_refs"] = _loads_json(decoded.pop("evidence_refs_json", "[]"), [])
            decoded["metadata"] = _loads_json(decoded.pop("metadata_json", "{}"), {})
            items_by_facet.setdefault(str(row["facet_id"]), []).append(decoded)
        facets_by_card: dict[str, list[dict[str, Any]]] = {}
        for row in facet_rows:
            decoded = dict(row)
            decoded["metadata"] = _loads_json(decoded.pop("metadata_json", "{}"), {})
            decoded["items"] = items_by_facet.get(str(row["id"]), [])
            facets_by_card.setdefault(str(row["card_id"]), []).append(decoded)

        result: list[dict[str, Any]] = []
        for document_id in ordered_ids:
            document = docs.get(document_id)
            card = card_by_doc.get(document_id)
            if document is None or card is None:
                continue
            stale_reasons = self._stale_reasons(card, document)
            decoded = dict(card)
            decoded["filename"] = document.get("filename")
            decoded["title"] = document.get("title")
            decoded["coverage"] = _loads_json(decoded.pop("coverage_json", "{}"), {})
            decoded["metadata"] = _loads_json(decoded.pop("metadata_json", "{}"), {})
            decoded["facets"] = [
                facet
                for facet in facets_by_card.get(str(card["id"]), [])
                if str(facet.get("facet")) in requested
            ]
            decoded["stale"] = bool(stale_reasons)
            decoded["stale_reasons"] = stale_reasons
            result.append(decoded)
        return result

    def coverage_matrix(
        self,
        document_ids: Iterable[str],
        requested_facets: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[CardCoverage]]:
        ordered_ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
        requested = normalize_facets(requested_facets)
        cards = self.load_cards(ordered_ids, requested_facets=requested)
        by_document = {str(card["document_id"]): card for card in cards}
        matrix: list[CardCoverage] = []
        for document_id in ordered_ids:
            card = by_document.get(document_id)
            usable = []
            if card and not card.get("stale") and card.get("status") in {"complete", "partial"}:
                usable = [
                    str(facet["facet"])
                    for facet in card.get("facets") or []
                    if facet.get("status") in {"complete", "partial"} and facet.get("items")
                ]
            missing = [facet for facet in requested if facet not in usable]
            matrix.append(
                CardCoverage(
                    document_id=document_id,
                    requested_facets=requested,
                    covered_facets=usable,
                    missing_facets=missing,
                    stale=bool(card and card.get("stale")),
                    status=str(card.get("status") if card else "missing"),
                )
            )
        return cards, matrix

    def render_navigation_context(
        self,
        cards: Iterable[dict[str, Any]],
        *,
        requested_facets: Iterable[str],
    ) -> str:
        requested = set(normalize_facets(requested_facets))
        blocks: list[str] = []
        for card in cards:
            if card.get("stale"):
                continue
            lines = [
                f"PAPER CARD: {card.get('filename') or card.get('document_id')}",
                f"document_id: {card.get('document_id')}",
            ]
            for facet in card.get("facets") or []:
                name = str(facet.get("facet") or "")
                if name not in requested or not facet.get("items"):
                    continue
                refs = []
                for item in facet.get("items") or []:
                    for ref in item.get("evidence_refs") or []:
                        ref_id = str(ref.get("source_id") or "")
                        if ref_id and ref_id not in refs:
                            refs.append(ref_id)
                lines.append(
                    f"- {name}: {str(facet.get('synopsis') or '').strip()} "
                    f"[canonical refs: {', '.join(refs)}]"
                )
            if len(lines) > 2:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def materialize_sources(
        self,
        cards: Iterable[dict[str, Any]],
        *,
        requested_facets: Iterable[str],
    ) -> list[dict[str, Any]]:
        """Resolve card refs back to canonical retrieval sources.

        The synopsis is deliberately not copied into source content.  Numeric
        validation therefore cannot circularly validate an LLM-generated card.
        """

        requested = set(normalize_facets(requested_facets))
        refs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        facets_by_key: dict[tuple[str, str], list[str]] = {}
        for card in cards:
            if card.get("stale"):
                continue
            for facet in card.get("facets") or []:
                name = str(facet.get("facet") or "")
                if name not in requested:
                    continue
                for item in facet.get("items") or []:
                    for ref in item.get("evidence_refs") or []:
                        key = (str(ref.get("source_kind") or ""), str(ref.get("source_id") or ""))
                        if not all(key):
                            continue
                        refs_by_key[key] = ref
                        facets_by_key.setdefault(key, [])
                        if name not in facets_by_key[key]:
                            facets_by_key[key].append(name)
        if not refs_by_key:
            return []
        with connect(self.db_path) as connection:
            sources = [
                self._source_record(connection, kind, source_id)
                for kind, source_id in refs_by_key
            ]
        materialized: list[dict[str, Any]] = []
        for key, source in zip(refs_by_key, sources, strict=True):
            if source is None:
                continue
            source["evidence_facets"] = facets_by_key[key]
            source["retrieval_channels"] = ["paper_evidence_card"]
            source["card_evidence"] = True
            materialized.append(source)
        return materialized

    def publish(self, draft: PaperEvidenceDraft) -> dict[str, Any]:
        """Validate a complete in-memory draft then atomically replace the card."""

        if draft.generator_model.strip() == "":
            raise ValueError("generator_model is required")
        with connect(self.db_path) as connection:
            document_row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (draft.document_id,)
            ).fetchone()
            if document_row is None:
                raise ValueError(f"Unknown document_id: {draft.document_id}")
            document = dict(document_row)
            validated_facets = self._validated_facets(
                connection,
                document=document,
                facets=draft.facets,
            )
            if not validated_facets:
                raise ValueError("Evidence card contains no valid canonical evidence")
            now = _utc_now()
            card_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paper-evidence:{draft.document_id}:{self.schema_version}"))
            complete_count = sum(facet["status"] == "complete" for facet in validated_facets)
            status = "complete" if complete_count == len(CORE_PAPER_FACETS) else "partial"
            coverage = {
                facet: next(
                    (item["status"] for item in validated_facets if item["facet"] == facet),
                    "unavailable",
                )
                for facet in ALL_PAPER_FACETS
            }
            # The old card stays queryable until every new row has passed
            # validation; replacement itself is one SQLite transaction.
            connection.execute("DELETE FROM paper_evidence_cards WHERE document_id = ?", (draft.document_id,))
            connection.execute(
                """
                INSERT INTO paper_evidence_cards (
                    id, document_id, document_content_hash, parser_name, parser_version,
                    schema_version, prompt_version, generator_provider, generator_model,
                    status, coverage_json, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    draft.document_id,
                    document["content_hash"],
                    document.get("parser_name"),
                    document.get("parser_version"),
                    self.schema_version,
                    self.prompt_version,
                    draft.generator_provider,
                    draft.generator_model,
                    status,
                    json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                    json.dumps(draft.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            for facet in validated_facets:
                facet_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{card_id}:{facet['facet']}"))
                connection.execute(
                    """
                    INSERT INTO paper_evidence_facets (
                        id, card_id, document_id, facet, synopsis, status, confidence,
                        source_count, facet_hash, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        facet_id,
                        card_id,
                        draft.document_id,
                        facet["facet"],
                        facet["synopsis"],
                        facet["status"],
                        facet["confidence"],
                        facet["source_count"],
                        facet["facet_hash"],
                        now,
                        now,
                        "{}",
                    ),
                )
                for ordinal, item in enumerate(facet["items"]):
                    item_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{facet_id}:{ordinal}"))
                    connection.execute(
                        """
                        INSERT INTO paper_evidence_items (
                            id, facet_id, document_id, ordinal, claim_text,
                            evidence_refs_json, validation_status, validation_reason,
                            created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, 'valid', ?, ?, '{}')
                        """,
                        (
                            item_id,
                            facet_id,
                            draft.document_id,
                            ordinal,
                            item["claim_text"],
                            json.dumps(item["evidence_refs"], ensure_ascii=False, sort_keys=True),
                            item["validation_reason"],
                            now,
                        ),
                    )
            connection.execute(
                """
                UPDATE paper_evidence_build_jobs
                SET status = 'complete', finished_at = ?, error_message = NULL
                WHERE document_id = ?
                """,
                (now, draft.document_id),
            )
        return self.card_for_document(draft.document_id) or {}

    def mark_job(self, document_id: str, status: str, *, model: str, error: str | None = None) -> None:
        now = _utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO paper_evidence_build_jobs (
                    document_id, status, attempt_count, requested_at, started_at,
                    finished_at, error_message, schema_version, prompt_version,
                    generator_model, metadata_json
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, '{}')
                ON CONFLICT(document_id) DO UPDATE SET
                    status = excluded.status,
                    attempt_count = paper_evidence_build_jobs.attempt_count + 1,
                    requested_at = excluded.requested_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    error_message = excluded.error_message,
                    schema_version = excluded.schema_version,
                    prompt_version = excluded.prompt_version,
                    generator_model = excluded.generator_model
                """,
                (
                    document_id,
                    status,
                    now,
                    now if status == "building" else None,
                    now if status in {"failed", "complete"} else None,
                    (error or "")[:1000] or None,
                    self.schema_version,
                    self.prompt_version,
                    model,
                ),
            )

    def candidate_sources(self, document_id: str, *, per_facet: int = 5) -> list[dict[str, Any]]:
        """Select a small diverse canonical candidate pool without an LLM."""

        with connect(self.db_path) as connection:
            document = connection.execute(
                "SELECT id, filename, title FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                raise ValueError(f"Unknown document_id: {document_id}")
            chunks = [dict(row) for row in connection.execute(
                """
                SELECT id, document_id, content, page_number, heading_path_json,
                       chunk_type, metadata_json
                FROM chunks WHERE document_id = ? ORDER BY order_index, chunk_index
                """,
                (document_id,),
            ).fetchall()]
            tables = [dict(row) for row in connection.execute(
                "SELECT * FROM document_tables WHERE document_id = ? ORDER BY table_index",
                (document_id,),
            ).fetchall()]
            figures = [dict(row) for row in connection.execute(
                "SELECT * FROM document_figures WHERE document_id = ? ORDER BY figure_index",
                (document_id,),
            ).fetchall()]

        candidates: list[dict[str, Any]] = []
        for chunk in chunks:
            heading = " / ".join(_loads_json(chunk.get("heading_path_json"), []))
            content = str(chunk.get("content") or "")
            candidates.append(
                _candidate_record(
                    kind="chunk",
                    source_id=str(chunk["id"]),
                    document_id=document_id,
                    page=chunk.get("page_number"),
                    label=heading,
                    content=content,
                    metadata=_loads_json(chunk.get("metadata_json"), {}),
                )
            )
        for table in tables:
            content = "\n".join(
                value for value in (str(table.get("caption") or "").strip(), str(table.get("markdown") or "").strip()) if value
            )
            candidates.append(
                _candidate_record(
                    kind="table",
                    source_id=str(table["id"]),
                    document_id=document_id,
                    page=table.get("page_number"),
                    label=str(table.get("caption") or ""),
                    content=content,
                    metadata=_loads_json(table.get("metadata_json"), {}),
                )
            )
        for figure in figures:
            metadata = _loads_json(figure.get("metadata_json"), {})
            quality = str(metadata.get("quality_status") or metadata.get("quality") or "").lower()
            if quality in {"rejected", "needs_review"}:
                continue
            content = "\n".join(
                value
                for value in (
                    str(figure.get("caption") or "").strip(),
                    str(figure.get("visual_summary") or "").strip(),
                )
                if value
            )
            if not content:
                continue
            candidates.append(
                _candidate_record(
                    kind="figure",
                    source_id=str(figure["id"]),
                    document_id=document_id,
                    page=figure.get("page_number"),
                    label=str(figure.get("caption") or ""),
                    content=content,
                    metadata=metadata,
                )
            )

        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for facet in CORE_PAPER_FACETS:
            ranked = sorted(
                candidates,
                key=lambda item: (-_candidate_score(item, facet), item.get("page") or 10_000, item["source_id"]),
            )
            for candidate in ranked[: max(1, per_facet)]:
                key = (candidate["source_kind"], candidate["source_id"])
                if key in seen:
                    continue
                seen.add(key)
                selected.append(candidate)
        return selected[: max(12, per_facet * len(CORE_PAPER_FACETS))]

    def _validated_facets(
        self,
        connection,
        *,
        document: dict[str, Any],
        facets: Iterable[EvidenceFacetDraft],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen_facets: set[str] = set()
        for draft in facets:
            normalized = normalize_facets([draft.facet])
            if not normalized or normalized[0] in seen_facets:
                continue
            facet = normalized[0]
            seen_facets.add(facet)
            valid_items: list[dict[str, Any]] = []
            dropped = 0
            for item in draft.items:
                refs: list[dict[str, Any]] = []
                source_documents: list[dict[str, Any]] = []
                invalid_reason: str | None = None
                for ref in item.evidence_refs:
                    source = self._source_record(connection, ref.source_kind, ref.source_id)
                    if source is None:
                        invalid_reason = "missing_source"
                        break
                    if str(source.get("document_id")) != str(document["id"]):
                        invalid_reason = "cross_document_source"
                        break
                    raw = str(source.get("content") or "")
                    actual_hash = _content_hash(raw)
                    if ref.source_content_hash and ref.source_content_hash != actual_hash:
                        invalid_reason = "source_hash_mismatch"
                        break
                    quote = _normalize_text(ref.quote)
                    if not quote or not _quote_found(ref.quote, raw):
                        invalid_reason = "quote_not_found"
                        break
                    refs.append(
                        {
                            "source_kind": ref.source_kind,
                            "source_id": ref.source_id,
                            "page": ref.page if ref.page is not None else source.get("page_number"),
                            "section_title": ref.section_title,
                            "quote": ref.quote.strip(),
                            "source_content_hash": actual_hash,
                            **ref.metadata,
                        }
                    )
                    source_documents.append(source)
                if invalid_reason or not refs or not item.claim_text.strip():
                    dropped += 1
                    continue
                validation = validate_answer_claims(
                    answer=item.claim_text,
                    documents=source_documents,
                    focus_document_ids=[str(document["id"])],
                )
                if not validation.valid:
                    dropped += 1
                    continue
                valid_items.append(
                    {
                        "claim_text": item.claim_text.strip(),
                        "evidence_refs": refs,
                        "validation_reason": validation.reason,
                    }
                )
            if not valid_items:
                continue
            # Navigation cards are routing hints, but they are still visible to
            # the answer model. Never persist a free-form synopsis containing
            # claims whose item-level evidence was dropped. Canonicalize the
            # synopsis exclusively from claims that survived quote ownership,
            # source freshness and quantitative validation above.
            synopsis = " ".join(item["claim_text"] for item in valid_items)
            status = "partial" if dropped or draft.status != "complete" else "complete"
            digest_payload = json.dumps(
                {"facet": facet, "synopsis": synopsis, "items": valid_items},
                ensure_ascii=False,
                sort_keys=True,
            )
            output.append(
                {
                    "facet": facet,
                    "synopsis": synopsis,
                    "status": status,
                    "confidence": max(0.0, min(float(draft.confidence), 1.0)),
                    "source_count": len({ref["source_id"] for item in valid_items for ref in item["evidence_refs"]}),
                    "facet_hash": hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
                    "items": valid_items,
                }
            )
        return output

    def _source_record(self, connection, source_kind: str, source_id: str) -> dict[str, Any] | None:
        kind = str(source_kind or "").strip().lower()
        if kind == "chunk":
            row = connection.execute(
                """
                SELECT chunks.*, documents.title
                FROM chunks JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id = ?
                """,
                (source_id,),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            item.update(
                {
                    "chunk_id": item["id"],
                    "artifact_type": "text",
                    "content": item.get("content") or "",
                    "page": item.get("page_number"),
                }
            )
            return item
        if kind == "table":
            row = connection.execute(
                """
                SELECT document_tables.*, documents.filename, documents.title
                FROM document_tables JOIN documents ON documents.id = document_tables.document_id
                WHERE document_tables.id = ?
                """,
                (source_id,),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            content = "\n".join(
                value for value in (str(item.get("caption") or "").strip(), str(item.get("markdown") or "").strip()) if value
            )
            item.update(
                {
                    "chunk_id": f"table:{item['id']}",
                    "table_id": item["id"],
                    "artifact_type": "table",
                    "content": content,
                    "page": item.get("page_number"),
                }
            )
            return item
        if kind == "figure":
            row = connection.execute(
                """
                SELECT document_figures.*, documents.filename, documents.title
                FROM document_figures JOIN documents ON documents.id = document_figures.document_id
                WHERE document_figures.id = ?
                """,
                (source_id,),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            content = "\n".join(
                value for value in (str(item.get("caption") or "").strip(), str(item.get("visual_summary") or "").strip()) if value
            )
            item.update(
                {
                    "chunk_id": f"figure:{item['id']}",
                    "figure_id": item["id"],
                    "artifact_type": "figure",
                    "content": content,
                    "page": item.get("page_number"),
                }
            )
            return item
        return None

    def _stale_reasons(self, card: dict[str, Any], document: dict[str, Any]) -> list[str]:
        checks = (
            ("document_content_hash", document.get("content_hash"), "content_hash"),
            ("parser_name", document.get("parser_name"), "parser_name"),
            ("parser_version", document.get("parser_version"), "parser_version"),
            ("schema_version", self.schema_version, "schema_version"),
            ("prompt_version", self.prompt_version, "prompt_version"),
        )
        return [reason for key, expected, reason in checks if str(card.get(key) or "") != str(expected or "")]


def _candidate_record(
    *,
    kind: str,
    source_id: str,
    document_id: str,
    page: int | None,
    label: str,
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized = content.strip()
    return {
        "source_kind": kind,
        "source_id": source_id,
        "document_id": document_id,
        "page": page,
        "label": label.strip(),
        "content": normalized[:5000],
        "source_content_hash": _content_hash(normalized),
        "metadata": metadata,
    }


def _quote_found(quote: str, source: str) -> bool:
    """Match quotes across layout-only PDF spacing/punctuation corruption.

    Docling can emit ``RobustAudio`` where the page visibly contains
    ``Robust Audio`` or split a hyphenated line as ``over- lapping``. Exact
    normalized matching stays authoritative first. The fallback removes only
    non-alphanumeric layout boundaries and requires a substantial quote, so it
    cannot turn a short topical keyword into provenance and still rejects
    reordered/paraphrased text.
    """

    normalized_quote = _normalize_text(quote)
    normalized_source = _normalize_text(source)
    if normalized_quote and normalized_quote in normalized_source:
        return True
    compact_quote = "".join(character.casefold() for character in quote if character.isalnum())
    if len(compact_quote) < 24:
        return False
    compact_source = "".join(
        character.casefold() for character in source if character.isalnum()
    )
    return compact_quote in compact_source


def _candidate_score(candidate: dict[str, Any], facet: str) -> float:
    haystack = f"{candidate.get('label', '')} {candidate.get('content', '')[:1200]}".lower()
    kind = str(candidate.get("source_kind") or "")
    markers = {
        "task": ("abstract", "introduction", "problem", "task", "objective", "we study", "we address"),
        "architecture": ("architecture", "method", "model", "framework", "pipeline", "fusion", "encoder"),
        "dataset_setup": ("dataset", "data", "experimental setup", "corpus", "split", "iemocap", "samples"),
        "benchmark_results": ("result", "performance", "comparison", "accuracy", "f1", "ccc", "table"),
        "contributions": ("contribution", "we propose", "novel", "our work", "conclusion"),
    }.get(facet, (facet.replace("_", " "),))
    score = sum(2.0 for marker in markers if marker in haystack)
    if facet == "benchmark_results" and kind == "table":
        score += 6.0
    if facet == "architecture" and kind == "figure":
        score += 4.0
    if facet in {"task", "contributions"} and (candidate.get("page") or 99) <= 2:
        score += 2.0
    return score


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _content_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _loads_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        decoded = json.loads(value or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    return decoded


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
