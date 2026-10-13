#!/usr/bin/env python3
"""Retrieval-only Evaluation Gate v2 for original/rewrite RRF.

The evaluator deliberately does not instantiate a router, query-rewrite LLM, or
answer model.  Rewrites are frozen in the suite manifest.  This keeps the A0,
A1 and B arms comparable and makes an accidental model call a hard failure.

The file is also importable by unit tests.  The qrel resolver is anchored to
source/content fingerprints and canonical locators rather than runtime UUIDs.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import unicodedata
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.sqlite import connect  # noqa: E402
from app.rag.evidence_identity import canonical_evidence_id, canonical_evidence_ids, evidence_kind  # noqa: E402
from app.rag.fusion import plan_query_variants  # noqa: E402
from app.rag.context import select_ranked_evidence  # noqa: E402
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider  # noqa: E402
from app.rag.reranker import rerank_candidates  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


SCHEMA_VERSION = "retrieval-eval-v2.1"
ARMS = ("A0", "A1", "B-full", "B-anchor")
POSITIVE_CATEGORIES = {
    "text",
    "table_numeric",
    "figure_page",
    "intra_document_multihop",
    "cross_document_compare",
}


class EvaluationContractError(ValueError):
    """Raised when a suite is incomplete, ambiguous, or unsafe to score."""


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(value.split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    """A source-stable locator which can resolve to fresh runtime IDs."""

    group_id: str
    kind: str
    document: dict[str, Any]
    page_number: int | None = None
    section: str | None = None
    quote: str | None = None
    quote_sha256: str | None = None
    caption: str | None = None
    caption_sha256: str | None = None
    artifact_index: int | None = None
    image_sha256: str | None = None
    required: bool = True
    alternatives: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, row: dict[str, Any], index: int) -> "EvidenceLocator":
        kind = str(row.get("kind") or "text").strip().lower()
        if kind not in {"text", "table", "figure", "page"}:
            raise EvaluationContractError(f"unsupported evidence kind: {kind!r}")
        document = row.get("document")
        if not isinstance(document, dict):
            raise EvaluationContractError(f"evidence group {index} has no document locator")
        quote = str(row.get("quote") or "").strip() or None
        caption = str(row.get("caption") or "").strip() or None
        alternatives = row.get("alternatives") or []
        if not isinstance(alternatives, list) or not all(isinstance(item, dict) for item in alternatives):
            raise EvaluationContractError(f"evidence group {index} has invalid alternatives")
        return cls(
            group_id=str(row.get("group_id") or f"group-{index + 1}"),
            kind=kind,
            document=document,
            page_number=(int(row["page_number"]) if row.get("page_number") is not None else None),
            section=str(row.get("section") or "").strip() or None,
            quote=quote,
            quote_sha256=str(row.get("quote_sha256") or "").strip() or (sha256_text(quote) if quote else None),
            caption=caption,
            caption_sha256=str(row.get("caption_sha256") or "").strip() or (sha256_text(caption) if caption else None),
            artifact_index=(int(row["artifact_index"]) if row.get("artifact_index") is not None else None),
            image_sha256=str(row.get("image_sha256") or "").strip() or None,
            required=bool(row.get("required", True)),
            alternatives=tuple(dict(item) for item in alternatives),
        )


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    category: str
    language: str
    raw_query: str
    enriched_query: str
    standalone_query: str
    standalone_full: str
    standalone_anchor: str
    required_entities: tuple[str, ...]
    conversation_context: tuple[dict[str, str], ...]
    working_state: dict[str, Any]
    rewrite_provenance: dict[str, tuple[dict[str, str], ...]]
    scope: str
    focus_documents: tuple[dict[str, Any], ...]
    evidence_groups: tuple[EvidenceLocator, ...]
    forbidden_documents: tuple[dict[str, Any], ...]
    expected_abstention: bool = False
    rewrite_fingerprint: str | None = None
    curation_status: str = "draft"
    language_reviewed: bool = False
    qrels_reviewed: bool = False
    audit_status: str = "pending"
    audit_reviewer: str | None = None
    audit_reviewed_at: str | None = None
    audit_notes: str | None = None

    @property
    def positive(self) -> bool:
        return self.category in POSITIVE_CATEGORIES and bool(self.evidence_groups)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EvalCase":
        required = row.get("required_entities") or []
        context = row.get("conversation_context") or []
        if not isinstance(context, list) or not all(isinstance(item, dict) for item in context):
            raise EvaluationContractError(f"{row.get('case_id')}: invalid conversation_context")
        evidence = row.get("evidence_groups") or []
        standalone_full = str(
            row.get("standalone_full") or row.get("enriched_query") or row.get("standalone_query") or ""
        ).strip()
        standalone_anchor = str(row.get("standalone_anchor") or standalone_full).strip()
        provenance_raw = row.get("rewrite_provenance") or {}
        if not isinstance(provenance_raw, dict):
            raise EvaluationContractError(f"{row.get('case_id')}: invalid rewrite_provenance")
        provenance = {
            name: tuple(item for item in values if isinstance(item, dict))
            for name, values in provenance_raw.items()
            if isinstance(values, list)
        }
        audit = row.get("audit") or {}
        if not isinstance(audit, dict):
            raise EvaluationContractError(f"{row.get('case_id')}: invalid audit")
        return cls(
            case_id=str(row.get("case_id") or "").strip(),
            category=str(row.get("category") or "").strip(),
            language=str(row.get("language") or "").strip().lower(),
            raw_query=str(row.get("raw_query") or row.get("message") or "").strip(),
            enriched_query=standalone_full,
            standalone_query=standalone_full,
            standalone_full=standalone_full,
            standalone_anchor=standalone_anchor,
            required_entities=tuple(str(item) for item in required if str(item).strip()),
            conversation_context=tuple(
                {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
                for item in context
            ),
            working_state=dict(row.get("working_state") or {}),
            rewrite_provenance=provenance,
            scope=str(row.get("scope") or "unscoped").strip().lower(),
            focus_documents=tuple(item for item in (row.get("focus_documents") or []) if isinstance(item, dict)),
            evidence_groups=tuple(
                EvidenceLocator.from_dict(item, index)
                for index, item in enumerate(evidence)
                if isinstance(item, dict)
            ),
            forbidden_documents=tuple(item for item in (row.get("forbidden_documents") or []) if isinstance(item, dict)),
            expected_abstention=bool(row.get("expected_abstention", False)),
            rewrite_fingerprint=str(row.get("rewrite_fingerprint") or "").strip() or None,
            curation_status=str(row.get("curation_status") or "draft").strip().lower(),
            language_reviewed=bool(row.get("language_reviewed", False)),
            qrels_reviewed=bool(row.get("qrels_reviewed", False)),
            audit_status=str(audit.get("status") or row.get("audit_status") or "pending").strip().lower(),
            audit_reviewer=str(audit.get("reviewer") or row.get("audit_reviewer") or "").strip() or None,
            audit_reviewed_at=str(audit.get("reviewed_at") or row.get("audit_reviewed_at") or "").strip() or None,
            audit_notes=str(audit.get("notes") or row.get("audit_notes") or "").strip() or None,
        )


@dataclass(frozen=True, slots=True)
class ResolvedGroup:
    locator: EvidenceLocator
    document_id: str
    evidence_ids: frozenset[str] = frozenset()
    page_keys: frozenset[tuple[str, int]] = frozenset()


@dataclass(frozen=True, slots=True)
class ResolvedCase:
    case: EvalCase
    document_ids: frozenset[str]
    focus_document_ids: tuple[str, ...]
    groups: tuple[ResolvedGroup, ...]
    forbidden_document_ids: frozenset[str]


def _document_matches(connection, locator: dict[str, Any]) -> list[dict[str, Any]]:
    filename = str(locator.get("filename") or "").strip()
    source_sha = str(locator.get("source_sha256") or "").strip()
    content_hash = str(locator.get("content_sha256") or locator.get("extracted_content_sha256") or "").strip()
    # Qrels are resolved against the same live logical-document surface that
    # production retrieval exposes.  A retired document (or a document whose
    # staged revision has not been activated) must not make preflight pass and
    # then disappear from the actual search path.
    try:
        rows = connection.execute(
            """
            SELECT id, source_path, filename, content_hash, active_revision_id,
                   revision_status
            FROM documents
            WHERE active_revision_id IS NOT NULL
              AND COALESCE(revision_status, 'active') != 'retired'
            ORDER BY indexed_at DESC
            """
        ).fetchall()
        revision_aware = True
    except sqlite3.OperationalError:
        # Keep the resolver importable against a pre-migration fixture used by
        # older unit tests.  Real benchmark databases always go through
        # ``init_db`` and therefore take the revision-aware branch above.
        rows = connection.execute(
            "SELECT id, source_path, filename, content_hash FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
        revision_aware = False
    matches: list[dict[str, Any]] = []
    for row in rows:
        if filename and str(row["filename"]) != filename:
            continue
        if content_hash and str(row["content_hash"]) != content_hash:
            continue
        if source_sha:
            source = Path(str(row["source_path"] or ""))
            if not source.is_file() or sha256_file(source) != source_sha:
                continue
        item = dict(row)
        if not revision_aware:
            item["active_revision_id"] = "legacy"
            item["revision_status"] = "active"
        matches.append(item)
    return matches


def _section_match(section: str | None, value: str | None) -> bool:
    if not section:
        return True
    return normalize_text(section) in normalize_text(value or "") or normalize_text(value or "") in normalize_text(section)


def _resolve_single_locator(connection, locator: EvidenceLocator) -> ResolvedGroup:
    documents = _document_matches(connection, locator.document)
    if len(documents) != 1:
        raise EvaluationContractError(
            f"{locator.group_id}: expected one document, resolved {len(documents)}"
        )
    document_id = str(documents[0]["id"])
    revision_aware = True
    try:
        document_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        revision_aware = "active_revision_id" in document_columns
    except sqlite3.OperationalError:
        revision_aware = False
    active_revision_id = str(documents[0].get("active_revision_id") or "")
    if revision_aware and not active_revision_id:
        raise EvaluationContractError(
            f"{locator.group_id}: document has no active revision"
        )
    page_keys = frozenset({(document_id, locator.page_number)}) if locator.page_number is not None else frozenset()
    ids: set[str] = set()
    if locator.kind == "text":
        if revision_aware:
            rows = connection.execute(
                """
                SELECT chunks.id, chunks.page_number, chunks.content, chunks.metadata_json
                FROM chunks
                JOIN documents d ON d.id = chunks.document_id
                WHERE chunks.document_id = ?
                  AND chunks.chunk_type = 'text'
                  AND (
                      chunks.revision_id = d.active_revision_id
                      OR (d.active_revision_id IS NULL AND chunks.revision_id IS NULL)
                  )
                ORDER BY chunks.chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, page_number, content, metadata_json
                FROM chunks
                WHERE document_id = ? AND chunk_type = 'text'
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        quote = normalize_text(locator.quote or "")
        for row in rows:
            if locator.page_number is not None and row["page_number"] != locator.page_number:
                continue
            metadata = json.loads(row["metadata_json"] or "{}")
            section_title = metadata.get("section_title")
            if not _section_match(locator.section, section_title):
                continue
            content = normalize_text(row["content"] or "")
            if quote and quote not in content:
                continue
            if quote or not locator.section and locator.page_number is None:
                ids.add(canonical_evidence_id({"chunk_id": str(row["id"]), "chunk_type": "text"}))
        if not ids:
            raise EvaluationContractError(f"{locator.group_id}: text span did not resolve")
    elif locator.kind == "table":
        if revision_aware:
            rows = connection.execute(
                """
                SELECT document_tables.id, document_tables.page_number,
                       document_tables.table_index, document_tables.caption,
                       document_tables.markdown
                FROM document_tables
                JOIN documents d ON d.id = document_tables.document_id
                WHERE document_tables.document_id = ?
                  AND (
                      document_tables.revision_id = d.active_revision_id
                      OR (d.active_revision_id IS NULL AND document_tables.revision_id IS NULL)
                  )
                ORDER BY table_index ASC
                """,
                (document_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, page_number, table_index, caption, markdown
                FROM document_tables WHERE document_id = ?
                ORDER BY table_index ASC
                """,
                (document_id,),
            ).fetchall()
        for row in rows:
            if locator.page_number is not None and row["page_number"] != locator.page_number:
                continue
            if locator.artifact_index is not None and row["table_index"] != locator.artifact_index:
                continue
            if locator.caption_sha256 and sha256_text(str(row["caption"] or "")) != locator.caption_sha256:
                continue
            if locator.caption and normalize_text(locator.caption) not in normalize_text(row["caption"] or ""):
                continue
            ids.add(canonical_evidence_id({"table_id": str(row["id"]), "artifact_type": "table"}))
        if not ids:
            raise EvaluationContractError(f"{locator.group_id}: table locator did not resolve")
    elif locator.kind == "figure":
        if revision_aware:
            rows = connection.execute(
                """
                SELECT document_figures.id, document_figures.page_number,
                       document_figures.figure_index, document_figures.caption,
                       document_figures.image_path, document_figures.metadata_json
                FROM document_figures
                JOIN documents d ON d.id = document_figures.document_id
                WHERE document_figures.document_id = ?
                  AND (
                      document_figures.revision_id = d.active_revision_id
                      OR (d.active_revision_id IS NULL AND document_figures.revision_id IS NULL)
                  )
                ORDER BY figure_index ASC
                """,
                (document_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, page_number, figure_index, caption, image_path, metadata_json
                FROM document_figures WHERE document_id = ?
                ORDER BY figure_index ASC
                """,
                (document_id,),
            ).fetchall()
        for row in rows:
            if locator.page_number is not None and row["page_number"] != locator.page_number:
                continue
            if locator.artifact_index is not None and row["figure_index"] != locator.artifact_index:
                continue
            if locator.caption_sha256 and sha256_text(str(row["caption"] or "")) != locator.caption_sha256:
                continue
            if locator.caption and normalize_text(locator.caption) not in normalize_text(row["caption"] or ""):
                continue
            if locator.image_sha256:
                metadata = json.loads(row["metadata_json"] or "{}")
                image_hash = str(metadata.get("image_hash") or metadata.get("image_sha256") or "")
                if image_hash != locator.image_sha256:
                    continue
            ids.add(canonical_evidence_id({"figure_id": str(row["id"]), "artifact_type": "figure"}))
        if not ids:
            raise EvaluationContractError(f"{locator.group_id}: figure locator did not resolve")
    else:
        if revision_aware:
            rows = connection.execute(
                """
                SELECT chunks.id
                FROM chunks
                JOIN documents d ON d.id = chunks.document_id
                WHERE chunks.document_id = ?
                  AND chunks.page_number = ?
                  AND (
                      chunks.revision_id = d.active_revision_id
                      OR (d.active_revision_id IS NULL AND chunks.revision_id IS NULL)
                  )
                LIMIT 1
                """,
                (document_id, locator.page_number),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id FROM chunks WHERE document_id = ? AND page_number = ? LIMIT 1",
                (document_id, locator.page_number),
            ).fetchall()
        if not rows:
            if revision_aware:
                rows = connection.execute(
                    """
                    SELECT document_figures.id
                    FROM document_figures
                    JOIN documents d ON d.id = document_figures.document_id
                    WHERE document_figures.document_id = ?
                      AND document_figures.page_number = ?
                      AND (
                          document_figures.revision_id = d.active_revision_id
                          OR (d.active_revision_id IS NULL AND document_figures.revision_id IS NULL)
                      )
                    LIMIT 1
                    """,
                    (document_id, locator.page_number),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id FROM document_figures WHERE document_id = ? AND page_number = ? LIMIT 1",
                    (document_id, locator.page_number),
                ).fetchall()
        if not rows:
            raise EvaluationContractError(f"{locator.group_id}: page locator did not resolve")
        ids.add(canonical_evidence_id({"artifact_type": "page", "document_id": document_id, "page_number": locator.page_number}))
    return ResolvedGroup(locator=locator, document_id=document_id, evidence_ids=frozenset(ids), page_keys=page_keys)


def _resolve_group(connection, locator: EvidenceLocator) -> ResolvedGroup:
    """Resolve a qrel group with one-or-more equivalent artifact locators.

    Alternatives are intentionally bounded to one document.  They model
    parser/chunk-boundary equivalence, not a fuzzy answer substitution across
    papers; cross-paper questions must still declare a required group for each
    document.
    """

    variants = [locator]
    for index, alternative in enumerate(locator.alternatives, start=1):
        protected = {"group_id", "kind", "document", "required", "alternatives"} & set(alternative)
        if protected:
            raise EvaluationContractError(
                f"{locator.group_id}: alternative cannot override {sorted(protected)}"
            )
        variants.append(
            EvidenceLocator.from_dict(
                {
                    "group_id": locator.group_id,
                    "kind": locator.kind,
                    "document": locator.document,
                    "required": locator.required,
                    **alternative,
                },
                index,
            )
        )
    resolved: list[ResolvedGroup] = []
    errors: list[EvaluationContractError] = []
    for variant in variants:
        try:
            resolved.append(_resolve_single_locator(connection, variant))
        except EvaluationContractError as exc:
            errors.append(exc)
    if not resolved:
        raise errors[0]
    document_ids = {value.document_id for value in resolved}
    if len(document_ids) != 1:
        raise EvaluationContractError(f"{locator.group_id}: alternatives resolved across documents")
    return ResolvedGroup(
        locator=locator,
        document_id=next(iter(document_ids)),
        evidence_ids=frozenset().union(*(value.evidence_ids for value in resolved)),
        page_keys=frozenset().union(*(value.page_keys for value in resolved)),
    )


def resolve_case(connection, case: EvalCase) -> ResolvedCase:
    groups = tuple(_resolve_group(connection, locator) for locator in case.evidence_groups)
    required_documents = frozenset(group.document_id for group in groups)

    def resolve_document_ref(ref: dict[str, Any]) -> str:
        matches = _document_matches(connection, ref)
        if len(matches) != 1:
            raise EvaluationContractError(
                f"{case.case_id}: document reference resolved {len(matches)} rows"
            )
        return str(matches[0]["id"])

    focus_ids = tuple(resolve_document_ref(ref) for ref in case.focus_documents)
    forbidden_ids = frozenset(resolve_document_ref(ref) for ref in case.forbidden_documents)
    return ResolvedCase(
        case=case,
        document_ids=required_documents,
        focus_document_ids=focus_ids,
        groups=groups,
        forbidden_document_ids=forbidden_ids,
    )


def load_cases(
    path: Path,
    *,
    split: str | None = None,
    allow_shadow: bool = False,
    qrels_path: Path | None = None,
    require_audited: bool = False,
    expected_rewrite_fingerprint: str | None = None,
) -> list[EvalCase]:
    if not path.is_file():
        raise EvaluationContractError(f"suite file does not exist: {path}")
    if "shadow-v2" in path.name:
        raise EvaluationContractError(
            "shadow-v2 is deprecated; rebuild a semantically curated shadow-v3 "
            "from public source question/evidence pairs"
        )
    if "shadow" in path.name and not allow_shadow:
        raise EvaluationContractError("shadow suite is sealed; pass --open-shadow")
    raw_rows: list[dict[str, Any]] = []
    qrels_by_case: dict[str, dict[str, Any]] = {}
    if qrels_path is not None:
        if not qrels_path.is_file():
            raise EvaluationContractError(f"qrels file does not exist: {qrels_path}")
        for line in qrels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in qrels_by_case:
                raise EvaluationContractError(f"duplicate/empty qrel case_id: {case_id!r}")
            qrels_by_case[case_id] = row
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            overlay = qrels_by_case.get(case_id)
            if overlay:
                row = {**row, **overlay}
            raw_rows.append(row)
            case = EvalCase.from_dict(row)
        except (json.JSONDecodeError, EvaluationContractError) as exc:
            raise EvaluationContractError(f"{path}:{line_number}: {exc}") from exc
        if not case.case_id or case.case_id in seen:
            raise EvaluationContractError(f"duplicate/empty case_id at {path}:{line_number}")
        if not case.raw_query or not case.enriched_query:
            raise EvaluationContractError(f"{case.case_id}: raw_query and enriched_query are required")
        if case.language not in {"vi", "en"}:
            raise EvaluationContractError(f"{case.case_id}: language must be vi or en")
        variants = plan_query_variants(
            original_query=case.raw_query,
            standalone_query=case.standalone_full,
            required_entities=list(case.required_entities),
        ).variants
        if not variants:
            raise EvaluationContractError(f"{case.case_id}: drift guard removed every query")
        if case.category in POSITIVE_CATEGORIES and not case.evidence_groups:
            raise EvaluationContractError(f"{case.case_id}: positive case has no evidence groups")
        if require_audited:
            _validate_audited_case(case)
        seen.add(case.case_id)
        cases.append(case)
    if split and len(cases) != int(split):
        raise EvaluationContractError(f"expected {split} cases, found {len(cases)}")
    if qrels_by_case and set(qrels_by_case) != {case.case_id for case in cases}:
        missing = sorted({case.case_id for case in cases} - set(qrels_by_case))
        extra = sorted(set(qrels_by_case) - {case.case_id for case in cases})
        raise EvaluationContractError(f"qrels case mismatch: missing={missing[:3]} extra={extra[:3]}")
    if require_audited:
        fingerprints = {case.rewrite_fingerprint for case in cases}
        if None in fingerprints or len(fingerprints) != 1:
            raise EvaluationContractError("audited suite has missing or inconsistent rewrite fingerprint")
        actual_fingerprint = next(iter(fingerprints))
        if expected_rewrite_fingerprint and actual_fingerprint != expected_rewrite_fingerprint:
            raise EvaluationContractError(
                "rewrite fingerprint mismatch: "
                f"expected {expected_rewrite_fingerprint!r}, got {actual_fingerprint!r}"
            )
        quality = suite_quality_diagnostics(cases)
        if not quality["promotion_safe"]:
            raise EvaluationContractError(
                "suite is not promotion-safe: " + "; ".join(quality["issues"])
            )
    return cases


def _validate_audited_case(case: EvalCase) -> None:
    if case.curation_status != "audited":
        raise EvaluationContractError(f"{case.case_id}: suite case is not audited")
    if case.audit_status != "approved":
        raise EvaluationContractError(f"{case.case_id}: audit is not approved")
    if not case.audit_reviewer or not case.audit_reviewed_at:
        raise EvaluationContractError(f"{case.case_id}: missing reviewer attestation")
    if not case.language_reviewed:
        raise EvaluationContractError(f"{case.case_id}: language has not been reviewed")
    if not case.qrels_reviewed:
        raise EvaluationContractError(f"{case.case_id}: qrels have not been semantically reviewed")
    if not case.standalone_full or not case.standalone_anchor:
        raise EvaluationContractError(f"{case.case_id}: missing frozen full/anchor rewrite")
    if not case.rewrite_fingerprint:
        raise EvaluationContractError(f"{case.case_id}: missing rewrite fingerprint")
    if case.category in POSITIVE_CATEGORIES and not case.rewrite_provenance:
        raise EvaluationContractError(f"{case.case_id}: audited rewrite has no provenance")
    if _is_deictic_follow_up(case.raw_query) and not (
        case.conversation_context or case.working_state or case.focus_documents
    ):
        raise EvaluationContractError(f"{case.case_id}: contextual follow-up has no prior context/state")
    allowed = {
        "raw_query": case.raw_query,
        "conversation": "\n".join(item.get("content") or "" for item in case.conversation_context),
        "working_state": json.dumps(case.working_state, ensure_ascii=False),
        "catalog": "\n".join(
            [*case.required_entities, *[str(item.get("filename") or "") for item in case.focus_documents]]
        ),
    }
    for variant in ("full", "anchor"):
        entries = case.rewrite_provenance.get(variant) or ()
        if not entries:
            raise EvaluationContractError(f"{case.case_id}: {variant} rewrite has no provenance entries")
        for entry in entries:
            source = str(entry.get("source") or "")
            text = str(entry.get("text") or "").strip()
            if source not in allowed or not text or normalize_text(text) not in normalize_text(allowed[source]):
                raise EvaluationContractError(f"{case.case_id}: unsupported {variant} rewrite provenance")
    supported_text = "\n".join(allowed.values())
    _validate_rewrite_anchors(case, supported_text)
    for locator in case.evidence_groups:
        sensitive_values = [locator.caption, locator.quote]
        if locator.page_number is not None:
            sensitive_values.append(f"page {locator.page_number}")
        if locator.artifact_index is not None:
            label = "table" if locator.kind == "table" else "figure"
            # Runtime indices are zero-based while paper labels are normally
            # one-based.  Guard both representations without assuming that a
            # correct query can never refer to one: it is allowed only when
            # that fact was already present in raw/context/state/catalog.
            sensitive_values.extend(
                [f"{label} {locator.artifact_index}", f"{label} {locator.artifact_index + 1}"]
            )
        for leaked in sensitive_values:
            leaked = str(leaked or "").strip()
            for rewrite in (case.standalone_full, case.standalone_anchor):
                if (
                    leaked
                    and normalize_text(leaked) in normalize_text(rewrite)
                    and normalize_text(leaked) not in normalize_text(supported_text)
                ):
                    raise EvaluationContractError(f"{case.case_id}: qrel evidence leaked into rewrite")
    if case.scope == "focused" and not case.focus_documents:
        raise EvaluationContractError(f"{case.case_id}: focused case has no catalog focus documents")


def _validate_rewrite_anchors(case: EvalCase, supported_text: str) -> None:
    """Reject introduced exact anchors without imposing a semantic length cap.

    Natural connective language is allowed: a rewrite is not a verbatim copy
    of chat history.  Exact identifiers, filenames and numbers are different:
    they can steer lexical/vector retrieval sharply and therefore must be
    traceable to raw chat, working state, or the catalog provenance recorded
    for the case.
    """

    supported = normalize_text(supported_text)
    for rewrite_name, rewrite in {
        "full": case.standalone_full,
        "anchor": case.standalone_anchor,
    }.items():
        for token in _exact_anchor_tokens(rewrite):
            if normalize_text(token) not in supported:
                raise EvaluationContractError(
                    f"{case.case_id}: unsupported {rewrite_name} rewrite entity {token!r}"
                )


def _exact_anchor_tokens(value: str) -> set[str]:
    # PDF names, numeric measurements/table labels, and technical acronyms
    # are deterministic retrieval anchors.  This intentionally has no word
    # count or broad noun extraction heuristic.
    filename_tokens = re.findall(r"[^\s/]+\.pdf\b", value, flags=re.IGNORECASE)
    numeric_tokens = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:%|m|ms|d)?(?!\w)", value, flags=re.IGNORECASE)
    acronym_tokens = re.findall(r"(?<!\w)[A-Z][A-Z0-9_\-/]{1,}(?!\w)", value)
    return {token for token in [*filename_tokens, *numeric_tokens, *acronym_tokens] if token}


def _is_deictic_follow_up(value: str) -> bool:
    normalized = normalize_text(value)
    markers = (
        "this table",
        "that table",
        "this figure",
        "that figure",
        "that paper",
        "those papers",
        "bảng này",
        "bảng đó",
        "hình này",
        "hình đó",
        "bài này",
        "bài đó",
        "hai bài đó",
    )
    return any(marker in normalized for marker in markers)


def _identity_set(result: dict[str, Any]) -> set[str]:
    return canonical_evidence_ids(result)


def _result_kind(result: dict[str, Any]) -> str:
    return evidence_kind(result)


def _group_hit(group: ResolvedGroup, results: Sequence[dict[str, Any]]) -> bool:
    for result in results:
        if str(result.get("document_id") or "") != group.document_id:
            continue
        if group.evidence_ids & _identity_set(result):
            return True
        if group.locator.kind == "page" and group.page_keys and (
            group.document_id,
            result.get("page_number"),
        ) in group.page_keys:
            return True
    return False


def _required_document_recall(resolved: ResolvedCase, results: Sequence[dict[str, Any]], cutoff: int) -> float:
    required = resolved.document_ids
    if not required:
        return 1.0
    found = {str(item.get("document_id") or "") for item in results[:cutoff]}
    return len(required & found) / len(required)


def score_results(resolved: ResolvedCase, results: Sequence[dict[str, Any]], *, cutoff: int) -> dict[str, Any]:
    top = list(results[:cutoff])
    groups = [group for group in resolved.groups if group.locator.required]
    hits = [group for group in groups if _group_hit(group, top)]
    by_kind: dict[str, list[ResolvedGroup]] = {}
    for group in groups:
        by_kind.setdefault(group.locator.kind, []).append(group)
    modality = {
        kind: sum(_group_hit(group, top) for group in values) / len(values)
        for kind, values in by_kind.items()
    }
    documents = {str(item.get("document_id") or "") for item in top}
    forbidden = sorted(documents & resolved.forbidden_document_ids)
    ranking = _ranking_metrics(resolved, top, cutoff=cutoff)
    return {
        "evidence_recall": len(hits) / len(groups) if groups else None,
        "document_recall": _required_document_recall(resolved, top, cutoff),
        "multi_paper_coverage": bool(resolved.document_ids <= documents) if len(resolved.document_ids) > 1 else None,
        "modality_recall": modality,
        "forbidden_documents": forbidden,
        "hit": bool(hits) if groups else not forbidden,
        "result_count": len(top),
        "result_kinds": [_result_kind(item) for item in top],
        **ranking,
    }


def _ranking_metrics(resolved: ResolvedCase, results: Sequence[dict[str, Any]], *, cutoff: int) -> dict[str, Any]:
    """Compute rank diagnostics against evidence groups, not runtime UUIDs."""

    groups = [group for group in resolved.groups if group.locator.required]
    if not groups:
        return {"hit_at_1": None, "hit_at_3": None, "hit_at_5": None, "hit_at_10": None, "mrr_at_10": None, "ndcg_at_10": None}
    relevance: list[int] = []
    for result in list(results)[:cutoff]:
        relevance.append(int(any(_group_hit(group, [result]) for group in groups)))
    def hit_at(rank: int) -> float:
        return float(any(relevance[:rank]))
    first_rank = next((index + 1 for index, value in enumerate(relevance[:10]) if value), None)
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance[:10]))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(groups), 10)))
    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "hit_at_10": hit_at(10),
        "mrr_at_10": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg_at_10": (dcg / ideal) if ideal else 0.0,
    }


class CountingEmbeddings:
    """Transparent provider wrapper used for per-arm call accounting."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.query_calls = 0
        self.text_calls = 0

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return await self.provider.embed_query(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls += 1
        return await self.provider.embed_texts(texts)


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, selected: list[dict[str, Any]] | None = None) -> float | None:
        values = [float(row[key]) for row in (selected or rows) if row.get(key) is not None]
        return round(statistics.fmean(values), 4) if values else None

    positive = [row for row in rows if row.get("positive")]
    negative = [row for row in rows if not row.get("positive")]
    modalities: dict[str, list[float]] = {}
    failure_counts: dict[str, int] = {}
    variant_reasons: dict[str, int] = {}
    for row in positive:
        for key, value in (row.get("modality_recall") or {}).items():
            modalities.setdefault(key, []).append(float(value))
    for row in rows:
        reason = str(row.get("raw_query_informativeness") or "single_query")
        variant_reasons[reason] = variant_reasons.get(reason, 0) + 1
        for failure in row.get("failure_taxonomy") or []:
            for reason in failure.get("reasons") or []:
                failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1
    def subset(name: str, selected: list[dict[str, Any]]) -> dict[str, float | int | None]:
        selected_positive = [row for row in selected if row.get("positive")]
        return {
            "cases": len(selected_positive),
            "evidence_recall_at_10": mean("evidence_recall", selected_positive),
            "document_recall_at_5": mean("document_recall", selected_positive),
        }
    return {
        "cases": len(rows),
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "candidate_recall_at_30": mean("candidate_recall", positive),
        "evidence_recall_at_10": mean("evidence_recall", positive),
        "document_recall_at_5": mean("document_recall", positive),
        "hit_at_1": mean("hit_at_1", positive),
        "hit_at_3": mean("hit_at_3", positive),
        "hit_at_5": mean("hit_at_5", positive),
        "hit_at_10": mean("hit_at_10", positive),
        "mrr_at_10": mean("mrr_at_10", positive),
        "ndcg_at_10": mean("ndcg_at_10", positive),
        "multi_paper_coverage": mean("multi_paper_coverage", [row for row in positive if row.get("multi_paper_coverage") is not None]),
        "modality_recall": {key: round(statistics.fmean(values), 4) for key, values in modalities.items()},
        "forbidden_scope_leaks": sum(len(row.get("forbidden_documents") or []) for row in rows),
        "failure_taxonomy": dict(sorted(failure_counts.items())),
        "latency_p50_ms": _percentile((row["latency_ms"] for row in rows), 0.50),
        "latency_p95_ms": _percentile((row["latency_ms"] for row in rows), 0.95),
        "embedding_query_calls": sum(int(row.get("embedding_query_calls") or 0) for row in rows),
        "rerank_embedding_calls": sum(
            int(row.get("rerank_embedding_query_calls") or 0)
            + int(row.get("rerank_embedding_text_calls") or 0)
            for row in rows
        ),
        "query_variant_activation": {
            "multi_query_cases": sum(len(row.get("query_origins") or []) > 1 for row in rows),
            "activation_rate": round(
                sum(len(row.get("query_origins") or []) > 1 for row in rows) / len(rows),
                4,
            ) if rows else 0.0,
            "reasons": dict(sorted(variant_reasons.items())),
        },
        "timing_p50_ms": _timing_percentiles(rows, 0.50),
        "timing_p95_ms": _timing_percentiles(rows, 0.95),
        "subsets": {
            "table_numeric": subset("table_numeric", [row for row in rows if row.get("category") == "table_numeric"]),
            "figure_page": subset("figure_page", [row for row in rows if row.get("category") == "figure_page"]),
            "focused": subset("focused", [row for row in rows if row.get("scope") == "focused"]),
        },
    }


def _timing_percentiles(rows: list[dict[str, Any]], fraction: float) -> dict[str, float | None]:
    keys = {
        key
        for row in rows
        for key in (row.get("timings_ms") or {})
        if isinstance((row.get("timings_ms") or {}).get(key), (int, float))
    }
    return {
        key: _percentile(
            (float((row.get("timings_ms") or {}).get(key)) for row in rows if (row.get("timings_ms") or {}).get(key) is not None),
            fraction,
        )
        for key in sorted(keys)
    }


def _length_bucket(words: int) -> str:
    if words <= 20:
        return "<=20"
    if words <= 40:
        return "21-40"
    return ">40"


def _paired_delta(full_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report B-full minus B-anchor without declaring either promotable."""

    anchor_by_case = {str(row["case_id"]): row for row in anchor_rows}
    dimensions = ("query_length_bucket", "contextual", "raw_query_informativeness", "category")
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for full in full_rows:
            anchor = anchor_by_case.get(str(full["case_id"]))
            if anchor is None or not full.get("positive"):
                continue
            key = str(full.get(dimension))
            groups.setdefault(key, []).append((full, anchor))
        output[dimension] = {
            key: {
                "cases": len(pairs),
                "candidate_recall_delta": round(statistics.fmean(float(full["candidate_recall"] or 0) - float(anchor["candidate_recall"] or 0) for full, anchor in pairs), 4),
                "evidence_recall_delta": round(statistics.fmean(float(full["evidence_recall"] or 0) - float(anchor["evidence_recall"] or 0) for full, anchor in pairs), 4),
                "document_recall_delta": round(statistics.fmean(float(full["document_recall"] or 0) - float(anchor["document_recall"] or 0) for full, anchor in pairs), 4),
            }
            for key, pairs in sorted(groups.items())
        }
    return output


def _recommend_action(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Decide what (if anything) the comparison actually supports.

    A passing promotion gate is necessary, but it is not sufficient to claim
    that dual-query was evaluated: the raw/rewrite planner may legitimately
    prune the raw branch for generic follow-ups.  Treating that no-op as a
    successful treatment made an earlier report recommend promotion while B
    had zero multi-query activations.  Keep the decision explicit and expose
    the measurements used so a report cannot hide that distinction.
    """

    a0 = reports["A0"]
    a1 = reports["A1"]
    b = reports["B-full"]
    a0_metrics = a0.get("metrics") or {}
    a1_metrics = a1.get("metrics") or {}
    b_metrics = b.get("metrics") or {}
    a0_candidate = float(a0_metrics.get("candidate_recall_at_30") or 0.0)
    a1_candidate = float(a1_metrics.get("candidate_recall_at_30") or 0.0)
    b_candidate = float(b_metrics.get("candidate_recall_at_30") or 0.0)
    a1_gain = a1_candidate - a0_candidate
    b_gain_vs_a1 = b_candidate - a1_candidate
    b_gain_vs_a0 = b_candidate - a0_candidate
    activation = b_metrics.get("query_variant_activation") or {}
    activation_rate = float(activation.get("activation_rate") or 0.0)
    multi_query_cases = int(activation.get("multi_query_cases") or 0)

    a1_passed = bool((a1.get("promotion") or {}).get("passed"))
    b_passed = bool((b.get("promotion") or {}).get("passed"))

    # Candidate union is a distinct intervention.  If it is the only arm that
    # demonstrates a meaningful gain, do not attribute that gain to RRF.
    if a1_passed and a1_gain >= 0.02 and (not b_passed or b_gain_vs_a1 < 0.01):
        action = "promote_candidate_union_only"
    elif activation_rate <= 0.0 or multi_query_cases <= 0:
        action = "keep_dual_query_disabled_no_treatment_activation"
    elif b_passed and (
        b_gain_vs_a1 >= 0.01
        or (
            b_gain_vs_a1 >= 0.0
            and float(b_metrics.get("latency_p50_ms") or float("inf"))
            < float(a0_metrics.get("latency_p50_ms") or float("inf"))
            and int(b_metrics.get("embedding_query_calls") or 0)
            <= int(a0_metrics.get("embedding_query_calls") or 0)
        )
    ):
        action = "promote_dual_query"
    else:
        action = "keep_dual_query_disabled"

    return {
        "A1_candidate_gain_vs_A0": round(a1_gain, 4),
        "B_candidate_gain_vs_A1": round(b_gain_vs_a1, 4),
        "B_candidate_gain_vs_A0": round(b_gain_vs_a0, 4),
        "B_multi_query_cases": multi_query_cases,
        "B_activation_rate": round(activation_rate, 4),
        "recommended_action": action,
    }


def promotion_gate(report: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    modalities = metrics.get("modality_recall") or {}
    checks = {
        "candidate_recall": (metrics.get("candidate_recall_at_30") or 0.0) >= 0.95,
        "evidence_recall": (metrics.get("evidence_recall_at_10") or 0.0) >= 0.85,
        "document_recall": (metrics.get("document_recall_at_5") or 0.0) >= 0.90,
        "multi_paper": (metrics.get("multi_paper_coverage") or 0.0) >= 0.95,
        "modality_floor": all(float(value) >= 0.80 for value in modalities.values()),
        "p50_latency": metrics.get("latency_p50_ms") is not None and metrics["latency_p50_ms"] <= 500,
        "p95_latency": metrics.get("latency_p95_ms") is not None and metrics["latency_p95_ms"] <= 1500,
        "qrels_resolved": report.get("qrels_resolved") is True,
        "no_scope_leak": int(metrics.get("forbidden_scope_leaks") or 0) == 0,
        "no_model_calls": int(report.get("model_calls") or 0) == 0,
    }
    if baseline:
        treatment = float(metrics.get("candidate_recall_at_30") or 0.0)
        control = float(baseline.get("candidate_recall_at_30") or 0.0)
        checks["candidate_gain_or_noninferior"] = (
            treatment >= control + 0.02
            if control < 0.95
            else treatment >= control - 0.01
        )
        checks["final_evidence_noninferior"] = float(
            metrics.get("evidence_recall_at_10") or 0.0
        ) >= float(baseline.get("evidence_recall_at_10") or 0.0)
        checks["document_recall_noninferior"] = float(
            metrics.get("document_recall_at_5") or 0.0
        ) >= float(baseline.get("document_recall_at_5") or 0.0)
        current_subsets = metrics.get("subsets") or {}
        baseline_subsets = baseline.get("subsets") or {}

        def subset_noninferior(name: str) -> bool:
            current = current_subsets.get(name) or {}
            control = baseline_subsets.get(name) or {}
            if not int(control.get("cases") or 0):
                return True
            return int(current.get("cases") or 0) == int(control.get("cases") or 0) and float(
                current.get("evidence_recall_at_10") or 0.0
            ) >= float(control.get("evidence_recall_at_10") or 0.0)

        checks["exact_artifact_noninferior"] = all(
            subset_noninferior(name) for name in ("table_numeric", "figure_page")
        )
        checks["focused_scope_noninferior"] = subset_noninferior("focused")
        checks["final_nonregression"] = (
            float(metrics.get("evidence_recall_at_10") or 0.0)
            >= float(baseline.get("evidence_recall_at_10") or 0.0)
            and float(metrics.get("document_recall_at_5") or 0.0)
            >= float(baseline.get("document_recall_at_5") or 0.0)
        )
    return {"passed": all(checks.values()), "checks": checks}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _index_fingerprint(db_path: Path) -> str:
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, filename, content_hash, chunk_count, table_count, figure_count FROM documents ORDER BY id"
        ).fetchall()
    return _json_hash([dict(row) for row in rows])


def _validate_revision_aware_eval_db(db_path: Path) -> None:
    """Fail clearly when the compare target has not completed foundation migration.

    The evaluator must never silently score a legacy path-only database as if
    staged/retired revisions were isolated.  Tests may still use the portable
    resolver against tiny legacy schemas, but a real benchmark needs the
    revision columns and v2 FTS projection that production retrieval filters.
    """

    if not db_path.is_file():
        raise EvaluationContractError(f"evaluation database does not exist: {db_path}")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        document_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        fts_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(rag_chunks_fts_v2)").fetchall()
        }
        required_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    except sqlite3.Error as exc:
        raise EvaluationContractError(f"evaluation database unreadable: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    missing = []
    if "active_revision_id" not in document_columns:
        missing.append("documents.active_revision_id")
    if "revision_status" not in document_columns:
        missing.append("documents.revision_status")
    if "revision_id" not in fts_columns:
        missing.append("rag_chunks_fts_v2.revision_id")
    if "document_revisions" not in required_tables:
        missing.append("document_revisions")
    if missing:
        raise EvaluationContractError(
            "evaluation database is not revision-aware (missing "
            + ", ".join(missing)
            + "); start the migrated isolated backend or pass an isolated foundation DB"
        )


async def _run_arm(
    resolved: ResolvedCase,
    *,
    arm: str,
    document_ids: list[str] | None,
    rag: RagService,
    store: LanceDBRetrievalStore,
    embeddings: CountingEmbeddings,
    settings: Any,
    top_k: int,
    final: bool,
) -> tuple[dict[str, Any], float]:
    case = resolved.case
    visual_boost = case.category in {"table_numeric", "figure_page"}
    started = time.perf_counter()
    query_variant_reason = "single_query"
    if arm == "A0":
        payload = await rag.search_hybrid(
            query=case.standalone_full,
            top_k=top_k,
            document_ids=document_ids,
            retrieval_store=store,
            embeddings=embeddings,
            rerank=False,
            rerank_mode=settings.rerank_mode,
            cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
            visual_boost=visual_boost,
            candidate_document_union=False,
        )
    elif arm == "A1":
        payload = await rag.search_hybrid(
            query=case.standalone_full,
            top_k=top_k,
            document_ids=document_ids,
            retrieval_store=store,
            embeddings=embeddings,
            rerank=False,
            rerank_mode=settings.rerank_mode,
            cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
            visual_boost=visual_boost,
            candidate_document_union=True,
        )
    else:
        standalone = case.standalone_full if arm == "B-full" else case.standalone_anchor
        plan = plan_query_variants(
            original_query=case.raw_query,
            standalone_query=standalone,
            required_entities=list(case.required_entities),
        )
        query_variant_reason = plan.reason
        queries = list(plan.variants)
        if len(queries) == 1:
            payload = await rag.search_hybrid(
                query=queries[0][1],
                top_k=top_k,
                document_ids=document_ids,
                retrieval_store=store,
                embeddings=embeddings,
                rerank=False,
                rerank_mode=settings.rerank_mode,
                cross_encoder_model_path=settings.rerank_cross_encoder_path,
                rerank_max_candidates=settings.rerank_max_candidates,
                visual_boost=visual_boost,
                candidate_document_union=True,
            )
            payload["queries"] = [queries[0][1]]
            payload["query_origins"] = [queries[0][0]]
            payload["dual_query_rrf"] = False
        else:
            payload = await rag.search_hybrid_queries(
                queries=queries,
                top_k=top_k,
                document_ids=document_ids,
                retrieval_store=store,
                embeddings=embeddings,
                rerank=False,
                rerank_mode=settings.rerank_mode,
                cross_encoder_model_path=settings.rerank_cross_encoder_path,
                rerank_max_candidates=settings.rerank_max_candidates,
                visual_boost=visual_boost,
                rrf_k=settings.dual_query_rrf_k,
            )
    payload["query_variant_reason"] = query_variant_reason
    return payload, round((time.perf_counter() - started) * 1000, 3)


def _evidence_identity_for_merge(item: dict[str, Any]) -> str:
    """Return a stable merge key even for legacy projections.

    The production comparison path retrieves each required document in its own
    branch.  Branches can therefore return the same logical artifact through
    different channels (FTS, vector, or a card).  Merging by a single
    canonical identity keeps the evaluator's candidate depth honest and
    avoids rewarding duplicate projections.
    """

    identity = canonical_evidence_id(item)
    if identity:
        return identity
    document_id = str(item.get("document_id") or "")
    for field in ("chunk_id", "table_id", "figure_id", "page_number", "id"):
        value = str(item.get(field) or "")
        if value:
            return f"legacy:{document_id}:{field}:{value}"
    return f"legacy:{document_id}:{_json_hash(item)}"


def _merge_decomposed_payloads(
    branch_payloads: list[tuple[str, dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Merge per-document branches the way the production compare path does.

    Scores from separately scoped searches are not calibrated across papers;
    rank-interleaving is consequently safer than sorting raw distances from
    unrelated branches.  It also guarantees that a long paper cannot consume
    the entire candidate frontier before final evidence selection.
    """

    candidate_lists = [
        list(payload.get("candidate_results") or payload.get("results") or [])
        for _document_id, payload in branch_payloads
    ]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_depth = max((len(values) for values in candidate_lists), default=0)
    for rank in range(max_depth):
        for values in candidate_lists:
            if rank >= len(values):
                continue
            item = dict(values[rank])
            identity = _evidence_identity_for_merge(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
            if len(merged) >= top_k:
                break
        if len(merged) >= top_k:
            break

    def merge_channel(name: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        channel_seen: set[str] = set()
        for _document_id, payload in branch_payloads:
            for raw in (payload.get("channel_candidates") or {}).get(name) or []:
                item = dict(raw)
                identity = _evidence_identity_for_merge(item)
                if identity in channel_seen:
                    continue
                channel_seen.add(identity)
                values.append(item)
        return values

    selected_document_ids = list(
        dict.fromkeys(
            document_id
            for _document_id, payload in branch_payloads
            for document_id in payload.get("selected_document_ids") or []
        )
    )
    return {
        "query": str(branch_payloads[0][1].get("query") or "") if branch_payloads else "",
        "queries": list(
            dict.fromkeys(
                query
                for _document_id, payload in branch_payloads
                for query in payload.get("queries") or [payload.get("query") or ""]
                if query
            )
        ),
        "query_origins": ["document_decomposition"],
        "selected_document_ids": selected_document_ids,
        "retrieval_channels": list(
            dict.fromkeys(
                channel
                for _document_id, payload in branch_payloads
                for channel in payload.get("retrieval_channels") or []
            )
        ),
        "document_card_results": [
            card
            for _document_id, payload in branch_payloads
            for card in payload.get("document_card_results") or []
        ],
        "candidate_document_union": any(
            bool(payload.get("candidate_document_union"))
            for _document_id, payload in branch_payloads
        ),
        "dual_query_rrf": any(
            bool(payload.get("dual_query_rrf")) for _document_id, payload in branch_payloads
        ),
        "candidate_results": merged,
        "channel_candidates": {
            "fts": merge_channel("fts"),
            "vector": merge_channel("vector"),
            "artifact": merge_channel("artifact"),
        },
        "branch_results": [
            {
                "origin": f"document:{document_id}",
                "results": list(payload.get("candidate_results") or payload.get("results") or []),
                "selected_document_ids": list(payload.get("selected_document_ids") or []),
                "channel_candidates": dict(payload.get("channel_candidates") or {}),
                "timings_ms": dict(payload.get("timings_ms") or {}),
            }
            for document_id, payload in branch_payloads
        ],
        "timings_ms": {
            "branches": max(
                (
                    float((payload.get("timings_ms") or {}).get("total") or 0.0)
                    for _document_id, payload in branch_payloads
                ),
                default=0.0,
            ),
            "fusion": 0.0,
        },
        "results": merged,
    }


async def _run_production_shaped_arm(
    resolved: ResolvedCase,
    *,
    arm: str,
    document_ids: list[str] | None,
    rag: RagService,
    store: LanceDBRetrievalStore,
    embeddings: CountingEmbeddings,
    settings: Any,
    top_k: int,
) -> tuple[dict[str, Any], float]:
    """Run an arm with the same per-document decomposition as production.

    The compare agent deliberately scopes each paper before retrieval.  A
    single global search would measure a different system and lets the longest
    paper monopolize top-k, which was the source of the old misleading Gate C
    misses.  Other categories retain the ordinary single-scope path.
    """

    case = resolved.case
    required_documents = list(dict.fromkeys(document_ids or []))
    if case.category != "cross_document_compare" or len(required_documents) < 2:
        return await _run_arm(
            resolved,
            arm=arm,
            document_ids=document_ids,
            rag=rag,
            store=store,
            embeddings=embeddings,
            settings=settings,
            top_k=top_k,
            final=False,
        )
    started = time.perf_counter()
    branch_results = await asyncio.gather(
        *[
            _run_arm(
                resolved,
                arm=arm,
                document_ids=[document_id],
                rag=rag,
                store=store,
                embeddings=embeddings,
                settings=settings,
                top_k=top_k,
                final=False,
            )
            for document_id in required_documents
        ]
    )
    payload = _merge_decomposed_payloads(
        list(zip(required_documents, (item[0] for item in branch_results))),
        top_k=top_k,
    )
    payload["decomposition"] = "compare_per_document"
    payload["query_variant_reason"] = "production_compare_decomposition"
    return payload, round((time.perf_counter() - started) * 1000, 3)


def _runtime_document_scope(
    rag: RagService,
    resolved: ResolvedCase,
) -> tuple[list[str] | None, str]:
    """Resolve suite scope through the same catalog signal as production.

    An ``unscoped`` suite label does not mean that the current turn is an
    anonymous corpus search: a raw filename or uniquely owned paper alias is
    authoritative in Aya's catalog resolver. This avoids scoring production
    document routing as a retrieval miss, without consulting qrels to choose
    the target document.
    """

    if resolved.case.scope == "focused":
        return list(resolved.focus_document_ids), "suite_focus"

    mentions = rag.resolve_catalog_mentions(
        query=resolved.case.raw_query,
        collection_id=None,
    )
    document_ids = list(mentions.document_ids)
    if not document_ids:
        return None, "unscoped"

    required = set(resolved.document_ids)
    selected = document_ids[:2] if resolved.case.category == "cross_document_compare" else document_ids[:1]
    if resolved.case.positive and not set(selected).issubset(required):
        raise EvaluationContractError(
            f"{resolved.case.case_id}: catalog scope {selected} does not match qrel documents"
        )
    return selected, "catalog_current_turn"


async def _run_final_stage(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    embeddings: CountingEmbeddings,
    settings: Any,
    required_document_ids: list[str] | None = None,
    required_facets_by_document: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Apply the production reranker once to an already-measured candidate pool."""

    started = time.perf_counter()
    if not settings.rerank_enabled or not candidates:
        selected, _diagnostics = select_ranked_evidence(
            list(candidates),
            limit=10,
            query=query,
            required_document_ids=required_document_ids,
            required_facets_by_document=required_facets_by_document,
        )
        return list(selected), round((time.perf_counter() - started) * 1000, 3)
    selected = await rerank_candidates(
        query=query,
        results=[dict(item) for item in candidates],
        embeddings=embeddings,
        top_k=10,
        mode=settings.rerank_mode,
        cross_encoder_model_path=settings.rerank_cross_encoder_path,
        max_candidates=settings.rerank_max_candidates,
    )
    return list(selected[:10]), round((time.perf_counter() - started) * 1000, 3)


def _failure_taxonomy(
    resolved: ResolvedCase,
    *,
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    final_results: list[dict[str, Any]],
    rerank_applied: bool,
) -> list[dict[str, Any]]:
    """Explain misses by retrieval stage without modifying relevance labels."""

    selected_documents = {str(value) for value in payload.get("selected_document_ids") or []}
    branches = payload.get("branch_results") or []
    branch_hits = {
        str(branch.get("origin") or "unknown"): list(branch.get("results") or [])
        for branch in branches
    }
    channel_candidates = payload.get("channel_candidates") or {}
    failures: list[dict[str, Any]] = []
    for group in resolved.groups:
        candidate_hit = _group_hit(group, candidates)
        final_hit = _group_hit(group, final_results)
        reasons: list[str] = []
        if not candidate_hit:
            if selected_documents and group.document_id not in selected_documents:
                reasons.append("document_selection_miss")
            branch_match = any(_group_hit(group, values) for values in branch_hits.values())
            if branch_match:
                reasons.append("fusion_loss")
            fts_hit = _group_hit(group, list(channel_candidates.get("fts") or []))
            vector_hit = _group_hit(group, list(channel_candidates.get("vector") or []))
            if fts_hit and not vector_hit:
                reasons.append("vector_miss")
            elif vector_hit and not fts_hit:
                reasons.append("fts_miss")
            elif not fts_hit and not vector_hit:
                same_document = [item for item in candidates if str(item.get("document_id") or "") == group.document_id]
                if same_document and not any(_result_kind(item) == group.locator.kind for item in same_document):
                    reasons.append("parser_artifact_generation_miss")
                else:
                    reasons.append("fts_vector_miss")
        elif not final_hit:
            # The production configuration may intentionally have no reranker.
            # In that case this is a top-k cutoff diagnostic, not an invented
            # reranker failure.
            reasons.append("rerank_loss" if rerank_applied else "final_top_k_cutoff")
        if reasons:
            failures.append(
                {
                    "group_id": group.locator.group_id,
                    "kind": group.locator.kind,
                    "document_id": group.document_id,
                    "reasons": reasons,
                }
            )
    if any(payload.get("scope_leak") or []):
        failures.append({"group_id": None, "kind": None, "document_id": None, "reasons": ["scope_leak"]})
    return failures


async def evaluate(
    cases: list[EvalCase],
    *,
    db_path: Path,
    lancedb_path: Path,
    artifact_root: Path,
    seed: int = 20260823,
) -> dict[str, Any]:
    settings = get_settings()
    if settings.dual_query_rrf_enabled:
        # The evaluator is a comparison harness, never a reason to mutate the
        # production flag.  It remains valid with the flag either way, but the
        # report must make an accidentally promoted runtime visible.
        runtime_flag = True
    else:
        runtime_flag = False
    try:
        with connect(db_path) as connection:
            resolved = [resolve_case(connection, case) for case in cases]
    except EvaluationContractError as exc:
        raise EvaluationContractError(f"qrel_preflight: {exc}") from exc
    if not resolved:
        raise EvaluationContractError("suite has no cases")

    store = LanceDBRetrievalStore(lancedb_path)
    rag = RagService(db_path, artifact_root=artifact_root)
    provider = CountingEmbeddings(
        OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    )
    await provider.embed_query(resolved[0].case.raw_query)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    rng = random.Random(seed)
    for resolved_case in resolved:
        document_ids, scope_source = _runtime_document_scope(rag, resolved_case)
        order = list(ARMS)
        rng.shuffle(order)
        for arm in order:
            retrieval_query_before = provider.query_calls
            retrieval_text_before = provider.text_calls
            payload, latency_ms = await _run_production_shaped_arm(
                resolved_case,
                arm=arm,
                document_ids=document_ids,
                rag=rag,
                store=store,
                embeddings=provider,
                settings=settings,
                top_k=30,
            )
            candidate_results = list(payload.get("candidate_results") or payload.get("results") or [])[:30]
            candidate_score = score_results(resolved_case, candidate_results, cutoff=30)
            rerank_query_before = provider.query_calls
            rerank_text_before = provider.text_calls
            final_results, rerank_latency_ms = await _run_final_stage(
                query=str(payload.get("query") or resolved_case.case.standalone_full),
                candidates=candidate_results,
                embeddings=provider,
                settings=settings,
                # Runtime focus/catalog scope is authoritative for selection.
                # Qrel document IDs are unordered gold scoring metadata: using
                # them here leaks targets into unscoped retrieval and makes
                # comparison ranking depend on Python's per-process hash seed.
                required_document_ids=document_ids,
            )
            final_score = score_results(resolved_case, final_results, cutoff=10)
            payload["scope_leak"] = list(final_score["forbidden_documents"])
            failures = _failure_taxonomy(
                resolved_case,
                payload=payload,
                candidates=candidate_results,
                final_results=final_results,
                rerank_applied=bool(settings.rerank_enabled),
            )
            timings = dict(payload.get("timings_ms") or {})
            timings["runner_retrieval"] = latency_ms
            timings["runner_final_stage"] = rerank_latency_ms
            timings["total"] = round(latency_ms + rerank_latency_ms, 3)
            full_words = len(resolved_case.case.standalone_full.split())
            rows_by_arm[arm].append(
                {
                    "case_id": resolved_case.case.case_id,
                    "category": resolved_case.case.category,
                    "language": resolved_case.case.language,
                    "scope": resolved_case.case.scope,
                    "effective_scope_source": scope_source,
                    "effective_focus_document_ids": list(document_ids or []),
                    "contextual": bool(resolved_case.case.conversation_context),
                    "query_length_bucket": _length_bucket(full_words),
                    "raw_query_informativeness": str(payload.get("query_variant_reason") or "single_query"),
                    "positive": resolved_case.case.positive,
                    "queries": payload.get("queries") or [resolved_case.case.standalone_full],
                    "query_origins": payload.get("query_origins") or ["single"],
                    "candidate_recall": candidate_score["evidence_recall"],
                    "evidence_recall": final_score["evidence_recall"],
                    "document_recall": final_score["document_recall"],
                    "multi_paper_coverage": final_score["multi_paper_coverage"],
                    "modality_recall": final_score["modality_recall"],
                    "hit_at_1": final_score["hit_at_1"],
                    "hit_at_3": final_score["hit_at_3"],
                    "hit_at_5": final_score["hit_at_5"],
                    "hit_at_10": final_score["hit_at_10"],
                    "mrr_at_10": final_score["mrr_at_10"],
                    "ndcg_at_10": final_score["ndcg_at_10"],
                    "forbidden_documents": final_score["forbidden_documents"],
                    "candidate_result_kinds": candidate_score["result_kinds"],
                    "final_result_kinds": final_score["result_kinds"],
                    "failure_taxonomy": failures,
                    "timings_ms": timings,
                    "latency_ms": timings["total"],
                    "retrieval_embedding_query_calls": rerank_query_before - retrieval_query_before,
                    "retrieval_embedding_text_calls": rerank_text_before - retrieval_text_before,
                    "rerank_embedding_query_calls": provider.query_calls - rerank_query_before,
                    "rerank_embedding_text_calls": provider.text_calls - rerank_text_before,
                    "embedding_query_calls": provider.query_calls - retrieval_query_before,
                    "dual_query_rrf": bool(payload.get("dual_query_rrf")),
                }
            )
    reports: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        metrics = _aggregate(rows)
        reports[arm] = {
            "metrics": metrics,
            "rows": rows,
            "promotion": None,
        }
    baseline_metrics = reports["A0"]["metrics"]
    for arm, report in reports.items():
        report["promotion"] = promotion_gate(
            {"metrics": report["metrics"], "qrels_resolved": True, "model_calls": 0},
            baseline=baseline_metrics if arm != "A0" else None,
        )
    comparison = _recommend_action(reports)
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_cases": len(cases),
        "arms": reports,
        "comparison": comparison,
        "paired_full_vs_anchor": _paired_delta(rows_by_arm["B-full"], rows_by_arm["B-anchor"]),
        "qrels_resolved": True,
        "final_selection_scope": "runtime_focus_or_catalog_only",
        "qrel_scope_used_for_selection": False,
        "preflight": {"status": "passed", "resolved_cases": len(resolved), "failures": []},
        "model_calls": 0,
        "rewrite_calls": 0,
        "rewrite_fingerprint": next(iter({case.rewrite_fingerprint for case in cases}), None),
        "runtime_dual_query_flag": runtime_flag,
        "seed": seed,
        "index_fingerprint": _index_fingerprint(db_path),
        "embedding": {
            "model": settings.embedding_model,
            "query_prefix": settings.embedding_query_prefix,
            "document_prefix": settings.embedding_document_prefix,
        },
        "git_sha": _git_sha(),
        "production_corpus_modified": False,
    }


def _validate_suite_shape(cases: list[EvalCase], expected: int) -> None:
    if len(cases) != expected:
        raise EvaluationContractError(f"expected {expected} cases, found {len(cases)}")
    category_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
        language_counts[case.language] = language_counts.get(case.language, 0) + 1
    expected_per_category = expected // 6
    categories = {*POSITIVE_CATEGORIES, "negative_unanswerable"}
    if any(category_counts.get(category, 0) != expected_per_category for category in categories):
        raise EvaluationContractError(f"suite category balance invalid: {category_counts}")
    if language_counts != {"vi": expected // 2, "en": expected // 2}:
        raise EvaluationContractError(f"suite language balance invalid: {language_counts}")
    contextual = sum(bool(case.conversation_context) for case in cases)
    if expected == 60 and contextual != 30:
        raise EvaluationContractError(f"suite contextual balance invalid: {contextual}/60")
    if expected == 60:
        document_counts: dict[str, int] = {}
        for case in cases:
            for group in case.evidence_groups:
                filename = str(group.document.get("filename") or "")
                if filename:
                    document_counts[filename] = document_counts.get(filename, 0) + 1
        if len(document_counts) < 21 or any(count < 2 for count in document_counts.values()):
            raise EvaluationContractError(f"suite document coverage invalid: {document_counts}")
        if max(document_counts.values(), default=0) > 8:
            raise EvaluationContractError(f"suite document dominance invalid: {document_counts}")


def suite_quality_diagnostics(cases: Sequence[EvalCase]) -> dict[str, Any]:
    """Check benchmark design validity separately from qrel resolution.

    A suite can have portable, perfectly resolving qrels and still be a poor
    experiment.  In particular, language and contextuality must not be
    perfectly confounded, and repeated template questions must not dominate a
    category.  These checks are deliberately structural: they do not inspect
    answers or tune retrieval to individual papers.
    """

    issues: list[str] = []
    matrix = {
        (language, contextual): sum(
            1
            for case in cases
            if case.language == language and bool(case.conversation_context) == contextual
        )
        for language in ("vi", "en")
        for contextual in (False, True)
    }
    if any(matrix.values()) and len([count for count in matrix.values() if count]) < 4:
        issues.append(f"language_context_confounded:{matrix}")
    if len(cases) >= 40:
        expected_cell = len(cases) // 4
        if any(count != expected_cell for count in matrix.values()):
            issues.append(f"language_context_matrix_unbalanced:{matrix}")

    by_category: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_category[case.category].append(case)
        if case.scope == "focused" and not case.focus_documents:
            issues.append(f"focused_scope_missing_catalog:{case.case_id}")
        if case.scope == "focused" and case.focus_documents:
            scoped_names = {
                normalize_text(str(item.get("filename") or ""))
                for item in case.focus_documents
                if str(item.get("filename") or "").strip()
            }
            named_context = normalize_text(
                " ".join(
                    [
                        *case.required_entities,
                        json.dumps(case.working_state, ensure_ascii=False),
                        " ".join(str(item.get("content") or "") for item in case.conversation_context),
                    ]
                )
            )
            for filename in scoped_names:
                if filename and filename not in named_context:
                    issues.append(f"focused_scope_not_grounded:{case.case_id}:{filename[:48]}")
    for category, category_cases in sorted(by_category.items()):
        category_matrix = {
            (language, contextual): sum(
                1
                for case in category_cases
                if case.language == language and bool(case.conversation_context) == contextual
            )
            for language in ("vi", "en")
            for contextual in (False, True)
        }
        if len(category_cases) >= 8 and any(count < 1 for count in category_matrix.values()):
            issues.append(f"category_language_context_gap:{category}={category_matrix}")
        raw_counts = Counter(normalize_text(case.raw_query) for case in category_cases)
        if raw_counts:
            most_common_query, count = raw_counts.most_common(1)[0]
            if len(category_cases) >= 8 and count / len(category_cases) > 0.4:
                issues.append(
                    f"template_dominance:{category}:{count}/{len(category_cases)}:{most_common_query[:48]}"
                )

    return {"promotion_safe": not issues, "issues": issues, "language_context_matrix": matrix}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.jsonl",
        help="audited Gate-C suite (v2.1 is retained as a diagnostic-only fixture)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--lancedb-path", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--open-shadow", action="store_true")
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--expected-rewrite-fingerprint",
        default=None,
        help="Fail closed unless the frozen audited suite has this rewrite-policy fingerprint.",
    )
    parser.add_argument("--allow-draft", action="store_true", help="Allow non-audited fixtures for local contract debugging only")
    args = parser.parse_args()
    cases = load_cases(
        args.suite,
        allow_shadow=args.open_shadow,
        qrels_path=args.qrels,
        require_audited=not args.allow_draft,
        expected_rewrite_fingerprint=args.expected_rewrite_fingerprint,
    )
    if args.expected_cases is not None:
        _validate_suite_shape(cases, args.expected_cases)
    settings = get_settings()
    db_path = args.db_path or settings.sqlite_db_path
    lancedb_path = args.lancedb_path or settings.lancedb_path
    artifact_root = args.artifact_root or settings.artifacts_path
    try:
        _validate_revision_aware_eval_db(db_path)
        report = asyncio.run(
            evaluate(
                cases,
                db_path=db_path,
                lancedb_path=lancedb_path,
                artifact_root=artifact_root,
                seed=args.seed,
            )
        )
    except LanceDBUnavailable as exc:
        raise SystemExit(f"LanceDB unavailable: {exc}") from exc
    except EmbeddingError as exc:
        raise SystemExit(f"Embedding provider unavailable: {exc}") from exc
    if args.allow_draft:
        # A structural proposal can exercise resolver/schema/timing code, but
        # it is not a source of promotion evidence.  Make that fact durable in
        # the JSON itself so a copied metric table cannot lose the caveat.
        report["status"] = "draft_contract_smoke_invalid_for_promotion"
        report["promotion_eligible"] = False
        report["comparison"]["recommended_action"] = "do_not_promote_draft_suite"
        for arm_report in report["arms"].values():
            arm_report["promotion"] = {
                "passed": False,
                "checks": {"suite_audited": False},
                "blockers": ["suite_not_audited"],
            }
    else:
        report["status"] = "audited_evaluation"
        action = str(report["comparison"].get("recommended_action") or "")
        report["promotion_eligible"] = action in {
            "promote_candidate_union_only",
            "promote_dual_query",
        }
        report["dual_query_promotion_eligible"] = action == "promote_dual_query"
    print(json.dumps({arm: value["metrics"] for arm, value in report["arms"].items()}, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
