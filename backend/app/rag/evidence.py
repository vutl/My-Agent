"""Typed, modality-aware evidence contracts for retrieval and future tools.

The runtime still accepts the historical ``dict`` source shape at the context
boundary.  These contracts provide a lossless, typed envelope around that
shape so retrieval, validation, and future tools can share provenance without
flattening tables and figures into indistinguishable text snippets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.rag.evidence_identity import canonical_evidence_id, evidence_kind


EvidenceKind = Literal["text", "table", "figure", "page"]
ReadinessState = Literal["ready", "partial", "failed"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceScores:
    """Retrieval scores retained for audit rather than collapsed into one float."""

    lexical: float | None = None
    vector: float | None = None
    rrf: float | None = None
    rerank: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "lexical": self.lexical,
            "vector": self.vector,
            "rrf": self.rrf,
            "rerank": self.rerank,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceCandidate:
    """Common envelope for one retrievable unit.

    ``kind`` is the tagged-union discriminator.  Subclasses below make the
    expected modality explicit to callers while preserving one serializable
    envelope for tool integrations.
    """

    evidence_id: str
    kind: EvidenceKind
    content: str
    document_id: str | None = None
    source_path: str | None = None
    filename: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    parent_id: str | None = None
    facets: tuple[str, ...] = ()
    query_origins: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    scores: EvidenceScores = field(default_factory=EvidenceScores)
    extraction_confidence: float | None = None
    readiness: ReadinessState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "content": self.content,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "filename": self.filename,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "parent_id": self.parent_id,
            "facets": list(self.facets),
            "query_origins": list(self.query_origins),
            "provenance": dict(self.provenance),
            "scores": self.scores.to_dict(),
            "extraction_confidence": self.extraction_confidence,
            "readiness": self.readiness,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TextEvidence(EvidenceCandidate):
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True, kw_only=True)
class TableEvidence(EvidenceCandidate):
    kind: Literal["table"] = "table"
    table_id: str | None = None
    table_index: int | None = None
    caption: str | None = None
    image_path: str | None = None
    image_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = EvidenceCandidate.to_dict(self)
        value.update(
            {
                "table_id": self.table_id,
                "table_index": self.table_index,
                "caption": self.caption,
                "image_path": self.image_path,
                "image_url": self.image_url,
            }
        )
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FigureEvidence(EvidenceCandidate):
    kind: Literal["figure"] = "figure"
    figure_id: str | None = None
    figure_index: int | None = None
    caption: str | None = None
    image_path: str | None = None
    image_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = EvidenceCandidate.to_dict(self)
        value.update(
            {
                "figure_id": self.figure_id,
                "figure_index": self.figure_index,
                "caption": self.caption,
                "image_path": self.image_path,
                "image_url": self.image_url,
            }
        )
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PageEvidence(EvidenceCandidate):
    kind: Literal["page"] = "page"
    image_path: str | None = None
    image_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = EvidenceCandidate.to_dict(self)
        value.update({"image_path": self.image_path, "image_url": self.image_url})
        return value


def evidence_candidate_from_result(
    result: dict[str, Any],
    *,
    query_origins: tuple[str, ...] = (),
    readiness: ReadinessState | None = None,
) -> EvidenceCandidate:
    """Convert a legacy retrieval result without dropping provenance fields."""

    artifact_type = evidence_kind(result)
    evidence_id = canonical_evidence_id(result)
    scores = EvidenceScores(
        lexical=_optional_float(result.get("raw_fts_score")),
        vector=_optional_float(result.get("raw_vector_score")),
        rrf=_optional_float(result.get("rrf_score")),
        rerank=_optional_float(result.get("rerank_score") or result.get("rerank_boost")),
    )
    common = {
        "evidence_id": evidence_id,
        "content": str(result.get("content") or result.get("expanded_content") or ""),
        "document_id": _optional_str(result.get("document_id")),
        "source_path": _optional_str(result.get("source_path")),
        "filename": _optional_str(result.get("filename")),
        "page_number": _optional_int(result.get("page_number")),
        "section_title": _optional_str(result.get("section_title")),
        "parent_id": _optional_str(result.get("parent_chunk_id")),
        "facets": tuple(str(item) for item in result.get("evidence_facets") or ()),
        "query_origins": query_origins or tuple(str(item) for item in result.get("query_origins") or ()),
        "provenance": {
            "retrieval_channels": list(result.get("retrieval_channels") or []),
            "heading_path": list(result.get("heading_path") or []),
            "chunk_index": result.get("chunk_index"),
        },
        "scores": scores,
        "extraction_confidence": _optional_float(result.get("extraction_confidence")),
        "readiness": readiness,
    }
    if artifact_type == "table":
        return TableEvidence(
            **common,
            # An image-backed table may have been physically stored in the
            # figures projection.  Its figure ID remains a stable artifact
            # alias until a table row is available, but the typed contract is
            # still table-shaped for routing/validation.
            table_id=_optional_str(result.get("table_id") or result.get("figure_id")),
            table_index=_optional_int(result.get("table_index")),
            caption=_optional_str(result.get("caption")),
            image_path=_optional_str(result.get("image_path")),
            image_url=_optional_str(result.get("image_url")),
        )
    if artifact_type == "figure":
        return FigureEvidence(
            **common,
            figure_id=_optional_str(result.get("figure_id")),
            figure_index=_optional_int(result.get("figure_index")),
            caption=_optional_str(result.get("caption")),
            image_path=_optional_str(result.get("image_path")),
            image_url=_optional_str(result.get("image_url")),
        )
    if artifact_type == "page":
        return PageEvidence(
            **common,
            image_path=_optional_str(result.get("image_path")),
            image_url=_optional_str(result.get("image_url")),
        )
    return TextEvidence(**common)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceBundle:
    """Evidence envelope shared by retrieval and future tool adapters."""

    query: str
    query_origins: tuple[str, ...] = ()
    candidates: tuple[EvidenceCandidate, ...] = ()
    selected: tuple[EvidenceCandidate, ...] = ()
    document_ids: tuple[str, ...] = ()
    facet_coverage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    readiness_by_document: dict[str, ReadinessState] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_origins": list(self.query_origins),
            "candidates": [item.to_dict() for item in self.candidates],
            "selected": [item.to_dict() for item in self.selected],
            "document_ids": list(self.document_ids),
            "facet_coverage": {key: list(value) for key, value in self.facet_coverage.items()},
            "readiness_by_document": dict(self.readiness_by_document),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_results(
        cls,
        *,
        query: str,
        results: list[dict[str, Any]],
        selected: list[dict[str, Any]] | None = None,
        query_origins: tuple[str, ...] = (),
        readiness_by_document: dict[str, ReadinessState] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "EvidenceBundle":
        candidates = tuple(
            evidence_candidate_from_result(
                result,
                query_origins=tuple(result.get("query_origins") or query_origins),
                readiness=(readiness_by_document or {}).get(str(result.get("document_id"))),
            )
            for result in results
        )
        selected_candidates = tuple(
            evidence_candidate_from_result(
                result,
                query_origins=tuple(result.get("query_origins") or query_origins),
                readiness=(readiness_by_document or {}).get(str(result.get("document_id"))),
            )
            for result in (selected if selected is not None else results)
        )
        facet_map: dict[str, set[str]] = {}
        for item in selected_candidates:
            if item.document_id:
                facet_map.setdefault(item.document_id, set()).update(item.facets)
        return cls(
            query=query,
            query_origins=query_origins,
            candidates=candidates,
            selected=selected_candidates,
            document_ids=tuple(
                dict.fromkeys(
                    item.document_id
                    for item in selected_candidates
                    if item.document_id
                )
            ),
            facet_coverage={key: tuple(sorted(value)) for key, value in facet_map.items()},
            readiness_by_document=dict(readiness_by_document or {}),
            diagnostics=dict(diagnostics or {}),
        )


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None
