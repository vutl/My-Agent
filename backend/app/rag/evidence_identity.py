"""Stable identities for evidence across retrieval, fusion, and evaluation.

Retrieval results can carry both a vector-row ``chunk_id`` and a canonical
table/figure record ID.  The latter is authoritative for visual artifacts.
Keeping this rule in one module prevents a typed :class:`EvidenceBundle` from
being scored differently from the legacy result shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


EvidenceIdentityKind = Literal["text", "table", "figure", "page"]


def canonical_artifact_type(
    payload: Mapping[str, Any],
    *,
    default: str | None = None,
) -> str:
    """Normalize an artifact modality across legacy and adapter result shapes.

    Some PDF/benchmark adapters persist an image-backed table in the
    ``document_figures`` table (because it is physically an image) while
    storing ``asset_type=table`` in metadata.  Treating that row as a figure
    loses table reservations, row/column prompting and table qrel identity.
    Metadata is therefore allowed to refine the physical storage type, while
    page/visual-page remain explicit page projections.
    """

    raw = str(
        payload.get("artifact_type")
        or payload.get("chunk_type")
        or payload.get("kind")
        or default
        or ""
    ).strip().lower()
    if raw in {"page", "visual_page"}:
        return raw
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        declared = str(
            metadata.get("asset_type")
            or metadata.get("artifact_type")
            or metadata.get("asset_kind")
            or ""
        ).strip().lower()
        if declared in {"table", "figure", "image", "page", "visual_page"}:
            # A table declaration is authoritative even when the storage row
            # came from the figures table.  Other declarations refine only an
            # untyped/visual row; an explicit canonical table/figure wins.
            if declared == "table" or raw not in {"table", "figure", "image"}:
                return declared
    if raw in {"table", "figure", "image"}:
        return raw
    if payload.get("table_id"):
        return "table"
    if payload.get("figure_id"):
        return "figure"
    return raw or "text"


def evidence_kind(payload: Mapping[str, Any]) -> EvidenceIdentityKind:
    """Return the modality represented by a legacy or typed evidence payload."""

    value = canonical_artifact_type(payload)
    if value == "visual_page":
        return "page"
    if value in {"table", "figure", "page"}:
        return value
    if payload.get("table_id"):
        return "table"
    if payload.get("figure_id"):
        return "figure"
    return "text"


def canonical_evidence_id(payload: Mapping[str, Any]) -> str:
    """Return a tagged, source-stable identity for one evidence unit.

    Table and figure IDs win over their vector chunk IDs.  A page has no
    independent record ID, so its document/page locator is the identity.
    Existing tagged IDs remain idempotent to make round-trips safe.
    """

    kind = evidence_kind(payload)
    if kind == "table":
        return _tagged(
            "table",
            payload.get("table_id")
            or payload.get("figure_id")
            or payload.get("evidence_id")
            or payload.get("chunk_id")
            or payload.get("id"),
        )
    if kind == "figure":
        return _tagged("figure", payload.get("figure_id") or payload.get("evidence_id") or payload.get("chunk_id") or payload.get("id"))
    if kind == "page":
        document_id = _text(payload.get("document_id"))
        page_number = payload.get("page_number")
        if document_id and page_number is not None:
            return f"page:{document_id}:{page_number}"
        return _tagged("page", payload.get("evidence_id") or payload.get("chunk_id") or payload.get("id"))
    return _tagged("text", payload.get("chunk_id") or payload.get("evidence_id") or payload.get("id"))


def canonical_evidence_ids(payload: Mapping[str, Any]) -> set[str]:
    """Return every safe identity represented by ``payload``.

    The canonical ID is sufficient for new code.  The compatible tagged raw
    aliases keep evaluator fixtures readable while all callers migrate.
    """

    values = {canonical_evidence_id(payload)}
    kind = evidence_kind(payload)
    for key in ("evidence_id", "chunk_id", "table_id", "figure_id", "id"):
        value = _text(payload.get(key))
        if value:
            values.add(_tagged(kind, value))
    return {value for value in values if value}


def _tagged(kind: str, value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    prefix = f"{kind}:"
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _text(value: Any) -> str:
    return str(value or "").strip()


# Formats whose ``page_number`` is a physical rendered page. Markdown, text and
# DOCX parsers put every artifact on page 1, so "same page" means nothing there.
_PAGED_SUFFIXES = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif",
    ".ppt", ".pptx", ".odp",
})
_UNPAGED_SUFFIXES = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".docx", ".doc", ".odt",
    ".rtf", ".csv", ".json",
})
SAME_PAGE_CONTEXT_MARKER = "[same-page canonical artifact context]"


def has_physical_pages(payload: Mapping[str, Any]) -> bool:
    """Whether page identity is meaningful for this evidence's document.

    Unknown formats keep page semantics (the previous behaviour)."""
    for field in ("filename", "source_path"):
        name = str(payload.get(field) or "").strip().lower()
        if "." not in name:
            continue
        suffix = name[name.rfind("."):]
        if suffix in _PAGED_SUFFIXES:
            return True
        if suffix in _UNPAGED_SUFFIXES:
            return False
    return True


def own_evidence_text(text: str) -> str:
    """Text that belongs to this evidence, without projected sibling context."""
    return str(text or "").split(SAME_PAGE_CONTEXT_MARKER, 1)[0]
