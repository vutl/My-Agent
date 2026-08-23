"""Deterministic page/document readiness diagnostics for arbitrary uploads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.parsers import ParsedDocument, ParsedPage


PageReadinessState = Literal["native_text_ready", "ocr_text_ready", "visual_only", "unresolved"]
DocumentReadinessState = Literal["ready", "partial", "failed"]


@dataclass(frozen=True, slots=True)
class PageReadiness:
    page_number: int
    state: PageReadinessState
    character_count: int
    alphanumeric_ratio: float
    text_origin: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentReadinessReport:
    state: DocumentReadinessState
    page_count: int
    native_text_pages: int
    ocr_text_pages: int
    visual_only_pages: int
    unresolved_pages: int
    table_count: int
    figure_count: int
    figure_quality_counts: dict[str, int] = field(default_factory=dict)
    parser_fingerprint: str = ""
    warnings: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    pages: tuple[PageReadiness, ...] = ()

    @property
    def retrievable_page_count(self) -> int:
        return self.native_text_pages + self.ocr_text_pages

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pages"] = [page.to_dict() for page in self.pages]
        return payload


def assess_document_readiness(
    parsed: "ParsedDocument",
    *,
    parser_error: str | None = None,
) -> DocumentReadinessReport:
    """Classify extracted pages without pretending images are text.

    The thresholds are intentionally conservative and document-agnostic.  A
    page with no text is ``visual_only``; non-empty but unusable text is
    ``unresolved`` so callers can decide whether an OCR/layout retry is worth
    the cost.  The report is diagnostic and does not mutate the parsed object.
    """

    pages = tuple(parsed.pages or ())
    if pages:
        page_reports = tuple(_classify_page(page) for page in pages)
    elif str(parsed.text or "").strip():
        # DOCX and plain-text parsers do not expose physical pages, but their
        # text is still a valid searchable unit. Represent it as one logical
        # native page instead of reporting zero retrievable pages.
        compact = re.sub(r"\s+", "", str(parsed.text))
        page_reports = (
            PageReadiness(
                page_number=1,
                state="native_text_ready",
                character_count=len(compact),
                alphanumeric_ratio=(
                    sum(char.isalnum() for char in compact) / len(compact)
                    if compact
                    else 0.0
                ),
                text_origin="native",
            ),
        )
    else:
        page_reports = ()
    native = sum(item.state == "native_text_ready" for item in page_reports)
    ocr = sum(item.state == "ocr_text_ready" for item in page_reports)
    visual = sum(item.state == "visual_only" for item in page_reports)
    unresolved = sum(item.state == "unresolved" for item in page_reports)
    table_count = len(parsed.tables or ())
    figure_count = len(parsed.figures or ())
    quality_counts: dict[str, int] = {}
    for figure in parsed.figures or ():
        quality = str((figure.metadata or {}).get("quality_status") or "unknown")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    warnings: list[str] = []
    failure_reasons: list[str] = []
    if parser_error:
        failure_reasons.append(parser_error)
    if unresolved:
        warnings.append(f"{unresolved} page(s) have text but are not reliably searchable")
    if visual:
        warnings.append(f"{visual} page(s) have no text layer and remain visual-only")
    if not (native or ocr or table_count or figure_count):
        failure_reasons.append("no_retrievable_text_or_artifacts")

    page_count = int(parsed.page_count or len(pages) or (1 if parsed.text.strip() else 0))
    if failure_reasons:
        state: DocumentReadinessState = "failed" if not (native or ocr or table_count or figure_count) else "partial"
    elif unresolved or visual:
        state = "partial"
    else:
        state = "ready"
    return DocumentReadinessReport(
        state=state,
        page_count=page_count,
        native_text_pages=native,
        ocr_text_pages=ocr,
        visual_only_pages=visual,
        unresolved_pages=unresolved,
        table_count=table_count,
        figure_count=figure_count,
        figure_quality_counts=quality_counts,
        parser_fingerprint=f"{parsed.parser_name}:{parsed.parser_version}",
        warnings=tuple(warnings),
        failure_reasons=tuple(failure_reasons),
        pages=page_reports,
    )


def _classify_page(page: "ParsedPage") -> PageReadiness:
    text = str(page.text or "")
    compact = re.sub(r"\s+", "", text)
    character_count = len(compact)
    alphanumeric = sum(char.isalnum() for char in compact)
    ratio = alphanumeric / character_count if character_count else 0.0
    origin = str(getattr(page, "text_origin", "native") or "native").lower()
    if character_count == 0:
        state: PageReadinessState = "visual_only"
        warnings = ("no_searchable_text",)
    elif character_count >= 40 and ratio >= 0.45:
        state = "ocr_text_ready" if origin == "ocr" else "native_text_ready"
        warnings = ()
    elif character_count < 40:
        state = "unresolved"
        warnings = ("sparse_text_layer",)
    else:
        state = "unresolved"
        warnings = ("low_alphanumeric_signal",)
    return PageReadiness(
        page_number=int(page.page_number),
        state=state,
        character_count=character_count,
        alphanumeric_ratio=ratio,
        text_origin=origin,
        warnings=warnings,
    )
