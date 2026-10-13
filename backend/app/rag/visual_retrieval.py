"""Feature-gated visual page retrieval contract.

The live text/table/figure LanceDB index remains the default. Visual
multi-vector implementations can plug into this protocol in an isolated
index without changing canonical evidence or answer contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class VisualPageRecord:
    page_id: str
    document_id: str
    revision_id: str
    page_number: int
    image_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisualSearchResult:
    page_id: str
    document_id: str
    revision_id: str
    page_number: int
    score: float
    image_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualRetrievalProvider(Protocol):
    """Provider contract for isolated page-image retrieval backends."""

    name: str
    model_fingerprint: str

    async def index_pages(
        self,
        pages: list[VisualPageRecord],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        ...

    async def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[VisualSearchResult]:
        ...

    async def get_pages(
        self,
        *,
        page_numbers: list[int],
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
    ) -> list[VisualSearchResult]:
        """Return explicitly requested page images without re-embedding."""
        ...

    async def delete_document(self, document_id: str, revision_id: str | None = None) -> None:
        ...


class DisabledVisualRetrievalProvider:
    """Explicit default; avoids pretending text retrieval is visual retrieval."""

    name = "disabled"
    model_fingerprint = "disabled:v1"

    async def index_pages(
        self,
        pages: list[VisualPageRecord],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        del progress_callback
        return {
            "ok": False,
            "status": "disabled",
            "indexed_pages": 0,
            "reason": "visual_retrieval_disabled",
        }

    async def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[VisualSearchResult]:
        return []

    async def get_pages(
        self,
        *,
        page_numbers: list[int],
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
    ) -> list[VisualSearchResult]:
        return []

    async def delete_document(self, document_id: str, revision_id: str | None = None) -> None:
        return None


def render_pdf_page_records(
    *,
    source_path: Path,
    document_id: str,
    revision_id: str,
    output_root: Path,
    scale: float = 1.25,
    max_render_pixels: int = 2_000_000,
) -> list[VisualPageRecord]:
    """Render every PDF page into a revision-scoped, resumable visual corpus.

    The raw source hash and rendering configuration are authoritative.  A
    revision never reuses page pixels from different bytes or a different
    render scale, and each page is atomically committed so an interrupted
    background job can resume without rerendering its completed prefix.
    """

    source = source_path.expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != ".pdf":
        raise ValueError(f"Visual PDF source does not exist: {source}")
    if not document_id.strip() or not revision_id.strip():
        raise ValueError("document_id and revision_id are required")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if max_render_pixels < 1:
        raise ValueError("max_render_pixels must be positive")

    source_sha = _sha256_file(source)
    render_config = {
        "schema": "visual-pages-v1",
        "source_sha256": source_sha,
        "document_id": document_id,
        "revision_id": revision_id,
        "scale": round(float(scale), 4),
        "max_render_pixels": int(max_render_pixels),
        "format": "png",
    }
    render_fingerprint = hashlib.sha256(
        json.dumps(render_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    page_dir = output_root.expanduser().resolve() / render_fingerprint
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "manifest.json"

    import fitz

    document = fitz.open(source)
    try:
        records: list[VisualPageRecord] = []
        page_entries: list[dict[str, Any]] = []
        for index in range(document.page_count):
            page_number = index + 1
            image_path = page_dir / f"page-{page_number:05d}.png"
            page = document.load_page(index)
            rect = page.rect
            source_area = max(float(rect.width) * float(rect.height), 1.0)
            requested_pixels = source_area * float(scale) * float(scale)
            effective_scale = float(scale)
            if requested_pixels > max_render_pixels:
                effective_scale *= (max_render_pixels / requested_pixels) ** 0.5
            # Preserve a useful image rather than rounding one dimension to
            # zero for a pathological PDF media box.
            effective_scale = max(effective_scale, 1.0 / max(rect.width, rect.height, 1.0))
            if not image_path.is_file() or image_path.stat().st_size == 0:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(effective_scale, effective_scale),
                    alpha=False,
                )
                rendered_width, rendered_height = pixmap.width, pixmap.height
                payload = pixmap.tobytes("png")
                temporary = image_path.with_suffix(".png.tmp")
                temporary.write_bytes(payload)
                temporary.replace(image_path)
            else:
                from PIL import Image

                with Image.open(image_path) as existing:
                    rendered_width, rendered_height = existing.size
            image_sha = _sha256_file(image_path)
            page_id = f"page:{document_id}:{revision_id}:{page_number}"
            metadata = {
                "artifact_type": "page",
                "source_path": str(source),
                "source_sha256": source_sha,
                "image_sha256": image_sha,
                "render_fingerprint": render_fingerprint,
                "requested_render_scale": round(float(scale), 6),
                "effective_render_scale": round(effective_scale, 6),
                "render_max_pixels": int(max_render_pixels),
                "image_pixels": int(rendered_width * rendered_height),
                "renderer": f"pymupdf:{getattr(fitz, '__version__', 'unknown')}",
            }
            records.append(
                VisualPageRecord(
                    page_id=page_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    page_number=page_number,
                    image_path=str(image_path),
                    metadata=metadata,
                )
            )
            page_entries.append(
                {
                    "page_id": page_id,
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "image_sha256": image_sha,
                    "requested_render_scale": round(float(scale), 6),
                    "effective_render_scale": round(effective_scale, 6),
                }
            )
            # Persist a resumable checkpoint after every committed page.  A
            # render can be interrupted by host sleep, a worker restart or a
            # malformed later page; already-rendered images must remain
            # discoverable instead of waiting for the whole PDF to finish.
            # The manifest is atomically replaced, so readers see either the
            # previous complete prefix or the new complete prefix, never a
            # half-written JSON document.
            _atomic_json_write(
                manifest_path,
                {
                    **render_config,
                    "render_fingerprint": render_fingerprint,
                    "status": "partial",
                    "page_count": document.page_count,
                    "completed_pages": len(page_entries),
                    "pages": page_entries,
                },
            )
        _atomic_json_write(
            manifest_path,
            {
                **render_config,
                "render_fingerprint": render_fingerprint,
                "status": "complete",
                "page_count": document.page_count,
                "completed_pages": len(page_entries),
                "pages": page_entries,
            },
        )
        return records
    finally:
        document.close()


def page_records_from_document_rows(rows: list[dict[str, Any]]) -> list[VisualPageRecord]:
    """Build revision-aware page records from canonical figure/page rows."""

    records: list[VisualPageRecord] = []
    for row in rows:
        image_path = str(row.get("image_path") or "").strip()
        if not image_path or not Path(image_path).is_file():
            continue
        document_id = str(row.get("document_id") or "").strip()
        revision_id = str(row.get("revision_id") or "").strip()
        if not document_id or not revision_id:
            continue
        page_number = row.get("page_number")
        synthetic_page = page_number is None
        if page_number is None:
            figure_index = row.get("figure_index")
            if figure_index is None:
                continue
            page_number = int(figure_index) + 1
        raw_metadata = row.get("metadata") or row.get("metadata_json") or {}
        if isinstance(raw_metadata, str):
            try:
                raw_metadata = json.loads(raw_metadata)
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_metadata = {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                "artifact_type": row.get("asset_type")
                or row.get("artifact_type")
                or metadata.get("asset_type")
                or "page",
                "caption": row.get("caption"),
                "page_number_is_synthetic": synthetic_page,
            }
        )
        page_id = str(row.get("id") or f"page:{document_id}:{revision_id}:{page_number}")
        records.append(
            VisualPageRecord(
                page_id=page_id,
                document_id=document_id,
                revision_id=revision_id,
                page_number=int(page_number),
                image_path=image_path,
                metadata=metadata,
            )
        )
    return records


def visual_search_results_as_evidence(
    results: list[VisualSearchResult],
    *,
    filename_by_document: dict[str, str] | None = None,
    source_path_by_document: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert visual page hits into the canonical evidence shape.

    Content intentionally describes provenance only.  OCR/captions may be
    attached by callers, but the retriever must never hallucinate page text.
    """

    filenames = filename_by_document or {}
    source_paths = source_path_by_document or {}
    return [
        {
            "id": result.page_id,
            "chunk_id": result.page_id,
            "document_id": result.document_id,
            "revision_id": result.revision_id,
            "filename": filenames.get(result.document_id),
            "source_path": source_paths.get(result.document_id),
            "page_number": result.page_number,
            "artifact_type": "page",
            "chunk_type": "page",
            "image_path": result.image_path,
            "content": "Visual page evidence selected from the active document revision.",
            "score": result.score,
            "visual_score": result.score,
            "retrieval_channels": ["visual_page", result.metadata.get("provider", "visual")],
            "metadata": dict(result.metadata),
        }
        for result in results
    ]


def reciprocal_rank_fuse_keys(
    rankings: list[list[str]],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[str]:
    """Fuse heterogeneous text/visual rankings by stable canonical key.

    Scores from BM25/vector/ColSmol are not calibrated to one another.  Rank
    fusion avoids pretending they are, deduplicates repeated chunks from the
    same page/artifact within each channel, and uses lexical key order only as
    a deterministic final tie-break.
    """

    if rrf_k < 1 or top_k < 1:
        raise ValueError("rrf_k and top_k must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for raw_key in ranking:
            key = str(raw_key).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            rank += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], best_rank[key], key))
    return ordered[:top_k]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
