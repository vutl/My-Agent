"""Small, local OCR adapter for approved image artifacts.

This is intentionally separate from page OCR: image artifacts in Markdown,
DOCX and dataset adapters have no searchable text layer.  The result is a
derived diagnostic field, keyed by the image SHA-256, and never replaces the
original image/provenance or claims to be a canonical table extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any


@dataclass(frozen=True)
class ImageOCRResult:
    text: str
    status: str
    image_sha256: str
    engine: str = "rapidocr"


_ENGINE: Any | None = None
_ENGINE_ERROR: str | None = None
_ENGINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _rapidocr_engine() -> tuple[Any | None, str | None]:
    global _ENGINE, _ENGINE_ERROR
    # Cache successful construction, but do not permanently poison the
    # process after a transient import/runtime failure.  Optional OCR may be
    # installed or become usable later in a long-lived folder worker (and
    # tests may provide an isolated adapter); a subsequent call should be
    # allowed to retry rather than silently returning the first error forever.
    if _ENGINE is not None:
        return _ENGINE, _ENGINE_ERROR
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE, _ENGINE_ERROR
        try:
            from rapidocr import RapidOCR  # type: ignore
            from rapidocr.utils.typings import EngineType  # type: ignore

            params = {f"{stage}.engine_type": EngineType.TORCH for stage in ("Det", "Cls", "Rec")}
            _ENGINE = RapidOCR(params=params)
        except Exception as exc:  # OCR remains optional enrichment.
            _ENGINE_ERROR = f"{type(exc).__name__}"
    return _ENGINE, _ENGINE_ERROR


def apple_vision_page_text(png_bytes: bytes) -> str | None:
    """Page OCR with the macOS Vision framework, layout kept: text boxes on one
    visual line are joined left to right (wide gaps become column breaks), so
    scanned tables keep their rows. Vision reads Vietnamese diacritics that the
    bundled RapidOCR models drop. ``None`` off macOS or when Vision fails."""
    if sys.platform != "darwin":
        return None
    try:
        from io import BytesIO

        from ocrmac import ocrmac  # type: ignore
        from PIL import Image

        with _INFERENCE_LOCK:
            results = ocrmac.OCR(
                Image.open(BytesIO(png_bytes)),
                recognition_level="accurate",
                language_preference=["vi-VT", "en-US"],
            ).recognize()
    except Exception:  # noqa: BLE001 - optional engine; RapidOCR is the fallback
        return None
    return layout_lines(
        [(str(text), x, 1.0 - (y + height), width, height) for text, _confidence, (x, y, width, height) in results]
    )


def layout_lines(boxes: list[tuple[str, float, float, float, float]]) -> str:
    """Rebuild reading lines from OCR boxes (text, left, top, width, height in
    one coordinate system, y growing downwards): boxes whose vertical centres
    are within half a line height form one line, ordered left to right; a gap
    wider than a line height becomes a column break."""
    items = [box for box in boxes if str(box[0]).strip()]
    if not items:
        return ""
    heights = sorted(box[4] for box in items)
    line_height = heights[len(heights) // 2] or 1e-6
    items.sort(key=lambda box: (box[2] + box[4] / 2, box[1]))
    lines: list[list[tuple[str, float, float, float, float]]] = []
    centres: list[float] = []
    for box in items:
        centre = box[2] + box[4] / 2
        if lines and abs(centre - centres[-1]) <= 0.5 * line_height:
            lines[-1].append(box)
            centres[-1] += (centre - centres[-1]) / len(lines[-1])
        else:
            lines.append([box])
            centres.append(centre)
    rendered = []
    for line in lines:
        line.sort(key=lambda box: box[1])
        parts = [line[0][0].strip()]
        for previous, current in zip(line, line[1:]):
            gap = current[1] - (previous[1] + previous[3])
            parts.append(("   " if gap > 1.2 * line_height else " ") + current[0].strip())
        rendered.append("".join(parts))
    return "\n".join(rendered)


def get_rapidocr_engine() -> tuple[Any | None, str | None]:
    """Return the process-local OCR engine used by all local OCR paths.

    Page OCR and image-artifact OCR share the same RapidOCR weights.  Keeping
    one lazy singleton avoids reloading the detector/classifier/recognizer for
    every document during a folder scan or public benchmark, while the caller
    still decides whether an unavailable engine is a hard or soft failure.
    """

    return _rapidocr_engine()


def _normalise_text(values: Any, *, max_chars: int) -> str:
    if isinstance(values, str):
        raw = values
    else:
        raw = "\n".join(str(value).strip() for value in (values or ()) if str(value).strip())
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def extract_image_ocr(
    image_path: Path,
    *,
    cache_dir: Path | None = None,
    max_chars: int = 12_000,
) -> ImageOCRResult:
    """Return cached/local OCR text without changing canonical image identity."""

    try:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    except OSError as exc:
        return ImageOCRResult("", "unavailable", "", engine=f"rapidocr:{type(exc).__name__}")

    cache_path = cache_dir / f"{digest}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_text = _normalise_text(payload.get("text"), max_chars=max_chars)
            return ImageOCRResult(
                cached_text,
                str(payload.get("status") or ("text" if cached_text else "empty")),
                digest,
                str(payload.get("engine") or "rapidocr"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    text = ""
    status = "empty"
    engine_name = "rapidocr"
    try:
        engine, engine_error = _rapidocr_engine()
        if engine is None:
            raise RuntimeError(engine_error or "rapidocr_unavailable")
        with _INFERENCE_LOCK:
            result = engine(image_path.read_bytes())
        text = _normalise_text(getattr(result, "txts", None), max_chars=max_chars)
        status = "text" if text else "empty"
    except Exception as exc:  # OCR is enrichment; it must never fail parsing.
        status = "failed"
        engine_name = f"rapidocr:{type(exc).__name__}"

    result = ImageOCRResult(text, status, digest, engine_name)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": "image-ocr-v1",
                        "image_sha256": digest,
                        "text": text,
                        "status": status,
                        "engine": engine_name,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        except OSError:
            pass
    return result
