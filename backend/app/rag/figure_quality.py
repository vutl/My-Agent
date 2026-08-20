from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


_FIGURE_LABEL_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|hình)\s*(\d+)(?:\s*([a-z]))?\s*[\.:–-]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FigureLabel:
    number: int
    label: str
    panel: str | None = None


@dataclass(frozen=True)
class AssetQualityDecision:
    status: str
    asset_kind: str
    confidence: float
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.asset_kind in {"figure", "panel"}

    def metadata_patch(self) -> dict[str, Any]:
        return {
            "quality_status": self.status,
            "asset_kind": self.asset_kind,
            "quality_confidence": round(self.confidence, 3),
            "quality_reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VisualAssetGroup:
    page_number: int | None
    member_indices: tuple[int, ...]
    bbox: dict[str, float] | None
    logical_group_id: str
    figure_label: str | None = None
    figure_number: int | None = None


def extract_figure_label(caption: str | None) -> FigureLabel | None:
    match = _FIGURE_LABEL_RE.match((caption or "").strip())
    if not match:
        return None
    number = int(match.group(2))
    panel = (match.group(3) or "").lower() or None
    prefix = "Hình" if match.group(1).lower().startswith("hình") else "Figure"
    label = f"{prefix} {number}"
    if panel:
        label += panel
    return FigureLabel(number=number, label=label, panel=panel)


def classify_visual_asset(
    *,
    caption: str | None,
    extraction_method: str | None,
    bbox: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    repeated_asset_count: int = 1,
) -> AssetQualityDecision:
    """Classify a raw visual candidate without paper-specific rules.

    This is deliberately conservative. Ambiguous assets remain ``needs_review``
    so a structured VLM pass can decide; they are not silently treated as
    retrieval-ready scientific figures.
    """

    metadata = metadata or {}
    method = str(extraction_method or "").lower()
    caption_text = " ".join(str(caption or "").split())
    lowered = caption_text.lower()
    reasons: list[str] = []

    asset_type = str(metadata.get("asset_type") or "").lower()
    if asset_type == "page" or method in {"page_screenshot", "page_visual_fallback", "visual_fallback"}:
        return AssetQualityDecision("rejected", "page", 0.99, ("page_render_fallback",))

    if "visual fallback" in lowered:
        return AssetQualityDecision("rejected", "page", 0.99, ("page_render_fallback",))

    caption_source = str(metadata.get("caption_source") or "").lower()
    direct_caption = caption_source in {"docling_direct", "spatial_anchor", "caption_crop"}
    label = extract_figure_label(caption_text)
    if label:
        reasons.append("figure_caption")
    if direct_caption:
        reasons.append("direct_caption_anchor")

    width = _number(metadata.get("width"))
    height = _number(metadata.get("height"))
    if width and height:
        if width < 32 or height < 32:
            return AssetQualityDecision("rejected", "decorative", 0.98, ("tiny_raster",))
        if width * height < 25_000 and not direct_caption:
            reasons.append("small_raster_without_direct_caption")

    page_width = _number(metadata.get("page_width"))
    page_height = _number(metadata.get("page_height"))
    normalized = normalized_bbox(bbox, page_width=page_width, page_height=page_height)
    area_ratio = _bbox_area_ratio(normalized)
    if area_ratio is not None:
        if area_ratio < 0.002 and not direct_caption:
            reasons.append("tiny_page_area")
        center_y = (normalized["y0"] + normalized["y1"]) / 2
        if (center_y < 0.09 or center_y > 0.91) and area_ratio < 0.08 and not direct_caption:
            reasons.append("header_footer_asset")

    if repeated_asset_count >= 2 and not direct_caption:
        reasons.append("repeated_across_pages")

    if any(reason in reasons for reason in ("header_footer_asset", "repeated_across_pages")):
        return AssetQualityDecision("rejected", "branding", 0.94, tuple(reasons))

    if "small_raster_without_direct_caption" in reasons or "tiny_page_area" in reasons:
        # A panel may legitimately be small; retain it for grouping/review but do
        # not index it as a standalone figure.
        return AssetQualityDecision("needs_review", "panel", 0.72, tuple(reasons))

    if direct_caption:
        return AssetQualityDecision("accepted", "figure", 0.95, tuple(reasons))
    if label and caption_source not in {"page_sequence", "fallback_sequence"}:
        return AssetQualityDecision("accepted", "figure", 0.88, tuple(reasons))

    if lowered.startswith("figure extracted from page") or not caption_text:
        reasons.append("missing_caption_anchor")
    elif label:
        reasons.append("unverified_caption_assignment")
    else:
        reasons.append("non_figure_caption")
    return AssetQualityDecision("needs_review", "unknown", 0.5, tuple(reasons))


def figure_is_indexable(
    *,
    caption: str | None,
    extraction_method: str | None,
    metadata: Mapping[str, Any] | None,
) -> bool:
    metadata = metadata or {}
    status = str(metadata.get("quality_status") or "").lower()
    kind = str(metadata.get("asset_kind") or metadata.get("asset_type") or "figure").lower()
    is_content = metadata.get("is_content")
    is_complete = metadata.get("is_complete")

    if status in {"rejected", "needs_review"}:
        return False
    if kind in {"logo", "branding", "publisher_mark", "decorative", "page", "panel"}:
        return False
    if is_content is False or is_complete is False:
        return False
    if status == "accepted":
        return True

    # Backward compatibility for artifacts created before quality metadata was
    # introduced. Generic page crops stay excluded until the document is rebuilt.
    text = " ".join(str(caption or "").lower().split())
    method = str(extraction_method or "").lower()
    return not (
        "visual fallback" in text
        or text.startswith("figure extracted from page")
        or method in {"page_screenshot", "page_visual_fallback", "visual_fallback"}
    )


def merge_visual_quality_metadata(
    metadata: dict[str, object],
    patch: dict[str, object],
) -> None:
    """Merge VLM semantics without overruling stronger geometry evidence.

    Caption-derived fallback context has no visual completeness judgement. It
    can improve search text, but cannot promote an ambiguous crop or downgrade
    a complete logical figure selected by the geometry/caption pipeline.
    """
    if not patch:
        return
    pre_status = str(metadata.get("quality_status") or "")
    patch_status = str(patch.get("quality_status") or "")
    if (
        pre_status == "needs_review"
        and patch_status == "accepted"
        and patch.get("is_complete") is not True
    ) or (
        pre_status == "accepted"
        and patch_status == "needs_review"
        and patch.get("is_complete") is None
    ):
        patch = {key: value for key, value in patch.items() if key != "quality_status"}
    metadata.update(patch)


def group_visual_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    document_key: str,
) -> list[VisualAssetGroup]:
    """Group adjacent same-page assets into stable logical visual regions.

    Distinct directly anchored Figure labels are hard boundaries. Remaining
    candidates are connected only when their boxes overlap strongly on one axis
    and have a panel-sized gap on the other.
    """

    if not candidates:
        return []

    boxes = [_candidate_bbox(candidate) for candidate in candidates]
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if candidates[left].get("page_number") != candidates[right].get("page_number"):
                continue
            if boxes[left] is None or boxes[right] is None:
                continue
            left_label = extract_figure_label(str(candidates[left].get("caption") or ""))
            right_label = extract_figure_label(str(candidates[right].get("caption") or ""))
            if left_label and right_label and left_label.number != right_label.number:
                continue
            if _boxes_are_panel_neighbors(boxes[left], boxes[right]):
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        components.setdefault(find(index), []).append(index)

    groups: list[VisualAssetGroup] = []
    for indices in components.values():
        group_boxes = [boxes[index] for index in indices if boxes[index] is not None]
        union_bbox = _union_boxes(group_boxes)
        labels = [
            label
            for index in indices
            if (label := extract_figure_label(str(candidates[index].get("caption") or "")))
        ]
        label = labels[0] if labels and all(item.number == labels[0].number for item in labels) else None
        page_number = candidates[indices[0]].get("page_number")
        stable_payload = {
            "document": document_key,
            "page": page_number,
            "bbox": {key: round(value, 3) for key, value in (union_bbox or {}).items()},
            "members": sorted(_candidate_identity(candidates[index], boxes[index]) for index in indices),
        }
        digest = hashlib.sha256(
            json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        groups.append(
            VisualAssetGroup(
                page_number=int(page_number) if page_number is not None else None,
                member_indices=tuple(sorted(indices)),
                bbox=union_bbox,
                logical_group_id=f"visual:{digest}",
                figure_label=label.label if label else None,
                figure_number=label.number if label else None,
            )
        )

    return sorted(groups, key=lambda group: _group_sort_key(group, candidates))


def normalized_bbox(
    bbox: Mapping[str, Any] | None,
    *,
    page_width: float | None,
    page_height: float | None,
) -> dict[str, float] | None:
    box = _coerce_bbox(bbox)
    if box is None or not page_width or not page_height:
        return None
    return {
        "x0": max(0.0, min(1.0, box["x0"] / page_width)),
        "y0": max(0.0, min(1.0, box["y0"] / page_height)),
        "x1": max(0.0, min(1.0, box["x1"] / page_width)),
        "y1": max(0.0, min(1.0, box["y1"] / page_height)),
    }


def _candidate_bbox(candidate: Mapping[str, Any]) -> dict[str, float] | None:
    return _coerce_bbox(candidate.get("bbox"))


def _coerce_bbox(bbox: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    try:
        x0 = float(bbox["x0"])
        y0 = float(bbox["y0"])
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "x0": min(x0, x1),
        "y0": min(y0, y1),
        "x1": max(x0, x1),
        "y1": max(y0, y1),
    }


def _boxes_are_panel_neighbors(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    left_width = left["x1"] - left["x0"]
    right_width = right["x1"] - right["x0"]
    left_height = left["y1"] - left["y0"]
    right_height = right["y1"] - right["y0"]
    if min(left_width, right_width, left_height, right_height) <= 0:
        return False

    horizontal_overlap = _overlap(left["x0"], left["x1"], right["x0"], right["x1"])
    vertical_overlap = _overlap(left["y0"], left["y1"], right["y0"], right["y1"])
    horizontal_ratio = horizontal_overlap / min(left_width, right_width)
    vertical_ratio = vertical_overlap / min(left_height, right_height)
    horizontal_gap = _gap(left["x0"], left["x1"], right["x0"], right["x1"])
    vertical_gap = _gap(left["y0"], left["y1"], right["y0"], right["y1"])

    side_by_side = vertical_ratio >= 0.55 and horizontal_gap <= max(18.0, 0.38 * min(left_width, right_width))
    stacked = horizontal_ratio >= 0.55 and vertical_gap <= max(18.0, 0.38 * min(left_height, right_height))
    return side_by_side or stacked


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _gap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0.0


def _union_boxes(boxes: Iterable[Mapping[str, float]]) -> dict[str, float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return {
        "x0": min(box["x0"] for box in boxes),
        "y0": min(box["y0"] for box in boxes),
        "x1": max(box["x1"] for box in boxes),
        "y1": max(box["y1"] for box in boxes),
    }


def _candidate_identity(candidate: Mapping[str, Any], bbox: Mapping[str, float] | None) -> str:
    metadata = candidate.get("metadata") or {}
    if isinstance(metadata, Mapping):
        image_hash = metadata.get("image_hash")
        if image_hash:
            return str(image_hash)
    if candidate.get("image_hash"):
        return str(candidate["image_hash"])
    return json.dumps(
        {
            "bbox": {key: round(value, 3) for key, value in (bbox or {}).items()},
            "caption": " ".join(str(candidate.get("caption") or "").split()),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _group_sort_key(
    group: VisualAssetGroup,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[int, float, float]:
    page = group.page_number or 0
    bbox = group.bbox or {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
    # Docling PDF coordinates commonly use bottom-left origin. X is the stable
    # primary key for multi-column figures; Y only breaks ties.
    return (page, round(bbox["x0"], 2), -round(bbox["y1"], 2))


def _bbox_area_ratio(bbox: Mapping[str, float] | None) -> float | None:
    if bbox is None:
        return None
    return max(0.0, bbox["x1"] - bbox["x0"]) * max(0.0, bbox["y1"] - bbox["y0"])


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
