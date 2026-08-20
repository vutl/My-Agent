from __future__ import annotations

import json
import re

from app.rag.figure_quality import extract_figure_label

_CAPTION_START = re.compile(r"^(fig\.?|figure|hình)\s*(\d+)\s*([\.:–-])", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"-\s*\n+\s*")


def normalize_caption_text(text: str) -> str:
    cleaned = _HYPHEN_BREAK.sub("", text or "")
    cleaned = re.sub(r"-\s+(?=[a-z])", "", cleaned)
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    return " ".join(cleaned.split()).strip()


def dedupe_repeated_figure_prefix(text: str) -> str:
    normalized = normalize_caption_text(text)
    if not normalized:
        return normalized

    match = _CAPTION_START.match(normalized)
    if not match:
        return normalized

    figure_number = match.group(2)
    splitter = re.compile(
        rf"(?=(?:fig\.?|figure|hình)\s*{re.escape(figure_number)}\s*[\.:–-])",
        re.IGNORECASE,
    )
    parts = [part.strip() for part in splitter.split(normalized) if part.strip()]
    if len(parts) <= 1:
        return normalized
    return max(parts, key=len)


def extract_figure_title_sentence(text: str, *, max_chars: int = 180) -> str:
    normalized = dedupe_repeated_figure_prefix(text)
    if not normalized:
        return normalized

    match = _CAPTION_START.search(normalized)
    if not match:
        return normalized[:max_chars] + ("…" if len(normalized) > max_chars else "")

    body = normalized[match.start() :]
    period = body.find(". ")
    if 20 < period < max_chars:
        return body[: period + 1].strip()
    if len(body) <= max_chars:
        return body.strip()
    return body[: max_chars - 1].rstrip() + "…"


def extract_figure_caption_from_content(
    content: str,
    *,
    figure_number: int | None = None,
    figure_index: int | None = None,
) -> str | None:
    if not content:
        return None

    # figure_index is extraction order, not the paper's Figure N. Keep the
    # keyword temporarily for call-site compatibility, but never derive a label
    # from it.
    _ = figure_index
    target_number = figure_number
    paragraphs = re.split(r"\n\s*\n", _HYPHEN_BREAK.sub("", content))
    for paragraph in paragraphs:
        stripped = " ".join(paragraph.split()).strip()
        if not stripped:
            continue
        match = _CAPTION_START.match(stripped)
        if not match:
            continue
        number = int(match.group(2))
        if target_number is not None and number != target_number:
            continue
        return extract_figure_title_sentence(stripped)

    return None


def best_figure_caption(
    *,
    caption: str | None = None,
    content: str | None = None,
    visual_summary: str | None = None,
    figure_number: int | None = None,
    figure_index: int | None = None,
) -> str:
    normalized_caption = dedupe_repeated_figure_prefix(caption or "")
    if (
        normalized_caption
        and extract_figure_label(normalized_caption)
        and not caption_looks_truncated(normalized_caption)
    ):
        # The paper-authored Figure/Hình caption is the display authority. Raw
        # structured VLM output may be longer but belongs in retrieval context,
        # never in the UI caption.
        return extract_figure_title_sentence(normalized_caption)

    candidates: list[str] = []

    if normalized_caption:
        candidates.append(extract_figure_title_sentence(normalized_caption))

    summary_title = _visual_summary_title(visual_summary)
    if summary_title:
        candidates.append(extract_figure_title_sentence(summary_title))

    if content:
        caption_label = extract_figure_label(caption)
        target_number = figure_number or (caption_label.number if caption_label else None)
        extracted = extract_figure_caption_from_content(
            content,
            figure_number=target_number,
        )
        if extracted:
            candidates.append(extracted)

    if not candidates:
        return "Figure"

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _visual_summary_title(summary: str | None) -> str | None:
    raw = (summary or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        for key in ("title", "figure_title"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    for line in raw.splitlines():
        match = re.match(r"^(?:title|figure_title)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    # Preserve old plain-text summaries, but never flatten a structured field
    # dump into a caption.
    if not _FIELD_DUMP_RE.search(raw):
        return normalize_caption_text(raw)
    return None


_FIELD_DUMP_RE = re.compile(
    r"(?im)^(?:asset_kind|is_content|is_complete|confidence|figure_type|observed_visual|"
    r"contextual_role|search_phrases)\s*:",
)


def caption_looks_truncated(caption: str | None) -> bool:
    normalized = normalize_caption_text(caption or "")
    if not normalized:
        return True
    if normalized.endswith("-") or normalized.endswith("–"):
        return True
    if len(normalized) < 24:
        return False
    tail = normalized.rsplit(" ", 1)[-1]
    if len(tail) <= 3 and tail.isalpha() and tail.islower():
        return True
    return False


def requested_figure_number(query: str) -> int | None:
    lowered = query.lower()
    patterns = (
        r"\bfig(?:ure)?\.?\s*(\d+)\b",
        r"\bhình\s*(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def figure_relevance_score(
    figure: dict,
    *,
    preferred_figure_number: int | None = None,
    query: str | None = None,
) -> float:
    """Rank by retrieval score + general query↔figure-text overlap.

    No paper-specific or Fig-N hard rules (except explicit "Figure N" in the query).
    Overlap uses tokens from the user query against caption + VLM search text only.
    """
    score = float(figure.get("score") or 0)
    caption = str(figure.get("caption") or "").lower()
    content = str(figure.get("content") or "").lower()
    blob = f"{caption}\n{content}"
    metadata = figure.get("metadata") if isinstance(figure.get("metadata"), dict) else {}
    figure_type = str(
        figure.get("figure_type") or metadata.get("figure_type") or ""
    ).strip().lower()
    if not figure_type:
        type_match = re.search(
            r"(?im)^figure_type:\s*(architecture|diagram|chart|plot|table|photo|other)\b",
            blob,
        )
        figure_type = type_match.group(1).lower() if type_match else ""
    figure_number = figure.get("figure_number") or metadata.get("figure_number")
    if not isinstance(figure_number, int):
        label = extract_figure_label(str(figure.get("caption") or ""))
        figure_number = label.number if label else None
    query_text = " ".join((query or "").lower().split())

    if caption.startswith("figure extracted from page") or content.startswith("figure extracted from page"):
        return score - 200
    if "visual fallback" in caption or "visual fallback" in content:
        return score - 200

    if preferred_figure_number is not None:
        if figure_number == preferred_figure_number:
            score += 80
        elif caption.startswith(f"fig. {preferred_figure_number}") or caption.startswith(
            f"figure {preferred_figure_number}"
        ):
            score += 60

    if query_text:
        architecture_intent = any(
            marker in query_text
            for marker in (
                "architecture",
                "architectural",
                "kiến trúc",
                "cấu trúc",
                "pipeline",
                "framework",
                "model overview",
                "overall model",
                "sơ đồ tổng thể",
            )
        )
        if architecture_intent:
            if figure_type == "architecture":
                score += 70
            elif figure_type == "diagram":
                score += 50
            elif figure_type in {"chart", "plot", "photo"}:
                score -= 15

        quantitative_intent = any(
            marker in query_text
            for marker in (
                "benchmark",
                "ablation",
                "confusion matrix",
                "kết quả",
                "so sánh",
                "accuracy",
                " acc",
                " f1",
                " ccc",
                "uar",
            )
        )
        if quantitative_intent and figure_type in {"chart", "plot", "table"}:
            score += 35

        stop = {
            "của",
            "cho",
            "với",
            "the",
            "and",
            "for",
            "về",
            "các",
            "một",
            "này",
            "đó",
            "làm",
            "sao",
            "gì",
            "như",
            "thế",
            "tôi",
            "mình",
            "bài",
            "paper",
            "pdf",
        }
        tokens = [
            token
            for token in re.findall(r"[a-z0-9à-ỹ]{3,}", query_text)
            if token not in stop
        ]
        if tokens:
            overlap = sum(1 for token in tokens if token in blob)
            # Reward figures whose own text mentions the same terms the user used.
            score += 20.0 * overlap

    return score
