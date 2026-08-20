from dataclasses import dataclass
from pathlib import Path
import re


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "been",
    "between",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "with",
    "using",
    "for",
    "our",
    "can",
    "will",
    "not",
    "was",
    "were",
}

TAG_RULES: dict[str, tuple[str, ...]] = {
    "rag": (r"\brag\b", r"retrieval[- ]augmented"),
    "lancedb": (r"\blancedb\b", r"\blance\b"),
    "sqlite": (r"\bsqlite\b", r"\bfts5\b"),
    "agent": (r"\bagent\b", r"\blanggraph\b", r"tool call"),
    "visual_rag": (r"\bvisual\b", r"\bfigure\b", r"\bchart\b", r"\bdiagram\b", r"\bmultimodal\b"),
    "speech_emotion_recognition": (r"speech emotion", r"\bser\b", r"emotion recognition"),
    "audio_visual": (r"audio[- ]visual", r"\baudiovisual\b", r"visual[- ]audio"),
    "multimodal_fusion": (r"\bmultimodal\b", r"\bfusion\b", r"cross[- ]modal"),
    "valence_arousal": (r"\bvalence\b", r"\barousal\b", r"\bdominance\b"),
    "dataset": (r"\bdataset\b", r"\bcorpus\b", r"\bbenchmark\b"),
}


@dataclass(frozen=True)
class DocumentCardDraft:
    title_guess: str
    doc_type: str
    language: str
    short_summary: str
    topic_tags: list[str]
    project_tags: list[str]
    keywords: list[str]
    importance_score: float
    confidence: float
    should_deep_index: bool


def build_document_card(path: Path, text: str) -> DocumentCardDraft:
    normalized = _normalize_text(text)
    title = _guess_title(path, normalized)
    keywords = _keywords(normalized)
    tags = _topic_tags(path.name, normalized)
    doc_type = _doc_type(path.name, normalized, tags)
    project_tags = _project_tags(path.name, normalized)

    importance = 0.45
    if doc_type in {"project_plan", "technical_plan"}:
        importance += 0.35
    if "speech_emotion_recognition" in tags or "multimodal_fusion" in tags:
        importance += 0.25
    if len(normalized) > 2_000:
        importance += 0.1

    return DocumentCardDraft(
        title_guess=title,
        doc_type=doc_type,
        language=_language(normalized),
        short_summary=_summary(normalized),
        topic_tags=tags,
        project_tags=project_tags,
        keywords=keywords,
        importance_score=min(importance, 1.0),
        confidence=0.65 if normalized else 0.2,
        should_deep_index=True,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _guess_title(path: Path, normalized: str) -> str:
    for line in normalized.split(". "):
        cleaned = line.strip().lstrip("#").strip()
        if 8 <= len(cleaned) <= 160 and len(cleaned.split()) <= 18:
            return cleaned
    return path.stem.replace("_", " ").strip()


def _summary(normalized: str, max_chars: int = 700) -> str:
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    boundary = normalized.rfind(". ", 0, max_chars)
    if boundary < 180:
        boundary = max_chars
    return normalized[:boundary].strip()


def _keywords(normalized: str, limit: int = 18) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", normalized.lower()):
        if token in STOPWORDS or len(token) > 40:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def _topic_tags(filename: str, normalized: str) -> list[str]:
    haystack = f"{filename} {normalized}".lower()
    tags = [
        tag
        for tag, patterns in TAG_RULES.items()
        if any(re.search(pattern, haystack) for pattern in patterns)
    ]
    return tags[:12]


def _project_tags(filename: str, normalized: str) -> list[str]:
    haystack = f"{filename} {normalized}".lower()
    tags: list[str] = []
    if "project_plan" in haystack or "local ai agent" in haystack or "lancedb" in haystack:
        tags.append("local_ai_agent")
    if "speech emotion" in haystack or "multimodal" in haystack or "ser" in haystack:
        tags.append("ser_research")
    return tags


def _doc_type(filename: str, normalized: str, tags: list[str]) -> str:
    haystack = f"{filename} {normalized}".lower()
    if "project_plan" in haystack:
        return "project_plan"
    if "lancedb" in haystack or "rag" in tags or "agent" in tags:
        return "technical_plan"
    if "abstract" in haystack and ("references" in haystack or "introduction" in haystack):
        return "research_paper"
    if filename.lower().endswith(".md"):
        return "markdown_note"
    return "document"


def _language(normalized: str) -> str:
    vietnamese_markers = ("đ", "Đ", "ă", "â", "ê", "ô", "ơ", "ư")
    if any(marker in normalized for marker in vietnamese_markers):
        return "vi"
    return "en"
