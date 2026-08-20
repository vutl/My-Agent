"""Vietnamese-aware tokenization for SQLite FTS queries."""

from __future__ import annotations

import re

_VI_WORD = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_FTS_STOPWORDS = frozenset(
    {
        "and",
        "cho",
        "cua",
        "của",
        "for",
        "from",
        "là",
        "la",
        "the",
        "this",
        "that",
        "with",
        "trong",
        "như",
        "nhu",
        "gì",
        "gi",
        "có",
        "co",
        "không",
        "khong",
        "một",
        "mot",
        "các",
        "cac",
        "và",
        "va",
        "được",
        "duoc",
    }
)


def tokenize_for_fts(text: str, *, max_tokens: int = 12) -> list[str]:
    """Tokenize query text for FTS5, with optional pyvi syllable splits."""
    lowered = text.lower().strip()
    if not lowered:
        return []

    tokens: list[str] = []
    seen: set[str] = set()

    def add_token(raw: str) -> None:
        cleaned = raw.strip("._-")
        if len(cleaned) < 2 or cleaned in _FTS_STOPWORDS:
            return
        if cleaned not in seen:
            seen.add(cleaned)
            tokens.append(cleaned)

    for match in _VI_WORD.finditer(lowered):
        add_token(match.group(0))

    for syllable in _pyvi_tokens(lowered):
        add_token(syllable)

    return tokens[:max_tokens]


def build_fts_query(text: str, *, max_tokens: int = 10) -> str:
    tokens = tokenize_for_fts(text, max_tokens=max_tokens)
    if not tokens:
        return ""
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def _pyvi_tokens(text: str) -> list[str]:
    try:
        from pyvi.ViTokenizer import tokenize
    except ImportError:
        return []

    try:
        tagged = tokenize(text)
    except Exception:
        return []

    result: list[str] = []
    for piece in tagged.split():
        if "_" in piece:
            for part in piece.split("_"):
                if part:
                    result.append(part.lower())
        elif piece:
            result.append(piece.lower())
    return result
