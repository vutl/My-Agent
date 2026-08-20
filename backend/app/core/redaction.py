"""Fail-closed redaction and UTF-8 byte bounding for local debug traces."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


_DENIED_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "api_key",
    "apikey",
    "x_api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "auth_token",
    "session_token",
    "token",
    "password",
    "passcode",
    "client_secret",
    "secret",
    "secret_key",
    "secret_access_key",
    "private_key",
    "credential",
    "credentials",
}

_DENIED_KEY_SUFFIXES = tuple(sorted(_DENIED_KEYS, key=len, reverse=True))

_SECRET_LABEL = (
    r"(?:[a-z0-9]+[-_])*"
    r"(?:x[-_]?api[-_]?key|api[-_]?key|apikey|client[-_]?secret|"
    r"access[-_]?token|refresh[-_]?token|id[-_]?token|auth[-_]?token|"
    r"session[-_]?token|token|password|passcode|private[-_]?key|"
    r"secret[-_]?access[-_]?key|secret[-_]?key|secret|credentials?)"
)

# Prompt and draft fields are opaque strings, so structured-key filtering alone
# is insufficient.  Preserve the label for debugging while replacing its value,
# including JSON/YAML/env/query-string spellings.
_LABELED_SECRET_RE = re.compile(
    rf"(?ix)"
    rf"(?P<prefix>(?<![\w-])[\"']?{_SECRET_LABEL}[\"']?\s*[:=]\s*)"
    rf"(?:"
    rf"(?P<quote>[\"'])(?P<quoted_value>[^\"'\r\n]*)(?P=quote)"
    rf"|(?P<bare_value>[^\s,;&#}}\]\r\n]+)"
    rf")"
)

# Header values commonly contain spaces or multiple cookies.  Redact the full
# line so credentials after a scheme name or semicolon cannot survive.
_SENSITIVE_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>(?<![\w-])[\"']?"
    r"(?:authorization|proxy[-_ ]authorization|cookie|set[-_ ]cookie)"
    r"[\"']?\s*:\s*)[^\r\n]+"
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"-----BEGIN [^-\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\n]*PRIVATE KEY-----",
        re.IGNORECASE,
    ),
    re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@"),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"
    ),
    re.compile(r"(?i)data:[^,;\s]+(?:;base64)?,[A-Za-z0-9+/=_-]{32,}"),
)

_LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:/Users/|/home/|/private/|[A-Za-z]:\\Users\\)"
    r"[^\n\r\t<>\"']+"
)


def redact_and_bound(
    payload: dict[str, Any],
    *,
    max_bytes: int,
    exact_secrets: tuple[str, ...] = (),
) -> tuple[dict[str, Any], int, bool, int]:
    """Return sanitized payload, replacement count, truncation flag and bytes."""

    replacements = 0

    def redact(value: Any, key: str | None = None) -> Any:
        nonlocal replacements
        if key and _is_denied_key(key):
            replacements += 1
            return "[REDACTED_SECRET]"
        if isinstance(value, dict):
            return {str(item_key): redact(item_value, str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return [redact(item) for item in value]
        if not isinstance(value, str):
            return deepcopy(value)

        text = value
        for secret in exact_secrets:
            candidate = str(secret or "")
            if len(candidate) >= 8 and candidate.casefold() != "any" and candidate in text:
                occurrences = text.count(candidate)
                text = text.replace(candidate, "[REDACTED_CONFIGURED_SECRET]")
                replacements += occurrences
        text, count = _SENSITIVE_HEADER_RE.subn(
            lambda match: f"{match.group('prefix')}[REDACTED_SECRET]",
            text,
        )
        replacements += count
        for pattern in _SECRET_PATTERNS:
            text, count = pattern.subn(_pattern_replacement(pattern), text)
            replacements += count
        text, count = _LABELED_SECRET_RE.subn(_labeled_secret_replacement, text)
        replacements += count
        text, count = _LOCAL_PATH_RE.subn("[REDACTED_LOCAL_PATH]", text)
        replacements += count
        return text

    sanitized = redact(payload)
    truncated = _clip_large_strings(sanitized, max_chars=16_384)
    encoded = _encoded(sanitized)
    while len(encoded) > max_bytes:
        slot = _largest_string_slot(sanitized)
        if slot is None or len(slot[2]) <= 256:
            sanitized = {
                "schema_version": sanitized.get("schema_version", 1),
                "capture": {
                    "redacted": True,
                    "truncated": True,
                    "reason": "payload_exceeded_hard_cap",
                },
            }
            truncated = True
            encoded = _encoded(sanitized)
            break
        container, field, text = slot
        target_chars = max(256, len(text) // 2)
        replacement = _head_tail_clip(text, target_chars)
        container[field] = replacement
        truncated = True
        encoded = _encoded(sanitized)
    return sanitized, replacements, truncated, len(encoded)


def _is_denied_key(key: str) -> bool:
    # Preserve CamelCase word boundaries before case-folding so structured
    # payloads cannot bypass the same denylist with clientSecret/accessToken.
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    return any(
        normalized == denied or normalized.endswith(f"_{denied}")
        for denied in _DENIED_KEY_SUFFIXES
    )


def _labeled_secret_replacement(match: re.Match[str]) -> str:
    value = match.group("quoted_value") or match.group("bare_value") or ""
    if value.startswith("[REDACTED_"):
        return match.group(0)
    quote = match.group("quote") or ""
    return f"{match.group('prefix')}{quote}[REDACTED_SECRET]{quote}"


def _pattern_replacement(pattern: re.Pattern[str]) -> str:
    raw = pattern.pattern.casefold()
    if "https?://" in raw:
        return r"\1[REDACTED_CREDENTIALS]@"
    if "api[_-]?key" in raw:
        return r"\1[REDACTED_SECRET]"
    return "[REDACTED_SECRET]"


def _clip_large_strings(value: Any, *, max_chars: int) -> bool:
    truncated = False
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, str) and len(item) > max_chars:
                value[key] = _head_tail_clip(item, max_chars)
                truncated = True
            else:
                truncated = _clip_large_strings(item, max_chars=max_chars) or truncated
    elif isinstance(value, list):
        for index, item in enumerate(list(value)):
            if isinstance(item, str) and len(item) > max_chars:
                value[index] = _head_tail_clip(item, max_chars)
                truncated = True
            else:
                truncated = _clip_large_strings(item, max_chars=max_chars) or truncated
    return truncated


def _largest_string_slot(value: Any) -> tuple[Any, Any, str] | None:
    slots: list[tuple[Any, Any, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, str):
                    slots.append((item, key, child))
                else:
                    visit(child)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, str):
                    slots.append((item, index, child))
                else:
                    visit(child)

    visit(value)
    return max(slots, key=lambda item: len(item[2]), default=None)


def _head_tail_clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[TRUNCATED_DEBUG_TRACE]...\n"
    budget = max(0, max_chars - len(marker))
    head = budget * 2 // 3
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _encoded(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
