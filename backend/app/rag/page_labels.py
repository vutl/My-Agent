"""Canonical helpers for printed-page labels.

PDF page identity is always the physical page number.  A printed label is
only an additional provenance alias and may describe a facing-page spread
(for example ``14 15`` or ``14-15``).  Keeping parsing here avoids subtly
different page-hint behaviour between ingestion, retrieval and multimodal
attachment code.
"""

from __future__ import annotations

import re
import unicodedata


_ROMAN_RE = re.compile(
    r"^(?=[MDCLXVI]+$)M{0,4}(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_LABEL_TOKEN = r"(?:\d{1,4}|[MDCLXVI]{1,12})"
_LABEL_RE = re.compile(
    rf"^\s*({_LABEL_TOKEN})\s*(?:(?:[-\u2010\u2011\u2012\u2013\u2014/]|\bto\b)\s*({_LABEL_TOKEN})|\s+({_LABEL_TOKEN}))?\s*$",
    re.IGNORECASE,
)


def page_label_number(value: object) -> int | None:
    """Return the first numeric value represented by a printed label."""

    values = page_label_values(value)
    return values[0] if values else None


def page_label_values(value: object) -> tuple[int, ...]:
    """Parse a printed label or facing-page range into numeric aliases.

    The parser accepts standalone Arabic/Roman labels and ranges represented
    by a dash, slash, ``to``, or two whitespace-separated labels.  The latter
    form is common in extracted facing-page PDF text (``14 15``).  Arbitrary
    prose is rejected so numeric table values cannot become page aliases.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ()
    text = text.strip(".,;:()[]{}")
    match = _LABEL_RE.fullmatch(text)
    if not match:
        return ()
    raw_values = [match.group(1), match.group(2) or match.group(3)]
    parsed: list[int] = []
    for raw in raw_values:
        if not raw:
            continue
        if raw.isdigit():
            number = int(raw)
            if number <= 0:
                return ()
        else:
            roman = raw.upper()
            if not _ROMAN_RE.fullmatch(roman):
                return ()
            values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            number = 0
            previous = 0
            for symbol in reversed(roman):
                current = values[symbol]
                number += -current if current < previous else current
                previous = max(previous, current)
            if number <= 0:
                return ()
        if number not in parsed:
            parsed.append(number)
    if len(parsed) == 2 and parsed[1] <= parsed[0]:
        return ()
    return tuple(parsed)


def page_label_matches(value: object, page_number: int) -> bool:
    """Whether a physical-page hint matches any value in a printed label."""

    try:
        target = int(page_number)
    except (TypeError, ValueError):
        return False
    return target > 0 and target in page_label_values(value)


def canonical_page_label(value: object) -> str | None:
    """Return a stable compact representation for parsed label values."""

    values = page_label_values(value)
    if not values:
        return None
    return "-".join(str(item) for item in values)

