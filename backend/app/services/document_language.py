"""Shared, corpus-independent language predicates for document turns."""

from __future__ import annotations

import re


_COMPARISON_CUE_RE = re.compile(
    r"(?<!\w)(?:"
    r"so\s+sánh|so\s+sanh|đối\s+chiếu|doi\s+chieu|phân\s+biệt|phan\s+biet|"
    r"khác\s+nhau|khac\s+nhau|giống\s+nhau|giong\s+nhau|"
    r"khác\s+với|khac\s+voi|giống\s+với|giong\s+voi|"
    r"compare|comparison|compared\s+(?:with|to)|versus|against|contrast|"
    r"differences?|similarities?"
    r")(?!\w)",
    flags=re.IGNORECASE,
)
_VS_RE = re.compile(r"(?<![a-z0-9])vs\.?(?![a-z0-9])", flags=re.IGNORECASE)
_CHOICE_COMPARISON_RE = re.compile(
    r"(?<!\w)(?:hay|or)(?!\w).{0,96}"
    r"(?<!\w)(?:tốt\s+hơn|tot\s+hon|better|stronger|worse|prefer(?:able)?)(?!\w)",
    flags=re.IGNORECASE | re.DOTALL,
)


def looks_like_document_comparison(query: str) -> bool:
    """Detect comparison style independently from document cardinality."""

    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        _COMPARISON_CUE_RE.search(normalized)
        or _VS_RE.search(normalized)
        or _CHOICE_COMPARISON_RE.search(normalized)
    )
