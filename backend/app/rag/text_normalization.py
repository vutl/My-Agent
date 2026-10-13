"""Search-side text normalization.

PDF math is often extracted as Unicode "mathematical alphanumeric" letters
("𝐿𝑡𝑜𝑡𝑎𝑙", "CCC 𝐴", "𝜆"). They look like letters to a reader but match no
query word, so the passage defining a loss or metric ranks near the bottom of
lexical and embedding search. ``searchable_text`` maps only those letter
blocks to their plain forms; superscripts/subscripts and every other
character are left untouched (``10²`` must not become ``102``).

Displayed content is not changed; only the text given to the FTS index and to
the embedding model is normalized.
"""

from __future__ import annotations

import unicodedata

_LETTERLIKE = range(0x2100, 0x2150)          # ℒ ℛ ℎ ...
_MATH_ALPHANUMERIC = range(0x1D400, 0x1D800)  # 𝐀 𝑡 𝜆 𝟏 ...


def _plain(char: str) -> str:
    code = ord(char)
    if code in _MATH_ALPHANUMERIC or code in _LETTERLIKE:
        mapped = unicodedata.normalize("NFKC", char)
        if mapped.isalnum() or mapped.isalpha():
            return mapped
    return char


def _is_math_letter(char: str) -> bool:
    code = ord(char)
    return code in _MATH_ALPHANUMERIC or code in _LETTERLIKE


def searchable_text(text: str) -> str:
    value = str(text or "")
    if not any(ord(char) >= 0x2100 for char in value):
        return value
    out: list[str] = []
    previous = ""
    for char in value:
        # "L𝑡𝑜𝑡𝑎𝑙" is L with subscript "total": keep the subscript a separate word.
        if _is_math_letter(char) and previous and previous.isalnum() and not _is_math_letter(previous):
            out.append(" ")
        out.append(_plain(char))
        previous = char
    return "".join(out)


def needs_search_normalization(text: str) -> bool:
    return searchable_text(text) != str(text or "")
