"""Document title extraction.

A catalog that knows each document's real title is what lets an assistant map
"the paper about key-sparse attention" or a misspelled name back to the right
file. The previous heuristic took the first "sentence" of whitespace-collapsed
text, which returned author lines or mid-body sentences for most PDFs.

Order of evidence, strongest first:
1. PDF: the largest-font line block in the upper part of page 1 (how papers,
   reports and slides print their title), checked against PDF metadata.
2. Markdown: the first ``# `` heading.
3. Any text: the first short line that does not look like an author/venue line.
4. The file stem.
"""

from __future__ import annotations

from pathlib import Path
import re

_AUTHOR_LINE_RE = re.compile(
    r"(?:@|\b(?:ieee|acm|member|student|university|universit[éa]|institute|department|"
    r"laboratory|school of|college|inc\.|ltd|arxiv|preprint|proceedings|conference|"
    r"vol\.|pp\.|doi|copyright|©)\b|\d{4,}\s*$)",
    re.IGNORECASE,
)
_GENERIC_METADATA_TITLES = re.compile(
    r"^(?:untitled|microsoft word|document\d*|slide\s*\d+|title|paper|manuscript|"
    r"[\w-]+\.(?:docx?|pdf|tex|pptx?))$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", str(text or ""))  # "Mul- timodal"
    text = re.sub(r"\s+", " ", text).strip(" .:;-–—")
    return text


def _plausible(title: str) -> bool:
    words = title.split()
    return 2 <= len(words) <= 30 and 6 <= len(title) <= 250 and not _AUTHOR_LINE_RE.search(title)


_STAMP_RE = re.compile(r"^(?:arxiv:|preprint|under review|accepted|submitted|published)", re.IGNORECASE)
_AUTHOR_MARK_RE = re.compile(r"[⋆∗*†‡§¶]|\d\s*,|^\W*(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+\d")


def _line_text(raw_line: dict) -> str:
    """Rebuild a line's text from character boxes, inserting the spaces that
    some PDFs omit ("MultimodalEmotionDetection")."""
    pieces: list[str] = []
    previous_right = None
    for span in raw_line.get("spans", []):
        size = float(span.get("size", 10.0)) or 10.0
        for char in span.get("chars", []):
            symbol = str(char.get("c", ""))
            left, _top, right, _bottom = char.get("bbox", (0, 0, 0, 0))
            if previous_right is not None and left - previous_right > 0.18 * size:
                if pieces and not pieces[-1].endswith(" ") and symbol != " ":
                    pieces.append(" ")
            pieces.append(symbol)
            previous_right = right
    return "".join(pieces)


def _readable_case(title: str) -> str:
    letters = [char for char in title if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        small = {"a", "an", "and", "the", "of", "for", "in", "on", "with", "to", "via", "by", "from"}
        words = title.lower().split()
        return " ".join(
            word if index and word in small else word[:1].upper() + word[1:]
            for index, word in enumerate(words)
        )
    return title


def pdf_title(path: Path) -> str | None:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            return None
    try:
        document = pymupdf.open(str(path))
    except Exception:  # noqa: BLE001 - unreadable PDFs fall back to text heuristics
        return None
    try:
        if document.page_count == 0:
            return None
        page = document[0]
        height = page.rect.height or 1.0
        # (size, bold, y, text) of horizontal lines in the upper page.
        lines: list[tuple[float, bool, float, str]] = []
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                direction = line.get("dir", (1.0, 0.0))
                if abs(direction[1]) > 0.1:  # rotated side stamps (arXiv ids)
                    continue
                spans = [span for span in line.get("spans", []) if span.get("chars")]
                if not spans:
                    continue
                text = _clean(_line_text(line))
                if len(text) < 2 or _STAMP_RE.match(text):
                    continue
                size = max(float(span.get("size", 0.0)) for span in spans)
                bold = all(int(span.get("flags", 0)) & 16 or "bold" in str(span.get("font", "")).lower()
                           for span in spans)
                y = float(line.get("bbox", (0, 0, 0, 0))[1])
                if y <= height * 0.45:
                    lines.append((size, bold, y, text))
        candidate = None
        if lines:
            top = max(size for size, _bold, _y, _text in lines)
            title_lines = sorted(
                ((y, bold, text) for size, bold, y, text in lines if size >= top - 0.6),
                key=lambda item: item[0],
            )
            first_y, first_bold, first_text = title_lines[0]
            run = [first_text]
            previous_y = first_y
            for y, bold, text in title_lines[1:]:
                if y - previous_y > top * 2.2 or bold != first_bold:
                    break
                if _AUTHOR_LINE_RE.search(text) or _AUTHOR_MARK_RE.search(text):
                    break
                run.append(text)
                previous_y = y
            candidate = _readable_case(_clean(" ".join(run)))
        metadata = _clean(str((document.metadata or {}).get("title") or ""))
        if metadata and _plausible(metadata) and not _GENERIC_METADATA_TITLES.match(metadata):
            # Prefer metadata when the visual title is missing or is a fragment of it.
            if not candidate or not _plausible(candidate) or candidate.casefold() in metadata.casefold():
                return metadata
        if candidate and _plausible(candidate):
            return candidate
        return None
    finally:
        document.close()


def text_title(text: str, *, allow_plain_lines: bool = True) -> str | None:
    """Heading title; for PDF-extracted text also a wrapped first line."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    for level in (r"#", r"##"):  # a document's H1 outranks any section heading
        for line in lines[:60]:
            heading = re.match(rf"^{level}\s+(.+)$", line)
            if heading:
                title = _clean(heading.group(1))
                # "0. Purpose of this addendum" names a section, not the document.
                if level == r"##" and re.match(r"^\d+(?:\.\d+)*\.?\s", title):
                    continue
                if _plausible(title):
                    return title
    if not allow_plain_lines:
        return None
    # Quotes, rules, tables, list items and code are never a document title.
    content = [line for line in lines[:30] if line and not re.match(r"^(?:#|>|---|\||```|[-*+]\s|\d+[.)]\s)", line)]
    for index, line in enumerate(content):
        title = _clean(line.lstrip("#").strip())
        if not (title and _plausible(title) and not title.endswith((",", ";")) and len(title.split()) >= 3):
            continue
        # Titles wrap over several short lines; stop at the author/affiliation
        # line or at anything that reads like body text.
        for following in content[index + 1:index + 4]:
            part = _clean(following)
            if (
                not part
                or len(part.split()) > 12
                or part.endswith((".", ":", ";"))
                or _AUTHOR_LINE_RE.search(part)
                or _AUTHOR_MARK_RE.search(part)
                or len((title + " " + part).split()) > 30
            ):
                break
            title = f"{title} {part}"
        return title
    return None


def guess_title(path: Path, text: str) -> str:
    if path.suffix.lower() == ".pdf" and path.exists():
        title = pdf_title(path)
        if title:
            return title
    # Authored text formats name themselves with a heading; their first prose
    # line is body text. Plain-line titles are only for PDF-extracted text.
    title = text_title(text, allow_plain_lines=path.suffix.lower() == ".pdf")
    if title:
        return title
    return path.stem.replace("_", " ").strip()
