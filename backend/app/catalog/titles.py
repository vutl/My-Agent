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

# Bump when the rules change: app startup re-derives older titles
# (app/catalog/title_refresh.py) without re-parsing documents.
TITLE_RULES_VERSION = "titles-v3"

_AUTHOR_LINE_RE = re.compile(
    r"(?:@|\b(?:ieee|acm|member|student|university|universit[éa]|institute|department|"
    r"laboratory|school of|college|inc\.|ltd|arxiv|preprint|proceedings|conference|"
    r"vol\.|pp\.|doi|copyright|©)\b|\d{5,}\s*$)",
    re.IGNORECASE,
)
_GENERIC_DOC_TYPE_RE = re.compile(
    r"(?:thông\s+cáo\s+báo\s+chí|thông\s+cáo|báo\s+cáo|thông\s+báo|quyết\s+định|nghị\s+định|thông\s+tư|"
    r"công\s+văn|tờ\s+trình|press\s+release|news\s+release|report)",
    re.IGNORECASE,
)
_GENERIC_METADATA_TITLES = re.compile(
    r"^(?:untitled|microsoft word|document\d*|slide\s*\d+|title|paper|manuscript|"
    r"[\w-]+\.(?:docx?|pdf|tex|pptx?))$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    # Markdown headings may embed HTML badges/images ("Title <img src=…>").
    text = re.sub(r"<[^>]{1,500}>", " ", str(text or ""))
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # "Mul- timodal"
    # A footnote mark glued to a closing year ("năm 20261" = 2026 + note 1).
    text = re.sub(r"\b((?:19|20)\d\d)\d{1,2}\s*$", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .:;-–—")
    return text


# Text in a legacy Vietnamese font encoding (TCVN3/VNI: "Bé N«ng nghiÖp vµ
# PTNT") read as Latin-1; PDF metadata titles often carry it.
_LEGACY_ENCODING_RE = re.compile(r"[«µ¸¶·¹®¬ª¢£¤¥¦§»¼½¾¿ÐÕÖ×ÜÞß]")
# A chapter or section heading, or a bare section name, is not a document title.
_SECTION_HEADING_RE = re.compile(
    r"^(?:(?:chương|phần|mục|chapter|part|section)\s+(?:[ivxlc]+|\d+)\b"
    r"|(?:abstract|introduction|get(?:ting)?\s+started|overview|contents|table\s+of\s+contents|"
    r"(?:latest\s+)?updates?|news|readme|summary|motivation|background|rationale|specification|preface|foreword|tóm\s+tắt|mục\s+lục|giới\s+thiệu)\b\W*(?:--|$))",
    re.IGNORECASE,
)


def _plausible(title: str) -> bool:
    words = title.split()
    return (
        2 <= len(words) <= 30 and 6 <= len(title) <= 250 and not _AUTHOR_LINE_RE.search(title)
        and len(_LEGACY_ENCODING_RE.findall(title)) < 2
        and not _SECTION_HEADING_RE.match(title)
    )


_STAMP_RE = re.compile(r"^(?:arxiv:|preprint|under review|accepted|submitted|published)", re.IGNORECASE)
# Vietnamese administrative headers (national motto, issuing agency, number,
# place and date) precede the real title of official documents.
_ADMIN_HEADER_RE = re.compile(
    r"^(?:cộng\s+hòa\s+xã\s+hội\s+chủ\s+nghĩa\s+việt\s+nam|độc\s+lập\s*[-–]\s*tự\s+do\s*[-–]\s*hạnh\s+phúc|"
    r"(?:bộ|cục|tổng\s+cục|ủy\s+ban|uỷ\s+ban|ubnd|sở|văn\s+phòng|chính\s+phủ|quốc\s+hội)\b[^.]{0,60}$|"
    r"số\s*:|[^,]{0,40},\s*ngày\s*\d*\s+tháng|văn\s+bản\s+quy\s+phạm\s+pháp\s+luật\s*$|công\s+báo\b|_{3,}|-{3,})",
    re.IGNORECASE,
)
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
        originals = title.split()
        return " ".join(
            originals[index] if re.fullmatch(r"[IVXLC]+", originals[index])  # roman numerals: QUÝ II
            else word if index and word in small else word[:1].upper() + word[1:]
            for index, word in enumerate(words)
        )
    return title


def pdf_has_text_layer(path: Path, min_chars: int = 50) -> bool:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore
    try:
        with pymupdf.open(str(path)) as document:
            return sum(len(page.get_text().strip()) for page in document) >= min_chars
    except Exception:  # noqa: BLE001
        return False


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
                if len(text) < 2 or _STAMP_RE.match(text) or _ADMIN_HEADER_RE.match(text):
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
            if _GENERIC_DOC_TYPE_RE.fullmatch(_clean(" ".join(run))):
                # "THÔNG CÁO BÁO CHÍ" / "PRESS RELEASE" names the kind of
                # document; the next line says what it is about.
                following = sorted(
                    (y, text) for size, _bold, y, text in lines if y > previous_y and not _ADMIN_HEADER_RE.match(text)
                )
                if following and following[0][0] - previous_y <= top * 4:
                    run.append(following[0][1])
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
    # RFC-822 style header ("PEP: 703" / "Title: Making the GIL Optional…").
    for line in lines[:30]:
        field = re.match(r"^title\s*:\s*(.+)$", line, re.IGNORECASE)
        if field and _plausible(_clean(field.group(1))):
            return _clean(field.group(1))
    for level in (r"#", r"##"):  # a document's H1 outranks any section heading
        for line in lines[:60]:
            heading = re.match(rf"^{level}\s+(.+)$", line)
            if heading:
                title = _clean(heading.group(1))
                # "0. Purpose of this addendum" names a section, not the document.
                if level == r"##" and re.match(r"^\d+(?:\.\d+)*\.?\s", title):
                    continue
                if _plausible(title) and not _SECTION_HEADING_RE.match(title):
                    return title
    for index, line in enumerate(lines[:60]):
        # Setext heading: "ripgrep (rg)" underlined with --- or ===.
        # A "Source: https://…" line closing a banner is metadata, not a title.
        if re.search(r"https?://|www\.", line):
            continue
        if index + 1 < len(lines) and re.fullmatch(r"(?:=|-){3,}", lines[index + 1]) and line:
            title = _clean(line)
            if _SECTION_HEADING_RE.match(title):
                continue
            if _plausible(title) or (title and len(title.split()) >= 1 and 3 <= len(title) <= 120):
                return title
    if not allow_plain_lines:
        return None
    # Quotes, rules, tables, list items and code are never a document title.
    content = [
        line for line in lines[:30]
        if line and not re.match(r"^(?:#|>|---|\||```|[-*+]\s|\d+[.)]\s)", line) and not _ADMIN_HEADER_RE.match(line)
    ]
    for index, line in enumerate(content):
        title = _clean(line.lstrip("#").strip())
        if not (title and _plausible(title) and not title.endswith((",", ";")) and len(title.split()) >= 3):
            continue
        # Titles wrap over several short lines; stop at the author/affiliation
        # line or at anything that reads like body text.
        generic = bool(_GENERIC_DOC_TYPE_RE.fullmatch(title))
        for following in content[index + 1:index + 4]:
            part = _clean(following)
            if (
                not part
                or len(part.split()) > (25 if generic else 12)
                or part.endswith((".", ":", ";"))
                or _AUTHOR_LINE_RE.search(part)
                or _AUTHOR_MARK_RE.search(part)
                or len((title + " " + part).split()) > 30
            ):
                break
            title = f"{title} {part}"
        return title
    return None


def docx_title(path: Path) -> str | None:
    """Title-styled paragraph, else the leading short title lines, else the
    core property (unless it only names the issuing agency)."""
    try:
        import docx  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        document = docx.Document(str(path))
    except Exception:  # noqa: BLE001 - unreadable files fall back to text heuristics
        return None
    paragraphs = [(p.style.name if p.style is not None else "", _clean(p.text)) for p in document.paragraphs]
    for index, (style, text) in enumerate(paragraphs[:40]):
        if _GENERIC_DOC_TYPE_RE.fullmatch(text or "") and style.lower() in {"title", "heading 1"}:
            # "THÔNG TƯ" names the kind of document; the next line says what
            # it is about ("Ban hành Danh mục…").
            following = next(
                (later for _later_style, later in paragraphs[index + 1:index + 4]
                 if later and not _ADMIN_HEADER_RE.match(later) and len(later.split()) <= 30),
                "",
            )
            if following:
                return _readable_case(f"{text} {following}")
        if text and style.lower() in {"title", "heading 1"} and _plausible(text) or (
            text and style.lower() == "title" and 1 <= len(text.split()) <= 30
            and not _SECTION_HEADING_RE.match(text) and not _GENERIC_DOC_TYPE_RE.fullmatch(text)
        ):
            return _readable_case(text)
    run: list[str] = []
    for _style, text in paragraphs[:15]:
        if not text:
            if run:
                break
            continue
        if _ADMIN_HEADER_RE.match(text):
            if run:
                break
            continue
        generic_run = bool(run) and bool(_GENERIC_DOC_TYPE_RE.fullmatch(" ".join(run)))
        if len(text.split()) > (30 if generic_run else 14) or text.endswith((".", ":", ";")):
            break
        run.append(text)
    if run and _plausible(" ".join(run)):
        return _readable_case(" ".join(run))
    core = _clean(str(document.core_properties.title or ""))
    if core and _plausible(core) and not _ADMIN_HEADER_RE.match(core):
        return core
    return None


def guess_title(path: Path, text: str) -> str:
    if path.suffix.lower() == ".pdf" and path.exists():
        title = pdf_title(path)
        if title:
            return title
    if path.suffix.lower() == ".docx" and path.exists():
        title = docx_title(path)
        if title:
            return title
    if path.suffix.lower() == ".pdf" and path.exists() and not pdf_has_text_layer(path):
        # A scanned PDF's OCR text starts with noisy headers; the file name
        # is a safer title than garbled OCR.
        return path.stem.replace("_", " ").strip()
    # Authored text formats name themselves with a heading; their first prose
    # line is body text. Plain-line titles are only for PDF-extracted text.
    title = text_title(text, allow_plain_lines=path.suffix.lower() == ".pdf")
    if title:
        return title
    return path.stem.replace("_", " ").strip()
