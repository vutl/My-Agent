from dataclasses import dataclass
from functools import lru_cache
import re

from app.rag.parsers import ParsedDocument


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    token_count: int | None = None
    parent_index: int | None = None
    parent_text: str | None = None
    parent_token_count: int | None = None
    child_index: int | None = None


DEFAULT_CHILD_TOKENS = 384
DEFAULT_CHILD_OVERLAP_TOKENS = 48
DEFAULT_PARENT_TOKENS = 1536
DEFAULT_TOKENIZER = "unicode_lexical_v1"


def chunk_text(text: str, *, chunk_size: int = 1400, overlap: int = 200) -> list[str]:
    return [
        chunk.text
        for chunk in _chunk_text_units(text, chunk_size=chunk_size, overlap=overlap)
    ]


def _chunk_text_units(text: str, *, chunk_size: int = 1400, overlap: int = 200) -> list[TextChunk]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_section: str | None = None
    current_length = 0

    for block, section in _blocks_with_sections(normalized):
        starts_new_section = bool(
            _detect_section_title(block)
            and section
            and current_section
            and _normalize_section(section) != _normalize_section(current_section)
        )
        if starts_new_section and current_parts:
            chunks.append(TextChunk(text="\n\n".join(current_parts).strip(), section_title=current_section))
            current_parts = []
            current_length = 0

        for piece in _split_oversized_block(block, chunk_size=chunk_size):
            piece_length = len(piece)
            if current_parts and current_length + piece_length + 2 > chunk_size:
                chunk_text_value = "\n\n".join(current_parts).strip()
                chunks.append(TextChunk(text=chunk_text_value, section_title=current_section))
                overlap_text = "" if section and current_section and _normalize_section(section) != _normalize_section(current_section) else _tail_overlap(chunk_text_value, overlap)
                current_parts = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            if section:
                current_section = section
            elif current_section is None:
                current_section = _detect_section_title(piece)
            current_parts.append(piece)
            current_length += piece_length + (2 if current_parts else 0)

    if current_parts:
        chunks.append(TextChunk(text="\n\n".join(current_parts).strip(), section_title=current_section))

    return [chunk for chunk in chunks if chunk.text]


def chunk_parsed_document(
    parsed: ParsedDocument,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_tokens: int = DEFAULT_CHILD_TOKENS,
    overlap_tokens: int = DEFAULT_CHILD_OVERLAP_TOKENS,
    parent_max_tokens: int = DEFAULT_PARENT_TOKENS,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[TextChunk]:
    """Chunk a parsed document for precise retrieval and coherent expansion.

    Production indexing uses token-aware child chunks linked to larger parent
    passages.  The character-size arguments remain available for compatibility
    with callers/tests that explicitly request the previous v2 strategy.
    """

    if chunk_size is not None or overlap is not None:
        legacy_size = chunk_size if chunk_size is not None else 1400
        legacy_overlap = overlap if overlap is not None else 200
        return _chunk_parsed_document_by_characters(
            parsed,
            chunk_size=legacy_size,
            overlap=legacy_overlap,
        )

    pages = parsed.pages or [None]
    parent_offset = 0
    chunks: list[TextChunk] = []
    for page in pages:
        page_text = page.text if page is not None else parsed.text
        page_number = page.page_number if page is not None else None
        page_chunks, parent_count = _chunk_page_by_tokens(
            page_text,
            page_number=page_number,
            parent_offset=parent_offset,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            parent_max_tokens=parent_max_tokens,
            tokenizer_name=tokenizer_name,
        )
        chunks.extend(page_chunks)
        parent_offset += parent_count
    return chunks


def _chunk_parsed_document_by_characters(
    parsed: ParsedDocument,
    *,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if parsed.pages:
        chunks: list[TextChunk] = []
        for page in parsed.pages:
            for chunk in _chunk_text_units(page.text, chunk_size=chunk_size, overlap=overlap):
                chunks.append(
                    TextChunk(
                        text=chunk.text,
                        page_number=page.page_number,
                        section_title=chunk.section_title,
                    )
                )
        return chunks

    return [
        TextChunk(text=chunk.text, page_number=None, section_title=chunk.section_title)
        for chunk in _chunk_text_units(parsed.text, chunk_size=chunk_size, overlap=overlap)
    ]


def count_tokens(text: str, *, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    if not text:
        return 0
    return len(_tokenizer(tokenizer_name).encode(text, disallowed_special=()))


def _chunk_page_by_tokens(
    text: str,
    *,
    page_number: int | None,
    parent_offset: int,
    max_tokens: int,
    overlap_tokens: int,
    parent_max_tokens: int,
    tokenizer_name: str,
) -> tuple[list[TextChunk], int]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if parent_max_tokens < max_tokens:
        raise ValueError("parent_max_tokens must be greater than or equal to max_tokens")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens")

    units = _token_structural_units(
        text,
        max_tokens=max_tokens,
        tokenizer_name=tokenizer_name,
    )
    parents = _group_parent_units(
        units,
        parent_max_tokens=parent_max_tokens,
        tokenizer_name=tokenizer_name,
    )
    chunks: list[TextChunk] = []
    for local_parent_index, (parent_units, section_title) in enumerate(parents):
        parent_text = "\n\n".join(unit for unit, _section in parent_units).strip()
        parent_token_count = count_tokens(parent_text, tokenizer_name=tokenizer_name)
        child_texts = _group_child_units(
            parent_units,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_name=tokenizer_name,
        )
        parent_index = parent_offset + local_parent_index
        for child_index, child_text in enumerate(child_texts):
            chunks.append(
                TextChunk(
                    text=child_text,
                    page_number=page_number,
                    section_title=section_title,
                    token_count=count_tokens(child_text, tokenizer_name=tokenizer_name),
                    parent_index=parent_index,
                    parent_text=parent_text,
                    parent_token_count=parent_token_count,
                    child_index=child_index,
                )
            )
    return chunks, len(parents)


def _token_structural_units(
    text: str,
    *,
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[str, str | None]]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    units: list[tuple[str, str | None]] = []
    for block, section in _blocks_with_sections(normalized):
        if count_tokens(block, tokenizer_name=tokenizer_name) <= max_tokens:
            units.append((block, section))
            continue
        for sentence in _sentence_units(block):
            if count_tokens(sentence, tokenizer_name=tokenizer_name) <= max_tokens:
                units.append((sentence, section))
                continue
            units.extend(
                (piece, section)
                for piece in _hard_split_tokens(
                    sentence,
                    max_tokens=max_tokens,
                    tokenizer_name=tokenizer_name,
                )
            )
    return units


def _group_parent_units(
    units: list[tuple[str, str | None]],
    *,
    parent_max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[list[tuple[str, str | None]], str | None]]:
    parents: list[tuple[list[tuple[str, str | None]], str | None]] = []
    current: list[tuple[str, str | None]] = []
    current_section: str | None = None
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_section, current_tokens
        if current:
            parents.append((current, current_section))
        current = []
        current_section = None
        current_tokens = 0

    for unit, section in units:
        unit_tokens = count_tokens(unit, tokenizer_name=tokenizer_name)
        section_changed = bool(
            current
            and section
            and current_section
            and _normalize_section(section) != _normalize_section(current_section)
        )
        if section_changed or (current and current_tokens + unit_tokens > parent_max_tokens):
            flush()
        current.append((unit, section))
        current_tokens += unit_tokens
        if section:
            current_section = section
    flush()
    return parents


def _group_child_units(
    units: list[tuple[str, str | None]],
    *,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer_name: str,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit, _section in units:
        unit_tokens = count_tokens(unit, tokenizer_name=tokenizer_name)
        if current and current_tokens + unit_tokens > max_tokens:
            chunk_text_value = "\n\n".join(current).strip()
            chunks.append(chunk_text_value)
            overlap_text = _tail_token_overlap(
                chunk_text_value,
                overlap_tokens=overlap_tokens,
                tokenizer_name=tokenizer_name,
            )
            if (
                overlap_text
                and count_tokens(overlap_text, tokenizer_name=tokenizer_name) + unit_tokens
                > max_tokens
            ):
                overlap_text = ""
            current = [overlap_text] if overlap_text else []
            current_tokens = count_tokens(overlap_text, tokenizer_name=tokenizer_name)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _hard_split_tokens(
    text: str,
    *,
    max_tokens: int,
    tokenizer_name: str,
) -> list[str]:
    tokenizer = _tokenizer(tokenizer_name)
    tokens = tokenizer.encode(text, disallowed_special=())
    return [
        tokenizer.decode(tokens[start : start + max_tokens]).strip()
        for start in range(0, len(tokens), max_tokens)
        if tokens[start : start + max_tokens]
    ]


def _tail_token_overlap(
    text: str,
    *,
    overlap_tokens: int,
    tokenizer_name: str,
) -> str:
    if overlap_tokens <= 0:
        return ""
    tokenizer = _tokenizer(tokenizer_name)
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= overlap_tokens:
        return text.strip()
    tail = tokenizer.decode(tokens[-overlap_tokens:]).strip()
    sentence_boundary = re.search(r"(?<=[.!?。！？])\s+", tail)
    if sentence_boundary and sentence_boundary.end() < len(tail) - 20:
        tail = tail[sentence_boundary.end() :].strip()
    return tail


@lru_cache(maxsize=4)
def _tokenizer(name: str):
    if name != DEFAULT_TOKENIZER:
        raise ValueError(f"Unsupported offline tokenizer: {name}")
    return _UnicodeLexicalTokenizer()


class _UnicodeLexicalTokenizer:
    """Small offline codec for stable token-budget chunking.

    Tokens retain their leading whitespace so decoding a slice reconstructs
    the source text without downloading a provider-specific vocabulary.  This
    is intentionally tokenizer-independent: the embedding endpoint still owns
    final truncation, while indexing gets deterministic word/punctuation
    budgets in English and Vietnamese.
    """

    _pattern = re.compile(r"\s*(?:[\w]+|[^\w\s])", flags=re.UNICODE)

    def encode(
        self,
        text: str,
        *,
        disallowed_special: tuple = (),
    ) -> list[str]:
        del disallowed_special
        return self._pattern.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


def _blocks_with_sections(text: str) -> list[tuple[str, str | None]]:
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if len(raw_blocks) <= 1:
        raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

    blocks: list[tuple[str, str | None]] = []
    current_section: str | None = None
    for block in raw_blocks:
        heading = _detect_section_title(block)
        if heading:
            current_section = heading
        blocks.append((block, current_section))
    return blocks


def _detect_section_title(block: str) -> str | None:
    first_line = block.strip().splitlines()[0].strip()
    if not first_line:
        return None
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()[:160] or None
    if len(first_line) > 140 or first_line.endswith("."):
        return None
    if re.match(r"^(?:[IVX]+\.|\d+(?:\.\d+)*\.?)\s+[A-ZÀ-ỸA-Za-z][^\n]{2,}$", first_line):
        return first_line[:160]
    if first_line.isupper() and 4 <= len(first_line) <= 80:
        return first_line[:160]
    return None


def _normalize_section(section: str) -> str:
    return " ".join(section.lower().replace("#", " ").split())


def _split_oversized_block(block: str, *, chunk_size: int) -> list[str]:
    if len(block) <= chunk_size:
        return [block]

    sentences = _sentence_units(block)
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_length = 0
            pieces.extend(_hard_split(sentence, chunk_size))
            continue
        if current and current_length + len(sentence) + 1 > chunk_size:
            pieces.append(" ".join(current).strip())
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence) + 1
    if current:
        pieces.append(" ".join(current).strip())
    return [piece for piece in pieces if piece]


def _sentence_units(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-ZÀ-Ỹ0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _hard_split(text: str, chunk_size: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        pieces.append(text[start:end].strip())
        start = max(end, start + 1)
    return [piece for piece in pieces if piece]


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    start = max(0, len(text) - overlap)
    boundary = max(text.find(". ", start), text.find("\n\n", start), text.find("\n", start))
    if boundary > start and boundary < len(text) - 20:
        start = boundary + 1
    return text[start:].strip()
