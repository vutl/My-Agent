from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ComposedContext:
    sources: list[dict[str, Any]]
    context_text: str
    stats: dict[str, Any]


def compose_retrieval_context(
    results: list[dict[str, Any]],
    *,
    max_sources: int = 8,
    max_chars: int = 5_500,
    max_chars_per_source: int = 1_200,
    max_chunks_per_document: int = 2,
    min_figures: int = 0,
    min_tables: int = 0,
    query: str | None = None,
    max_table_chars: int = 4_200,
    max_figure_chars: int = 2_000,
    required_document_ids: list[str] | None = None,
    required_facets_by_document: dict[str, list[str]] | None = None,
) -> ComposedContext:
    ordered_results = _prioritize_results_for_context(
        results,
        min_figures=min_figures,
        min_tables=min_tables,
        query=query,
    )
    ordered_results, reserved_result_ids, _ = (
        _reserve_required_document_sources(
            ordered_results,
            required_document_ids=required_document_ids or [],
            required_facets_by_document=required_facets_by_document or {},
            prefer_tables=min_tables > 0,
            query=query,
        )
    )
    sources: list[dict[str, Any]] = []
    per_document_count: dict[str, int] = defaultdict(int)
    seen_chunks: set[str] = set()
    used_chars = 0
    reservation_content_budgets = _reservation_content_budgets(
        ordered_results,
        reserved_result_ids=reserved_result_ids,
        max_chars=max_chars,
    )

    effective_max_sources = max(max_sources, len(reserved_result_ids))
    for result in ordered_results:
        chunk_id = str(result.get("chunk_id") or result.get("id") or "")
        document_id = str(result.get("document_id") or "")
        is_figure = bool(result.get("figure_id") or result.get("artifact_type") == "figure")
        is_table = bool(result.get("table_id") or result.get("artifact_type") == "table")
        is_artifact = is_figure or is_table
        is_required_reservation = id(result) in reserved_result_ids
        if chunk_id and chunk_id in seen_chunks:
            continue
        if (
            not is_required_reservation
            and not is_artifact
            and document_id
            and per_document_count[document_id] >= max_chunks_per_document
        ):
            continue

        raw_content = str(result.get("expanded_content") or result.get("content") or result.get("text") or "")
        content = _clean_content(raw_content, preserve_lines=is_table)
        if not content:
            continue
        reservation_content_budget = reservation_content_budgets.get(id(result))
        if is_table:
            content = _truncate_table(
                content,
                min(max_table_chars, reservation_content_budget)
                if reservation_content_budget
                else max_table_chars,
                query=query,
            )
        elif is_figure:
            content = _truncate(
                content,
                min(max_figure_chars, reservation_content_budget)
                if reservation_content_budget
                else max_figure_chars,
            )
        else:
            content = _truncate_text(
                content,
                min(max_chars_per_source, reservation_content_budget)
                if reservation_content_budget
                else max_chars_per_source,
                query=query,
            )
        source_id = f"SOURCE {len(sources) + 1}"
        citation_label = _citation_label(result)
        block = _format_source_block(source_id, citation_label, result, content)
        if (
            used_chars
            and used_chars + len(block) > max_chars
            and not is_required_reservation
        ):
            # A large table/figure should not prevent a later compact source from
            # being considered. Required facet cells are a correctness
            # invariant, so their already-projected source may exceed this soft
            # prompt target instead of disappearing silently.
            continue
        if not used_chars and len(block) > max_chars:
            available = max(300, max_chars - (len(block) - len(content)))
            content = (
                _truncate_table(content, available, query=query)
                if is_table
                else _truncate(content, available)
            )
            block = _format_source_block(source_id, citation_label, result, content)

        source = {
            **result,
            "source_id": source_id,
            "citation_label": citation_label,
            "content": content,
        }
        sources.append(source)
        used_chars += len(block)
        if chunk_id:
            seen_chunks.add(chunk_id)
        if document_id and not is_artifact:
            per_document_count[document_id] += 1
        if len(sources) >= effective_max_sources:
            break

    context_text = "\n\n".join(
        _format_source_block(source["source_id"], source["citation_label"], source, source["content"])
        for source in sources
    )
    if not context_text:
        context_text = "No local document excerpts were retrieved."
    return ComposedContext(
        sources=sources,
        context_text=context_text,
        stats={
            "source_count": len(sources),
            "character_count": len(context_text),
            "document_count": len({source.get("document_id") for source in sources if source.get("document_id")}),
            "table_source_count": sum(
                1 for source in sources if source.get("table_id") or source.get("artifact_type") == "table"
            ),
            "figure_source_count": sum(
                1 for source in sources if source.get("figure_id") or source.get("artifact_type") == "figure"
            ),
            "required_facet_coverage": _required_facet_coverage(
                sources,
                required_facets_by_document or {},
            ),
        },
    )


def _reserve_required_document_sources(
    results: list[dict[str, Any]],
    *,
    required_document_ids: list[str],
    required_facets_by_document: dict[str, list[str]],
    prefer_tables: bool,
    query: str | None,
) -> tuple[list[dict[str, Any]], set[int], int]:
    """Reserve the smallest useful source set for every document/facet cell.

    A multi-document prompt is invalid if one prolific paper consumes every
    context slot.  A card-backed prompt is also incomplete when a paper is
    present but one of its requested facets was evicted. Evidence sources can
    cover several facets, so a deterministic greedy set cover keeps the prompt
    compact instead of blindly reserving one source per facet. Documents with
    no tagged facet source still retain one conservative fallback source.
    """

    ordered_ids = list(
        dict.fromkeys(str(value).strip() for value in required_document_ids if str(value).strip())
    )
    facet_requirements = {
        str(document_id).strip(): list(
            dict.fromkeys(
                str(facet).strip()
                for facet in facets
                if str(facet).strip()
            )
        )
        for document_id, facets in required_facets_by_document.items()
        if str(document_id).strip()
    }
    for document_id in facet_requirements:
        if document_id not in ordered_ids:
            ordered_ids.append(document_id)
    if not ordered_ids or not results:
        return results, set(), 0

    reserved: list[dict[str, Any]] = []
    reserved_result_ids: set[int] = set()
    for document_id in ordered_ids:
        candidates = [
            result
            for result in results
            if str(result.get("document_id") or "").strip() == document_id
        ]
        if not candidates:
            continue
        uncovered = set(facet_requirements.get(document_id, []))
        while uncovered:
            ranked_candidates: list[tuple[tuple[int, int, int, float], dict[str, Any]]] = []
            for index, candidate in enumerate(candidates):
                if id(candidate) in reserved_result_ids:
                    continue
                candidate_facets = _source_evidence_facets(candidate)
                covered_now = uncovered.intersection(candidate_facets)
                if not covered_now:
                    continue
                is_table = bool(
                    candidate.get("table_id")
                    or candidate.get("artifact_type") == "table"
                )
                is_figure = bool(
                    candidate.get("figure_id")
                    or candidate.get("artifact_type") == "figure"
                )
                artifact_fit = int(
                    ("benchmark_results" in covered_now and is_table)
                    or ("visual_evidence" in covered_now and is_figure)
                )
                table_match, rank_score = _table_query_match_score(
                    candidate,
                    query=query or "",
                )
                ranked_candidates.append(
                    (
                        (len(covered_now), artifact_fit, table_match, rank_score - index * 1e-9),
                        candidate,
                    )
                )
            if not ranked_candidates:
                break
            _score, candidate = max(ranked_candidates, key=lambda item: item[0])
            reserved.append(candidate)
            reserved_result_ids.add(id(candidate))
            uncovered.difference_update(_source_evidence_facets(candidate))

        if any(
            str(result.get("document_id") or "").strip() == document_id
            for result in reserved
        ):
            continue
        table_candidates = [
            result
            for result in candidates
            if result.get("table_id") or result.get("artifact_type") == "table"
        ]
        if prefer_tables and table_candidates:
            candidate = max(
                table_candidates,
                key=lambda item: _table_query_match_score(item, query=query or ""),
            )
        else:
            candidate = candidates[0]
        reserved.append(candidate)
        reserved_result_ids.add(id(candidate))

    if not reserved:
        return results, set(), 0
    return (
        reserved + [result for result in results if id(result) not in reserved_result_ids],
        reserved_result_ids,
        len(reserved),
    )


def _source_evidence_facets(source: dict[str, Any]) -> set[str]:
    raw = source.get("evidence_facets") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(facet).strip() for facet in raw if str(facet).strip()}


def _reservation_content_budgets(
    ordered_results: list[dict[str, Any]],
    *,
    reserved_result_ids: set[int],
    max_chars: int,
) -> dict[int, int]:
    """Fit every required source into the shared character budget.

    Metadata overhead varies substantially with long paper filenames and table
    captions. A fixed per-source subtraction can therefore reserve a source in
    ranking but still evict it during rendering. Compute the real formatted
    overhead first, then divide only the remaining content capacity. Artifact
    sources receive a modestly larger share without compromising inclusion.
    """

    reserved = [
        result for result in ordered_results if id(result) in reserved_result_ids
    ]
    if not reserved:
        return {}
    overhead = 0
    weights: list[float] = []
    for index, result in enumerate(reserved):
        source_id = f"SOURCE {index + 1}"
        overhead += len(
            _format_source_block(
                source_id,
                _citation_label(result),
                result,
                "",
            )
        )
        is_artifact = bool(
            result.get("table_id")
            or result.get("figure_id")
            or result.get("artifact_type") in {"table", "figure"}
        )
        weights.append(1.35 if is_artifact else 1.0)
    available = max(0, max_chars - overhead - 2 * max(0, len(reserved) - 1))
    weight_total = sum(weights) or 1.0
    return {
        id(result): max(24, int(available * weight / weight_total) - 4)
        for result, weight in zip(reserved, weights, strict=True)
    }


def _required_facet_coverage(
    sources: list[dict[str, Any]],
    requirements: dict[str, list[str]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for document_id, requested_facets in requirements.items():
        requested = list(
            dict.fromkeys(
                str(item).strip()
                for item in requested_facets
                if str(item).strip()
            )
        )
        covered: set[str] = set()
        for source in sources:
            if str(source.get("document_id") or "").strip() == str(document_id).strip():
                covered.update(_source_evidence_facets(source))
        coverage.append(
            {
                "document_id": str(document_id),
                "requested_facets": requested,
                "covered_facets": [facet for facet in requested if facet in covered],
                "missing_facets": [facet for facet in requested if facet not in covered],
            }
        )
    return coverage


def _prioritize_results_for_context(
    results: list[dict[str, Any]],
    *,
    min_figures: int,
    min_tables: int,
    query: str | None,
) -> list[dict[str, Any]]:
    if (min_figures <= 0 and min_tables <= 0) or not results:
        return results

    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    seen_figure_ids: set[str] = set()
    seen_table_ids: set[str] = set()
    for result in results:
        figure_id = str(result.get("figure_id") or "")
        table_id = str(result.get("table_id") or "")
        if figure_id and min_figures > 0:
            if figure_id in seen_figure_ids:
                continue
            seen_figure_ids.add(figure_id)
            figures.append(result)
        elif min_tables > 0 and (table_id or result.get("artifact_type") == "table"):
            identity = table_id or str(result.get("chunk_id") or result.get("id") or "")
            if identity and identity in seen_table_ids:
                continue
            if identity:
                seen_table_ids.add(identity)
            tables.append(result)
        else:
            others.append(result)

    if not figures and not tables:
        return results

    if min_tables > 0 and query:
        # Lance similarity is still the primary candidate generator.  Within
        # its table candidates, prefer the artifact that covers more of the
        # explicitly requested metrics/terms before reserving the bounded
        # context slot.
        tables.sort(
            key=lambda item: _table_query_match_score(item, query=query),
            reverse=True,
        )

    prioritized = tables[:min_tables] + figures[:min_figures] + others
    seen_chunks: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for result in prioritized:
        chunk_id = str(result.get("chunk_id") or result.get("id") or "")
        if chunk_id and chunk_id in seen_chunks:
            continue
        if chunk_id:
            seen_chunks.add(chunk_id)
        deduped.append(result)
    for result in results:
        chunk_id = str(result.get("chunk_id") or result.get("id") or "")
        if chunk_id and chunk_id in seen_chunks:
            continue
        if chunk_id:
            seen_chunks.add(chunk_id)
        deduped.append(result)
    return deduped


def _table_query_match_score(result: dict[str, Any], *, query: str) -> tuple[int, float]:
    tokens = _table_query_tokens(query)
    searchable = "\n".join(
        str(result.get(field) or "")
        for field in ("caption", "content", "text")
    ).lower()
    matched = sum(1 for token in tokens if token in searchable)
    rank_score = float(result.get("rerank_score") or result.get("score") or 0.0)
    return matched, rank_score


def _format_source_block(source_id: str, citation_label: str, result: dict[str, Any], content: str) -> str:
    filename = result.get("filename") or "unknown"
    source_path = result.get("source_path") or ""
    page_number = result.get("page_number")
    raw_channels = result.get("retrieval_channels") or result.get("retrieval_channel") or []
    channels = raw_channels if isinstance(raw_channels, str) else ", ".join(raw_channels)
    ranks = _rank_text(result)
    page_line = f"page: {page_number}\n" if page_number is not None else ""
    section_line = f"section: {result.get('section_title')}\n" if result.get("section_title") else ""
    caption_line = f"caption: {result.get('caption')}\n" if result.get("caption") else ""
    artifact_line = f"artifact_type: {result.get('artifact_type')}\n" if result.get("artifact_type") else ""
    table_line = f"table_id: {result.get('table_id')}\n" if result.get("table_id") else ""
    figure_line = f"figure_id: {result.get('figure_id')}\n" if result.get("figure_id") else ""
    label_line = f"figure_label: {result.get('figure_label')}\n" if result.get("figure_label") else ""
    quality_line = (
        f"visual_quality: {result.get('quality_status')} / {result.get('asset_kind')}\n"
        if result.get("quality_status") or result.get("asset_kind")
        else ""
    )
    image_line = f"image: {result.get('image_path')}\n" if result.get("image_path") else ""
    return (
        f"[{source_id}]\n"
        f"file: {filename}\n"
        f"path: {source_path}\n"
        f"{page_line}"
        f"{section_line}"
        f"{caption_line}"
        f"{artifact_line}"
        f"{table_line}"
        f"{figure_line}"
        f"{label_line}"
        f"{quality_line}"
        f"{image_line}"
        f"retrieval: {channels or 'unknown'}{ranks}\n"
        f"content:\n{content}"
    )


def _rank_text(result: dict[str, Any]) -> str:
    parts = []
    if result.get("vector_rank") is not None:
        parts.append(f"vector_rank={result['vector_rank']}")
    if result.get("fts_rank") is not None:
        parts.append(f"fts_rank={result['fts_rank']}")
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def _citation_label(result: dict[str, Any]) -> str:
    filename = str(result.get("filename") or "source").strip() or "source"
    stem = filename.rsplit(".", 1)[0]
    label = " ".join(stem.replace("_", " ").replace("-", " ").split())
    if len(label) > 32:
        label = label[:29].rstrip() + "..."
    page_number = result.get("page_number")
    if page_number is not None:
        return f"{label} p.{page_number}"
    chunk_index = result.get("chunk_index")
    if chunk_index is not None:
        return f"{label} chunk {int(chunk_index) + 1}"
    return label


def _clean_content(content: str, *, preserve_lines: bool = False) -> str:
    if not preserve_lines:
        return " ".join(content.split())
    lines = [" ".join(line.split()) for line in content.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    boundary = content.rfind(". ", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = content.rfind(" ", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = max_chars
    return content[:boundary].rstrip() + "..."


_RESULT_QUERY_RE = re.compile(
    r"(?<!\w)(?:results?|performance|benchmark|metrics?|scores?|"
    r"kết\s*quả|hiệu\s*suất|chỉ\s*số|so\s*sánh|compare|comparison)(?!\w)",
    re.IGNORECASE,
)
_RESULT_EVIDENCE_TOKENS = (
    "accuracy",
    "acc",
    "f1",
    "ccc",
    "uar",
    "war",
    "wer",
    "mae",
    "mse",
    "rmse",
    "precision",
    "recall",
)
_PROPOSED_TABLE_ROW_RE = re.compile(
    r"(?<!\w)(?:ours?|proposed|full\s+(?:model|method|system)|"
    r"our\s+(?:model|method|system|approach)|"
    r"(?:mo\s+hinh|phuong\s+phap|he\s+thong)\s+de\s+xuat)(?!\w)",
    re.IGNORECASE,
)
_TABLE_QUERY_STOPWORDS = {
    "about",
    "and",
    "answer",
    "based",
    "benchmark",
    "bảng",
    "compare",
    "comparison",
    "dataset",
    "document",
    "from",
    "giữa",
    "kết",
    "label",
    "method",
    "model",
    "multi",
    "paper",
    "performance",
    "quả",
    "result",
    "results",
    "single",
    "system",
    "table",
    "the",
    "với",
}


def _truncate_text(content: str, max_chars: int, *, query: str | None) -> str:
    """Project a long passage around query evidence instead of its prefix.

    Parent expansion deliberately gives the model a larger section, but the
    per-source prompt budget is smaller. Prefix-only truncation can therefore
    discard the retrieved child or the result sentence that caused the hit.
    Candidate windows are scored from query vocabulary; result questions also
    reward metric/value density. This stays document- and domain-neutral.
    """

    if len(content) <= max_chars or not query:
        return _truncate(content, max_chars)

    # Parent/neighbor expansion is supporting context.  The retrieved child is
    # the passage that actually matched the query, so a later dense parent
    # region must never evict it from the per-source prompt budget.  Project
    # the child first; use leftover space for the expanded context only when
    # the child itself is compact enough.
    primary, supporting = _expanded_text_parts(content)
    if primary:
        label = "[retrieved chunk] "
        primary_budget = max(160, max_chars - len(label))
        projected_primary = _project_text_window(primary, primary_budget, query=query)
        rendered_primary = label + projected_primary
        remaining = max_chars - len(rendered_primary) - 3
        if supporting and remaining >= 180:
            projected_supporting = _project_text_window(
                supporting,
                remaining,
                query=query,
            )
            return rendered_primary + " | " + projected_supporting
        return _truncate(rendered_primary, max_chars)

    return _project_text_window(content, max_chars, query=query)


_PRIMARY_EXPANSION_MARKERS = (
    "[retrieved chunk / child]",
    "[retrieved chunk]",
)
_SUPPORTING_EXPANSION_MARKERS = (
    "[parent section context]",
    "[previous context]",
    "[next context]",
)


def _expanded_text_parts(content: str) -> tuple[str, str]:
    """Split retrieval expansion into authoritative child and supporting text."""

    marker_start = -1
    marker_end = -1
    for marker in _PRIMARY_EXPANSION_MARKERS:
        position = content.find(marker)
        if position >= 0 and (marker_start < 0 or position < marker_start):
            marker_start = position
            marker_end = position + len(marker)
    if marker_start < 0:
        return "", ""

    supporting_start = len(content)
    for marker in _SUPPORTING_EXPANSION_MARKERS:
        position = content.find(marker, marker_end)
        if position >= 0:
            supporting_start = min(supporting_start, position)
    primary = content[marker_end:supporting_start].strip()
    supporting = content[supporting_start:].strip() if supporting_start < len(content) else ""
    return primary, supporting


def _project_text_window(content: str, max_chars: int, *, query: str | None) -> str:
    """Select the best bounded window from one homogeneous text segment."""

    if len(content) <= max_chars or not query:
        return _truncate(content, max_chars)

    normalized_content = content.casefold()
    query_tokens = _table_query_tokens(query)
    result_intent = bool(_RESULT_QUERY_RE.search(query))
    if result_intent:
        query_tokens.update(_RESULT_EVIDENCE_TOKENS)
    query_tokens = {token for token in query_tokens if len(token) >= 2}
    if not query_tokens:
        return _truncate(content, max_chars)

    anchor_positions: list[int] = [0]
    for token in query_tokens:
        anchor_positions.extend(
            match.start()
            for match in re.finditer(re.escape(token.casefold()), normalized_content)
        )
    if len(anchor_positions) == 1:
        return _truncate(content, max_chars)

    candidates: list[tuple[tuple[int, int, int], int, int]] = []
    for position in dict.fromkeys(anchor_positions):
        start = max(0, min(len(content) - max_chars, position - max_chars // 2))
        if start:
            boundary = content.find(" ", start, min(len(content), start + 80))
            if boundary >= 0:
                start = boundary + 1
        end = min(len(content), start + max_chars)
        window = normalized_content[start:end]
        covered = sum(1 for token in query_tokens if token in window)
        occurrences = sum(min(window.count(token), 4) for token in query_tokens)
        numeric_density = (
            len(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", window))
            if result_intent
            else 0
        )
        candidates.append(((covered, occurrences, numeric_density), start, end))

    _score, start, end = max(candidates, key=lambda item: item[0])
    prefix = "... " if start else ""
    suffix = " ..." if end < len(content) else ""
    projected = content[start:end].strip()
    available = max(1, max_chars - len(prefix) - len(suffix))
    if len(projected) > available:
        # When the selected window reaches the end of the passage, preserve
        # that end (where a matched claim often completes) and trim its left
        # context.  Otherwise retain the selected window's leading anchor.
        projected = (
            projected[-available:].lstrip()
            if start and not suffix
            else projected[:available].rstrip()
        )
    return prefix + projected + suffix


def _truncate_table(content: str, max_chars: int, *, query: str | None) -> str:
    if len(content) <= max_chars:
        return content

    lines = content.splitlines()
    pipe_indices = [index for index, line in enumerate(lines) if "|" in line]
    if len(pipe_indices) < 2:
        return _truncate(content, max_chars)

    start = pipe_indices[0]
    table_lines: list[str] = []
    for line in lines[start:]:
        if "|" not in line:
            if table_lines:
                break
            continue
        table_lines.append(line)
    if len(table_lines) < 2:
        return _truncate(content, max_chars)

    prefix = lines[:start]
    header = table_lines[:2]
    rows = table_lines[2:]
    query_tokens = _table_query_tokens(query)
    if query_tokens:
        matching = [row for row in rows if any(token in row.lower() for token in query_tokens)]
        proposed = [
            row
            for row in rows
            if row not in matching and _PROPOSED_TABLE_ROW_RE.search(_fold_ascii(row))
        ]
        remaining = [row for row in rows if row not in matching and row not in proposed]
        rows = [*matching, *proposed, *remaining]

    selected = [*prefix, *header]
    for row in rows:
        candidate = "\n".join([*selected, row])
        if len(candidate) > max_chars:
            break
        selected.append(row)
    if len(selected) <= len(prefix) + len(header) and rows:
        # Never cut through a row; retain the first relevant row even when it
        # slightly exceeds the preferred per-source budget.
        selected.append(rows[0])
    if len(selected) < len(prefix) + len(table_lines):
        selected.append("<!-- additional table rows omitted after evidence-aware projection -->")
    return "\n".join(selected)


def _table_query_tokens(query: str | None) -> set[str]:
    if not query:
        return set()
    normalized = query.lower()
    metric_tokens = {
        token
        for token in ("acc", "accuracy", "f1", "ccc", "uar", "war", "wa", "ua", "wer", "mae", "mse", "rmse")
        if token in normalized
    }
    raw_words = re.findall(r"[a-z0-9à-ỹ]+(?:[-_][a-z0-9à-ỹ]+)*", normalized)
    words: set[str] = set()
    for raw_word in raw_words:
        variants = {raw_word, *re.split(r"[-_]+", raw_word)}
        for word in variants:
            if len(word) >= 3 and word not in _TABLE_QUERY_STOPWORDS:
                words.add(word)
    return metric_tokens | words


def _fold_ascii(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold().replace("đ", "d"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))
