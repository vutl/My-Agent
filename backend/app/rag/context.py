from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any

from app.rag.evidence_identity import (
    canonical_artifact_type,
    canonical_evidence_id,
    has_physical_pages,
    own_evidence_text,
)
from app.rag.paper_facets import extract_query_facets, requested_paper_facets


@dataclass(frozen=True)
class ComposedContext:
    sources: list[dict[str, Any]]
    context_text: str
    stats: dict[str, Any]


def merge_page_coverage_sources(
    text_pages: list[dict[str, Any]],
    visual_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join lexical/OCR and visual page frontiers deterministically.

    Coverage prompts have a finite source budget.  Appending visual hits after
    a lexical frontier can therefore discard every new visual page before the
    composer sees it.  This shared helper merges same-page provenance and
    interleaves the two ranked frontiers so both channels get a bounded chance
    to contribute.  It is a context operation only; it never changes canonical
    evidence identity or ordinary text-only ordering.
    """

    merged = list(text_pages)
    by_page = {
        (str(item.get("document_id") or ""), int(item.get("page_number"))): index
        for index, item in enumerate(merged)
        if item.get("document_id") and item.get("page_number") is not None
    }
    for visual in visual_pages:
        document_id = str(visual.get("document_id") or "")
        page = visual.get("page_number")
        if not document_id or page is None:
            continue
        key = (document_id, int(page))
        existing_index = by_page.get(key)
        if existing_index is None:
            # Visual hits are page-coverage evidence even when they do not
            # have a lexical/OCR counterpart. Mark the copy here so the
            # composer can exempt it from the ordinary per-document chunk
            # cap and preserve it for multimodal prompts.
            merged.append(
                {
                    **visual,
                    "visual_page": True,
                    "coverage_mode": True,
                }
            )
            by_page[key] = len(merged) - 1
            continue
        existing = merged[existing_index]
        channels = list(existing.get("retrieval_channels") or [])
        for channel in visual.get("retrieval_channels") or []:
            if channel not in channels:
                channels.append(channel)
        existing.update(
            {
                "image_path": visual.get("image_path") or existing.get("image_path"),
                "image_url": visual.get("image_url") or existing.get("image_url"),
                "visual_score": visual.get("visual_score"),
                "visual_page": True,
                "coverage_mode": True,
                "retrieval_channels": channels,
                "metadata": {
                    **(existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}),
                    **(visual.get("metadata") if isinstance(visual.get("metadata"), dict) else {}),
                },
            }
        )
    if not visual_pages:
        return merged

    text_keys = [
        (str(item.get("document_id") or ""), int(item["page_number"]))
        for item in text_pages
        if item.get("document_id") and item.get("page_number") is not None
    ]
    visual_keys = [
        (str(item.get("document_id") or ""), int(item["page_number"]))
        for item in visual_pages
        if item.get("document_id") and item.get("page_number") is not None
    ]
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index in range(max(len(text_keys), len(visual_keys))):
        for keys in (visual_keys, text_keys):
            if index >= len(keys):
                continue
            key = keys[index]
            if key in seen or key not in by_page:
                continue
            ordered.append(merged[by_page[key]])
            seen.add(key)
    for item in merged:
        if not item.get("document_id") or item.get("page_number") is None:
            continue
        key = (str(item.get("document_id") or ""), int(item["page_number"]))
        if key not in seen:
            ordered.append(item)
            seen.add(key)
    return ordered


def select_ranked_evidence(
    results: list[dict[str, Any]],
    *,
    limit: int,
    query: str | None = None,
    required_document_ids: list[str] | None = None,
    required_facets_by_document: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded evidence set with deterministic coverage guarantees.

    First-stage retrieval is allowed to rank several near-duplicate chunks
    from one section highly.  This selector keeps the original score as the
    primary signal but reserves one strong result for every required document
    and, when structured section metadata supports it, every requested facet.
    It is intentionally metadata-only: it never reads qrels or invents paper
    names, and it does not re-embed candidates.
    """

    if limit < 1 or not results:
        return [], {"selected_count": 0, "reserved_count": 0, "covered_facets": {}}

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        identity = canonical_evidence_id(result)
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        deduped.append(result)

    ordered_ids = list(dict.fromkeys(str(value).strip() for value in (required_document_ids or []) if str(value).strip()))
    facet_requirements = {
        str(document_id).strip(): list(dict.fromkeys(str(facet).strip() for facet in facets if str(facet).strip()))
        for document_id, facets in (required_facets_by_document or {}).items()
        if str(document_id).strip()
    }
    if ordered_ids and not facet_requirements and query:
        requested = requested_paper_facets(query, focused_document_count=len(ordered_ids))
        facet_requirements = {document_id: list(requested) for document_id in ordered_ids}

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: dict[str, set[str]] = defaultdict(set)

    def add(result: dict[str, Any]) -> bool:
        if len(selected) >= limit:
            return False
        identity = canonical_evidence_id(result)
        if identity and identity in selected_ids:
            return False
        selected.append(result)
        if identity:
            selected_ids.add(identity)
        document_id = str(result.get("document_id") or "").strip()
        covered[document_id].update(_result_facets(result))
        return True

    def candidates_for(document_id: str) -> list[dict[str, Any]]:
        return [result for result in deduped if str(result.get("document_id") or "").strip() == document_id]

    preferred_artifact = _query_artifact_kind(query or "")

    # Exact artifact requests are a correctness constraint, not a soft score.
    # Reserve one canonical table/figure per required document before semantic
    # facet coverage can consume all ten slots.
    if preferred_artifact:
        for document_id in ordered_ids:
            artifact_candidates = [
                item
                for item in candidates_for(document_id)
                if str(item.get("artifact_type") or item.get("chunk_type") or "").lower() == preferred_artifact
                or bool(item.get(f"{preferred_artifact}_id"))
            ]
            if artifact_candidates:
                add(
                    max(
                        artifact_candidates,
                        key=lambda item: (
                            _artifact_query_match(item, preferred_artifact, query or ""),
                            len(set(facet_requirements.get(document_id, [])).intersection(_result_facets(item))),
                            _table_query_match_score(item, query=query or "") if preferred_artifact == "table" else (),
                            _selection_score(item),
                        ),
                    )
                )
                if len(selected) >= limit:
                    break

    # Reserve facet cells first. A single result may satisfy multiple cells.
    for document_id in ordered_ids:
        candidates = candidates_for(document_id)
        uncovered = set(facet_requirements.get(document_id, []))
        while uncovered and len(selected) < limit:
            available = [candidate for candidate in candidates if canonical_evidence_id(candidate) not in selected_ids]
            if not available:
                break
            candidate = max(
                available,
                key=lambda item: (
                    len(uncovered.intersection(_result_facets(item))),
                    _facet_artifact_fit(item, uncovered.intersection(_result_facets(item))),
                    _table_query_match_score(item, query=query or ""),
                    _selection_score(item),
                ),
            )
            facets = uncovered.intersection(_result_facets(candidate))
            if not facets:
                break
            add(candidate)
            uncovered.difference_update(facets)
        if len(selected) >= limit:
            break
        if candidates and not any(str(item.get("document_id") or "").strip() == document_id for item in selected):
            add(max(candidates, key=_selection_score))

    # Keep a mild per-document cap for multi-document requests. This prevents
    # one paper from consuming every remaining slot after the reservations.
    per_document_cap = None
    if len(ordered_ids) > 1:
        per_document_cap = max(2, math.ceil(limit / len(ordered_ids)) + 1)
    counts: dict[str, int] = defaultdict(int)
    for item in selected:
        counts[str(item.get("document_id") or "").strip()] += 1

    # A comparison/multi-document answer needs a second independent source
    # opportunity per document when the budget permits.  This is deliberately
    # document-level, not paper-name or qrel-specific: it prevents an early
    # cluster of near-duplicates from monopolising the final ten while keeping
    # the existing per-document cap and score ordering for the remainder.
    if len(ordered_ids) > 1 and limit >= 2 * len(ordered_ids):
        target_per_document = min(2, limit // len(ordered_ids))
        for document_id in ordered_ids:
            while counts[document_id] < target_per_document and len(selected) < limit:
                available = [
                    candidate
                    for candidate in deduped
                    if str(candidate.get("document_id") or "").strip() == document_id
                    and canonical_evidence_id(candidate) not in selected_ids
                ]
                if not available:
                    break
                candidate = max(available, key=_selection_score)
                if add(candidate):
                    counts[document_id] += 1
                else:
                    break

    for candidate in sorted(deduped, key=_selection_score, reverse=True):
        if len(selected) >= limit:
            break
        identity = canonical_evidence_id(candidate)
        if identity and identity in selected_ids:
            continue
        document_id = str(candidate.get("document_id") or "").strip()
        if per_document_cap is not None and counts[document_id] >= per_document_cap:
            continue
        if add(candidate):
            counts[document_id] += 1

    return selected[:limit], {
        "selected_count": len(selected[:limit]),
        "reserved_count": min(len(selected), sum(1 for item in selected if canonical_evidence_id(item) in selected_ids)),
        "required_documents": ordered_ids,
        "covered_facets": {document_id: sorted(values) for document_id, values in covered.items() if document_id},
    }


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float]:
    score = float(result.get("score") or result.get("rrf_score") or 0.0)
    lexical = float(result.get("raw_lexical_score") or result.get("fts_score") or 0.0)
    vector = float(result.get("raw_vector_score") or 0.0)
    return score, lexical, -vector


def _facet_artifact_fit(result: dict[str, Any], facets: set[str]) -> int:
    """Prefer typed evidence for a facet it independently supplies.

    Dataset/count tables answer dataset questions just as result tables answer
    score questions. Artifact identity alone does not confer a facet; callers
    first intersect the shared own-evidence classifier with the request.
    """
    artifact_type = canonical_artifact_type(result)
    is_table = bool(result.get("table_id") or artifact_type == "table")
    is_figure = bool(result.get("figure_id") or artifact_type == "figure")
    return int(
        (is_table and bool(facets.intersection({"dataset_setup", "benchmark_results"})))
        or (is_figure and bool(facets.intersection({"architecture", "visual_evidence"})))
    )


def _result_facets(result: dict[str, Any]) -> set[str]:
    raw = result.get("evidence_facets") or result.get("facets") or []
    if isinstance(raw, str):
        raw = [raw]
    facets = {str(value).strip() for value in raw if str(value).strip()}
    artifact_type = canonical_artifact_type(result)
    is_table = bool(result.get("table_id") or artifact_type == "table")
    if is_table and _table_has_benchmark_evidence(result):
        facets.add("benchmark_results")
    if result.get("figure_id") or artifact_type == "figure":
        facets.add("visual_evidence")
    labels = " ".join(
        [
            str(result.get("section_title") or ""),
            " ".join(str(item) for item in result.get("heading_path") or []),
            str(result.get("caption") or ""),
        ]
    ).casefold()
    # Some older/local projections do not persist ``evidence_facets`` on text
    # chunks.  The candidate is still usable when its section or content
    # carries an unambiguous structural cue.  Keep this inference bounded to
    # metadata + a short content prefix: it is a routing hint, never a qrel or
    # answer label, and it avoids turning an arbitrary long passage into every
    # possible facet.
    content = " ".join(
        own_evidence_text(str(result.get(field) or ""))
        for field in ("content", "text", "expanded_content")
    )[:1_200].casefold()
    markers = {
        "task": ("abstract", "introduction", "problem", "objective", "purpose", "background"),
        "architecture": ("architecture", "method", "model", "framework", "pipeline", "proposed method", "design"),
        "dataset_setup": ("dataset", "data", "experimental setup", "protocol", "corpus"),
        "benchmark_results": ("result", "experiment", "evaluation", "benchmark", "table"),
        "contributions": ("contribution", "novel", "proposed", "we present"),
        "training_method": ("training", "loss", "optimizer", "objective"),
        "ablation": ("ablation", "without", "component"),
        "limitations": ("limitation", "threats", "failure"),
    }
    for facet, values in markers.items():
        # An artifact's role belongs to its own caption/header, not the page's
        # surrounding prose. A dataset/count table alongside a results section
        # does not itself supply benchmark scores.
        if is_table and facet == "benchmark_results":
            continue
        # Broad vocabulary is safe in section/caption metadata, where
        # ``model`` or ``results`` usually denotes the section itself.  Text
        # content needs narrower anchors: a table header containing ``Model``
        # must not be mislabeled as architecture and evict the actual method
        # chunk from a bounded context.
        content_values = {
            "task": (
                "abstract", "introduction", "problem", "objective", "purpose",
                "we propose", "we address", "in this paper", "this paper",
                "presents", "provides", "solves", "designed to", "aims to",
                "is a", "là một", "là gì", "giải quyết", "mục tiêu",
            ),
            "architecture": ("architecture", "framework", "pipeline", "encoder", "fusion", "cross-attention", "module", "component", "layer", "proposed method"),
            "dataset_setup": ("dataset", "data split", "corpus", "protocol", "participants", "experimental setup"),
            "benchmark_results": ("results", "result", "experiment", "evaluation", "benchmark", "accuracy", "f1", "ccc", "performance", "table"),
            "contributions": ("contribution", "novel", "we propose", "we present", "proposed method"),
            "training_method": ("training", "loss", "optimizer", "objective"),
            "ablation": ("ablation", "without component", "remove the"),
            "limitations": ("limitation", "threats", "failure cases"),
            "visual_evidence": ("figure", "diagram", "chart", "plot"),
        }.get(facet, values)
        if any(value in labels for value in values) or any(value in content for value in content_values):
            facets.add(facet)
    return facets


def _table_has_benchmark_evidence(result: dict[str, Any]) -> bool:
    """Conservative purpose signal, independent of artifact number/identity.

    Captions support image-backed tables; metric headers support unnamed text
    tables. Neither neighboring context nor a substring such as F10 can turn
    input/corpus statistics into performance evidence. This is a selection
    hint, not a claim that every requested value is readable or supported.
    """

    caption = str(result.get("caption") or "").casefold()
    # "Training-set statistics for benchmark variants" describes data, not
    # performance. Benchmark/evaluation alone names a protocol or setting;
    # require a result/score/performance role, or actual metric columns.
    if _RESULT_CAPTION_RE.search(caption) or re.search(r"(?<!\w)ablation(?!\w)", caption):
        return True
    return _table_has_metric_columns(result)


# Result/comparison captions (EN + VI). Checked before statistics cues so
# "Results under distribution shift" is a result table.
_RESULT_CAPTION_RE = re.compile(
    r"(?<!\w)(?:results?|comparison|comparative|compared|performance|evaluation scores?|"
    r"state[- ]of[- ]the[- ]art|sota|kết\s+quả|so\s+sánh|hiệu\s+(?:năng|suất)|"
    r"độ\s+chính\s+xác)(?!\w)",
    re.IGNORECASE,
)
_STATISTICS_CAPTION_RE = re.compile(
    r"(?<!\w)(?:statistics|distribution|dataset summary|hyperparameters?|"
    r"training settings?|thống\s+kê|phân\s+(?:bố|phối)|siêu\s+tham\s+số|"
    r"cấu\s+hình\s+huấn\s+luyện)(?!\w)",
    re.IGNORECASE,
)
_ABLATION_CAPTION_RE = re.compile(
    r"(?<!\w)(?:ablation|ablations|sensitivity|component removal|"
    r"loại\s+bỏ\s+thành\s+phần|phân\s+tích\s+loại\s+bỏ)(?!\w)",
    re.IGNORECASE,
)
# Domain-neutral evaluation metrics; a table whose header carries them reports
# measurements even when its caption is missing or unhelpful.
_METRIC_COLUMN_RE = re.compile(
    r"(?<!\w)(?:accuracy|acc|precision|recall|(?:micro|macro)[ -]?f1|mif1|maf1|f1|"
    r"ccc(?:[ _-]?[avd])?|ua|wa|uar|war|wer|cer|bleu|rouge(?:[ _-]?[12l])?|meteor|cider|"
    r"pearson(?: correlation)?|spearman(?: correlation)?|auc|roc|rmse|mae|mse|mape|"
    r"map|ap50|ap|iou|miou|dice|em|exact match|top-?[15]|hit@\d+|recall@\d+|"
    r"ndcg(?:@\d+)?|mrr|psnr|ssim|fid|lpips|perplexity|ppl|r2|"
    r"error rate|độ chính xác|sai số)(?!\w)",
    re.IGNORECASE,
)


def _table_has_metric_columns(result: dict[str, Any]) -> bool:
    content = str(result.get("content") or result.get("markdown") or result.get("text") or "")
    # All non-separator Markdown rows are considered: multi-task tables can
    # repeat their schema deep in the artifact. Never use expanded_content,
    # which may belong to a different artifact on the same page.
    rows = [line for line in content.splitlines() if line.strip().startswith("|")]
    # Header rows are those directly above a |---| separator (tables may
    # repeat their header deep in the artifact); body cells are not schema.
    separator = re.compile(r"^\s*\|?\s*:?-{2,}")
    headers = [row for row, following in zip(rows, rows[1:]) if separator.match(following)]
    schema = "\n".join(headers) if headers else (rows[0] if rows else content[:1_200])
    return bool(_METRIC_COLUMN_RE.search(schema))


def table_result_role(result: dict[str, Any]) -> str:
    """Classify a table's own purpose, never its index or neighboring prose.

    Ablation/sensitivity and corpus statistics are not interchangeable with
    primary benchmark comparisons. Unknown captions remain unknown; this is
    a selection hint, not authorization for any quantitative claim.
    """
    caption = str(result.get("caption") or "").casefold()
    if _ABLATION_CAPTION_RE.search(caption):
        return "ablation"
    if _RESULT_CAPTION_RE.search(caption):
        return "main"
    if _STATISTICS_CAPTION_RE.search(caption):
        return "statistics"
    # No role in the caption: a header of evaluation metrics still marks a
    # measurement table (e.g. "Accuracy (%) of different models").
    if _table_has_metric_columns(result) or _METRIC_COLUMN_RE.search(caption):
        return "main"
    return "unknown"


def _query_artifact_kind(query: str) -> str | None:
    lowered = " ".join(str(query or "").casefold().split())
    figure_markers = ("figure", "fig.", "diagram", "chart", "plot", "hình", "sơ đồ", "biểu đồ")
    table_markers = ("table", "bảng", "metric", "metrics", "accuracy", "f1", "ccc", "results", "kết quả", "điểm")
    if any(marker in lowered for marker in figure_markers):
        return "figure"
    if any(marker in lowered for marker in table_markers):
        return "table"
    return None


def _artifact_query_match(result: dict[str, Any], kind: str, query: str) -> int:
    """Prefer the explicitly requested Table/Figure number when present."""

    lowered = " ".join(str(query or "").casefold().split())
    marker = "table" if kind == "table" else "figure"
    localized = "bảng" if kind == "table" else "hình"
    short = "fig\\.?" if kind == "figure" else "table"
    match = re.search(
        rf"(?<!\w)(?:{marker}|{short}|{localized})\s*"
        rf"(?:#|no\.?|number|số)?\s*(\d+)(?!\w)",
        lowered,
    )
    if match is None:
        return 0
    requested = int(match.group(1))
    raw_index = result.get(f"{kind}_index")
    try:
        index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        index = None
    if index is not None and index == requested - 1:
        return 4
    if index is not None and index == requested:
        # Legacy projections occasionally stored one-based artifact indices;
        # keep this as a weaker compatibility match than the canonical
        # zero-based representation.
        return 2
    caption = str(result.get("caption") or "").casefold()
    if re.search(
        rf"(?<!\w)(?:{marker}|{short}|{localized})\s*{requested}(?:\D|$)",
        caption,
    ):
        return 4
    return 0


def compose_retrieval_context(
    results: list[dict[str, Any]],
    *,
    max_sources: int = 8,
    max_chars: int = 5_500,
    max_chars_per_source: int = 1_200,
    max_chunks_per_document: int = 2,
    min_figures: int = 0,
    min_tables: int = 0,
    min_visual_pages: int = 0,
    query: str | None = None,
    max_table_chars: int = 4_200,
    max_figure_chars: int = 2_000,
    required_document_ids: list[str] | None = None,
    required_facets_by_document: dict[str, list[str]] | None = None,
    coverage_mode: bool = False,
    selection_trace_limit: int = 256,
) -> ComposedContext:
    # A page-coverage merge marks visual pages as coverage evidence.  Preserve
    # one of those pages even when an upstream caller predates the explicit
    # ``min_visual_pages`` parameter; this keeps older LightRAG/fallback
    # composition paths multimodal-safe without reserving images in ordinary
    # text retrieval.
    effective_min_visual_pages = int(min_visual_pages or 0)
    if effective_min_visual_pages <= 0 and coverage_mode:
        effective_min_visual_pages = int(
            any(_is_visual_page_result(result) for result in results)
        )
    ordered_document_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in (required_document_ids or [])
            if str(value).strip()
        )
    )
    effective_facet_requirements = {
        str(document_id).strip(): list(
            dict.fromkeys(
                str(facet).strip()
                for facet in facets
                if str(facet).strip()
            )
        )
        for document_id, facets in (required_facets_by_document or {}).items()
        if str(document_id).strip()
    }
    if ordered_document_ids and not effective_facet_requirements and query:
        requested = requested_paper_facets(
            query,
            focused_document_count=len(ordered_document_ids),
        )
        effective_facet_requirements = {
            document_id: list(requested)
            for document_id in ordered_document_ids
        }
    ordered_results = _prioritize_results_for_context(
        results,
        min_figures=min_figures,
        min_tables=min_tables,
        min_visual_pages=effective_min_visual_pages,
        query=query,
    )
    ordered_results, reserved_result_ids, _ = (
        _reserve_required_document_sources(
            ordered_results,
            required_document_ids=ordered_document_ids,
            required_facets_by_document=effective_facet_requirements,
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
    selection_counts: Counter[str] = Counter()
    selection_events: list[dict[str, Any]] = []

    def record_selection(result: dict[str, Any], index: int, reason: str, **details: Any) -> None:
        selection_counts[reason] += 1
        if len(selection_events) < max(0, selection_trace_limit):
            selection_events.append({
                "ordered_index": index,
                "evidence_id": canonical_evidence_id(result),
                "document_id": result.get("document_id"),
                "artifact_type": canonical_artifact_type(result),
                "reason": reason, **details,
            })

    for index, result in enumerate(ordered_results):
        evidence_id = canonical_evidence_id(result)
        document_id = str(result.get("document_id") or "")
        result_type = canonical_artifact_type(result)
        is_figure = bool(result.get("figure_id") or result_type == "figure")
        is_table = bool(result.get("table_id") or result_type == "table")
        is_coverage_page = bool(
            (result.get("coverage_mode") or _is_visual_page_result(result))
            and result_type in {"page", "visual_page"}
            and result.get("page_number") is not None
        )
        is_artifact = is_figure or is_table
        is_required_reservation = id(result) in reserved_result_ids
        # Page position is not artifact identity: separate crops/figures on a
        # page can contain independent evidence, and a crop does not cover the
        # whole page. Only duplicate canonical units are suppressed. Mark a
        # unit as seen after it actually survives content/budget selection.
        if evidence_id and evidence_id in seen_chunks:
            record_selection(result, index, "duplicate_canonical_identity")
            continue
        if (
            not is_required_reservation
            and not is_artifact
            and document_id
            and not (coverage_mode and is_coverage_page)
            and per_document_count[document_id] >= max_chunks_per_document
        ):
            record_selection(result, index, "per_document_limit")
            continue

        raw_content = str(result.get("expanded_content") or result.get("content") or result.get("text") or "")
        content = _clean_content(raw_content, preserve_lines=is_table)
        if not content:
            record_selection(result, index, "empty_content")
            continue
        cleaned_char_count = len(content)
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
            record_selection(result, index, "character_budget")
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
        record_selection(result, index, "selected", input_chars=cleaned_char_count,
                         output_chars=len(content), projected=len(content) < cleaned_char_count)
        used_chars += len(block)
        if evidence_id:
            seen_chunks.add(evidence_id)
        if document_id and not is_artifact:
            per_document_count[document_id] += 1
        if len(sources) >= effective_max_sources:
            # These candidates were not examined, so do not pretend to know
            # whether they were duplicates, empty, or otherwise irrelevant.
            for remaining_index in range(index + 1, len(ordered_results)):
                record_selection(ordered_results[remaining_index], remaining_index, "source_limit_unexamined")
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
            "selection_trace": {
                "schema_version": "context-selection-v1", "input_count": len(ordered_results),
                "counts": dict(selection_counts), "events": selection_events,
                "omitted_events": len(ordered_results) - len(selection_events),
                "complete": len(selection_events) == len(ordered_results),
                "boundary": "composer_input_to_prompt_sources",
            },
            "source_count": len(sources),
            "character_count": len(context_text),
            "document_count": len({source.get("document_id") for source in sources if source.get("document_id")}),
            "table_source_count": sum(
                1 for source in sources if source.get("table_id") or source.get("artifact_type") == "table"
            ),
            "figure_source_count": sum(
                1 for source in sources if source.get("figure_id") or source.get("artifact_type") == "figure"
            ),
            "page_source_count": sum(
                1
                for source in sources
                if source.get("coverage_mode") and source.get("page_number") is not None
            ),
            "visual_page_source_count": sum(
                1
                for source in sources
                if _is_visual_page_result(source) and source.get("page_number") is not None
            ),
            "coverage_mode": bool(coverage_mode),
            "required_facet_coverage": _required_facet_coverage(
                sources,
                effective_facet_requirements,
            ),
        },
    )


def _is_visual_page_result(result: dict[str, Any]) -> bool:
    """Return whether a source carries a selected page-image provenance.

    Visual retrieval providers use ``artifact_type=page`` because a page is
    the canonical multimodal evidence unit.  The provider-specific channel or
    explicit marker is therefore the stable discriminator; callers must not
    infer visual evidence merely from OCR text or a page number.
    """

    channels = result.get("retrieval_channels")
    if isinstance(channels, str):
        channels = [channels]
    artifact_type = str(
        result.get("artifact_type") or result.get("chunk_type") or ""
    ).casefold()
    return bool(
        result.get("visual_page")
        or "visual_page" in {str(value).strip() for value in (channels or [])}
        or artifact_type == "visual_page"
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
    # Production callers often know the focused documents but do not build a
    # separate facet map.  Derive the same bounded semantic contract used by
    # ``select_ranked_evidence`` so context composition cannot silently spend
    # all slots on one high-scoring section.  Explicit requirements always win.
    if ordered_ids and not facet_requirements and query:
        requested = requested_paper_facets(query, focused_document_count=len(ordered_ids))
        facet_requirements = {document_id: list(requested) for document_id in ordered_ids}
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
            ranked_candidates: list[tuple[tuple, dict[str, Any]]] = []
            for index, candidate in enumerate(candidates):
                if id(candidate) in reserved_result_ids:
                    continue
                candidate_facets = _source_evidence_facets(candidate)
                covered_now = uncovered.intersection(candidate_facets)
                if not covered_now:
                    continue
                artifact_fit = _facet_artifact_fit(candidate, covered_now)
                table_match = _table_query_match_score(
                    candidate,
                    query=query or "",
                )
                ranked_candidates.append(
                    (
                        (len(covered_now), artifact_fit, *table_match, -index),
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
    # Keep context composition and evaluator/final selection on one facet
    # contract.  Older sources may have no persisted facet array; the shared
    # classifier can recover conservative structural hints from artifact,
    # section and content metadata without consulting evaluation labels.
    return _result_facets(source)


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
    min_visual_pages: int,
    query: str | None,
) -> list[dict[str, Any]]:
    if (min_figures <= 0 and min_tables <= 0 and min_visual_pages <= 0) or not results:
        return _promote_exact_query_matches(results, query=query)

    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    visual_pages: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    seen_figure_ids: set[str] = set()
    seen_table_ids: set[str] = set()
    seen_visual_pages: set[tuple[str, int]] = set()
    for result in results:
        figure_id = str(result.get("figure_id") or "")
        table_id = str(result.get("table_id") or "")
        is_visual_page = _is_visual_page_result(result)
        if figure_id and min_figures > 0:
            if figure_id in seen_figure_ids:
                continue
            seen_figure_ids.add(figure_id)
            figures.append(result)
        elif is_visual_page and min_visual_pages > 0:
            # A visual worker can return the same page through multiple
            # projection rows (for example a page hit merged with an image
            # artifact). They carry different runtime chunk IDs but represent
            # one multimodal evidence unit; keeping both can evict the exact
            # text chunk that explains the page. Deduplicate by canonical page
            # identity before the bounded source budget is applied.
            document_id = str(result.get("document_id") or "").strip()
            raw_page = result.get("page_number")
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                page_number = None
            visual_key = (document_id, page_number) if document_id and page_number else None
            if visual_key is not None and visual_key in seen_visual_pages:
                continue
            if visual_key is not None:
                seen_visual_pages.add(visual_key)
            visual_pages.append(result)
        elif min_tables > 0 and (table_id or result.get("artifact_type") == "table"):
            identity = table_id or str(result.get("chunk_id") or result.get("id") or "")
            if identity and identity in seen_table_ids:
                continue
            if identity:
                seen_table_ids.add(identity)
            tables.append(result)
        else:
            others.append(result)

    if not figures and not tables and not visual_pages:
        return _promote_exact_query_matches(results, query=query)

    if min_tables > 0 and query:
        # Lance similarity is still the primary candidate generator.  Within
        # its table candidates, prefer the artifact that covers more of the
        # explicitly requested metrics/terms before reserving the bounded
        # context slot.
        tables.sort(
            key=lambda item: _table_query_match_score(item, query=query),
            reverse=True,
        )
    if min_figures > 0 and query:
        # Match the explicitly requested figure label before applying the
        # visual-intent ranking.  A page/figure retriever may return several
        # visually similar figures; reserving the first one by vector score
        # can silently discard ``Figure N`` even when that exact artifact is
        # already present in the candidate frontier.
        figures.sort(
            key=lambda item: (
                _artifact_query_match(item, kind="figure", query=query),
                _selection_score(item),
            ),
            reverse=True,
        )

    # Keep the explicit artifact reservations first, but apply a bounded
    # lexical-preservation pass to the ordinary text frontier.  A semantic
    # hit can be ranked highly while the exact row/paragraph containing a
    # rare acronym, number or identifier sits just beyond the prompt budget.
    # This pass does not replace retrieval ranking or invent evidence: it only
    # reorders already retrieved candidates when at least two content-bearing
    # query anchors occur in the candidate.  It is therefore useful for
    # arbitrary manuals, reports and papers alike (e.g. ``UL`` + ``CSA``), not
    # a paper/case-specific rule.
    prioritized = (
        tables[:min_tables]
        + figures[:min_figures]
        + visual_pages[:min_visual_pages]
        + _promote_exact_query_matches(others, query=query)
    )
    seen_chunks: set[str] = set()
    seen_visual_source_pages: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for result in prioritized:
        artifact_type = canonical_artifact_type(result)
        is_visual_source = bool(
            _is_visual_page_result(result)
            or result.get("figure_id")
            or artifact_type == "figure"
        )
        if is_visual_source:
            document_id = str(result.get("document_id") or "").strip()
            raw_page = result.get("page_number")
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                page_number = None
            page_key = (
                (document_id, page_number)
                if document_id and page_number is not None and has_physical_pages(result)
                else None
            )
            if page_key is not None and page_key in seen_visual_source_pages:
                continue
            if page_key is not None:
                seen_visual_source_pages.add(page_key)
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


_QUERY_MATCH_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
        "did", "does", "for", "from", "how", "in", "is", "it", "of", "on", "or",
        "that", "the", "their", "this", "to", "was", "what", "when", "which",
        "with", "who", "why", "where", "about", "answer", "based", "document",
        "file", "paper", "users", "user", "second", "first", "bullet", "point",
        "number", "value", "values", "much", "please", "give", "tell", "show",
    }
)

def _query_anchor_tokens(query: str | None) -> tuple[str, ...]:
    """Return conservative content-bearing anchors for context projection.

    This is intentionally not a semantic rewrite.  Preserve short technical
    tokens (``UL``, ``CSA``, ``F1``), numbers and hyphenated identifiers while
    ignoring question boilerplate.  The original spelling is retained for
    special-token detection, but matching is case-insensitive.
    """

    if not query:
        return ()
    raw_tokens = re.findall(
        r"[A-Za-zÀ-ỹ0-9]+(?:[&][A-Za-zÀ-ỹ0-9]+|[-_./][A-Za-zÀ-ỹ0-9]+)*",
        str(query),
        flags=re.UNICODE,
    )
    anchors: list[str] = []
    seen: set[str] = set()
    for raw in raw_tokens:
        normalized = raw.casefold()
        if normalized in _QUERY_MATCH_STOPWORDS:
            continue
        compact = normalized.strip("._-/")
        if not compact or compact in seen:
            continue
        # Keep ordinary words at length >= 3 and all compact technical tokens
        # that carry digits, punctuation or an uppercase/acronym signal.
        technical = (
            len(compact) >= 2
            and (
                any(char.isdigit() for char in compact)
                or any(mark in compact for mark in ("&", "-", "_", "/", "."))
                or (raw.isupper() and raw.isascii())
            )
        )
        if len(compact) >= 3 or technical:
            anchors.append(compact)
            seen.add(compact)
    return tuple(anchors)


def _query_match_signature(
    result: dict[str, Any],
    query: str | None,
    *,
    anchors: tuple[str, ...] | None = None,
) -> tuple[int, int, int]:
    """Score exact anchor coverage for bounded context ordering.

    The tuple is (anchor count, special-token count, phrase count).  It is
    deliberately small and deterministic; retrieval score remains the tie
    breaker through the original result order.
    """

    effective_anchors = anchors if anchors is not None else _query_anchor_tokens(query)
    if not effective_anchors:
        return (0, 0, 0)
    searchable = " ".join(
        str(result.get(field) or "")
        for field in (
            "section_title", "caption", "heading_path", "content",
            "text", "expanded_content",
        )
    ).casefold()
    matched = [
        anchor
        for anchor in effective_anchors
        if re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", searchable)
    ]
    special = [
        anchor
        for anchor in matched
        if len(anchor) <= 6
        and (
            any(char.isdigit() for char in anchor)
            or any(mark in anchor for mark in ("&", "-", "_", "/", "."))
            or (anchor.isascii() and len(anchor) <= 4)
        )
    ]
    phrase_count = 0
    for left, right in zip(effective_anchors, effective_anchors[1:]):
        if left in searchable and right in searchable and re.search(
            rf"{re.escape(left)}(?:\W+\w+){{0,2}}\W+{re.escape(right)}",
            searchable,
        ):
            phrase_count += 1
    return (len(matched), len(special), phrase_count)


def _promote_exact_query_matches(
    results: list[dict[str, Any]], *, query: str | None
) -> list[dict[str, Any]]:
    """Stable promotion of already-retrieved exact lexical evidence.

    Require two matched anchors, or one special technical anchor, before
    changing order.  That conservative threshold prevents a common word such
    as ``model`` or ``safety`` from displacing a strong semantic result.
    """

    if not results or not _query_anchor_tokens(query):
        return results
    # Multi-evidence retrieval may have already marked one candidate for each
    # query-derived obligation (ratio numerator/denominator, comparison
    # operands, or explicit multi-hop clauses). Preserve the first candidate
    # for every distinct obligation before applying lexical promotion; without
    # this, a high-overlap narrative chunk can evict the complementary value
    # from a bounded prompt even though retrieval found it correctly.
    obligation_prefix: list[dict[str, Any]] = []
    obligation_ids: set[str] = set()
    for result in results:
        ids = result.get("evidence_obligation_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        new_ids = [
            str(value).strip()
            for value in ids
            if str(value).strip() and str(value).strip() not in obligation_ids
        ]
        if not new_ids:
            continue
        obligation_prefix.append(result)
        obligation_ids.update(new_ids)
    if obligation_prefix:
        remainder = [result for result in results if result not in obligation_prefix]
        promoted_remainder = _promote_exact_query_matches(remainder, query=query)
        return [*obligation_prefix, *promoted_remainder]
    anchors = _query_anchor_tokens(query)
    # Terms repeated on most candidates (often a document title/header) do
    # not identify the value-bearing passage. Exclude those terms from the
    # promotion signature while retaining rare acronyms, numbers and labels.
    searchable_by_result = [
        " ".join(
            str(result.get(field) or "")
            for field in (
                "section_title", "caption", "heading_path", "content",
                "text", "expanded_content",
            )
        ).casefold()
        for result in results
    ]
    max_common_frequency = max(2, len(results) // 2)
    discriminative = tuple(
        anchor
        for anchor in anchors
        if sum(
            bool(re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", searchable))
            for searchable in searchable_by_result
        ) <= max_common_frequency
    )
    if not discriminative:
        return results
    decorated: list[tuple[int, tuple[int, int, int], int, dict[str, Any]]] = []
    for index, result in enumerate(results):
        signature = _query_match_signature(result, query, anchors=discriminative)
        eligible = int(signature[0] >= 2 or signature[1] >= 1)
        decorated.append((eligible, signature, -index, result))
    return [item[3] for item in sorted(decorated, key=lambda item: item[:3], reverse=True)]


def _table_query_match_score(result: dict[str, Any], *, query: str) -> tuple[int, int, int, int, float]:
    tokens = _table_query_tokens(query)
    searchable = "\n".join(
        str(result.get(field) or "")
        for field in ("caption", "content", "text")
    ).lower()
    matched = sum(1 for token in tokens if token in searchable)
    rank_score = float(result.get("rerank_score") or result.get("score") or 0.0)
    if not (result.get("table_id") or canonical_artifact_type(result) == "table"):
        return 0, 0, 0, matched, rank_score
    metric_patterns = (
        r"(?<!\w)(?:acc(?:uracy)?)(?!\w)", r"(?<!\w)(?:(?:micro|macro)[ -]?)?f1(?!\w)",
        r"(?<!\w)ccc(?:[ _-]?[avd])?(?!\w)", r"(?<!\w)ua(?:r)?(?!\w)",
        r"(?<!\w)wa(?:r)?(?!\w)", r"(?<!\w)(?:wer|cer)(?!\w)",
        r"(?<!\w)bleu(?!\w)", r"(?<!\w)rouge(?:[ _-]?[12l])?(?!\w)",
        r"(?<!\w)(?:rmse|mae|mse)(?!\w)", r"(?<!\w)precision(?!\w)",
        r"(?<!\w)recall(?!\w)", r"(?<!\w)auc(?!\w)",
    )
    metric_matches = sum(
        bool(re.search(pattern, query, re.I) and re.search(pattern, searchable, re.I))
        for pattern in metric_patterns
    )
    facets = set(extract_query_facets(query))
    role = table_result_role(result)
    purpose = 0
    if "ablation" in facets:
        purpose = int(role == "ablation")
    elif "benchmark_results" in facets:
        purpose = {"main": 2, "unknown": 0, "ablation": -1, "statistics": -2}[role]
    return _artifact_query_match(result, "table", query), metric_matches, purpose, matched, rank_score


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
    query_tokens = _table_query_tokens(query)
    # Select complete row bundles, never naked rows. A single canonical table
    # can contain repeated headers with different metrics, grouped datasets,
    # or several Markdown blocks. Flattening all rows under the first header
    # manufactures a metric/condition assignment not present in the source.
    units: list[tuple[int, tuple[int, ...], int, int]] = []
    schema: tuple[int, ...] = ()
    group: tuple[int, ...] = ()
    header_cells: list[str] = []
    contiguous = False
    pending_headings: list[int] = []
    for index in range(start, len(lines)):
        line = lines[index]
        cells = _projection_table_cells(line)
        if len(cells) < 2:
            contiguous = False
            if line.lstrip().startswith("#"):
                pending_headings.append(index)
            continue
        if _projection_separator(cells):
            continue
        next_cells = _projection_table_cells(lines[index + 1]) if index + 1 < len(lines) else []
        repeated_header = bool(
            header_cells and cells[0].casefold() == header_cells[0].casefold()
            and len([cell for cell in cells if cell]) >= 2
            and all(not re.fullmatch(r"[≈~<>+\-\s]*\d[\d.,%\s]*", cell) for cell in cells if cell)
        )
        if not contiguous or _projection_separator(next_cells) or repeated_header:
            schema = (*pending_headings, index)
            if _projection_separator(next_cells):
                schema = (*schema, index + 1)
            header_cells = cells
            group = ()
            pending_headings = []
            contiguous = True
            continue
        if cells[0] and not any(cells[1:]):
            # An explicit full-width label owns subsequent rows until the
            # next such label/header. Never move its rows to a different group.
            group = (index,)
            continue
        bundle = (*schema, *group, index)
        row_match = sum(token in line.casefold() for token in query_tokens)
        header_match = sum(token in " ".join(lines[i] for i in (*schema, *group)).casefold()
                           for token in query_tokens)
        proposed = int(bool(_PROPOSED_TABLE_ROW_RE.search(_fold_ascii(line))))
        units.append((index, bundle, row_match * 2 + header_match, proposed))
    if not units:
        return _truncate(content, max_chars)

    prefix_ids = set(range(start))
    selected_ids = set(prefix_ids)
    ranked = sorted(units, key=lambda unit: (-unit[2], -unit[3], unit[0]))

    def render(indices: set[int]) -> str:
        return "\n".join(lines[index] for index in sorted(indices))

    for _index, bundle, _match, _proposed in ranked:
        candidate_ids = selected_ids.union(bundle)
        if len(render(candidate_ids)) <= max_chars:
            selected_ids = candidate_ids
        elif selected_ids == prefix_ids:
            # Existing per-source budget is soft; preserve one complete
            # schema/condition/row rather than emitting a broken partial row.
            selected_ids = candidate_ids
    projected = render(selected_ids)
    if any(index not in selected_ids for index, *_rest in units):
        projected += "\n<!-- additional table rows omitted after evidence-aware projection -->"
    return projected


def _projection_table_cells(line: str) -> list[str]:
    if "|" not in line:
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _projection_separator(cells: list[str]) -> bool:
    return len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


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
