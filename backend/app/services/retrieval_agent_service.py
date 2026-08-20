from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.rag.paper_facets import (
    FACET_MARKERS as _FACET_MARKERS,
    FACET_QUERY_TERMS as _FACET_QUERY_TERMS,
    extract_query_facets,
    normalize_facets as _shared_normalize_facets,
)
from app.services.evidence_validator import EvidenceValidationResult
from app.services.query_rewrite_service import enrich_retrieval_query, has_visual_intent


MAX_ADAPTIVE_SUBQUERIES = 3
MAX_RETRIEVAL_HOPS = 2


@dataclass(frozen=True)
class RetrievalGapAssessment:
    needs_second_pass: bool
    reason: str
    missing_document_ids: list[str]
    missing_figure_document_ids: list[str]


@dataclass(frozen=True)
class RetrievalFacetCoverage:
    requested_facets: list[str]
    covered_facets: list[str]
    missing_facets: list[str]
    coverage_observed: bool


@dataclass(frozen=True)
class RetrievalBranch:
    query: str
    focus_document_ids: list[str]
    reason: str
    hop: int = 1
    facets: list[str] = field(default_factory=list)
    bridge_anchors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SecondRetrievalPlan:
    """A bounded second-hop plan.

    ``query`` remains the compatibility field used by the existing single-query
    caller. New callers should execute ``branches`` so every query keeps its
    canonical document scope. The planner never emits more than three branches
    and never plans beyond hop two.
    """

    query: str
    reasons: list[str]
    queries: list[str] = field(default_factory=list)
    branches: list[RetrievalBranch] = field(default_factory=list)
    hop_count: int = MAX_RETRIEVAL_HOPS
    missing_facets: list[str] = field(default_factory=list)
    bridge_anchors: list[str] = field(default_factory=list)


_BROAD_QUERY_MARKERS = (
    "across papers",
    "across documents",
    "all papers",
    "literature survey",
    "survey",
    "nhiều paper",
    "cac paper",
    "các paper",
    "toàn bộ paper",
    "tong hop",
    "tổng hợp",
    "so sánh",
    "compare",
)

_GRAPH_CHANNELS = frozenset({"lightrag_entity", "lightrag_relation", "graph_bridge"})
_BRIDGE_ANCHOR_KEYS = (
    "anchor",
    "anchors",
    "entity_name",
    "entity_names",
    "source_entity",
    "target_entity",
    "src_id",
    "tgt_id",
    "keywords",
    "relation_keywords",
)
_GENERIC_BRIDGE_ANCHORS = frozenset(
    {
        "entity",
        "figure",
        "graph",
        "ieee",
        "journal",
        "model",
        "paper",
        "relation",
        "table",
    }
)
_STRUCTURAL_ANCHOR_RE = re.compile(
    r"^(?:fig(?:ure)?\.?|table)\s*[\divxlcdm]+[a-z]?$",
    flags=re.IGNORECASE,
)


def plan_retrieval_decomposition(
    *,
    query: str,
    answer_intent: str,
    focus_document_ids: list[str],
    enabled: bool,
    must_cover_all: bool = False,
) -> list[RetrievalBranch]:
    """Split broad comparisons into independently scoped retrieval branches.

    Per-document branches guarantee evidence coverage without spending another
    LLM call on decomposition.  They are merged only after every engine has
    applied the same canonical focus filter.
    """

    unique_focus_ids = _normalize_document_ids(focus_document_ids)
    if (
        not enabled
        or (answer_intent != "compare" and not must_cover_all)
        or len(unique_focus_ids) < 2
        or len(unique_focus_ids) > MAX_ADAPTIVE_SUBQUERIES
    ):
        return []
    return [
        RetrievalBranch(
            query=query,
            focus_document_ids=[str(document_id)],
            reason="compare_per_document",
        )
        for document_id in unique_focus_ids
    ]


def smart_retrieval_enabled(
    agent_reasoning: str,
    *,
    answer_intent: str,
    query: str,
    focus_document_ids: list[str],
) -> bool:
    mode = (agent_reasoning or "auto").lower()
    if mode == "fast":
        return False
    if mode == "smart":
        return True
    if answer_intent in {"compare", "infer_structure"}:
        return True
    if has_visual_intent(query):
        return True
    # A single-paper request can still need evidence from several independent
    # sections (for example architecture + training + benchmark results).
    # Planning remains deterministic and a second hop is only emitted when the
    # first-hop structured metadata proves a coverage gap.
    if len(extract_query_facets(query)) >= 2:
        return True
    return len(focus_document_ids) >= 2


def assess_facet_coverage(
    *,
    query: str,
    documents: list[dict],
    focus_document_ids: list[str] | None = None,
    requested_facets: list[str] | None = None,
    covered_facets: list[str] | None = None,
) -> RetrievalFacetCoverage:
    """Compare requested facets with explicit retrieval metadata.

    Raw chunk text is intentionally not classified here: arbitrary prose is not
    a durable evidence-coverage signal.
    """

    requested = _normalize_facets(
        requested_facets if requested_facets is not None else extract_query_facets(query)
    )
    observed = covered_facets is not None
    covered = _normalize_facets(covered_facets or [])
    allowed_ids = _normalize_document_ids(focus_document_ids)

    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if allowed_ids and document_id not in allowed_ids:
            continue
        structured_facets, structured_observed = _structured_document_facets(document)
        if structured_observed:
            observed = True
            covered.extend(structured_facets)
        for container in _facet_metadata_containers(document):
            for key in ("covered_facets", "evidence_facets", "retrieval_facets", "facets"):
                if key not in container:
                    continue
                observed = True
                covered.extend(_normalize_facets(_as_string_list(container.get(key))))

    covered = _dedupe_strings(covered)
    covered_set = set(covered)
    missing = [facet for facet in requested if facet not in covered_set] if observed else []
    return RetrievalFacetCoverage(
        requested_facets=requested,
        covered_facets=covered,
        missing_facets=missing,
        coverage_observed=observed,
    )


def extract_graph_bridge_anchors(
    *,
    documents: list[dict],
    graph_bridge_metadata: list[dict[str, Any]] | None = None,
    focus_document_ids: list[str] | None = None,
    unresolved_only: bool = False,
) -> list[str]:
    """Read graph anchors from explicit metadata fields, never source prose."""

    allowed_ids = _normalize_document_ids(focus_document_ids)
    anchors: list[str] = []
    for record, inherited_document_id in _iter_graph_bridge_records(
        documents,
        graph_bridge_metadata or [],
    ):
        document_id = str(record.get("document_id") or inherited_document_id or "").strip()
        # Focused planning rejects graph metadata without canonical provenance.
        if allowed_ids and document_id not in allowed_ids:
            continue
        if unresolved_only and not _bridge_record_requires_followup(record):
            continue
        for key in _BRIDGE_ANCHOR_KEYS:
            for candidate in _as_anchor_list(record.get(key)):
                if _useful_bridge_anchor(candidate):
                    anchors.append(candidate)
    return _dedupe_strings(anchors)


def assess_retrieval_gaps(
    *,
    documents: list[dict],
    answer_intent: str,
    focus_document_ids: list[str],
    query: str,
    must_cover_all: bool = False,
) -> RetrievalGapAssessment:
    text_document_ids = {
        str(document.get("document_id"))
        for document in documents
        if document.get("document_id") and not document.get("figure_id")
    }
    figure_document_ids = {
        str(document.get("document_id"))
        for document in documents
        if document.get("document_id") and document.get("figure_id")
    }

    focus_ids = _normalize_document_ids(focus_document_ids)
    if not focus_ids:
        if (answer_intent == "compare" or must_cover_all) and len(text_document_ids) < 2:
            return RetrievalGapAssessment(
                needs_second_pass=True,
                reason="insufficient_compare_document_coverage",
                missing_document_ids=[],
                missing_figure_document_ids=[],
            )
        return RetrievalGapAssessment(False, "no_focus", [], [])

    missing_text = [document_id for document_id in focus_ids if document_id not in text_document_ids]

    if (answer_intent == "compare" or must_cover_all) and len(focus_ids) >= 2 and missing_text:
        return RetrievalGapAssessment(
            needs_second_pass=True,
            reason="missing_compare_document_text",
            missing_document_ids=missing_text,
            missing_figure_document_ids=[],
        )

    if has_visual_intent(query):
        missing_figures = [document_id for document_id in focus_ids if document_id not in figure_document_ids]
        if missing_figures:
            return RetrievalGapAssessment(
                needs_second_pass=True,
                reason="missing_focus_figures",
                missing_document_ids=[],
                missing_figure_document_ids=missing_figures,
            )

    return RetrievalGapAssessment(False, "sufficient", [], [])


def plan_second_retrieval_pass(
    *,
    retrieval_query: str,
    original_task: str,
    topic: str | None,
    entities: list[str],
    answer_intent: str,
    focus_document_ids: list[str],
    validation: EvidenceValidationResult,
    smart_allowed: bool,
    documents: list[dict],
    graph_bridge_metadata: list[dict[str, Any]] | None = None,
    requested_facets: list[str] | None = None,
    covered_facets: list[str] | None = None,
    completed_hops: int = 1,
    previous_queries: list[str] | None = None,
    must_cover_all: bool = False,
    retry_budget_available: bool = True,
) -> SecondRetrievalPlan | None:
    if not retry_budget_available or completed_hops >= MAX_RETRIEVAL_HOPS:
        return None

    focus_ids = _normalize_document_ids(focus_document_ids)
    gap = assess_retrieval_gaps(
        documents=documents,
        answer_intent=answer_intent,
        focus_document_ids=focus_ids,
        query=original_task,
        must_cover_all=must_cover_all,
    )

    # Fast mode normally stays one-hop.  The exception is an atomic scope
    # obligation: when the user named several documents, returning evidence for
    # only a subset is incorrect rather than merely less thorough.  Keep that
    # repair deterministic, document-scoped, and bounded by MAX_RETRIEVAL_HOPS.
    mandatory_scope_retry = bool(
        must_cover_all and focus_ids and gap.needs_second_pass
    )
    if not smart_allowed and not mandatory_scope_retry:
        return None

    reasons: list[str] = []
    facet_coverage = assess_facet_coverage(
        query=original_task,
        documents=documents,
        focus_document_ids=focus_ids,
        requested_facets=requested_facets,
        covered_facets=covered_facets,
    )
    bridge_anchors = extract_graph_bridge_anchors(
        documents=documents,
        graph_bridge_metadata=graph_bridge_metadata,
        focus_document_ids=focus_ids,
    )
    unresolved_bridge_anchors = extract_graph_bridge_anchors(
        documents=documents,
        graph_bridge_metadata=graph_bridge_metadata,
        focus_document_ids=focus_ids,
        unresolved_only=True,
    )

    # In fast mode, do not broaden the retry because of weak optional facets,
    # graph hints, or generic validation.  Only repair the mandatory scope gap.
    if smart_allowed and validation.retry_required:
        reasons.append(validation.reason)
    if gap.needs_second_pass:
        reasons.append(gap.reason)
    if smart_allowed and facet_coverage.missing_facets:
        reasons.append("missing_query_facets")
    if smart_allowed and unresolved_bridge_anchors:
        reasons.append("unresolved_graph_bridge")
    if (
        smart_allowed
        and not documents
        and not focus_ids
        and _is_broad_multidocument_query(answer_intent, original_task)
    ):
        reasons.append("no_broad_evidence")

    reasons = _dedupe_strings(reasons)

    if not reasons:
        return None

    # Never turn ordinary unscoped chat into a corpus-wide retry. The only
    # unscoped exception is an explicitly broad/compare task with a concrete
    # coverage failure recorded above.
    if not focus_ids and not _is_broad_multidocument_query(answer_intent, original_task):
        return None

    branches = _build_second_hop_branches(
        retrieval_query=retrieval_query,
        original_task=original_task,
        topic=topic,
        entities=entities,
        answer_intent=answer_intent,
        focus_document_ids=focus_ids,
        missing_entities=validation.missing_entities,
        gap=gap,
        reasons=reasons,
        missing_facets=facet_coverage.missing_facets if smart_allowed else [],
        bridge_anchors=bridge_anchors if smart_allowed else [],
        previous_queries=previous_queries or [],
    )
    if not branches:
        return None

    queries = [branch.query for branch in branches]
    return SecondRetrievalPlan(
        query=queries[0],
        reasons=reasons,
        queries=queries,
        branches=branches,
        hop_count=min(MAX_RETRIEVAL_HOPS, max(1, completed_hops) + 1),
        missing_facets=facet_coverage.missing_facets if smart_allowed else [],
        bridge_anchors=bridge_anchors if smart_allowed else [],
    )


def build_refined_retrieval_query(
    retrieval_query: str,
    *,
    original_task: str,
    topic: str | None,
    entities: list[str],
    answer_intent: str,
    focus_document_ids: list[str],
    missing_entities: list[str],
    gap: RetrievalGapAssessment,
    reasons: list[str],
    missing_facets: list[str] | None = None,
    bridge_anchors: list[str] | None = None,
) -> str:
    anchors: list[str] = []

    if validation_reasons_contain(reasons, "missing_required_entities", "no_documents"):
        anchors.extend(missing_entities)

    for facet in missing_facets or []:
        anchors.append(_FACET_QUERY_TERMS.get(facet, facet))

    if "missing_compare_document_text" in reasons:
        anchors.extend(["overview", "method", "evidence"])

    if "insufficient_compare_document_coverage" in reasons:
        anchors.extend(["comparison", "method", "results"])

    if (
        gap.missing_figure_document_ids
        or "missing_focus_figures" in reasons
        or has_visual_intent(original_task)
    ):
        anchors.extend(
            [
                "figure",
                "diagram",
                "caption",
            ]
        )

    if validation_reasons_contain(
        reasons,
        "no_documents",
        "focus_document_mismatch",
        "mixed_focus_documents",
        "no_broad_evidence",
    ):
        anchors.extend(["evidence", "source passage"])

    anchors.extend(bridge_anchors or [])

    augmented = retrieval_query
    for anchor in anchors:
        token = anchor.strip()
        if token and token.lower() not in augmented.lower():
            augmented = f"{token} {augmented}"

    return enrich_retrieval_query(
        augmented,
        topic=topic,
        entities=[*entities, *missing_entities],
        answer_intent=answer_intent,
        focus_document_ids=focus_document_ids,
    )


def _build_second_hop_branches(
    *,
    retrieval_query: str,
    original_task: str,
    topic: str | None,
    entities: list[str],
    answer_intent: str,
    focus_document_ids: list[str],
    missing_entities: list[str],
    gap: RetrievalGapAssessment,
    reasons: list[str],
    missing_facets: list[str],
    bridge_anchors: list[str],
    previous_queries: list[str],
) -> list[RetrievalBranch]:
    if gap.missing_document_ids:
        scopes = [[document_id] for document_id in gap.missing_document_ids]
    elif answer_intent == "compare" and 1 < len(focus_document_ids) <= MAX_ADAPTIVE_SUBQUERIES:
        scopes = [[document_id] for document_id in focus_document_ids]
    else:
        # More than three focused papers stay in one canonical allow-list so the
        # hard subquery cap cannot silently omit a paper.
        scopes = [list(focus_document_ids)]

    if len(scopes) > 1:
        candidate_specs = [
            (scope, list(missing_facets[:MAX_ADAPTIVE_SUBQUERIES]))
            for scope in scopes[:MAX_ADAPTIVE_SUBQUERIES]
        ]
    elif missing_facets:
        candidate_specs = [
            (scopes[0], [facet])
            for facet in missing_facets[:MAX_ADAPTIVE_SUBQUERIES]
        ]
    else:
        candidate_specs = [(scopes[0], [])]

    prior_queries = {
        _normalize_query(value)
        for value in [retrieval_query, *previous_queries]
        if _normalize_query(value)
    }
    seen_branches: set[tuple[str, tuple[str, ...]]] = set()
    branches: list[RetrievalBranch] = []
    for scope, facets in candidate_specs:
        refined_query = build_refined_retrieval_query(
            retrieval_query,
            original_task=original_task,
            topic=topic,
            entities=entities,
            answer_intent=answer_intent,
            focus_document_ids=scope,
            missing_entities=missing_entities,
            gap=gap,
            reasons=reasons,
            missing_facets=facets,
            bridge_anchors=bridge_anchors[:MAX_ADAPTIVE_SUBQUERIES],
        )
        normalized_query = _normalize_query(refined_query)
        branch_key = (normalized_query, tuple(scope))
        if not normalized_query or normalized_query in prior_queries or branch_key in seen_branches:
            continue
        seen_branches.add(branch_key)
        branches.append(
            RetrievalBranch(
                query=refined_query,
                focus_document_ids=list(scope),
                reason=f"adaptive_second_hop:{reasons[0]}",
                hop=MAX_RETRIEVAL_HOPS,
                facets=list(facets),
                bridge_anchors=list(bridge_anchors[:MAX_ADAPTIVE_SUBQUERIES]),
            )
        )
        if len(branches) >= MAX_ADAPTIVE_SUBQUERIES:
            break
    return branches


def _is_broad_multidocument_query(answer_intent: str, query: str) -> bool:
    if (answer_intent or "").lower() in {
        "compare",
        "research",
        "survey",
        "synthesize",
        "broad",
    }:
        return True
    lowered = " ".join((query or "").lower().split())
    return any(marker in lowered for marker in _BROAD_QUERY_MARKERS)


def _normalize_document_ids(document_ids: list[str] | None) -> list[str]:
    return _dedupe_strings(
        str(document_id).strip()
        for document_id in document_ids or []
        if str(document_id).strip()
    )


def _normalize_facets(facets: list[str]) -> list[str]:
    return _shared_normalize_facets(facets)


def _facet_metadata_containers(document: dict) -> list[dict]:
    containers = [document]
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata)
    return containers


def _structured_document_facets(document: dict) -> tuple[list[str], bool]:
    """Infer coverage from retrieval structure, never arbitrary passage text."""

    facets: list[str] = []
    observed = False
    for container in _facet_metadata_containers(document):
        for key in (
            "section_title",
            "heading_path",
            "heading",
            "section",
            "figure_type",
            "table_type",
        ):
            values = _as_string_list(container.get(key))
            if not values:
                continue
            observed = True
            for value in values:
                facets.extend(_structured_label_facets(value))

        artifact_type = str(
            container.get("artifact_type") or container.get("chunk_type") or ""
        ).strip().lower()
        if artifact_type in {"figure", "image", "visual_page", "page"}:
            observed = True
            facets.append("visual_evidence")
        elif artifact_type == "table":
            observed = True
            facets.append("benchmark_results")

        if container.get("figure_id"):
            observed = True
            facets.append("visual_evidence")
        if container.get("table_id"):
            observed = True
            facets.append("benchmark_results")

    return _dedupe_strings(facets), observed


def _structured_label_facets(value: str) -> list[str]:
    label = " ".join(str(value or "").lower().replace("_", " ").split())
    facets = extract_query_facets(label)
    structural_markers = {
        "architecture": ("system design", "model structure", "proposed model"),
        "training_method": (
            "method",
            "methods",
            "approach",
            "training",
            "optimization",
            "objective",
            "loss",
        ),
        "benchmark_results": (
            "experiment",
            "experiments",
            "experimental results",
            "evaluation",
            "discussion",
        ),
        "dataset_setup": ("data", "corpus", "setup"),
    }
    for facet, markers in structural_markers.items():
        if any(marker in label for marker in markers):
            facets.append(facet)
    return _dedupe_strings(facets)


def _iter_graph_bridge_records(
    documents: list[dict],
    graph_bridge_metadata: list[dict[str, Any]],
):
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        channels = set(_as_string_list(document.get("retrieval_channels")))
        if channels & _GRAPH_CHANNELS:
            yield document, document_id

        metadata = document.get("metadata")
        if isinstance(metadata, dict):
            for key in ("graph_bridge", "graph_bridge_metadata"):
                yield from _iter_nested_bridge_records(metadata.get(key), document_id)

    for record in graph_bridge_metadata:
        if not isinstance(record, dict):
            continue
        document_id = str(record.get("document_id") or "").strip()
        yield record, document_id
        for key in ("graph_bridge", "graph_bridge_metadata"):
            yield from _iter_nested_bridge_records(record.get(key), document_id)


def _iter_nested_bridge_records(value: Any, document_id: str):
    if isinstance(value, dict):
        yield value, document_id
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                yield item, document_id


def _bridge_record_requires_followup(record: dict[str, Any]) -> bool:
    if record.get("requires_followup") is True:
        return True
    if record.get("covered") is False or record.get("resolved") is False:
        return True
    status = str(record.get("coverage_status") or record.get("status") or "").strip().lower()
    return status in {"missing", "partial", "unresolved"}


def _as_anchor_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [
            part.strip()
            for part in re.split(r"(?:<SEP>|[;|\n])", value)
            if part.strip()
        ]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [
            part.strip()
            for part in re.split(r"[,;|\n]", value)
            if part.strip()
        ]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _useful_bridge_anchor(value: str) -> bool:
    anchor = " ".join(str(value or "").strip().strip("\"'<>").split())
    lowered = anchor.lower()
    if not anchor or len(anchor) < 3 or len(anchor) > 120:
        return False
    if len(anchor.split()) > 14:
        return False
    if lowered in _GENERIC_BRIDGE_ANCHORS or _STRUCTURAL_ANCHOR_RE.fullmatch(lowered):
        return False
    if re.fullmatch(r"[0-9a-f-]{24,}", lowered) or "-chunk-" in lowered:
        return False
    return any(character.isalnum() for character in anchor)


def _normalize_query(query: str) -> str:
    return " ".join((query or "").lower().split())


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def validation_reasons_contain(reasons: list[str], *candidates: str) -> bool:
    return any(reason in candidates for reason in reasons)
