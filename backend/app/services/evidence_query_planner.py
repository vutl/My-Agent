"""Bounded semantic expansion for explicit multi-evidence retrieval.

The planner is deliberately narrower than general query rewriting.  It is
called only when the deterministic evidence-obligation parser has established
that a question requires complementary facts (ratio, comparison, or explicit
multi-hop anaphora).  The model sees only the user's query and query-derived
operands; it never sees qrels, gold pages, answers, or retrieved document text.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from app.rag.evidence_obligations import (
    EvidenceObligation,
    EvidenceObligationPlan,
    plan_evidence_obligations,
)


EVIDENCE_QUERY_PLANNER_VERSION = "evidence-query-planner-v1"
_GENERIC_RETRIEVAL_TERMS = frozenset(
    {
        "annual", "document", "figure", "file", "paper", "report", "result",
        "results", "table", "value", "values",
    }
)
_SEMANTIC_STOPWORDS = _GENERIC_RETRIEVAL_TERMS | frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "what", "which", "with",
    }
)


class SupportsChat(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int = 512,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class EvidenceQueryPlanningResult:
    plan: EvidenceObligationPlan | None
    model_used: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EvidenceQueryPlannerService:
    client: SupportsChat
    default_model: str

    async def plan(
        self,
        *,
        query: str,
        model: str | None = None,
        base_plan: EvidenceObligationPlan | None = None,
    ) -> EvidenceQueryPlanningResult:
        deterministic = base_plan or plan_evidence_obligations(query)
        if deterministic is None or not deterministic.enabled:
            return EvidenceQueryPlanningResult(
                plan=deterministic,
                model_used=False,
                diagnostics={
                    "version": EVIDENCE_QUERY_PLANNER_VERSION,
                    "reason": "single_evidence_query",
                },
            )

        # Conditional lookups deliberately keep the anchor and target search
        # branches independent.  The join value (often a year) is discovered
        # from the anchor evidence and may not occur in the target chart/table;
        # an LLM expansion that copies it into both branches would suppress
        # the complementary artifact.  The deterministic plan is therefore
        # the safe production path for this structure and avoids an avoidable
        # gateway round-trip.
        if deterministic.kind == "conditional_multihop":
            return EvidenceQueryPlanningResult(
                plan=deterministic,
                model_used=False,
                diagnostics={
                    "version": EVIDENCE_QUERY_PLANNER_VERSION,
                    "reason": "conditional_branches_keep_join_key_independent",
                    "expanded": False,
                },
            )

        # Ratio/comparison plans already carry bounded operand aliases from
        # the deterministic vocabulary. A second Sol round-trip adds several
        # seconds to every structured question while often repeating the same
        # labels; keep the model planner available for ratio queries (where
        # domain terminology can be especially varied), but make comparison
        # retrieval latency local-first and auditable.
        if deterministic.kind == "comparison":
            return EvidenceQueryPlanningResult(
                plan=deterministic,
                model_used=False,
                diagnostics={
                    "version": EVIDENCE_QUERY_PLANNER_VERSION,
                    "reason": "comparison_uses_deterministic_operand_aliases",
                    "expanded": False,
                },
            )

        selected_model = model or self.default_model
        operands = "\n".join(
            f"- {item.obligation_id}: {item.query}"
            for item in deterministic.obligations
        )
        completion = await self.client.chat(
            model=selected_model,
            temperature=0.0,
            num_predict=320,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce search queries, not answers. Decompose an "
                        "explicit multi-evidence document question into one "
                        "retrieval query per required operand. Expand common "
                        "technical, scientific, or accounting aliases only when "
                        "useful for finding how a source document may phrase the "
                        "same concept (including alternate row labels or units). "
                        "For a metric, include likely synonyms such as cost/"
                        "expense wording when appropriate, but never add an "
                        "answer. Preserve named entities, dates, datasets, "
                        "and acronyms from the question. Never invent numeric "
                        "values, page/table/figure numbers, citations, results, or "
                        "an answer. Return JSON only with this shape: "
                        '{"queries":[{"id":"operand_1","query":"..."}]}. '
                        "Return exactly the supplied operand IDs in the same order."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Required operands:\n{operands}"
                    ),
                },
            ],
        )
        raw = str(getattr(completion, "message", None) or completion)
        try:
            payload = _parse_json_object(raw)
            expanded = _validated_expansion(
                query=query,
                base_plan=deterministic,
                payload=payload,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # A malformed planning payload must not manufacture search terms.
            # The query-derived deterministic plan remains safe and auditable;
            # provider/transport failures are deliberately not caught here.
            return EvidenceQueryPlanningResult(
                plan=deterministic,
                model_used=True,
                diagnostics={
                    "version": EVIDENCE_QUERY_PLANNER_VERSION,
                    "reason": "invalid_model_plan",
                    "error_type": type(exc).__name__,
                    "error": " ".join(str(exc).split())[:240],
                    "expanded": False,
                },
            )
        return EvidenceQueryPlanningResult(
            plan=expanded,
            model_used=True,
            diagnostics={
                "version": EVIDENCE_QUERY_PLANNER_VERSION,
                "reason": "validated_semantic_expansion",
                "expanded": [item.query for item in expanded.obligations]
                != [item.query for item in deterministic.obligations],
                "model": selected_model,
            },
        )


def _validated_expansion(
    *,
    query: str,
    base_plan: EvidenceObligationPlan,
    payload: dict[str, Any],
) -> EvidenceObligationPlan:
    rows = payload.get("queries")
    if not isinstance(rows, list) or len(rows) != len(base_plan.obligations):
        raise ValueError("query_count_mismatch")
    expected_ids = [item.obligation_id for item in base_plan.obligations]
    actual_ids = [str(row.get("id") or "") for row in rows if isinstance(row, dict)]
    if actual_ids != expected_ids:
        raise ValueError("operand_id_mismatch")

    allowed_numbers = set(_numeric_tokens(query))
    shared_anchors = set(base_plan.obligations[0].anchors)
    for obligation in base_plan.obligations[1:]:
        shared_anchors.intersection_update(obligation.anchors)
    required_shared = {
        _normalize_token(anchor)
        for anchor in shared_anchors
        if len(_normalize_token(anchor)) >= 2
    }

    expanded: list[EvidenceObligation] = []
    for base, row in zip(base_plan.obligations, rows, strict=True):
        value = _compact_search_query(str(row.get("query") or ""))
        if not value or len(value) > 400 or len(re.findall(r"\S+", value)) > 64:
            raise ValueError(f"invalid_query_size:{base.obligation_id}")
        introduced_numbers = set(_numeric_tokens(value)) - allowed_numbers
        if introduced_numbers:
            raise ValueError(f"invented_numeric_anchor:{base.obligation_id}")
        normalized_value = _normalize_token(value)
        missing_shared = {
            anchor for anchor in required_shared if anchor not in normalized_value
        }
        if missing_shared:
            raise ValueError(f"dropped_shared_anchor:{base.obligation_id}")
        expanded.append(
            EvidenceObligation(
                obligation_id=base.obligation_id,
                query=value,
                # Reservation remains grounded in query-derived anchors. Model-
                # added aliases may retrieve candidates but cannot themselves
                # certify that a candidate satisfies an operand.
                core_anchors=base.core_anchors,
                anchors=base.anchors,
                semantic_anchors=_semantic_anchors(
                    value,
                    shared=required_shared,
                ),
                semantic_phrases=_semantic_phrases(
                    value,
                    shared=required_shared,
                ),
            )
        )
    return EvidenceObligationPlan(
        kind=base_plan.kind,
        obligations=tuple(expanded),
        reason=f"{base_plan.reason}+semantic_expansion",
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("json_object_missing")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("json_object_required")
    return parsed


def _numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+(?:[.,]\d+)*", str(value or "")))


def _normalize_token(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _semantic_anchors(value: str, *, shared: set[str]) -> tuple[str, ...]:
    tokens = [
        token.casefold()
        for token in re.findall(r"[\w&+./_-]{2,}", value, flags=re.UNICODE)
    ]
    return tuple(
        dict.fromkeys(
            token
            for token in tokens
            if token not in shared
            and token not in _SEMANTIC_STOPWORDS
            and not token.isdigit()
        )
    )[:16]


def _semantic_phrases(value: str, *, shared: set[str]) -> tuple[str, ...]:
    # Build phrases from the original token stream instead of the filtered
    # anchor list.  Filtering first used to turn ``total assets`` into only
    # ``assets`` and ``product development costs`` into disconnected terms;
    # the exact row/section phrase is often the strongest signal for selecting
    # the complementary evidence of a multi-hop question.  Common function
    # words and shared document/date anchors are still removed, but semantic
    # role words such as ``total`` and ``cost`` remain available in phrases.
    tokens = [
        token.casefold()
        for token in re.findall(r"[\w&+./_-]{2,}", value, flags=re.UNICODE)
    ]
    filtered = [
        token
        for token in tokens
        if token not in shared
        and token not in _SEMANTIC_STOPWORDS
        and not token.isdigit()
    ]
    return tuple(
        dict.fromkeys(
            " ".join(pair)
            for pair in zip(filtered, filtered[1:])
            if len(pair[0]) >= 3 and len(pair[1]) >= 3
        )
    )[:8]


def _compact_search_query(value: str) -> str:
    """Remove generic document boilerplate while retaining content terms.

    A planner often appends ``annual report`` or ``paper`` to every operand.
    Those words add no discrimination inside an already scoped corpus and can
    push a high-value term out of the embedding/FTS focus.  This is a generic
    token policy, not a paper-specific rewrite or a word-count heuristic.
    """

    tokens = re.findall(r"[\w&+./_-]+", " ".join(str(value or "").split()), flags=re.UNICODE)
    kept = [
        token
        for token in tokens
        if token.casefold() not in _GENERIC_RETRIEVAL_TERMS
    ]
    return " ".join(kept).strip()
