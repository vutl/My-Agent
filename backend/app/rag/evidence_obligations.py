"""Bounded, query-only evidence obligations for multi-evidence retrieval.

The ordinary dense/lexical query is intentionally kept as the primary search.
Some questions, however, cannot be answered from one locally relevant chunk:
ratios need a numerator and denominator, comparisons need both sides, and an
anaphoric second sentence may depend on a fact established by the first.  This
module recognizes only those explicit structures and exposes independent
retrieval obligations.  It never consults qrels, document contents, filenames,
page numbers, or answer text.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable

from app.rag.evidence_identity import canonical_evidence_id


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    obligation_id: str
    query: str
    core_anchors: tuple[str, ...]
    anchors: tuple[str, ...]
    # Optional aliases supplied by a validated semantic query planner.  These
    # are retrieval hints only; they never become evidence or answer text.
    semantic_anchors: tuple[str, ...] = ()
    semantic_phrases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceObligationPlan:
    kind: str
    obligations: tuple[EvidenceObligation, ...]
    reason: str

    @property
    def enabled(self) -> bool:
        return len(self.obligations) >= 2


_QUESTION_PREFIX_RE = re.compile(
    r"^(?:what\s+is|what\s+was|calculate|compute|find|determine|"
    r"how\s+much|bao\s+nhi[eề]u|t[ií]nh|x[aá]c\s+[đd]ịnh)\s+",
    re.IGNORECASE,
)
_RATIO_OF_RE = re.compile(
    r"\b(?:the\s+)?(?:ratio|proportion)\s+of\s+(?P<left>.+?)\s+"
    r"(?:to|over|versus|vs\.?)\s+(?P<right>.+?)"
    r"(?P<context>\s+(?:for|in|during|on)\s+.+)?$",
    re.IGNORECASE,
)
_TO_RATIO_RE = re.compile(
    r"(?P<left>.+?)\s+(?:to|over|versus|vs\.?)\s+(?P<right>.+?)\s+"
    r"(?:ratio|proportion)(?P<context>\s+(?:for|in|during|on)\s+.+)?$",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"\b(?:difference|gap|change|growth|increase|decrease|delta|"
    r"ch[eê]nh\s+l[eệ]ch|thay\s+[đd][oổ]i|m[uứ]c\s+t[aă]ng|m[uứ]c\s+gi[aả]m)"
    r"(?:\s+(?:in|of|for|v[eề]))?\s+(?P<context>.+?)?\b"
    r"(?:between|gi[uữ]a)\s+(?P<left>.+?)\s+(?:and|v[aà])\s+(?P<right>.+)$",
    re.IGNORECASE,
)
_HIGHER_LOWER_RE = re.compile(
    r"(?:how\s+much\s+)?(?:higher|lower|greater|smaller|larger|less)\s+"
    r"(?:is|was|were|are|did)?\s*(?P<left>.+?)\s+than\s+(?P<right>.+)$",
    re.IGNORECASE,
)
_VI_HIGHER_LOWER_RE = re.compile(
    r"(?P<left>.+?)\s+(?:cao|th[aấ]p|l[oớ]n|nh[oỏ])\s+h[oơ]n\s+"
    r"(?P<right>.+?)(?:\s+bao\s+nhi[eề]u)?$",
    re.IGNORECASE,
)
_VI_RATIO_RE = re.compile(
    r"(?:t[yỷ]\s+l[eệ]|t[iỉ]\s+l[eệ])\s+(?:c[uủ]a\s+)?(?P<left>.+?)\s+"
    r"(?:tr[eê]n|so\s+v[oớ]i|v[oớ]i)\s+(?P<right>.+)$",
    re.IGNORECASE,
)
_CONDITIONAL_LOOKUP_RE = re.compile(
    r"\b(?:in|during|for)\s+(?:the\s+)?"
    r"(?:year|period|time|date)\s+(?:in\s+which|when|where)\s+"
    r"(?P<left>.+?)(?:,|;)?\s+"
    r"(?:what|which|how\s+much|how\s+many)\s+(?P<right>.+)$",
    re.IGNORECASE,
)
_ANAPHOR_RE = re.compile(
    r"\b(?:this|that|these|those|such|it|its|the\s+former|the\s+latter|"
    r"n[aà]y|[đd][oó]|[đd][aấ]y|n[oó]|c[uủ]a\s+n[oó]|nh[oó]m\s+n[aà]y|"
    r"m[oô]\s+h[iì]nh\s+n[aà]y|b[aà]i\s+n[aà]y)\b",
    re.IGNORECASE,
)
_MEANINGFUL_RE = re.compile(r"[\w][\w&+./_-]*", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
        "do", "does", "for", "from", "how", "in", "is", "it", "of", "on",
        "or", "that", "the", "this", "to", "was", "were", "what", "which",
        "with", "bao", "bị", "có", "của", "được", "gì", "giữa", "hơn", "là",
        "này", "nhiêu", "so", "thế", "trong", "và", "với",
    }
)

# Small, language-aware retrieval aliases for *roles*, not paper-specific
# entities.  They are used only to widen an explicit operand branch; the
# original operand remains the authority for obligation tagging and answer
# validation.  Keeping this vocabulary deterministic means a live agent does
# not become dependent on an extra planner call merely to find common row
# labels (for example, ``R&D`` versus ``Product development``).
_OPERAND_ALIASES: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset({"r&d", "rd", "research", "development"}),
        ("research and development", "product development", "development costs", "nghiên cứu phát triển"),
    ),
    (
        frozenset({"asset", "assets"}),
        ("total assets", "tổng tài sản"),
    ),
    (
        frozenset({"revenue", "revenues", "sales"}),
        ("net revenue", "net revenues", "doanh thu thuần"),
    ),
    (
        frozenset({"expense", "expenses", "cost", "costs"}),
        ("operating expense", "operating expenses", "chi phí hoạt động"),
    ),
    (
        frozenset({"employees", "employee", "headcount", "workforce"}),
        ("number of employees", "số lượng nhân viên"),
    ),
)


def plan_evidence_obligations(query: str) -> EvidenceObligationPlan | None:
    """Return explicit multi-evidence obligations without an LLM call.

    The parser is deliberately conservative.  A false negative keeps the
    existing single-query path; a weak or ambiguous split must not create a
    noisy extra dense search.
    """

    raw = " ".join(unicodedata.normalize("NFKC", str(query or "")).split()).strip()
    if not raw:
        return None

    sentences = [
        sentence.strip(" \t\r\n?!.")
        for sentence in re.split(r"(?<=[?.!])\s+", raw)
        if sentence.strip(" \t\r\n?!.")
    ]
    if len(sentences) >= 2 and _ANAPHOR_RE.search(sentences[-1]):
        first = " ".join(sentences[:-1]).strip()
        second = sentences[-1].strip()
        obligations = _make_obligations((first, second), shared=())
        if len(obligations) >= 2:
            return EvidenceObligationPlan(
                kind="anaphoric_multihop",
                obligations=obligations,
                reason="later_sentence_depends_on_fact_from_earlier_sentence",
            )

    stripped = _QUESTION_PREFIX_RE.sub("", raw).strip(" ?.")
    conditional = _CONDITIONAL_LOOKUP_RE.search(raw)
    if conditional is not None:
        left = _clean_operand(conditional.group("left") or "")
        right = _clean_operand(conditional.group("right") or "")
        # The join key (often a year) is discovered from the first operand's
        # evidence.  Do not copy its numeric value into the second search
        # branch: that value may be absent from the target chart/table and
        # would suppress the complementary evidence needed for the join.
        obligations = _make_obligations((left, right), shared=())
        if len(obligations) >= 2:
            return EvidenceObligationPlan(
                kind="conditional_multihop",
                obligations=obligations,
                reason="conditional_lookup_requires_anchor_and_target_evidence",
            )
    for kind, pattern in (
        ("ratio", _RATIO_OF_RE),
        ("ratio", _TO_RATIO_RE),
        ("comparison", _HIGHER_LOWER_RE),
        ("comparison", _VI_HIGHER_LOWER_RE),
        ("ratio", _VI_RATIO_RE),
        ("temporal_or_value_comparison", _BETWEEN_RE),
    ):
        match = pattern.search(stripped)
        if match is None:
            continue
        left = _clean_operand(match.groupdict().get("left") or "")
        right = _clean_operand(match.groupdict().get("right") or "")
        context = _clean_context(match.groupdict().get("context") or "")
        operand_tokens = {
            token.casefold()
            for token in (*_tokens(left), *_tokens(right))
        }
        shared = tuple(
            anchor
            for anchor in dict.fromkeys((*_shared_anchors(raw), *_tokens(context)))
            if anchor.casefold() not in operand_tokens
        )
        obligations = _make_obligations((left, right), shared=shared)
        if len(obligations) >= 2:
            return EvidenceObligationPlan(
                kind=kind,
                obligations=obligations,
                reason="explicit_operator_requires_complementary_operands",
            )
    return None


def reserve_obligation_coverage(
    results: Iterable[dict[str, Any]],
    *,
    plan: EvidenceObligationPlan | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep one best lexical candidate per obligation inside a bounded list.

    Candidates are still produced by the normal retriever.  This function only
    performs a constrained final selection, analogous to reserving one source
    per required document in a comparison.  It never creates evidence.
    """

    ranked = list(results)
    if limit < 1 or not ranked or plan is None or not plan.enabled:
        return ranked[: max(0, limit)]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for obligation in plan.obligations:
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for rank, item in enumerate(ranked, start=1):
            identity = canonical_evidence_id(item)
            if identity and identity in selected_ids:
                continue
            core_coverage = _anchor_coverage(item, obligation.core_anchors)
            coverage = _anchor_coverage(item, obligation.anchors)
            semantic_coverage = _anchor_coverage(
                item,
                obligation.semantic_anchors,
            )
            semantic_phrase_coverage = _anchor_coverage(
                item,
                obligation.semantic_phrases,
            )
            numeric_proximity = _numeric_proximity_score(
                item,
                obligation.semantic_phrases,
            )
            structured_score = _structured_numeric_evidence_score(item)
            tagged = obligation.obligation_id in set(item.get("evidence_obligation_ids") or [])
            # A branch origin is diagnostic metadata, not proof that its
            # result contains the requested operand. Shared anchors such as a
            # year/company may occur on hundreds of pages in a long report;
            # require at least one operand-specific anchor before reserving a
            # slot. If none exists, fail open to the ordinary ranking rather
            # than promoting branch noise.
            if core_coverage <= 0 and semantic_coverage <= 0:
                continue
            candidates.append(
                (
                    (
                        float(semantic_phrase_coverage),
                        float(numeric_proximity),
                        float(structured_score),
                        float(semantic_coverage),
                        float(core_coverage),
                        float(coverage),
                        1.0 if tagged else 0.0,
                        -float(rank),
                    ),
                    item,
                )
            )
        if not candidates:
            continue
        # Prefer a branch-unique candidate when one exists.  A shared chunk
        # often ranks highly for both operands (for example a comparison
        # table header), but retaining only that shared chunk can evict the
        # complementary operand from the bounded final context.  Branch
        # uniqueness is derived from retrieval provenance, never from gold
        # answers or document-specific rules; if no unique candidate exists,
        # the normal evidence-coverage score remains the fallback.
        branch_unique = [
            pair
            for pair in candidates
            if obligation.obligation_id
            in set(pair[1].get("evidence_obligation_ids") or [])
            and set(pair[1].get("evidence_obligation_ids") or [])
            <= {obligation.obligation_id}
        ]
        if branch_unique:
            candidates = branch_unique
        _score, chosen = max(candidates, key=lambda pair: pair[0])
        copy = dict(chosen)
        ids = list(copy.get("evidence_obligation_ids") or [])
        if obligation.obligation_id not in ids:
            ids.append(obligation.obligation_id)
        copy["evidence_obligation_ids"] = ids
        selected.append(copy)
        identity = canonical_evidence_id(copy)
        if identity:
            selected_ids.add(identity)

    for item in ranked:
        if len(selected) >= limit:
            break
        identity = canonical_evidence_id(item)
        if identity and identity in selected_ids:
            continue
        selected.append(item)
        if identity:
            selected_ids.add(identity)
    return selected[:limit]


def tag_obligation_results(
    results: Iterable[dict[str, Any]],
    obligation: EvidenceObligation,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for item in results:
        copy = dict(item)
        ids = list(copy.get("evidence_obligation_ids") or [])
        if obligation.obligation_id not in ids:
            ids.append(obligation.obligation_id)
        copy["evidence_obligation_ids"] = ids
        tagged.append(copy)
    return tagged


def obligation_search_query(obligation: EvidenceObligation) -> str:
    """Build the bounded query sent to the local retriever for one obligation.

    ``EvidenceObligation.query`` is the immutable query-derived anchor.  The
    optional semantic fields are retrieval-only aliases (from the deterministic
    vocabulary or a validated planner response); keeping this assembly here
    makes it impossible for callers to accidentally use aliases as evidence.
    """

    return " ".join(
        dict.fromkeys(
            part.strip()
            for part in (
                obligation.query,
                *obligation.semantic_phrases,
            )
            if part and part.strip()
        )
    )[:400]


def _make_obligations(
    values: Iterable[str],
    *,
    shared: tuple[str, ...],
) -> tuple[EvidenceObligation, ...]:
    obligations: list[EvidenceObligation] = []
    seen: set[str] = set()
    for value in values:
        operand = _clean_operand(value)
        core_anchors = tuple(
            token
            for token in _tokens(operand)
            if token.casefold() not in _STOPWORDS
        )
        anchors = tuple(dict.fromkeys((*core_anchors, *shared)))
        if not operand or not core_anchors:
            continue
        aliases = _operand_aliases(core_anchors)
        # Keep the deterministic operand query stable and auditable.  Alias
        # terms live in the separate semantic fields and are appended by the
        # retrieval executor, so qrel/rewrite contracts never change merely
        # because the local vocabulary evolves.
        branch_query = " ".join(dict.fromkeys((operand, *shared))).strip()
        key = branch_query.casefold()
        if key in seen:
            continue
        seen.add(key)
        obligations.append(
            EvidenceObligation(
                obligation_id=f"operand_{len(obligations) + 1}",
                query=branch_query,
                core_anchors=core_anchors[:12],
                anchors=anchors[:16],
                semantic_anchors=tuple(
                    token
                    for token in _tokens(" ".join(aliases))
                    if token.casefold() not in core_anchors
                )[:16],
                semantic_phrases=tuple(aliases)[:8],
            )
        )
        if len(obligations) >= 3:
            break
    return tuple(obligations)


def _operand_aliases(core_anchors: tuple[str, ...]) -> tuple[str, ...]:
    """Return bounded row-label aliases for an explicit operand.

    Alias groups are activated by token overlap and are intentionally capped
    per operand.  This is not a synonym thesaurus for arbitrary queries: the
    caller must already have proven that the question has complementary
    evidence obligations, so the additional terms cannot turn ordinary fact
    retrieval into a broad noisy expansion.
    """

    tokens = {token.casefold() for token in core_anchors}
    aliases: list[str] = []
    for triggers, values in _OPERAND_ALIASES:
        if not tokens.intersection(triggers):
            continue
        aliases.extend(values)
    return tuple(dict.fromkeys(aliases))[:6]


def _clean_operand(value: str) -> str:
    cleaned = _QUESTION_PREFIX_RE.sub("", " ".join(str(value or "").split()))
    cleaned = re.sub(
        r"^(?:compared\s+with\s+[^,]+,?\s*|in\s+[^,]+,?\s*)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,;:?.")[:240]


def _clean_context(value: str) -> str:
    return " ".join(str(value or "").strip(" ,;:?.").split())[:160]


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _MEANINGFUL_RE.findall(str(value or ""))
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    )


def _shared_anchors(query: str) -> tuple[str, ...]:
    anchors: list[str] = []
    for quoted in re.findall(r"[\"']([^\"']{2,80})[\"']", query):
        anchors.append(quoted.strip())
    for token in _MEANINGFUL_RE.findall(query):
        if any(char.isdigit() for char in token):
            anchors.append(token)
        elif token.isupper() and len(token) >= 2:
            anchors.append(token)
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _anchor_coverage(item: dict[str, Any], anchors: tuple[str, ...]) -> int:
    searchable = unicodedata.normalize(
        "NFKC",
        "\n".join(
            str(item.get(field) or "")
            for field in (
                "filename",
                "section_title",
                "caption",
                "content",
                "text",
                "expanded_content",
            )
        ),
    ).casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in searchable)


def _structured_numeric_evidence_score(item: dict[str, Any]) -> int:
    """Prefer compact numeric/table-like evidence for computed obligations.

    This is a format-neutral signal: a candidate with several values, a
    currency/percent marker, or an explicit table artifact is more likely to
    contain a numerator/denominator than a prose mention of the same term.
    It does not inspect answers, qrels, page numbers, or document-specific
    labels.
    """

    artifact_type = str(item.get("artifact_type") or item.get("chunk_type") or "").casefold()
    searchable = "\n".join(
        str(item.get(field) or "")
        for field in ("caption", "content", "text", "expanded_content")
    )
    numbers = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", searchable)
    score = min(len(numbers), 8)
    if item.get("table_id") or artifact_type == "table":
        score += 8
    if re.search(r"[$€£%]|\|", searchable):
        score += 3
    if re.search(r"\b(?:total|amounts?|year ended|at December)\b", searchable, re.IGNORECASE):
        score += 2
    return score


def _numeric_proximity_score(
    item: dict[str, Any],
    phrases: tuple[str, ...],
) -> int:
    """Score numeric support immediately adjacent to a semantic row label.

    Long reports often mention a requested label in narrative prose (for
    example, saying that a segment does *not* disclose total assets) as well
    as in the actual statement row.  A phrase hit alone cannot distinguish the
    two.  Numbers/currency near the phrase are a format-neutral signal that a
    candidate carries the value-bearing row, without requiring a known answer
    or page.
    """

    searchable = unicodedata.normalize(
        "NFKC",
        "\n".join(
            str(item.get(field) or "")
            for field in ("caption", "content", "text", "expanded_content")
        ),
    )
    lowered = searchable.casefold()
    best = 0
    for phrase in phrases:
        needle = str(phrase or "").casefold().strip()
        if not needle:
            continue
        start = 0
        while True:
            offset = lowered.find(needle, start)
            if offset < 0:
                break
            window = searchable[offset : offset + len(needle) + 180]
            number_count = len(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", window))
            marker = bool(re.search(r"[$€£%]|\|", window))
            score = min(number_count, 6) + (2 if marker else 0)
            best = max(best, score)
            start = offset + max(1, len(needle))
    return best
