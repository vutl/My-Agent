from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Any


@dataclass(frozen=True)
class EvidenceValidationResult:
    valid: bool
    retry_required: bool
    reason: str
    required_entities: list[str]
    matched_entities: list[str]
    missing_entities: list[str]
    missing_document_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "retry_required": self.retry_required,
            "reason": self.reason,
            "required_entities": self.required_entities,
            "matched_entities": self.matched_entities,
            "missing_entities": self.missing_entities,
            "missing_document_ids": self.missing_document_ids,
        }


@dataclass(frozen=True)
class MetricValueClaim:
    """A bounded, machine-checkable quantitative claim.

    This intentionally does not attempt to validate every number in an answer.
    Only metric/value pairs (Acc, F1, CCC, WER, parameter counts, etc.) and bare
    percentages are checked. Figure numbers, years, layer counts, and qualitative
    statements are left to the answer prompt instead of being rejected by a
    brittle parser.
    """

    metric: str
    value: str
    percentage: bool
    subjects: tuple[str, ...]
    text: str
    qualifiers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "percentage": self.percentage,
            "subjects": list(self.subjects),
            "qualifiers": list(self.qualifiers),
            "text": self.text,
        }


@dataclass(frozen=True)
class AnswerClaimValidationResult:
    valid: bool
    retry_required: bool
    reason: str
    checked_claims: list[MetricValueClaim]
    supported_claims: list[MetricValueClaim]
    unsupported_claims: list[MetricValueClaim]
    foreign_document_ids: list[str]
    unparsed_signals: list[dict[str, Any]] = field(default_factory=list)
    covered_document_ids: list[str] = field(default_factory=list)
    missing_document_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "retry_required": self.retry_required,
            "reason": self.reason,
            "checked_claims": [claim.to_dict() for claim in self.checked_claims],
            "supported_claims": [claim.to_dict() for claim in self.supported_claims],
            "unsupported_claims": [claim.to_dict() for claim in self.unsupported_claims],
            "foreign_document_ids": self.foreign_document_ids,
            "unparsed_signals": self.unparsed_signals,
            "covered_document_ids": self.covered_document_ids,
            "missing_document_ids": self.missing_document_ids,
        }


def validate_retrieval_evidence(
    *,
    documents: list[dict],
    required_entities: list[str],
    current_topic: str | None,
    is_followup: bool,
    focus_document_ids: list[str] | None,
    require_all_focus_documents: bool = False,
) -> EvidenceValidationResult:
    entities = _normalize_required_entities(required_entities, current_topic)
    strict_entities = _strict_entities(required_entities, current_topic)
    focus_ids = _normalize_document_ids(focus_document_ids)
    if not documents:
        return EvidenceValidationResult(
            valid=False,
            retry_required=bool(is_followup or focus_document_ids or entities),
            reason="no_documents",
            required_entities=entities,
            matched_entities=[],
            missing_entities=entities,
            missing_document_ids=(
                sorted(focus_ids) if require_all_focus_documents else []
            ),
        )

    if focus_ids:
        explicit_ids = {
            str(document.get("document_id")).strip()
            for document in documents
            if document.get("document_id")
        }
        in_focus = explicit_ids & focus_ids
        foreign_ids = explicit_ids - focus_ids
        if not in_focus:
            return EvidenceValidationResult(
                valid=False,
                retry_required=True,
                reason="focus_document_mismatch",
                required_entities=entities,
                matched_entities=[],
                missing_entities=entities,
            )
        if foreign_ids:
            return EvidenceValidationResult(
                valid=False,
                retry_required=True,
                reason="mixed_focus_documents",
                required_entities=entities,
                matched_entities=[],
                missing_entities=entities,
            )
        if require_all_focus_documents and in_focus != focus_ids:
            missing_focus_ids = sorted(focus_ids - in_focus)
            return EvidenceValidationResult(
                valid=False,
                retry_required=True,
                reason="missing_focus_documents",
                required_entities=entities,
                matched_entities=[
                    entity
                    for entity in entities
                    if _entity_matches_any_document(documents, entity)
                ],
                missing_entities=[],
                missing_document_ids=missing_focus_ids,
            )

        # A canonical document_id is stronger paper identity evidence than a
        # chunk mentioning the title. Many valid inner-page chunks omit it.
        matched = [entity for entity in entities if _entity_matches_any_document(documents, entity)]
        return EvidenceValidationResult(
            valid=True,
            retry_required=False,
            reason="focused_documents_present",
            required_entities=entities,
            matched_entities=matched,
            missing_entities=[],
        )

    if not strict_entities:
        return EvidenceValidationResult(
            valid=True,
            retry_required=False,
            reason="no_required_entities",
            required_entities=[],
            matched_entities=[],
            missing_entities=[],
        )

    matched: list[str] = []
    for entity in entities:
        require_identity = bool(current_topic and entity.lower() == current_topic.strip().lower())
        if _entity_matches_any_document(documents, entity, require_identity=require_identity):
            matched.append(entity)
    strict_missing = [entity for entity in strict_entities if entity not in matched]
    if strict_missing:
        return EvidenceValidationResult(
            valid=False,
            retry_required=True,
            reason="missing_required_entities",
            required_entities=entities,
            matched_entities=matched,
            missing_entities=strict_missing,
        )

    return EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="required_entities_present",
        required_entities=entities,
        matched_entities=matched,
        missing_entities=[],
    )


def validate_answer_claims(
    *,
    answer: str,
    documents: list[dict],
    focus_document_ids: list[str] | None = None,
    require_all_focus_documents: bool = False,
    answer_document_ids: list[str] | None = None,
) -> AnswerClaimValidationResult:
    """Validate exact metric/value claims against retrieved evidence.

    When a sticky paper focus is supplied, foreign and unprovenanced chunks are
    deliberately ineligible as support. This prevents a numerically identical
    result from another paper from legitimising a claim about the active paper.
    The function is side-effect free so the agent can run it after generation
    and decide whether to retry or replace unsupported quantitative sentences.
    """

    focus_ids = _normalize_document_ids(focus_document_ids)
    foreign_ids: set[str] = set()
    eligible_documents: list[dict] = []
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if focus_ids:
            if not document_id:
                continue
            if document_id not in focus_ids:
                foreign_ids.add(document_id)
                continue
        eligible_documents.append(document)

    claims = _claims_with_answer_document_context(
        answer,
        documents=eligible_documents,
        focus_ids=focus_ids,
    )

    identity_document_ids = (
        _normalize_document_ids(answer_document_ids) & focus_ids
        if answer_document_ids is not None
        else _answer_identity_document_ids(
            answer,
            eligible_documents,
            focus_ids=focus_ids,
        )
    )
    covered_document_ids = set(identity_document_ids)

    def missing_required_documents() -> list[str]:
        if not require_all_focus_documents or len(focus_ids) < 2:
            return []
        return sorted(focus_ids - covered_document_ids)

    if not claims:
        # A parser miss must never turn an obviously quantitative answer into a
        # successful validation. This is deliberately fail-closed: the answer
        # guard gets one constrained retry and then emits its number-free
        # fallback. Structural counts such as "top 12 transformer layers" are
        # filtered by the same bounded-number rules and do not trigger this.
        unparsed_signals = _unparsed_metric_value_signals(answer)
        if unparsed_signals:
            missing_document_ids = missing_required_documents()
            return AnswerClaimValidationResult(
                valid=False,
                retry_required=True,
                reason="unparsed_metric_values",
                checked_claims=[],
                supported_claims=[],
                unsupported_claims=[],
                foreign_document_ids=sorted(foreign_ids),
                unparsed_signals=unparsed_signals,
                covered_document_ids=sorted(covered_document_ids),
                missing_document_ids=missing_document_ids,
            )
        missing_document_ids = missing_required_documents()
        return AnswerClaimValidationResult(
            valid=not missing_document_ids,
            retry_required=bool(missing_document_ids),
            reason=(
                "missing_answer_documents"
                if missing_document_ids
                else "no_metric_value_claims"
            ),
            checked_claims=[],
            supported_claims=[],
            unsupported_claims=[],
            foreign_document_ids=sorted(foreign_ids),
            covered_document_ids=sorted(covered_document_ids),
            missing_document_ids=missing_document_ids,
        )

    evidence_claims: list[tuple[MetricValueClaim, frozenset[str], str]] = []
    for document in eligible_documents:
        document_id = str(document.get("document_id") or "").strip()
        identity = _document_identity_text(document)
        identity_subjects = frozenset(_subject_identifiers(identity))
        global_metric_qualifiers = _document_metric_qualifiers(document)
        global_percentage_metrics = _document_percentage_metrics(document)
        for claim in _extract_metric_claims(_document_claim_text(document)):
            claim = _claim_with_global_metric_metadata(
                claim,
                global_metric_qualifiers,
                global_percentage_metrics,
            )
            evidence_claims.append(
                (
                    _claim_with_identity_subjects(claim, identity),
                    identity_subjects,
                    document_id,
                )
            )

    # For a quantitative answer, merely naming a paper is not enough: at least
    # one claim must both name that catalog identity and be supported by evidence
    # from that same document. Qualitative answers have no bounded claims, so
    # explicit identities remain the only conservative coverage signal.
    covered_document_ids = (
        _answer_insufficiency_document_ids(
            answer,
            eligible_documents,
            focus_ids=focus_ids,
        )
        & identity_document_ids
    )
    supported: list[MetricValueClaim] = []
    unsupported: list[MetricValueClaim] = []
    for claim in claims:
        candidate_evidence = evidence_claims
        if claim.qualifiers:
            qualified_evidence = [
                item
                for item in evidence_claims
                if item[0].metric == claim.metric and item[0].qualifiers
            ]
            # Once the retrieved evidence exposes a dimension-aware schema for
            # this metric, an unqualified prose duplicate must not bypass it.
            # Preserve legacy prose-only validation when no such schema exists.
            if qualified_evidence:
                candidate_evidence = qualified_evidence
        supporting_document_ids = {
            document_id
            for evidence_claim, identity_subjects, document_id in candidate_evidence
            if _claim_is_supported(
                claim,
                evidence_claim,
                evidence_identity_subjects=identity_subjects,
            )
        }
        if not supporting_document_ids and not claim.subjects:
            # A generated comparison can put the row owner in a neighboring
            # Markdown cell that is lost by the prose fallback parser. Accept
            # an owner-free measurement only when metric, value, units and
            # qualifiers identify exactly one evidence owner in exactly one
            # focused document. Repeated values, dimensions or owners remain
            # ambiguous and fail closed; explicitly named wrong owners never
            # enter this path.
            ownerless_candidates = {
                (
                    document_id,
                    evidence_claim.subjects,
                    evidence_claim.qualifiers,
                )
                for evidence_claim, _identity_subjects, document_id in candidate_evidence
                if _claim_measurement_matches(claim, evidence_claim)
            }
            if len(ownerless_candidates) == 1:
                unique_document_id, _subjects, _qualifiers = next(
                    iter(ownerless_candidates)
                )
                if unique_document_id in identity_document_ids:
                    supporting_document_ids = {unique_document_id}
        if supporting_document_ids:
            supported.append(claim)
            coverage_document_ids = {
                document_id
                for document_id in supporting_document_ids
                if document_id and document_id in identity_document_ids
            }
            if (
                len(coverage_document_ids) > 1
                and claim.subjects
                and set(claim.subjects).issubset(
                    {_PROPOSED_OWNER_SUBJECT, _BASELINE_OWNER_SUBJECT}
                )
            ):
                # ``Ours`` and ``Baseline`` are document-local roles. If two
                # papers report the same value, one generic claim must not
                # satisfy both coverage obligations merely because both titles
                # occur elsewhere in the answer. A per-paper line remains
                # valid because its claim text names exactly one local identity.
                local_identity_ids = _answer_identity_document_ids(
                    claim.text,
                    eligible_documents,
                    focus_ids=focus_ids,
                )
                coverage_document_ids = (
                    coverage_document_ids & local_identity_ids
                    if len(local_identity_ids) == 1
                    else set()
                )
            covered_document_ids.update(coverage_document_ids)
        else:
            unsupported.append(claim)

    missing_document_ids = missing_required_documents()
    if unsupported:
        reason = "unsupported_metric_values"
    elif missing_document_ids:
        reason = "missing_answer_documents"
    else:
        reason = "metric_values_supported"

    return AnswerClaimValidationResult(
        valid=not unsupported and not missing_document_ids,
        retry_required=bool(unsupported or missing_document_ids),
        reason=reason,
        checked_claims=claims,
        supported_claims=supported,
        unsupported_claims=unsupported,
        foreign_document_ids=sorted(foreign_ids),
        covered_document_ids=sorted(covered_document_ids),
        missing_document_ids=missing_document_ids,
    )


def _claims_with_answer_document_context(
    answer: str,
    *,
    documents: list[dict],
    focus_ids: set[str],
) -> list[MetricValueClaim]:
    """Bind subjectless metric lines to their nearest document section.

    Generated comparisons commonly use ``### Paper A`` followed by compact
    ``UA/WA`` lines.  The paper identity is structurally local even though it
    is not repeated on every line.  Ambiguous signatures seen under multiple
    document sections remain unbound and therefore fail closed.
    """

    claims = _extract_metric_claims(answer)
    if len(focus_ids) < 2 or not claims:
        return claims

    identity_by_document_id = {
        str(document.get("document_id") or "").strip(): _document_identity_text(document)
        for document in documents
        if document.get("document_id")
    }
    owner_subjects_by_document_id: dict[str, set[str]] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            continue
        identity_subjects = set(
            _subject_identifiers(identity_by_document_id.get(document_id, ""))
        )
        owners: set[str] = set()
        for evidence_claim in _extract_metric_claims(_document_claim_text(document)):
            owners.update(set(evidence_claim.subjects) - identity_subjects)
        owner_subjects_by_document_id[document_id] = owners

    contexts_by_signature: dict[
        tuple[Any, ...],
        set[tuple[str, tuple[str, ...]]],
    ] = {}
    active_document_id: str | None = None
    active_section_subjects: tuple[str, ...] = ()
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        line_document_ids = _answer_identity_document_ids(
            stripped,
            documents,
            focus_ids=focus_ids,
        )
        if len(line_document_ids) == 1 and _is_document_section_label(stripped):
            active_document_id = next(iter(line_document_ids))
            identity_subjects = set(
                _subject_identifiers(
                    identity_by_document_id.get(active_document_id, "")
                )
            )
            allowed_owners = owner_subjects_by_document_id.get(
                active_document_id,
                set(),
            )
            active_section_subjects = tuple(
                subject
                for subject in _subject_identifiers(stripped)
                if subject in identity_subjects or subject in allowed_owners
            )
        elif len(line_document_ids) > 1:
            active_document_id = None
            active_section_subjects = ()
        if not active_document_id:
            continue
        if _is_local_owner_section_label(stripped):
            allowed_owners = owner_subjects_by_document_id.get(
                active_document_id,
                set(),
            )
            local_owners = tuple(
                subject
                for subject in _subject_identifiers(stripped)
                if subject in allowed_owners
            )
            # Model ownership is often expressed as a Markdown subheading or
            # list label, with its metric values on the following line. Carry
            # only owners that actually occur in this document's evidence;
            # an arbitrary heading can therefore never authorize a baseline
            # value. A new structural label without a known owner closes the
            # previous local owner scope instead of leaking it indefinitely.
            active_section_subjects = local_owners
        for line_claim in _extract_line_claims(stripped):
            contexts_by_signature.setdefault(
                _claim_context_signature(line_claim),
                set(),
            ).add((active_document_id, active_section_subjects))

    contextualized: list[MetricValueClaim] = []
    for claim in claims:
        contexts = contexts_by_signature.get(_claim_context_signature(claim), set())
        if len(contexts) != 1:
            contextualized.append(claim)
            continue
        document_id, section_subjects = next(iter(contexts))
        identity_subjects = tuple(
            _subject_identifiers(identity_by_document_id.get(document_id, ""))
        )
        contextualized.append(
            MetricValueClaim(
                metric=claim.metric,
                value=claim.value,
                percentage=claim.percentage,
                subjects=tuple(
                    dict.fromkeys(
                        [*identity_subjects, *section_subjects, *claim.subjects]
                    )
                ),
                text=claim.text,
                qualifiers=claim.qualifiers,
            )
        )
    return contextualized


def _claim_context_signature(claim: MetricValueClaim) -> tuple[Any, ...]:
    return (
        claim.metric,
        claim.value,
        claim.percentage,
        claim.qualifiers,
    )


def _is_document_section_label(line: str) -> bool:
    compact = " ".join(str(line or "").split())
    if not compact:
        return False
    if re.match(r"^#{1,6}\s+", compact):
        return True
    unwrapped = compact.strip("*_` ")
    return bool(
        len(unwrapped) <= 120
        and not _METRIC_RE.search(unwrapped)
        and not _NUMBER_RE.search(unwrapped)
        and (
            compact.startswith(("**", "__"))
            or compact.endswith(":")
        )
    )


_EXPLICIT_INSUFFICIENCY_RE = re.compile(
    r"(?:"
    r"(?<!\w)(?:not\s+enough|insufficient)\s+(?:canonical\s+)?evidence(?!\w)"
    r"|(?<!\w)(?:can(?:not|'t)|could(?:\s+not|n't)|unable\s+to)\s+"
    r"(?:identify|find|confirm|verify|determine)(?!\w)"
    r"|(?<!\w)(?:is|are|was|were)\s+not\s+"
    r"(?:provided|reported|stated|specified|mentioned|available)(?!\w)"
    r"|(?<!\w)(?:không|chưa)\s+có\s+đủ\s+(?:canonical\s+)?"
    r"(?:evidence|bằng\s+chứng)(?!\w)"
    r"|(?<!\w)(?:không|chưa)\s+(?:được\s+)?"
    r"(?:nêu|cung\s+cấp|đề\s+cập|báo\s+cáo|xác\s+nhận)(?!\w)"
    r"|(?<!\w)(?:không|chưa)\s+thể\s+xác\s+nhận(?!\w)"
    r")",
    flags=re.IGNORECASE,
)


def _answer_insufficiency_document_ids(
    answer: str,
    documents: list[dict],
    *,
    focus_ids: set[str],
) -> set[str]:
    """Return documents with their own explicit, identity-bound abstention.

    A global ``not enough evidence`` sentence cannot satisfy multiple papers.
    The statement must occur under a structural line that resolves to exactly
    one focused document, matching the same section contract used by
    progressive multi-paper streaming.
    """

    if len(focus_ids) < 2 or not answer.strip():
        return set()
    sections: dict[str, list[str]] = {}
    active_document_id: str | None = None
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        line_ids = _answer_identity_document_ids(
            stripped,
            documents,
            focus_ids=focus_ids,
        )
        if len(line_ids) == 1 and _is_document_section_label(stripped):
            active_document_id = next(iter(line_ids))
            sections.setdefault(active_document_id, []).append(stripped)
            continue
        if len(line_ids) > 1:
            active_document_id = None
            continue
        if active_document_id:
            sections.setdefault(active_document_id, []).append(stripped)
    return {
        document_id
        for document_id, lines in sections.items()
        if _EXPLICIT_INSUFFICIENCY_RE.search("\n".join(lines))
    }


def _is_local_owner_section_label(line: str) -> bool:
    """Recognize a structural owner label without paper/model name lists.

    Generated answers may place ``Model 2.0 (baseline)`` on one Markdown line
    and ``UA ... / WA ...`` on the next. The caller intersects every candidate
    with owners extracted from the active document, so version-like numbers in
    a model label are harmless while arbitrary headings cannot authorize a
    value. Metric-labelled lines are never promoted to persistent context.
    """

    compact = " ".join(str(line or "").split())
    if not compact or len(compact) > 160 or _METRIC_RE.search(compact):
        return False
    if re.match(r"^#{1,6}\s+", compact):
        return True
    if re.match(r"^(?:[-+*]|\d+[.)])\s+", compact):
        return True
    unwrapped = compact.strip("*_` ")
    return bool(compact.startswith(("**", "__")) or unwrapped.endswith(":"))


def _normalize_required_entities(required_entities: list[str], current_topic: str | None) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for value in [*(required_entities or []), current_topic]:
        if value is None:
            continue
        entity = str(value).strip()
        if not entity or entity.lower() in {"null", "none", "n/a"}:
            continue
        if len(entity) < 2:
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity)
    return entities[:8]


def _strict_entities(required_entities: list[str], current_topic: str | None) -> list[str]:
    if current_topic:
        # The rewrite service can include metric/query terms alongside the
        # active paper. Those are retrieval anchors, not identity constraints.
        return _normalize_required_entities([], current_topic)

    strict: list[str] = []
    for entity in required_entities or []:
        text = str(entity).strip()
        if not text:
            continue
        if " " in text and not any(char.isupper() for char in text):
            continue
        strict.append(text)
        if len(strict) >= 2:
            break
    return _normalize_required_entities(strict, None)


def _normalize_document_ids(document_ids: list[str] | None) -> set[str]:
    return {str(document_id).strip() for document_id in document_ids or [] if str(document_id).strip()}


def _document_text(document: dict) -> str:
    return " ".join(
        str(part)
        for part in [
            _document_identity_text(document),
            document.get("caption"),
            document.get("content"),
        ]
        if part
    )


def _document_identity_text(document: dict) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    source_path = str(document.get("source_path") or "").replace("\\", "/")
    source_basename = source_path.rsplit("/", 1)[-1] if source_path else None
    catalog_aliases = metadata.get("catalog_aliases")
    if not isinstance(catalog_aliases, list):
        catalog_aliases = []
    return " ".join(
        str(part)
        for part in [
            document.get("filename"),
            source_basename,
            document.get("title"),
            document.get("document_title"),
            document.get("paper_title"),
            metadata.get("filename"),
            metadata.get("title"),
            metadata.get("document_title"),
            metadata.get("paper_title"),
            *catalog_aliases,
        ]
        if part
    )


def _answer_identity_document_ids(
    answer: str,
    documents: list[dict],
    *,
    focus_ids: set[str],
) -> set[str]:
    """Find uniquely owned catalog identities explicitly named in an answer."""

    answer_words = _coverage_words(answer)
    alias_owners: dict[tuple[str, ...], set[str]] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if not document_id or (focus_ids and document_id not in focus_ids):
            continue
        for alias in _document_coverage_aliases(document):
            alias_owners.setdefault(alias, set()).add(document_id)

    covered: set[str] = set()
    for alias, owners in alias_owners.items():
        if len(owners) != 1 or not _contains_word_tuple(answer_words, alias):
            continue
        covered.update(owners)
    return covered


def _document_coverage_aliases(document: dict) -> set[tuple[str, ...]]:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    aliases: set[tuple[str, ...]] = set()
    raw_values = [
        document.get("filename"),
        document.get("title"),
        document.get("document_title"),
        document.get("paper_title"),
        metadata.get("filename"),
        metadata.get("title"),
        metadata.get("document_title"),
        metadata.get("paper_title"),
    ]
    catalog_aliases = metadata.get("catalog_aliases")
    if isinstance(catalog_aliases, list):
        raw_values.extend(catalog_aliases)
    for raw_value in raw_values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        basename = re.split(r"[/\\]", value)[-1]
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", basename)
        words = tuple(_coverage_words(stem))
        if not words:
            continue
        if len("".join(words)) >= 3:
            aliases.add(words)
            if len(words) >= 2:
                aliases.add(("".join(words),))
    return aliases


def _coverage_words(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_text.casefold())


def _contains_word_tuple(words: list[str], sequence: tuple[str, ...]) -> bool:
    if not words or not sequence or len(sequence) > len(words):
        return False
    width = len(sequence)
    return any(
        tuple(words[index : index + width]) == sequence
        for index in range(len(words) - width + 1)
    )


def _document_claim_text(document: dict) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    return "\n".join(
        str(part)
        for part in [
            document.get("caption"),
            document.get("content"),
            metadata.get("caption"),
            metadata.get("content"),
        ]
        if part
    )


_METRIC_VARIANT_WORDS = ("macro", "micro", "weighted", "unweighted")


def _document_metric_qualifiers(document: dict) -> dict[str, tuple[str, ...]]:
    """Propagate an unambiguous metric variant from an artifact caption.

    Table captions often define a global schema such as ``F1 (Macro)`` while
    the individual columns are labelled only ``LR.F1``/``Within.F1``.  The
    row parser correctly preserves the per-column condition, but without the
    caption qualifier a generated ``Macro F1`` claim cannot match the same
    value.  Only one unique variant per metric is propagated; captions that
    mention both macro and micro variants remain fail-closed.
    """

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    caption = " ".join(
        str(value)
        for value in (document.get("caption"), metadata.get("caption"))
        if value
    )
    if not caption:
        return {}

    variants_by_metric: dict[str, set[str]] = {}
    folded = _fold_for_matching(caption)
    for metric_match in _METRIC_RE.finditer(folded):
        metric = _metric_from_alias(metric_match.group("metric"))
        window = folded[
            max(0, metric_match.start() - 36) : min(len(folded), metric_match.end() + 36)
        ]
        variants = {
            variant
            for variant in _METRIC_VARIANT_WORDS
            if re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", window)
        }
        if variants:
            variants_by_metric.setdefault(metric, set()).update(variants)

    return {
        metric: tuple(sorted(variants))
        for metric, variants in variants_by_metric.items()
        if len(variants) == 1
    }


def _document_percentage_metrics(document: dict) -> set[str]:
    """Read unambiguous ``Metric (%)`` units declared by an artifact caption."""

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    caption = " ".join(
        str(value)
        for value in (document.get("caption"), metadata.get("caption"))
        if value
    )
    folded = _fold_for_matching(caption)
    metrics: set[str] = set()
    for metric_match in _METRIC_RE.finditer(folded):
        suffix = folded[metric_match.end() : metric_match.end() + 24]
        if re.search(r"(?:\(\s*%\s*\)|\bpercent(?:age)?\b)", suffix):
            metrics.add(_metric_from_alias(metric_match.group("metric")))
    return metrics


def _claim_with_global_metric_metadata(
    claim: MetricValueClaim,
    qualifiers_by_metric: dict[str, tuple[str, ...]],
    percentage_metrics: set[str],
) -> MetricValueClaim:
    qualifiers = list(claim.qualifiers)
    for qualifier in qualifiers_by_metric.get(claim.metric, ()):
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)
    percentage = bool(claim.percentage or claim.metric in percentage_metrics)
    if tuple(qualifiers) == claim.qualifiers and percentage == claim.percentage:
        return claim
    return MetricValueClaim(
        metric=claim.metric,
        value=claim.value,
        percentage=percentage,
        subjects=claim.subjects,
        text=claim.text,
        qualifiers=tuple(qualifiers),
    )


def _entity_matches_any_document(
    documents: list[dict],
    entity: str,
    *,
    require_identity: bool = False,
) -> bool:
    has_identity = any(_document_identity_text(document).strip() for document in documents)
    for document in documents:
        haystack = _document_identity_text(document) if require_identity and has_identity else _document_text(document)
        if _contains_entity(haystack.lower(), entity):
            return True
    return False


def _contains_entity(text: str, entity: str) -> bool:
    normalized = entity.lower().strip()
    if not normalized:
        return False
    for candidate in _entity_aliases(normalized):
        if re.search(r"\w", candidate):
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text):
                return True
        elif candidate in text:
            return True
    return False


def _entity_aliases(normalized_entity: str) -> list[str]:
    # Keep only aliases specific enough to establish paper identity. Short
    # component names such as ES/GS/LES used to create cross-paper false hits.
    aliases = {
        "kst": ["kst", "kolmogorov-smirnov", "kolmogorov smirnov", "emotion primitives", "tc-lstm"],
        "msf-ser": ["msf-ser", "multi-granularity semantic fusion", "fm-moe"],
        "msfser": ["msf-ser", "multi-granularity semantic fusion", "fm-moe"],
        "whiser": ["whiser"],
    }
    return aliases.get(normalized_entity, [normalized_entity])


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "accuracies", "acc"),
    "f1": (
        "macro-f1-score",
        "macro f1 score",
        "micro-f1-score",
        "micro f1 score",
        "macro-f1",
        "macro f1",
        "micro-f1",
        "micro f1",
        "ma-f1",
        "mi-f1",
        "maf1",
        "mif1",
        "f1-score",
        "f1 score",
        "f1",
    ),
    "ccc": (
        "concordance correlation coefficient",
        "ccc-avg",
        "ccc avg",
        "ccc-v",
        "ccc v",
        "ccc-a",
        "ccc a",
        "ccc-d",
        "ccc d",
        "ccc",
    ),
    "uar": ("unweighted average recall", "uar"),
    "wa": ("weighted accuracy", "wa"),
    "ua": ("unweighted accuracy", "ua"),
    "wer": ("word error rate", "wer"),
    "mae": ("mean absolute error", "mae"),
    "mse": ("mean squared error", "mse"),
    "rmse": ("root mean squared error", "rmse"),
    "precision": ("precision",),
    "recall": ("recall",),
    "auc": ("roc-auc", "roc auc", "auc"),
    "map": ("mean average precision", "mAP"),
    "eer": ("equal error rate", "eer"),
    "hamming_loss": ("hamming loss", "hl"),
    "jaccard": ("jaccard index", "jaccard", "jac"),
    "exact_accuracy": ("exact match accuracy", "exact accuracy", "exacc"),
    "binary_accuracy": ("per-label accuracy", "binary accuracy", "biacc"),
    "parameters": ("trainable parameters", "parameter count", "parameters", "params", "tham số"),
}

def _metric_alias_key(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(value).casefold()).strip()


def _metric_alias_pattern(value: str) -> str:
    parts = [part for part in re.split(r"[\s_-]+", str(value)) if part]
    return r"[\s_-]+".join(re.escape(part) for part in parts)


_ALIAS_TO_METRIC = {
    _metric_alias_key(alias): metric
    for metric, aliases in _METRIC_ALIASES.items()
    for alias in aliases
}
_METRIC_PATTERN = "|".join(
    sorted(
        {
            _metric_alias_pattern(alias)
            for aliases in _METRIC_ALIASES.values()
            for alias in aliases
        },
        key=len,
        reverse=True,
    )
)
_METRIC_RE = re.compile(rf"(?<![\w-])(?P<metric>{_METRIC_PATTERN})(?![\w-])", re.IGNORECASE)


def _metric_from_alias(value: str) -> str:
    return _ALIAS_TO_METRIC[_metric_alias_key(value)]
_NUMBER_RE = re.compile(
    # A leading decimal separator must touch its digits.  Allowing whitespace
    # here made punctuation such as `Accuracy, 76.31` parse as the bogus value
    # `0.76`.  OCR-spaced decimals that start with a digit (`0 , 3`) remain
    # supported by the first alternative.
    r"(?<![\w.-])(?P<value>[+-]?(?:\d+(?:\s*[.,]\s*\d+)?|[.,]\d+))"
    r"\s*(?P<magnitude>[kmb])?\s*(?P<percent>%?)(?![%\w-])",
    re.IGNORECASE,
)
_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_GENERIC_METRIC_HEADER_RE = re.compile(
    r"\b(?:metric|metrics|measure|measures|chi\s*so)\b",
    re.IGNORECASE,
)
_GENERIC_VALUE_HEADER_RE = re.compile(
    r"\b(?:result|results|value|values|score|scores|ket\s*qua|gia\s*tri)\b",
    re.IGNORECASE,
)
_MODEL_HEADER_RE = re.compile(
    r"\b(?:model|method|system|approach|paper|mo\s*hinh|phuong\s*phap)\b",
    re.IGNORECASE,
)
_STRUCTURAL_COUNT_SUFFIX_RE = re.compile(
    r"^\s*(?:[-–—,:;/]\s*)?"
    r"(?:(?:top|transformer|encoder|decoder|attention|hidden|metric|main|key|"
    r"primary|fft|chính)\s+)*"
    r"(?:layers?|heads?|epochs?|folds?|stages?|streams?|blocks?|modules?|"
    r"encoders?|decoders?|tokens?|samples?|groups?|parts?|types?|categories?|"
    r"emotions?|classes?|bands?|milliseconds?|seconds?|ms|metrics?|"
    r"contributions?|steps?|reasons?|nhóm|phần|loại|chỉ\s*số|"
    r"cảm\s*xúc|đóng\s*góp|bước|lý\s*do)\b",
    re.IGNORECASE,
)
_PARAMETER_MILLIONS_HEADER_RE = re.compile(
    r"^(?:trainable\s+)?(?:p|params?|parameters?)\s*"
    r"\(\s*m(?:illion)?s?\s*\)\.?$",
    re.IGNORECASE,
)

# These are semantic dimensions commonly abbreviated beside a metric (for
# example ``CCC A/V/D``).  Arbitrary hierarchy labels are not enumerated here:
# Markdown section rows are normalized dynamically by
# ``_canonical_table_qualifier``.  The aliases only bridge prose abbreviations
# to the full section labels extracted from a table.
_METRIC_QUALIFIER_ALIASES: dict[str, tuple[str, ...]] = {
    "arousal": ("arousal", "activation", "aro", "ar", "a"),
    "valence": ("valence", "val", "v"),
    "dominance": ("dominance", "dom", "d"),
}
_QUALIFIER_ALIAS_TO_CANONICAL = {
    alias.casefold(): qualifier
    for qualifier, aliases in _METRIC_QUALIFIER_ALIASES.items()
    for alias in aliases
}
_METRIC_QUALIFIER_PATTERN = "|".join(
    re.escape(alias)
    for alias in sorted(_QUALIFIER_ALIAS_TO_CANONICAL, key=len, reverse=True)
)
_METRIC_QUALIFIER_RE = re.compile(
    rf"(?<!\w)(?P<qualifier>{_METRIC_QUALIFIER_PATTERN})(?!\w)",
    re.IGNORECASE,
)
_GENERIC_TABLE_QUALIFIERS = {
    "metric",
    "metrics",
    "measure",
    "measures",
    "model",
    "method",
    "system",
    "approach",
    "result",
    "results",
    "score",
    "scores",
    "value",
    "values",
}


def _extract_metric_claims(text: str) -> list[MetricValueClaim]:
    if not text or not text.strip():
        return []

    claims = _extract_markdown_table_claims(text)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped and stripped.count("|") >= 2:
            # A prose-oriented table (for example ``Contribution | Meaning``)
            # can contain a complete metric sentence inside one cell. The
            # schema parser intentionally ignores those columns, so inspect
            # each cell independently as well. Numeric result tables remain
            # safe: their metric headers and value cells are separate, and
            # any duplicate inline claim is removed below.
            for cell in _table_cells(stripped):
                normalized_cell = _normalize_jammed_metric_numbers(cell)
                if _METRIC_RE.search(normalized_cell):
                    claims.extend(_extract_line_claims(normalized_cell))
            continue
        claims.extend(_extract_line_claims(stripped))
    return _deduplicate_claims(claims)


def _extract_line_claims(line: str) -> list[MetricValueClaim]:
    line = _normalize_jammed_metric_numbers(line)
    metric_matches = list(_METRIC_RE.finditer(line))
    number_matches = [
        match
        for match in _NUMBER_RE.finditer(line)
        if not _looks_like_year(match)
        and not _looks_like_reference_number(line, match)
        and not _looks_like_model_version_number(line, match)
    ]
    if not number_matches:
        return []

    claims: list[MetricValueClaim] = []
    claimed_number_spans: set[tuple[int, int]] = set()
    if metric_matches:
        if _uses_aligned_slash_metrics(line, metric_matches, number_matches):
            pairs = [
                (metric_matches[index % len(metric_matches)], number_match)
                for index, number_match in enumerate(number_matches)
            ]
        elif len(metric_matches) == len(number_matches):
            pairs = list(zip(metric_matches, number_matches))
        else:
            # Generated prose often repeats a compact heading later beside the
            # actual values ("Acc/F1: ... 75.86 Acc ... 76.31 F1"). Pair from
            # each number outward so an early F1 label cannot steal the Acc
            # value. A single metric still naturally owns ranges/uncertainty.
            pairs = [
                (
                    _metric_match_for_number(
                        line,
                        metric_matches=metric_matches,
                        number_match=number_match,
                    ),
                    number_match,
                )
                for number_match in number_matches
            ]

        for metric_match, number_match in pairs:
            if _span_distance(metric_match.span(), number_match.span()) > 72:
                continue
            metric = _metric_from_alias(metric_match.group("metric"))
            if not _metric_number_is_eligible(
                metric=metric,
                text=line,
                number_match=number_match,
                metric_span=metric_match.span(),
            ):
                continue
            claim = _claim_from_matches(
                metric_match,
                number_match,
                line,
                _comparison_subjects_near_number(line, number_match)
                or _subjects_near_number(line, number_match),
                qualifiers=_metric_qualifiers_near_number(
                    line,
                    metric_match=metric_match,
                    number_match=number_match,
                    number_matches=number_matches,
                ),
            )
            if claim is not None:
                claims.append(claim)
                claimed_number_spans.add(number_match.span())

    # A percentage is itself a bounded performance/value claim even when the
    # author omitted the metric name (e.g. "improves by 4.2%").
    for number_match in number_matches:
        if not number_match.group("percent") or number_match.span() in claimed_number_spans:
            continue
        value = _canonical_number(
            number_match.group("value"),
            number_match.group("magnitude"),
        )
        if value is None:
            continue
        claims.append(
            MetricValueClaim(
                metric="percent",
                value=value,
                percentage=True,
                subjects=_subjects_near_number(line, number_match),
                text=_compact_claim_text(line),
            )
        )
    return claims


def _uses_aligned_slash_metrics(
    line: str,
    metric_matches: list[re.Match[str]],
    number_matches: list[re.Match[str]],
) -> bool:
    """Recognize compact aligned vectors such as `61.7/62.3 ... UA/WA`.

    Generated comparisons often place two metric labels after two or more
    slash-separated value groups.  Nearest-neighbour pairing assigns every
    value to the first metric.  Positional cycling is safe only when both the
    metric labels and at least one complete value group explicitly use `/` and
    the arities align.
    """

    if len(metric_matches) < 2 or len(number_matches) <= len(metric_matches):
        return False
    if len(number_matches) % len(metric_matches) != 0:
        return False
    metric_region = line[metric_matches[0].start() : metric_matches[-1].end()]
    if "/" not in metric_region:
        return False
    for left, right in zip(number_matches, number_matches[1:]):
        if "/" in line[left.end() : right.start()]:
            return True
    return False


def _metric_match_for_number(
    text: str,
    *,
    metric_matches: list[re.Match[str]],
    number_match: re.Match[str],
) -> re.Match[str]:
    """Bind a value vector to its leading metric until the next label.

    ``CCC 0.41/0.49/0.39; F1 0.67`` is one CCC vector followed by a new
    metric. Pure nearest-neighbour binding incorrectly lends the last CCC
    values to F1. Value-first prose remains supported when a following label is
    immediately adjacent and the preceding metric belongs to an earlier
    clause.
    """

    number_end = max(
        (
            number_match.end(group)
            for group in ("value", "magnitude", "percent")
            if number_match.group(group)
        ),
        default=number_match.end(),
    )
    preceding = [match for match in metric_matches if match.end() <= number_match.start()]
    following = [match for match in metric_matches if match.start() >= number_end]
    if not preceding:
        return following[0] if following else metric_matches[0]
    owner = preceding[-1]
    if not following:
        return owner

    next_metric = following[0]
    after_value = text[number_end : next_metric.start()]
    immediate_value_first_label = bool(
        re.fullmatch(r"[\s:*_`=-]*", after_value)
    )
    return next_metric if immediate_value_first_label else owner


def _comparison_subjects_near_number(
    text: str,
    number_match: re.Match[str],
) -> tuple[str, ...]:
    """Bind `from A/B to C/D ... compared with Y` to the right owners.

    The values before the transition belong to the comparator/baseline; the
    values after it belong to the leading target.  This is deliberately
    limited to explicit comparison grammar so ordinary prose keeps the more
    conservative nearest-subject logic.
    """

    from_matches = list(re.finditer(r"(?<!\w)(?:from|từ)(?!\w)", text, re.IGNORECASE))
    if not from_matches:
        return ()
    from_match = max(
        (match for match in from_matches if match.end() <= number_match.start()),
        key=lambda match: match.start(),
        default=None,
    )
    if from_match is None:
        return ()
    transition = re.search(
        r"(?<!\w)(?:to|lên|đến|tới)(?!\w)",
        text[from_match.end() :],
        re.IGNORECASE,
    )
    if transition is None:
        return ()
    transition_start = from_match.end() + transition.start()
    transition_end = from_match.end() + transition.end()
    comparator = re.search(
        r"(?<!\w)(?:so\s+với|compared\s+(?:with|to)|versus|vs\.?)\s+",
        text[transition_end:],
        re.IGNORECASE,
    )
    if comparator is None:
        return ()
    comparator_start = transition_end + comparator.start()
    comparator_end = transition_end + comparator.end()

    if from_match.end() <= number_match.start() < transition_start:
        comparator_subjects = _noncontextual_subject_identifiers(text[comparator_end:])
        return comparator_subjects[:1]
    if transition_end <= number_match.start() < comparator_start:
        target_subjects = _noncontextual_subject_identifiers(text[: from_match.start()])
        return target_subjects[-1:]
    return ()


def _normalize_jammed_metric_numbers(text: str) -> str:
    """Restore a missing boundary between a metric phrase and its value.

    PDF extraction commonly emits ``accuracy of75.86%`` or ``F10.74``.
    Relaxing the global number lookbehind would also treat digits in model
    names as measurements, so repair only a bounded metric+connector+number
    sequence before the normal parser runs.
    """

    # GPT/PDF formatting can render an abbreviation period as a detached
    # leading decimal (``Acc . 75.86``). Convert that separator first so the
    # number parser cannot truncate the claim to ``.75``.
    dotted_separator = re.compile(
        rf"(?<![\w-])(?P<metric>{_METRIC_PATTERN})\s*\.\s+"
        r"(?=(?:[+-]?\d+(?:\s*[.,]\s*\d+)?))",
        re.IGNORECASE,
    )
    text = dotted_separator.sub(lambda match: f"{match.group('metric')}: ", text)

    jammed = re.compile(
        rf"(?<![\w-])(?P<metric>{_METRIC_PATTERN})"
        r"(?P<link>\s*(?:(?:of|is|at|=|:)\s*)?)"
        r"(?P<number>[+-]?(?:\d+(?:\s*[.,]\s*\d+)?|[.,]\s*\d+))",
        re.IGNORECASE,
    )
    return jammed.sub(
        lambda match: (
            f"{match.group('metric')}{match.group('link')} {match.group('number')}"
        ),
        text,
    )


def _extract_markdown_table_claims(text: str) -> list[MetricValueClaim]:
    claims: list[MetricValueClaim] = []
    table_lines: list[str] = []

    def flush() -> None:
        if len(table_lines) < 2:
            table_lines.clear()
            return
        rows = [_table_cells(line) for line in table_lines]
        header_index = _markdown_table_header_index(rows)
        if header_index is None:
            table_lines.clear()
            return
        header = rows[header_index]

        hierarchical_claims = _hierarchical_global_metric_table_claims(
            rows,
            header_index=header_index,
        )
        if hierarchical_claims is not None:
            claims.extend(hierarchical_claims)
            table_lines.clear()
            return

        # Some generated answers use a row-oriented schema such as
        # "Dataset | Metric | Kết quả". In that layout the metric name lives in
        # each data row, and a single result cell may hold aligned Acc/F1 values.
        row_metric_column = next(
            (
                index
                for index, cell in enumerate(header)
                if _GENERIC_METRIC_HEADER_RE.search(_fold_for_matching(cell))
            ),
            None,
        )
        row_value_columns = [
            index
            for index, cell in enumerate(header)
            if _GENERIC_VALUE_HEADER_RE.search(_fold_for_matching(cell))
        ]
        if row_metric_column is not None and row_value_columns:
            for row in rows[header_index + 1 :]:
                if _is_table_separator_row(row) or row_metric_column >= len(row):
                    continue
                metric_cell = row[row_metric_column]
                metric_matches = list(_METRIC_RE.finditer(metric_cell))
                if not metric_matches:
                    continue
                row_text = " | ".join(row)
                subjects = _table_row_subjects(header, row)
                for value_column in row_value_columns:
                    if value_column >= len(row):
                        continue
                    value_cell = row[value_column]
                    number_matches = [
                        match
                        for match in _NUMBER_RE.finditer(value_cell)
                        if not _looks_like_year(match)
                        and not _looks_like_reference_number(value_cell, match)
                    ]
                    claims.extend(
                        _claims_from_table_cells(
                            metric_cell=metric_cell,
                            metric_matches=metric_matches,
                            value_cell=value_cell,
                            number_matches=number_matches,
                            subjects=subjects,
                            row_text=row_text,
                            percentage_context=(
                                "%" in metric_cell
                                or "%" in header[value_column]
                            ),
                        )
                    )
            table_lines.clear()
            return

        metric_columns = _metric_columns_from_header(header)
        for row in rows[header_index + 1 :]:
            if _is_table_separator_row(row):
                continue
            # Docling can serialize two logical tables/sections into one
            # Markdown block and omit a second separator row. Treat any
            # metric-labelled, value-free row as a fresh column schema so the
            # following rows are interpreted against the new metrics.
            repeated_metric_columns = _metric_columns_from_header(row)
            if repeated_metric_columns and not _row_has_data_number(row):
                header = row
                metric_columns = repeated_metric_columns
                continue
            row_text = " | ".join(row)
            row_subjects = _table_row_subjects(header, row)
            for column, metric in metric_columns.items():
                if column >= len(row) or metric is None:
                    continue
                inline_metric_matches = list(_METRIC_RE.finditer(row[column]))
                if inline_metric_matches:
                    # A comparison matrix may declare a broad metric family in
                    # its header, then put a compact multi-metric summary in
                    # one body cell (``CCC-V .63, CCC-A .68`` or ``miF1 .51,
                    # maF1 .30, HL .18``). Parse that cell by its own labels;
                    # applying the first header metric to every value corrupts
                    # both metric identity and the owner in adjacent columns.
                    claims.extend(_extract_line_claims(row[column]))
                    continue
                number_matches = [
                    match
                    for match in _NUMBER_RE.finditer(row[column])
                    if not _looks_like_year(match)
                    and not _looks_like_reference_number(row[column], match)
                ]
                for number_match in number_matches:
                    if not _metric_number_is_eligible(
                        metric=metric,
                        text=row[column],
                        number_match=number_match,
                    ):
                        continue
                    value = _canonical_number(
                        number_match.group("value"),
                        number_match.group("magnitude")
                        or _metric_header_magnitude(header[column], metric),
                    )
                    if value is None:
                        continue
                    header_percent = "%" in header[column]
                    claims.append(
                        MetricValueClaim(
                            metric=metric,
                            value=value,
                            percentage=bool(number_match.group("percent") or header_percent),
                            subjects=(
                                _noncontextual_subject_identifiers(row[column])
                                or row_subjects
                            ),
                            text=_compact_claim_text(f"{row_text} [{header[column]}]"),
                            qualifiers=_metric_qualifiers_from_table_label(
                                header[column],
                                metric=metric,
                            ),
                        )
                    )
        table_lines.clear()

    for line in text.splitlines():
        if "|" in line and line.count("|") >= 2:
            table_lines.append(line.strip())
        else:
            flush()
    flush()
    return claims


def _markdown_table_header_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        if _is_table_separator_row(row) and index > 0:
            return index - 1
    return next(
        (
            index
            for index, row in enumerate(rows)
            if any(_canonical_metric(cell) for cell in row)
            or any(_GENERIC_METRIC_HEADER_RE.search(_fold_for_matching(cell)) for cell in row)
        ),
        None,
    )


def _metric_columns_from_header(row: list[str]) -> dict[int, str]:
    columns: dict[int, str] = {}
    for index, cell in enumerate(row):
        metric = _canonical_metric(cell)
        if metric is not None:
            columns[index] = metric
    return columns


def _hierarchical_global_metric_table_claims(
    rows: list[list[str]],
    *,
    header_index: int,
) -> list[MetricValueClaim] | None:
    """Parse a metric table whose row hierarchy carries the missing schema.

    Docling commonly emits a logical table such as ``CCC | LR | ADA | Within``
    followed by a repeated section row (``Arousal`` in every cell) and then
    model rows.  The single metric in the leading header applies to every
    numeric column.  Activation is deliberately structural: exactly one metric
    header, at least one repeated/single-cell section row, and numeric child
    rows are required.  No document, model, or section name is hard-coded.
    """

    header = rows[header_index]
    metric_columns = _metric_columns_from_header(header)
    if len(metric_columns) != 1:
        return None
    metric_column, metric = next(iter(metric_columns.items()))
    value_columns = [
        index
        for index, cell in enumerate(header)
        if index != metric_column and _canonical_table_qualifier(cell)
    ]
    if not value_columns:
        return None

    active_section: str | None = None
    saw_section = False
    claims: list[MetricValueClaim] = []
    for row in rows[header_index + 1 :]:
        if _is_table_separator_row(row):
            continue
        # A second real metric header starts a new logical schema. The ordinary
        # repeated-header path handles that block; do not smear the first
        # global metric into it.
        repeated_metrics = _metric_columns_from_header(row)
        if repeated_metrics and not _row_has_data_number(row):
            break

        section = _hierarchical_section_qualifier(row)
        if section:
            active_section = section
            saw_section = True
            continue
        if active_section is None or metric_column >= len(row):
            continue

        subjects = _hierarchical_owner_subjects(row[metric_column])
        if not subjects:
            continue
        row_text = " | ".join(row)
        for column in value_columns:
            if column >= len(row):
                continue
            number_matches = [
                match
                for match in _NUMBER_RE.finditer(row[column])
                if not _looks_like_year(match)
                and not _looks_like_reference_number(row[column], match)
            ]
            for number_match in number_matches:
                if not _metric_number_is_eligible(
                    metric=metric,
                    text=row[column],
                    number_match=number_match,
                ):
                    continue
                value = _canonical_number(
                    number_match.group("value"),
                    number_match.group("magnitude")
                    or _metric_header_magnitude(header[metric_column], metric),
                )
                if value is None:
                    continue
                column_qualifier = _canonical_table_qualifier(header[column])
                qualifiers = tuple(
                    qualifier
                    for qualifier in (active_section, column_qualifier)
                    if qualifier
                )
                claims.append(
                    MetricValueClaim(
                        metric=metric,
                        value=value,
                        percentage=bool(
                            number_match.group("percent")
                            or "%" in header[metric_column]
                            or "%" in header[column]
                        ),
                        subjects=subjects,
                        text=_compact_claim_text(
                            f"{row_text} [{header[metric_column]} / "
                            f"{active_section} / {header[column]}]"
                        ),
                        qualifiers=qualifiers,
                    )
                )

    return claims if saw_section and claims else None


def _hierarchical_section_qualifier(row: list[str]) -> str | None:
    nonempty_cells = [
        cell
        for cell in row
        if re.sub(r"[*_`~\s]", "", str(cell)).strip("-:|")
    ]
    labels = [_canonical_table_qualifier(cell) for cell in nonempty_cells]
    if (
        not nonempty_cells
        or any(label is None for label in labels)
        or len(set(labels)) != 1
        or not re.search(r"[a-z]", labels[0] or "")
    ):
        return None
    return labels[0]


def _hierarchical_owner_subjects(value: str) -> tuple[str, ...]:
    subjects = _noncontextual_subject_identifiers(value)
    if subjects:
        return subjects
    # A table's leading child cell is an explicit owner even for lowercase or
    # ordinary-word model labels that the conservative prose subject parser
    # intentionally ignores. Table-to-table validation can therefore remain
    # exact without broadening prose ownership heuristics.
    owner = _canonical_table_qualifier(value)
    return (owner.upper(),) if owner else ()


def _metric_qualifiers_from_table_label(
    value: str,
    *,
    metric: str,
    metric_match: re.Match[str] | None = None,
) -> tuple[str, ...]:
    if metric_match is None:
        metric_match = next(
            (
                match
                for match in _METRIC_RE.finditer(value)
                if _metric_from_alias(match.group("metric")) == metric
            ),
            None,
        )
    if metric_match is None:
        return ()

    left_boundary = max(
        value.rfind(separator, 0, metric_match.start())
        for separator in ("/", "|", ",", ";")
    )
    right_candidates = [
        position
        for separator in ("/", "|", ",", ";")
        if (position := value.find(separator, metric_match.end())) >= 0
    ]
    right_boundary = min(right_candidates) if right_candidates else len(value)
    segment = value[left_boundary + 1 : right_boundary]
    local_start = metric_match.start() - (left_boundary + 1)
    local_end = metric_match.end() - (left_boundary + 1)
    remainder = f"{segment[:local_start]} {segment[local_end:]}"

    qualifiers: list[str] = []
    alias_variant = _metric_variant_from_alias(metric_match.group("metric"))
    if alias_variant:
        qualifiers.append(alias_variant)
    for match in _METRIC_QUALIFIER_RE.finditer(remainder):
        qualifier = _QUALIFIER_ALIAS_TO_CANONICAL[match.group("qualifier").casefold()]
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)
    if qualifiers:
        return tuple(qualifiers)

    qualifier = _canonical_table_qualifier(remainder)
    return (qualifier,) if qualifier else ()


def _canonical_table_qualifier(value: str) -> str | None:
    plain = re.sub(r"<[^>]+>", " ", str(value))
    plain = re.sub(r"[*_`~]", "", plain)
    folded = _fold_for_matching(plain).strip()
    folded = re.sub(r"\[[^\]]*\]", " ", folded)
    folded = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    if not folded:
        return None
    alias = _QUALIFIER_ALIAS_TO_CANONICAL.get(folded)
    if alias:
        return alias
    if folded in _GENERIC_TABLE_QUALIFIERS or not re.search(r"[a-z]", folded):
        return None
    if _canonical_metric(plain) is not None:
        return None
    return folded[:80]


def _row_has_data_number(row: list[str]) -> bool:
    for cell in row:
        for match in _NUMBER_RE.finditer(cell):
            if _looks_like_year(match) or _looks_like_reference_number(cell, match):
                continue
            return True
    return False


def _is_table_separator_row(row: list[str]) -> bool:
    return bool(row) and all(
        _TABLE_SEPARATOR_RE.fullmatch(cell.replace(" ", ""))
        for cell in row
    )


def _table_row_subjects(header: list[str], row: list[str]) -> tuple[str, ...]:
    """Prefer the model/method column over dataset and task identifiers.

    Generated comparison tables often add a Dataset column beside Model. If
    every uppercase token in the row is treated as a joint owner, a valid
    ``ASPIRE / IEMOCAP / 75.86`` claim cannot match the canonical source row
    ``ASPIRE / 75.86``. A labelled model column is the strongest owner; a
    model name embedded in a result header (``Kết quả ASPIRE``) is next.
    """

    for index, header_cell in enumerate(header):
        if index >= len(row) or not _MODEL_HEADER_RE.search(_fold_for_matching(header_cell)):
            continue
        subjects = _noncontextual_subject_identifiers(row[index])
        if subjects:
            return subjects

    header_subjects: list[str] = []
    for header_cell in header:
        folded = _fold_for_matching(header_cell)
        if not _GENERIC_VALUE_HEADER_RE.search(folded):
            continue
        for subject in _subject_identifiers(header_cell):
            if subject not in header_subjects:
                header_subjects.append(subject)
    if header_subjects:
        return tuple(header_subjects)

    return _noncontextual_subject_identifiers(" | ".join(row))


def _claims_from_table_cells(
    *,
    metric_cell: str,
    metric_matches: list[re.Match[str]],
    value_cell: str,
    number_matches: list[re.Match[str]],
    subjects: tuple[str, ...],
    row_text: str,
    percentage_context: bool,
) -> list[MetricValueClaim]:
    if not metric_matches or not number_matches:
        return []
    if len(metric_matches) == len(number_matches):
        pairs = list(zip(metric_matches, number_matches))
    elif len(metric_matches) == 1:
        pairs = [(metric_matches[0], number_match) for number_match in number_matches]
    elif len(number_matches) == 1:
        pairs = [(metric_match, number_matches[0]) for metric_match in metric_matches]
    else:
        # Slash-separated cells normally align by position. If one side is
        # incomplete, validate the pairs that are explicit and let the
        # fail-closed signal protect a future completely unparsed layout.
        pairs = list(zip(metric_matches, number_matches))

    claims: list[MetricValueClaim] = []
    for metric_match, number_match in pairs:
        metric = _metric_from_alias(metric_match.group("metric"))
        if not _metric_number_is_eligible(
            metric=metric,
            text=value_cell,
            number_match=number_match,
        ):
            continue
        value = _canonical_number(
            number_match.group("value"),
            number_match.group("magnitude")
            or _metric_header_magnitude(metric_cell, metric),
        )
        if value is None:
            continue
        claims.append(
            MetricValueClaim(
                metric=metric,
                value=value,
                percentage=bool(number_match.group("percent") or percentage_context),
                subjects=subjects,
                text=_compact_claim_text(f"{row_text} [{metric_cell}]"),
                qualifiers=_metric_qualifiers_from_table_label(
                    metric_cell,
                    metric=metric,
                    metric_match=metric_match,
                ),
            )
        )
    return claims


def _table_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _canonical_metric(value: str) -> str | None:
    if _PARAMETER_MILLIONS_HEADER_RE.fullmatch(_fold_for_matching(value).strip()):
        return "parameters"
    match = _METRIC_RE.search(value)
    return _metric_from_alias(match.group("metric")) if match else None


def _metric_header_magnitude(value: str, metric: str) -> str | None:
    if metric != "parameters":
        return None
    if _PARAMETER_MILLIONS_HEADER_RE.fullmatch(_fold_for_matching(value).strip()):
        return "m"
    return None


def _fold_for_matching(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _metric_variant_from_alias(value: str) -> str | None:
    """Recover a semantic qualifier encoded inside a compact metric alias.

    Scientific tables commonly abbreviate micro-F1 and macro-F1 as ``miF1``
    and ``maF1``. They belong to the F1 family, but must remain distinct so a
    value from one variant cannot support a claim about the other.
    """

    compact = re.sub(r"[^a-z0-9]+", "", _fold_for_matching(value))
    if compact in {"mif1", "microf1", "microf1score"}:
        return "micro"
    if compact in {"maf1", "macrof1", "macrof1score"}:
        return "macro"
    if compact == "cccv":
        return "valence"
    if compact == "ccca":
        return "arousal"
    if compact == "cccd":
        return "dominance"
    if compact == "cccavg":
        return "avg"
    return None


def _metric_number_is_eligible(
    *,
    metric: str,
    text: str,
    number_match: re.Match[str],
    metric_span: tuple[int, int] | None = None,
) -> bool:
    prefix = text[max(0, number_match.start() - 48) : number_match.start()]
    if re.search(
        r"(?:(?:decision\s+)?threshold|ngưỡng)\s*(?:[a-z]\s*)?[:=]?\s*$|"
        r"(?<!\w)t\s*=\s*$",
        _fold_for_matching(prefix),
        re.IGNORECASE,
    ):
        # A nearby metric name often describes how checkpoint selection was
        # performed, while the following value belongs to a decision threshold
        # (``best val miF1 ... threshold t = 0.45``). Do not turn that
        # hyperparameter into a reported F1 score.
        return False
    suffix = text[number_match.end() : number_match.end() + 64]
    if _STRUCTURAL_COUNT_SUFFIX_RE.search(suffix):
        return False
    # Parameter language frequently appears in the conclusion of an unrelated
    # architectural sentence ("top 12 layers ... fewer parameters"). Unlike a
    # table's explicit Metric/Result cells, prose needs close lexical binding.
    if (
        metric == "parameters"
        and metric_span is not None
        and _span_distance(metric_span, number_match.span()) > 40
    ):
        return False
    return True


def _unparsed_metric_value_signals(text: str) -> list[dict[str, Any]]:
    """Detect an obvious metric + value layout that the bounded parser missed.

    Blocks, rather than the entire answer, are inspected so an unrelated number
    several paragraphs later does not poison a qualitative metric discussion.
    This also covers fragmented Markdown such as a metric label on one line and
    its value on the next line.
    """

    signals: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text):
        metric_matches = list(_METRIC_RE.finditer(block))
        if not metric_matches:
            continue
        number_matches = [
            match
            for match in _NUMBER_RE.finditer(block)
            if not _looks_like_year(match)
            and not _looks_like_reference_number(block, match)
            and not _looks_like_model_version_number(block, match)
        ]
        for metric_match in metric_matches:
            metric = _metric_from_alias(metric_match.group("metric"))
            for number_match in number_matches:
                distance = _span_distance(metric_match.span(), number_match.span())
                if distance > 72 or not _metric_number_is_eligible(
                    metric=metric,
                    text=block,
                    number_match=number_match,
                    metric_span=metric_match.span(),
                ):
                    continue
                value = _canonical_number(
                    number_match.group("value"),
                    number_match.group("magnitude"),
                )
                signals.append(
                    {
                        "metric": metric,
                        "value": value,
                        "distance": distance,
                        "text": _compact_claim_text(block),
                    }
                )
                if len(signals) >= 8:
                    return signals
    return signals


def _has_unparsed_metric_value_signal(text: str) -> bool:
    return bool(_unparsed_metric_value_signals(text))


def _metric_qualifiers_near_number(
    text: str,
    *,
    metric_match: re.Match[str],
    number_match: re.Match[str],
    number_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    """Extract an explicit dimension near a prose metric/value claim.

    Full table section labels are discovered dynamically.  This helper only
    needs to bridge compact prose such as ``CCC arousal 0.418`` and
    ``CCC A/V/D 0.418/0.483/0.387`` back to those labels. Qualifiers may be
    prefix or postfix (``0.418 for arousal``), while parenthesized labels remain
    a general, schema-free form (``Accuracy (Outdoor)``).
    """

    following_metric = next(_METRIC_RE.finditer(text, metric_match.end()), None)
    metric_end = following_metric.start() if following_metric is not None else len(text)
    local_numbers = [
        match
        for match in number_matches
        if metric_match.end() <= match.start() < metric_end
    ]
    if number_match not in local_numbers:
        return ()
    number_index = local_numbers.index(number_match)

    # Compact qualifier vectors are the one case where a prefix intentionally
    # owns several later values. Bind them only when arity is exact; a single
    # qualifier can safely describe every value in the same metric region.
    vector_context = text[metric_match.end() : local_numbers[0].start()]
    vector_qualifiers = _prose_qualifiers_from_context(vector_context)
    vector_qualifier: tuple[str, ...] = ()
    if len(vector_qualifiers) == len(local_numbers) and len(local_numbers) > 1:
        vector_qualifier = (vector_qualifiers[number_index],)
    elif len(vector_qualifiers) == 1:
        vector_qualifier = vector_qualifiers

    prefix_start = metric_match.end()
    if number_index > 0:
        prefix_start = local_numbers[number_index - 1].end()
    prefix_context = text[prefix_start : number_match.start()]
    if number_index > 0:
        separators = list(
            re.finditer(
                r"[,;|/]|(?<!\w)(?:and|while|versus|vs\.?|và|còn)(?!\w)",
                prefix_context,
                re.IGNORECASE,
            )
        )
        # Without a connector this span normally describes the previous value
        # (``0.418 arousal 0.483``), so do not let it leak to the current one.
        prefix_context = (
            prefix_context[separators[-1].end() :] if separators else ""
        )
    explicit_prefix_qualifiers = _prose_qualifiers_from_context(prefix_context)
    if explicit_prefix_qualifiers:
        # Interleaved vectors such as ``CCC V=.63, A=.68, D=.60`` put a new
        # qualifier before each later value. That local label overrides the
        # first prefix; otherwise ``V`` would incorrectly leak to A and D.
        prefix_qualifiers = explicit_prefix_qualifiers
        vector_qualifier = ()
    else:
        prefix_qualifiers = ()

    suffix_end = metric_end
    if number_index + 1 < len(local_numbers):
        suffix_end = min(suffix_end, local_numbers[number_index + 1].start())
    suffix_context = text[number_match.end() : suffix_end]
    if number_index + 1 < len(local_numbers):
        # In ``0.55 on Outdoor scenes and 0.61 on Indoor scenes``, the
        # conjunction separates claims; it is not part of the first dynamic
        # section label.  Only trim it when another value really follows so a
        # legitimate standalone label containing ``and`` remains untouched.
        suffix_context = re.sub(
            r"\s+(?:and|while|versus|vs\.?|và|còn)\s*$",
            "",
            suffix_context,
            flags=re.IGNORECASE,
        )
    suffix_delimiter = re.search(r"[,;|/]|[.!?](?=\s|$)", suffix_context)
    if suffix_delimiter is not None:
        suffix_context = suffix_context[: suffix_delimiter.start()]
    suffix_qualifiers = _prose_qualifiers_from_context(suffix_context)

    result: list[str] = []
    alias_variant = _metric_variant_from_alias(metric_match.group("metric"))
    if alias_variant and (number_index == 0 or not explicit_prefix_qualifiers):
        result.append(alias_variant)
    for qualifier in (*vector_qualifier, *prefix_qualifiers, *suffix_qualifiers):
        if qualifier not in result:
            result.append(qualifier)
    return tuple(result)


def _prose_qualifiers_from_context(context: str) -> tuple[str, ...]:
    qualifiers: list[str] = []
    folded = _fold_for_matching(context)
    compact_single_letter_context = bool(
        re.fullmatch(
            r"\s*(?:(?:for|on|in|under|dimension|dim|tren|o)\s+)?"
            r"[avd](?:\s*/\s*[avd])*"
            r"\s*(?:(?:score|value|is|at|equals?|la|dat)\s*)?[:=.-]?\s*",
            folded,
            re.IGNORECASE,
        )
    )
    for match in _METRIC_QUALIFIER_RE.finditer(context):
        raw = match.group("qualifier").casefold()
        if len(raw) == 1 and not compact_single_letter_context:
            continue
        qualifier = _QUALIFIER_ALIAS_TO_CANONICAL[raw]
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)

    assignment = re.fullmatch(
        r"\s*(?P<label>[a-z][a-z0-9 _-]{0,40})\s*[:=]\s*",
        folded,
    )
    if assignment is not None:
        qualifier = _canonical_table_qualifier(assignment.group("label"))
        if qualifier and qualifier not in qualifiers:
            qualifiers.append(qualifier)

    groups = re.findall(r"\(([^)]{1,80})\)|\[([^\]]{1,80})\]", context)
    for parenthesized, bracketed in groups:
        raw_group = parenthesized or bracketed
        for raw_part in re.split(r"[,;/]", raw_group):
            qualifier = _canonical_table_qualifier(raw_part)
            if qualifier and qualifier not in qualifiers:
                qualifiers.append(qualifier)

    connector_re = re.compile(
        r"(?<!\w)(?:for|on|in|under|trên|ở|cho)(?!\w)\s+"
        r"(?P<label>[^,;|/()[\]]{1,64})",
        re.IGNORECASE,
    )
    for match in connector_re.finditer(context):
        raw_label = re.sub(
            r"\s+(?:score|value|is|at|equals?|là|đạt)\s*$",
            "",
            match.group("label"),
            flags=re.IGNORECASE,
        )
        if _is_composite_prose_qualifier_label(raw_label):
            # A qualifier tuple represents one conjunctive schema path (for
            # example ``outdoor-scenes`` + ``fine-tuned``). Phrases such as
            # ``both Test1 and Test2`` denote several alternative slices and
            # must not be collapsed into a fabricated single table label.
            continue
        qualifier = _canonical_table_qualifier(raw_label)
        if qualifier and qualifier not in qualifiers:
            qualifiers.append(qualifier)
    return tuple(qualifiers)


def _is_composite_prose_qualifier_label(value: str) -> bool:
    folded = " ".join(_fold_for_matching(value).split())
    if re.match(
        r"^(?:both|all|two|multiple|several|ca|hai|tat\s+ca|nhieu)(?:\s|$)",
        folded,
    ):
        return True
    return bool(
        re.search(
            r"(?<!\w)(?:and|or|versus|vs\.?|va|hoac)(?!\w)",
            folded,
        )
    )


def _claim_from_matches(
    metric_match: re.Match[str],
    number_match: re.Match[str],
    line: str,
    subjects: tuple[str, ...],
    *,
    qualifiers: tuple[str, ...] = (),
) -> MetricValueClaim | None:
    value = _canonical_number(
        number_match.group("value"),
        number_match.group("magnitude"),
    )
    if value is None:
        return None
    return MetricValueClaim(
        metric=_metric_from_alias(metric_match.group("metric")),
        value=value,
        percentage=bool(number_match.group("percent")),
        subjects=subjects,
        text=_compact_claim_text(line),
        qualifiers=qualifiers,
    )


def _canonical_number(raw: str, magnitude: str | None = None) -> str | None:
    # Docling occasionally emits OCR/table decimals as ``21 . 7`` or
    # ``0 , 3``. Whitespace inside a numeric token is layout noise, not a
    # thousands separator, so normalize it before Decimal parsing.
    normalized = re.sub(r"\s+", "", raw).replace(",", ".")
    if normalized.startswith("."):
        normalized = f"0{normalized}"
    if normalized.startswith("+.") or normalized.startswith("-."):
        normalized = f"{normalized[:1]}0{normalized[1:]}"
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    multiplier = {
        "k": Decimal(1_000),
        "m": Decimal(1_000_000),
        "b": Decimal(1_000_000_000),
    }.get((magnitude or "").lower(), Decimal(1))
    value *= multiplier
    result = format(value.normalize(), "f")
    return "0" if result in {"-0", "+0"} else result


def _claim_is_supported(
    answer_claim: MetricValueClaim,
    evidence_claim: MetricValueClaim,
    *,
    evidence_identity_subjects: frozenset[str] = frozenset(),
) -> bool:
    if not _claim_measurement_matches(answer_claim, evidence_claim):
        return False
    answer_subjects = set(answer_claim.subjects)
    evidence_subjects = set(evidence_claim.subjects)
    if _PROPOSED_OWNER_SUBJECT in answer_subjects:
        explicit_subjects = answer_subjects - {
            _PROPOSED_OWNER_SUBJECT,
            *evidence_identity_subjects,
        }
        if not explicit_subjects.issubset(evidence_subjects):
            return False
        if (
            _PROPOSED_OWNER_SUBJECT not in evidence_subjects
            and not evidence_subjects.intersection(evidence_identity_subjects)
        ):
            return False
    elif answer_subjects:
        # A comparison row commonly names both the enclosing paper and the
        # model that owns the metric (``Paper A — Model X — F1 0.80``).  Strip
        # only identities belonging to this exact evidence document, then bind
        # every remaining subject to the evidence row owner.  If the answer
        # names only the paper, the row itself must represent that paper/its
        # proposed system; this prevents ``Paper A`` from borrowing a baseline
        # value merely because the baseline is printed inside Paper A.
        owner_subjects = answer_subjects - evidence_identity_subjects
        if owner_subjects:
            if not owner_subjects.issubset(evidence_subjects):
                return False
        elif not (
            evidence_subjects.intersection(evidence_identity_subjects)
            or _PROPOSED_OWNER_SUBJECT in evidence_subjects
        ):
            return False
    if not answer_claim.subjects:
        # A genuinely owner-free answer may use evidence for the proposed
        # system, but never a baseline row that merely happens to contain the
        # same value. Preserve the legacy active-document identity fallback for
        # named model rows and unlabelled prose.
        if _BASELINE_OWNER_SUBJECT in evidence_subjects:
            return False
        if _PROPOSED_OWNER_SUBJECT in evidence_subjects:
            return True
        if evidence_identity_subjects:
            return bool(evidence_subjects & evidence_identity_subjects)
        return not evidence_claim.subjects
    return True


def _claim_measurement_matches(
    answer_claim: MetricValueClaim,
    evidence_claim: MetricValueClaim,
) -> bool:
    if answer_claim.metric != evidence_claim.metric:
        return False
    if not _values_equivalent(answer_claim, evidence_claim):
        return False
    return not (
        answer_claim.qualifiers
        and evidence_claim.qualifiers
        and not set(answer_claim.qualifiers).issubset(evidence_claim.qualifiers)
    )


def _values_equivalent(left: MetricValueClaim, right: MetricValueClaim) -> bool:
    try:
        left_value = Decimal(left.value)
        right_value = Decimal(right.value)
    except InvalidOperation:
        return False
    if left_value == right_value:
        return True
    if left.percentage != right.percentage:
        return left_value == right_value * Decimal(100) or right_value == left_value * Decimal(100)
    return False


def _claim_with_identity_subjects(claim: MetricValueClaim, identity: str) -> MetricValueClaim:
    # An explicit name beside a value wins over the enclosing filename. This
    # stops a related-work sentence such as "Wav2Small: 91.7%" inside
    # ASPIRE.pdf from becoming evidence for "ASPIRE: 91.7%". If the excerpt
    # uses only "the proposed model", fall back to the canonical file identity.
    subjects = claim.subjects or tuple(_subject_identifiers(identity))
    return MetricValueClaim(
        metric=claim.metric,
        value=claim.value,
        percentage=claim.percentage,
        subjects=subjects,
        text=claim.text,
        qualifiers=claim.qualifiers,
    )


_SUBJECT_RE = re.compile(
    r"(?<!\w)(?:"
    r"[A-Z][A-Z0-9]*(?:[-_][A-Za-z0-9]+)*|"
    r"[A-Za-z]+\d+[A-Za-z0-9._-]*|"
    # Mixed-case acronym/model tokens can start lowercase (wavLM) or use an
    # unusual acronym casing (WHiSER). Requiring two uppercase characters
    # avoids promoting ordinary sentence words to owners.
    r"[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9-]*|"
    # Model names are frequently sentence-cased after a hyphen
    # (Pitch-fusion, Cross-attention), not title-cased on every segment.
    r"[A-Z][a-z]+(?:-[A-Za-z][A-Za-z0-9]+)+"
    r")(?!\w)"
)
_PROPOSED_OWNER_SUBJECT = "PROPOSED-MODEL"
_BASELINE_OWNER_SUBJECT = "BASELINE"
_OWNER_ROLE_RE = re.compile(
    r"(?P<proposed>"
    r"(?<!\w)(?:"
    r"the\s+proposed\s+(?:model|method|system|approach)|"
    r"(?:our|proposed|full|complete)\s+(?:model|method|system|approach)|"
    r"ours|proposed|"
    r"(?:mô\s+hình|phương\s+pháp|hệ\s+thống)\s+đề\s+xuất"
    r")(?!\w)"
    r")|"
    r"(?P<baseline>"
    r"(?<!\w)(?:"
    r"baselines?|"
    r"(?:base|reference)\s+(?:model|method|system|approach)|"
    r"mô\s+hình\s+cơ\s+sở|đường\s+cơ\s+sở"
    r")(?!\w)"
    r")",
    re.IGNORECASE,
)
_BLOCKED_SUBJECTS = {
    "ACC",
    "ACCURACY",
    "F1",
    "MIF1",
    "MAF1",
    "CCC",
    "UAR",
    "WA",
    "UA",
    "WER",
    "MAE",
    "MSE",
    "RMSE",
    "AUC",
    "MAP",
    "EER",
    "HL",
    "JAC",
    "JACCARD",
    "EXACC",
    "BIACC",
    "PARAMETERS",
    "PARAMS",
    "FIG",
    "FIGURE",
    "TABLE",
    "SOURCE",
    "SOTA",
    "ROC",
    # Task/domain/modality shorthand, not model or paper identities. Keeping
    # this list explicit avoids weakening cross-paper guards for real acronyms.
    "SER",
    "ASR",
    "SSL",
    "AV",
    "AVD",
    "GT",
    "LR",
    "A",
    "T",
}
_SENTENCE_DELIMITER_RE = re.compile(r"[.!?](?=\s|$)")
_CLAUSE_DELIMITER_RE = re.compile(r"[,;|]|[.!?](?=\s|$)")
_CONTEXTUAL_SUBJECT_PREFIX_RE = re.compile(
    r"(?:"
    r"\bon(?:\s+the)?(?:\s+dataset)?|"
    r"\bin|\busing|\bevaluated\s+on|"
    r"\btrên(?:\s+(?:dataset|tập\s+dữ\s+liệu))?|\bở"
    r")\s*$",
    re.IGNORECASE,
)
_FOLLOWING_OWNER_PREFIX_RE = re.compile(
    r"(?:\bcủa|\bof|\bfor|\bby)\s*$",
    re.IGNORECASE,
)
_FOLLOWING_EXAMPLE_PREFIX_RE = re.compile(
    r"(?:^\s*(?:như|such\s+as|e\.g\.?|for\s+example|namely|including|gồm)\b|"
    r"(?:\bnhư|\bsuch\s+as|\be\.g\.?|\bfor\s+example|\bnamely|"
    r"\bincluding|\bgồm)\s*$)",
    re.IGNORECASE,
)


def _subject_occurrences(text: str) -> list[tuple[str, tuple[int, int]]]:
    result: list[tuple[str, tuple[int, int]]] = []
    for match in _SUBJECT_RE.finditer(text):
        canonical = match.group().upper()
        if (
            canonical in _BLOCKED_SUBJECTS
            or re.fullmatch(r"[IVXLCDM]+", canonical)
            or re.fullmatch(
                r"(?:MACRO|MICRO|WEIGHTED|UNWEIGHTED)[-_]?F1(?:[-_]?SCORE)?",
                canonical,
            )
            or re.fullmatch(r"TEST[-_]?\d+", canonical)
            or re.fullmatch(r"CCC[-_]?(?:V|A|D|AVG)", canonical)
        ):
            continue
        result.append((canonical, match.span()))
    for match in _OWNER_ROLE_RE.finditer(text):
        canonical = (
            _PROPOSED_OWNER_SUBJECT
            if match.group("proposed")
            else _BASELINE_OWNER_SUBJECT
        )
        if not any(
            existing_span[0] <= match.start()
            and existing_span[1] >= match.end()
            for _existing, existing_span in result
        ):
            result.append((canonical, match.span()))
    return sorted(result, key=lambda item: (item[1][0], item[1][1]))


def _subject_identifiers(text: str) -> list[str]:
    result: list[str] = []
    for canonical, _span in _subject_occurrences(text):
        if canonical not in result:
            result.append(canonical)
    return result


def _noncontextual_subject_identifiers(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for subject, span in _subject_occurrences(text):
        if _subject_occurrence_is_contextual(text, span):
            continue
        if subject not in result:
            result.append(subject)
    return tuple(result)


def _subjects_near_number(text: str, number_match: re.Match[str]) -> tuple[str, ...]:
    occurrences = [
        item
        for item in _subject_occurrences(text)
        if not _subject_occurrence_is_contextual(text, item[1])
    ]
    if not occurrences:
        return ()

    number_span = number_match.span()
    preceding_delimiters = [
        match.end()
        for match in _CLAUSE_DELIMITER_RE.finditer(text, 0, number_span[0])
    ]
    has_sentence_boundary = bool(
        _SENTENCE_DELIMITER_RE.search(text, 0, number_span[0])
    )
    following_delimiter = _CLAUSE_DELIMITER_RE.search(text, number_span[1])
    clause_start = preceding_delimiters[-1] if preceding_delimiters else 0
    clause_end = following_delimiter.start() if following_delimiter else len(text)
    local = [
        item
        for item in occurrences
        if item[1][0] >= clause_start and item[1][1] <= clause_end
    ]

    def closest_preceding(pool: list[tuple[str, tuple[int, int]]]) -> str | None:
        candidates = [item for item in pool if item[1][1] <= number_span[0]]
        if not candidates:
            return None
        return min(candidates, key=lambda item: number_span[0] - item[1][1])[0]

    def closest_following_owner(pool: list[tuple[str, tuple[int, int]]]) -> str | None:
        candidates: list[tuple[str, tuple[int, int]]] = []
        for item in pool:
            if item[1][0] < number_span[1]:
                continue
            between = text[number_span[1] : item[1][0]]
            # Ownership must bind this value, not a later value in the same
            # sentence (6.35M ... 21.7M của AFEA-Net).
            if not _NUMBER_RE.search(between) and _FOLLOWING_OWNER_PREFIX_RE.search(between):
                candidates.append(item)
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1][0] - number_span[1])[0]

    def closest_following_example(pool: list[tuple[str, tuple[int, int]]]) -> str | None:
        candidates: list[tuple[str, tuple[int, int]]] = []
        for item in pool:
            if item[1][0] < number_span[1]:
                continue
            between = text[number_span[1] : item[1][0]]
            if not _NUMBER_RE.search(between) and _FOLLOWING_EXAMPLE_PREFIX_RE.search(between):
                candidates.append(item)
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1][0] - number_span[1])[0]

    selected = (
        # Explicit postposed ownership beats an earlier discourse subject:
        # "ASPIRE ... 21.7M của AFEA-Net" belongs to AFEA-Net.
        closest_following_owner(local)
        or closest_following_owner(occurrences)
        # A same-clause exemplar owns the comparison value: "0.717 such as
        # PCM". Restrict this to the local clause so "75.86, higher than
        # baselines such as AFEA-Net" keeps 75.86 attached to the target.
        or closest_following_example(local)
        or closest_preceding(local)
        # Do not let a model named in a previous sentence steal a later value.
        # A subjectless later sentence can still be grounded conservatively by
        # the active document identity in `_claim_is_supported`.
        or (None if has_sentence_boundary else closest_preceding(occurrences))
    )
    return (selected,) if selected else ()


def _subject_occurrence_is_contextual(text: str, span: tuple[int, int]) -> bool:
    prefix = text[max(0, span[0] - 80) : span[0]]
    return bool(_CONTEXTUAL_SUBJECT_PREFIX_RE.search(prefix))


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def _looks_like_year(match: re.Match[str]) -> bool:
    if match.group("percent") or match.group("magnitude"):
        return False
    try:
        value = Decimal(match.group("value").replace(",", "."))
    except InvalidOperation:
        return False
    return value == value.to_integral_value() and Decimal(1900) <= value <= Decimal(2100)


def _looks_like_reference_number(line: str, match: re.Match[str]) -> bool:
    prefix = line[: match.start()]
    suffix = line[match.end() :]
    if not prefix.strip() and re.match(r"\s*[.)]\s+", suffix):
        return True
    if re.search(r"\[\s*$", prefix):
        return True
    if _looks_like_numeric_list(prefix, match.group("value"), suffix):
        return True
    return bool(
        re.search(
            r"(?:figure|fig\.?|table|page|hình|bảng|trang|section|sec\.?)"
            r"\s*(?:[*_`~]+\s*)*$",
            prefix,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_model_version_number(line: str, match: re.Match[str]) -> bool:
    """Exclude version suffixes such as `Wav2Vec 2.0` from metric values.

    The global number regex cannot reject every decimal following a word,
    because `ASPIRE 75.86 Acc` is a valid result layout.  A preceding mixed
    alphanumeric model token is the bounded distinction needed for names such
    as Wav2Vec/Qwen3, while explicit `version`/`ver` markers cover ordinary
    family names.
    """

    if match.group("percent") or match.group("magnitude"):
        return False
    prefix = line[: match.start()]
    token_match = re.search(r"([A-Za-z][A-Za-z0-9._-]{1,40})\s*$", prefix)
    if token_match is None:
        return bool(re.search(r"(?:\bversion|\bver\.?)\s*$", prefix, re.IGNORECASE))
    token = token_match.group(1)
    if _METRIC_RE.fullmatch(token):
        return False
    if any(character.isdigit() for character in token):
        return True
    return bool(re.fullmatch(r"(?:gpt|bert|roberta|hubert|wavlm|whisper|resnet)", token, re.IGNORECASE))


def _looks_like_numeric_list(prefix: str, raw_value: str, suffix: str) -> bool:
    # Decimal whitespace is accepted, but a visibly spaced separator can also
    # be list punctuation. Reject clear enumerations/references while keeping
    # isolated OCR decimals such as a table cell containing ``≈ 21 . 7``.
    if re.search(r"\d\s*[.,;]\s*\d\s*[.,;]\s*$", prefix):
        return True
    if not re.search(r"\d(?:\s+[.,]\s*|\s*[.,]\s+)\d", raw_value):
        return False
    if re.search(
        r"(?:items?|steps?|points?|references?|refs?|entries)\s*$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(r"\s*[.,;]\s*\d", suffix):
        return True
    return False


def _compact_claim_text(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 1].rstrip()}…"


def _deduplicate_claims(claims: list[MetricValueClaim]) -> list[MetricValueClaim]:
    result: list[MetricValueClaim] = []
    seen: set[tuple[str, str, bool, tuple[str, ...], tuple[str, ...], str]] = set()
    for claim in claims:
        key = (
            claim.metric,
            claim.value,
            claim.percentage,
            claim.subjects,
            claim.qualifiers,
            claim.text,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(claim)
    return result
