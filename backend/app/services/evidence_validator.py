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
    claim_text_by_document: dict[str, list[str]] = {}
    for document in eligible_documents:
        document_id = str(document.get("document_id") or "").strip()
        if document_id:
            claim_text_by_document.setdefault(document_id, []).append(
                _document_claim_text(document)
            )
    metric_qualifiers_by_document = {
        document_id: _document_metric_qualifiers(
            {"caption": "\n".join(part for part in parts if part)}
        )
        for document_id, parts in claim_text_by_document.items()
    }
    for document in eligible_documents:
        document_id = str(document.get("document_id") or "").strip()
        identity = _document_identity_text(document)
        identity_subjects = frozenset(_identity_subject_identifiers(identity))
        global_metric_qualifiers = metric_qualifiers_by_document.get(document_id, {})
        global_context_qualifiers = _document_global_context_qualifiers(document)
        global_percentage_metrics = _document_percentage_metrics(document)
        for claim in _extract_metric_claims(
            _document_claim_text(document),
            join_soft_wrapped=True,
        ):
            claim = _claim_with_global_metric_metadata(
                claim,
                global_metric_qualifiers,
                global_percentage_metrics,
                global_context_qualifiers,
            )
            evidence_claims.append(
                (
                    _claim_with_identity_subjects(claim, identity),
                    identity_subjects,
                    document_id,
                )
            )

    claims = [
        _resolve_unlabeled_metric_claim(claim, evidence_claims)
        for claim in claims
    ]

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
            # Markdown cell, or use only the paper/dataset as shorthand. Bind
            # such a claim only when metric, value, units and conditions point
            # to exactly one evidence owner in one focused document. A real
            # model subject that disagrees with the row is never discarded;
            # repeated values/owners remain ambiguous and fail closed.
            ownerless_candidates: set[
                tuple[str, tuple[str, ...], tuple[str, ...]]
            ] = set()
            for evidence_claim, identity_subjects, document_id in candidate_evidence:
                if not _claim_measurement_matches(claim, evidence_claim):
                    continue
                ownerless_candidates.add(
                    (
                        document_id,
                        evidence_claim.subjects,
                        evidence_claim.qualifiers,
                    )
                )
            if len(ownerless_candidates) == 1:
                unique_document_id, _subjects, _qualifiers = next(
                    iter(ownerless_candidates)
                )
                if unique_document_id in identity_document_ids:
                    supporting_document_ids = {unique_document_id}
        if not supporting_document_ids and claim.subjects:
            # A row condition can be rendered in prose where the conservative
            # subject lexer sees it as a model-like token (``Audio-text: CCC
            # ...``). Reclassify it only when every non-identity subject maps
            # to evidence qualifiers and the resulting measurement has one
            # unique owner. A paper/model identity alone never enters this
            # path, preserving the cross-assignment guard above.
            context_only_candidates: set[
                tuple[str, tuple[str, ...], tuple[str, ...]]
            ] = set()
            for evidence_claim, identity_subjects, document_id in candidate_evidence:
                if not _claim_measurement_matches(claim, evidence_claim):
                    continue
                non_identity_subjects = set(claim.subjects) - set(identity_subjects)
                if not non_identity_subjects or not all(
                    _subject_is_measurement_context(
                        subject,
                        evidence_claim.qualifiers,
                    )
                    for subject in non_identity_subjects
                ):
                    continue
                context_only_candidates.add(
                    (
                        document_id,
                        evidence_claim.subjects,
                        evidence_claim.qualifiers,
                    )
                )
            if len(context_only_candidates) == 1:
                unique_document_id, _subjects, _qualifiers = next(
                    iter(context_only_candidates)
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
    if not claims:
        return claims

    identity_by_document_id = {
        str(document.get("document_id") or "").strip(): _document_identity_text(document)
        for document in documents
        if document.get("document_id")
    }
    owner_subjects_by_document_id: dict[str, set[str]] = {}
    qualifiers_by_document_id: dict[str, set[str]] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            continue
        identity_subjects = set(
            _identity_subject_identifiers(identity_by_document_id.get(document_id, ""))
        )
        owners: set[str] = set()
        qualifiers: set[str] = set()
        for evidence_claim in _extract_metric_claims(_document_claim_text(document)):
            owners.update(set(evidence_claim.subjects) - identity_subjects)
            qualifiers.update(evidence_claim.qualifiers)
        owner_subjects_by_document_id[document_id] = owners
        qualifiers_by_document_id[document_id] = qualifiers

    contexts_by_signature: dict[
        tuple[Any, ...],
        set[tuple[str, tuple[str, ...], tuple[str, ...], bool]],
    ] = {}
    active_document_id: str | None = (
        next(iter(focus_ids)) if len(focus_ids) == 1 else None
    )
    active_section_subjects: tuple[str, ...] = ()
    active_section_qualifiers: tuple[str, ...] = ()
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        line_document_ids = _answer_identity_document_ids(
            stripped,
            documents,
            focus_ids=focus_ids,
        )
        line_document_id = (
            next(iter(line_document_ids)) if len(line_document_ids) == 1 else None
        )
        if line_document_id and _is_document_section_label(stripped):
            active_document_id = line_document_id
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
            active_section_qualifiers = ()
        elif len(line_document_ids) > 1:
            active_document_id = None
            active_section_subjects = ()
            active_section_qualifiers = ()

        # A metric line may name a different paper without being a Markdown
        # heading (``Paper B: F1 ...``). Bind only that line to the explicit
        # identity; an incidental prose mention must not permanently switch
        # the following section.
        context_document_id = line_document_id or active_document_id
        if not context_document_id:
            continue
        context_subjects = (
            active_section_subjects
            if context_document_id == active_document_id
            else ()
        )
        context_qualifiers = (
            active_section_qualifiers
            if context_document_id == active_document_id
            else ()
        )
        if _is_local_owner_section_label(stripped):
            allowed_owners = owner_subjects_by_document_id.get(
                context_document_id,
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
            known_qualifiers = qualifiers_by_document_id.get(
                context_document_id,
                set(),
            )
            local_qualifier = _structural_answer_qualifier(
                stripped,
                known_qualifiers=known_qualifiers,
            )
            local_qualifiers = (local_qualifier,) if local_qualifier else ()
            if context_document_id == active_document_id:
                active_section_subjects = local_owners
                active_section_qualifiers = local_qualifiers
            context_subjects = local_owners
            context_qualifiers = local_qualifiers
        line_claims = _extract_line_claims(stripped)
        allowed_owners = owner_subjects_by_document_id.get(
            context_document_id,
            set(),
        )
        leading_line_owners = _leading_metric_bullet_owners(
            stripped,
            allowed_owners=allowed_owners,
        )
        leading_sentence_owner = _unique_prevalue_line_owner(
            stripped,
            allowed_owners=allowed_owners,
            known_qualifiers=qualifiers_by_document_id.get(
                context_document_id,
                set(),
            ),
        )
        for line_claim in line_claims:
            replace_claim_subjects = False
            claim_owners = tuple(
                subject
                for subject in line_claim.subjects
                if subject in allowed_owners
            )
            claim_context_subjects = context_subjects
            if claim_owners:
                # An owner written on the metric line is stronger than a
                # structural owner carried from a preceding bullet/heading.
                # This prevents ``Fusion-Gate`` from leaking into the next
                # explicit ``Fusion-Stack: miF1 ...`` row while retaining
                # heading context for genuinely subjectless continuation rows.
                claim_context_subjects = tuple(
                    subject for subject in context_subjects if subject in claim_owners
                )
            elif len(leading_line_owners) == 1 and not claim_owners:
                # A quantitative bullet labels its own row before the metric.
                # Apply that owner to the same line immediately; carrying the
                # previous bullet until after parsing swaps adjacent rows.
                claim_context_subjects = leading_line_owners
                replace_claim_subjects = True
            elif leading_sentence_owner and not claim_owners:
                # One explicit evidence owner on a metric sentence owns its
                # later coordinated or parenthetically annotated measurements.
                # Multiple owners remain ambiguous and never inherit across
                # comparison rows.
                claim_context_subjects = (leading_sentence_owner,)
                replace_claim_subjects = True
            contexts_by_signature.setdefault(
                _claim_context_signature(line_claim),
                set(),
            ).add(
                (
                    context_document_id,
                    claim_context_subjects,
                    context_qualifiers,
                    replace_claim_subjects,
                )
            )
        if context_document_id == active_document_id and len(leading_line_owners) == 1:
            active_section_subjects = leading_line_owners
            active_section_qualifiers = ()

    contextualized: list[MetricValueClaim] = []
    for claim in claims:
        contexts = contexts_by_signature.get(_claim_context_signature(claim), set())
        if len(contexts) != 1:
            contextualized.append(claim)
            continue
        (
            document_id,
            section_subjects,
            section_qualifiers,
            replace_claim_subjects,
        ) = next(iter(contexts))
        identity_subjects = tuple(
            _identity_subject_identifiers(identity_by_document_id.get(document_id, ""))
        )
        contextualized.append(
            MetricValueClaim(
                metric=claim.metric,
                value=claim.value,
                percentage=claim.percentage,
                subjects=tuple(
                    dict.fromkeys(
                        [
                            *(identity_subjects if len(focus_ids) >= 2 else ()),
                            *section_subjects,
                            *(() if replace_claim_subjects else claim.subjects),
                        ]
                    )
                ),
                text=claim.text,
                qualifiers=tuple(
                    dict.fromkeys([*section_qualifiers, *claim.qualifiers])
                ),
            )
        )
    return contextualized


def _leading_metric_bullet_owners(
    line: str,
    *,
    allowed_owners: set[str],
) -> tuple[str, ...]:
    """Return an owner explicitly labeling a quantitative list item.

    Only the prefix before the first metric/colon can establish persistent
    context. A later comparison target in ``Acc 75.86 > Baseline 75.10`` is an
    owner for its own value, not a heading for the following bullet.
    """

    compact = " ".join(str(line or "").split())
    if not re.match(r"^(?:[-+*]|\d+[.)])\s+", compact):
        return ()
    metric_match = _METRIC_RE.search(compact)
    boundary = metric_match.start() if metric_match is not None else len(compact)
    colon = compact.find(":")
    if 0 <= colon < boundary:
        boundary = colon
    prefix = compact[:boundary]
    owners = tuple(
        subject
        for subject in _subject_identifiers(prefix)
        if subject in allowed_owners
    )
    return tuple(dict.fromkeys(owners))


def _unique_prevalue_line_owner(
    line: str,
    *,
    allowed_owners: set[str],
    known_qualifiers: set[str] | None = None,
) -> str | None:
    """Resolve one owner introduced before the first value in its sentence.

    This supports ``With all modalities, FF-MT obtains CCC ... and MAE ...``
    while refusing comparison prose such as ``Compared with AFEA, ASPIRE uses
    6.35M``: the latter has multiple subject identities before the value.
    """

    first_number = next(_NUMBER_RE.finditer(line), None)
    if first_number is None:
        return None
    sentence_start = 0
    for boundary in _SENTENCE_DELIMITER_RE.finditer(line, 0, first_number.start()):
        sentence_start = boundary.end()
    prefix = line[sentence_start : first_number.start()]
    occurrences = tuple(
        (subject, span)
        for subject, span in _subject_occurrences(prefix)
        if not _subject_occurrence_is_contextual(prefix, span)
        and subject not in {_BASELINE_OWNER_SUBJECT, _PROPOSED_OWNER_SUBJECT}
    )
    subjects = tuple(dict.fromkeys(subject for subject, _span in occurrences))
    owner_subjects = tuple(subject for subject in subjects if subject in allowed_owners)
    if len(owner_subjects) != 1:
        return None
    qualifier_set = known_qualifiers or set()
    if any(
        subject not in allowed_owners
        and not _subject_is_measurement_context(subject, qualifier_set)
        and not _span_is_inside_parentheses(prefix, span)
        for subject, span in occurrences
    ):
        return None
    return owner_subjects[0]


def _span_is_inside_parentheses(text: str, span: tuple[int, int]) -> bool:
    """Return whether a subject occurrence belongs to a local annotation."""

    opening = text.rfind("(", 0, span[0] + 1)
    closing = text.find(")", span[1])
    return opening >= 0 and closing >= span[1]


def _claim_context_signature(claim: MetricValueClaim) -> tuple[Any, ...]:
    return (
        claim.metric,
        claim.value,
        claim.percentage,
        claim.qualifiers,
        claim.text,
    )


def _structural_answer_qualifier(
    line: str,
    *,
    known_qualifiers: set[str],
) -> str | None:
    """Bind a subheading only to a schema already present in evidence.

    This supports arbitrary labels such as ``Valence``, ``Indoor scenes`` or
    ``audio-text`` without a paper-specific vocabulary. A random heading
    cannot authorize a value because it must map uniquely to a qualifier from
    the same focused document.
    """

    plain = re.sub(r"^#{1,6}\s+", "", str(line or "")).strip()
    plain = re.sub(r"^(?:[-+*]|\d+[.)])\s+", "", plain).strip()
    plain = plain.strip("*_` :")
    candidate = _canonical_table_qualifier(plain)
    if not candidate:
        return None
    matches = [
        qualifier
        for qualifier in known_qualifiers
        if _canonical_qualifier_matches(candidate, qualifier)
    ]
    return matches[0] if len(matches) == 1 else None


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
    """Propagate an unambiguous metric variant from a document schema.

    Table captions often define a global schema such as ``F1 (Macro)`` while
    the individual columns are labelled only ``LR.F1``/``Within.F1``. Papers
    also commonly declare ``Macro F1-score (F1)`` in an evaluation-metrics
    paragraph on a different page/source row. The caller aggregates canonical
    excerpts by document before invoking this helper. Only one unique variant
    per metric is propagated; a document mentioning both macro and micro
    variants remains fail-closed.
    """

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    schema_text = " ".join(
        str(value)
        for value in (
            document.get("caption"),
            document.get("content"),
            metadata.get("caption"),
            metadata.get("content"),
        )
        if value
    )
    if not schema_text:
        return {}

    variants_by_metric: dict[str, set[str]] = {}
    folded = _fold_for_matching(schema_text)
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
    global_context_qualifiers: tuple[str, ...] = (),
) -> MetricValueClaim:
    qualifiers = list(claim.qualifiers)
    for qualifier in global_context_qualifiers:
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)
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


def _document_global_context_qualifiers(document: dict) -> tuple[str, ...]:
    """Extract explicit dataset/corpus context declared by an artifact.

    A table row may encode modality while its caption declares the dataset.
    Both are orthogonal parts of the measurement path. Propagate only labels
    introduced by an explicit dataset/corpus/benchmark marker; arbitrary title
    words are never promoted to qualifiers.
    """

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    caption = " ".join(
        str(value)
        for value in (document.get("caption"), metadata.get("caption"))
        if value
    )
    folded = _fold_for_matching(caption)
    qualifiers: list[str] = []
    pattern = re.compile(
        r"(?<!\w)(?:based\s+on|evaluated\s+on|tested\s+on|trained\s+on|"
        r"using|from|on)\s+(?:the\s+)?"
        r"(?P<label>[a-z0-9][a-z0-9._+/-]*(?:\s+[a-z0-9][a-z0-9._+/-]*){0,4}?)"
        r"(?=\s+(?:dataset|corpus|benchmark)\b)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(folded):
        qualifier = _canonical_table_qualifier(match.group("label"))
        if qualifier and qualifier not in qualifiers:
            qualifiers.append(qualifier)
    return tuple(qualifiers)


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
    "war": ("weighted average recall", "war"),
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
    "pearson_r": (
        "pearson correlation coefficient",
        "pearson correlations",
        "pearson correlation",
        "pearson r",
    ),
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
_BYTE_SIZE_RE = re.compile(
    r"(?<![\w.-])(?P<value>\d+(?:\s*[.,]\s*\d+)?)\s*"
    r"(?P<prefix>[kmgt])?(?P<binary>i)?b(?!\w)",
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
    r"times?|repetitions?|trials?|iterations?|documents?|questions?|"
    r"contributions?|steps?|reasons?|nhóm|phần|loại|chỉ\s*số|"
    r"lớp|lop|cảm\s*xúc|đóng\s*góp|bước|lý\s*do|lần|lượt|"
    r"tài\s*liệu|câu\s*hỏi)\b",
    re.IGNORECASE,
)
_STRUCTURAL_RUN_COUNT_SUFFIX_RE = re.compile(r"^\s+runs?\b", re.IGNORECASE)
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
    "avg": ("average", "avg", "mean", "trung bình", "trung binh"),
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
    "efficiency",
}


def _extract_metric_claims(
    text: str,
    *,
    join_soft_wrapped: bool = False,
) -> list[MetricValueClaim]:
    if not text or not text.strip():
        return []

    table_claims = _extract_markdown_table_claims(text)
    claims = list(table_claims)
    claims.extend(_extract_caption_declared_metric_table_claims(text))
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
            row_text = " | ".join(_table_cells(stripped))
            if any(claim.text.startswith(row_text) for claim in table_claims):
                # The structural table parser already interpreted this row.
                # Re-parsing an individual body cell can lose the composite
                # header and manufacture a second claim with the wrong metric.
                continue
            for cell in _table_cells(stripped):
                normalized_cell = _normalize_jammed_metric_numbers(cell)
                if _METRIC_RE.search(normalized_cell):
                    claims.extend(_extract_line_claims(normalized_cell))
            continue
        claims.extend(_extract_line_claims(stripped))
    if join_soft_wrapped:
        for logical_line in _soft_wrapped_prose_lines(text):
            claims.extend(_extract_line_claims(logical_line))
    return _deduplicate_claims(claims)


def _soft_wrapped_prose_lines(text: str) -> list[str]:
    """Rejoin PDF physical lines without flattening document structure.

    Some parsers place blank lines between every visual line, so a scientific
    claim can arrive as ``72 K`` on one line and ``parameters`` on the next.
    Join only bounded non-table prose until sentence punctuation; Markdown
    rows, headings, and long buffers close the scope.
    """

    result: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if len(buffer) >= 2:
            result.append(" ".join(buffer))
        buffer.clear()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if (
            ("|" in line and line.count("|") >= 2)
            or re.match(r"^#{1,6}\s+", line)
            or _TABLE_SEPARATOR_RE.fullmatch(line.replace(" ", ""))
        ):
            flush()
            continue
        if buffer and (len(" ".join(buffer)) + len(line) > 1_200 or len(buffer) >= 24):
            flush()
        buffer.append(line)
        if re.search(r"[.!?](?:[*_`~)\]]*)$", line):
            flush()
    flush()
    return result


def _extract_caption_declared_metric_table_claims(
    text: str,
) -> list[MetricValueClaim]:
    """Apply one explicit caption metric to a qualifier-column table.

    Some PDF serializers emit ``CCC`` only in the caption while the table
    headers are ``Arousal | Valence | Mean``. Activation requires exactly one
    canonical metric in the immediately preceding prose, a labelled
    model/method column, and at least two textual qualifier columns. This is a
    schema rule; no document, metric, model, or dimension is hard-coded.
    """

    claims: list[MetricValueClaim] = []
    pending_context: list[str] = []
    table_lines: list[str] = []

    def flush() -> None:
        nonlocal table_lines
        if len(table_lines) < 2:
            table_lines = []
            return
        rows = [_table_cells(line) for line in table_lines]
        header_index = _markdown_table_header_index(rows)
        if header_index is None:
            table_lines = []
            return
        header = rows[header_index]
        if _metric_columns_from_header(header):
            table_lines = []
            return
        context = " ".join(pending_context[-2:])
        metric_matches = list(_METRIC_RE.finditer(context))
        metrics = {
            _metric_from_alias(match.group("metric")) for match in metric_matches
        }
        if len(metrics) != 1:
            table_lines = []
            return
        metric = next(iter(metrics))
        model_column = next(
            (
                index
                for index, cell in enumerate(header)
                if _MODEL_HEADER_RE.search(_fold_for_matching(cell))
            ),
            None,
        )
        if model_column is None:
            table_lines = []
            return
        qualifier_columns = {
            index: qualifier
            for index, cell in enumerate(header)
            if index != model_column
            and (qualifier := _canonical_table_qualifier(cell)) is not None
        }
        if len(qualifier_columns) < 2:
            table_lines = []
            return
        for row in rows[header_index + 1 :]:
            if _is_table_separator_row(row) or model_column >= len(row):
                continue
            subjects = _hierarchical_owner_subjects(row[model_column])
            if not subjects:
                continue
            row_text = " | ".join(row)
            for column, qualifier in qualifier_columns.items():
                if column >= len(row):
                    continue
                number_matches = list(_NUMBER_RE.finditer(row[column]))
                for number_match in number_matches:
                    if _looks_like_year(number_match) or _looks_like_reference_number(
                        row[column],
                        number_match,
                    ):
                        continue
                    value = _canonical_number(
                        number_match.group("value"),
                        number_match.group("magnitude"),
                    )
                    if value is None:
                        continue
                    claims.append(
                        MetricValueClaim(
                            metric=metric,
                            value=value,
                            percentage=bool(number_match.group("percent")),
                            subjects=subjects,
                            text=_compact_claim_text(f"{row_text} [{context}]"),
                            qualifiers=tuple(
                                dict.fromkeys(
                                    [
                                        qualifier,
                                        *_uncertainty_qualifiers(
                                            row[column],
                                            number_match=number_match,
                                            number_matches=number_matches,
                                        ),
                                    ]
                                )
                            ),
                        )
                    )
        table_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if "|" in stripped and stripped.count("|") >= 2:
            table_lines.append(stripped)
            continue
        if table_lines:
            flush()
        if stripped:
            pending_context.append(stripped)
            pending_context = pending_context[-2:]
    if table_lines:
        flush()
    return claims


def _extract_line_claims(line: str) -> list[MetricValueClaim]:
    """Extract claims without lending labels across sentence boundaries.

    A generated bullet can contain several complete sentences.  Treating it
    as one flat token stream lets a metric or model in sentence one steal a
    structural count/value in sentence two.  We retain only one explicitly
    aligned metric schema (for example UA/WA) across adjacent sentences; this
    supports compact follow-on result rows while keeping owners sentence-local.
    """

    line = _normalize_jammed_metric_numbers(line)
    boundaries = list(_SENTENCE_DELIMITER_RE.finditer(line))
    if not boundaries:
        return _extract_line_claims_segment(line)

    segments: list[str] = []
    start = 0
    for boundary in boundaries:
        end = boundary.end()
        if segment := line[start:end].strip():
            segments.append(segment)
        start = end
    if segment := line[start:].strip():
        segments.append(segment)
    if len(segments) <= 1:
        return _extract_line_claims_segment(line)

    claims: list[MetricValueClaim] = []
    active_schema: tuple[str, ...] = ()
    for segment in segments:
        metric_matches = list(_METRIC_RE.finditer(segment))
        number_matches = _eligible_number_matches(segment)
        if not metric_matches and active_schema:
            inherited = _claims_from_inherited_metric_schema(
                segment,
                schema=active_schema,
                number_matches=number_matches,
            )
            if inherited:
                claims.extend(inherited)
                continue
        segment_claims = _extract_line_claims_segment(segment)
        claims.extend(segment_claims)
        schema = _explicit_aligned_metric_schema(
            segment,
            metric_matches=metric_matches,
            number_matches=number_matches,
        )
        if schema:
            active_schema = schema
    return claims


def _eligible_number_matches(line: str) -> list[re.Match[str]]:
    return [
        match
        for match in _NUMBER_RE.finditer(line)
        if not _looks_like_year(match)
        and not _looks_like_reference_number(line, match)
        and not _looks_like_model_version_number(line, match)
    ]


def _explicit_aligned_metric_schema(
    line: str,
    *,
    metric_matches: list[re.Match[str]],
    number_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    if not _uses_aligned_slash_metrics(line, metric_matches, number_matches):
        return ()
    schema = tuple(
        dict.fromkeys(_metric_from_alias(match.group("metric")) for match in metric_matches)
    )
    return schema if len(schema) >= 2 else ()


def _claims_from_inherited_metric_schema(
    line: str,
    *,
    schema: tuple[str, ...],
    number_matches: list[re.Match[str]],
) -> list[MetricValueClaim]:
    """Apply a prior explicit schema only to complete slash-aligned blocks."""

    if (
        len(schema) < 2
        or len(number_matches) < len(schema)
        or len(number_matches) % len(schema) != 0
    ):
        return []
    for block_start in range(0, len(number_matches), len(schema)):
        block = number_matches[block_start : block_start + len(schema)]
        if not all(
            "/" in line[left.end() : right.start()]
            for left, right in zip(block, block[1:])
        ):
            return []
    claims: list[MetricValueClaim] = []
    for index, number_match in enumerate(number_matches):
        value = _canonical_number(
            number_match.group("value"),
            number_match.group("magnitude"),
        )
        if value is None:
            continue
        claims.append(
            MetricValueClaim(
                metric=schema[index % len(schema)],
                value=value,
                percentage=bool(number_match.group("percent")),
                subjects=_subjects_near_number(line, number_match),
                text=_compact_claim_text(line),
            )
        )
    return claims


def _extract_line_claims_segment(line: str) -> list[MetricValueClaim]:
    metric_matches = list(_METRIC_RE.finditer(line))
    number_matches = _eligible_number_matches(line)
    if not number_matches:
        return _byte_size_claims(line)

    claims: list[MetricValueClaim] = _byte_size_claims(line)
    claimed_number_spans: set[tuple[int, int]] = set()
    trailing_experiment_conditions = _trailing_experiment_conditions(
        line,
        metric_matches=metric_matches,
        number_matches=number_matches,
    )
    if metric_matches:
        if _uses_aligned_slash_metrics(line, metric_matches, number_matches):
            pairs = [
                (metric_matches[index % len(metric_matches)], number_match)
                for index, number_match in enumerate(number_matches)
            ]
        elif _uses_explicit_one_to_one_alignment(
            line,
            metric_matches,
            number_matches,
        ):
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
            if _number_is_part_of_byte_size(line, number_match):
                continue
            if _number_has_unresolved_following_metric(
                line,
                number_match=number_match,
                metric_matches=metric_matches,
                number_matches=number_matches,
            ):
                value = _canonical_number(
                    number_match.group("value"),
                    number_match.group("magnitude"),
                )
                if value is not None:
                    claims.append(
                        MetricValueClaim(
                            metric=_UNLABELED_METRIC,
                            value=value,
                            percentage=bool(number_match.group("percent")),
                            subjects=_subjects_near_number(line, number_match),
                            text=_compact_claim_text(line),
                        )
                    )
                    claimed_number_spans.add(number_match.span())
                continue
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
            sentence_conditions = _leading_experiment_conditions(
                line,
                position=number_match.start(),
            )
            aligned_trailing_condition = _aligned_trailing_experiment_condition(
                line,
                number_match=number_match,
                number_matches=number_matches,
            )
            local_claim_qualifiers = _metric_qualifiers_near_number(
                line,
                metric_match=metric_match,
                number_match=number_match,
                number_matches=number_matches,
            )
            claim_qualifiers = tuple(
                dict.fromkeys(
                    [
                        *sentence_conditions,
                        *(() if aligned_trailing_condition else trailing_experiment_conditions),
                        *aligned_trailing_condition,
                        *local_claim_qualifiers,
                        *_uncertainty_qualifiers(
                            line,
                            number_match=number_match,
                            number_matches=number_matches,
                        ),
                    ]
                )
            )
            claim = _claim_from_matches(
                metric_match,
                number_match,
                line,
                _comparison_subjects_near_number(line, number_match)
                or _anaphoric_subjects_near_number(line, number_match)
                or _subjects_near_number(line, number_match),
                qualifiers=claim_qualifiers,
            )
            shared_dimension_qualifiers = (
                _shared_dimension_qualifiers_before_metric(
                    line,
                    metric_match=metric_match,
                    number_match=number_match,
                )
                if not local_claim_qualifiers
                else ()
            )
            if claim is not None:
                if shared_dimension_qualifiers:
                    claims.extend(
                        MetricValueClaim(
                            metric=claim.metric,
                            value=claim.value,
                            percentage=claim.percentage,
                            subjects=claim.subjects,
                            text=claim.text,
                            qualifiers=tuple(
                                dict.fromkeys([*sentence_conditions, qualifier])
                            ),
                        )
                        for qualifier in shared_dimension_qualifiers
                    )
                else:
                    claims.append(claim)
                claimed_number_spans.add(number_match.span())

    # A percentage is itself a bounded performance/value claim even when the
    # author omitted the metric name (e.g. "improves by 4.2%").
    for number_match in number_matches:
        percentage_range_endpoint = _is_percentage_range_endpoint(
            line,
            number_match=number_match,
            number_matches=number_matches,
        )
        if (
            not number_match.group("percent")
            and not percentage_range_endpoint
        ) or number_match.span() in claimed_number_spans:
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


def _byte_size_claims(text: str) -> list[MetricValueClaim]:
    claims: list[MetricValueClaim] = []
    decimal_factors = {
        "": Decimal(1),
        "k": Decimal(1_000),
        "m": Decimal(1_000_000),
        "g": Decimal(1_000_000_000),
        "t": Decimal(1_000_000_000_000),
    }
    binary_factors = {
        "": Decimal(1),
        "k": Decimal(1_024),
        "m": Decimal(1_048_576),
        "g": Decimal(1_073_741_824),
        "t": Decimal(1_099_511_627_776),
    }
    for match in _BYTE_SIZE_RE.finditer(text):
        if _byte_size_is_model_variant(text, match):
            continue
        raw = re.sub(r"\s+", "", match.group("value")).replace(",", ".")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        prefix = (match.group("prefix") or "").casefold()
        factors = binary_factors if match.group("binary") else decimal_factors
        canonical = format((value * factors[prefix]).normalize(), "f")
        claims.append(
            MetricValueClaim(
                metric="storage_bytes",
                value=canonical,
                percentage=False,
                subjects=_subjects_near_number(text, match),
                text=_compact_claim_text(text),
            )
        )
    return claims


def _number_is_part_of_byte_size(
    text: str,
    number_match: re.Match[str],
) -> bool:
    byte_match = _BYTE_SIZE_RE.match(text, number_match.start())
    return bool(
        byte_match
        and byte_match.start() == number_match.start()
        and not _byte_size_is_model_variant(text, byte_match)
    )


def _byte_size_is_model_variant(text: str, match: re.Match[str]) -> bool:
    """Reject parameter-count model tags such as ``qwen3.5:4b``."""

    return match.start() > 0 and text[match.start() - 1] == ":"


_UNLABELED_METRIC = "__unlabeled__"


def _number_has_unresolved_following_metric(
    text: str,
    *,
    number_match: re.Match[str],
    metric_matches: list[re.Match[str]],
    number_matches: list[re.Match[str]],
) -> bool:
    """Detect a leading value separated from the next labelled measurement.

    In ``75.86, F1 76.31`` the comma means the visible F1 label belongs to the
    second value. The first metric must be resolved from canonical evidence,
    not copied backwards. A directly postfix form such as ``0.8 F1`` remains
    explicit and is not treated as unresolved.
    """

    following_metric = next(
        (match for match in metric_matches if match.start() >= number_match.end()),
        None,
    )
    if following_metric is None:
        return False
    if any(match.end() <= number_match.start() for match in metric_matches):
        return False
    if not any(match.start() >= following_metric.end() for match in number_matches):
        return False
    between = text[number_match.end() : following_metric.start()]
    return bool(
        re.search(
            r"[,;|/]|(?<!\w)(?:and|while|và|còn)(?!\w)",
            between,
            re.IGNORECASE,
        )
    )


def _trailing_experiment_conditions(
    text: str,
    *,
    metric_matches: list[re.Match[str]],
    number_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    """Share a final explicit dataset/split phrase across a metric bundle."""

    if not number_matches:
        return ()
    participating_metrics = [
        match for match in metric_matches if match.start() < number_matches[-1].end()
    ]
    if len(participating_metrics) < 2:
        return ()
    suffix = text[number_matches[-1].end() :]
    if not re.match(
        r"\s*(?:[*_`~),;:]\s*)*"
        r"(?:(?:on|in|for|under|trên|ở|cho)(?!\w))",
        suffix,
        re.IGNORECASE,
    ):
        return ()
    return _prose_qualifiers_from_context(suffix)


def _aligned_trailing_experiment_condition(
    text: str,
    *,
    number_match: re.Match[str],
    number_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    """Bind a trailing split vector to the corresponding value.

    Examples include ``0.51/0.52 on Test1/Test2`` and
    ``.70 and .72 for indoor and outdoor``.  Alignment is accepted only when
    the number of explicit labels exactly matches the number of values, so a
    broad prose phrase cannot create an arbitrary qualifier.
    """

    if len(number_matches) < 2:
        return ()
    suffix = text[number_matches[-1].end() :]
    connector = re.match(
        r"\s*(?:[*_`~),;:]\s*)*"
        r"(?:on|in|for|under|trên|ở|cho)(?!\w)\s+(?P<labels>.+?)"
        r"(?:[.!?]\s*)?$",
        suffix,
        re.IGNORECASE,
    )
    if connector is None:
        return ()
    raw_labels = connector.group("labels").strip(" *_`~.,;:()[]")
    parts = [
        part.strip()
        for part in re.split(
            r"\s*(?:/|\||,|(?<!\w)(?:and|và)(?!\w))\s*",
            raw_labels,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]
    if len(parts) != len(number_matches):
        return ()
    labels = tuple(_canonical_table_qualifier(part) for part in parts)
    if any(label is None for label in labels):
        return ()
    try:
        index = number_matches.index(number_match)
    except ValueError:
        return ()
    return (str(labels[index]),)


def _is_percentage_range_endpoint(
    text: str,
    *,
    number_match: re.Match[str],
    number_matches: list[re.Match[str]],
) -> bool:
    """Propagate a trailing percent sign to the first endpoint of a range."""

    try:
        index = number_matches.index(number_match)
    except ValueError:
        return False
    if index + 1 >= len(number_matches):
        return False
    following = number_matches[index + 1]
    return bool(
        following.group("percent")
        and re.fullmatch(r"\s*[–—]\s*", text[number_match.end() : following.start()])
    )


def _uncertainty_qualifiers(
    text: str,
    *,
    number_match: re.Match[str],
    number_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    """Mark the second endpoint of an explicit estimate ± uncertainty pair."""

    try:
        index = number_matches.index(number_match)
    except ValueError:
        return ()
    if index == 0:
        return ()
    previous = number_matches[index - 1]
    between = text[previous.end() : number_match.start()]
    if re.fullmatch(r"\s*(?:±|\+\s*/\s*-)\s*", between):
        return ("uncertainty",)
    return ()


def _leading_experiment_conditions(
    text: str,
    *,
    position: int | None = None,
) -> tuple[str, ...]:
    """Recover an explicit row/slice condition introduced before a claim.

    Scientific prose often linearizes a result-table row as ``With all
    modalities, ...`` or ``Combining all modalities gives ...``.  The metric
    and semantic dimension alone are insufficient when the same value occurs
    under several configurations.  This grammar is deliberately structural:
    it accepts only a short leading experimental clause with an explicit
    boundary, and never infers a condition from a paper/model name or value.
    """

    sentence_start = 0
    if position is not None:
        for boundary in _SENTENCE_DELIMITER_RE.finditer(text, 0, position):
            sentence_start = boundary.end()
    segment = text[sentence_start:]
    local_position = None if position is None else max(0, position - sentence_start)
    folded = _fold_for_matching(re.sub(r"[*_`~]", "", segment)).strip()
    match = re.match(
        r"^(?:(?:with|using|under|for|on|combining|"
        r"voi|dung|khi\s+dung|ket\s+hop)\s+)"
        r"(?P<label>[a-z0-9][a-z0-9 _+&/-]{0,63}?)"
        r"\s*(?:,|(?=\s+(?:gives?|yields?|achieves?|reports?|"
        r"dat|cho|tao\s+ra)\b))",
        folded,
        re.IGNORECASE,
    )
    if match is None:
        return ()
    if local_position is not None:
        competing_clause = re.search(
            r"(?:;|[.!?])\s*(?:with|using|under|for|on|in|"
            r"voi|dung|tren|o)\s+[a-z0-9]",
            folded[match.end() :],
            re.IGNORECASE,
        )
        if (
            competing_clause is not None
            and local_position >= match.end() + competing_clause.start()
        ):
            return ()
    label = " ".join(match.group("label").split())
    # A leading clause can describe the task before introducing the actual
    # experimental slice: ``For attribute prediction under condition Within``
    # / ``Với dự đoán thuộc tính trong điều kiện Within``. The complete clause
    # is discourse, while the value after the explicit condition marker is the
    # row qualifier. Keep this open-vocabulary and structurally bounded.
    nested_condition = re.search(
        r"(?:under|in)\s+(?:the\s+)?condition\s+"
        r"(?P<condition>[a-z0-9][a-z0-9 _+&/-]{0,31})$|"
        r"(?:trong|o)\s+(?:dieu\s+kien)\s+"
        r"(?P<condition_vi>[a-z0-9][a-z0-9 _+&/-]{0,31})$",
        label,
        re.IGNORECASE,
    )
    if nested_condition is not None:
        label = " ".join(
            (
                nested_condition.group("condition")
                or nested_condition.group("condition_vi")
                or ""
            ).split()
        )
    scope = re.fullmatch(
        r"(?P<quantity>all|both|tat\s+ca|ca\s+hai)\s+"
        r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)?"
        r"(?:available\s+)?(?:modalities|modality|streams?|inputs?|"
        r"features?|views?|signals?|phuong\s+thuc)",
        label,
    )
    if scope is not None:
        quantity = scope.group("quantity")
        condition = "all" if quantity in {"all", "tat ca"} else "both"
    else:
        condition = _canonical_table_qualifier(label)
    return (condition,) if condition else ()


def _shared_dimension_qualifiers_before_metric(
    text: str,
    *,
    metric_match: re.Match[str],
    number_match: re.Match[str],
) -> tuple[str, ...]:
    """Expand ``both with MAE 0.08`` across named semantic dimensions.

    A single shared value is a conjunction of separate measurements, not one
    fabricated composite qualifier. Expansion is enabled only by explicit
    distributive language and full dimension names already supported by the
    bounded qualifier grammar.
    """

    sentence_start = 0
    for boundary in _SENTENCE_DELIMITER_RE.finditer(text, 0, metric_match.start()):
        sentence_start = boundary.end()
    prefix = text[sentence_start : metric_match.start()]
    prefix_distributes = bool(re.search(
        r"(?<!\w)(?:both|each|respectively|cả\s+hai|đều)"
        r"(?:\s+(?:with|for|on|cùng|với))?\s*$",
        prefix,
        re.IGNORECASE,
    ))
    sentence_end_match = _SENTENCE_DELIMITER_RE.search(text, number_match.end())
    sentence_end = sentence_end_match.start() if sentence_end_match else len(text)
    suffix = text[number_match.end() : sentence_end]
    postfix_distributes = bool(
        re.match(
            r"\s*(?:,?\s*(?:for|across|on|over|cho|trên)\s+)?"
            r"(?:both|each|cả\s+hai|đều)(?!\w)",
            suffix,
            re.IGNORECASE,
        )
    )
    if not prefix_distributes and not postfix_distributes:
        return ()
    qualifiers: list[str] = []
    for match in _METRIC_QUALIFIER_RE.finditer(prefix):
        raw = match.group("qualifier").casefold()
        if len(raw) <= 1:
            continue
        qualifier = _QUALIFIER_ALIAS_TO_CANONICAL[raw]
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)
    return tuple(qualifiers) if len(qualifiers) >= 2 else ()


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
    if "/" in metric_region:
        # ``from 61/62 to 72/71 UA/WA`` declares the aligned schema directly
        # in the metric vector even though every value precedes its label.
        return any(
            "/" in line[left.end() : right.start()]
            for left, right in zip(number_matches, number_matches[1:])
        )

    # The first complete value block must already bind one-to-one to the
    # declared metrics under the ordinary local grammar. This distinguishes
    # ``72 UA, 71 WA, baseline 61/62`` from ``CCC .41/.49/.39; F1 .67``:
    # only the former establishes an aligned two-column schema that may cycle
    # over the later compact pair.
    for index, metric_match in enumerate(metric_matches):
        if _metric_match_for_number(
            line,
            metric_matches=metric_matches,
            number_match=number_matches[index],
        ) is not metric_match:
            return False

    # Require at least one slash-delimited complete block. Merely having the
    # same number of labels and values is not enough to infer positional
    # cycling from prose.
    return any(
        "/" in line[left.end() : right.start()]
        for left, right in zip(number_matches, number_matches[1:])
    )


def _uses_explicit_one_to_one_alignment(
    line: str,
    metric_matches: list[re.Match[str]],
    number_matches: list[re.Match[str]],
) -> bool:
    """Recognize a real equal-arity metric/value list, not coincidental counts."""

    if not metric_matches or len(metric_matches) != len(number_matches):
        return False
    locally_aligned = all(
        _metric_match_for_number(
            line,
            metric_matches=metric_matches,
            number_match=number_matches[index],
        )
        is metric_match
        for index, metric_match in enumerate(metric_matches)
    )
    if locally_aligned:
        return True

    metric_vector = line[metric_matches[0].start() : metric_matches[-1].end()]
    for match in reversed(metric_matches):
        local_start = match.start() - metric_matches[0].start()
        local_end = match.end() - metric_matches[0].start()
        metric_vector = metric_vector[:local_start] + " " + metric_vector[local_end:]
    metric_vector = _fold_for_matching(metric_vector)
    metric_vector = re.sub(
        r"(?<!\w)(?:and|while|versus|vs|va|con)(?!\w)",
        " ",
        metric_vector,
    )
    return not bool(re.sub(r"[\s,;/|&+()\[\]_-]+", "", metric_vector))


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


def _anaphoric_subjects_near_number(
    text: str,
    number_match: re.Match[str],
) -> tuple[str, ...]:
    """Resolve a leading owner referenced later by ``it/this model``.

    PDF prose often inserts deployment qualifiers between the model and its
    measurement: ``Model-X with an ONNX runtime ... as it has 72K parameters``.
    The acronym inside that modifier is context, while the pronoun points back
    to the leading owner. Activation requires an explicit anaphoric verb and a
    bounded modifier, so ordinary multi-model comparisons remain unchanged.
    """

    prefix = text[: number_match.start()]
    anaphor = list(
        re.finditer(
            r"(?<!\w)(?:as|because|since|and|,)?\s*"
            r"(?:it|this\s+(?:model|method|system|architecture))\s+"
            r"(?:has|uses|contains|requires|reports?|achieves?|đạt|có|dùng)"
            r"(?!\w)",
            prefix,
            re.IGNORECASE,
        )
    )
    if not anaphor:
        return ()
    following_number = next(
        (
            match
            for match in _NUMBER_RE.finditer(text, number_match.end())
            if not _looks_like_year(match)
            and not _looks_like_reference_number(text, match)
        ),
        None,
    )
    local_end = following_number.start() if following_number is not None else len(text)
    local_suffix = text[number_match.end() : local_end]
    for _subject, span in _subject_occurrences(local_suffix):
        if _FOLLOWING_OWNER_PREFIX_RE.search(local_suffix[: span[0]]):
            # ``3.12M parameters for MobileNet`` names a new explicit owner;
            # it is not governed by an earlier ``it has`` anaphor.
            return ()
    before = prefix[: anaphor[-1].start()]
    modifier = re.search(
        r"(?<!\w)(?:with|using|under|on|in|với|dùng|trên|ở)(?!\w)",
        before,
        re.IGNORECASE,
    )
    owner_region = before[: modifier.start()] if modifier is not None else before
    subjects = _noncontextual_subject_identifiers(owner_region)
    return subjects[-1:] if subjects else ()


def _normalize_jammed_metric_numbers(text: str) -> str:
    """Restore a missing boundary between a metric phrase and its value.

    PDF extraction commonly emits ``accuracy of75.86%`` or ``F10.74``.
    Relaxing the global number lookbehind would also treat digits in model
    names as measurements, so repair only a bounded metric+connector+number
    sequence before the normal parser runs.
    """

    # An ASCII hyphen between two percentages is a range delimiter, not a
    # numeric sign. Normalizing only the bounded ``number-number%`` form lets
    # the ordinary number lexer see both endpoints without weakening its
    # protection for model versions, dates, or subtraction.
    text = re.sub(
        r"(?P<low>\d+(?:[.,]\d+)?)\s*-\s*"
        r"(?P<high>\d+(?:[.,]\d+)?)(?=\s*%)",
        r"\g<low>–\g<high>",
        text,
    )

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
        lambda match: " ".join(
            part
            for part in (
                match.group("metric"),
                match.group("link").strip(),
                match.group("number"),
            )
            if part
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

        dimension_matrix_claims = _row_metric_dimension_matrix_claims(
            rows,
            header_index=header_index,
        )
        if dimension_matrix_claims is not None:
            claims.extend(dimension_matrix_claims)
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
        inferred_column_qualifiers = _inferred_first_metric_group_qualifiers(
            header,
            metric_columns,
        )
        active_row_section: str | None = None
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
                inferred_column_qualifiers = _inferred_first_metric_group_qualifiers(
                    header,
                    metric_columns,
                )
                active_row_section = None
                continue
            row_section = _hierarchical_section_qualifier(row)
            if row_section and not _row_has_data_number(row):
                active_row_section = row_section
                continue
            row_text = " | ".join(row)
            for column in metric_columns:
                if column >= len(row):
                    continue
                metric = metric_columns[column]
                header_metric_matches = list(_METRIC_RE.finditer(header[column]))
                row_subjects = _table_column_subjects(
                    header,
                    row,
                    metric_column=column,
                )
                row_qualifiers = _table_row_qualifiers(
                    header,
                    row,
                    metric_column=column,
                    active_section=active_row_section,
                )
                row_qualifiers = tuple(
                    qualifier
                    for qualifier in row_qualifiers
                    if not any(
                        _canonical_qualifier_matches(qualifier, subject)
                        for subject in row_subjects
                    )
                )
                row_qualifiers = tuple(
                    dict.fromkeys(
                        [
                            *row_qualifiers,
                            *inferred_column_qualifiers.get(column, ()),
                        ]
                    )
                )
                inline_metric_matches = list(_METRIC_RE.finditer(row[column]))
                number_matches = [
                    match
                    for match in _NUMBER_RE.finditer(row[column])
                    if not _looks_like_year(match)
                    and not _looks_like_reference_number(row[column], match)
                ]
                if inline_metric_matches:
                    if (
                        len(header_metric_matches) > 1
                        and len(inline_metric_matches) < len(header_metric_matches)
                        and len(number_matches) == len(header_metric_matches)
                    ):
                        # A generated cell can repeat only the second label
                        # from a composite header: ``Acc / F1`` with
                        # ``75.86, F1 76.31``. The explicit header still owns
                        # the complete aligned schema; parsing only inline
                        # labels would incorrectly turn both values into F1.
                        claims.extend(
                            _claims_from_table_cells(
                                metric_cell=header[column],
                                metric_matches=header_metric_matches,
                                value_cell=row[column],
                                number_matches=number_matches,
                                subjects=row_subjects,
                                row_text=row_text,
                                percentage_context=False,
                                extra_qualifiers=row_qualifiers,
                            )
                        )
                        continue
                    # A comparison matrix may declare a broad metric family in
                    # its header, then put a compact multi-metric summary in
                    # one body cell (``CCC-V .63, CCC-A .68`` or ``miF1 .51,
                    # maF1 .30, HL .18``). Parse that cell by its own labels;
                    # applying the first header metric to every value corrupts
                    # both metric identity and the owner in adjacent columns.
                    claims.extend(
                        _claims_with_table_context(
                            _extract_line_claims(row[column]),
                            subjects=row_subjects,
                            qualifiers=row_qualifiers,
                        )
                    )
                    continue
                if len(header_metric_matches) > 1:
                    claims.extend(
                        _claims_from_table_cells(
                            metric_cell=header[column],
                            metric_matches=header_metric_matches,
                            value_cell=row[column],
                            number_matches=number_matches,
                            subjects=row_subjects,
                            row_text=row_text,
                            percentage_context=False,
                            extra_qualifiers=row_qualifiers,
                        )
                    )
                    continue
                metric_match = header_metric_matches[0] if header_metric_matches else None
                if metric_match is not None:
                    metric = _metric_from_alias(metric_match.group("metric"))
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
                            qualifiers=tuple(
                                dict.fromkeys(
                                    [
                                        *_metric_qualifiers_from_table_label(
                                            header[column],
                                            metric=metric,
                                            metric_match=metric_match,
                                            extra_qualifiers=row_qualifiers,
                                        ),
                                        *_uncertainty_qualifiers(
                                            row[column],
                                            number_match=number_match,
                                            number_matches=number_matches,
                                        ),
                                    ]
                                )
                            ),
                        )
                    )
            # Some generated tables put a compact metric bundle inside a
            # semantic column (``Valence | CCC 0.70, MAE 0.08``). The header
            # is a condition, not itself a metric, so the ordinary metric-
            # column loop cannot see it. Bind inline claims to both the row
            # condition and this column label.
            for column, cell in enumerate(row):
                if column in metric_columns:
                    continue
                inline_metric_matches = list(_METRIC_RE.finditer(cell))
                if not inline_metric_matches or not _NUMBER_RE.search(cell):
                    continue
                header_cell = header[column] if column < len(header) else ""
                folded_header = _fold_for_matching(header_cell)
                # Only a recognized measurement dimension (Valence,
                # Arousal, Dominance, Average, including their aliases) may
                # qualify an inline metric bundle.  Treating every prose
                # column title as a condition makes ordinary tables such as
                # ``Contribution | Meaning`` acquire invented qualifiers
                # (``contribution``, ``meaning``) and rejects otherwise
                # grounded claims.  Unknown headers therefore fall through
                # to the independent prose-cell parser below.
                semantic_header = _QUALIFIER_ALIAS_TO_CANONICAL.get(
                    re.sub(r"[*_`~]", "", folded_header).strip()
                )
                if semantic_header is None:
                    continue
                row_qualifiers = list(
                    _table_row_qualifiers(
                        header,
                        row,
                        metric_column=column,
                        active_section=active_row_section,
                    )
                )
                if semantic_header and semantic_header not in row_qualifiers:
                    row_qualifiers.append(semantic_header)
                claims.extend(
                    _claims_with_table_context(
                        _extract_line_claims(cell),
                        subjects=_table_column_subjects(
                            header,
                            row,
                            metric_column=column,
                        ),
                        qualifiers=tuple(row_qualifiers),
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
                        qualifiers=tuple(
                            dict.fromkeys(
                                [
                                    *qualifiers,
                                    *_uncertainty_qualifiers(
                                        row[column],
                                        number_match=number_match,
                                        number_matches=number_matches,
                                    ),
                                ]
                            )
                        ),
                    )
                )

    return claims if saw_section and claims else None


_AFFECT_DIMENSION_QUALIFIERS = frozenset({"valence", "arousal", "dominance"})


def _row_metric_dimension_matrix_claims(
    rows: list[list[str]],
    *,
    header_index: int,
) -> list[MetricValueClaim] | None:
    """Parse matrices whose rows declare metrics and columns declare outputs.

    Generated comparisons often transpose a canonical table into
    ``Scheme | V | A | D`` and put ``CCC``/``MAE``/``Pearson r`` in each row.
    This is schema-driven: activation requires at least two explicit semantic
    dimension columns and exactly one known metric in a numeric row. Model
    ownership is intentionally left empty because compressed labels such as
    ``FF-MT/XGB`` are lossy; metric + value + complete row/dimension qualifiers
    must instead resolve to one canonical evidence owner.
    """

    header = rows[header_index]
    dimension_columns: dict[int, str] = {}
    for index, cell in enumerate(header):
        qualifier = _canonical_table_qualifier(cell)
        if qualifier in _AFFECT_DIMENSION_QUALIFIERS:
            dimension_columns[index] = qualifier
    if len(dimension_columns) < 2:
        return None

    claims: list[MetricValueClaim] = []
    saw_metric_row = False
    for row in rows[header_index + 1 :]:
        if _is_table_separator_row(row):
            continue
        metric_cells = [
            (index, match)
            for index, cell in enumerate(row)
            for match in _METRIC_RE.finditer(cell)
        ]
        if len(metric_cells) != 1:
            continue
        metric_cell_index, metric_match = metric_cells[0]
        metric_cell = row[metric_cell_index]
        metric = _metric_from_alias(metric_match.group("metric"))
        row_qualifiers = _row_metric_matrix_qualifiers(
            metric_cell,
            metric_match=metric_match,
        )
        row_text = " | ".join(row)
        row_claims: list[MetricValueClaim] = []
        for column, dimension in dimension_columns.items():
            if column >= len(row):
                continue
            value_cell = row[column]
            number_matches = [
                match
                for match in _NUMBER_RE.finditer(value_cell)
                if not _looks_like_year(match)
                and not _looks_like_reference_number(value_cell, match)
            ]
            if len(number_matches) != 1:
                continue
            number_match = number_matches[0]
            if not _metric_number_is_eligible(
                metric=metric,
                text=value_cell,
                number_match=number_match,
            ):
                continue
            value = _canonical_number(
                number_match.group("value"),
                number_match.group("magnitude"),
            )
            if value is None:
                continue
            row_claims.append(
                MetricValueClaim(
                    metric=metric,
                    value=value,
                    percentage=bool(
                        number_match.group("percent")
                        or "%" in metric_cell
                        or "%" in header[column]
                    ),
                    subjects=(),
                    text=_compact_claim_text(
                        f"{row_text} [{dimension} / {metric}]"
                    ),
                    qualifiers=tuple(
                        dict.fromkeys([*row_qualifiers, dimension])
                    ),
                )
            )
        if row_claims:
            saw_metric_row = True
            claims.extend(row_claims)
    return claims if saw_metric_row else None


def _row_metric_matrix_qualifiers(
    metric_cell: str,
    *,
    metric_match: re.Match[str],
) -> tuple[str, ...]:
    """Recover conditions before a row-declared metric without model lists."""

    prefix = metric_cell[: metric_match.start()]
    # In ``audio-video — FF-MT/XGB, CCC`` the text before the dash is the
    # measurement condition; the model vector after it is deliberately not
    # guessed. Without a dash, comma-separated labels such as
    # ``Zero-shot, EmoBank Pearson r`` are independent conditions.
    parts = re.split(r"\s+[—–-]{1,2}\s+", prefix, maxsplit=1)
    condition_text = parts[0] if len(parts) > 1 else prefix
    qualifiers: list[str] = []
    for part in re.split(r"\s*[,;]\s*", condition_text):
        qualifier = _canonical_table_qualifier(part)
        if qualifier and qualifier not in qualifiers:
            qualifiers.append(qualifier)
    return tuple(qualifiers)


def _hierarchical_section_qualifier(row: list[str]) -> str | None:
    nonempty_cells = [
        cell
        for cell in row
        if re.sub(r"[*_`~\s]", "", str(cell)).strip("-:|")
    ]
    # Parenthetical glosses commonly translate a dataset/section label
    # (``ViSEC (Vietnamese)``). They describe the section, but are not part of
    # its stable identity and should not prevent ``ViSEC`` from matching it.
    labels = [
        _canonical_table_qualifier(re.sub(r"\([^)]*\)", " ", cell))
        for cell in nonempty_cells
    ]
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
    extra_qualifiers: tuple[str, ...] = (),
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
    if metric_match is None and metric == "pearson_r":
        parenthetical_r = re.search(r"\(\s*r\s*\)", value, re.IGNORECASE)
        if parenthetical_r is not None:
            remainder = (
                f"{value[:parenthetical_r.start()]} "
                f"{value[parenthetical_r.end():]}"
            )
            qualifiers = list(extra_qualifiers)
            for match in _METRIC_QUALIFIER_RE.finditer(remainder):
                qualifier = _QUALIFIER_ALIAS_TO_CANONICAL[
                    match.group("qualifier").casefold()
                ]
                if qualifier not in qualifiers:
                    qualifiers.append(qualifier)
            return tuple(qualifiers)
    if metric_match is None:
        return extra_qualifiers

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
    if metric == "parameters":
        remainder = re.sub(
            r"\(\s*m(?:illion)?s?\s*\)",
            " ",
            remainder,
            flags=re.IGNORECASE,
        )

    qualifiers: list[str] = list(extra_qualifiers)
    alias_variant = _metric_variant_from_alias(metric_match.group("metric"))
    if alias_variant:
        qualifiers.append(alias_variant)
    for match in _METRIC_QUALIFIER_RE.finditer(remainder):
        qualifier = _QUALIFIER_ALIAS_TO_CANONICAL[match.group("qualifier").casefold()]
        if qualifier not in qualifiers:
            qualifiers.append(qualifier)
    qualifier = _canonical_table_qualifier(remainder)
    if qualifier and qualifier not in qualifiers:
        qualifiers.append(qualifier)
    return tuple(qualifiers)


def _inferred_first_metric_group_qualifiers(
    header: list[str],
    metric_columns: dict[int, str],
) -> dict[int, tuple[str, ...]]:
    """Infer an omitted first group label from a repeated table schema.

    Exporters frequently flatten a two-part header into ``miF1 | maF1 | HL |
    Test2.miF1 | Test2.maF1 | Test2.HL``.  Calling the unlabelled columns
    ``Test1`` is safe only when at least two distinct metric signatures have
    both an unlabelled occurrence and the same explicit numbered group.  A
    lone ``F1 | Test2.F1`` or unrelated repeated metric stays unqualified.
    """

    group_re = re.compile(
        r"(?<![a-z0-9])(?P<family>test|split|fold|partition)"
        r"[\s._/-]*(?P<number>[2-9]\d*)(?!\d)",
        re.IGNORECASE,
    )
    unqualified_by_signature: dict[tuple[str, str | None], list[int]] = {}
    explicit_by_group: dict[
        tuple[str, int],
        set[tuple[str, str | None]],
    ] = {}
    for column, metric in metric_columns.items():
        if column >= len(header):
            continue
        label = _fold_for_matching(header[column])
        metric_match = next(
            (
                match
                for match in _METRIC_RE.finditer(header[column])
                if _metric_from_alias(match.group("metric")) == metric
            ),
            None,
        )
        signature = (
            metric,
            _metric_variant_from_alias(metric_match.group("metric"))
            if metric_match is not None
            else None,
        )
        group_match = group_re.search(label)
        if group_match is None:
            unqualified_by_signature.setdefault(signature, []).append(column)
            continue
        group = (
            group_match.group("family").casefold(),
            int(group_match.group("number")),
        )
        explicit_by_group.setdefault(group, set()).add(signature)

    inferred: dict[int, tuple[str, ...]] = {}
    for (family, number), signatures in explicit_by_group.items():
        shared_signatures = {
            signature for signature in signatures if signature in unqualified_by_signature
        }
        if len(shared_signatures) < 2:
            continue
        first_group = f"{family}1" if number == 2 else None
        if first_group is None:
            continue
        for signature in shared_signatures:
            for column in unqualified_by_signature[signature]:
                inferred[column] = (first_group,)
    return inferred


_MODALITY_TOKEN_ALIASES = {
    "a": "audio",
    "acoustic": "audio",
    "audio": "audio",
    "s": "audio",
    "speech": "audio",
    "semantic": "text",
    "t": "text",
    "text": "text",
    "transcript": "text",
    "v": "visual",
    "video": "visual",
    "visual": "visual",
}


def _canonical_modality_composition(value: str) -> str | None:
    """Normalize open modality compositions without paper-specific aliases."""

    folded = _fold_for_matching(value)
    parts = [part for part in re.split(r"[^a-z0-9]+", folded) if part]
    if len(parts) < 2 or not all(part in _MODALITY_TOKEN_ALIASES for part in parts):
        return None
    modalities = tuple(dict.fromkeys(_MODALITY_TOKEN_ALIASES[part] for part in parts))
    return "-".join(modalities) if len(modalities) >= 2 else None


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
    # Hierarchical headers frequently label a semantic output as ``Valence
    # model`` / ``Arousal predictor`` while another serializer emits simply
    # ``Valence``. Strip a generic component suffix only when the remaining
    # label is already a known semantic qualifier; arbitrary model names are
    # never rewritten.
    semantic_component = re.fullmatch(
        r"(?P<label>.+?)-(?:model|regressor|predictor|head)",
        folded,
    )
    if semantic_component is not None:
        alias = _QUALIFIER_ALIAS_TO_CANONICAL.get(semantic_component.group("label"))
        if alias:
            return alias
    modality = _canonical_modality_composition(plain)
    if modality:
        return modality
    if folded in _GENERIC_TABLE_QUALIFIERS or not re.search(r"[a-z]", folded):
        return None
    if _canonical_metric(plain) is not None:
        return None
    return folded[:80]


def _canonical_qualifier_matches(left: str, right: str) -> bool:
    left_key = _canonical_table_qualifier(left)
    right_key = _canonical_table_qualifier(right)
    return bool(left_key and right_key and left_key == right_key)


def _subject_is_measurement_context(
    subject: str,
    qualifiers: tuple[str, ...],
) -> bool:
    return any(
        _canonical_qualifier_matches(subject, qualifier)
        for qualifier in qualifiers
    )


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


def _table_column_subjects(
    header: list[str],
    row: list[str],
    *,
    metric_column: int,
) -> tuple[str, ...]:
    """Resolve the owner column local to one metric group.

    Wide scientific tables may repeat ``model, metric, metric`` groups for
    several targets in the same row. Using the first model cell for every
    metric silently assigns later values to the wrong method. Prefer the
    nearest labelled model/method column on the left, then fall back to the
    ordinary single-owner table rule.
    """

    model_columns = [
        index
        for index, cell in enumerate(header)
        if _MODEL_HEADER_RE.search(_fold_for_matching(cell))
    ]
    preceding = [index for index in model_columns if index <= metric_column]
    candidates = list(reversed(preceding))
    if not candidates and len(model_columns) == 1:
        candidates = model_columns
    for index in candidates:
        if index >= len(row):
            continue
        subjects = _noncontextual_subject_identifiers(row[index])
        if subjects:
            return subjects
        owner = _canonical_table_qualifier(row[index])
        if owner:
            return (owner.upper(),)
    return _table_row_subjects(header, row)


def _table_row_qualifiers(
    header: list[str],
    row: list[str],
    *,
    metric_column: int,
    active_section: str | None,
) -> tuple[str, ...]:
    """Extract row conditions independently from the model owner.

    Dataset, modality, split and unlabeled leading section cells describe the
    measurement rather than owning it. The rule is schema-driven: textual
    cells outside model/metric/value columns become qualifiers; numeric class
    counts and empty cells do not.
    """

    qualifiers: list[str] = []
    if active_section:
        qualifiers.append(active_section)
    for index, cell in enumerate(row):
        if index == metric_column or not str(cell).strip():
            continue
        header_cell = header[index] if index < len(header) else ""
        folded_header = _fold_for_matching(header_cell)
        if (
            _MODEL_HEADER_RE.search(folded_header)
            or _METRIC_RE.search(header_cell)
            or _GENERIC_METRIC_HEADER_RE.search(folded_header)
            or _GENERIC_VALUE_HEADER_RE.search(folded_header)
        ):
            continue
        if any(
            not _looks_like_year(match)
            and not _looks_like_reference_number(str(cell), match)
            for match in _NUMBER_RE.finditer(str(cell))
        ):
            continue
        qualifier = _canonical_table_qualifier(cell)
        if qualifier and qualifier not in qualifiers:
            qualifiers.append(qualifier)
    return tuple(qualifiers)


def _claims_with_table_context(
    claims: list[MetricValueClaim],
    *,
    subjects: tuple[str, ...],
    qualifiers: tuple[str, ...],
) -> list[MetricValueClaim]:
    return [
        MetricValueClaim(
            metric=claim.metric,
            value=claim.value,
            percentage=claim.percentage,
            subjects=claim.subjects or subjects,
            text=claim.text,
            qualifiers=tuple(dict.fromkeys([*qualifiers, *claim.qualifiers])),
        )
        for claim in claims
    ]


def _claims_from_table_cells(
    *,
    metric_cell: str,
    metric_matches: list[re.Match[str]],
    value_cell: str,
    number_matches: list[re.Match[str]],
    subjects: tuple[str, ...],
    row_text: str,
    percentage_context: bool,
    extra_qualifiers: tuple[str, ...] = (),
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

    shared_header_qualifiers = _shared_composite_header_qualifiers(
        metric_cell,
        metric_matches,
    )
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
                qualifiers=tuple(
                    dict.fromkeys(
                        [
                            *_metric_qualifiers_from_table_label(
                                metric_cell,
                                metric=metric,
                                metric_match=metric_match,
                                extra_qualifiers=tuple(
                                    dict.fromkeys(
                                        [*extra_qualifiers, *shared_header_qualifiers]
                                    )
                                ),
                            ),
                            *_uncertainty_qualifiers(
                                value_cell,
                                number_match=number_match,
                                number_matches=number_matches,
                            ),
                        ]
                    )
                ),
            )
        )
    return claims


def _shared_composite_header_qualifiers(
    header: str,
    metric_matches: list[re.Match[str]],
) -> tuple[str, ...]:
    """Return a leading condition shared by every metric in one column.

    In ``LR F1 / UAR`` the ``LR`` condition applies to both values, whereas
    the ordinary local-segment parser sees it only beside F1. The prefix is
    accepted only when a column explicitly contains multiple metric labels.
    """

    if len(metric_matches) < 2:
        return ()
    prefix = header[: metric_matches[0].start()]
    qualifier = _canonical_table_qualifier(prefix)
    return (qualifier,) if qualifier else ()


def _table_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _canonical_metric(value: str) -> str | None:
    if _PARAMETER_MILLIONS_HEADER_RE.fullmatch(_fold_for_matching(value).strip()):
        return "parameters"
    match = _METRIC_RE.search(value)
    if match:
        return _metric_from_alias(match.group("metric"))
    # VAD result tables conventionally serialize Pearson correlation as
    # ``V (r)``, ``A (r)`` and ``D (r)``. A bare prose ``r`` stays ignored;
    # only a parenthesized r attached to an explicit semantic dimension is a
    # sufficiently bounded metric header.
    folded = _fold_for_matching(value)
    if re.search(r"\(\s*r\s*\)", folded) and any(
        re.search(rf"(?<!\w){alias}(?!\w)", folded)
        for alias in ("valence", "arousal", "dominance", "v", "a", "d")
    ):
        return "pearson_r"
    return None


def _metric_header_magnitude(value: str, metric: str) -> str | None:
    if metric != "parameters":
        return None
    folded = _fold_for_matching(value).strip()
    if (
        _PARAMETER_MILLIONS_HEADER_RE.fullmatch(folded)
        or re.search(
            r"(?<!\w)(?:p|params?|parameters?)\s*"
            r"\(\s*m(?:illion)?s?\s*\)",
            folded,
        )
    ):
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
    if metric_span is not None:
        between_start = min(metric_span[1], number_match.end())
        between_end = max(metric_span[0], number_match.start())
        between = text[between_start:between_end]
        if between_start < between_end and _SENTENCE_DELIMITER_RE.search(
            text,
            between_start,
            between_end,
        ):
            # A metric mentioned in the next/previous sentence cannot own a
            # nearby structural count merely because both fit the distance
            # window (``20 documents. Metrics include Recall@5``).
            return False
        if re.fullmatch(r"\s*(?:@|at)\s*", between, re.IGNORECASE):
            # Recall@5 / Hit@10 names a retrieval cutoff, not a reported
            # recall or hit value.
            return False
        if (
            metric_span[0] >= number_match.end()
            and number_match.group("magnitude")
            and re.search(r"[:,;]", between)
        ):
            # ``Model, 3.02M: miF1 .50`` uses a magnitude as compact model
            # size metadata.  It cannot be a value owned by the later metric.
            # The rule is structural and also covers K/B magnitudes without
            # guessing whether the omitted noun was parameters or samples.
            return False
        compact_between = re.sub(r"[*_`~\s]", "", between.casefold())
        if re.fullmatch(r"(?:@|at)?\d+(?:/\d+)*/", compact_between):
            # The later endpoint in ``Recall@5/10`` is part of the same cutoff
            # vector. It must not become a recall score merely because another
            # integer lies between the metric label and this endpoint.
            return False
    prefix = text[max(0, number_match.start() - 96) : number_match.start()]
    if re.search(
        r"(?:(?:decision\s+)?threshold|ngưỡng)\s*"
        r"(?:(?:fixed|decision|evaluation|global|assessment|"
        r"đánh\s*giá|danh\s*gia|cố\s*định|co\s*dinh)\s+)*"
        r"(?:[a-z]\s*)?(?:(?:is|was|là|la)\s*)?[:=]?\s*[*_`~]*\s*$|"
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
    if _STRUCTURAL_COUNT_SUFFIX_RE.search(suffix) or _STRUCTURAL_RUN_COUNT_SUFFIX_RE.search(
        suffix
    ):
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
        if _uses_aligned_slash_metrics(block, metric_matches, number_matches):
            pairs = [
                (metric_matches[index % len(metric_matches)], number_match)
                for index, number_match in enumerate(number_matches)
            ]
        elif _uses_explicit_one_to_one_alignment(
            block,
            metric_matches,
            number_matches,
        ):
            pairs = list(zip(metric_matches, number_matches))
        else:
            pairs = [
                (
                    _metric_match_for_number(
                        block,
                        metric_matches=metric_matches,
                        number_match=number_match,
                    ),
                    number_match,
                )
                for number_match in number_matches
            ]
        for metric_match, number_match in pairs:
            metric = _metric_from_alias(metric_match.group("metric"))
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
    aligned_dimension = _aligned_dimension_qualifier_before_values(
        text,
        metric_match=metric_match,
        local_numbers=local_numbers,
        number_index=number_index,
    )
    parenthetical_qualifier, parenthetical_close = _parenthesized_metric_qualifier(
        text,
        metric_match=metric_match,
    )
    leading_metric_qualifier = _leading_metric_dimension_qualifier(
        text,
        metric_match=metric_match,
    )

    # Compact qualifier vectors are the one case where a prefix intentionally
    # owns several later values. Bind them only when arity is exact; a single
    # qualifier can safely describe every value in the same metric region.
    vector_context = text[metric_match.end() : local_numbers[0].start()]
    vector_qualifiers = _prose_qualifiers_from_context(vector_context)
    vector_qualifier: tuple[str, ...] = ()
    if aligned_dimension:
        vector_qualifier = aligned_dimension
    elif len(vector_qualifiers) == len(local_numbers) and len(local_numbers) > 1:
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
    if explicit_prefix_qualifiers and not aligned_dimension:
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
    if parenthetical_close is not None and parenthetical_close >= number_match.end():
        suffix_context = text[number_match.end() : parenthetical_close]
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
    elif following_metric is not None:
        # In ``MAE .08, and arousal CCC .57`` the dimension after the
        # conjunction belongs to the next metric, not to this MAE value.
        next_dimension = re.search(
            rf"\s*,?\s*(?:and|while|whereas|và|còn)\s+"
            rf"(?=(?:{_METRIC_QUALIFIER_PATTERN})(?!\w))",
            suffix_context,
            re.IGNORECASE,
        )
        if next_dimension is not None:
            suffix_context = suffix_context[: next_dimension.start()]
    if re.match(
        r"\s*(?:,?\s*(?:for|on|in|under|cho|trên|ở)\s+)?"
        r"[^,;|/]{1,48}\s*/\s*[^,;|/]{1,48}",
        suffix_context,
        re.IGNORECASE,
    ):
        # This is an alternative-slice vector (Test1/Test2, indoor/outdoor),
        # not one compound qualifier owned by the current value.  Exact
        # positional binding is handled by
        # `_aligned_trailing_experiment_condition`.
        suffix_context = ""
    suffix_delimiter = re.search(r"[,;|/]|[.!?](?=\s|$)", suffix_context)
    if suffix_delimiter is not None:
        suffix_context = suffix_context[: suffix_delimiter.start()]
    suffix_qualifiers = _prose_qualifiers_from_context(suffix_context)
    if leading_metric_qualifier:
        # ``valence CCC .70 and arousal CCC .57`` gives each metric an
        # explicit left owner. Do not borrow the next conjunct's qualifier as
        # a postfix label for the previous value.
        suffix_qualifiers = ()
    inherited_qualifier: tuple[str, ...] = ()
    if not any(
        (
            parenthetical_qualifier,
            leading_metric_qualifier,
            vector_qualifier,
            prefix_qualifiers,
            suffix_qualifiers,
        )
    ) and not re.match(
        r"\s*(?:,?\s*(?:for|across|on|over|cho|trên)\s+)?"
        r"(?:both|each|cả\s+hai|đều)(?!\w)",
        text[number_match.end() :],
        re.IGNORECASE,
    ):
        inherited_qualifier = _inherited_metric_dimension_qualifier(
            text,
            metric_match=metric_match,
        )

    result: list[str] = []
    alias_variant = _metric_variant_from_alias(metric_match.group("metric"))
    if alias_variant and (number_index == 0 or not explicit_prefix_qualifiers):
        result.append(alias_variant)
    for qualifier in (
        *parenthetical_qualifier,
        *leading_metric_qualifier,
        *inherited_qualifier,
        *vector_qualifier,
        *prefix_qualifiers,
        *suffix_qualifiers,
    ):
        if qualifier not in result:
            result.append(qualifier)
    return tuple(result)


def _aligned_dimension_qualifier_before_values(
    text: str,
    *,
    metric_match: re.Match[str],
    local_numbers: list[re.Match[str]],
    number_index: int,
) -> tuple[str, ...]:
    """Resolve an explicit dimension vector followed by aligned values.

    This handles open prose forms such as ``CCC for V/A/D respectively
    .63/.68/.60`` without assuming a paper or a fixed metric table.  A vector
    needs at least two known dimensions, an explicit separator, and a
    distributive cue; any later value (for example ``mean .64``) remains free
    to carry its own local qualifier.
    """

    if not local_numbers:
        return ()
    context = text[metric_match.end() : local_numbers[0].start()]
    folded = _fold_for_matching(re.sub(r"[*_`~]", "", context))
    if not re.search(r"(?<!\w)(?:respectively|lần\s+lượt|lan\s+luot)(?!\w)", folded):
        return ()
    qualifier_matches = list(_METRIC_QUALIFIER_RE.finditer(context))
    if len(qualifier_matches) < 2:
        return ()
    between_vector = context[qualifier_matches[0].end() : qualifier_matches[-1].start()]
    if not re.search(r"[/,]|(?<!\w)(?:and|và)(?!\w)", between_vector, re.IGNORECASE):
        return ()
    qualifiers = tuple(
        _QUALIFIER_ALIAS_TO_CANONICAL[match.group("qualifier").casefold()]
        for match in qualifier_matches
    )
    if number_index >= len(qualifiers):
        return ()
    # The aligned prefix must cover a slash/comma-separated value vector.
    vector_numbers = local_numbers[: len(qualifiers)]
    if len(vector_numbers) != len(qualifiers) or not all(
        re.search(r"[/,]", text[left.end() : right.start()])
        for left, right in zip(vector_numbers, vector_numbers[1:])
    ):
        return ()
    return (qualifiers[number_index],)


def _inherited_metric_dimension_qualifier(
    text: str,
    *,
    metric_match: re.Match[str],
) -> tuple[str, ...]:
    """Carry a dimension across coordinated metrics in one conjunct.

    ``valence CCC .70 and MAE .08`` is a compact serialization of two table
    columns under Valence. Inheritance is allowed only when the text between
    the last full dimension and the current metric consists solely of metric
    labels, numeric values, and coordination punctuation. A sentence or
    semicolon closes the scope.
    """

    sentence_start = 0
    for boundary in _SENTENCE_DELIMITER_RE.finditer(text, 0, metric_match.start()):
        sentence_start = boundary.end()
    prefix = text[sentence_start : metric_match.start()]
    matches = [
        match
        for match in _METRIC_QUALIFIER_RE.finditer(prefix)
        if len(match.group("qualifier")) > 1
    ]
    if not matches:
        return ()
    if len(matches) > 1 and not _METRIC_RE.search(prefix):
        # A leading vector such as ``valence and arousal CCC`` names several
        # dimensions for aligned values; the last word is not a singular scope
        # to inherit.
        return ()
    last = matches[-1]
    between = prefix[last.end() :]
    if ";" in between:
        return ()
    between = _METRIC_RE.sub(" ", between)
    between = _NUMBER_RE.sub(" ", between)
    between = re.sub(
        r"(?<!\w)(?:and|plus|with|along\s+with|respectively|"
        r"và|cùng|với|đồng\s+thời)(?!\w)",
        " ",
        between,
        flags=re.IGNORECASE,
    )
    if re.sub(r"[\s*_`~,:=()\[\]/-]+", "", between):
        return ()
    raw = last.group("qualifier").casefold()
    return (_QUALIFIER_ALIAS_TO_CANONICAL[raw],)


def _leading_metric_dimension_qualifier(
    text: str,
    *,
    metric_match: re.Match[str],
) -> tuple[str, ...]:
    """Return a full semantic dimension immediately preceding a metric."""

    sentence_start = 0
    for boundary in _SENTENCE_DELIMITER_RE.finditer(text, 0, metric_match.start()):
        sentence_start = boundary.end()
    previous_metrics = list(_METRIC_RE.finditer(text, sentence_start, metric_match.start()))
    prefix_start = previous_metrics[-1].end() if previous_metrics else sentence_start
    prefix = text[prefix_start : metric_match.start()]
    matches = list(_METRIC_QUALIFIER_RE.finditer(prefix))
    if not matches:
        return ()
    if len(matches) != 1:
        # ``valence and arousal CCC`` is a multi-dimension vector, not a
        # singular prefix owner. It is handled by value alignment/uniqueness.
        return ()
    last = matches[-1]
    raw = last.group("qualifier").casefold()
    trailing = prefix[last.end() :].strip(" *_`~:-,;()/")
    if len(raw) <= 1:
        return ()
    if trailing and not re.fullmatch(
        r"(?:(?:on|for|in|under|trên|ở|cho)\s+(?:the\s+)?"
        r"[a-z0-9][a-z0-9 ._+/-]{0,64}?\s+"
        r"(?:dataset|corpus|benchmark|split)\s+)?"
        r"(?:(?:with|at|achieving|reporting|dat|voi)\s+)?"
        r"(?:a|an|the)?",
        _fold_for_matching(trailing),
        re.IGNORECASE,
    ):
        return ()
    return (_QUALIFIER_ALIAS_TO_CANONICAL[raw],)


def _parenthesized_metric_qualifier(
    text: str,
    *,
    metric_match: re.Match[str],
) -> tuple[tuple[str, ...], int | None]:
    """Bind ``Valence (CCC 0.59)`` without crossing into the next conjunct.

    The semantic label immediately before an opening parenthesis owns metrics
    inside that parenthesis.  Closing the local scope also prevents the first
    claim in ``Valence (...) and Arousal (...)`` from borrowing ``Arousal`` as
    a postfix qualifier.
    """

    open_at = text.rfind("(", 0, metric_match.start())
    if open_at < 0 or text.rfind(")", 0, metric_match.start()) > open_at:
        return (), None
    close_at = text.find(")", metric_match.end())
    if close_at < 0:
        return (), None
    prefix = text[max(0, open_at - 48) : open_at]
    matches = list(_METRIC_QUALIFIER_RE.finditer(prefix))
    if not matches:
        return (), close_at
    last = matches[-1]
    if prefix[last.end() :].strip(" *_`~:-"):
        return (), close_at
    raw = last.group("qualifier").casefold()
    if len(raw) <= 1:
        return (), close_at
    return (_QUALIFIER_ALIAS_TO_CANONICAL[raw],), close_at


def _prose_qualifiers_from_context(context: str) -> tuple[str, ...]:
    qualifiers: list[str] = []
    folded = _fold_for_matching(re.sub(r"[*_`~]", "", context))
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
        raw_label = re.split(
            r"\s+(?:using|with|where|which|that|dùng|với)\b",
            raw_label,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        if _is_generic_prose_context_label(raw_label):
            continue
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


def _is_generic_prose_context_label(value: str) -> bool:
    """Reject discourse scaffolding accidentally captured after ``in/on``.

    Dynamic evidence sections remain open-vocabulary, but phrases such as
    ``in this paper—approximately`` are not measurement dimensions. Keeping
    this linguistic filter separate from document/model names prevents a
    correct value from acquiring a fabricated qualifier while preserving real
    labels such as ``IEMOCAP`` or ``Outdoor scenes``.
    """

    folded = " ".join(_fold_for_matching(value).split())
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if not folded:
        return True
    if re.search(
        r"(?<!\w)(?:approximately|approx|about|around|roughly|nearly|"
        r"xap\s+xi|khoang)(?!\w)",
        folded,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:(?:this|the|current|present|same|our)\s+)?"
            r"(?:paper|study|work|article|document|excerpt|source|evidence|"
            r"table|figure|result|experiment)s?",
            folded,
        )
    )


def _is_composite_prose_qualifier_label(value: str) -> bool:
    folded = " ".join(_fold_for_matching(value).split())
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()
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
        explicit_subjects = {
            subject
            for subject in explicit_subjects
            if not _subject_is_measurement_context(
                subject,
                evidence_claim.qualifiers,
            )
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
        owner_subjects = {
            subject
            for subject in owner_subjects
            if not _subject_is_measurement_context(
                subject,
                evidence_claim.qualifiers,
            )
        }
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
    if ("uncertainty" in answer_claim.qualifiers) != (
        "uncertainty" in evidence_claim.qualifiers
    ):
        # An error/dispersion endpoint is not an alternative estimate of the
        # metric. Subset matching for orthogonal context must never let a
        # reported ± value support the mean itself (or vice versa).
        return False
    if answer_claim.qualifiers and evidence_claim.qualifiers:
        answer_qualifiers = set(answer_claim.qualifiers)
        evidence_qualifiers = set(evidence_claim.qualifiers)
        # One side may state additional orthogonal context (task/dataset) that
        # the other table omits. They remain compatible only when one complete
        # qualifier path contains the other. Serializers are free to preserve
        # a composite cell as one qualifier (``IEMOCAP, 5-fold, LOSO``) or as
        # several atomic qualifiers, so compare both representations. Exact
        # token containment keeps conflicting dimensions such as arousal vs
        # valence, indoor vs outdoor, or test-1 vs test-2 fail-closed.
        if not (
            answer_qualifiers.issubset(evidence_qualifiers)
            or evidence_qualifiers.issubset(answer_qualifiers)
            or _qualifier_token_path_contains(
                answer_claim.qualifiers,
                evidence_claim.qualifiers,
            )
            or _qualifier_token_path_contains(
                evidence_claim.qualifiers,
                answer_claim.qualifiers,
            )
        ):
            return False
    return True


def _qualifier_token_path_contains(
    containing: tuple[str, ...],
    contained: tuple[str, ...],
) -> bool:
    """Compare equivalent qualifier paths across serializer boundaries.

    Markdown/Docling may turn several adjacent context cells into one
    hyphenated label. Flattening only for containment preserves the existing
    rule that one side may omit orthogonal context without treating a merely
    overlapping label as equivalent.
    """

    containing_tokens = {
        token
        for qualifier in containing
        for token in re.findall(r"[a-z0-9]+", _fold_for_matching(qualifier))
    }
    contained_tokens = {
        token
        for qualifier in contained
        for token in re.findall(r"[a-z0-9]+", _fold_for_matching(qualifier))
    }
    return bool(contained_tokens and contained_tokens.issubset(containing_tokens))


def _resolve_unlabeled_metric_claim(
    claim: MetricValueClaim,
    evidence_claims: list[tuple[MetricValueClaim, frozenset[str], str]],
) -> MetricValueClaim:
    """Resolve a missing metric label only when canonical evidence is unique."""

    if claim.metric != _UNLABELED_METRIC:
        return claim
    candidate_metrics: set[str] = set()
    for evidence_claim, _identity_subjects, _document_id in evidence_claims:
        candidate = MetricValueClaim(
            metric=evidence_claim.metric,
            value=claim.value,
            percentage=claim.percentage,
            subjects=claim.subjects,
            text=claim.text,
            qualifiers=claim.qualifiers,
        )
        if _claim_measurement_matches(candidate, evidence_claim):
            candidate_metrics.add(evidence_claim.metric)
    if len(candidate_metrics) != 1:
        return claim
    return MetricValueClaim(
        metric=next(iter(candidate_metrics)),
        value=claim.value,
        percentage=claim.percentage,
        subjects=claim.subjects,
        text=claim.text,
        qualifiers=claim.qualifiers,
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
    explicit_roles = _explicit_proposed_role_subjects(claim.text)
    subjects = tuple(dict.fromkeys([*claim.subjects, *explicit_roles]))
    if not subjects:
        subjects = tuple(_identity_subject_identifiers(identity))
    return MetricValueClaim(
        metric=claim.metric,
        value=claim.value,
        percentage=claim.percentage,
        subjects=subjects,
        text=claim.text,
        qualifiers=claim.qualifiers,
    )


def _explicit_proposed_role_subjects(text: str) -> tuple[str, ...]:
    """Retain the named component behind a proposed-system role.

    ``the proposed teacher`` denotes both the paper's proposed component and
    the concrete ``TEACHER`` owner. Keeping both lets an answer use the shorter
    role name while still preventing unrelated model rows from borrowing it.
    """

    role_pattern = re.compile(
        r"(?<!\w)(?:the\s+)?(?:proposed|our)\s+"
        r"(?P<role>teacher|student|encoder|decoder|predictor|classifier|"
        r"regressor|backbone|network|architecture|model|method|system|approach)"
        r"(?!\w)",
        re.IGNORECASE,
    )
    roles = [match.group("role").upper() for match in role_pattern.finditer(text)]
    if not roles:
        return ()
    return tuple(dict.fromkeys([_PROPOSED_OWNER_SUBJECT, *roles]))


def _identity_subject_identifiers(identity: str) -> list[str]:
    """Canonicalize file extensions before lexing document identities."""

    without_extensions = re.sub(
        r"\.(?:pdf|docx?|pptx?|xlsx?|txt|md|html?)\b",
        "",
        str(identity or ""),
        flags=re.IGNORECASE,
    )
    return _subject_identifiers(without_extensions)


_SUBJECT_RE = re.compile(
    r"(?<!\w)(?:"
    # Keep a complete hyphenated identity before the acronym alternatives;
    # otherwise ``MobileNet-V4-Small`` is split into unrelated owners.
    r"[A-Z][A-Za-z0-9]*(?:[-_][A-Z0-9][A-Za-z0-9]*)+|"
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
_EXPLICIT_PROSE_OWNER_RE = re.compile(
    r"(?<!\w)(?P<owner>(?-i:[A-Z][a-z]{2,}))"
    r"(?:[*_`~]+)?\s+(?:achieves?|attains?|reports?|gets?|has|uses|"
    r"đạt|dat|có|co|dùng|dung)(?!\w)",
    re.IGNORECASE,
)
_GENERIC_PROSE_OWNER_TOKENS = {
    "APPROACH",
    "BASELINE",
    "FUSION",
    "METHOD",
    "MODEL",
    "SYSTEM",
}
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
    "WAR",
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
    "KB",
    "MB",
    "GB",
    "TB",
    "KIB",
    "MIB",
    "GIB",
    "TIB",
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
                r"(?:(?:MACRO|MICRO|WEIGHTED|UNWEIGHTED)[-_]?)?F1(?:[-_]?SCORE)?",
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
    for match in _EXPLICIT_PROSE_OWNER_RE.finditer(text):
        owner = match.group("owner")
        # The regex is case-insensitive for Vietnamese verbs, but the owner
        # itself must still look like an explicit proper/model token.  This
        # admits sentence-cased names such as Crab while rejecting ordinary
        # lowercase discourse words.
        if not owner[:1].isupper():
            continue
        canonical = owner.upper()
        owner_span = match.span("owner")
        if canonical in _BLOCKED_SUBJECTS or canonical in _GENERIC_PROSE_OWNER_TOKENS:
            continue
        if not any(
            existing_span[0] <= owner_span[0]
            and existing_span[1] >= owner_span[1]
            for _existing, existing_span in result
        ):
            result.append((canonical, owner_span))
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
    has_strong_clause_boundary = bool(
        re.search(r"[;|]", text[: number_span[0]])
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
        or (
            None
            if has_sentence_boundary or has_strong_clause_boundary
            else closest_preceding(occurrences)
        )
    )
    return (selected,) if selected else ()


def _subject_occurrence_is_contextual(text: str, span: tuple[int, int]) -> bool:
    prefix = text[max(0, span[0] - 80) : span[0]]
    plain_prefix = re.sub(r"[*_`~]", "", prefix)
    if _CONTEXTUAL_SUBJECT_PREFIX_RE.search(plain_prefix):
        return True
    suffix = text[span[1] : span[1] + 32]
    if re.search(
        r"(?<!\w)(?:quantized|quantised|exported|runtime|format|backend|"
        r"deployment|deploy|lượng\s+tử|triển\s+khai)(?!\w)",
        plain_prefix[-64:] + " " + suffix,
        re.IGNORECASE,
    ):
        # Deployment target acronyms (ONNX/TFLite/etc.) describe the runtime
        # around a size/latency claim; they are not the measured model owner.
        return True
    # Dataset/corpus lists remain contextual after coordination, e.g.
    # ``on IEMOCAP and MSP-Podcast``. Looking only at the immediately previous
    # token classified the second dataset as a model owner. Accept a bounded
    # prepositional list only when no clause/sentence boundary or predicate
    # occurs after its introducer; comparison/model ownership stays intact.
    folded_prefix = _fold_for_matching(plain_prefix)
    introductions = list(
        re.finditer(
            r"(?<!\w)(?:evaluated\s+on|on(?:\s+the)?(?:\s+dataset)?|in|using|"
            r"tren(?:\s+(?:dataset|tap\s+du\s+lieu))?|o)(?!\w)\s+",
            folded_prefix,
            re.IGNORECASE,
        )
    )
    if not introductions:
        return False
    tail = folded_prefix[introductions[-1].end() :]
    if re.search(r"[,;|.!?]", tail):
        return False
    if re.search(
        r"(?<!\w)(?:achieves?|reports?|reaches?|obtains?|outperforms?|"
        r"dat|bao\s+cao|vuot)(?!\w)",
        tail,
        re.IGNORECASE,
    ):
        return False
    return bool(re.search(r"(?<!\w)(?:and|or|va|hoac|&)(?!\w)", tail))


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
        if any(
            prior.metric == claim.metric
            and prior.value == claim.value
            and prior.percentage == claim.percentage
            and set(claim.subjects).issubset(prior.subjects)
            and set(claim.qualifiers).issubset(prior.qualifiers)
            and (
                set(claim.subjects) != set(prior.subjects)
                or set(claim.qualifiers) != set(prior.qualifiers)
            )
            and claim.text
            and claim.text in prior.text
            for prior in result
        ):
            # The schema-aware table pass precedes the cell fallback. Prefer
            # its richer owner/condition binding over an otherwise identical
            # inline claim extracted from the same cell.
            continue
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
