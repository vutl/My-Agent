from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
audit = importlib.import_module("audit_public_retrieval_frontier")


def test_composer_boundary_reports_only_the_observed_frontier():
    context = importlib.import_module("evaluate_public_agent_context")
    names = {"table:a": "a.png", "figure:b": "b.png"}
    trace = {"complete": True, "events": [
        {"evidence_id": "table:a", "reason": "character_budget"},
        {"evidence_id": "figure:b", "reason": "selected"},
        {"evidence_id": "figure:b", "reason": "duplicate_canonical_identity"}]}
    assert context.composer_artifact_boundaries(trace, names, ["a.png", "b.png", "c.png"]) == {
        "a.png": ["character_budget"], "b.png": "selected_in_composer",
        "c.png": "absent_from_composer_input"}
    assert context.composer_artifact_boundaries({**trace, "complete": False}, names, ["c.png"]) == {
        "c.png": "unobserved_in_incomplete_trace"}


def row(**patch):
    return {"case_id": "case-1", "status": "ok", "candidate_artifact_hits": {"a.png": True},
            "referred_artifact_hits": {"a.png": False}, "image_count": 0,
            "artifact_support_required_by_query": False,
            "candidate_evidence_hits": [True], "evidence_phrase_hits": [False], **patch}


def test_implicit_question_loss_is_not_excluded_or_blamed_on_vector_model():
    result = audit.audit_case(row())
    assert result["implicit_artifact_question"]
    assert result["lost_final_artifacts"] == ["a.png"]
    assert result["labels"] == ["final_context_artifact_loss", "final_context_phrase_loss"]
    assert not result["pixel_identity_verified"]


def test_artifact_name_is_not_proof_of_pixels():
    result = audit.audit_case(row(referred_artifact_hits={"a.png": True}))
    assert "artifact_name_present_without_attached_pixels" in result["labels"]
    assert not result["pixel_identity_verified"]
    assert not result["lost_final_artifacts"]


def test_parent_expansion_recovery_is_not_classified_as_schema_error():
    result = audit.audit_case(row(candidate_evidence_hits=[False], evidence_phrase_hits=[True]))
    assert "phrase_recovered_after_candidate_frontier" in result["labels"]
    assert "final_context_phrase_loss" not in result["labels"]


def test_negative_is_not_in_positive_recall_and_requires_answer_check():
    result = audit.audit_case(row(unanswerable=True))
    assert result["labels"] == ["negative_requires_e2e_abstention_check"]
    assert not result["missing_candidate_artifacts"]


@pytest.mark.parametrize("patch", [
    {"candidate_artifact_hits": {"a.png": "False"}},
    {"referred_artifact_hits": {"different.png": True}},
    {"image_count": -1}, {"image_count": True},
    {"candidate_evidence_hits": [True, False]},
    {"evidence_phrase_hits": [1]},
])
def test_malformed_telemetry_fails_closed(patch):
    with pytest.raises(ValueError):
        audit.audit_case(row(**patch))


def test_frontier_completeness_and_duplicates_fail_closed():
    report = {"selected_case_ids": ["case-1", "case-2"], "rows": [row()]}
    with pytest.raises(ValueError, match="incomplete"):
        audit.audit_report(report)
    partial = audit.audit_report(report, allow_partial=True)
    assert not partial["frontier_complete"]
    assert not partial["promotion_eligible"]
    assert partial["observed_cases"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_report({**report, "rows": [row(), row()]}, allow_partial=True)


def test_final_report_uses_the_frozen_contract_without_changing_scores():
    source = {"checkpoint_contract": {"case_ids": ["case-1"]}, "rows": [row()],
              "aggregate": {"artifact_recall": 0.5}}
    result = audit.audit_report(source)
    assert result["frontier_complete"]
    assert source["aggregate"] == {"artifact_recall": 0.5}
    assert result["no_model_calls"]
    assert not result["promotion_eligible"]
