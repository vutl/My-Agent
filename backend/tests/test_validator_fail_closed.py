"""Fail-closed regressions: a claim that loses its written owner must never
borrow another row's value through the unique-value fallback.

Each wrong-value case has a matching correct-value control in the same layout
so the guard is not satisfied by rejecting everything.
"""
import pytest

from app.services.evidence_validator import validate_answer_claims

_TABLE = "| Model | CCC avg | F1 |\n|---|---|---|\n| ModelA | 0.63 | 0.70 |\n| ModelB | 0.55 | 0.60 |\n"
_DOCS = [{"document_id": "s", "filename": "Study.pdf", "content": _TABLE}]
_VAD = [{
    "document_id": "o",
    "filename": "ORION.pdf",
    "content": (
        "| Model | CCC V | CCC A | CCC D | CCC avg |\n|---|---|---|---|---|\n"
        "| ORION | 0.74 | 0.63 | 0.42 | 0.61 |"
    ),
}]
_LOSS = [{
    "document_id": "s",
    "filename": "Study.pdf",
    "content": "| SSL | Loss | WAR |\n|---|---|---:|\n| AlphaNet | CE | 0.61 |\n| AlphaNet | MLCS | 0.72 |\n",
}]


def _valid(answer, documents=_DOCS, focus=("s",)):
    return validate_answer_claims(
        answer=answer, documents=documents, focus_document_ids=list(focus) if focus else None,
    ).valid


@pytest.mark.parametrize("answer", [
    "### Study.pdf\n**ModelB:**\nCCC avg 0.55.\n- So sánh CCC avg 0.63 và F1 0.70 với baseline.",
    "### Study.pdf\n**ModelB:**\nCCC avg 0.55.\n- Comparing CCC avg 0.63 and F1 0.70, the model is balanced.",
])
def test_cross_metric_reference_cannot_borrow_another_owner(answer):
    assert _valid(answer) is False


def test_cross_metric_reference_with_section_owner_values_is_supported():
    answer = "### Study.pdf\n**ModelB:**\nCCC avg 0.55.\n- Comparing CCC avg 0.55 and F1 0.60, the model is balanced."
    assert _valid(answer) is True


@pytest.mark.parametrize("label", ["- Model: ModelB", "- Mô hình: **ModelB**", "- System: ModelB (ours)"])
def test_list_label_value_after_colon_owns_sub_bullets(label):
    assert _valid(f"### Study.pdf\n{label}\n  - CCC avg 0.63\n") is False
    assert _valid(f"### Study.pdf\n{label}\n  - CCC avg 0.55\n") is True


def test_descriptive_bullet_body_is_still_not_an_owner():
    # "uses ModelB" describes a component; it must not become a section owner.
    answer = "### Study.pdf\n- Architecture: uses ModelB as encoder\n- ModelA: CCC avg 0.63"
    assert _valid(answer) is True


@pytest.mark.parametrize("phrase", ["w/o MLCS", "without MLCS", "không dùng MLCS", "không có MLCS", "bỏ MLCS", "removing MLCS"])
def test_negated_condition_cannot_take_the_positive_condition_value(phrase):
    assert _valid(f"AlphaNet {phrase} achieves WAR 0.72.", _LOSS, None) is False


def test_positive_condition_value_is_supported():
    assert _valid("AlphaNet with MLCS achieves WAR 0.72.", _LOSS, None) is True
    assert _valid("AlphaNet dùng MLCS đạt WAR 0.72.", _LOSS, None) is True


def test_unqualified_metric_is_ambiguous_when_evidence_has_several_dimensions():
    assert _valid("ORION achieves CCC 0.42.", _VAD, ("o",)) is False
    assert _valid("ORION achieves CCC avg 0.61.", _VAD, ("o",)) is True
    assert _valid("ORION achieves valence CCC 0.74.", _VAD, ("o",)) is True


@pytest.mark.parametrize("notation", ["F_1", "F_{1}", "$F_1$"])
def test_latex_metric_notation_is_checked(notation):
    assert _valid(f"ModelA reaches {notation} 0.99.") is False
    assert _valid(f"ModelA reaches {notation} 0.70.") is True


def test_vietnamese_achieve_is_not_a_comparison_cue():
    # "đạt" (achieve) folded to "dat" must not exempt a line from owner binding.
    answer = "### Study.pdf\n**ModelB:**\n- Mô hình đạt CCC avg 0.63 và F1 0.70."
    assert _valid(answer) is False
