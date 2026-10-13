"""Condition/owner identity tests use synthetic names, not paper answers."""

import pytest

from app.services.evidence_validator import _extract_metric_claims, validate_answer_claims


@pytest.mark.parametrize("scope", ["Test2", "FoldB", "Holdout"])
def test_dotted_metric_cells_keep_row_owner_and_scope(scope):
    source = (f"| Model | {scope}.F1 | {scope}.HL |\n|---|---|---|\n"
              "| NovaNet | 0.74 | 0.19 |\n| RivalNet | 0.68 | 0.24 |")
    documents = [{"document_id": "b", "filename": "Study.pdf", "content": source}]
    answer = ("| Model | Results |\n|---|---|\n"
              f"| NovaNet | {scope}.F1 = 0.74; {scope}.HL = 0.19 |")
    assert validate_answer_claims(answer=answer, documents=documents).valid
    for bad in (answer.replace("NovaNet", "RivalNet"),
                answer.replace(scope, "UnknownSlice"), answer.replace("0.74", "0.68")):
        assert not validate_answer_claims(answer=bad, documents=documents).valid


def test_long_packed_table_row_is_not_reparsed_as_ownerless_prose():
    source = "| Model | F1 | HL |\n|---|---|---|\n| NovaNet | 0.74 | 0.19 |"
    documents = [{"document_id": "b", "filename": "Study.pdf", "content": source}]
    row = "| NovaNet | " + "A source-supported nonnumeric description. " * 12 + " | F1 0.74; HL 0.19 |"
    answer = "| Model | Notes | Results |\n|---|---|---|\n" + row
    claims = _extract_metric_claims(answer, bind_condition_owners=True, preserve_occurrences=True)
    assert len(claims) == 2
    assert all("NOVANET" in claim.subjects for claim in claims)
    assert validate_answer_claims(answer=answer, documents=documents).valid
    assert not validate_answer_claims(answer=answer.replace("NovaNet", "UnknownNet"), documents=documents).valid


def test_clipped_row_prefix_cannot_bind_a_different_paper_cell():
    documents = [
        {"document_id": "a", "filename": "AlphaStudy.pdf", "content":
         "| Model | F1 |\n|---|---|\n| SharedNet | 0.74 |"},
        {"document_id": "b", "filename": "BetaStudy.pdf", "content":
         "| Model | F1 |\n|---|---|\n| SharedNet | 0.68 |"},
    ]
    notes = "A nonnumeric annotation. " * 20
    answer = ("| Notes | Paper | Model | F1 |\n|---|---|---|---|\n"
              f"| {notes} | AlphaStudy | SharedNet | 0.74 |\n"
              f"| {notes} | BetaStudy | SharedNet | 0.68 |")
    assert validate_answer_claims(answer=answer, documents=documents).valid
    assert not validate_answer_claims(answer=answer.replace("0.68", "0.74"), documents=documents).valid


def test_each_sentence_keeps_its_own_document_and_model_in_summary_paragraph():
    documents = [
        {"document_id": "a", "filename": "ORION.pdf", "content":
         "| Model | F1 |\n|---|---|\n| OrionNet | 0.61 |"},
        {"document_id": "b", "filename": "From_Signals_To_Nova_Fusion_Model.pdf", "content":
         "| Model | F1 |\n|---|---|\n| NovaNet | 0.74 |\n| RivalNet | 0.68 |"},
    ]
    prefix = ("### ORION.pdf\nOrionNet F1 0.61.\n"
              "### From_Signals_To_Nova_Fusion_Model.pdf\n")
    paragraph = "ORION uses an acoustic encoder. NovaNet achieves F1 0.74."
    assert validate_answer_claims(answer=prefix + paragraph, documents=documents,
                                 focus_document_ids=["a", "b"]).valid
    for bad in (paragraph.replace("0.74", "0.61"),
                paragraph.replace("NovaNet", "RivalNet"),
                "ORION: NovaNet achieves F1 0.74.",
                paragraph.replace("NovaNet", "UnknownNet")):
        assert not validate_answer_claims(answer=prefix + bad, documents=documents,
                                         focus_document_ids=["a", "b"]).valid


def test_sanitizer_drops_unsafe_sentence_even_after_summary_label():
    from app.api.agent import _remove_unsupported_claim_lines

    answer = ("### Study\n| Model | F1 |\n|---|---|\n| NovaNet | 0.74 |\n"
              "**Summary:** The architecture is compact. NovaNet F1 was 0.99.")
    documents = [{"document_id": "study", "filename": "Study.pdf", "content":
                  "| Model | F1 |\n|---|---|\n| NovaNet | 0.74 |"}]
    validation = validate_answer_claims(answer=answer, documents=documents)
    assert not validation.valid
    safe = _remove_unsupported_claim_lines(answer, validation.to_dict())
    assert "0.99" not in safe
    assert "NovaNet | 0.74" in safe
    assert validate_answer_claims(answer=safe, documents=documents).valid


@pytest.mark.parametrize("fence", ["```text", "~~~text"])
def test_diagram_scope_is_local_and_does_not_own_following_prose(fence):
    documents = [
        {"document_id": "a", "filename": "ORION.pdf", "content":
         "| Model | F1 |\n|---|---|\n| OrionNet | 0.61 |"},
        {"document_id": "b", "filename": "From_Signals_To_Nova_Fusion_Model.pdf", "content":
         "| Model | F1 |\n|---|---|\n| NovaNet | 0.74 |\n| RivalNet | 0.68 |"},
    ]
    prefix = ("### ORION.pdf\nOrionNet F1 0.61.\n"
              "### From_Signals_To_Nova_Fusion_Model.pdf\nNovaNet F1 0.74.\n"
              f"{fence}\nORION: Audio -> Encoder -> Classifier\n"
              f"The other pipeline: Audio -> Fusion\n{fence[:3]}\n")
    correct = prefix + "NovaNet achieves F1 0.74."
    assert validate_answer_claims(answer=correct, documents=documents,
                                 focus_document_ids=["a", "b"]).valid
    for bad in (prefix + "NovaNet achieves F1 0.61.",
                prefix + "RivalNet achieves F1 0.74.",
                prefix + "ORION: OrionNet achieves F1 0.74.",
                prefix + "OrionNet achieves F1 0.74."):
        assert not validate_answer_claims(answer=bad, documents=documents,
                                         focus_document_ids=["a", "b"]).valid
    inside_wrong = prefix.replace("ORION: Audio -> Encoder -> Classifier",
                                 "ORION: OrionNet F1 0.74.")
    assert not validate_answer_claims(answer=inside_wrong, documents=documents,
                                     focus_document_ids=["a", "b"]).valid


VECTORIZED = """| SSL | Loss | WAR | UAR |
|---|---|---:|---:|
| AlphaNet | CE MLCS | 0.61 0.72 | 0.50 0.63 |
| BetaNet | CE MLCS | 0.64 0.75 | 0.53 0.66 |
"""


def docs(content=VECTORIZED):
    return [{"document_id": "study", "filename": "Study.pdf", "content": content}]


def test_loss_vector_pairs_each_measurement_without_promoting_loss_to_owner():
    claims = _extract_metric_claims(VECTORIZED)
    assert [(c.metric, c.value, c.subjects, c.qualifiers) for c in claims] == [
        ("war", "0.61", ("ALPHANET",), ("ce",)),
        ("war", "0.72", ("ALPHANET",), ("mlcs",)),
        ("uar", "0.5", ("ALPHANET",), ("ce",)),
        ("uar", "0.63", ("ALPHANET",), ("mlcs",)),
        ("war", "0.64", ("BETANET",), ("ce",)),
        ("war", "0.75", ("BETANET",), ("mlcs",)),
        ("uar", "0.53", ("BETANET",), ("ce",)),
        ("uar", "0.66", ("BETANET",), ("mlcs",)),
    ]


@pytest.mark.parametrize("condition", ["AlphaNet + MLCS", "AlphaNet with MLCS"])
def test_owner_inside_condition_column_is_evidence_bound(condition):
    answer = f"""| Model | Condition | Metric | Value |
|---|---|---|---:|
| Study experiment | {condition} | WAR | 0.72 |
| Study experiment | {condition} | UAR | 0.63 |
"""
    assert validate_answer_claims(answer=answer, documents=docs(), focus_document_ids=["study"]).valid
    for bad in [answer.replace("MLCS", "CE"), answer.replace("AlphaNet", "BetaNet"),
                answer.replace("AlphaNet", "UnknownNet"), answer.replace("MLCS", "UNSEEN")]:
        assert not validate_answer_claims(answer=bad, documents=docs(), focus_document_ids=["study"]).valid


def test_loss_named_in_prose_is_a_condition_not_a_model():
    answer = "Study: AlphaNet with MLCS achieves WAR 0.72 and UAR 0.63."
    assert validate_answer_claims(answer=answer, documents=docs(), focus_document_ids=["study"]).valid
    assert not validate_answer_claims(answer=answer.replace("MLCS", "CE"), documents=docs()).valid


def test_declared_model_rowspan_carries_owner_but_not_across_table_boundary():
    source = """| Model | Loss | WAR |
|---|---|---:|
| AlphaNet | CE | 0.61 |
| | MLCS | 0.72 |

| Model | Loss | WAR |
|---|---|---:|
| | MLCS | 0.75 |
"""
    claims = _extract_metric_claims(source)
    assert claims[1].subjects == ("ALPHANET",)
    assert "ALPHANET" not in claims[2].subjects
    assert validate_answer_claims(answer="AlphaNet with MLCS WAR 0.72.", documents=docs(source)).valid
    assert not validate_answer_claims(answer="AlphaNet with MLCS WAR 0.75.", documents=docs(source)).valid


@pytest.mark.parametrize("values", ["0.72", "0.61 0.72 0.83"])
def test_ambiguous_vector_length_never_authorizes_one_specific_loss(values):
    source = f"""| Model | Loss | WAR |
|---|---|---:|
| AlphaNet | CE MLCS | {values} |
"""
    for loss in ["CE", "MLCS"]:
        result = validate_answer_claims(answer=f"AlphaNet with {loss} WAR 0.72.", documents=docs(source))
        assert not result.valid


def test_dataset_and_loss_have_independent_matching_axes():
    source = """| Model | Dataset | Loss | WAR |
|---|---|---|---:|
| AlphaNet | Corpus-A | CE MLCS | 0.61 0.72 |
| AlphaNet | Corpus-B | CE MLCS | 0.64 0.75 |
"""
    answer = """| Model | Dataset | Loss | WAR |
|---|---|---|---:|
| AlphaNet | Corpus-A | MLCS | 0.72 |
"""
    assert validate_answer_claims(answer=answer, documents=docs(source)).valid
    assert not validate_answer_claims(answer=answer.replace("Corpus-A", "Corpus-B"), documents=docs(source)).valid


@pytest.mark.parametrize("loss,value", [("CE", "0.61"), ("MLCS", "0.72")])
def test_unknown_prose_encoder_cannot_borrow_known_encoder_result(loss, value):
    answer = f"UnknownNet with {loss} reaches WAR {value}."
    assert not validate_answer_claims(answer=answer, documents=docs()).valid


@pytest.mark.parametrize("phrase", ["without MLCS", "excluding MLCS", "không có MLCS"])
def test_negated_configuration_cannot_authorize_positive_configuration(phrase):
    assert not validate_answer_claims(
        answer=f"AlphaNet {phrase} achieves WAR 0.72.", documents=docs(),
    ).valid


def test_metric_named_loss_remains_configuration_not_result_metric():
    source = VECTORIZED.replace("CE MLCS", "MSE FOCAL")
    assert validate_answer_claims(answer="AlphaNet using MSE reaches WAR 0.61.", documents=docs(source)).valid
    assert not validate_answer_claims(answer="AlphaNet using MSE reaches WAR 0.72.", documents=docs(source)).valid
    assert validate_answer_claims(answer="AlphaNet using FOCAL reaches WAR 0.72.", documents=docs(source)).valid


def test_multiple_documents_resolve_local_encoder_and_loss_without_numeric_lookup():
    documents = [*docs(), {"document_id": "second", "filename": "Second.pdf", "content": VECTORIZED.replace("AlphaNet", "GammaNet").replace("BetaNet", "DeltaNet")}]
    assert validate_answer_claims(answer="AlphaNet with MLCS achieves WAR 0.72.", documents=documents).valid
    assert not validate_answer_claims(answer="AlphaNet with CE achieves WAR 0.72.", documents=documents).valid
    assert not validate_answer_claims(answer="UnknownNet with MLCS achieves WAR 0.72.", documents=documents).valid


def test_each_clause_keeps_its_own_configuration():
    answer = "AlphaNet with CE reaches WAR 0.61; BetaNet with MLCS reaches WAR 0.75."
    assert validate_answer_claims(answer=answer, documents=docs()).valid
    for wrong in [answer.replace("with CE", "with MLCS"), answer.replace("BetaNet with MLCS", "BetaNet with CE")]:
        assert not validate_answer_claims(answer=wrong, documents=docs()).valid


def test_prose_document_identity_does_not_hide_explicit_wrong_encoder():
    assert validate_answer_claims(answer="Study: AlphaNet with MLCS WAR 0.72.", documents=docs()).valid
    assert not validate_answer_claims(answer="Study: UnknownNet with MLCS WAR 0.72.", documents=docs()).valid
    assert not validate_answer_claims(answer="Study: BetaNet with MLCS WAR 0.72.", documents=docs()).valid


def test_blank_row_and_repeated_header_end_owner_continuation():
    source = """| Model | Loss | WAR |
|---|---|---:|
| AlphaNet | CE | 0.61 |
| | | |
| | MLCS | 0.72 |
| Model | Loss | WAR |
| | MLCS | 0.75 |
"""
    claims = _extract_metric_claims(source)
    assert "ALPHANET" not in claims[1].subjects
    assert "ALPHANET" not in claims[2].subjects


def test_configuration_name_can_be_owner_in_another_document():
    documents = [*docs(), {"document_id": "other", "filename": "Other.pdf", "content": "| Model | WAR |\n|---|---:|\n| MLCS | 0.72 |\n"}]
    assert validate_answer_claims(answer="Study: AlphaNet with MLCS WAR 0.72.", documents=documents).valid
    assert not validate_answer_claims(answer="Study: AlphaNet with CE WAR 0.72.", documents=documents).valid


def test_explicit_negative_source_configuration_is_preserved():
    source = "| Model | Condition | WAR |\n|---|---|---:|\n| AlphaNet | without MLCS | 0.72 |\n"
    assert validate_answer_claims(answer="AlphaNet without MLCS WAR 0.72.", documents=docs(source)).valid
    assert not validate_answer_claims(answer="AlphaNet with MLCS WAR 0.72.", documents=docs(source)).valid


def test_uppercase_subscript_component_has_one_canonical_identity():
    source = "Table 1: WA results.\n| Fold | H s | H t |\n|---|---:|---:|\n| Avg | 0.61 | 0.72 |\n"
    assert validate_answer_claims(answer="H_s WA 0.61.", documents=docs(source)).valid
    assert not validate_answer_claims(answer="H_t WA 0.61.", documents=docs(source)).valid


def test_loss_only_table_does_not_turn_loss_into_model_owner():
    source = "| Loss | WAR |\n|---|---:|\n| CE | 0.61 |\n| MLCS | 0.72 |\n"
    claims = _extract_metric_claims(source)
    assert claims[1].subjects == ()
    assert claims[1].qualifiers == ("mlcs",)
    assert validate_answer_claims(answer="Study with MLCS reaches WAR 0.72.", documents=docs(source)).valid
    assert not validate_answer_claims(answer="Study with CE reaches WAR 0.72.", documents=docs(source)).valid


@pytest.mark.parametrize("missing", ["Not provided", "Unknown", "Không nêu trong excerpt"])
def test_missing_context_cell_is_not_an_experimental_condition(missing):
    source = "| Model | Dataset | WAR |\n|---|---|---:|\n| AlphaNet | Corpus-A | 0.72 |\n"
    answer = f"| Model | Dataset | WAR |\n|---|---|---:|\n| AlphaNet | {missing} | 0.72 |\n"
    assert validate_answer_claims(answer=answer, documents=docs(source)).valid
    assert not validate_answer_claims(answer=answer.replace("AlphaNet", "UnknownNet"), documents=docs(source)).valid


@pytest.mark.parametrize("unit", ["hr", "hours", "minutes"])
def test_training_duration_is_not_a_following_accuracy_measurement(unit):
    # AlphaNet reports WAR under CE and MLCS; an unconditioned 0.72 is ambiguous.
    answer = f"AlphaNet Large (316.62M, 94k {unit}) with MLCS reaches WAR 0.72 and UAR 0.63."
    claims = _extract_metric_claims(answer)
    assert not any(c.value == "94000" for c in claims)
    # Pretraining duration is stated in the source; grounding requires it.
    source = VECTORIZED + f"\n| Model | Params (M) | Pretraining |\n|---|---:|---|\n| AlphaNet | 316.62 | 94k {unit} |\n"
    assert validate_answer_claims(answer=answer, documents=docs(source)).valid
