"""Review regressions: quantities are checked even outside the metric lexicon."""
import pytest

from app.services.evidence_validator import validate_answer_claims
from app.services.fact_extraction_service import validate_fact
from app.services.numeric_grounding import ground_answer_numbers


def _docs(text):
    return [{"document_id": "manual", "filename": "manual.txt", "content": text}]


@pytest.mark.parametrize("first, second", [("County A", "County B"), ("Region X", "Region Y")])
def test_explicit_prose_subject_and_counted_measure_are_bound(first, second):
    documents = _docs(f"{first} reported 1284 confirmed cases and 37 hospitalizations; "
                      f"{second} reported 962 confirmed cases and 51 hospitalizations.")
    for answer in (f"{second} reported 37 hospitalizations.",
                   f"{second} reported 51 confirmed cases.",
                   f"{first} and {second} reported 962 and 1284 confirmed cases, respectively."):
        assert not ground_answer_numbers(answer, documents).valid, answer
    for answer in (f"{second} reported 51 hospitalizations.",
                   f"{first} reported 1284 confirmed cases and 37 hospitalizations.",
                   f"{first} and {second} reported 1284 and 962 confirmed cases, respectively."):
        assert ground_answer_numbers(answer, documents).valid, answer


def test_prose_counted_noun_allows_omitted_descriptor_without_swapping_owner():
    documents = _docs("Zone A reported 1284 confirmed influenza cases and 37 hospitalizations; "
                      "Zone B reported 962 confirmed cases and 51 hospitalizations.")
    assert ground_answer_numbers(
        "Zone A and Zone B reported 1,284 and 962 confirmed cases, respectively.", documents,
    ).valid
    assert not ground_answer_numbers("Zone B reported 51 confirmed cases.", documents).valid


def test_vietnamese_explicit_prose_owners_keep_quantity_attribution():
    documents = _docs("Khu A ghi nhận 120 ca nhiễm và 17 ca nhập viện; "
                      "Khu B ghi nhận 85 ca nhiễm và 23 ca nhập viện.")
    assert ground_answer_numbers("Khu B ghi nhận 23 ca nhập viện.", documents).valid
    assert not ground_answer_numbers("Khu B ghi nhận 17 ca nhập viện.", documents).valid
    assert not ground_answer_numbers("Khu B ghi nhận 23 ca nhiễm.", documents).valid


@pytest.mark.parametrize("answer, evidence, question", [
    ("Annual revenue was 999 million.", "No amount is stated in this report.", None),
    ("The specified torque is 9 Nm.", "The specified torque is 4 Nm.", None),
    ("Annual revenue was 999 million.", "Annual revenue was 438 million.", "Was annual revenue 999 million?"),
    ("Annual revenue was 2025 million.", "Annual revenue was 438 million.", None),
    ("The company was founded in 2025.", "The company was founded in 2024.", None),
    ("The specified torque is 9 Nm.", "", None),
])
def test_unverified_quantities_are_not_success(answer, evidence, question):
    result = validate_answer_claims(answer=answer, documents=_docs(evidence), question=question)
    assert not result.valid
    assert result.unsupported_claims


@pytest.mark.parametrize("answer", [
    "The specified torque is 4 Nm.",
    "1. The specified torque is 4 Nm.",
    "The company was founded in 2024.",
    "Annual revenue was 438 million.",
])
def test_matching_correct_control_still_passes(answer):
    evidence = "The specified torque is 4 Nm. The company was founded in 2024. Annual revenue was 438 million."
    assert validate_answer_claims(answer=answer, documents=_docs(evidence)).valid


def test_fact_value_is_a_whole_number_token():
    raw = {"entity": "Alpha", "attribute": "count", "conditions": [], "value": "20",
           "quote": "Alpha count was 120."}
    assert validate_fact(raw, raw["quote"]) is None


def test_fact_cannot_assign_beta_quote_to_alpha():
    source = "Alpha revenue was 100 million. Beta revenue was 200 million."
    raw = {"entity": "Alpha", "attribute": "revenue", "conditions": [], "value": "200",
           "quote": "Beta revenue was 200 million."}
    assert validate_fact(raw, source) is None
    result = ground_answer_numbers("Alpha revenue was 200 million.", [
        {**_docs(source)[0], "facts": [raw]},
    ])
    assert not result.valid


def test_same_value_in_one_correct_claim_does_not_license_another():
    documents = _docs("| Region | Revenue | Staff |\n|---|---|---|\n| North | 45 | 30 |\n| South | 30 | 45 |")
    assert not validate_answer_claims(
        answer="North revenue was 45. South revenue was 45.", documents=documents,
    ).valid


@pytest.mark.parametrize("answer, evidence", [
    ("The balance is 45 dollars.", "The balance is -45 dollars."),
    ("The dose is 5 mg.", "The dose is 0.05 mg."),
    ("The dose is 50 g.", "The dose is 50 mg."),
    ("The incidence is 50%.", "The incidence is 0.5 cases per patient."),
    ("The value is 0.05.", "The value is <0.05."),
])
def test_sign_unit_scale_and_comparator_are_part_of_the_claim(answer, evidence):
    assert not ground_answer_numbers(answer, _docs(evidence)).valid


def test_result_records_checked_occurrences_not_just_a_set_of_values():
    result = ground_answer_numbers("North revenue was 45. South revenue was 45.",
                                  _docs("| Region | Revenue |\n|---|---|\n| North | 45 |\n| South | 30 |"))
    assert len(result.assessments) == 2
    assert result.assessments[0].status == "supported"
    assert result.assessments[1].status == "contradicted"
    assert result.assessments[0].claim_id != result.assessments[1].claim_id
    assert result.assessments[0].evidence_locators


def test_repeated_identical_sentences_have_distinct_offsets():
    result = ground_answer_numbers("Revenue was 45. Revenue was 45.", _docs("Revenue was 45."))
    assert [a.start for a in result.assessments] == [12, 28]
    assert len({a.claim_id for a in result.assessments}) == 2


def test_fact_does_not_hide_an_unrelated_occurrence_of_same_value():
    source = "Alpha revenue was 45 million. Beta staff count was 45."
    fact = {"entity": "Alpha", "attribute": "revenue", "value": "45",
            "quote": "Alpha revenue was 45 million."}
    result = ground_answer_numbers("Beta staff count was 45.", [{**_docs(source)[0], "facts": [fact]}])
    assert result.valid


def test_supported_metric_does_not_license_same_value_in_unknown_quantity():
    result = validate_answer_claims(answer="Alpha F1 was 45. Revenue was 45 million.",
                                   documents=_docs("Alpha F1 was 45. Revenue was 30 million."))
    assert not result.valid
    assert any(c.metric == "number" and "Revenue" in c.text for c in result.unsupported_claims)


def test_correct_metric_cell_does_not_override_wrong_same_value_cell():
    result = validate_answer_claims(answer="| Model | F1 | Recall |\n|---|---|---|\n| Alpha | 45 | 45 |",
        documents=_docs("| Model | F1 | Recall |\n|---|---|---|\n| Alpha | 45 | 30 |"))
    assert not result.valid


def test_unknown_count_is_checked_even_next_to_a_supported_metric():
    result = validate_answer_claims(answer="Alpha F1 was 45, using 9 layers.",
                                   documents=_docs("Alpha F1 was 45, using 4 layers."))
    assert not result.valid


@pytest.mark.parametrize("header", ["Paper/model and benchmark", "Model and evaluation"])
def test_composite_benchmark_cell_preserves_model_and_split(header):
    source = [{"document_id": "alpha", "filename": "AlphaStudy.pdf",
        "caption": "Benchmark results on Corpus-X test sets.",
        "content": "| Model | Test1.miF1 | Test2.miF1 |\n|---|---|---|\n| Lynx | 0.510 | 0.514 |"}]
    answer = f"| {header} | Result |\n|---|---|\n| Lynx — Corpus-X Test2 | miF1 0.514 |"
    assert validate_answer_claims(answer=answer, documents=source).valid
    assert not validate_answer_claims(answer=answer.replace("Test2", "Test1"), documents=source).valid
    assert not validate_answer_claims(answer=answer.replace("Lynx", "UnknownModel"), documents=source).valid


def test_comparison_table_local_owner_is_not_overwritten_by_prior_paper_heading():
    sources = [{"document_id": "a", "filename": "AlphaStudy.pdf",
        "caption": "Results on Corpus-X.",
        "content": "| Model | CCC avg |\n|---|---|\n| Lynx | 0.638 |"},
        {"document_id": "b", "filename": "BetaStudy.pdf",
         "caption": "Results on Corpus-Y.",
         "content": "| Model | miF1 |\n|---|---|\n| Falcon | 0.514 |"}]
    answer = ("### AlphaStudy.pdf\n| Paper / model | Dataset | Result |\n|---|---|---|\n"
              "| Lynx | Corpus-X | CCC avg 0.638 |\n| Falcon | Corpus-Y | miF1 0.514 |")
    options = {"documents": sources, "focus_document_ids": ["a", "b"], "require_all_focus_documents": True}
    assert validate_answer_claims(answer=answer, **options).valid
    assert not validate_answer_claims(answer=answer.replace("Falcon", "Lynx"), **options).valid
    assert not validate_answer_claims(answer=answer.replace("Corpus-Y", "Corpus-X"), **options).valid
    assert not validate_answer_claims(answer=answer.replace("| Falcon |", "| AlphaStudy — Falcon |"), **options).valid
    assert not validate_answer_claims(answer=answer, documents=sources, focus_document_ids=["a"]).valid


def test_packed_measurement_cells_take_precedence_over_preceding_prose_metric():
    sources = [{"document_id": "a", "filename": "AlphaStudy.pdf",
                "content": "| Model | CCC avg |\n|---|---|\n| Lynx | 0.638 |"},
               {"document_id": "b", "filename": "BetaStudy.pdf",
                "content": "| Model | Test2.miF1 | Test2.maF1 |\n|---|---|---|\n| Falcon | 0.514 | 0.258 |"}]
    answer = ("### AlphaStudy.pdf\n### BetaStudy.pdf\n"
              "The selection metric is miF1.\n"
              "| Paper/model | Dataset/condition | Main results |\n|---|---|---|\n"
              "| Lynx | Corpus-X | CCC avg 0.638 |\n"
              "| Falcon | Corpus-Y Test2 | miF1 0.514; maF1 0.258 |")
    options = {"documents": sources, "focus_document_ids": ["a", "b"],
               "require_all_focus_documents": True}
    assert validate_answer_claims(answer=answer, **options).valid
    assert not validate_answer_claims(answer=answer.replace("CCC avg 0.638", "CCC avg 0.514"), **options).valid
    assert not validate_answer_claims(answer=answer.replace("maF1 0.258", "maF1 0.514"), **options).valid


def test_separate_paper_and_model_columns_keep_both_ownership_constraints():
    sources = [{"document_id": "a", "filename": "AlphaStudy.pdf",
                "content": "| Model | miF1 |\n|---|---|\n| Lynx | 0.638 |"},
               {"document_id": "b", "filename": "BetaStudy.pdf",
                "content": "| Model | miF1 |\n|---|---|\n| Falcon | 0.514 |\n| Raven | 0.481 |"}]
    answer = ("| Paper | Model | Metric | Value |\n|---|---|---|---|\n"
              "| BetaStudy.pdf | Falcon | miF1 | 0.514 |")
    assert validate_answer_claims(answer=answer, documents=sources).valid
    assert not validate_answer_claims(answer=answer.replace("Falcon", "Raven"), documents=sources).valid
    assert not validate_answer_claims(answer=answer.replace("BetaStudy", "AlphaStudy"), documents=sources).valid


def test_shortened_paper_cell_resolves_identity_without_erasing_model_ownership():
    sources = [{"document_id": "a", "filename": "AlphaStudy.pdf",
                "content": "| Model | Test2.miF1 |\n|---|---|\n| RavenNet | 0.638 |"},
               {"document_id": "b", "filename": "From_Signals_To_Aster_Based_Dynamic_Fusion_Model.pdf",
                "content": "| Model | Test2.miF1 |\n|---|---|\n| LynxNet | 0.514 |\n| RavenNet | 0.481 |"}]
    answer = ("### AlphaStudy.pdf\n| Paper | Model | Split | Metric | Value |\n|---|---|---|---|---|\n"
              "| Aster-Based Dynamic Fusion Model | LynxNet | Test2 | miF1 | 0.514 |")
    assert validate_answer_claims(answer=answer, documents=sources).valid
    for wrong in (answer.replace("LynxNet", "RavenNet"),
                  answer.replace("Aster-Based Dynamic Fusion Model", "AlphaStudy.pdf"),
                  answer.replace("Aster-Based Dynamic Fusion Model", "Unknown New Study")):
        assert not validate_answer_claims(answer=wrong, documents=sources).valid
    collision = {"document_id": "c", "filename": "Other_Aster_Based_Dynamic_Fusion_Model.pdf",
                 "content": "| Model | Test2.miF1 |\n|---|---|\n| LynxNet | 0.514 |"}
    assert not validate_answer_claims(answer=answer, documents=[*sources, collision]).valid


def test_paper_binding_is_provenance_even_without_acronym_identity_tokens():
    sources = [{"document_id": "a", "filename": "first.md",
                "content": "| Model | F1 |\n|---|---|\n| LynxNet | 0.7 |"},
               {"document_id": "b", "filename": "second.md",
                "content": "| Model | F1 |\n|---|---|\n| LynxNet | 0.8 |"}]
    answer = "| Paper | Model | F1 |\n|---|---|---|\n| first.md | LynxNet | 0.7 |"
    result = validate_answer_claims(answer=answer, documents=sources)
    assert result.valid
    assert result.checked_claims[0].to_dict()["document_id"] == "a"
    assert not validate_answer_claims(answer=answer.replace("0.7", "0.8"), documents=sources).valid


@pytest.mark.parametrize("separator", [" — ", " / "])
def test_declared_composite_owner_scope_supports_structural_cell_separators(separator):
    source = [{"document_id": "b", "filename": "BetaStudy.pdf",
               "caption": "Benchmark results on Corpus-X multi-label test sets.",
               "content": "| Model | Test1.miF1 | Test2.miF1 |\n|---|---|---|\n| LynxNet | 0.510 | 0.514 |\n| RavenNet | 0.514 | 0.520 |"}]
    answer = ("| Model / dataset | miF1 |\n|---|---|\n"
              f"| LynxNet{separator}Corpus-X Test2 | 0.514 |")
    assert validate_answer_claims(answer=answer, documents=source).valid
    for wrong in (answer.replace("LynxNet", "RavenNet"),
                  answer.replace("Test2", "Test1"), answer.replace("Corpus-X", "Corpus-Y")):
        assert not validate_answer_claims(answer=wrong, documents=source).valid


def test_threshold_before_packed_metrics_remains_a_checked_condition_not_a_metric():
    source = [{"document_id": "b", "filename": "BetaStudy.pdf",
        "caption": "Results on Corpus-Y use a fixed threshold t = 0.45.",
        "content": "| Model | Test1.miF1 | Test2.miF1 |\n|---|---|---|\n| Falcon | 0.510 | 0.514 |"}]
    answer = "### BetaStudy.pdf\n**Falcon:**\nFalcon — Corpus-Y Test2, t = 0.45 | miF1 0.514."
    assert validate_answer_claims(answer=answer, documents=source).valid
    assert not validate_answer_claims(answer=answer.replace("0.45", "0.46"), documents=source).valid
    assert not validate_answer_claims(answer=answer.replace("Test2", "Test1"), documents=source).valid


@pytest.mark.parametrize("caveat", [
    "Không nên đối chiếu trực tiếp CCC 0.638 với miF1 0.514.",
    "Không nên so trực tiếp 0.638 với 0.514.",
    "Do not compare 0.638 with 0.514.",
])
def test_cross_source_reference_does_not_inherit_last_heading(caveat):
    source = [{"document_id": "a", "filename": "AlphaStudy.pdf",
        "content": "| Model | CCC avg |\n|---|---|\n| AlphaStudy | 0.638 |"},
        {"document_id": "b", "filename": "BetaStudy.pdf",
        "content": "| Model | miF1 |\n|---|---|\n| BetaStudy | 0.514 |"}]
    answer = "### BetaStudy\n" + caveat
    assert validate_answer_claims(answer=answer, documents=source).valid
    assert not validate_answer_claims(answer=answer.replace("0.514", "0.555"), documents=source).valid


def test_reference_cue_never_discards_explicit_wrong_owner():
    source = _docs("| Model | Revenue |\n|---|---|\n| Alpha | 45 |\n| Beta | 30 |")
    assert not ground_answer_numbers("Do not compare Alpha revenue 30 with Beta revenue 45.", source).valid


def test_inline_unknown_quantity_in_answer_table_is_checked():
    source = _docs("| Item | Specification |\n|---|---|\n| Alpha | Required torque is 4 Nm |")
    answer = "| Item | Specification |\n|---|---|\n| Alpha | Required torque is 9 Nm |"
    assert not validate_answer_claims(answer=answer, documents=source).valid
    assert validate_answer_claims(answer=answer.replace("9 Nm", "4 Nm"), documents=source).valid


def test_rescued_metric_occurrence_records_actual_source():
    result = validate_answer_claims(answer="Alpha accuracy was 82.5%.",
                                   documents=_docs("Alpha accuracy was 0.825."))
    assert result.valid
    supported = [a for a in result.numeric_assessments if a["status"] == "supported"]
    assert supported and supported[0]["evidence_locators"][0]["document_id"] == "manual"
    assert supported[0]["evidence_locators"][0]["source_claim"]["value"] == "0.825"


def test_repeated_supported_metric_occurrences_are_counted_not_collapsed():
    documents = _docs("Alpha accuracy was 0.825.")
    answer = "Alpha accuracy was 82.5%. Alpha accuracy was 82.5%."
    result = validate_answer_claims(answer=answer, documents=documents)
    assert result.valid
    assert len(result.numeric_assessments) == 2
    assert all(a["status"] == "supported" for a in result.numeric_assessments)
    # Repeated metric support does not lend a value to a different quantity.
    assert not validate_answer_claims(
        answer=answer + " Alpha recall was 82.5%.", documents=documents,
    ).valid


def test_ocr_spaced_decimal_keeps_literal_quote_and_number_offsets():
    source = "The fixed threshold t = 0 . 45 . A different score was 0.46."
    documents = _docs(source)
    result = ground_answer_numbers("The fixed threshold t = 0.45.", documents)
    assert result.valid
    locator = result.assessments[0].evidence_locators[0]
    assert locator["quote"][locator["number_start"]:locator["number_end"]] == "0 . 45"
    assert not ground_answer_numbers("The fixed threshold t = 0.46.", documents).valid
    # Ordinary whitespace and a sentence/list period do not merge integers.
    assert not ground_answer_numbers("The count was 75.123.", _docs("The counts were 75 123.")).valid
    assert not ground_answer_numbers("The count was 3.12.", _docs("The count was 3. 12 samples were excluded.")).valid


@pytest.mark.parametrize("filename", ["BetaStudy (7).pdf", "BetaStudy_2024.pdf"])
def test_numbered_document_heading_replaces_previous_document_scope(filename):
    documents = [
        {"document_id": "a", "filename": "AlphaStudy.pdf",
         "content": "The fixed threshold t = 0.30."},
        {"document_id": "b", "filename": filename,
         "content": "The fixed threshold t = 0 . 45 ."},
    ]
    answer = f"### AlphaStudy.pdf\n### {filename}\nThe fixed threshold t = 0.45."
    assert ground_answer_numbers(answer, documents).valid
    assert not ground_answer_numbers(answer.replace("t = 0.45", "t = 0.30"), documents).valid
    # A quantity printed alongside the document name is still checked.
    assert not ground_answer_numbers(
        answer.replace(f"### {filename}", f"### {filename} — threshold t = 0.99"), documents,
    ).valid


@pytest.mark.parametrize("condition", ["Week", "Dose", "Iteration"])
def test_numeric_row_bindings_are_not_independent_value_matches(condition):
    source = _docs(f"| Group | {condition} | Output |\n|---|---|---|\n"
                   "| Amber | 7 | 31.4 |\n| Amber | 14 | 35.8 |\n| Violet | 7 | 35.8 |")
    assert ground_answer_numbers(f"Amber output was 35.8 at {condition} 14.", source).valid
    assert not ground_answer_numbers(f"Amber output was 35.8 at {condition} 7.", source).valid
    assert not ground_answer_numbers(f"Violet output was 31.4 at {condition} 7.", source).valid


def test_numeric_binding_cannot_license_a_value_from_another_column():
    source = _docs("| Site | Revenue | Staff |\n|---|---|---|\n| East | 45 | 30 |\n| West | 30 | 45 |")
    assert not ground_answer_numbers("West revenue was 45.", source).valid


@pytest.mark.parametrize("group", ["Test", "Validation"])
def test_omitted_group_is_ambiguous_but_named_group_and_equal_values_are_valid(group):
    source = _docs(f"| Region | Base | | {group} | |\n"
                   "| | Cost | Days | Cost | Days |\n|---|---|---|---|---|\n| East | 120 | 8 | 170 | 8 |")
    assert not ground_answer_numbers("East cost was 170.", source).valid
    assert ground_answer_numbers(f"East {group} cost was 170.", source).valid
    assert ground_answer_numbers("East days were 8.", source).valid


def test_bullet_owner_is_inherited_by_later_numbered_clauses_only_when_not_overridden():
    source = _docs("- Team: Silver\n- Points: 39\n- Goals scored: 52\n\n"
                   "- Team: Gold\n- Points: 28\n- Goals scored: 41")
    assert ground_answer_numbers("- Silver: 39 points, 52 goals scored\n- Gold: 28 points, 41 goals scored", source).valid
    assert not ground_answer_numbers("- Gold: 28 points, 52 goals scored", source).valid
    assert ground_answer_numbers("Silver scored 52 goals, while Gold scored 41 goals.", source).valid


@pytest.mark.parametrize("obj", ["adapter", "insurance", "taxes"])
def test_inclusion_qualifier_belongs_to_its_local_quantity(obj):
    source = _docs(f"The package costs 73 dollars excluding {obj} and 89 dollars including {obj}.")
    assert ground_answer_numbers(f"The package costs 73 dollars excluding {obj}.", source).valid
    assert ground_answer_numbers(f"Including {obj}, the package costs 89 dollars.", source).valid
    assert not ground_answer_numbers(f"The package costs 73 dollars including {obj}.", source).valid
    assert not ground_answer_numbers(f"The package costs 89 dollars without {obj}.", source).valid
    assert not ground_answer_numbers("The package costs 89 dollars.", source).valid


@pytest.mark.parametrize("qualifier", ["chưa bao gồm", "chưa", "đã gồm"])
def test_vietnamese_inclusion_binding_is_literal_and_value_specific(qualifier):
    source = _docs("Gói dịch vụ: 730.000 đồng chưa bao gồm phụphí; giá đã gồm phụphí là 890.000 đồng.")
    value = "890.000" if qualifier == "đã gồm" else "730.000"
    other = "730.000" if qualifier == "đã gồm" else "890.000"
    assert ground_answer_numbers(f"Gói dịch vụ có giá {value} đồng, {qualifier} phụphí.", source).valid
    assert not ground_answer_numbers(f"Gói dịch vụ có giá {other} đồng, {qualifier} phụphí.", source).valid


def test_negative_inclusion_and_separate_owners_keep_correct_controls():
    source = _docs("Bronze costs 73 dollars without insurance. Silver costs 89 dollars including insurance.")
    assert ground_answer_numbers("Bronze costs 73 dollars not including insurance.", source).valid
    assert not ground_answer_numbers("Silver costs 89 dollars not including insurance.", source).valid


def test_multiple_conditioned_clauses_cannot_union_their_numeric_row_bindings():
    source = _docs("| Group | Iteration | Output |\n|---|---|---|\n"
                   "| Amber | 7 | 31.4 |\n| Amber | 14 | 35.8 |")
    assert ground_answer_numbers("At Iteration 7, Amber output was 31.4, while at Iteration 14, Amber output was 35.8.", source).valid
    assert not ground_answer_numbers("At Iteration 7, Amber output was 35.8, while at Iteration 14, Amber output was 31.4.", source).valid


def test_answer_table_numeric_conditions_bind_the_other_cells():
    source = _docs("| Group | Week | Change |\n|---|---|---|\n| Amber | 7 | 31.4 |\n| Amber | 14 | 35.8 |")
    answer = "| Group | Week | Change |\n|---|---|---|\n| Amber | 7 | 31.4 |"
    assert ground_answer_numbers(answer, source).valid
    assert not ground_answer_numbers(answer.replace("31.4", "35.8"), source).valid
