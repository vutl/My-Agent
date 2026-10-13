"""Metric-agnostic numeric grounding: every number must come from the evidence,
from the row/column the answer names. Domains here are deliberately unlike the
speech-emotion papers the metric validator was built on."""
import pytest

from app.services.evidence_validator import validate_answer_claims
from app.services.numeric_grounding import ground_answer_numbers

_ENERGY = [{
    "document_id": "energy",
    "filename": "plant-report.pdf",
    "content": (
        "Table 4: Output by site (GWh)\n"
        "| Site | 2022 output | 2023 output | Capacity factor (%) |\n"
        "|---|---|---|---|\n"
        "| Hoa Binh | 8 160 | 9 340 | 54.2 |\n"
        "| Son La | 9 870 | 10 120 | 48.7 |\n"
    ),
}]
_TRIAL = [{
    "document_id": "trial",
    "filename": "trial.pdf",
    "content": (
        "| Group | Week | Mean change (mmHg) |\n"
        "|---|---|---|\n"
        "| Drug X | 12 | -14,6 |\n"
        "| Drug X | 24 | -16,1 |\n"
        "| Control | 12 | -5,3 |\n"
    ),
}]
_GROUPED = [{
    "document_id": "bench",
    "filename": "bench.pdf",
    "content": (
        "| System | Dev | | Test | |\n"
        "|---|---|---|---|---|\n"
        "| | P | R | P | R |\n"
        "| Lynx | 71.0 | 64.2 | 69.5 | 62.8 |\n"
        "| Otter | 74.3 | 60.9 | 72.1 | 59.7 |\n"
    ),
}]
_ABLATION = [{
    "document_id": "abl",
    "filename": "abl.pdf",
    "content": (
        "| Variant | BLEU |\n|---|---|\n"
        "| Full system | 31.4 |\n| w/o copy mechanism | 28.9 |\n| w/o coverage | 30.2 |\n"
    ),
}]


def _ok(answer, documents):
    return ground_answer_numbers(answer, documents).valid


@pytest.mark.parametrize("answer, expected", [
    ("Son La produced 10120 GWh in 2023.", True),
    ("Son La produced 11200 GWh in 2023.", False),          # fabricated
    ("Hoa Binh produced 10120 GWh in 2023.", False),        # other row's value
    ("Son La produced 10120 GWh in 2022.", False),          # other column's value
    ("Hoa Binh output grew by 1180 GWh from 2022 to 2023.", True),  # difference
    ("Hệ số công suất của Sơn La là 48,7%.", True),
])
def test_unknown_metric_values_are_grounded_by_row_and_column(answer, expected):
    assert _ok(answer, _ENERGY) is expected


def test_years_named_as_columns_still_require_matching_cells():
    assert _ok("Hoa Binh reached 9340 GWh in 2023.", _ENERGY)
    assert not _ok("Hoa Binh reached 9340 GWh in 2022.", _ENERGY)


@pytest.mark.parametrize("answer, expected", [
    ("Ở tuần 24, nhóm Drug X giảm 16,1 mmHg.", True),
    ("Nhóm Control giảm 14,6 mmHg ở tuần 12.", False),      # other arm
])
def test_vietnamese_decimal_comma_and_conditions(answer, expected):
    assert _ok(answer, _TRIAL) is expected


def test_numeric_condition_column_is_bound():
    assert not _ok("Ở tuần 12, nhóm Drug X giảm 16,1 mmHg.", _TRIAL)


@pytest.mark.parametrize("answer, expected", [
    ("Otter reaches a test precision of 72.1.", True),
    ("Otter reaches a dev precision of 72.1.", False),      # group header
    ("Lynx reaches a test recall of 62.8.", True),
    ("| System | Split | P |\n|---|---|---|\n| Otter | Test | 72.1 |", True),
    ("| System | Split | P |\n|---|---|---|\n| Lynx | Test | 72.1 |", False),
])
def test_group_headers_below_the_separator(answer, expected):
    assert _ok(answer, _GROUPED) is expected


@pytest.mark.parametrize("answer, expected", [
    ("Without the copy mechanism, BLEU drops to 28.9.", True),
    ("Without the copy mechanism, BLEU is 31.4.", False),   # full-system value
    ("Removing coverage gives 30.2 BLEU.", True),
    ("The full system reaches 31.4 BLEU.", True),
])
def test_labels_that_denote_absence(answer, expected):
    assert _ok(answer, _ABLATION) is expected


def test_family_name_without_member_is_ambiguous_only_when_values_differ():
    docs = [{"document_id": "d", "filename": "d.pdf", "content": (
        "| Region | Revenue H1 | Revenue H2 | Staff H1 | Staff H2 |\n|---|---|---|---|---|\n"
        "| North | 120.5 | 133.0 | 40 | 40 |\n")}]
    assert not _ok("North revenue was 120.5.", docs)        # H1 or H2?
    assert _ok("North revenue in H1 was 120.5.", docs)
    assert _ok("North had 40 staff.", docs)                 # same value both halves


def test_respectively_pairs_across_conjunctions():
    assert _ok("Lynx and Otter reach a dev precision of 71.0 and 74.3, respectively.", _GROUPED)


def test_validator_rejects_ungrounded_number_the_metric_parser_ignores():
    result = validate_answer_claims(
        answer="Son La produced 11200 GWh in 2023.", documents=_ENERGY, focus_document_ids=["energy"],
    )
    assert result.valid is False
    assert any(claim.metric == "number" for claim in result.unsupported_claims)
    assert validate_answer_claims(
        answer="Son La produced 10120 GWh in 2023.", documents=_ENERGY, focus_document_ids=["energy"],
    ).valid


_AIR_2023 = {"document_id": "air23", "filename": "air_2023.md", "content": (
    "# Air quality report 2023\n\n| City | Mean PM2.5 (µg/m³) | Bad days |\n|---|---|---|\n"
    "| Hanoi | 41.2 | 118 |\n| Hue | 18.6 | 22 |\n")}
_AIR_2024 = {"document_id": "air24", "filename": "air_2024.md", "content": (
    "# Air quality report 2024\n\n| City | Mean PM2.5 (µg/m³) | Bad days |\n|---|---|---|\n"
    "| Hanoi | 38.9 | 104 |\n| Hue | 17.1 | 19 |\n")}


@pytest.mark.parametrize("answer, expected", [
    ("In 2024, Hanoi's mean PM2.5 was 38.9 µg/m³.", True),
    ("In 2023, Hanoi's mean PM2.5 was 38.9 µg/m³.", False),   # other year's report
    ("Hanoi's mean PM2.5 was 38.9 µg/m³.", False),            # which year?
    ("Hanoi had 118 bad days in 2023.", True),
])
def test_document_level_conditions(answer, expected):
    assert _ok(answer, [_AIR_2023, _AIR_2024]) is expected


_QUARTERS = [{"document_id": "q", "filename": "q.md", "content": (
    "| Chỉ tiêu | Hà Nội | Đà Nẵng |\n|---|---|---|\n"
    "| **Quý 1** | | |\n| Lượt khách (nghìn) | 1.240 | 860 |\n"
    "| **Quý 2** | | |\n| Lượt khách (nghìn) | 1.410 | 1.020 |\n")}]


@pytest.mark.parametrize("answer, expected", [
    ("Quý 2, Đà Nẵng đón 1.020 nghìn lượt khách.", True),
    ("Quý 1, Đà Nẵng đón 1.020 nghìn lượt khách.", False),    # other section
    ("Đà Nẵng đón 1.020 nghìn lượt khách.", False),           # which quarter?
    ("Quý 2, Hà Nội đón 1.020 nghìn lượt khách.", False),      # other column
])
def test_section_rows_and_owner_columns(answer, expected):
    assert _ok(answer, _QUARTERS) is expected


def test_respectively_order_is_checked():
    docs = [_AIR_2024]
    assert _ok("In 2024, Hanoi and Hue had 104 and 19 bad days, respectively.", docs)
    assert not _ok("In 2024, Hanoi and Hue had 19 and 104 bad days, respectively.", docs)


def test_stated_difference_must_match_the_stated_numbers():
    docs = [_AIR_2023]
    assert _ok("Hanoi's PM2.5 (41.2) exceeds Hue's (18.6) by 22.6 µg/m³ in 2023.", docs)
    assert not _ok("Hanoi's PM2.5 (41.2) exceeds Hue's (18.6) by 23.6 µg/m³ in 2023.", docs)
