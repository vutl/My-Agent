from app.rag.evidence_obligations import (
    obligation_search_query,
    plan_evidence_obligations,
    reserve_obligation_coverage,
    tag_obligation_results,
)


def _result(chunk_id: str, content: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "content": content,
        "score": score,
        "artifact_type": "table",
        "table_id": chunk_id,
    }


def test_ratio_query_creates_two_operand_obligations() -> None:
    plan = plan_evidence_obligations(
        "what is R&D to asset ratio for Activision Blizzard in FY 2019?"
    )

    assert plan is not None
    assert plan.kind == "ratio"
    assert [item.query for item in plan.obligations] == [
        "R&D FY 2019 Activision Blizzard",
        "asset FY 2019 Activision Blizzard",
    ]
    assert "R&D" in plan.obligations[0].anchors
    assert "asset" in plan.obligations[1].anchors


def test_obligation_search_query_adds_generic_row_label_aliases_without_changing_anchor_query() -> None:
    plan = plan_evidence_obligations(
        "what is R&D to asset ratio for Activision Blizzard in FY 2019?"
    )
    assert plan is not None
    assert plan.obligations[0].query == "R&D FY 2019 Activision Blizzard"
    assert "product development" in obligation_search_query(plan.obligations[0])
    assert "total assets" in obligation_search_query(plan.obligations[1])
    # The answer value is never part of a query-derived alias expansion.
    assert "998" not in obligation_search_query(plan.obligations[0])
    assert "19,845" not in obligation_search_query(plan.obligations[1])


def test_anaphoric_multisentence_query_creates_fact_and_target_obligations() -> None:
    plan = plan_evidence_obligations(
        "A particular racial group was treated least fairly in promotion. "
        "What percentage of this group was denied a promotion?"
    )

    assert plan is not None
    assert plan.kind == "anaphoric_multihop"
    assert len(plan.obligations) == 2
    assert "racial" in {token.casefold() for token in plan.obligations[0].anchors}
    assert "percentage" in {
        token.casefold() for token in plan.obligations[1].anchors
    }


def test_conditional_lookup_keeps_anchor_and_target_searches_independent() -> None:
    plan = plan_evidence_obligations(
        "In the year in which the No of internet users was 330 million, "
        "what percentage of Indians were Debit Card users?"
    )

    assert plan is not None
    assert plan.kind == "conditional_multihop"
    assert "330" in plan.obligations[0].query
    assert "Debit Card" in plan.obligations[1].query
    # The target chart may not repeat the anchor value; do not pollute its
    # lexical/vector branch with the first operand's number.
    assert "330" not in plan.obligations[1].query


def test_plain_single_fact_query_does_not_activate_obligations() -> None:
    assert plan_evidence_obligations("What dataset does ASPIRE use?") is None


def test_obligation_coverage_keeps_both_operands_inside_final_limit() -> None:
    plan = plan_evidence_obligations("What is R&D to asset ratio in FY 2019?")
    assert plan is not None
    noise = [_result(f"noise-{index}", "generic annual report", 10 - index) for index in range(8)]
    numerator = tag_obligation_results(
        [_result("rd", "Research and development R&D 998", 0.2)],
        plan.obligations[0],
    )[0]
    denominator = tag_obligation_results(
        [_result("assets", "Total assets 19,845", 0.1)],
        plan.obligations[1],
    )[0]

    selected = reserve_obligation_coverage(
        [*noise, numerator, denominator],
        plan=plan,
        limit=4,
    )

    assert {item["table_id"] for item in selected} >= {"rd", "assets"}


def test_obligation_reservation_never_synthesizes_a_missing_operand() -> None:
    plan = plan_evidence_obligations("What is revenue to asset ratio in 2024?")
    assert plan is not None
    only_revenue = _result("revenue", "Revenue was 10", 1.0)

    selected = reserve_obligation_coverage(
        [only_revenue],
        plan=plan,
        limit=3,
    )

    assert len(selected) == 1
    assert selected[0]["table_id"] == "revenue"


def test_obligation_reservation_prefers_branch_unique_evidence_over_shared_header() -> None:
    plan = plan_evidence_obligations("What is revenue to asset ratio in 2024?")
    assert plan is not None
    shared = _result("shared", "Revenue and total assets table header", 1.0)
    shared["evidence_obligation_ids"] = [
        item.obligation_id for item in plan.obligations
    ]
    numerator = tag_obligation_results(
        [_result("revenue", "Revenue 10", 0.2)], plan.obligations[0]
    )[0]
    denominator = tag_obligation_results(
        [_result("assets", "Total assets 20", 0.1)], plan.obligations[1]
    )[0]

    selected = reserve_obligation_coverage(
        [shared, numerator, denominator], plan=plan, limit=2
    )

    assert {item["table_id"] for item in selected} == {"revenue", "assets"}
