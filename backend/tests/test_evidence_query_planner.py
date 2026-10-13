from dataclasses import dataclass
import asyncio

from app.rag.evidence_obligations import plan_evidence_obligations
from app.rag.evidence_obligations import reserve_obligation_coverage
from app.rag.fusion import fuse_ranked_results
from app.services.evidence_query_planner import EvidenceQueryPlannerService


@dataclass
class _Completion:
    message: str


class _Client:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        return _Completion(self.message)


def test_planner_expands_aliases_without_answer_or_gold_context() -> None:
    client = _Client(
        '{"queries":['
        '{"id":"operand_1","query":"Activision Blizzard FY 2019 research and development R&D expense"},'
        '{"id":"operand_2","query":"Activision Blizzard FY 2019 total assets"}'
        ']}'
    )
    service = EvidenceQueryPlannerService(client=client, default_model="cx/gpt-5.6-sol")

    result = asyncio.run(
        service.plan(
            query="what is R&D to asset ratio for Activision Blizzard in FY 2019?"
        )
    )

    assert client.calls == 1
    assert result.model_used is True
    assert result.diagnostics["reason"] == "validated_semantic_expansion"
    assert [item.query for item in result.plan.obligations] == [
        "Activision Blizzard FY 2019 research and development R&D expense",
        "Activision Blizzard FY 2019 total assets",
    ]


def test_planner_rejects_invented_numbers_and_keeps_query_derived_plan() -> None:
    query = "What is revenue to assets ratio for Example Corp in FY 2024?"
    base = plan_evidence_obligations(query)
    assert base is not None
    client = _Client(
        '{"queries":['
        '{"id":"operand_1","query":"Example Corp FY 2024 revenue 123"},'
        '{"id":"operand_2","query":"Example Corp FY 2024 total assets"}'
        ']}'
    )
    service = EvidenceQueryPlannerService(client=client, default_model="cx/gpt-5.6-sol")

    result = asyncio.run(service.plan(query=query, base_plan=base))

    assert result.model_used is True
    assert result.diagnostics["reason"] == "invalid_model_plan"
    assert result.plan == base


def test_planner_does_not_call_model_for_single_fact_query() -> None:
    client = _Client("not used")
    service = EvidenceQueryPlannerService(client=client, default_model="cx/gpt-5.6-sol")

    result = asyncio.run(service.plan(query="What dataset does ASPIRE use?"))

    assert client.calls == 0
    assert result.plan is None
    assert result.model_used is False


def test_planner_keeps_comparison_queries_local_first() -> None:
    client = _Client("should not be called")
    service = EvidenceQueryPlannerService(client=client, default_model="cx/gpt-5.6-sol")

    result = asyncio.run(
        service.plan(
            query=(
                "Compared with 22Q1, in 23Q1, how much higher is the increase "
                "rate of daily creators than daily video views?"
            )
        )
    )

    assert result.plan is not None
    assert result.plan.kind == "comparison"
    assert client.calls == 0
    assert result.model_used is False
    assert result.diagnostics["reason"] == "comparison_uses_deterministic_operand_aliases"


def test_semantic_aliases_can_rank_a_candidate_without_certifying_new_facts() -> None:
    client = _Client(
        '{"queries":['
        '{"id":"operand_1","query":"Example Corp FY 2024 product development expense"},'
        '{"id":"operand_2","query":"Example Corp FY 2024 total assets"}'
        ']}'
    )
    service = EvidenceQueryPlannerService(client=client, default_model="cx/gpt-5.6-sol")
    plan = asyncio.run(
        service.plan(
            query="What is R&D to asset ratio for Example Corp in FY 2024?"
        )
    ).plan
    assert plan is not None
    candidates = [
        {
            "chunk_id": "wrong",
            "document_id": "doc-1",
            "content": "Total intangible assets in FY 2024",
            "score": 1.0,
        },
        {
            "chunk_id": "right",
            "document_id": "doc-1",
            "content": "Product development expense in FY 2024",
            "score": 0.2,
        },
    ]
    selected = reserve_obligation_coverage(
        candidates,
        plan=plan,
        limit=2,
    )
    assert selected[0]["chunk_id"] == "right"


def test_fusion_preserves_obligation_tags_across_duplicate_identity() -> None:
    original = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": "value",
        "score": 0.4,
    }
    tagged = {**original, "evidence_obligation_ids": ["operand_2"]}

    fused = fuse_ranked_results(
        [("original", original), ("operand_2", tagged)],
        top_k=2,
    )

    assert fused[0]["evidence_obligation_ids"] == ["operand_2"]
