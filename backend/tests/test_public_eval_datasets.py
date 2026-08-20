from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from app.services.tool_decision_service import IntentRouterService


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = ROOT / "data" / "retrieval_eval" / "public"


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_downloads_match_pinned_revisions_and_counts() -> None:
    validator = _load_script("validate_public_eval_datasets")
    report = validator.validate(full=False)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["sources"] == 7
    assert report["counts"] == validator.EXPECTED_COUNTS


def test_public_catalog_build_is_isolated_and_deterministic(tmp_path: Path) -> None:
    preparer = _load_script("prepare_public_eval_datasets")
    preparer.PREPARED_ROOT = tmp_path
    summary = preparer.build_catalog()

    assert summary["cases"] == 15_892
    assert summary["runnable_agent_routing_cases"] == 50
    assert summary["production_corpus_modified"] is False
    assert sum(summary["by_suite"].values()) == 15_892
    assert sum(summary["by_runner_mode"].values()) == 15_892

    catalog = _jsonl(tmp_path / "catalog-v1.jsonl")
    routing = _jsonl(tmp_path / "wildbench-routing-v1.jsonl")
    assert len(catalog) == 15_892
    assert len(routing) == 50
    assert len({item["case_id"] for item in catalog}) == len(catalog)
    assert len({item["case_id"] for item in routing}) == len(routing)
    assert all(item["expected_route"] == "chat" for item in routing)
    assert all(item["expected_document_ids"] == [] for item in routing)
    assert all(item["provenance"]["suite"] == "WildBench-v2" for item in routing)
    assert summary["by_suite"]["spiqa-test-b"] == 228
    assert summary["by_suite"]["spiqa-test-c"] == 493
    spiqa_c = [item for item in catalog if item["suite"] == "spiqa-test-c"]
    assert len({item["case_id"] for item in spiqa_c}) == 493
    assert all("question_index" in item["source_locator"] for item in spiqa_c)


def test_conversation_evaluator_accepts_prepared_public_routing_suite() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    cases = evaluator._load_cases(  # noqa: SLF001 - schema boundary regression
        PUBLIC_ROOT / "prepared" / "wildbench-routing-v1.jsonl"
    )

    assert len(cases) == 50
    assert cases[0]["turn_index"] == 1


def test_conversation_evaluator_reference_token_f1_is_bounded() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    assert evaluator._token_f1("market cap equals company value", "company market cap") > 0
    assert evaluator._token_f1("unrelated pasta recipe", "company market cap") == 0
    assert evaluator._token_f1("", "company market cap") == 0


def test_conversation_evaluator_abstention_markers_cover_natural_phrasings() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    assert evaluator._has_abstention("There isn't enough evidence here.")
    assert evaluator._has_abstention("The provided references are insufficient.")
    assert evaluator._has_abstention("I do not have any information about that.")
    assert evaluator._has_abstention("I can't identify a Table 99 in this paper.")
    assert evaluator._has_abstention("The random seed was not provided.")
    assert evaluator._has_abstention("Random seed không được nêu trong source.")
    assert not evaluator._has_abstention("The references provide enough evidence.")


def test_conversation_evaluator_infers_current_limited_tool_contract() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    paper = {
        "expected_artifacts": {"table_ids": [], "figure_ids": []},
    }
    figure = {
        "expected_artifacts": {"table_ids": [], "figure_ids": ["figure-1"]},
    }

    assert evaluator._expected_tools(paper, ["file_qa"]) == ["search_local_docs"]
    assert evaluator._expected_tools(figure, ["file_qa"]) == [
        "search_local_docs",
        "retrieve_visual_assets",
    ]
    assert evaluator._expected_tools({}, ["chat"]) == []


def test_conversation_evaluator_checks_agent_layers_beyond_retrieval() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    document_id = "doc-1"
    case = {
        "case_id": "agent-layer-t1",
        "conversation_group": "agent-layer",
        "turn_index": 1,
        "message": "What does this paper do?",
        "expected_route": "file_qa",
        "expected_document_ids": [document_id],
        "expected_facets": {},
        "expected_artifacts": {"table_ids": [], "figure_ids": []},
        "expected_abstention": False,
    }

    def event(name: str, data: dict) -> object:
        return evaluator.Event(name, data)

    events = [
        event("run.started", {"model": evaluator.DEFAULT_MODEL}),
        event(
            "agent.route.decided",
            {"route": "file_qa", "selected_tools": ["search_local_docs"]},
        ),
        event("query.rewritten", {"focus_document_ids": [document_id]}),
        event("tool.started", {"tool_name": "search_local_docs"}),
        event(
            "retrieval.completed",
            {
                "documents": [{"document_id": document_id}],
                "retrieval_mode": "hybrid",
                "evidence_validation": {"valid": True, "reason": "focused"},
            },
        ),
        event("tool.completed", {"tool_name": "search_local_docs"}),
        event(
            "answer.evidence.validated",
            {"valid": True, "reason": "claims_supported"},
        ),
        event("message.delta", {"delta": "Supported answer."}),
        event("message.finished", {"finish_reason": "stop"}),
        event("run.completed", {"run_id": "run-1"}),
    ]
    result = evaluator._evaluate_case(
        case,
        {
            "events": events,
            "answer": "Supported answer.",
            "elapsed_ms": 10.0,
            "client_first_delta_ms": 8.0,
        },
        model=evaluator.DEFAULT_MODEL,
    )

    assert result["ok"] is True
    assert result["selected_tools"] == ["search_local_docs"]
    assert result["retrieval_validation"]["valid"] is True
    assert result["answer_validation"]["valid"] is True


def test_conversation_evaluator_allows_semantically_valid_optional_visual_tool() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    document_id = "doc-1"
    case = {
        "case_id": "missing-figure-t1",
        "conversation_group": "missing-figure",
        "turn_index": 1,
        "message": "Give me Figure 99 from this paper.",
        "expected_route": "file_qa",
        "expected_document_ids": [document_id],
        "expected_facets": {},
        "expected_artifacts": {"table_ids": [], "figure_ids": []},
        "expected_abstention": True,
    }
    events = [
        evaluator.Event("run.started", {"model": evaluator.DEFAULT_MODEL}),
        evaluator.Event(
            "agent.route.decided",
            {
                "route": "file_qa",
                "selected_tools": ["search_local_docs", "retrieve_visual_assets"],
            },
        ),
        evaluator.Event("query.rewritten", {"focus_document_ids": [document_id]}),
        evaluator.Event("tool.started", {"tool_name": "search_local_docs"}),
        evaluator.Event("tool.completed", {"tool_name": "search_local_docs"}),
        evaluator.Event("tool.started", {"tool_name": "retrieve_visual_assets"}),
        evaluator.Event("tool.completed", {"tool_name": "retrieve_visual_assets"}),
        evaluator.Event(
            "retrieval.completed",
            {
                "documents": [{"document_id": document_id}],
                "evidence_validation": {"valid": True},
            },
        ),
        evaluator.Event("answer.evidence.validated", {"valid": True}),
        evaluator.Event("message.delta", {"delta": "I can't identify Figure 99."}),
        evaluator.Event("message.finished", {"finish_reason": "stop"}),
        evaluator.Event("run.completed", {"run_id": "run-1"}),
    ]

    result = evaluator._evaluate_case(
        case,
        {
            "events": events,
            "answer": "I can't identify Figure 99.",
            "elapsed_ms": 10.0,
            "client_first_delta_ms": 8.0,
        },
        model=evaluator.DEFAULT_MODEL,
    )

    assert result["ok"] is True


def test_conversation_evaluator_rejects_chat_that_uses_local_paper_tool() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    case = {
        "case_id": "chat-leak-t1",
        "conversation_group": "chat-leak",
        "turn_index": 1,
        "message": "Let's talk about lunch.",
        "expected_route": "chat",
        "expected_document_ids": [],
        "expected_artifacts": {"table_ids": [], "figure_ids": []},
    }
    events = [
        evaluator.Event("run.started", {"model": evaluator.DEFAULT_MODEL}),
        evaluator.Event(
            "agent.route.decided",
            {"route": "chat", "selected_tools": ["search_local_docs"]},
        ),
        evaluator.Event("tool.started", {"tool_name": "search_local_docs"}),
        evaluator.Event("tool.completed", {"tool_name": "search_local_docs"}),
        evaluator.Event("message.delta", {"delta": "Lunch sounds good."}),
        evaluator.Event("message.finished", {"finish_reason": "stop"}),
        evaluator.Event("run.completed", {"run_id": "run-1"}),
    ]

    result = evaluator._evaluate_case(
        case,
        {
            "events": events,
            "answer": "Lunch sounds good.",
            "elapsed_ms": 10.0,
            "client_first_delta_ms": 8.0,
        },
        model=evaluator.DEFAULT_MODEL,
    )

    assert result["ok"] is False
    assert any(item.startswith("selected_tools:") for item in result["failures"])
    assert any(item.startswith("started_tools:") for item in result["failures"])


def test_conversation_evaluator_rejects_antigravity_draft_schema(
    tmp_path: Path,
) -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    path = tmp_path / "misleading-draft.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "What does the paper do?",
                "expected_documents": ["ASPIRE.pdf"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported conversation-eval schema"):
        evaluator._load_cases(path)  # noqa: SLF001 - schema boundary regression


def test_all_public_wildbench_negatives_reject_unscoped_local_rag() -> None:
    class _AlwaysLocalRouter:
        async def chat(self, **kwargs):  # noqa: ANN003
            class _Completion:
                message = (
                    '{"route":"file_qa","selected_tools":["search_local_docs"],'
                    '"answer_intent":"elaborate","answer_depth":"detailed",'
                    '"confidence":"high","needs_fallback":false,'
                    '"reason":"generic_document_request"}'
                )

            return _Completion()

    async def evaluate() -> list:
        router = IntentRouterService(client=_AlwaysLocalRouter(), default_model="test")
        rows = _jsonl(PUBLIC_ROOT / "prepared" / "wildbench-routing-v1.jsonl")
        return [
            await router.decide(
                task=row["message"],
                mode="auto",
                previous_messages=[],
                has_recent_retrieval=False,
                allowed_tools=["search_local_docs", "retrieve_visual_assets"],
            )
            for row in rows
        ]

    decisions = asyncio.run(evaluate())
    assert len(decisions) == 50
    assert all(decision.route == "chat" for decision in decisions)
    assert all(decision.selected_tools == [] for decision in decisions)
    assert {decision.reason for decision in decisions} <= {
        "unscoped_local_retrieval_rejected",
        "deterministic_casual_chat",
    }
    assert sum(
        decision.reason == "unscoped_local_retrieval_rejected"
        for decision in decisions
    ) == 49
