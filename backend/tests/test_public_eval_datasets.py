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
    # The manifest is versioned: new corrected releases are additive metadata,
    # so this assertion must follow the manifest rather than hard-code a count.
    manifest = json.loads((PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert report["sources"] == len(manifest["sources"])
    assert report["counts"] == validator.EXPECTED_COUNTS


def test_public_case_manifest_resolver_is_strict_and_ordered() -> None:
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    lib = _load_script("public_multimodal_eval_lib")
    cases = [
        lib.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id="case-a",
            document_id="doc-a",
            title="A",
            question="q",
            answers=("a",),
            answer_format="Str",
        ),
        lib.BenchmarkCase(
            suite="spiqa-test-c",
            case_id="case-b",
            document_id="doc-b",
            title="B",
            question="q",
            answers=("b",),
            answer_format="free_form",
        ),
    ]
    assert [case.case_id for case in lib.resolve_case_ids(cases, ["case-b", "case-a"])] == [
        "case-b",
        "case-a",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        lib.resolve_case_ids(cases, ["case-a", "case-a"])
    with pytest.raises(ValueError, match="unknown"):
        lib.resolve_case_ids(cases, ["missing"])


def test_visual_answer_quality_pairing_requires_exact_case_frontier() -> None:
    pair = _load_script("attach_answer_quality_to_visual_gate")
    visual = {
        "schema_version": "visual-retrieval-gate-v1",
        "no_llm_calls": True,
        "pdf_ocr_enabled": True,
        "evaluation_configuration": {
            "render_scale": 1.25,
            "device": "auto",
            "local_files_only": True,
            "stage_version": "foundation-v3",
        },
        "model": "vidore/colSmol-256M",
        "selection": {"case_ids": ["a", "b"]},
        "rows": [{"case_id": "a"}, {"case_id": "b"}],
        "aggregate": {},
        "worker_health": {"ok": True},
    }
    answer = {
        "status": "complete",
        "model": "cx/gpt-5.6-sol",
        "selection": {"case_ids": ["a", "b"]},
        "answer_quality": {
            "status": "complete",
            "paired_cases": 2,
            "overall": {"aya_vs_full_delta": 0.1},
            "per_case": [
                {"case_id": "a", "status": "ok", "delta_aya_vs_full": 0.1},
                {"case_id": "b", "status": "ok", "delta_aya_vs_full": 0.0},
            ],
        },
    }
    # A tiny report remains non-safe even when it is otherwise complete.
    paired = pair.attach(visual, answer)
    assert paired["answer_quality"]["promotion_safe"] is False
    assert paired["promotion"]["passed"] is False
    with pytest.raises(ValueError, match="frontier mismatch"):
        pair.attach(visual, {**answer, "selection": {"case_ids": ["a"]}})
    # A matching declared selection must not attest scores from foreign cases.
    for scored_ids in (["a", "foreign"], ["a", "a"]):
        bad_quality = {**answer["answer_quality"], "per_case": [
            {"case_id": case_id, "status": "ok", "delta_aya_vs_full": 0.1}
            for case_id in scored_ids
        ]}
        with pytest.raises(ValueError, match="scored answer rows"):
            pair.attach(visual, {**answer, "answer_quality": bad_quality})


def test_answer_quality_merge_excludes_audited_non_frontier_cases(tmp_path: Path) -> None:
    merger = _load_script("merge_answer_quality_reports")

    def report(case_id: str) -> dict:
        per_case = {
            "case_id": case_id,
            "suite": "mmlongbench-doc",
            "stratum": "table",
            "document_profile": "native_text",
            "status": "ok",
            "scores": {
                "gold_evidence": 1.0,
                "full_extracted_document": 1.0,
                "aya_pipeline": 1.0,
            },
            "delta_aya_vs_full": 0.0,
            "delta_aya_vs_gold": 0.0,
        }
        rows = [
            {
                "case_id": case_id,
                "mode": mode,
                "status": "ok",
                "score": {"official_mmlong_score": 1.0},
                "generation": {"first_token_ms": 1.0},
                "total_ms": 2.0,
                "input": {},
            }
            for mode in merger.MODES
        ]
        return {
            "status": "complete",
            "model": "cx/gpt-5.6-sol",
            "evaluator_fingerprint": "test-evaluator-v1",
            "modes": list(merger.MODES),
            "visual_pages_enabled": True,
            "pdf_ocr_enabled": True,
            "retrieval_configuration": {"rerank_enabled": False},
            "selection": {"case_ids": [case_id]},
            "answer_quality": {
                "status": "complete",
                "per_case": [per_case],
            },
            "case_results": rows,
            "config_fingerprint": case_id,
        }

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(report("keep")), encoding="utf-8")
    second.write_text(json.dumps(report("exclude")), encoding="utf-8")
    merged = merger.merge(
        [first, second], ["keep"], {"exclude"}
    )
    assert merged["case_count"] == 1
    assert merged["selection"]["case_ids"] == ["keep"]

    stale = report("stale")
    stale.pop("evaluator_fingerprint")
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(merger.MergeContractError, match="evaluator_fingerprint"):
        merger.merge([stale_path, first], ["stale", "keep"])


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


def test_public_frontier_document_exclusion_resolves_all_dataset_versions() -> None:
    builder = _load_script("build_public_foundation_frontier")
    library = _load_script("public_multimodal_eval_lib")
    old = library.load_mmlong_cases(version="v1")[0]
    new = library.load_mmlong_cases(version="v2")[0]
    assert builder.previous_documents({old.case_id, new.case_id}, [old, new]) == {old.document_id, new.document_id}
    with pytest.raises(ValueError, match="unresolved previous frontier"):
        builder.previous_documents({new.case_id}, [old])
    assert builder.document_family("2307.09288v2.pdf") == builder.document_family("2307.09288v3.pdf")


def test_conversation_evaluator_reference_token_f1_is_bounded() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    assert evaluator._token_f1("market cap equals company value", "company market cap") > 0
    assert evaluator._token_f1("unrelated pasta recipe", "company market cap") == 0
    assert evaluator._token_f1("", "company market cap") == 0


def test_format_lifecycle_smoke_distinguishes_text_hits_from_non_searchable_pages() -> None:
    smoke = _load_script("evaluate_format_lifecycle_smoke")

    # Native/OCR-searchable material must produce the expected lexical hit;
    # projection success alone cannot hide a real local-search regression.
    assert smoke._retrieval_contract(
        retrieval_hit=True,
        readiness={"state": "ready", "native_text_pages": 1, "ocr_text_pages": 0},
        projection_ok=True,
    ) == (True, True)
    assert smoke._retrieval_contract(
        retrieval_hit=False,
        readiness={"state": "ready", "native_text_pages": 1, "ocr_text_pages": 0},
        projection_ok=True,
    ) == (True, False)

    # A scan/image-only page with no usable native/OCR text is accepted only as
    # an honest partial/visual/unresolved projection. It must not be labelled
    # ``ready`` merely because the vector projection succeeded.
    assert smoke._retrieval_contract(
        retrieval_hit=False,
        readiness={"state": "partial", "native_text_pages": 0, "ocr_text_pages": 0},
        projection_ok=True,
    ) == (False, True)
    assert smoke._retrieval_contract(
        retrieval_hit=False,
        readiness={"state": "ready", "native_text_pages": 0, "ocr_text_pages": 0},
        projection_ok=True,
    ) == (False, False)


def test_conversation_evaluator_abstention_markers_cover_natural_phrasings() -> None:
    evaluator = _load_script("evaluate_agent_conversations")
    assert evaluator._has_abstention("There isn't enough evidence here.")
    assert evaluator._has_abstention("The provided references are insufficient.")
    assert evaluator._has_abstention("I do not have any information about that.")
    assert evaluator._has_abstention("I can't identify a Table 99 in this paper.")
    assert evaluator._has_abstention("I can’t find Figure 99 in this paper.")
    assert evaluator._has_abstention("The random seed was not provided.")
    assert evaluator._has_abstention("The provided source does not specify the random seed.")
    assert evaluator._has_abstention("Neither source excerpt states the number of runs.")
    assert evaluator._has_abstention("No meaningful comparison is possible from the evidence.")
    assert evaluator._has_abstention("Random seed không được nêu trong source.")
    assert evaluator._has_abstention("ASPIRE.pdf không có Table 99.")
    assert evaluator._has_abstention("ViSEC không có Figure 99.")
    assert evaluator._has_abstention("ViSEC không có **Figure 99**.")
    assert evaluator._has_abstention("ViSEC paper does not have Figure 99.")
    assert evaluator._has_abstention("The source contains no result for that split.")
    assert not evaluator._has_abstention("The references provide enough evidence.")
    assert not evaluator._has_abstention("ASPIRE does not use video input.")


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
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs):  # noqa: ANN003
            self.calls += 1

            class _Completion:
                message = (
                    '{"route":"file_qa","selected_tools":["search_local_docs"],'
                    '"answer_intent":"elaborate","answer_depth":"detailed",'
                    '"confidence":"high","needs_fallback":false,'
                    '"reason":"generic_document_request"}'
                )

            return _Completion()

    client = _AlwaysLocalRouter()

    async def evaluate() -> list:
        router = IntentRouterService(client=client, default_model="test")
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
    # Every model request for unscoped local retrieval must be rejected,
    # regardless of how many complete social messages take a shortcut.
    assert client.calls > 0
    assert sum(
        decision.reason == "unscoped_local_retrieval_rejected"
        for decision in decisions
    ) == client.calls


def test_public_answer_eval_timeout_contract_is_fingerprinted() -> None:
    evaluator = _load_script("evaluate_public_multimodal_three_mode")
    common = {
        "model": "cx/gpt-5.6-sol",
        "modes": ("gold_evidence",),
        "mmlong_cases": 1,
        "spiqa_cases": 0,
        "mmlong_cases_per_document": 1,
        "spiqa_cases_per_document": 1,
        "top_k": 8,
        "max_context_chars": 90_000,
        "max_full_images": 4,
        "max_context_tokens": 96_000,
        "output_reserve_tokens": 16_000,
        "mmlong_version": "v2",
        "visual_pages": False,
        "selected_case_ids": ["case-a"],
    }
    first = evaluator._evaluation_config_fingerprint(
        **common, retrieval_timeout_seconds=30, generation_timeout_seconds=60
    )
    second = evaluator._evaluation_config_fingerprint(
        **common, retrieval_timeout_seconds=31, generation_timeout_seconds=60
    )
    third = evaluator._evaluation_config_fingerprint(
        **common, retrieval_timeout_seconds=30, generation_timeout_seconds=61
    )
    assert first != second
    assert first != third


def test_public_answer_eval_generation_timeout_is_cancellable() -> None:
    evaluator = _load_script("evaluate_public_multimodal_three_mode")

    class _SlowClient:
        async def stream_chat(self, **_kwargs):  # noqa: ANN003
            await asyncio.sleep(60)
            yield None

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                evaluator._generate(
                    client=_SlowClient(),
                    model="cx/gpt-5.6-sol",
                    prompt="{}",
                    images=[],
                ),
                timeout=0.01,
            )

    asyncio.run(run())
