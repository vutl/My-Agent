from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dual_query_retrieval import (  # noqa: E402
    EvalCase,
    EvaluationContractError,
    EvidenceLocator,
    _validate_suite_shape,
    promotion_gate,
    resolve_case,
    score_results,
    load_cases,
    _run_arm,
    _run_final_stage,
    _failure_taxonomy,
    _runtime_document_scope,
    _merge_decomposed_payloads,
    suite_quality_diagnostics,
    _recommend_action,
)
from evaluate_agent_policy_retrieval import _branch_specs, _round_robin  # noqa: E402


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            source_path TEXT,
            filename TEXT,
            content_hash TEXT,
            chunk_count INTEGER,
            table_count INTEGER,
            figure_count INTEGER,
            indexed_at TEXT
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_type TEXT,
            page_number INTEGER,
            section_title TEXT,
            content TEXT,
            chunk_index INTEGER,
            metadata_json TEXT
        );
        CREATE TABLE document_tables (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            table_index INTEGER,
            caption TEXT,
            markdown TEXT
        );
        CREATE TABLE document_figures (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            figure_index INTEGER,
            caption TEXT,
            image_path TEXT,
            metadata_json TEXT
        );
        """
    )
    return connection


def _case(*, kind: str = "text") -> EvalCase:
    locator = {
        "group_id": "g1",
        "kind": kind,
        "document": {
            "filename": "paper.pdf",
            "content_sha256": "content-hash",
        },
        "page_number": 2,
        "quote": "Pitch fusion improves tonal emotion recognition.",
    }
    return EvalCase.from_dict(
        {
            "case_id": "case-1",
            "category": "text" if kind == "text" else "table_numeric",
            "language": "en",
            "raw_query": "what does it improve?",
            "enriched_query": "Pitch fusion improves tonal emotion recognition",
            "standalone_query": "Pitch fusion improves tonal emotion recognition",
            "required_entities": ["Pitch fusion"],
            "scope": "unscoped",
            "evidence_groups": [locator],
        }
    )


def test_qrel_resolves_against_content_fingerprint_not_runtime_uuid() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("runtime-uuid-2", "/tmp/paper.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "chunk-runtime-2",
            "runtime-uuid-2",
            "text",
            2,
            "Results",
            "Pitch fusion improves tonal emotion recognition.",
            0,
            "{}",
        ),
    )
    resolved = resolve_case(connection, _case())
    assert resolved.document_ids == {"runtime-uuid-2"}
    assert resolved.groups[0].evidence_ids == {"text:chunk-runtime-2"}


def test_qrel_resolver_ignores_retired_revision_rows() -> None:
    """Preflight must mirror production active-revision filtering."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            source_path TEXT,
            filename TEXT,
            content_hash TEXT,
            active_revision_id TEXT,
            revision_status TEXT,
            indexed_at TEXT
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            revision_id TEXT,
            chunk_type TEXT,
            page_number INTEGER,
            section_title TEXT,
            content TEXT,
            chunk_index INTEGER,
            metadata_json TEXT
        );
        CREATE TABLE document_tables (
            id TEXT PRIMARY KEY, document_id TEXT, revision_id TEXT,
            page_number INTEGER, table_index INTEGER, caption TEXT, markdown TEXT
        );
        CREATE TABLE document_figures (
            id TEXT PRIMARY KEY, document_id TEXT, revision_id TEXT,
            page_number INTEGER, figure_index INTEGER, caption TEXT,
            image_path TEXT, metadata_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("doc-live", "/tmp/paper.pdf", "paper.pdf", "content-hash", "rev-2", "active", "now"),
    )
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("chunk-old", "doc-live", "rev-1", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
            ("chunk-live", "doc-live", "rev-2", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
        ],
    )
    resolved = resolve_case(connection, _case())
    assert resolved.groups[0].evidence_ids == {"text:chunk-live"}

    connection.execute(
        "UPDATE documents SET revision_status = 'retired', active_revision_id = 'rev-1' WHERE id = 'doc-live'"
    )
    with pytest.raises(EvaluationContractError, match="resolved 0"):
        resolve_case(connection, _case())


def test_score_results_handles_required_evidence_and_forbidden_documents() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/a.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("chunk-a", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
    )
    resolved = resolve_case(connection, _case())
    scored = score_results(
        resolved,
        [{"chunk_id": "chunk-a", "document_id": "doc-a", "page_number": 2}],
        cutoff=10,
    )
    assert scored["evidence_recall"] == 1.0
    assert scored["document_recall"] == 1.0
    assert scored["forbidden_documents"] == []


def test_runtime_scope_uses_catalog_mentions_without_qrel_targeting() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/paper.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("chunk-a", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
    )
    resolved = resolve_case(connection, _case())

    class CatalogRag:
        def resolve_catalog_mentions(self, *, query, collection_id):  # noqa: ANN001
            assert query == "what does it improve?"
            assert collection_id is None
            return SimpleNamespace(document_ids=("doc-a",))

    assert _runtime_document_scope(CatalogRag(), resolved) == (["doc-a"], "catalog_current_turn")


def test_runtime_scope_fails_closed_when_catalog_target_conflicts_with_qrel() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/paper.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("chunk-a", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
    )
    resolved = resolve_case(connection, _case())

    class ConflictingCatalogRag:
        def resolve_catalog_mentions(self, **_kwargs):
            return SimpleNamespace(document_ids=("wrong-doc",))

    with pytest.raises(EvaluationContractError, match="does not match qrel"):
        _runtime_document_scope(ConflictingCatalogRag(), resolved)


def test_failure_taxonomy_distinguishes_final_cutoff_from_reranker_loss() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/a.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("chunk-a", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
    )
    resolved = resolve_case(connection, _case())
    candidate = {"chunk_id": "chunk-a", "document_id": "doc-a", "page_number": 2}
    cutoff = _failure_taxonomy(
        resolved,
        payload={},
        candidates=[candidate],
        final_results=[],
        rerank_applied=False,
    )
    reranked = _failure_taxonomy(
        resolved,
        payload={},
        candidates=[candidate],
        final_results=[],
        rerank_applied=True,
    )
    assert cutoff[0]["reasons"] == ["final_top_k_cutoff"]
    assert reranked[0]["reasons"] == ["rerank_loss"]


def test_qrel_alternatives_resolve_and_score_as_one_evidence_group() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/a.pdf", "paper.pdf", "content-hash", 2, 0, 0, "now"),
    )
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("chunk-a", "doc-a", "text", 2, "Results", "Other result.", 0, "{}"),
            ("chunk-b", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 1, "{}"),
        ],
    )
    case = EvalCase.from_dict(
        {
            "case_id": "alternatives",
            "category": "text",
            "language": "en",
            "raw_query": "what improves emotion recognition?",
            "standalone_full": "pitch fusion emotion recognition",
            "standalone_anchor": "pitch fusion",
            "evidence_groups": [
                {
                    "group_id": "task",
                    "kind": "text",
                    "document": {"filename": "paper.pdf", "content_sha256": "content-hash"},
                    "page_number": 2,
                    "quote": "not present in this re-index",
                    "alternatives": [
                        {"page_number": 2, "quote": "Pitch fusion improves tonal emotion recognition."}
                    ],
                }
            ],
        }
    )
    resolved = resolve_case(connection, case)
    assert resolved.groups[0].evidence_ids == {"text:chunk-b"}
    assert score_results(resolved, [{"chunk_id": "chunk-b", "document_id": "doc-a"}], cutoff=10)["evidence_recall"] == 1.0


def test_typed_artifact_schema_cannot_change_candidate_or_final_recall() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/a.pdf", "paper.pdf", "content-hash", 0, 1, 0, "now"),
    )
    connection.execute(
        "INSERT INTO document_tables VALUES (?, ?, ?, ?, ?, ?)",
        ("table-record", "doc-a", 5, 1, "Table 2. Results", "| model | score |"),
    )
    case = EvalCase.from_dict(
        {
            "case_id": "typed-table",
            "category": "table_numeric",
            "language": "en",
            "raw_query": "show Table 2",
            "standalone_full": "paper.pdf Table 2",
            "standalone_anchor": "paper.pdf Table 2",
            "evidence_groups": [
                {
                    "group_id": "result-table",
                    "kind": "table",
                    "document": {"filename": "paper.pdf", "content_sha256": "content-hash"},
                    "page_number": 5,
                    "artifact_index": 1,
                    "caption": "Table 2. Results",
                }
            ],
        }
    )
    resolved = resolve_case(connection, case)
    legacy = {
        "chunk_id": "lancedb-row-7",
        "table_id": "table-record",
        "artifact_type": "table",
        "document_id": "doc-a",
        "caption": "Table 2. Results",
    }
    from app.rag.evidence import evidence_candidate_from_result

    typed = evidence_candidate_from_result(legacy).to_dict()
    assert score_results(resolved, [legacy], cutoff=30)["evidence_recall"] == 1.0
    assert score_results(resolved, [typed], cutoff=10)["evidence_recall"] == 1.0


def test_promotion_gate_requires_absolute_and_latency_gates() -> None:
    report = {
        "qrels_resolved": True,
        "model_calls": 0,
        "metrics": {
            "candidate_recall_at_30": 0.96,
            "evidence_recall_at_10": 0.90,
            "document_recall_at_5": 0.95,
            "multi_paper_coverage": 1.0,
            "modality_recall": {"text": 0.9, "table": 0.85},
            "latency_p50_ms": 200,
            "latency_p95_ms": 900,
            "forbidden_scope_leaks": 0,
        },
    }
    assert promotion_gate(report)["passed"] is True
    report["metrics"]["latency_p95_ms"] = 1501
    assert promotion_gate(report)["passed"] is False


def test_promotion_gate_rejects_final_or_exact_artifact_regression() -> None:
    metrics = {
        "candidate_recall_at_30": 0.98,
        "evidence_recall_at_10": 0.91,
        "document_recall_at_5": 0.96,
        "multi_paper_coverage": 1.0,
        "modality_recall": {"text": 0.9, "table": 0.9, "figure": 0.9},
        "latency_p50_ms": 100,
        "latency_p95_ms": 200,
        "forbidden_scope_leaks": 0,
        "subsets": {
            "table_numeric": {"cases": 2, "evidence_recall_at_10": 1.0},
            "figure_page": {"cases": 2, "evidence_recall_at_10": 1.0},
            "focused": {"cases": 2, "evidence_recall_at_10": 1.0},
        },
    }
    report = {"qrels_resolved": True, "model_calls": 0, "metrics": metrics}
    baseline = {**metrics, "evidence_recall_at_10": 0.92}
    result = promotion_gate(report, baseline=baseline)
    assert result["passed"] is False
    assert result["checks"]["final_evidence_noninferior"] is False


def test_suite_shape_rejects_imbalanced_cases() -> None:
    with pytest.raises(EvaluationContractError, match="category balance"):
        _validate_suite_shape([_case()], expected=1)


def test_dev_v21_proposal_has_balanced_shape_but_cannot_be_promoted() -> None:
    suite = ROOT / "evals/retrieval_v2/dev-v2.1.proposed.jsonl"
    if not suite.is_file():
        pytest.skip("retrieval-v2 fixtures have not been curated in this checkout")
    cases = load_cases(suite)
    _validate_suite_shape(cases, expected=60)
    with pytest.raises(EvaluationContractError, match="not audited"):
        load_cases(suite, require_audited=True)
    from app.core.config import get_settings
    from app.db.sqlite import connect

    with connect(get_settings().sqlite_db_path) as connection:
        resolved = [resolve_case(connection, case) for case in cases]
    assert len(resolved) == 60
    for case in cases:
        for locator in case.evidence_groups:
            caption = (locator.caption or "").strip()
            if len(caption) >= 24:
                assert caption.casefold() not in case.standalone_full.casefold()


def test_dev_v21_artifact_proposals_use_caption_backed_human_labels() -> None:
    """Never expose ingestion-order IDs as paper-facing Table/Figure numbers."""

    suite = ROOT / "evals/retrieval_v2/dev-v2.1.proposed.jsonl"
    if not suite.is_file():
        pytest.skip("retrieval-v2 fixtures have not been curated in this checkout")
    rows = [json.loads(line) for line in suite.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("category") not in {"table_numeric", "figure_page"}:
            continue
        locator = row["evidence_groups"][0]
        kind = str(locator["kind"])
        caption = str(locator.get("caption") or "")
        match = re.search(
            r"\b(?:table)\s*([0-9]+|[ivxlcdm]+)\b" if kind == "table" else r"\b(?:figure|fig\.?)\s*([0-9]+|[ivxlcdm]+)\b",
            caption,
            flags=re.IGNORECASE,
        )
        assert match is not None, row["case_id"]
        label = f"{'Table' if kind == 'table' else 'Figure'} {match.group(1)}"
        assert label.casefold() in str(row["standalone_full"]).casefold(), row["case_id"]


def test_v21_is_not_promotion_safe_when_language_and_context_are_confounded() -> None:
    suite = ROOT / "evals/retrieval_v2/dev-v2.1.jsonl"
    if not suite.is_file():
        pytest.skip("retrieval-v2 fixtures have not been curated in this checkout")
    cases = load_cases(suite)
    diagnostics = suite_quality_diagnostics(cases)
    assert diagnostics["promotion_safe"] is False
    assert any("language_context_confounded" in issue for issue in diagnostics["issues"])
    assert any("template_dominance" in issue for issue in diagnostics["issues"])


def test_v22_proposal_has_orthogonal_language_context_matrix() -> None:
    suite = ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl"
    if not suite.is_file():
        pytest.skip("v2.2 proposal has not been generated in this checkout")
    cases = load_cases(suite)
    diagnostics = suite_quality_diagnostics(cases)
    assert diagnostics["promotion_safe"] is True
    assert diagnostics["language_context_matrix"] == {
        ("vi", False): 15,
        ("vi", True): 15,
        ("en", False): 15,
        ("en", True): 15,
    }


def test_policy_round_robin_reserves_each_branch_and_deduplicates() -> None:
    branches = [
        [{"chunk_id": "a1"}, {"chunk_id": "a2"}],
        [{"chunk_id": "b1"}, {"chunk_id": "a2"}, {"chunk_id": "b3"}],
    ]
    merged = _round_robin(branches, 4)
    assert [item["chunk_id"] for item in merged] == ["a1", "b1", "a2", "b3"]


def test_policy_branch_specs_split_compare_per_document_and_multihop_by_facets() -> None:
    compare = SimpleNamespace(
        category="cross_document_compare",
        standalone_full="a.pdf b.pdf Compare their purposes and architectures",
    )
    specs = _branch_specs(compare, ["doc-a", "doc-b"])
    assert [spec[2] for spec in specs] == [["doc-a"], ["doc-b"]]

    multihop = SimpleNamespace(
        category="intra_document_multihop",
        standalone_full="paper.pdf task and architecture",
    )
    facet_specs = _branch_specs(multihop, ["doc-a"])
    assert len(facet_specs) >= 2
    assert all(spec[2] == ["doc-a"] for spec in facet_specs)


def test_audited_suite_requires_named_review_attestation(tmp_path) -> None:
    suite = tmp_path / "unattested.jsonl"
    suite.write_text(
        json.dumps(
            {
                "case_id": "negative",
                "category": "negative_unanswerable",
                "language": "en",
                "raw_query": "Does this library contain a weather forecast?",
                "standalone_full": "Does this library contain a weather forecast?",
                "standalone_anchor": "Does this library contain a weather forecast?",
                "rewrite_provenance": {
                    "full": [{"source": "raw_query", "text": "Does this library contain a weather forecast?"}],
                    "anchor": [{"source": "raw_query", "text": "Does this library contain a weather forecast?"}],
                },
                "evidence_groups": [],
                "curation_status": "audited",
                "rewrite_fingerprint": "rewrite-policy-v1",
                "language_reviewed": True,
                "qrels_reviewed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="audit is not approved"):
        load_cases(suite, require_audited=True)


def test_shadow_qrels_require_explicit_open_and_match_questions() -> None:
    questions = ROOT / "evals/retrieval_v2/shadow-v2.questions.jsonl"
    qrels = ROOT / "data/retrieval_eval/private/shadow-v2.qrels.jsonl"
    if not questions.is_file() or not qrels.is_file():
        pytest.skip("retrieval-v2 shadow fixtures have not been curated in this checkout")
    with pytest.raises(EvaluationContractError, match="deprecated"):
        load_cases(questions)


def test_runner_arms_keep_union_and_query_fusion_separate() -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc-a", "/tmp/a.pdf", "paper.pdf", "content-hash", 1, 0, 0, "now"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("chunk-a", "doc-a", "text", 2, "Results", "Pitch fusion improves tonal emotion recognition.", 0, "{}"),
    )
    from evaluate_dual_query_retrieval import CountingEmbeddings

    resolved = resolve_case(connection, _case())

    class FakeRag:
        async def search_hybrid(self, *, query, candidate_document_union, **_kwargs):
            return {
                "query": query,
                "results": [{"chunk_id": "chunk-a", "document_id": "doc-a", "page_number": 2}],
                "candidate_document_union": candidate_document_union,
                "query_origins": ["single"],
            }

        async def search_hybrid_queries(self, *, queries, **_kwargs):
            return {
                "queries": [query for _origin, query in queries],
                "query_origins": [origin for origin, _query in queries],
                "dual_query_rrf": True,
                "evidence_bundle": {
                    "candidates": [{"chunk_id": "chunk-a", "document_id": "doc-a", "page_number": 2}]
                },
                "results": [{"chunk_id": "chunk-a", "document_id": "doc-a", "page_number": 2}],
            }

    settings = SimpleNamespace(
        rerank_enabled=False,
        rerank_mode="embedding",
        rerank_cross_encoder_path=None,
        rerank_max_candidates=20,
        dual_query_rrf_k=60,
    )
    embeddings = CountingEmbeddings(SimpleNamespace())
    for arm in ("A0", "A1", "B-full", "B-anchor"):
        payload, _latency = asyncio.run(
            _run_arm(
                resolved,
                arm=arm,
                document_ids=list(resolved.focus_document_ids),
                rag=FakeRag(),
                store=object(),
                embeddings=embeddings,
                settings=settings,
                top_k=30,
                final=False,
            )
        )
        if arm.startswith("B-"):
            assert payload["query_origins"] == ["rewrite"]
        else:
            assert payload["candidate_document_union"] is (arm == "A1")


def _recommendation_reports(*, activation_rate: float, multi_query_cases: int, b_passed: bool = True,
                            a1_passed: bool = False, a0_candidate: float = 0.97,
                            a1_candidate: float | None = None, b_candidate: float | None = None) -> dict:
    a1_candidate = a0_candidate if a1_candidate is None else a1_candidate
    b_candidate = a1_candidate if b_candidate is None else b_candidate

    def report(candidate: float, *, passed: bool, activation: float = 0.0, multi: int = 0) -> dict:
        return {
            "metrics": {
                "candidate_recall_at_30": candidate,
                "evidence_recall_at_10": 0.95,
                "document_recall_at_5": 1.0,
                "multi_paper_coverage": 1.0,
                "modality_recall": {"text": 0.9, "table": 0.9, "figure": 0.9},
                "latency_p50_ms": 100.0,
                "latency_p95_ms": 200.0,
                "forbidden_scope_leaks": 0,
                "embedding_query_calls": 10,
                "query_variant_activation": {
                    "activation_rate": activation,
                    "multi_query_cases": multi,
                },
                "subsets": {},
            },
            "promotion": {"passed": passed},
        }

    return {
        "A0": report(a0_candidate, passed=True),
        "A1": report(a1_candidate, passed=a1_passed),
        "B-full": report(b_candidate, passed=b_passed, activation=activation_rate, multi=multi_query_cases),
        "B-anchor": report(b_candidate, passed=False),
    }


def test_recommendation_does_not_promote_noop_dual_query() -> None:
    reports = _recommendation_reports(activation_rate=0.0, multi_query_cases=0)
    decision = _recommend_action(reports)
    assert decision["recommended_action"] == "keep_dual_query_disabled_no_treatment_activation"
    assert decision["B_multi_query_cases"] == 0


def test_recommendation_attributes_gain_to_candidate_union() -> None:
    reports = _recommendation_reports(
        activation_rate=0.0,
        multi_query_cases=0,
        a1_passed=True,
        a0_candidate=0.80,
        a1_candidate=0.95,
        b_candidate=0.95,
    )
    assert _recommend_action(reports)["recommended_action"] == "promote_candidate_union_only"


def test_recommendation_requires_active_dual_treatment_for_promotion() -> None:
    reports = _recommendation_reports(
        activation_rate=0.5,
        multi_query_cases=2,
        b_passed=True,
        a0_candidate=0.80,
        a1_candidate=0.80,
        b_candidate=0.95,
    )
    decision = _recommend_action(reports)
    assert decision["recommended_action"] == "promote_dual_query"
    assert decision["B_candidate_gain_vs_A0"] == 0.15


def test_final_stage_obeys_reranker_configuration_without_second_retrieval(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_rerank(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        return list(reversed(kwargs["results"]))[: kwargs["top_k"]]

    monkeypatch.setattr("evaluate_dual_query_retrieval.rerank_candidates", fake_rerank)
    settings = SimpleNamespace(
        rerank_enabled=True,
        rerank_mode="embedding",
        rerank_cross_encoder_path=None,
        rerank_max_candidates=20,
    )
    selected, _ = asyncio.run(
        _run_final_stage(
            query="query",
            candidates=[{"chunk_id": "one"}, {"chunk_id": "two"}],
            embeddings=SimpleNamespace(),
            settings=settings,
        )
    )
    assert [item["chunk_id"] for item in selected] == ["two", "one"]
    assert calls[0]["top_k"] == 10

    settings.rerank_enabled = False
    selected, _ = asyncio.run(
        _run_final_stage(
            query="query",
            candidates=[{"chunk_id": "one"}, {"chunk_id": "two"}],
            embeddings=SimpleNamespace(),
            settings=settings,
        )
    )
    assert [item["chunk_id"] for item in selected] == ["one", "two"]
    assert len(calls) == 1


@pytest.mark.parametrize("runtime_scope", [None, ["beta", "alpha"]])
def test_evaluate_final_selection_uses_runtime_scope_not_gold_documents(monkeypatch, tmp_path, runtime_scope):
    from contextlib import nullcontext
    import evaluate_dual_query_retrieval as runner

    case = _case()
    # Gold documents deliberately differ from anonymous runtime scope, and
    # have no authoritative order. They may score retrieval, never route it.
    resolved = SimpleNamespace(case=case, document_ids=frozenset({"alpha", "beta"}))
    monkeypatch.setattr(runner, "get_settings", lambda: SimpleNamespace(
        dual_query_rrf_enabled=False, ollama_host="unused", embedding_model="unused",
        request_timeout_seconds=1, embedding_query_prefix="", embedding_document_prefix="",
    ))
    monkeypatch.setattr(runner, "connect", lambda _: nullcontext(None))
    monkeypatch.setattr(runner, "resolve_case", lambda *_: resolved)
    monkeypatch.setattr(runner, "LanceDBRetrievalStore", lambda *_: None)
    monkeypatch.setattr(runner, "RagService", lambda *_, **__: None)

    class Embeddings:
        async def embed_query(self, query):
            return [1.0]

    monkeypatch.setattr(runner, "OllamaEmbeddingProvider", lambda **_: Embeddings())
    monkeypatch.setattr(runner, "_runtime_document_scope", lambda *_: (runtime_scope, "runtime"))

    async def retrieve(*_, **__):
        return {"candidate_results": [], "query": "runtime query"}, 0.0

    async def final(**kwargs):
        assert kwargs["required_document_ids"] == runtime_scope
        raise RuntimeError("runtime-scope-verified")

    monkeypatch.setattr(runner, "_run_production_shaped_arm", retrieve)
    monkeypatch.setattr(runner, "score_results", lambda *_, **__: {})
    monkeypatch.setattr(runner, "_run_final_stage", final)
    with pytest.raises(RuntimeError, match="runtime-scope-verified"):
        asyncio.run(runner.evaluate(
            [case], db_path=tmp_path / "db", lancedb_path=tmp_path / "lance", artifact_root=tmp_path / "assets",
        ))


def test_audited_rewrite_fails_closed_when_gold_caption_leaks(tmp_path) -> None:
    suite = tmp_path / "leaky.jsonl"
    suite.write_text(
        json.dumps(
            {
                "case_id": "leaky",
                "category": "table_numeric",
                "language": "en",
                "raw_query": "What does the table report?",
                "standalone_full": "paper.pdf Table 2 Results of the proposed pitch fusion model",
                "standalone_anchor": "paper.pdf Table 2",
                "required_entities": ["paper.pdf", "Table 2"],
                "conversation_context": [],
                "working_state": {"current_artifact": "Table 2"},
                "rewrite_provenance": {
                    "full": [
                        {"source": "raw_query", "text": "What does the table report?"},
                        {"source": "catalog", "text": "paper.pdf"},
                        {"source": "working_state", "text": "{\"current_artifact\": \"Table 2\"}"},
                    ],
                    "anchor": [
                        {"source": "catalog", "text": "paper.pdf"},
                        {"source": "working_state", "text": "{\"current_artifact\": \"Table 2\"}"},
                    ],
                },
                "scope": "focused",
                "focus_documents": [],
                "evidence_groups": [
                    {
                        "group_id": "table",
                        "kind": "table",
                        "document": {"filename": "paper.pdf", "content_sha256": "hash"},
                        "page_number": 2,
                        "artifact_index": 1,
                        "caption": "Results of the proposed pitch fusion model",
                    }
                ],
                "curation_status": "audited",
                "rewrite_fingerprint": "test-v1",
                "language_reviewed": True,
                "qrels_reviewed": True,
                "audit": {
                    "status": "approved",
                    "reviewer": "test-reviewer",
                    "reviewed_at": "2026-08-24T00:00:00Z",
                    "notes": "fixture",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="qrel evidence leaked"):
        load_cases(suite, require_audited=True)


def test_audited_suite_fails_closed_on_rewrite_fingerprint_mismatch(tmp_path) -> None:
    suite = tmp_path / "fingerprint.jsonl"
    suite.write_text(
        json.dumps(
            {
                "case_id": "negative",
                "category": "negative_unanswerable",
                "language": "en",
                "raw_query": "Does this library contain a weather forecast?",
                "standalone_full": "Does this library contain a weather forecast?",
                "standalone_anchor": "Does this library contain a weather forecast?",
                "rewrite_provenance": {
                    "full": [{"source": "raw_query", "text": "Does this library contain a weather forecast?"}],
                    "anchor": [{"source": "raw_query", "text": "Does this library contain a weather forecast?"}],
                },
                "evidence_groups": [],
                "curation_status": "audited",
                "rewrite_fingerprint": "rewrite-policy-v1",
                "language_reviewed": True,
                "qrels_reviewed": True,
                "audit": {
                    "status": "approved",
                    "reviewer": "test-reviewer",
                    "reviewed_at": "2026-08-24T00:00:00Z",
                    "notes": "fixture",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="rewrite fingerprint mismatch"):
        load_cases(
            suite,
            require_audited=True,
            expected_rewrite_fingerprint="rewrite-policy-v2",
        )


def test_decomposed_payload_merge_interleaves_documents_and_deduplicates() -> None:
    merged = _merge_decomposed_payloads(
        [
            (
                "doc-a",
                {
                    "query": "compare",
                    "candidate_results": [
                        {"chunk_id": "a1", "document_id": "doc-a", "content": "a1"},
                        {"chunk_id": "a2", "document_id": "doc-a", "content": "a2"},
                    ],
                    "channel_candidates": {"fts": [{"chunk_id": "a1", "document_id": "doc-a"}], "vector": []},
                    "selected_document_ids": ["doc-a"],
                    "retrieval_channels": ["fts"],
                },
            ),
            (
                "doc-b",
                {
                    "query": "compare",
                    "candidate_results": [
                        {"chunk_id": "b1", "document_id": "doc-b", "content": "b1"},
                        {"chunk_id": "b2", "document_id": "doc-b", "content": "b2"},
                    ],
                    "channel_candidates": {"fts": [{"chunk_id": "b1", "document_id": "doc-b"}], "vector": []},
                    "selected_document_ids": ["doc-b"],
                    "retrieval_channels": ["fts"],
                },
            ),
        ],
        top_k=3,
    )
    assert [item["chunk_id"] for item in merged["candidate_results"]] == ["a1", "b1", "a2"]
    assert merged["selected_document_ids"] == ["doc-a", "doc-b"]
    assert merged["branch_results"][0]["origin"] == "document:doc-a"
