from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mtrag_eval_lib import (  # noqa: E402
    DOMAINS,
    build_fts_index,
    ensure_isolated_index_path,
    fetch_passages,
    load_human_retrieval_cases,
    load_un_retrieval_cases,
    read_index_manifest,
    reciprocal_rank_fusion,
    retrieval_metrics,
    search_fts,
)
from evaluate_mtrag_candidate_rerank import (  # noqa: E402
    CandidateEmbeddingCache,
    _fingerprint,
    rank_candidate_ids,
)
from prepare_mtrag_agent_smoke import prepare_cases  # noqa: E402
from evaluate_mtrag_aya_rewrite import (  # noqa: E402
    _ensure_public_results_path,
    conversation_for_rewrite,
)
from evaluate_mtrag_aya_e2e import select_generation_tasks  # noqa: E402
from external_rag_eval_contract import (  # noqa: E402
    ExternalPassage,
    adapt_passages_to_aya_documents,
    format_conversation_context,
    is_abstention,
    rouge_l_f1,
    stable_stratified_sample,
    token_f1,
    token_recall,
)


def _write_corpus(root: Path, domain: str, rows: list[dict]) -> None:
    archive = root / "corpora" / "passage_level" / f"{domain}.jsonl.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr(f"{domain}.jsonl", payload)


def test_isolated_mtrag_index_build_search_and_manifest(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    mtrag_root = public_root / "raw" / "github" / "mtrag"
    output = public_root / "indexes" / "mtrag-test.sqlite"
    production_sentinel = tmp_path / "data" / "sqlite" / "app.db"
    production_sentinel.parent.mkdir(parents=True)
    production_sentinel.write_bytes(b"production-must-not-change")

    _write_corpus(
        mtrag_root,
        "fiqa",
        [
            {
                "_id": "rel-1",
                "title": "Market capitalization",
                "text": "Market cap is share price multiplied by outstanding shares.",
            },
            {
                "_id": "neg-1",
                "title": "Cooking",
                "text": "Boil pasta in salted water.",
            },
        ],
    )

    report = build_fts_index(
        output,
        domains=["fiqa"],
        mtrag_root=mtrag_root,
        public_root=public_root,
        batch_size=1,
    )

    assert report["ok"] is True
    assert report["counts"] == {"fiqa": 2}
    assert report["production_corpus_modified"] is False
    assert production_sentinel.read_bytes() == b"production-must-not-change"
    assert read_index_manifest(output)["evaluation_only"] is True
    with closing(sqlite3.connect(output)) as connection:
        hits = search_fts(
            connection,
            domain="fiqa",
            query="What does market cap mean?",
            top_k=2,
        )
    assert hits[0]["passage_id"] == "rel-1"
    with closing(sqlite3.connect(output)) as connection:
        passages = fetch_passages(connection, ["neg-1", "rel-1", "missing"])
    assert set(passages) == {"rel-1", "neg-1"}
    assert passages["rel-1"]["title"] == "Market capitalization"


def test_mtrag_index_rejects_any_path_outside_public_indexes(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    with pytest.raises(ValueError, match="must be below"):
        ensure_isolated_index_path(
            tmp_path / "data" / "lancedb" / "mtrag.sqlite",
            public_root=public_root,
        )


def test_candidate_embedding_cache_rejects_production_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be below"):
        CandidateEmbeddingCache(
            tmp_path / "data" / "lancedb" / "candidate.sqlite",
            fingerprint="test",
        )


def test_candidate_rerank_uses_only_lexical_candidates() -> None:
    semantic, hybrid = rank_candidate_ids(
        ["a", "b", "c"],
        query_vector=[1.0, 0.0],
        document_vectors={
            "a": [0.0, 1.0],
            "b": [1.0, 0.0],
            "c": [0.5, 0.5],
            "qrel_not_retrieved": [1.0, 0.0],
        },
    )
    assert semantic == ["b", "c", "a"]
    assert set(hybrid) == {"a", "b", "c"}
    assert "qrel_not_retrieved" not in semantic + hybrid


def test_candidate_embedding_fingerprint_changes_with_corpus_content() -> None:
    common = {
        "model": "embeddinggemma:300m",
        "document_prefix": "document: ",
        "max_chars": 8_000,
    }
    first = _fingerprint(**common, source_sha256={"fiqa": "sha-a"})
    second = _fingerprint(**common, source_sha256={"fiqa": "sha-b"})
    assert first != second


def test_mtrag_generation_task_conversation_adapter_excludes_latest_turn() -> None:
    latest, previous = conversation_for_rewrite(
        {
            "task_id": "conversation<::>2",
            "input": [
                {"speaker": "user", "text": "Who is Ada?"},
                {"speaker": "agent", "text": "Ada is a mathematician."},
                {"speaker": "user", "text": "Where was she born?"},
            ],
        },
        expected_latest_query="|user|: Where was she born?",
    )
    assert latest == "Where was she born?"
    assert previous == [
        {"role": "user", "content": "Who is Ada?"},
        {"role": "assistant", "content": "Ada is a mathematician."},
    ]


def test_mtrag_rewrite_cache_rejects_paths_outside_public_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be below"):
        _ensure_public_results_path(tmp_path / "rewrite.jsonl")


def test_mtrag_metric_math_handles_multiple_relevant_passages() -> None:
    metrics = retrieval_metrics(
        ["wrong", "rel-a", "rel-b"],
        {"rel-a", "rel-b"},
        cutoffs=(1, 3),
    )
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert 0.0 < metrics["ndcg@3"] < 1.0


def test_reciprocal_rank_fusion_rewards_agreement_and_keeps_unique_candidates() -> None:
    fused = reciprocal_rank_fusion(
        [["shared", "lexical", "duplicate", "duplicate"], ["semantic", "shared"]]
    )
    assert fused[0] == "shared"
    assert set(fused) == {"shared", "lexical", "duplicate", "semantic"}


def test_official_mtrag_case_adapters_match_pinned_release() -> None:
    human = load_human_retrieval_cases(query_mode="rewrite", domains=DOMAINS)
    un = load_un_retrieval_cases(query_mode="questions", domains=DOMAINS)

    assert len(human) == 777
    # The pinned public checkout currently includes qrels/corpora for the four
    # shared domains only. Banking/Telco are still announced as coming soon.
    assert len(un) == 332
    assert {case["domain"] for case in human} == set(DOMAINS)
    assert {case["domain"] for case in un} == set(DOMAINS)
    assert all(case["relevant_passage_ids"] for case in human + un)


def test_mtrag_reference_agent_smoke_is_deterministic_and_isolated() -> None:
    first = prepare_cases(turns_per_domain=3, max_context_chars=2_000)
    second = prepare_cases(turns_per_domain=3, max_context_chars=2_000)

    assert first == second
    assert len(first) == 12
    assert len({case["conversation_group"] for case in first}) == 4
    assert {case["provenance"]["domain"] for case in first} == set(DOMAINS)
    assert all(case["mode"] == "chat" for case in first)
    assert all(case["expected_route"] == "chat" for case in first)
    assert all(case["expected_document_ids"] == [] for case in first)
    assert all("not private indexed content" in case["message"] for case in first)
    assert all("Answer in English" in case["message"] for case in first)
    assert all("minimum_token_f1" not in case for case in first)


def test_external_passage_adapter_is_dataset_neutral_and_does_not_accept_qrels() -> None:
    documents = adapt_passages_to_aya_documents(
        [
            ExternalPassage("p-2", "arbitrary-corpus", "Second", "Canonical text 2"),
            ExternalPassage("p-1", "arbitrary-corpus", "First", "Canonical text 1"),
        ],
        channels_by_id={"p-2": ["lexical", "semantic"]},
    )

    assert [document["external_passage_id"] for document in documents] == ["p-2", "p-1"]
    assert documents[0]["document_id"] == "external:arbitrary-corpus:p-2"
    assert documents[0]["source_path"] == "external://arbitrary-corpus/p-2"
    assert documents[0]["retrieval_channels"] == ["lexical", "semantic"]
    assert "qrel" not in json.dumps(documents).lower()
    assert "target" not in json.dumps(documents).lower()


def test_external_eval_selection_is_stable_stratified_not_input_order_dependent() -> None:
    rows = [
        {"id": "a-1", "domain": "a", "label": "yes"},
        {"id": "a-2", "domain": "a", "label": "yes"},
        {"id": "b-1", "domain": "b", "label": "no"},
        {"id": "b-2", "domain": "b", "label": "no"},
    ]
    first = stable_stratified_sample(
        rows, strata=("domain", "label"), id_field="id", per_stratum=1
    )
    second = stable_stratified_sample(
        list(reversed(rows)), strata=("domain", "label"), id_field="id", per_stratum=1
    )
    assert first == second
    assert {(row["domain"], row["label"]) for row in first} == {("a", "yes"), ("b", "no")}


def test_external_eval_text_metrics_and_context_are_general() -> None:
    prediction = "The Cardinals play in London."
    target = "The Cardinals played a game in London."
    assert 0.0 < token_recall(prediction, target) <= 1.0
    assert 0.0 < token_f1(prediction, target) <= 1.0
    assert 0.0 < rouge_l_f1(prediction, target) <= 1.0
    assert rouge_l_f1("alpha", "omega") == 0.0
    assert is_abstention("I don't have enough information to answer that.")
    assert is_abstention(
        "The available excerpts don't directly confirm whether this plan allows it."
    )
    assert not is_abstention(prediction)
    assert format_conversation_context(
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
    ) == "User: First question\nAssistant: First answer"


def test_mtrag_generation_selection_covers_every_available_domain_label_stratum() -> None:
    rows = [
        {
            "task_id": f"{domain}-{label}-{index}",
            "_domain": domain,
            "_answerability": label,
        }
        for domain, label in (("a", "ANSWERABLE"), ("a", "PARTIAL"), ("b", "ANSWERABLE"))
        for index in range(2)
    ]
    selected = select_generation_tasks(
        rows,
        labels={"ANSWERABLE", "PARTIAL"},
        per_stratum=1,
    )
    assert len(selected) == 3
    assert {(row["_domain"], row["_answerability"]) for row in selected} == {
        ("a", "ANSWERABLE"),
        ("a", "PARTIAL"),
        ("b", "ANSWERABLE"),
    }
