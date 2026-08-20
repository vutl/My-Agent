#!/usr/bin/env python3
"""Run bounded end-to-end Aya generation on an isolated official MTRAG index.

The runner uses Aya's real QueryRewriteService, generic AgentService graph and
streaming answer client.  MTRAG remains an external passage collection: no
passage is ingested as a paper and no production data directory is opened.
Qrels and reference targets are used after generation for scoring only.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import APPROVED_9ROUTER_MODELS, get_settings  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402
from app.services.agent_service import AgentService  # noqa: E402
from app.services.evidence_validator import validate_answer_claims  # noqa: E402
from app.services.evidence_sufficiency_service import EvidenceSufficiencyService  # noqa: E402
from app.services.query_rewrite_service import QueryRewriteService  # noqa: E402
from app.services.tool_decision_service import IntentRouterService  # noqa: E402
from evaluate_mtrag_aya_rewrite import (  # noqa: E402
    _ensure_public_results_path,
    conversation_for_rewrite,
)
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
from mtrag_eval_lib import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DOMAINS,
    MTRAG_ROOT,
    PUBLIC_ROOT,
    domain_from_collection,
    fetch_passages,
    load_human_retrieval_cases,
    mean_metrics,
    reciprocal_rank_fusion,
    retrieval_metrics,
    search_fts,
)


DEFAULT_OUTPUT = PUBLIC_ROOT / "results" / "mtrag-aya-e2e-sol.json"
SUPPORTED_LABELS = ("ANSWERABLE", "PARTIAL", "UNANSWERABLE", "CONVERSATIONAL")


def _read_generation_tasks() -> list[dict[str, Any]]:
    path = MTRAG_ROOT / "mtrag-human" / "generation_tasks" / "reference.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            domain = domain_from_collection(item.get("Collection"))
            if domain not in DOMAINS:
                continue
            label = str((item.get("Answerability") or ["UNKNOWN"])[0]).upper()
            item["_domain"] = domain
            item["_answerability"] = label
            rows.append(item)
    return rows


def select_generation_tasks(
    rows: list[dict[str, Any]],
    *,
    labels: set[str],
    per_stratum: int,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["_answerability"] in labels]
    return stable_stratified_sample(
        filtered,
        strata=("_domain", "_answerability"),
        id_field="task_id",
        per_stratum=per_stratum,
    )


def _previous_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    messages = list(task.get("input") or [])[:-1]
    return [
        {
            "role": "assistant" if message.get("speaker") == "agent" else "user",
            "content": str(message.get("text") or "").strip(),
        }
        for message in messages
        if str(message.get("text") or "").strip()
    ]


def _latest_question(task: dict[str, Any]) -> str:
    messages = list(task.get("input") or [])
    if not messages or messages[-1].get("speaker") != "user":
        raise ValueError(f"Task {task.get('task_id')} has no latest user turn")
    return str(messages[-1].get("text") or "").strip()


def _target_text(task: dict[str, Any]) -> str:
    return "\n".join(
        str(target.get("text") or "").strip()
        for target in task.get("targets") or []
        if str(target.get("text") or "").strip()
    )


def _limitation_answer(assessment: Any) -> str:
    if assessment.verdict == "ambiguous":
        return (
            "The retrieved evidence does not resolve this question unambiguously. "
            f"{assessment.reason} Please clarify the intended entity or situation."
        )
    return (
        "The retrieved evidence does not directly answer that question. "
        f"{assessment.reason}"
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": round(statistics.median(values), 3) if values else None,
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


async def evaluate(
    *,
    index_path: Path,
    per_stratum: int,
    labels: set[str],
    candidate_k: int,
    top_k: int,
    model: str,
    temperature: float,
    sufficiency_gate: bool,
) -> dict[str, Any]:
    if model not in APPROVED_9ROUTER_MODELS:
        raise ValueError(f"Refusing unapproved Aya model: {model}")
    tasks = select_generation_tasks(
        _read_generation_tasks(), labels=labels, per_stratum=per_stratum
    )
    if not tasks:
        raise ValueError("No MTRAG generation tasks selected")

    qrel_cases = {
        str(case["query_id"]): case
        for case in load_human_retrieval_cases(query_mode="lastturn", domains=DOMAINS)
    }
    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        raise RuntimeError("MTRAG Aya E2E requires configured 9router/openai_compatible")
    client = get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    health = await client.health()
    available_models = set(health.get("models") or [])
    if not health.get("reachable"):
        raise RuntimeError(
            f"9router preflight failed at {settings.openai_api_base}: "
            f"{health.get('error') or 'unreachable'}"
        )
    if model not in available_models:
        raise RuntimeError(f"9router preflight model unavailable: {model}")
    rewrite_service = QueryRewriteService(client=client, default_model=model)
    router_service = IntentRouterService(client=client, default_model=model)
    agent_service = AgentService(client=client, default_model=model)
    sufficiency_service = EvidenceSufficiencyService(client=client, default_model=model)

    case_results: list[dict[str, Any]] = []
    with closing(sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)) as index:
        for task in tasks:
            task_id = str(task["task_id"])
            question = _latest_question(task)
            previous = _previous_messages(task)

            routing_started = time.perf_counter()
            decision = await router_service.decide(
                task=question,
                mode="auto",
                previous_messages=previous,
                has_recent_retrieval=bool(previous),
                allowed_tools=["search_local_docs"],
                working_topic=f"external:{task['_domain']}",
                resolved_document_ids=[f"external-collection:{task['_domain']}"],
                model=model,
            )
            routing_ms = (time.perf_counter() - routing_started) * 1000

            rewrite_ms = 0.0
            standalone_query = question
            rewrite_used = False
            if decision.use_local_retrieval:
                rewrite_started = time.perf_counter()
                rewrite = await rewrite_service.rewrite(
                    query=question,
                    previous_messages=previous,
                    model=model,
                )
                rewrite_ms = (time.perf_counter() - rewrite_started) * 1000
                standalone_query = rewrite.standalone_query
                rewrite_used = rewrite.rewrite_used

            retrieval_started = time.perf_counter()
            raw_hits = (
                search_fts(
                    index,
                    domain=task["_domain"],
                    query=question,
                    top_k=candidate_k,
                )
                if decision.use_local_retrieval
                else []
            )
            rewrite_hits = (
                search_fts(
                    index,
                    domain=task["_domain"],
                    query=standalone_query,
                    top_k=candidate_k,
                )
                if decision.use_local_retrieval
                else []
            )
            raw_ids = [str(hit["passage_id"]) for hit in raw_hits]
            rewrite_ids = [str(hit["passage_id"]) for hit in rewrite_hits]
            fused_ids = reciprocal_rank_fusion([raw_ids, rewrite_ids])
            selected_ids = fused_ids[:top_k]
            canonical = fetch_passages(index, selected_ids)
            missing = set(selected_ids) - set(canonical)
            if missing:
                raise ValueError(f"Index failed to resolve {len(missing)} passages")
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

            channels_by_id: dict[str, list[str]] = defaultdict(list)
            for passage_id in raw_ids:
                channels_by_id[passage_id].append("fts_original")
            for passage_id in rewrite_ids:
                channels_by_id[passage_id].append("fts_aya_rewrite")
            passages = [
                ExternalPassage(
                    passage_id=passage_id,
                    collection=str(canonical[passage_id]["domain"]),
                    title=str(canonical[passage_id]["title"]),
                    text=str(canonical[passage_id]["text"]),
                )
                for passage_id in selected_ids
            ]
            documents = adapt_passages_to_aya_documents(
                passages,
                channels_by_id=dict(channels_by_id),
            )

            sufficiency_started = time.perf_counter()
            assessment = None
            if sufficiency_gate and decision.use_local_retrieval:
                assessment = await sufficiency_service.assess(
                    question=question,
                    documents=documents,
                    model=model,
                )
            sufficiency_ms = (time.perf_counter() - sufficiency_started) * 1000

            graph_started = time.perf_counter()
            graph = await agent_service.run_graph(
                run_id=f"mtrag-eval-{uuid4()}",
                conversation_id=str(task["conversation_id"]),
                user_message_id=task_id,
                task=question,
                resolved_task=standalone_query,
                conversation_context=format_conversation_context(previous),
                answer_intent=decision.answer_intent,
                answer_depth=decision.answer_depth,
                answer_style="natural_technical",
                mode="auto",
                model=model,
                temperature=temperature,
                retrieved_docs=documents,
                tool_decision=decision.to_dict(),
                evidence_sufficiency_context=(
                    json.dumps(assessment.to_dict(), ensure_ascii=False)
                    if assessment is not None
                    else ""
                ),
            )
            graph_ms = (time.perf_counter() - graph_started) * 1000

            generation_started = time.perf_counter()
            first_token_ms: float | None = None
            answer_parts: list[str] = []
            finish_reason: str | None = None
            if assessment is not None and not assessment.can_answer:
                answer_parts.append(_limitation_answer(assessment))
                first_token_ms = 0.0
                finish_reason = "evidence_sufficiency_gate"
            else:
                async for chunk in agent_service.stream_final_answer(
                    prompt=graph.final_prompt,
                    model=model,
                    temperature=temperature,
                    answer_intent=decision.answer_intent,
                    answer_depth=decision.answer_depth,
                ):
                    if chunk.content:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - generation_started) * 1000
                        answer_parts.append(chunk.content)
                    if chunk.done:
                        finish_reason = chunk.finish_reason
            generation_ms = (time.perf_counter() - generation_started) * 1000
            answer = "".join(answer_parts).strip()
            target = _target_text(task)
            validation = validate_answer_claims(answer=answer, documents=documents)

            qrel_case = qrel_cases.get(task_id)
            method_metrics: dict[str, dict[str, float]] = {}
            if qrel_case:
                relevant = set(qrel_case["relevant_passage_ids"])
                method_metrics = {
                    "original": retrieval_metrics(raw_ids, relevant, cutoffs=(3, 5, 10)),
                    "aya_rewrite": retrieval_metrics(
                        rewrite_ids, relevant, cutoffs=(3, 5, 10)
                    ),
                    "dual_rrf": retrieval_metrics(
                        fused_ids, relevant, cutoffs=(3, 5, 10)
                    ),
                }

            label = str(task["_answerability"])
            abstained = is_abstention(answer)
            case_results.append(
                {
                    "task_id": task_id,
                    "conversation_id": str(task["conversation_id"]),
                    "turn": task.get("turn"),
                    "domain": task["_domain"],
                    "answerability": label,
                    "question_type": task.get("Question Type"),
                    "multi_turn": task.get("Multi-Turn"),
                    "question": question,
                    "standalone_query": standalone_query,
                    "rewrite_used": rewrite_used,
                    "retrieved_passage_ids": selected_ids,
                    "retrieval_metrics": method_metrics,
                    "answer": answer,
                    "target": target,
                    "generation_metrics": {
                        "token_recall": round(token_recall(answer, target), 6),
                        "token_f1": round(token_f1(answer, target), 6),
                        "rouge_l_f1": round(rouge_l_f1(answer, target), 6),
                        "abstained": abstained,
                        "abstention_correct": (
                            abstained if label == "UNANSWERABLE" else not abstained
                        ),
                        "numeric_claims_valid": validation.valid,
                    },
                    "numeric_validation": validation.to_dict(),
                    "evidence_sufficiency": assessment.to_dict() if assessment else None,
                    "route": graph.route,
                    "route_reason": decision.reason,
                    "selected_tools": graph.selected_tools,
                    "finish_reason": finish_reason,
                    "latency_ms": {
                        "routing": round(routing_ms, 3),
                        "rewrite": round(rewrite_ms, 3),
                        "retrieval": round(retrieval_ms, 3),
                        "sufficiency": round(sufficiency_ms, 3),
                        "graph": round(graph_ms, 3),
                        "first_token": round(first_token_ms, 3)
                        if first_token_ms is not None
                        else None,
                        "generation": round(generation_ms, 3),
                        "total": round(
                            routing_ms
                            + rewrite_ms
                            + retrieval_ms
                            + sufficiency_ms
                            + graph_ms
                            + generation_ms,
                            3,
                        ),
                    },
                }
            )

    retrieval_rows = [row for row in case_results if row["retrieval_metrics"]]
    retrieval_methods = ("original", "aya_rewrite", "dual_rrf")
    generation_keys = ("token_recall", "token_f1", "rouge_l_f1")
    latency_keys = (
        "routing",
        "rewrite",
        "retrieval",
        "sufficiency",
        "graph",
        "first_token",
        "generation",
        "total",
    )
    return {
        "schema_version": 1,
        "ok": True,
        "suite": "mtrag-human-aya-e2e",
        "model": model,
        "cases": len(case_results),
        "selection": {
            "method": "stable_sha256_per_domain_and_answerability",
            "per_stratum": per_stratum,
            "labels": sorted(labels),
            "distribution": dict(Counter(row["answerability"] for row in case_results)),
        },
        "retrieval": {
            "scored_cases": len(retrieval_rows),
            "candidate_k": candidate_k,
            "top_k": top_k,
            "metrics": {
                method: mean_metrics(
                    row["retrieval_metrics"][method] for row in retrieval_rows
                )
                for method in retrieval_methods
            }
            if retrieval_rows
            else {},
        },
        "generation": {
            "metrics": {
                key: round(
                    sum(float(row["generation_metrics"][key]) for row in case_results)
                    / len(case_results),
                    6,
                )
                for key in generation_keys
            },
            "abstention_accuracy": round(
                sum(row["generation_metrics"]["abstention_correct"] for row in case_results)
                / len(case_results),
                6,
            ),
            "numeric_claim_validation_rate": round(
                sum(row["generation_metrics"]["numeric_claims_valid"] for row in case_results)
                / len(case_results),
                6,
            ),
            "route_distribution": dict(Counter(row["route"] for row in case_results)),
            "sufficiency_gate_enabled": sufficiency_gate,
            "sufficiency_verdicts": dict(
                Counter(
                    row["evidence_sufficiency"]["verdict"]
                    for row in case_results
                    if row["evidence_sufficiency"]
                )
            ),
        },
        "latency_ms": {
            key: _latency_summary(
                [
                    float(row["latency_ms"][key])
                    for row in case_results
                    if row["latency_ms"][key] is not None
                ]
            )
            for key in latency_keys
        },
        "case_results": case_results,
        "production_corpus_modified": False,
        "limitations": [
            "Reference targets are model-authored, so lexical generation metrics are diagnostic rather than a complete correctness score.",
            "This bounded run forces the benchmark's assigned external corpus scope; natural open-domain tool routing is evaluated separately.",
            "The adapter evaluates text passages only and does not impersonate Aya's paper/table/figure catalog.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-stratum", type=int, default=1)
    parser.add_argument("--labels", nargs="+", choices=SUPPORTED_LABELS, default=list(SUPPORTED_LABELS))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="cx/gpt-5.6-sol")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sufficiency-gate", action="store_true")
    args = parser.parse_args()
    output = _ensure_public_results_path(args.output)
    report = asyncio.run(
        evaluate(
            index_path=args.index,
            per_stratum=max(1, args.per_stratum),
            labels=set(args.labels),
            candidate_k=max(args.top_k, args.candidate_k),
            top_k=max(1, args.top_k),
            model=args.model,
            temperature=args.temperature,
            sufficiency_gate=args.sufficiency_gate,
        )
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
