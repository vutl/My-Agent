#!/usr/bin/env python3
"""Build lightweight adapters/catalogs for Aya's pinned public benchmarks."""

from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = PROJECT_ROOT / "data" / "retrieval_eval" / "public"
RAW_ROOT = PUBLIC_ROOT / "raw"
PREPARED_ROOT = PUBLIC_ROOT / "prepared"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative(path: Path) -> str:
    return str(path.relative_to(PUBLIC_ROOT))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _record(
    *,
    case_id: str,
    suite: str,
    source_path: Path,
    source_locator: dict[str, Any],
    capabilities: Iterable[str],
    runner_mode: str,
    turn_count: int,
    has_gold_answer: bool,
    has_inline_context: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "suite": suite,
        "source_path": _relative(source_path),
        "source_locator": source_locator,
        "capabilities": list(dict.fromkeys(capabilities)),
        "runner_mode": runner_mode,
        "turn_count": turn_count,
        "has_gold_answer": has_gold_answer,
        "has_inline_context": has_inline_context,
        "metadata": metadata or {},
    }


def _mtrag_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    conversations_path = (
        RAW_ROOT / "github" / "mtrag" / "mtrag-human" / "conversations" / "conversations.json"
    )
    for index, conversation in enumerate(_json(conversations_path)):
        user_turns = sum(
            message.get("speaker") == "user" for message in conversation.get("messages") or []
        )
        records.append(
            _record(
                case_id=f"mtrag-human:{index:04d}",
                suite="mtrag-human",
                source_path=conversations_path,
                source_locator={"row_index": index},
                capabilities=(
                    "multi_turn_rag",
                    "non_standalone_followup",
                    "answerability",
                    "retrieval_and_generation",
                ),
                runner_mode="isolated_rag_conversation",
                turn_count=user_turns,
                has_gold_answer=True,
                has_inline_context=True,
                metadata={"domain": conversation.get("domain")},
            )
        )

    un_path = (
        RAW_ROOT
        / "github"
        / "mtrag"
        / "mtragun-human"
        / "generation_tasks"
        / "reference.jsonl"
    )
    for index, task in enumerate(_jsonl(un_path)):
        records.append(
            _record(
                case_id=f"mtrag-un:{task.get('task_id') or index}",
                suite="mtrag-un",
                source_path=un_path,
                source_locator={"row_index": index},
                capabilities=(
                    "multi_turn_rag",
                    "unanswerable",
                    "underspecified",
                    "non_standalone_followup",
                ),
                runner_mode="inline_context_conversation",
                turn_count=sum(
                    item.get("speaker") == "user" for item in task.get("input") or []
                )
                + 1,
                has_gold_answer=bool(task.get("targets")),
                has_inline_context=bool(task.get("contexts")),
                metadata={"dataset": task.get("dataset"), "turn": task.get("turn")},
            )
        )
    return records


def _multichallenge_records() -> list[dict[str, Any]]:
    path = (
        RAW_ROOT
        / "github"
        / "multichallenge"
        / "data"
        / "benchmark_questions.jsonl"
    )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(_jsonl(path)):
        conversation = item.get("CONVERSATION") or []
        turn_count = len(conversation) if isinstance(conversation, list) else 1
        records.append(
            _record(
                case_id=f"multichallenge:{item.get('QUESTION_ID') or index}",
                suite="multichallenge",
                source_path=path,
                source_locator={"row_index": index},
                capabilities=(
                    "instruction_retention",
                    "inference_memory",
                    "self_coherence",
                    "version_editing",
                ),
                runner_mode="direct_model_conversation",
                turn_count=turn_count,
                has_gold_answer=False,
                has_inline_context=True,
                metadata={
                    "axis": item.get("AXIS"),
                    "pass_criteria": item.get("PASS_CRITERIA"),
                },
            )
        )
    return records


def _chatrag_records() -> list[dict[str, Any]]:
    base = RAW_ROOT / "huggingface" / "chatrag_bench" / "data"
    paths = {
        "convfinqa": base / "convfinqa" / "dev.json",
        "sqa": base / "sqa" / "test.json",
        "inscit": base / "inscit" / "dev.json",
        "doqa-cooking": base / "doqa" / "test_cooking.json",
        "doqa-movies": base / "doqa" / "test_movies.json",
        "doqa-travel": base / "doqa" / "test_travel.json",
        "hybridial": base / "hybridial" / "test.json",
    }
    records: list[dict[str, Any]] = []
    for subset, path in paths.items():
        for index, item in enumerate(_json(path)):
            capabilities = ["conversational_qa", "inline_context"]
            if subset in {"convfinqa", "sqa"}:
                capabilities.extend(("table_reasoning", "numeric_answer"))
            if subset == "inscit":
                capabilities.extend(("information_seeking", "grounded_context"))
            records.append(
                _record(
                    case_id=f"chatrag:{subset}:{index:05d}",
                    suite=f"chatrag-{subset}",
                    source_path=path,
                    source_locator={"row_index": index},
                    capabilities=capabilities,
                    runner_mode="inline_context_conversation",
                    turn_count=sum(
                        message.get("role") == "user" for message in item.get("messages") or []
                    ),
                    has_gold_answer=bool(item.get("answers")),
                    has_inline_context=bool(item.get("ctxs")),
                    metadata={"context_count": len(item.get("ctxs") or [])},
                )
            )
    return records


def _spiqa_records() -> list[dict[str, Any]]:
    base = RAW_ROOT / "huggingface" / "spiqa"
    records: list[dict[str, Any]] = []

    path_a = base / "test-A" / "SPIQA_testA.json"
    for paper_id, paper in _json(path_a).items():
        for qa_index, qa in enumerate(paper.get("qa") or []):
            reference = qa.get("reference")
            records.append(
                _record(
                    case_id=f"spiqa-a:{paper_id}:{qa_index:04d}",
                    suite="spiqa-test-a",
                    source_path=path_a,
                    source_locator={
                        "paper_id": paper_id,
                        "qa_index": qa_index,
                        "image_reference": reference,
                    },
                    capabilities=("scientific_figure_qa", "table_qa", "visual_grounding"),
                    runner_mode="multimodal_qa",
                    turn_count=1,
                    has_gold_answer=bool(qa.get("answer")),
                    has_inline_context=False,
                    metadata={"has_explanation": bool(qa.get("explanation"))},
                )
            )

    for split in ("B", "C"):
        path = base / f"test-{split}" / f"SPIQA_test{split}.json"
        for key, item in _json(path).items():
            questions = item.get("question") or []
            if not isinstance(questions, list):
                questions = [questions]
            answers = item.get("answer") or []
            for question_index, question in enumerate(questions):
                answer = answers[question_index] if question_index < len(answers) else None
                records.append(
                    _record(
                        case_id=f"spiqa-{split.lower()}:{key}:{question_index:04d}",
                        suite=f"spiqa-test-{split.lower()}",
                        source_path=path,
                        source_locator={
                            "record_key": key,
                            "question_index": question_index,
                        },
                        capabilities=(
                            "scientific_figure_qa",
                            "table_qa",
                            "multi_artifact_reasoning",
                            "paper_context",
                        ),
                        runner_mode="multimodal_qa",
                        turn_count=1,
                        has_gold_answer=bool(answer),
                        has_inline_context=bool(
                            item.get("full_text") or item.get("evidential_info")
                        ),
                        metadata={
                            "paper_id": item.get("paper_id") or item.get("arxiv_id"),
                            "question_key": _list_item(item.get("question_key"), question_index),
                            "figure_in_evidence": _list_item(
                                item.get("Is_figure_in_evidence"), question_index
                            ),
                            "table_in_evidence": _list_item(
                                item.get("Is_table_in_evidence"), question_index
                            ),
                            "question": question,
                        },
                    )
                )
    return records


def _list_item(value: Any, index: int) -> Any:
    if isinstance(value, list) and index < len(value):
        return value[index]
    return None


def _mmlong_records() -> list[dict[str, Any]]:
    path = RAW_ROOT / "github" / "mmlongbench_doc" / "data" / "samples.json"
    records: list[dict[str, Any]] = []
    for index, item in enumerate(_json(path)):
        records.append(
            _record(
                case_id=f"mmlongbench-doc:{index:04d}",
                suite="mmlongbench-doc",
                source_path=path,
                source_locator={"row_index": index, "document": item.get("doc_id")},
                capabilities=(
                    "long_document_qa",
                    "cross_page_reasoning",
                    "table_chart_image_layout",
                    "evidence_page_retrieval",
                ),
                runner_mode="isolated_multimodal_rag",
                turn_count=1,
                has_gold_answer=bool(item.get("answer")),
                has_inline_context=False,
                metadata={
                    "evidence_pages": item.get("evidence_pages"),
                    "evidence_sources": item.get("evidence_sources"),
                    "answer_format": item.get("answer_format"),
                },
            )
        )
    return records


def _wildbench_records_and_routing_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to prepare WildBench") from exc

    path = (
        RAW_ROOT
        / "huggingface"
        / "wildbench"
        / "v2"
        / "test-00000-of-00001.parquet"
    )
    rows = parquet.read_table(path).to_pylist()
    records: list[dict[str, Any]] = []
    eligible: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for index, item in enumerate(rows):
        messages = item.get("conversation_input") or []
        records.append(
            _record(
                case_id=f"wildbench-v2:{item.get('id') or index}",
                suite="wildbench-v2",
                source_path=path,
                source_locator={"row_index": index},
                capabilities=("real_user_prompt", "instruction_following", "checklist_judging"),
                runner_mode="direct_model_checklist",
                turn_count=len(messages),
                has_gold_answer=bool((item.get("references") or {}).get("gpt-4")),
                has_inline_context=True,
                metadata={
                    "primary_tag": item.get("primary_tag"),
                    "secondary_tags": item.get("secondary_tags") or [],
                    "checklist_count": len(item.get("checklist") or []),
                },
            )
        )
        if len(messages) != 1:
            continue
        message = messages[0]
        content = str(message.get("content") or "").strip()
        if (
            not content
            or len(content) > 800
            or message.get("toxic") is True
            or message.get("redacted") is True
            or str(message.get("language") or "").lower() != "english"
        ):
            continue
        eligible[str(item.get("primary_tag") or "other")].append(
            {"item": item, "content": content}
        )

    routing_cases: list[dict[str, Any]] = []
    buckets = deque(sorted(eligible))
    while buckets and len(routing_cases) < 50:
        tag = buckets.popleft()
        bucket = eligible[tag]
        if bucket:
            selected = bucket.popleft()
            item = selected["item"]
            index = len(routing_cases) + 1
            routing_cases.append(
                {
                    "case_id": f"public_wildbench_route_{index:03d}",
                    "conversation_group": f"public_wildbench_route_{index:03d}",
                    "turn_index": 1,
                    "message": selected["content"],
                    "expected_route": "chat",
                    "expected_document_ids": [],
                    "must_cover_all": False,
                    "expected_abstention": False,
                    "language": "en",
                    "category": "public_nonlocal_routing_negative",
                    "provenance": {
                        "suite": "WildBench-v2",
                        "source_id": item.get("id"),
                        "primary_tag": item.get("primary_tag"),
                    },
                }
            )
        if bucket:
            buckets.append(tag)
    if len(routing_cases) != 50:
        raise RuntimeError(f"Expected 50 WildBench routing cases, got {len(routing_cases)}")
    return records, routing_cases


def build_catalog() -> dict[str, Any]:
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    wildbench_records, routing_cases = _wildbench_records_and_routing_cases()
    records = [
        *wildbench_records,
        *_mtrag_records(),
        *_multichallenge_records(),
        *_chatrag_records(),
        *_spiqa_records(),
        *_mmlong_records(),
    ]
    records.sort(key=lambda item: item["case_id"])

    catalog_path = PREPARED_ROOT / "catalog-v1.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    routing_path = PREPARED_ROOT / "wildbench-routing-v1.jsonl"
    routing_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in routing_cases),
        encoding="utf-8",
    )

    by_suite: dict[str, int] = defaultdict(int)
    by_mode: dict[str, int] = defaultdict(int)
    for record in records:
        by_suite[record["suite"]] += 1
        by_mode[record["runner_mode"]] += 1
    summary = {
        "schema_version": 1,
        "cases": len(records),
        "by_suite": dict(sorted(by_suite.items())),
        "by_runner_mode": dict(sorted(by_mode.items())),
        "runnable_agent_routing_cases": len(routing_cases),
        "catalog": _display_path(catalog_path),
        "routing_suite": _display_path(routing_path),
        "production_corpus_modified": False,
    }
    (PREPARED_ROOT / "catalog-v1.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(build_catalog(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
