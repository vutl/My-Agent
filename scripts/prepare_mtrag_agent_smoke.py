#!/usr/bin/env python3
"""Prepare a small official-reference MTRAG suite for an isolated Aya backend."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from mtrag_eval_lib import DOMAINS, MTRAG_ROOT, PROJECT_ROOT, domain_from_collection


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "retrieval_eval"
    / "public"
    / "prepared"
    / "mtrag-reference-agent-smoke-v1.jsonl"
)


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _answerability(item: dict[str, Any]) -> str:
    raw = item.get("Answerability") or item.get("answerability") or []
    values = raw if isinstance(raw, list) else [raw]
    normalized = {str(value).upper() for value in values}
    if "ANSWERABLE" in normalized or "PARTIAL" in normalized:
        return "answerable"
    return "unanswerable"


def _current_question(item: dict[str, Any]) -> str:
    for message in reversed(item.get("input") or []):
        if message.get("speaker") == "user":
            return str(message.get("text") or "").strip()
    return ""


def _target_answers(item: dict[str, Any]) -> list[str]:
    return [
        str(target.get("text") or "").strip()
        for target in item.get("targets") or []
        if str(target.get("text") or "").strip()
    ]


def _reference_prompt(item: dict[str, Any], *, max_context_chars: int) -> str:
    contexts: list[str] = []
    used = 0
    for index, context in enumerate((item.get("contexts") or [])[:10], start=1):
        text = str(context.get("text") or "").strip()
        if not text:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        clipped = text[:remaining]
        contexts.append(f"[REFERENCE {index}]\n{clipped}")
        used += len(clipped)
    context_block = "\n\n".join(contexts) if contexts else "(no reference passage)"
    return (
        "This is an evaluation turn using public benchmark references, not "
        "private indexed content. Answer in English using only the reference "
        "passages below. If the references do not support an answer, say that the "
        "provided references are insufficient.\n\n"
        f"{context_block}\n\n"
        f"USER QUESTION:\n{_current_question(item)}"
    )


def prepare_cases(
    *,
    turns_per_domain: int = 3,
    max_context_chars: int = 6_000,
    mtrag_root: Path = MTRAG_ROOT,
) -> list[dict[str, Any]]:
    tasks = _read_tasks(mtrag_root / "mtrag-human" / "generation_tasks" / "reference.jsonl")
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in tasks:
        domain = domain_from_collection(item.get("Collection"))
        if domain in DOMAINS:
            by_conversation[str(item["conversation_id"])].append(item)
    for items in by_conversation.values():
        items.sort(key=lambda item: int(item.get("turn") or 0))

    selected: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        candidates = [
            items
            for items in by_conversation.values()
            if items and domain_from_collection(items[0].get("Collection")) == domain
            and len(items) >= turns_per_domain
        ]
        # Prefer a short conversation slice containing both answerability modes;
        # tie-breaking by conversation id makes the fixture deterministic.
        candidates.sort(
            key=lambda items: (
                -len({_answerability(item) for item in items[:turns_per_domain]}),
                str(items[0]["conversation_id"]),
            )
        )
        if not candidates:
            raise RuntimeError(f"No {domain} conversation has {turns_per_domain} turns")
        selected[domain] = candidates[0][:turns_per_domain]

    cases: list[dict[str, Any]] = []
    for domain in DOMAINS:
        items = selected[domain]
        group = f"mtrag_reference_{domain}_{items[0]['conversation_id'][:8]}"
        for turn_index, item in enumerate(items, start=1):
            answerability = _answerability(item)
            targets = _target_answers(item)
            cases.append(
                {
                    "case_id": f"{group}_t{turn_index}",
                    "conversation_group": group,
                    "turn_index": turn_index,
                    "message": _reference_prompt(
                        item,
                        max_context_chars=max_context_chars,
                    ),
                    # WildBench separately evaluates auto-routing. This suite
                    # isolates generation and conversational memory over the
                    # official supplied contexts.
                    "mode": "chat",
                    "expected_route": "chat",
                    "expected_document_ids": [],
                    "forbidden_document_ids": [],
                    "must_cover_all": False,
                    "expected_abstention": answerability == "unanswerable",
                    "reference_answers": targets,
                    "language": "en",
                    "category": "mtrag_official_reference_generation",
                    "provenance": {
                        "suite": "MTRAG Human",
                        "task_id": item.get("task_id"),
                        "conversation_id": item.get("conversation_id"),
                        "domain": domain,
                        "answerability": answerability,
                        "reference_context_count": len(item.get("contexts") or []),
                    },
                }
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turns-per-domain", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=6_000)
    args = parser.parse_args()
    cases = prepare_cases(
        turns_per_domain=max(1, args.turns_per_domain),
        max_context_chars=max(1_000, args.max_context_chars),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "cases": len(cases),
        "conversation_groups": len({case["conversation_group"] for case in cases}),
        "domains": list(DOMAINS),
        "answerable": sum(not case["expected_abstention"] for case in cases),
        "unanswerable": sum(case["expected_abstention"] for case in cases),
        "reference_token_f1": "diagnostic_only_no_invented_pass_threshold",
        "output": str(args.output.relative_to(PROJECT_ROOT)),
        "requires_isolated_backend": True,
        "production_corpus_modified": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
