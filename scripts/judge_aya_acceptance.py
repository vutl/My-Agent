#!/usr/bin/env python3
"""Content judge for an Aya acceptance report (plan step E).

The structural checks in ``evaluate_aya_acceptance.py`` cannot tell whether an
answer says the right thing. This script asks a different model than the one
that answered (default ``cx/gpt-5.6-terra``) whether each answer conveys every
reference point in the turn's ``must_mention`` (paraphrase and translation
count) and whether it contradicts any of them. Turn-level verdicts and a
per-group summary are written next to the report.

Usage:
  backend/.venv/bin/python scripts/judge_aya_acceptance.py \
      --report data/retrieval_eval/internal/aya-acceptance-round4-20261014.json \
      --dataset evals/aya_acceptance_v1.jsonl --dataset evals/aya_acceptance_g5_fuzzy.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402

DEFAULT_JUDGE_MODEL = "cx/gpt-5.6-terra"

_PROMPT = """You grade an assistant's answer about research papers.

Question: {question}

Reference points written from the paper (the answer must convey each one; paraphrase,
translation and equivalent notation count; extra correct detail is fine):
{points}

{extra}
Answer:
<<<
{answer}
>>>

Return JSON only:
{{"points": [{{"point": "<reference point>", "covered": true|false}}],
  "contradiction": "<a statement in the answer that contradicts a reference point, or empty>",
  "verdict": "pass"|"fail"}}
verdict is "pass" when every point is covered, nothing contradicts them, and no forbidden
statement below appears (as a claim the answer makes, not as a correction it rejects)."""


def _load_turns(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    turns: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                turn = json.loads(line)
                turns[(turn["conversation"], int(turn["turn"]))] = turn
    return turns


def _parse(payload: str) -> dict[str, Any]:
    try:
        return json.loads(payload)
    except ValueError:
        start, end = payload.find("{"), payload.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(payload[start : end + 1])
            except ValueError:
                pass
    return {}


async def _judge(client: Any, model: str, turn: dict[str, Any], row: dict[str, Any], gate: asyncio.Semaphore) -> dict[str, Any]:
    points = [str(point) for point in turn.get("must_mention") or []]
    base = {"conversation": row["conversation"], "turn": row["turn"], "group": row["group"]}
    if not points and turn.get("expect_abstain"):
        points = ["States that the paper/library does not contain or report what was asked"]
    if not points and not turn.get("must_not_mention"):
        return {**base, "verdict": "skipped"}
    forbidden = [str(item) for item in turn.get("must_not_mention") or []]
    extra = ""
    if forbidden:
        extra += "Forbidden (the answer must not assert these):\n" + "\n".join(f"- {item}" for item in forbidden) + "\n"
    if turn.get("expect_abstain"):
        extra += "The correct behaviour is to say the paper/library does not contain this, without inventing it.\n"
    prompt = _PROMPT.format(
        extra=extra,
        question=turn["message"],
        points="\n".join(f"- {point}" for point in points),
        answer=str(row.get("answer") or "")[:12_000],
    )
    completion = None
    for attempt in range(3):
        try:
            async with gate:
                completion = await client.chat(
                    model=model, temperature=0.0, num_predict=1_200,
                    messages=[{"role": "user", "content": prompt}],
                )
            break
        except Exception as exc:  # noqa: BLE001 - one provider error must not void the run
            if attempt == 2:
                return {**base, "verdict": "error", "missing": [], "contradiction": "", "error": str(exc)[:200]}
            await asyncio.sleep(20 * (attempt + 1))
    data = _parse(getattr(completion, "message", "") or "")
    verdict = data.get("verdict") if data.get("verdict") in {"pass", "fail"} else "error"
    return {
        **base,
        "verdict": verdict,
        "missing": [item.get("point") for item in data.get("points") or [] if isinstance(item, dict) and not item.get("covered")],
        "contradiction": data.get("contradiction") or "",
    }


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    client = get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=180,
    )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    turns = _load_turns(args.dataset)
    gate = asyncio.Semaphore(args.concurrency)
    rows = [row for row in report["rows"] if (row["conversation"], int(row["turn"])) in turns]
    results = await asyncio.gather(*[
        _judge(client, args.model, turns[(row["conversation"], int(row["turn"]))], row, gate) for row in rows
    ])
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        if result["verdict"] == "skipped":
            continue
        group = summary.setdefault(result["group"], {"passed": 0, "total": 0, "errors": 0})
        group["total"] += 1
        group["passed"] += result["verdict"] == "pass"
        group["errors"] += result["verdict"] == "error"
        if result["verdict"] != "pass":
            print(f"[{result['conversation']} t{result['turn']} {result['group']}] {result['verdict']} "
                  f"missing={result.get('missing')} contradiction={result.get('contradiction')!r}")
    output = args.output or args.report.with_name(args.report.stem + "-judge.json")
    output.write_text(json.dumps({"model": args.model, "summary": summary, "turns": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print({group: f"{value['passed']}/{value['total']}" for group, value in sorted(summary.items())})
    print(f"written {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, action="append")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
