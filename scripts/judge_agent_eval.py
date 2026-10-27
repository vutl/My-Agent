#!/usr/bin/env python3
"""Grade ``judge_rubric`` criteria of an agent-eval run with the harness judge role.

The judge is ``judge_ag`` (tools/harness/cx.py), never the model that answered. It sees the goal,
the rubric, the judge notes, the final answer, the tool calls and the workspace tree, and returns a
pass/fail per rubric criterion. Verdicts are written to ``judge.jsonl`` in the run directory and
``results.jsonl`` / ``report.md`` are recomputed.

Usage: python3 scripts/judge_agent_eval.py --run DIR [--role judge_ag]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval_lib import PASS, FAIL, PENDING, render_report, summarize  # noqa: E402

SCHEMA = ROOT / "tools/harness/schemas/agent_judge.json"


def build_prompt(result: dict, evidence: dict) -> str:
    rubrics = "\n".join(f"- [{item['id']}] {item['rubric']}" for item in result["judge_items"])
    calls = "\n".join(f"- {call['tool']} {json.dumps(call.get('args'), ensure_ascii=False)[:300]}"
                      for call in evidence.get("tool_calls", [])[:60]) or "(none)"
    tree = "\n".join(f"- {name}" for name in evidence.get("workspace_tree", [])[:80]) or "(empty)"
    return (
        "You grade one run of an AI assistant (Aya) against rubric criteria. Judge only what the evidence shows; "
        "a claim in the answer that the tool calls or files do not support is a failure. Reply with JSON matching "
        "the schema: one verdict per criterion id.\n\n"
        f"## Goal given to the assistant\n{result['goal']}\n\n"
        f"## Notes for the judge\n{result.get('notes_for_judge') or '(none)'}\n\n"
        f"## Criteria\n{rubrics}\n\n"
        f"## Final answer\n{evidence.get('answer') or '(empty)'}\n\n"
        f"## Tool calls\n{calls}\n\n## Files in the workspace afterwards\n{tree}\n"
    )


def parse_verdicts(text: str) -> dict[str, tuple[bool, str]]:
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    return {str(item["criterion"]): (bool(item["pass"]), str(item.get("note") or "")) for item in data["verdicts"]}


def run_judge(prompt: str, role: str = "judge_ag") -> str:
    with tempfile.TemporaryDirectory() as scratch:
        prompt_file = Path(scratch) / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(ROOT / "tools/harness/cx.py"), "run", role, "--prompt-file", str(prompt_file),
             "--schema", str(SCHEMA)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"judge failed: {done.stderr[-300:]}")
    return done.stdout


def apply_verdicts(result: dict, verdicts: dict[str, tuple[bool, str]]) -> dict:
    for criterion in result["criteria"]:
        if criterion["status"] != PENDING:
            continue
        if criterion["id"] in verdicts:
            passed, note = verdicts[criterion["id"]]
            criterion["status"], criterion["detail"] = (PASS if passed else FAIL), f"judge: {note}"[:300]
        else:
            criterion["status"], criterion["detail"] = FAIL, "judge returned no verdict"
    return {**result, **summarize(result["criteria"], result["forbidden"])}


def judge_run(run: Path, judge: Callable[[str], str] = run_judge) -> list[dict]:
    results = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    log = []
    for index, result in enumerate(results):
        if not any(item["status"] == PENDING for item in result["criteria"]):
            continue
        evidence = json.loads(Path(result["evidence"]).read_text(encoding="utf-8"))
        try:
            verdicts = parse_verdicts(judge(build_prompt(result, evidence)))
        except Exception as exc:  # noqa: BLE001 - leave the task pending so it can be rejudged
            log.append({"id": result["id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        results[index] = apply_verdicts(result, verdicts)
        log.append({"id": result["id"], "verdicts": {key: {"pass": value[0], "note": value[1]}
                                                     for key, value in verdicts.items()}})
    (run / "judge.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in log),
                                     encoding="utf-8")
    (run / "results.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
                                       encoding="utf-8")
    (run / "report.md").write_text(render_report(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", default="judge_ag")
    args = parser.parse_args(argv)
    results = judge_run(args.run.expanduser().resolve(), lambda prompt: run_judge(prompt, args.role))
    pending = sum(item["verdict"] == PENDING for item in results)
    print(f"judged; {pending} task(s) still pending (see judge.jsonl)")
    return 1 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
