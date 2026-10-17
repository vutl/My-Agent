#!/usr/bin/env python3
"""Plan → review the plan → execute in a worktree → test → review → fix.

  python tools/harness/pipeline.py --task task.md
  python tools/harness/pipeline.py --task task.md --plan approved_plan.md --executor coder

Artifacts in .harness/runs/<run>/; the change is left on branch harness/<run>
in .harness/wt/<run> for the head agent to read and merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import HARNESS_DIR, REPO_ROOT, STATE_DIR, QuotaExhausted, load_config, run_role  # noqa: E402

PROMPTS = HARNESS_DIR / "prompts"
VERDICT_SCHEMA = HARNESS_DIR / "schemas" / "verdict.json"


def _prompt(name: str, **values: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").format(**values)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _verdict(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        return {"verdict": "revise", "issues": [{"severity": "blocking", "file": None, "line": None,
                                                 "problem": "reviewer returned no JSON", "fix": text[:500]}]}


def _blocking(verdict: dict) -> list[dict]:
    return [issue for issue in verdict.get("issues") or [] if issue.get("severity") == "blocking"]


def _feedback(issues: list[dict]) -> str:
    return "Fix these review findings:\n" + "\n".join(
        f"- [{issue.get('file') or '-'}:{issue.get('line') or '-'}] {issue['problem']} → {issue['fix']}" for issue in issues
    )


def _make_worktree(run: str) -> tuple[Path, str]:
    base = _git("rev-parse", "HEAD")
    worktree = STATE_DIR / "wt" / run
    _git("worktree", "add", "-q", "-b", f"harness/{run}", str(worktree), base)
    # Untracked runtime inputs the tests read (settings, local data).
    for relative in ("backend/.env", "data"):
        source, target = REPO_ROOT / relative, worktree / relative
        if source.exists() and not target.exists():
            target.symlink_to(source)
    return worktree, base


def _run_tests(worktree: Path, log: Path) -> tuple[bool, str]:
    command = str(load_config()["tests"]["command"]).format(python=REPO_ROOT / "backend" / ".venv" / "bin" / "python")
    completed = subprocess.run(command, shell=True, cwd=worktree, capture_output=True, text=True, timeout=1800)
    output = (completed.stdout or "") + (completed.stderr or "")
    log.write_text(output, encoding="utf-8")
    summary = next((line for line in reversed(output.splitlines()) if " passed" in line or " failed" in line), output[-300:])
    return completed.returncode == 0, summary.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="skip planning and use this approved plan")
    parser.add_argument("--run-id")
    parser.add_argument("--planner", default="planner")
    parser.add_argument("--executor", default="executor")
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    run = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = STATE_DIR / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    task = args.task.read_text(encoding="utf-8")
    (run_dir / "task.md").write_text(task, encoding="utf-8")
    report = [f"# Harness run {run}", ""]

    def log(line: str) -> None:
        print(line, flush=True)
        report.append(f"- {line}")

    # 1. Plan, reviewed by a different model until approved.
    if args.plan:
        plan = args.plan.read_text(encoding="utf-8")
        log("plan: provided")
    else:
        feedback = ""
        plan = ""
        for round_number in range(1, args.max_rounds + 2):
            planner = args.planner
            try:
                result = run_role(planner, _prompt("plan", task=task, feedback=feedback), log_dir=run_dir)
            except QuotaExhausted:
                planner = "planner_codex"
                result = run_role(planner, _prompt("plan", task=task, feedback=feedback), log_dir=run_dir)
            plan = result.text
            (run_dir / f"plan.r{round_number}.md").write_text(plan, encoding="utf-8")
            review = _verdict(run_role("plan_review", _prompt("plan_review", task=task, plan=plan),
                                       schema=VERDICT_SCHEMA, log_dir=run_dir).text)
            (run_dir / f"plan_review.r{round_number}.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
            blocking = _blocking(review)
            log(f"plan round {round_number} ({planner}): {review.get('verdict')}, {len(blocking)} blocking")
            if review.get("verdict") == "approve" or not blocking:
                break
            feedback = "Revise your previous plan.\nPrevious plan:\n" + plan + "\n\n" + _feedback(blocking)
        else:
            log("plan not approved; stopping")
            (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
            return 2
    (run_dir / "plan.md").write_text(plan, encoding="utf-8")

    # 2. Execute in an isolated worktree, then test and review until clean.
    worktree, base = _make_worktree(run)
    tests_command = str(load_config()["tests"]["command"]).format(
        python=REPO_ROOT / "backend" / ".venv" / "bin" / "python")
    feedback = ""
    status = "changes"
    for round_number in range(1, args.max_rounds + 2):
        result = run_role(
            args.executor, _prompt("execute", plan=plan, feedback=feedback, tests=tests_command),
            cwd=worktree, log_dir=run_dir, add_dirs=[REPO_ROOT / ".git"], timeout=5400,
        )
        (run_dir / f"execute.r{round_number}.md").write_text(result.text, encoding="utf-8")
        passed, summary = _run_tests(worktree, run_dir / f"tests.r{round_number}.log")
        log(f"execute round {round_number} ({result.model} via {result.account}): tests {'pass' if passed else 'FAIL'} — {summary}")
        review = _verdict(run_role(
            "reviewer",
            _prompt("review", branch=f"harness/{run}", base=base, task=task, plan=plan, tests=summary),
            cwd=worktree, schema=VERDICT_SCHEMA, log_dir=run_dir,
        ).text)
        (run_dir / f"review.r{round_number}.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
        blocking = _blocking(review)
        log(f"review round {round_number}: {review.get('verdict')}, {len(blocking)} blocking")
        if passed and not blocking:
            status = "ready"
            break
        issues = blocking + ([] if passed else [{"severity": "blocking", "file": None, "line": None,
                                                 "problem": f"tests fail: {summary}", "fix": "make the test suite pass"}])
        feedback = _feedback(issues)
    commits = _git("log", "--oneline", f"{base}..HEAD", cwd=worktree)
    log(f"status: {status}; branch harness/{run} in {worktree}")
    report += ["", "## Commits", "", commits or "(none)"]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
