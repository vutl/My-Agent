#!/usr/bin/env python3
"""Judge saved assistant answers with the harness `judge` role, sharded by
conversation and run in parallel across the Codex accounts.

  python tools/harness/judge.py --set-dir data/retrieval_eval/sealed/v2 \
      --answers answers-rerun1-20261016.jsonl --out judged-rerun1-20261017.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import HARNESS_DIR, run_role  # noqa: E402


def _turn_keys(path: Path, conversations: set[str] | None = None) -> list[tuple[str, int]]:
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if conversations is None or row["conversation"] in conversations:
                keys.append((row["conversation"], int(row["turn"])))
    return keys


def _judge_shard(set_dir: Path, questions: str, answers: str, notes: str, shard: list[str], log_dir: Path) -> list[dict]:
    prompt = (HARNESS_DIR / "prompts" / "judge.md").read_text(encoding="utf-8").format(
        questions=questions, answers=answers, notes=notes, conversations=", ".join(shard),
    )
    result = run_role("judge", prompt, cwd=set_dir, schema=HARNESS_DIR / "schemas" / "judge.json", log_dir=log_dir)
    try:
        return list(json.loads(result.text).get("verdicts") or [])
    except ValueError:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--questions", default="questions.jsonl")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--notes", default="questions.notes.md")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-size", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()

    set_dir = args.set_dir.resolve()
    expected = _turn_keys(set_dir / args.answers)
    conversations = sorted({conversation for conversation, _ in expected})
    notes = f"{args.notes} explains the groups." if (set_dir / args.notes).exists() else ""
    log_dir = set_dir / (Path(args.out).stem + "-logs")
    out = set_dir / args.out
    expected_set = set(expected)
    verdicts: dict[tuple[str, int], dict] = {}
    # Resume: verdicts already written, and finished shards in the call logs.
    sources = [out] if out.exists() else []
    sources += sorted(log_dir.glob("judge-*.last.md")) if log_dir.exists() else []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        try:
            if source.suffix == ".md":
                rows = json.loads(text).get("verdicts") or []
            else:
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError:
            continue
        for row in rows:
            key = (str(row.get("conversation")), int(row.get("turn") or 0))
            if key in expected_set:
                verdicts[key] = row

    def save() -> None:
        out.write_text("".join(json.dumps(verdicts[key], ensure_ascii=False) + "\n" for key in expected if key in verdicts),
                       encoding="utf-8")
    for attempt in range(2):
        missing = sorted({conversation for conversation, turn in expected if (conversation, turn) not in verdicts})
        if not missing:
            break
        shards = [missing[index:index + args.shard_size] for index in range(0, len(missing), args.shard_size)]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for rows in pool.map(
                lambda shard: _judge_shard(set_dir, args.questions, args.answers, notes, shard, log_dir), shards,
            ):
                for row in rows:
                    key = (str(row.get("conversation")), int(row.get("turn") or 0))
                    if key in expected_set:
                        verdicts[key] = row
                save()
        print(f"attempt {attempt + 1}: {len(verdicts)}/{len(expected)} turns judged", flush=True)
    save()
    counts = {name: sum(1 for row in verdicts.values() if row["verdict"] == name) for name in ("correct", "partial", "incorrect")}
    print(json.dumps({**counts, "fabricated": sum(bool(r["fabricated_number"]) for r in verdicts.values()),
                      "false_abstention": sum(bool(r["false_abstention"]) for r in verdicts.values()),
                      "missing": len(expected) - len(verdicts), "out": str(out)}))
    return 0 if len(verdicts) == len(expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
