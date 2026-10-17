#!/usr/bin/env python3
"""Run one harness role, or show account status.

  python tools/harness/cx.py status
  python tools/harness/cx.py run judge --prompt-file p.md [--schema s.json] [--out answer.md]
  echo "review the diff on branch X" | python tools/harness/cx.py run reviewer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import CALL_LOG, REPO_ROOT, _read_state, accounts, load_config, run_role  # noqa: E402


def status() -> int:
    state = _read_state()
    now = time.time()
    print("accounts:")
    configured = {account["name"]: account for account in load_config().get("accounts", [])}
    logged_in = {account["name"] for account in accounts()}
    for name, account in configured.items():
        entry = state.get(name, {})
        cooling = entry.get("cooldown_until", 0) - now
        print(
            f"  {name:10s} {account['codex_home']:20s} "
            f"{'logged in' if name in logged_in else 'NOT LOGGED IN'}"
            + (f", cooling {int(cooling // 60)} min" if cooling > 0 else "")
        )
    print("roles:")
    for role, config in load_config().get("roles", {}).items():
        print(f"  {role:13s} {config['engine']:6s} {config['model']}")
    if CALL_LOG.exists():
        lines = CALL_LOG.read_text(encoding="utf-8").splitlines()[-8:]
        print("recent calls:")
        for line in lines:
            entry = json.loads(line)
            print(f"  {entry['ts']} {entry['role']:12s} {entry['model']:12s} {entry['account']:8s} "
                  f"{entry['seconds']:>6}s ok={entry['ok']} limited={entry['rate_limited']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("role")
    run.add_argument("prompt", nargs="?")
    run.add_argument("--prompt-file", type=Path)
    run.add_argument("--cwd", type=Path, default=REPO_ROOT)
    run.add_argument("--schema", type=Path)
    run.add_argument("--out", type=Path)
    run.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.command == "status":
        return status()
    prompt = (
        args.prompt_file.read_text(encoding="utf-8") if args.prompt_file
        else args.prompt if args.prompt
        else sys.stdin.read()
    )
    result = run_role(args.role, prompt, cwd=args.cwd, schema=args.schema, timeout=args.timeout)
    if args.out:
        args.out.write_text(result.text, encoding="utf-8")
    print(result.text)
    print(f"[{result.role} {result.model} via {result.account}, {result.seconds}s, ok={result.ok}, log={result.log_path}]",
          file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
