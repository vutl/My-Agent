#!/usr/bin/env python3
"""Generalisation check for a fix: an independent model writes variants of the
failure (same mechanism, other documents, languages, wordings, conversation
shapes) plus negative controls, against a library the fix was NOT debugged on.

  python tools/harness/variants.py --name pronoun-compare \
      --library-dir data/retrieval_eval/sealed/v1 \
      --failure "In a 2-paper comparison, 'bài kia' was read as the same paper" \
      --mechanism "resolving vague references (nó, bài đấy, the other one) to documents"

Writes <library-dir>/variants/<name>.jsonl (evaluate_aya_acceptance.py schema).
Then run Aya on the backend serving that library and judge it:
  backend/.venv/bin/python scripts/evaluate_aya_acceptance.py --dataset <file> --base-url <port> --output <run>.json
  python tools/harness/judge.py --set-dir <library-dir> --questions variants/<name>.jsonl --answers <answers>.jsonl --out ...
A fix counts as general when the variants pass and the negatives do not regress.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import HARNESS_DIR, run_role  # noqa: E402


def _library(library_dir: Path) -> str:
    manifest = library_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        documents = data.get("documents") or data.get("files") or []
        lines = []
        for item in documents:
            if isinstance(item, dict):
                lines.append(" | ".join(str(item.get(key) or "") for key in ("filename", "title", "format", "language", "description")))
        if lines:
            return "\n".join(f"- {line}" for line in lines)
    return "\n".join(f"- {path.name}" for path in sorted((library_dir / "raw").iterdir()) if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--failure", required=True, help="text, or @path to a file")
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--author", default="variant_author")
    args = parser.parse_args()

    library_dir = args.library_dir.resolve()
    failure = Path(args.failure[1:]).read_text(encoding="utf-8") if args.failure.startswith("@") else args.failure
    prompt = (HARNESS_DIR / "prompts" / "variants.md").read_text(encoding="utf-8").format(
        failure=failure, mechanism=args.mechanism, library=_library(library_dir),
        n=args.n, negatives=args.negatives,
    )
    result = run_role(args.author, prompt, cwd=library_dir, schema=HARNESS_DIR / "schemas" / "variants.json",
                      log_dir=library_dir / "variants" / "logs")
    try:
        cases = json.loads(result.text).get("cases") or []
    except ValueError:
        print(f"author returned no JSON; see {result.log_path}", file=sys.stderr)
        return 1
    out = library_dir / "variants" / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({**case, "conversation": f"{args.name}-{case['conversation']}"},
                                      ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    conversations = {case["conversation"] for case in cases}
    negatives = {case["conversation"] for case in cases if case.get("group") == "negative"}
    print(f"{len(cases)} turns in {len(conversations)} conversations ({len(negatives)} negative) → {out} "
          f"[{result.model} via {result.account}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
