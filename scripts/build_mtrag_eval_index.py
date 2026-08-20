#!/usr/bin/env python3
"""Build an atomic, full-corpus MTRAG FTS baseline outside Aya production stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mtrag_eval_lib import DEFAULT_INDEX_PATH, DOMAINS, build_fts_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()
    report = build_fts_index(args.output, domains=args.domains)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
