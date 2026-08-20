"""Explicitly download a local cross-encoder model for Aya.

Runtime inference is local-files-only and never performs a hidden network
request. Run this script deliberately, then point RERANK_CROSS_ENCODER_PATH at
the output directory and set RERANK_MODE=cross_encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "rerank" / "bge-reranker-base",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        local_dir=output,
        local_dir_use_symlinks=False,
    )
    print(output)


if __name__ == "__main__":
    main()
