#!/usr/bin/env python3
"""Re-embed LightRAG vector stores without rerunning graph extraction.

The graph and text-chunk KV files remain authoritative. This wrapper uses
LightRAG's official offline rebuild engine but injects Aya's runtime embedding
adapter, so the rebuild does not require the optional LightRAG API server
dependencies and cannot drift from the app's query prefixes/model settings.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools import rebuild_vdb as rebuild_vdb_module
from lightrag.tools.rebuild_vdb import RebuildTool, check_vdb_consistency

from app.core.config import Settings, get_settings
from app.lightrag.adapters import build_ollama_embedding_func


class AyaRebuildTool(RebuildTool):
    def __init__(
        self,
        settings: Settings,
        working_dir: Path,
        batch_size: int,
        embedding_batch_size: int,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.working_dir = working_dir
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size

    def build_embedding_func(self):
        embedding = build_ollama_embedding_func(self.settings)
        print(
            "- Embedding: binding=ollama "
            f"model={embedding.model_name} dim={embedding.embedding_dim} "
            f"max_tokens={embedding.max_token_size}"
        )
        return embedding

    def build_global_config(self):
        config = super().build_global_config()
        config["working_dir"] = str(self.working_dir)
        config["embedding_batch_num"] = self.embedding_batch_size
        return config


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    working_dir = args.working_dir.expanduser().resolve()
    required = (
        working_dir / "graph_chunk_entity_relation.graphml",
        working_dir / "kv_store_text_chunks.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"LightRAG authoritative sources missing: {missing}")
    if not args.yes:
        raise RuntimeError(
            "Refusing to rebuild without --yes. Stop every backend/LightRAG writer first."
        )

    if args.backup_dir:
        backup_dir = args.backup_dir.expanduser().resolve()
        if backup_dir.exists():
            raise RuntimeError(f"Backup path already exists: {backup_dir}")
        shutil.copytree(working_dir, backup_dir)
        print(f"Backup created: {backup_dir}")

    os.environ["WORKING_DIR"] = str(working_dir)
    # The upstream recovery tool normally accumulates ten input batches before
    # flushing. That can fan out hundreds of concurrent Ollama requests on a
    # local machine. Flush each batch and let NanoVectorDB split it into the
    # explicitly bounded embedding micro-batches below.
    rebuild_vdb_module.FLUSH_EVERY_N_BATCHES = 1
    tool = AyaRebuildTool(
        settings,
        working_dir,
        args.batch_size,
        args.embedding_batch_size,
    )
    initialize_share_data(workers=1)
    try:
        if not await tool.setup_storages():
            raise RuntimeError("LightRAG storage initialization failed")

        await tool.print_source_counts(include_graph=True, include_chunks=True)
        stats = await tool.run_rebuild_entities_relations()
        stats.extend(await tool.run_rebuild_chunks())
        if tool.report_rebuild(stats):
            raise RuntimeError("LightRAG vector rebuild completed with errors")

        report = await check_vdb_consistency(
            tool.graph,
            tool.entities_vdb,
            tool.relationships_vdb,
            batch_size=args.batch_size,
        )
        tool.print_check_report(report)
        if not report["consistent"]:
            raise RuntimeError("LightRAG vector consistency check failed")
    finally:
        for storage in tool.all_storages():
            if storage is not None:
                await storage.finalize()
        finalize_share_data()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild LightRAG VDBs with Aya's configured local embedder."
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=get_settings().lightrag_working_dir,
    )
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
