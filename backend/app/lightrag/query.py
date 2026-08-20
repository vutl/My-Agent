from __future__ import annotations

from lightrag import QueryParam

from app.lightrag.client import get_lightrag

VALID_MODES = {"local", "global", "hybrid", "naive", "mix"}


def resolve_query_mode(
    *,
    answer_intent: str | None = None,
    retrieval_mode: str | None = None,
) -> str:
    if retrieval_mode and retrieval_mode.lower() in VALID_MODES:
        return retrieval_mode.lower()
    if answer_intent in {"compare", "infer_structure"}:
        return "hybrid"
    if answer_intent in {"direct_answer", "define", "simplify", "example"}:
        return "local"
    return "mix"


async def query_lightrag(
    query: str,
    *,
    mode: str = "mix",
    top_k: int = 10,
    chunk_top_k: int = 8,
    enable_rerank: bool = False,
) -> dict:
    if mode not in VALID_MODES:
        mode = "mix"
    rag = get_lightrag()
    param = QueryParam(
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=enable_rerank,
        only_need_context=True,
    )
    return await rag.aquery_data(query, param)
