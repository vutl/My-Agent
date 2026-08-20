from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.lightrag.bridge import LightRAGBridge
from app.lightrag.client import get_lightrag
from app.lightrag.ingest import ingest_all_documents, ingest_document, prune_stale_documents
from app.lightrag.query import query_lightrag
from app.llm.openai_client import OpenAICompatibleClient

router = APIRouter(prefix="/rag/lightrag", tags=["lightrag"])


class LightRAGInsertRequest(BaseModel):
    document_id: str = Field(min_length=1)


class LightRAGInsertAllRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=5000)
    prune_stale: bool | None = None


class LightRAGQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="mix")
    top_k: int = Field(default=10, ge=1, le=50)
    chunk_top_k: int = Field(default=8, ge=1, le=50)


def _require_lightrag(settings: Settings) -> None:
    if not settings.lightrag_enabled:
        raise HTTPException(status_code=503, detail="LightRAG is disabled")


async def _require_lightrag_gateway(settings: Settings) -> None:
    """Preflight 9router/model availability before mutating the graph queue."""
    model = settings.lightrag_llm_model_chain[0]
    client = OpenAICompatibleClient(
        base_url=settings.lightrag_llm_api_base,
        api_key=settings.lightrag_llm_api_key or "any",
        timeout_seconds=settings.lightrag_llm_timeout_seconds,
    )
    health = await client.health()
    if not health.get("reachable"):
        raise HTTPException(
            status_code=503,
            detail=f"LightRAG gateway unavailable: {health.get('error') or 'unknown_error'}",
        )
    if model not in set(health.get("models") or []):
        raise HTTPException(
            status_code=503,
            detail=f"LightRAG model unavailable: {model}",
        )


@router.post("/insert")
async def insert_document(
    body: LightRAGInsertRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    _require_lightrag(settings)
    await _require_lightrag_gateway(settings)
    try:
        result = await ingest_document(settings.sqlite_db_path, body.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "ok": True,
        "document_id": result.document_id,
        "track_id": result.track_id,
        "source_path": result.source_path,
        "char_count": result.char_count,
        "skipped": result.skipped,
        "reason": result.reason,
    }


@router.post("/insert-all")
async def insert_all_documents(
    body: LightRAGInsertAllRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    _require_lightrag(settings)
    await _require_lightrag_gateway(settings)
    try:
        result = await ingest_all_documents(
            settings.sqlite_db_path,
            limit=body.limit,
            prune_stale=body.prune_stale,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "total": result.total,
        "inserted": result.inserted,
        "skipped": result.skipped,
        "failed": result.failed,
        "deleted_stale": result.deleted_stale,
        "prune_skipped_reason": result.prune_skipped_reason,
        "unready_document_ids": result.unready_document_ids,
        "results": result.results,
    }


@router.post("/prune-stale")
async def prune_stale(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Prune obsolete graph rows without spending LLM quota on re-ingestion."""
    _require_lightrag(settings)
    try:
        result = await prune_stale_documents(settings.sqlite_db_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "deleted_stale": result.deleted,
        "failed": result.failed,
        "prune_skipped_reason": result.skipped_reason,
        "unready_document_ids": result.unready_document_ids,
        "results": result.results,
    }


@router.post("/query")
async def debug_query(
    body: LightRAGQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    _require_lightrag(settings)
    try:
        raw = await query_lightrag(
            body.query,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            enable_rerank=False,
        )
        bridge = LightRAGBridge(settings)
        mapped = bridge._to_retrieval_results(  # noqa: SLF001 — debug endpoint
            raw,
            focus_document_ids=None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "query": body.query,
        "mode": body.mode,
        "mapped_results": mapped,
        "lightrag": raw,
    }


@router.get("/status")
async def lightrag_status(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    _require_lightrag(settings)
    try:
        rag = get_lightrag()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    status_counts: dict[str, int] = {}
    documents: list[dict] = []
    if hasattr(rag.doc_status, "get_all_status_counts"):
        status_counts = await rag.doc_status.get_all_status_counts()
    if hasattr(rag.doc_status, "get_docs_paginated"):
        page_rows, total = await rag.doc_status.get_docs_paginated(page=1, page_size=50)
        documents = [
            {
                "doc_id": doc_id,
                "status": getattr(status, "status", None),
                "file_path": getattr(status, "file_path", None),
                "track_id": getattr(status, "track_id", None),
                "updated_at": getattr(status, "updated_at", None),
            }
            for doc_id, status in page_rows
        ]
        status_counts = {**status_counts, "total": total}

    return {
        "enabled": settings.lightrag_enabled,
        "retrieval_engine": settings.retrieval_engine,
        "llm_model": settings.lightrag_llm_model,
        "llm_api_base": settings.lightrag_llm_api_base,
        "embedding_model": settings.embedding_model,
        "working_dir": str(settings.lightrag_working_dir),
        "status_counts": status_counts,
        "documents": documents,
    }
