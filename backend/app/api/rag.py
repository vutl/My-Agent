from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.lightrag.client import get_lightrag
from app.lightrag.ingest import _delete_by_doc_id_without_llm
from app.llm.openai_client import get_llm_client
from app.rag.context import compose_retrieval_context
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.figure_enrich_service import FigureEnrichService
from app.services.indexing_service import IndexingService
from app.services.paper_evidence_builder import PaperEvidenceBuilder
from app.services.paper_evidence_service import PaperEvidenceService
from app.services.rag_service import RagService
from app.services.vector_index_service import VectorIndexService

router = APIRouter(prefix="/rag", tags=["rag"])


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class IndexFileRequest(BaseModel):
    source_path: str = Field(min_length=1)
    collection_name: str | None = None
    collection_type: str = "manual"
    scope_type: str = "global"
    scope_id: str | None = None


class IndexSelectedFilesRequest(BaseModel):
    files: list[str] = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    collection_type: str = "manual"
    scope_type: str = "global"
    scope_id: str | None = None


class SearchInCollectionRequest(BaseModel):
    collection_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    collection_id: str | None = None


class VectorIndexDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1)


class VectorIndexAllRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=5000)


class EnrichFiguresRequest(BaseModel):
    document_id: str | None = None
    force: bool = False
    limit: int | None = Field(default=None, ge=1, le=5000)
    revector: bool = True


class BuildEvidenceCardRequest(BaseModel):
    document_id: str = Field(min_length=1)
    force: bool = False


class BuildAllEvidenceCardsRequest(BaseModel):
    document_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=5000)
    force: bool = False
    max_concurrency: int | None = Field(default=None, ge=1, le=4)


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> RagService:
    return RagService(settings.sqlite_db_path, artifact_root=settings.artifacts_path)


def get_indexing_service(settings: Annotated[Settings, Depends(get_settings)]) -> IndexingService:
    return IndexingService(
        db_path=settings.sqlite_db_path,
        artifact_root=settings.artifacts_path,
        ollama_host=settings.ollama_host,
        vision_model=settings.vision_model,
        vision_provider=settings.vision_provider,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        request_timeout_seconds=settings.request_timeout_seconds,
        paper_evidence_card_build_enabled=settings.paper_evidence_card_build_enabled,
        paper_evidence_card_model=settings.paper_evidence_card_model,
        paper_evidence_card_schema_version=settings.paper_evidence_card_schema_version,
        paper_evidence_card_prompt_version=settings.paper_evidence_card_prompt_version,
    )


def get_paper_evidence_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> PaperEvidenceService:
    return PaperEvidenceService(
        settings.sqlite_db_path,
        schema_version=settings.paper_evidence_card_schema_version,
        prompt_version=settings.paper_evidence_card_prompt_version,
    )


def _paper_evidence_builder(
    *,
    settings: Settings,
    service: PaperEvidenceService,
    max_concurrency: int | None = None,
) -> PaperEvidenceBuilder:
    if not settings.paper_evidence_card_build_enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "Paper evidence-card build is disabled. Set "
                "PAPER_EVIDENCE_CARD_BUILD_ENABLED=true only after approving corpus upload."
            ),
        )
    client = get_llm_client(
        provider="openai_compatible",
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return PaperEvidenceBuilder(
        service=service,
        client=client,
        model=settings.paper_evidence_card_model,
        max_concurrency=max_concurrency or settings.paper_evidence_card_max_concurrency,
    )


def get_embedding_provider(settings: Annotated[Settings, Depends(get_settings)]) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


def get_lancedb_store(settings: Annotated[Settings, Depends(get_settings)]) -> LanceDBRetrievalStore:
    try:
        return LanceDBRetrievalStore(settings.lancedb_path)
    except LanceDBUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_vector_index_service(
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[LanceDBRetrievalStore, Depends(get_lancedb_store)],
    embeddings: Annotated[OllamaEmbeddingProvider, Depends(get_embedding_provider)],
) -> VectorIndexService:
    return VectorIndexService(
        db_path=settings.sqlite_db_path,
        retrieval_store=store,
        embeddings=embeddings,
    )


@router.post("/index-file")
async def index_file(
    request: IndexFileRequest,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> dict:
    try:
        document = service.index_file(
            source_path=request.source_path,
            collection_name=request.collection_name,
            collection_type=request.collection_type,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"document": document}


@router.post("/index-selected-files")
async def index_selected_files(
    request: IndexSelectedFilesRequest,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> dict:
    try:
        return service.index_selected_files(
            source_paths=request.files,
            collection_name=request.collection_name,
            collection_type=request.collection_type,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search")
async def search(
    request: RagSearchRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> list[dict]:
    return [chunk.__dict__ for chunk in service.search(request.query, request.top_k)]


@router.post("/search-debug")
async def search_debug(
    request: RagSearchRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> dict:
    return service.search_debug(request.query, request.top_k)


@router.post("/search-in-collection")
async def search_in_collection(
    request: SearchInCollectionRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> dict:
    return service.search_in_collection(
        collection_id=request.collection_id,
        query=request.query,
        top_k=request.top_k,
    )


@router.post("/search-hybrid")
async def search_hybrid(
    request: HybridSearchRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    store: Annotated[LanceDBRetrievalStore, Depends(get_lancedb_store)],
    embeddings: Annotated[OllamaEmbeddingProvider, Depends(get_embedding_provider)],
) -> dict:
    try:
        result = await service.search_hybrid(
            query=request.query,
            top_k=request.top_k,
            collection_id=request.collection_id,
            retrieval_store=store,
            embeddings=embeddings,
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    composed = compose_retrieval_context(
        result["results"],
        query=request.query,
        max_sources=request.top_k,
        max_chunks_per_document=request.top_k,
    )
    result["context"] = {
        "text": composed.context_text,
        "sources": composed.sources,
        "stats": composed.stats,
    }
    return result


@router.post("/retrieve-figures")
@router.post("/search-figures")
async def retrieve_figures(
    request: HybridSearchRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    store: Annotated[LanceDBRetrievalStore, Depends(get_lancedb_store)],
    embeddings: Annotated[OllamaEmbeddingProvider, Depends(get_embedding_provider)],
) -> dict:
    try:
        return await service.retrieve_figures(
            query=request.query,
            top_k=request.top_k,
            collection_id=request.collection_id,
            retrieval_store=store,
            embeddings=embeddings,
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/vector/index-document")
async def vector_index_document(
    request: VectorIndexDocumentRequest,
    service: Annotated[VectorIndexService, Depends(get_vector_index_service)],
) -> dict:
    try:
        result = await service.index_document(request.document_id)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.post("/vector/index-all")
async def vector_index_all(
    request: VectorIndexAllRequest,
    service: Annotated[VectorIndexService, Depends(get_vector_index_service)],
) -> dict:
    try:
        return await service.index_all_documents(limit=request.limit)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/figures/enrich")
async def enrich_figures(
    request: EnrichFiguresRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    vector_service: Annotated[VectorIndexService, Depends(get_vector_index_service)],
) -> dict:
    if not settings.vision_model:
        raise HTTPException(status_code=503, detail="VISION_MODEL is not configured")

    enricher = FigureEnrichService(
        db_path=settings.sqlite_db_path,
        ollama_host=settings.ollama_host,
        vision_model=settings.vision_model,
        vision_provider=settings.vision_provider,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        request_timeout_seconds=settings.request_timeout_seconds,
        artifact_root=settings.artifacts_path,
    )
    result = enricher.enrich_document(
        request.document_id,
        force=request.force,
        limit=request.limit,
    )

    vector_result = None
    if request.revector and request.document_id and result.get("enriched", 0) > 0:
        try:
            vector_result = await vector_service.index_document(request.document_id)
        except EmbeddingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    elif request.revector and not request.document_id and result.get("enriched", 0) > 0:
        try:
            vector_result = await vector_service.index_all_documents(limit=request.limit)
        except EmbeddingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {**result, "vector_index": vector_result}


@router.post("/evidence-cards/build")
async def build_evidence_card(
    request: BuildEvidenceCardRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[PaperEvidenceService, Depends(get_paper_evidence_service)],
) -> dict:
    builder = _paper_evidence_builder(settings=settings, service=service)
    try:
        return await builder.build_document(request.document_id, force=request.force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evidence-cards/build-all")
async def build_all_evidence_cards(
    request: BuildAllEvidenceCardsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[PaperEvidenceService, Depends(get_paper_evidence_service)],
) -> dict:
    builder = _paper_evidence_builder(
        settings=settings,
        service=service,
        max_concurrency=request.max_concurrency,
    )
    try:
        return await builder.build_all(
            document_ids=request.document_ids,
            limit=request.limit,
            force=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/evidence-cards/status")
async def evidence_card_status(
    service: Annotated[PaperEvidenceService, Depends(get_paper_evidence_service)],
) -> dict:
    return service.get_status()


@router.get("/documents")
async def documents(
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> list[dict]:
    return service.list_documents()


@router.get("/documents/{document_id}")
async def document(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> dict:
    selected = service.get_document(document_id)
    if selected is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return selected


@router.get("/documents/{document_id}/evidence-card")
async def document_evidence_card(
    document_id: str,
    service: Annotated[PaperEvidenceService, Depends(get_paper_evidence_service)],
) -> dict:
    selected = service.card_for_document(document_id)
    if selected is None:
        raise HTTPException(status_code=404, detail="Evidence card not found")
    return selected


@router.get("/documents/{document_id}/chunks")
async def document_chunks(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> list[dict]:
    if service.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return service.list_document_chunks(document_id)


@router.get("/documents/{document_id}/tables")
async def document_tables(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> list[dict]:
    if service.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return service.list_document_tables(document_id)


@router.get("/documents/{document_id}/figures")
async def document_figures(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> list[dict]:
    if service.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return service.list_document_figures(document_id)


@router.get("/figures/{figure_id}/image")
async def figure_image(
    figure_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    figure = service.get_figure(figure_id)
    if figure is None or not figure.get("image_path"):
        raise HTTPException(status_code=404, detail="Figure image not found")

    image_path = Path(str(figure["image_path"])).expanduser().resolve()
    artifact_root = settings.artifacts_path.expanduser().resolve()
    if not image_path.is_file() or not image_path.is_relative_to(artifact_root):
        raise HTTPException(status_code=404, detail="Figure image not found")
    return FileResponse(image_path)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[LanceDBRetrievalStore, Depends(get_lancedb_store)],
) -> dict:
    if service.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    derived_indexes: dict[str, str] = {}
    if settings.lightrag_enabled:
        try:
            deletion = await _delete_by_doc_id_without_llm(
                get_lightrag(),
                document_id,
            )
            status = str(getattr(deletion, "status", "fail"))
        except Exception as exc:  # noqa: BLE001 — preserve canonical source on sync failure
            raise HTTPException(
                status_code=503,
                detail=f"LightRAG delete failed; canonical document kept: {exc}",
            ) from exc
        if status not in {"success", "not_found"}:
            raise HTTPException(
                status_code=409,
                detail=f"LightRAG delete blocked ({status}); canonical document kept",
            )
        derived_indexes["lightrag"] = status

    try:
        await store.delete_document(document_id)
    except Exception as exc:  # noqa: BLE001 — surface incomplete derived-index cleanup
        raise HTTPException(
            status_code=503,
            detail=f"LanceDB delete failed; canonical document kept: {exc}",
        ) from exc
    derived_indexes["lancedb"] = "success"

    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "ok": True,
        "document_id": document_id,
        "derived_indexes": derived_indexes,
    }
