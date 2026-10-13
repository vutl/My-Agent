from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.rag.parsers import SUPPORTED_TEXT_TYPES
from app.services.catalog_service import CatalogService
from app.services.document_processing_service import DocumentProcessingService
from app.services.indexing_service import IndexFolderResult, IndexingService

router = APIRouter(prefix="/files", tags=["files"])


class IndexFolderRequest(BaseModel):
    folder_path: str = Field(min_length=1)
    recursive: bool = False
    file_types: list[str] = Field(default_factory=lambda: ["txt", "md"])


class ProcessFolderRequest(IndexFolderRequest):
    # The durable lifecycle is the arbitrary-format folder entry point. Keep
    # the deprecated /index-folder request unchanged, but let new callers
    # process every parser-supported type by default.
    file_types: list[str] = Field(default_factory=lambda: sorted(SUPPORTED_TEXT_TYPES))
    metadata: dict = Field(default_factory=dict)


class ResolveFileRequest(BaseModel):
    filename_or_query: str = Field(min_length=1)
    base_folder: str | None = None
    allow_fuzzy: bool = True
    max_candidates: int = Field(default=10, ge=1, le=50)


class ReadFileRequest(BaseModel):
    source_path: str = Field(min_length=1)
    mode: str = "transient"
    max_tokens: int = Field(default=6000, ge=200, le=20000)


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
        pdf_ocr_page_budget=settings.pdf_ocr_page_budget,
        pdf_page_label_ocr_budget=settings.pdf_page_label_ocr_budget,
        paper_evidence_card_build_enabled=settings.paper_evidence_card_build_enabled,
        paper_evidence_card_model=settings.paper_evidence_card_model,
        paper_evidence_card_schema_version=settings.paper_evidence_card_schema_version,
        paper_evidence_card_prompt_version=settings.paper_evidence_card_prompt_version,
    )


def get_catalog_service(settings: Annotated[Settings, Depends(get_settings)]) -> CatalogService:
    return CatalogService(settings.sqlite_db_path)


def get_processing_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentProcessingService:
    return DocumentProcessingService.from_settings(settings)


@router.post("/index-folder", deprecated=True)
async def index_folder(
    request: IndexFolderRequest,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> IndexFolderResult:
    try:
        return service.index_folder(
            folder_path=request.folder_path,
            recursive=request.recursive,
            file_types=request.file_types,
            defer_activation=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/process-folder")
async def process_folder(
    request: ProcessFolderRequest,
    service: Annotated[DocumentProcessingService, Depends(get_processing_service)],
) -> dict:
    """Queue a local-first folder scan; the durable worker performs it in background."""

    try:
        job_id = service.enqueue_folder(
            folder_path=request.folder_path,
            recursive=request.recursive,
            file_types=request.file_types,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "pending"}


@router.get("/processing-jobs")
async def processing_jobs(
    service: Annotated[DocumentProcessingService, Depends(get_processing_service)],
) -> list[dict]:
    return [job.__dict__ for job in service.list_jobs()]


@router.get("/processing-jobs/{job_id}")
async def processing_job(
    job_id: str,
    service: Annotated[DocumentProcessingService, Depends(get_processing_service)],
) -> dict:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="processing_job_not_found")
    return job.__dict__


@router.get("/documents/{document_id}/status")
async def document_processing_status(
    document_id: str,
    service: Annotated[DocumentProcessingService, Depends(get_processing_service)],
) -> dict:
    status = service.document_status(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    return status


@router.get("/indexed-folders")
async def indexed_folders(
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> list[dict]:
    return service.list_folders()


@router.post("/resolve")
async def resolve_file(
    request: ResolveFileRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
) -> dict:
    return service.resolve_file(
        filename_or_query=request.filename_or_query,
        base_folder=request.base_folder,
        allow_fuzzy=request.allow_fuzzy,
        max_candidates=request.max_candidates,
    )


@router.post("/read")
async def read_file(
    request: ReadFileRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
) -> dict:
    if request.mode != "transient":
        raise HTTPException(status_code=400, detail="Only transient read mode is implemented")
    try:
        return service.read_file_direct(
            source_path=request.source_path,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
