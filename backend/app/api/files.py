from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.catalog_service import CatalogService
from app.services.indexing_service import IndexFolderResult, IndexingService

router = APIRouter(prefix="/files", tags=["files"])


class IndexFolderRequest(BaseModel):
    folder_path: str = Field(min_length=1)
    recursive: bool = False
    file_types: list[str] = Field(default_factory=lambda: ["txt", "md"])


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
        paper_evidence_card_build_enabled=settings.paper_evidence_card_build_enabled,
        paper_evidence_card_model=settings.paper_evidence_card_model,
        paper_evidence_card_schema_version=settings.paper_evidence_card_schema_version,
        paper_evidence_card_prompt_version=settings.paper_evidence_card_prompt_version,
    )


def get_catalog_service(settings: Annotated[Settings, Depends(get_settings)]) -> CatalogService:
    return CatalogService(settings.sqlite_db_path)


@router.post("/index-folder")
async def index_folder(
    request: IndexFolderRequest,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> IndexFolderResult:
    try:
        return service.index_folder(
            folder_path=request.folder_path,
            recursive=request.recursive,
            file_types=request.file_types,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
