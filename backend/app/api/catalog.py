from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.catalog_service import CatalogService

router = APIRouter(prefix="/catalog", tags=["catalog"])


class ScanFolderRequest(BaseModel):
    folder_path: str = Field(min_length=1)
    recursive: bool = False
    mode: str = "shallow"


class CatalogSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    folder_path: str | None = None
    top_k: int = Field(default=20, ge=1, le=50)


def get_catalog_service(settings: Annotated[Settings, Depends(get_settings)]) -> CatalogService:
    return CatalogService(settings.sqlite_db_path)


@router.post("/scan-folder")
async def scan_folder(
    request: ScanFolderRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
) -> dict:
    if request.mode != "shallow":
        raise HTTPException(status_code=400, detail="Only shallow scan mode is implemented")
    try:
        return service.scan_folder(
            folder_path=request.folder_path,
            recursive=request.recursive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search")
async def search_catalog(
    request: CatalogSearchRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
) -> dict:
    return service.search(
        query=request.query,
        folder_path=request.folder_path,
        top_k=request.top_k,
    )


@router.get("/collections")
async def collections(
    service: Annotated[CatalogService, Depends(get_catalog_service)],
) -> list[dict]:
    return service.list_collections()
