"""Public tool configuration status; credentials never leave the backend."""
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.tools.search_providers import build_providers

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/search-providers")
def search_provider_status(settings: Annotated[Settings, Depends(get_settings)]) -> list[dict[str, Any]]:
    return [provider.status() for provider in build_providers(settings).values()]
