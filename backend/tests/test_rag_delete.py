import asyncio
from dataclasses import dataclass

import pytest
from fastapi import HTTPException

from app.api import rag as rag_api
from app.core.config import Settings


class _RagService:
    def __init__(self) -> None:
        self.deleted = []

    def get_document(self, document_id):
        return {"id": document_id}

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        return True


class _LanceStore:
    def __init__(self) -> None:
        self.deleted = []

    async def delete_document(self, document_id):
        self.deleted.append(document_id)


@dataclass
class _Deletion:
    status: str


class _LightRAG:
    def __init__(self, status: str) -> None:
        self.status = status

    async def adelete_by_doc_id(self, _document_id):
        return _Deletion(self.status)


def test_delete_document_removes_derived_index_before_canonical() -> None:
    service = _RagService()
    lance = _LanceStore()

    result = asyncio.run(
        rag_api.delete_document(
            "doc-1",
            service=service,
            settings=Settings(lightrag_enabled=False),
            store=lance,
        )
    )

    assert result["derived_indexes"] == {"lancedb": "success"}
    assert lance.deleted == ["doc-1"]
    assert service.deleted == ["doc-1"]


def test_delete_document_keeps_canonical_when_lightrag_is_busy(monkeypatch) -> None:
    service = _RagService()
    lance = _LanceStore()
    monkeypatch.setattr(rag_api, "get_lightrag", lambda: _LightRAG("not_allowed"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            rag_api.delete_document(
                "doc-1",
                service=service,
                settings=Settings(lightrag_enabled=True),
                store=lance,
            )
        )

    assert exc_info.value.status_code == 409
    assert lance.deleted == []
    assert service.deleted == []
