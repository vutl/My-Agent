from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalFilter:
    document_ids: list[str] | None = None
    file_ids: list[str] | None = None
    collection_ids: list[str] | None = None
    project_id: str | None = None
    file_types: list[str] | None = None
    chunk_types: list[str] | None = None
    folder_path: str | None = None
    date_from: str | None = None
    date_to: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    text: str
    score: float
    source: Literal["document_card", "text_chunk", "table", "figure", "memory"]
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStore(Protocol):
    async def add_document_cards(self, records: list[VectorRecord]) -> None:
        ...

    async def add_text_chunks(self, records: list[VectorRecord]) -> None:
        ...

    async def add_table_chunks(self, records: list[VectorRecord]) -> None:
        ...

    async def add_figure_chunks(self, records: list[VectorRecord]) -> None:
        ...

    async def add_memory_chunks(self, records: list[VectorRecord]) -> None:
        ...

    async def search_document_cards(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...

    async def search_text_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...

    async def search_table_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...

    async def search_figure_chunks(
        self,
        query_embedding: list[float],
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...

    async def delete_document(self, document_id: str) -> None:
        ...
