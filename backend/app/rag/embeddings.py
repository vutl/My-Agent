from dataclasses import dataclass
import asyncio
import hashlib
from typing import Protocol

import httpx


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return vectors."""


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class OllamaEmbeddingProvider:
    host: str
    model: str = "embeddinggemma:300m"
    timeout_seconds: float = 120.0
    query_prefix: str = ""
    document_prefix: str = ""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{self.document_prefix}{text}" for text in texts]
        try:
            return await self._embed_batch_resilient(prepared)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise EmbeddingError(self._http_error_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

        return [await self._embed_legacy(text) for text in prepared]

    async def embed_query(self, text: str) -> list[float]:
        prepared = f"{self.query_prefix}{text}"
        try:
            return (await self._embed_batch([prepared]))[0]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise EmbeddingError(self._http_error_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc
        return await self._embed_legacy(prepared)

    async def _embed_batch_resilient(
        self,
        texts: list[str],
        transient_retries: int = 1,
    ) -> list[list[float]]:
        """Split rejected multi-input batches without changing document order."""
        try:
            return await self._embed_batch(texts)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            if len(texts) == 1:
                if transient_retries > 0 and "EOF" in exc.response.text:
                    return await self._embed_batch_resilient(texts, transient_retries - 1)
                raise
            midpoint = len(texts) // 2
            left = await self._embed_batch_resilient(texts[:midpoint], transient_retries)
            right = await self._embed_batch_resilient(texts[midpoint:], transient_retries)
            return left + right

    @staticmethod
    def _http_error_message(exc: httpx.HTTPStatusError) -> str:
        detail = exc.response.text.strip()
        suffix = f": {detail[:500]}" if detail else ""
        return f"Ollama embedding failed: {exc}{suffix}"

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
            "truncate": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.host.rstrip('/')}/api/embed", json=payload)
            response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError("Ollama embed response did not include the expected embeddings list")
        return [_normalize_embedding(embedding) for embedding in embeddings]

    async def _embed_legacy(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.host.rstrip('/')}/api/embeddings",
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingError("Ollama embedding response did not include an embedding list")
        return _normalize_embedding(embedding)


class HuggingFaceEmbeddingProvider:
    """Local Sentence Transformers provider for explicit staging/evaluation.

    Model loading and encoding are moved off the event loop. Production keeps
    using the configured Ollama provider unless it is changed explicitly.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        query_prefix: str = "",
        document_prefix: str = "",
        query_task: str | None = None,
        document_task: str | None = None,
        trust_remote_code: bool = False,
        truncate_dim: int | None = None,
        revision: str | None = None,
        code_revision: str | None = None,
        native_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_task = query_task
        self.document_task = document_task
        self.trust_remote_code = trust_remote_code
        self.truncate_dim = truncate_dim
        self.revision = revision
        self.code_revision = code_revision
        self.native_model = native_model
        self._model = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{self.document_prefix}{text}" for text in texts]
        return await asyncio.to_thread(self._encode, prepared, self.document_task)

    async def embed_query(self, text: str) -> list[float]:
        prepared = f"{self.query_prefix}{text}"
        vectors = await asyncio.to_thread(self._encode, [prepared], self.query_task)
        return vectors[0]

    def _load_model(self):
        if self._model is None:
            try:
                if self.native_model:
                    from transformers import AutoModel
                else:
                    from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "transformers/sentence-transformers is required for Hugging Face embeddings"
                ) from exc
            try:
                if self.native_model:
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                        revision=self.revision,
                        code_revision=self.code_revision,
                    )
                    self._model.to(self.device)
                    self._model.eval()
                else:
                    model_kwargs = (
                        {"code_revision": self.code_revision}
                        if self.code_revision
                        else None
                    )
                    config_kwargs = (
                        {"code_revision": self.code_revision}
                        if self.code_revision
                        else None
                    )
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                        truncate_dim=self.truncate_dim,
                        revision=self.revision,
                        model_kwargs=model_kwargs,
                        config_kwargs=config_kwargs,
                    )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load Hugging Face embedding model {self.model_name}: {exc}"
                ) from exc
        return self._model

    def _encode(self, texts: list[str], task: str | None) -> list[list[float]]:
        model = self._load_model()
        kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.native_model:
            kwargs["device"] = self.device
        if self.truncate_dim is not None:
            kwargs["truncate_dim"] = self.truncate_dim
        if task:
            kwargs["task"] = task
        try:
            embeddings = model.encode(texts, **kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"Hugging Face embedding failed for {self.model_name}: {exc}"
            ) from exc
        return [[float(value) for value in row] for row in embeddings]


@dataclass(frozen=True)
class HashEmbeddingProvider:
    """Deterministic small embedding provider for local tests.

    This is not used for production retrieval quality. It lets tests exercise
    vector-store behavior without depending on a running model server.
    """

    dimensions: int = 32

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest) % self.dimensions
            vector[bucket] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _normalize_embedding(embedding: list) -> list[float]:
    return [float(value) for value in embedding]
