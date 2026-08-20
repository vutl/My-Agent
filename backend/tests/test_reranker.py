import asyncio

import pytest

from app.rag.embeddings import HashEmbeddingProvider
from app.rag.reranker import (
    CrossEncoderUnavailable,
    rerank_candidates,
    rerank_with_embeddings,
)


def test_rerank_with_embeddings_adds_score_and_limits_top_k() -> None:
    results = [
        {"chunk_id": str(index), "content": f"chunk {index}", "score": index * 0.1}
        for index in range(5)
    ]
    reranked = asyncio.run(
        rerank_with_embeddings(
            query="chunk retrieval",
            results=results,
            embeddings=HashEmbeddingProvider(dimensions=64),
            top_k=2,
        )
    )
    assert len(reranked) == 2
    assert all("rerank_score" in item for item in reranked)
    assert all(item["rerank_backend"] == "embedding_cosine" for item in reranked)


def test_cross_encoder_requires_explicit_local_model_directory(tmp_path) -> None:
    with pytest.raises(CrossEncoderUnavailable, match="cross_encoder_model_missing"):
        asyncio.run(
            rerank_candidates(
                query="chunk retrieval",
                results=[{"chunk_id": "1", "content": "retrieval evidence"}],
                embeddings=HashEmbeddingProvider(dimensions=32),
                top_k=1,
                mode="cross_encoder",
                cross_encoder_model_path=tmp_path / "missing",
            )
        )
