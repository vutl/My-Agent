"""Configurable local reranking for hybrid retrieval candidates."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from app.rag.embeddings import EmbeddingProvider


class CrossEncoderUnavailable(RuntimeError):
    """Raised when explicitly enabled local cross-encoder assets are unavailable."""


async def rerank_candidates(
    *,
    query: str,
    results: list[dict],
    embeddings: EmbeddingProvider,
    top_k: int,
    mode: str = "embedding",
    cross_encoder_model_path: str | Path | None = None,
    max_candidates: int = 20,
) -> list[dict]:
    normalized_mode = (mode or "embedding").strip().lower()
    if normalized_mode == "cross_encoder":
        return await rerank_with_cross_encoder(
            query=query,
            results=results,
            model_path=cross_encoder_model_path,
            top_k=top_k,
            max_candidates=max_candidates,
        )
    if normalized_mode != "embedding":
        raise ValueError(f"Unsupported rerank mode: {mode}")
    return await rerank_with_embeddings(
        query=query,
        results=results,
        embeddings=embeddings,
        top_k=top_k,
        max_candidates=max_candidates,
    )


async def rerank_with_embeddings(
    *,
    query: str,
    results: list[dict],
    embeddings: EmbeddingProvider,
    top_k: int,
    max_candidates: int = 12,
) -> list[dict]:
    if not results:
        return []
    if len(results) <= top_k:
        return [_with_rerank_score(item, item.get("score", 0.0)) for item in results]

    candidates = results[:max_candidates]
    query_vector = await embeddings.embed_query(query)
    texts = [_result_text(item) for item in candidates]
    doc_vectors = await embeddings.embed_texts(texts)

    scored: list[tuple[dict, float]] = []
    for item, doc_vector in zip(candidates, doc_vectors, strict=True):
        base = float(item.get("score") or item.get("rrf_score") or 0.0)
        similarity = _cosine(query_vector, doc_vector)
        combined = base + (similarity * 0.35)
        scored.append((item, combined))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    reranked = [
        _with_rerank_score(item, score, backend="embedding_cosine")
        for item, score in scored[:top_k]
    ]
    if len(results) <= max_candidates:
        return reranked

    tail = [
        _with_rerank_score(
            item,
            float(item.get("score") or item.get("rrf_score") or 0.0),
            backend="embedding_cosine",
        )
        for item in results[max_candidates:]
    ]
    merged = reranked + tail
    merged.sort(key=lambda item: float(item.get("rerank_score") or item.get("score") or 0.0), reverse=True)
    return merged[:top_k]


async def rerank_with_cross_encoder(
    *,
    query: str,
    results: list[dict],
    model_path: str | Path | None,
    top_k: int,
    max_candidates: int = 20,
) -> list[dict]:
    if not results:
        return []
    resolved = Path(model_path).expanduser() if model_path else None
    if resolved is None or not resolved.is_dir():
        raise CrossEncoderUnavailable(
            "cross_encoder_model_missing: configure RERANK_CROSS_ENCODER_PATH "
            "to an explicitly downloaded local model directory"
        )

    candidates = results[:max_candidates]
    texts = [_result_text(item) for item in candidates]
    scores = await asyncio.to_thread(
        _score_cross_encoder_pairs,
        str(resolved.resolve()),
        query,
        texts,
    )
    scored = [
        (
            item,
            float(score),
        )
        for item, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    reranked = [
        _with_rerank_score(item, score, backend="cross_encoder")
        for item, score in scored[:top_k]
    ]
    if len(results) <= max_candidates:
        return reranked

    tail = [
        _with_rerank_score(
            item,
            float(item.get("score") or item.get("rrf_score") or 0.0),
            backend="first_stage_tail",
        )
        for item in results[max_candidates:]
    ]
    merged = reranked + tail
    merged.sort(
        key=lambda item: float(item.get("rerank_score") or item.get("score") or 0.0),
        reverse=True,
    )
    return merged[:top_k]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_path: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise CrossEncoderUnavailable(
            "cross_encoder_dependencies_missing: install backend[rerank]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, torch


def _score_cross_encoder_pairs(
    model_path: str,
    query: str,
    texts: list[str],
) -> list[float]:
    tokenizer, model, torch = _load_cross_encoder(model_path)
    encoded = tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    if logits.ndim == 2 and logits.shape[-1] > 1:
        values = torch.softmax(logits, dim=-1)[:, -1]
    else:
        values = torch.sigmoid(logits.reshape(-1))
    return [float(value) for value in values.detach().cpu().tolist()]


def _result_text(item: dict) -> str:
    parts = [
        str(item.get("filename") or ""),
        " ".join(str(part) for part in (item.get("heading_path") or [])),
        str(item.get("caption") or ""),
        str(item.get("content") or item.get("text") or ""),
    ]
    text = "\n".join(part for part in parts if part.strip())
    return text[:1800]


def _with_rerank_score(
    item: dict,
    rerank_score: float,
    *,
    backend: str = "embedding_cosine",
) -> dict:
    merged = dict(item)
    merged["rerank_score"] = round(rerank_score, 6)
    merged["score"] = merged["rerank_score"]
    merged["rerank_backend"] = backend
    return merged


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
