import asyncio

import httpx

from app.rag.embeddings import HuggingFaceEmbeddingProvider, OllamaEmbeddingProvider


def test_ollama_embedding_provider_separates_query_and_document_prefixes(monkeypatch) -> None:
    payloads: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, _url, *, json):
            payloads.append(json)
            count = len(json["input"]) if isinstance(json["input"], list) else 1
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2] for _ in range(count)]},
                request=httpx.Request("POST", "http://ollama/api/embed"),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = OllamaEmbeddingProvider(
        host="http://ollama",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    asyncio.run(provider.embed_texts(["alpha", "beta"]))
    asyncio.run(provider.embed_query("question"))

    assert payloads[0]["input"] == ["search_document: alpha", "search_document: beta"]
    assert payloads[1]["input"] == "search_query: question"


def test_ollama_embedding_provider_splits_rejected_batch(monkeypatch) -> None:
    payloads: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json):
            payloads.append(json)
            request = httpx.Request("POST", url)
            if isinstance(json["input"], list) and len(json["input"]) > 1:
                return httpx.Response(400, request=request, json={"error": "batch too large"})
            return httpx.Response(200, request=request, json={"embeddings": [[1.0, 0.0]]})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = OllamaEmbeddingProvider(host="http://ollama", model="test")

    result = asyncio.run(provider.embed_texts(["one", "two", "three"]))

    assert result == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert any(isinstance(payload["input"], list) for payload in payloads)


def test_huggingface_provider_separates_prefixes_and_tasks(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[1.0, 0.0] for _ in texts]

    provider = HuggingFaceEmbeddingProvider(
        model_name="fake",
        query_prefix="query: ",
        document_prefix="passage: ",
        query_task="retrieval.query",
        document_task="retrieval.passage",
    )
    provider._model = FakeModel()

    documents = asyncio.run(provider.embed_texts(["one", "two"]))
    query = asyncio.run(provider.embed_query("question"))

    assert documents == [[1.0, 0.0], [1.0, 0.0]]
    assert query == [1.0, 0.0]
    assert calls[0][0] == ["passage: one", "passage: two"]
    assert calls[0][1]["task"] == "retrieval.passage"
    assert calls[1][0] == ["query: question"]
    assert calls[1][1]["task"] == "retrieval.query"
