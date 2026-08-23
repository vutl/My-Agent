import asyncio
from pathlib import Path

from app.rag.fusion import fuse_ranked_results, normalized_query, valid_query_variants
from app.services.rag_service import RagService


def test_query_variants_reject_rewrite_that_drops_explicit_entity() -> None:
    variants = valid_query_variants(
        original_query="compare ASPIRE and ViSEC Table 2",
        standalone_query="compare the two methods",
        required_entities=["ASPIRE", "ViSEC", "Table 2"],
    )
    assert variants == ("compare ASPIRE and ViSEC Table 2",)


def test_query_variants_deduplicate_equivalent_rewrite() -> None:
    assert valid_query_variants(
        original_query="  Bảng 2 của ASPIRE ",
        standalone_query="BẢNG 2 CỦA ASPIRE",
    ) == ("Bảng 2 của ASPIRE",)
    assert normalized_query("  A\nB ") == "a b"


def test_rrf_fusion_retains_components_and_prefers_cross_query_agreement() -> None:
    original = [
        {"chunk_id": "a", "content": "a", "retrieval_channels": ["fts"]},
        {"chunk_id": "shared", "content": "shared", "retrieval_channels": ["fts"]},
    ]
    rewrite = [
        {"chunk_id": "shared", "content": "shared", "retrieval_channels": ["vector"]},
        {"chunk_id": "b", "content": "b", "retrieval_channels": ["vector"]},
    ]
    fused = fuse_ranked_results(
        [("original", item) for item in original]
        + [("rewrite", item) for item in rewrite],
        top_k=3,
        rrf_k=60,
    )

    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["query_origins"] == ["original", "rewrite"]
    assert set(fused[0]["retrieval_channels"]) == {"fts", "vector"}
    assert len(fused[0]["query_rrf_components"]) == 2


def test_search_hybrid_queries_exposes_typed_bundle_after_fusion() -> None:
    class FakeRag(RagService):
        async def search_hybrid(self, *, query, **_kwargs):  # noqa: ANN001
            if "rewrite" in query:
                results = [
                    {
                        "chunk_id": "shared",
                        "document_id": "doc-a",
                        "content": "shared",
                        "retrieval_channels": ["vector"],
                    },
                    {
                        "chunk_id": "rewrite-only",
                        "document_id": "doc-b",
                        "content": "rewrite",
                        "retrieval_channels": ["vector"],
                    },
                ]
            else:
                results = [
                    {
                        "chunk_id": "original-only",
                        "document_id": "doc-c",
                        "content": "original",
                        "retrieval_channels": ["fts"],
                    },
                    {
                        "chunk_id": "shared",
                        "document_id": "doc-a",
                        "content": "shared",
                        "retrieval_channels": ["fts"],
                    },
                ]
            return {
                "query": query,
                "results": results,
                "selected_document_ids": [item["document_id"] for item in results],
                "retrieval_channels": ["fts", "vector"],
                "document_card_results": [],
            }

    service = FakeRag(Path("/tmp/unused.db"))
    result = asyncio.run(
        service.search_hybrid_queries(
            queries=[("original", "original query"), ("rewrite", "rewrite query")],
            top_k=3,
            retrieval_store=None,  # type: ignore[arg-type]
            embeddings=None,  # type: ignore[arg-type]
        )
    )

    assert result["dual_query_rrf"] is True
    assert result["results"][0]["chunk_id"] == "shared"
    assert result["evidence_bundle"]["selected"][0]["kind"] == "text"
    assert result["evidence_bundle"]["selected"][0]["query_origins"] == ["original", "rewrite"]
    assert len(result["evidence_bundle"]["candidates"]) == 3
