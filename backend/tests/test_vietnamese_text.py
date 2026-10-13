from app.rag.retriever import build_fts_queries, build_fts_query
from app.rag.vietnamese_text import build_fts_query as vn_build_fts_query
from app.rag.vietnamese_text import tokenize_for_fts
from app.rag.paper_facets import retrieval_query_expansion


def test_tokenize_for_fts_handles_vietnamese_words() -> None:
    tokens = tokenize_for_fts("Giải thích kiến trúc mô hình MSF-SER trong paper")
    assert any("giải" in token for token in tokens)
    assert any("msf" in token for token in tokens)


def test_build_fts_query_uses_or_join() -> None:
    query = vn_build_fts_query("speech emotion fusion")
    assert " OR " in query
    assert "speech" in query


def test_retriever_build_fts_query_delegates_to_vietnamese_module() -> None:
    assert build_fts_query("tài liệu local") == vn_build_fts_query("tài liệu local")


def test_build_fts_queries_keeps_late_filename_and_tail_terms() -> None:
    variants = build_fts_queries(
        "Compare PROJECT_PLAN_mac_ai_agent.md and Robust_Audio-Visual_Fusion_for_Emotion_Recognition.pdf purpose"
    )
    assert variants
    # The first view remains backward-compatible; at least one bounded view
    # must retain identifiers that occur after the old ten-token window.
    joined = " ".join(variants)
    assert "project_plan" in joined or "robust" in joined
    assert len(variants) <= 3


def test_retrieval_query_expansion_adds_structural_anchors_without_replacing_raw_terms() -> None:
    expanded, facets = retrieval_query_expansion(
        "KST.pdf Bài này giải quyết bài toán gì và mô hình được thiết kế ra sao?",
        focused_document_count=1,
    )
    assert expanded.startswith("KST.pdf Bài này giải quyết")
    assert "abstract" in expanded
    assert "architecture" in expanded
    assert "task" in facets
    assert "architecture" in facets


def test_retrieval_query_expansion_keeps_exact_query_for_unscoped_narrow_terms() -> None:
    query = "CCC_A 0.714 Table 2"
    expanded, facets = retrieval_query_expansion(query)
    assert expanded == query
    assert facets == []
