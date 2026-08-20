from app.rag.retriever import build_fts_query
from app.rag.vietnamese_text import build_fts_query as vn_build_fts_query
from app.rag.vietnamese_text import tokenize_for_fts


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
