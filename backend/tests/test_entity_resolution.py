import sqlite3
from pathlib import Path

from app.api.agent import _canonicalize_explicit_document_target
from app.core.config import get_settings
from app.services.query_rewrite_service import QueryRewriteResult
from app.services.rag_service import RagService, document_match_tokens


def _rewrite(query: str, *, intent: str = "direct_answer") -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=query,
        standalone_query=query,
        is_followup=True,
        current_topic="ASPIRE",
        required_entities=["ASPIRE"],
        use_last_sources=True,
        answer_intent=intent,
        answer_depth="brief",
        rewrite_used=False,
        diagnostics={"reason": "test"},
    )


def test_document_match_tokens_prefers_mamba_fusion_compound() -> None:
    tokens = document_match_tokens(
        entities=[],
        query="đưa tôi figure architecture của mamba fusion",
    )
    assert "mambafusion" in tokens
    assert "fusion" not in tokens


def test_resolve_mamba_fusion_document_not_robust_av_fusion() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    ids = rag.resolve_document_ids_for_entities(
        entities=[],
        query="đưa tôi figure architecture của mamba fusion",
    )
    assert ids
    conn = sqlite3.connect(settings.sqlite_db_path)
    filename = conn.execute("SELECT filename FROM documents WHERE id=?", (ids[0],)).fetchone()[0]
    assert "MAMBA" in filename.upper()
    assert "Robust_Audio-Visual" not in filename


def test_resolve_compare_kst_and_mamba_fusion() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    ids = rag.resolve_document_ids_for_entities(
        entities=[],
        query="so sánh KST với mamba fusion",
    )
    assert len(ids) >= 2
    conn = sqlite3.connect(settings.sqlite_db_path)
    filenames = {
        conn.execute("SELECT filename FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
        for doc_id in ids[:2]
    }
    joined = " ".join(filenames).upper()
    assert "KST" in joined
    assert "MAMBA" in joined


def test_resolve_msf_ser_acronym_to_its_canonical_pdf() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    ids = rag.resolve_document_ids_for_entities(
        entities=["MSF-SER"],
        query="MSF-SER",
    )

    assert ids
    conn = sqlite3.connect(settings.sqlite_db_path)
    filename = conn.execute("SELECT filename FROM documents WHERE id=?", (ids[0],)).fetchone()[0]
    assert filename == "MSF-SER.pdf"


def test_every_catalog_filename_is_an_explicit_resolvable_target() -> None:
    """Guard the whole current corpus, not a hand-picked paper pair."""

    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    conn = sqlite3.connect(settings.sqlite_db_path)
    rows = conn.execute("SELECT id, filename FROM documents ORDER BY filename").fetchall()

    assert len(rows) >= 20
    failures: list[str] = []
    for document_id, filename in rows:
        canonical, resolved = _canonicalize_explicit_document_target(
            rag,
            rewrite=_rewrite(f"mở file {filename}"),
            collection_id=None,
        )
        if resolved != [document_id]:
            failures.append(f"{filename}: {resolved}")
            continue
        assert canonical.use_last_sources is False
        assert canonical.diagnostics["reason"] == "explicit_catalog_document_target"

    assert failures == []


def test_every_multiword_catalog_stem_resolves_without_file_extension() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    conn = sqlite3.connect(settings.sqlite_db_path)
    rows = conn.execute("SELECT id, filename FROM documents ORDER BY filename").fetchall()

    failures: list[str] = []
    tested = 0
    for document_id, filename in rows:
        stem = Path(filename).stem
        if len(stem.replace("_", " ").replace("-", " ").split()) < 2:
            continue
        tested += 1
        _, resolved = _canonicalize_explicit_document_target(
            rag,
            rewrite=_rewrite(f"đưa bảng kết quả của bài {stem}"),
            collection_id=None,
        )
        if resolved != [document_id]:
            failures.append(f"{filename}: {resolved}")

    assert tested >= 10
    assert failures == []


def test_catalog_aliases_resolve_across_different_naming_styles() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)
    cases = {
        "đưa bảng 2 bài LPMN": "An Effective Local Prototypical Mapping Network",
        "đưa bảng 2 bài pitch fusion": "ICASSP_2024___ViSEC.pdf",
        "đưa bảng 2 bài KS-Transformer": "KST.pdf",
        "đưa bảng kết quả bài mamba fusion": "FROM_SINGLE_TO_MULTI_LABEL_SER",
    }

    for query, expected_filename_part in cases.items():
        _, resolved = _canonicalize_explicit_document_target(
            rag,
            rewrite=_rewrite(query),
            collection_id=None,
        )
        assert len(resolved) == 1, query
        document = rag.get_document(resolved[0]) or {}
        assert expected_filename_part in str(document.get("filename")), query


def test_catalog_resolution_handles_correction_compare_ambiguity_and_baselines() -> None:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, settings.artifacts_path)

    corrected, corrected_ids = _canonicalize_explicit_document_target(
        rag,
        rewrite=_rewrite("đây là bảng bài ASPIRE rồi, bài KST cơ"),
        collection_id=None,
    )
    assert len(corrected_ids) == 1
    assert (rag.get_document(corrected_ids[0]) or {}).get("filename") == "KST.pdf"
    assert corrected.required_entities == ["KST"]

    compared, compared_ids = _canonicalize_explicit_document_target(
        rag,
        rewrite=_rewrite("so sánh bài ASPIRE với bài MSF-SER", intent="compare"),
        collection_id=None,
    )
    assert len(compared_ids) == 2
    assert {
        (rag.get_document(document_id) or {}).get("filename") for document_id in compared_ids
    } == {"ASPIRE.pdf", "MSF-SER.pdf"}
    assert compared.required_entities == ["ASPIRE", "MSF-SER"]

    unchanged, ambiguous_ids = _canonicalize_explicit_document_target(
        rag,
        rewrite=_rewrite("đưa bảng của bài 9router"),
        collection_id=None,
    )
    assert ambiguous_ids == []
    assert unchanged.current_topic == "ASPIRE"

    unchanged, baseline_ids = _canonicalize_explicit_document_target(
        rag,
        rewrite=_rewrite("bảng 2 so sánh Pitch-fusion với Wav2Vec 2.0"),
        collection_id=None,
    )
    assert baseline_ids == []
    assert unchanged.current_topic == "ASPIRE"
