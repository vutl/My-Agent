from __future__ import annotations

from pathlib import Path
import re
import sqlite3

import pytest

from app.core.config import get_settings
from app.services.conversation_state import ConversationWorkingState, DocumentThreadState
from app.services.document_scope_service import resolve_document_scope
from app.services.rag_service import RagService


def _live_rag() -> RagService:
    settings = get_settings()
    return RagService(settings.sqlite_db_path, settings.artifacts_path)


def _catalog_rows() -> list[tuple[str, str]]:
    settings = get_settings()
    connection = sqlite3.connect(settings.sqlite_db_path)
    try:
        return connection.execute(
            "SELECT id, filename FROM documents ORDER BY filename"
        ).fetchall()
    finally:
        connection.close()


def _compact_stem(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", Path(filename).stem)


def test_every_catalog_stem_resolves_across_separator_case_and_compact_variants() -> None:
    rag = _live_rag()
    rows = _catalog_rows()
    assert len(rows) >= 20

    failures: list[str] = []
    for document_id, filename in rows:
        stem = Path(filename).stem
        variants = {
            stem.lower(),
            stem.upper(),
            re.sub(r"[-_]+", " ", stem).lower(),
            _compact_stem(filename).lower(),
        }
        for variant in variants:
            resolved = rag.resolve_document_mentions_for_query(query=variant)
            if document_id not in resolved:
                failures.append(f"{filename!r} via {variant!r}: {resolved}")

    assert failures == []


def test_retrieval_source_identity_enrichment_reuses_unique_catalog_aliases() -> None:
    rag = _live_rag()
    visec_ids = rag.resolve_document_mentions_for_query(query="ViSEC paper")
    assert len(visec_ids) == 1

    source = rag.enrich_source_identities(
        [{"document_id": visec_ids[0], "filename": "ICASSP_2024___ViSEC.pdf"}]
    )[0]
    aliases = set(source["metadata"]["catalog_aliases"])
    assert "visec" in aliases


def test_every_catalog_pair_resolves_in_mention_order_without_sticky_overwrite() -> None:
    rag = _live_rag()
    rows = _catalog_rows()
    failures: list[str] = []

    for index, (first_id, first_filename) in enumerate(rows):
        for second_id, second_filename in rows[index + 1 :]:
            first = _compact_stem(first_filename)
            second = _compact_stem(second_filename)
            queries = (
                f"{first} vs {second}",
                f"{first} và {second}",
                f"{first} với {second}",
                f"{first} and {second}",
                f"{first} & {second}",
                f"{first} + {second}",
                f"2 cái {first} {second}",
                f"{first}, {second}",
                f"hai bài {first} và {second}",
                f"paper {first} and {second}",
            )
            for query in queries:
                scope = resolve_document_scope(
                    rag,
                    query=query,
                    collection_id=None,
                    working_state=ConversationWorkingState(
                        active_document_ids=["stale-document"],
                        active_topic="stale",
                    ),
                    previous_messages=[],
                )
                if list(scope.document_ids) != [first_id, second_id]:
                    failures.append(
                        f"{query}: expected={[first_id, second_id]}, "
                        f"actual={scope.document_ids}"
                    )
                if not scope.must_cover_all or not scope.authoritative:
                    failures.append(f"non-authoritative pair: {query}")

    assert failures == []


def test_aliases_fail_closed_for_generic_words_baselines_and_ambiguous_names() -> None:
    rag = _live_rag()

    for query in (
        "data",
        "while models use data",
        "model",
        "speech",
        "emotion",
        "fusion",
        "SER",
        "access",
        "user",
        "aspiration",
        "whisper",
        "Wav2Vec",
        "project là gì?",
        "lập plan giúp tôi",
        "addendum nghĩa là gì?",
        "storage hoạt động thế nào?",
        "catalog là gì?",
        "năm 2024 có gì mới?",
        "page 1127 nói gì?",
        "semi-supervised learning là gì?",
        "frame-level features là gì?",
        "F1-score là gì?",
        "key-sparse attention là gì?",
    ):
        assert rag.resolve_document_mentions_for_query(query=query) == [], query

    ambiguous = rag.resolve_catalog_mentions(query="9router")
    assert ambiguous.document_ids == ()
    assert ambiguous.ambiguous_mentions
    assert len(ambiguous.ambiguous_mentions[0].candidate_ids) >= 2

    expected_alias_files = {
        "Pitch-fusion kết quả": "ICASSP_2024___ViSEC.pdf",
        "FM-MOE kết quả": "MSF-SER.pdf",
        "KS-Transformer kết quả": "KST.pdf",
        "LPMN kết quả": (
            "An Effective Local Prototypical Mapping Network for Speech "
            "Emotion Recognition.pdf"
        ),
    }
    for query, filename in expected_alias_files.items():
        ids = rag.resolve_document_mentions_for_query(query=query)
        assert len(ids) == 1, query
        assert (rag.get_document(ids[0]) or {}).get("filename") == filename, query


def test_natural_alias_pair_and_single_paper_baseline_comparison() -> None:
    rag = _live_rag()
    pair = resolve_document_scope(
        rag,
        query="thế model msf ser versus wav2small",
        collection_id=None,
        working_state=ConversationWorkingState(
            active_document_ids=["stale"], active_topic="WhiSER"
        ),
        previous_messages=[],
    )
    pair_files = {
        (rag.get_document(document_id) or {}).get("filename")
        for document_id in pair.document_ids
    }
    assert pair_files == {"MSF-SER.pdf", "wav2small.pdf"}
    assert pair.must_cover_all is True

    within_paper = resolve_document_scope(
        rag,
        query="Pitch-fusion vs Wav2Vec 2.0",
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )
    assert len(within_paper.document_ids) == 1
    assert (rag.get_document(within_paper.document_ids[0]) or {}).get("filename") == (
        "ICASSP_2024___ViSEC.pdf"
    )
    assert within_paper.must_cover_all is False


def test_plural_referent_is_durable_and_legacy_history_recovers_only_user_pair() -> None:
    rag = _live_rag()
    ids = rag.resolve_document_mentions_for_query(query="WhiSER vs wav2small")
    assert len(ids) == 2

    durable = resolve_document_scope(
        rag,
        query="cho tôi bảng kết quả của chúng",
        collection_id=None,
        working_state=ConversationWorkingState(
            active_document_ids=["some-later-paper"],
            referent_document_ids=ids,
            referent_filenames=["WhiSER.pdf", "wav2small.pdf"],
        ),
        previous_messages=[
            {"role": "user", "content": "hôm nay trời đẹp"},
            {"role": "assistant", "content": "Ừ."},
        ],
    )
    assert list(durable.document_ids) == ids
    assert durable.source == "plural_referent"
    assert durable.must_cover_all is True

    for query in ("bảng của hai cái đó", "tụi nó dùng dataset gì?", "bọn nó khác gì?"):
        colloquial = resolve_document_scope(
            rag,
            query=query,
            collection_id=None,
            working_state=ConversationWorkingState(
                active_document_ids=["some-later-paper"],
                referent_document_ids=ids,
                referent_filenames=["WhiSER.pdf", "wav2small.pdf"],
            ),
            previous_messages=[],
        )
        assert list(colloquial.document_ids) == ids, query
        assert colloquial.source == "plural_referent", query

    legacy = resolve_document_scope(
        rag,
        query="what datasets do both use?",
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[
            {"role": "user", "content": "WhiSER versus wav2small"},
            {
                "role": "assistant",
                "content": "ASPIRE and KST are merely baseline names in this answer.",
            },
            {"role": "user", "content": "nói linh tinh một chút"},
        ],
    )
    assert list(legacy.document_ids) == ids
    assert legacy.source == "plural_referent"


def test_explicit_two_papers_resolves_active_plus_nearest_suspended_thread() -> None:
    rag = _live_rag()
    msf_id = rag.resolve_document_mentions_for_query(query="MSF-SER")[0]
    mamba_id = rag.resolve_document_mentions_for_query(query="Mamba fusion")[0]
    scope = resolve_document_scope(
        rag,
        query="So sánh kết quả hai bài đó.",
        collection_id=None,
        working_state=ConversationWorkingState(
            active_document_ids=[mamba_id],
            recent_document_threads=(
                DocumentThreadState(document_ids=(msf_id,), topic="MSF-SER"),
            ),
        ),
        previous_messages=[],
    )

    assert list(scope.document_ids) == [msf_id, mamba_id]
    assert scope.source == "plural_referent"
    assert scope.must_cover_all is True


def test_first_person_chung_ta_does_not_bind_document_referents() -> None:
    rag = _live_rag()
    ids = rag.resolve_document_mentions_for_query(query="WhiSER vs wav2small")
    state = ConversationWorkingState(
        active_document_ids=[],
        referent_document_ids=ids,
        referent_filenames=["WhiSER.pdf", "wav2small.pdf"],
    )
    for query in ("chúng ta làm gì tiếp?", "chúng tôi nên chọn gì?"):
        scope = resolve_document_scope(
            rag,
            query=query,
            collection_id=None,
            working_state=state,
            previous_messages=[],
        )
        assert scope.source == "none"
        assert scope.document_ids == ()


def test_correction_semantics_override_loose_comma_list_scope() -> None:
    rag = _live_rag()
    scope = resolve_document_scope(
        rag,
        query="đây là bảng ASPIRE rồi, ý tôi là MSF-SER",
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert len(scope.document_ids) == 1
    assert (rag.get_document(scope.document_ids[0]) or {}).get("filename") == "MSF-SER.pdf"
    assert scope.must_cover_all is False


@pytest.mark.parametrize(
    ("query", "expected_filename"),
    [
        ("ASPIRE chỉ là ví dụ; cho tôi KST", "KST.pdf"),
        ("đừng lấy ASPIRE; lấy KST", "KST.pdf"),
        ("ý tôi là KST, không phải ASPIRE", "KST.pdf"),
        ("I mean KST, not ASPIRE", "KST.pdf"),
        ("ASPIRE không đúng, KST mới đúng", "KST.pdf"),
        ("không lấy ASPIRE, cho KST", "KST.pdf"),
        ("không phải ASPIRE, KST cũng không, ASPIRE cơ", "ASPIRE.pdf"),
        ("paper ASPIRE rồi, ý tôi là KST", "KST.pdf"),
        ("use KST instead of ASPIRE", "KST.pdf"),
        ("use KST rather than ASPIRE", "KST.pdf"),
        ("KST chứ không phải ASPIRE", "KST.pdf"),
        ("ASPIRE is just an example; use KST", "KST.pdf"),
        ("don't use ASPIRE; use KST", "KST.pdf"),
        ("ASPIRE is wrong; KST is correct", "KST.pdf"),
        ("KST, rather than ASPIRE", "KST.pdf"),
        ("KST (not ASPIRE)", "KST.pdf"),
        ("KST; not ASPIRE", "KST.pdf"),
        ("ASPIRE? No, KST.", "KST.pdf"),
        ("ASPIRE, sorry, KST", "KST.pdf"),
        ("ASPIRE, correction: KST", "KST.pdf"),
        ("ASPIRE, sửa lại là KST", "KST.pdf"),
        ("replace ASPIRE with KST", "KST.pdf"),
        ("use KST over ASPIRE", "KST.pdf"),
    ],
)
def test_correction_grammar_selects_last_non_negated_catalog_identity(
    query: str,
    expected_filename: str,
) -> None:
    rag = _live_rag()
    scope = resolve_document_scope(
        rag,
        query=query,
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert len(scope.document_ids) == 1
    assert (rag.get_document(scope.document_ids[0]) or {}).get("filename") == expected_filename
    assert scope.must_cover_all is False


@pytest.mark.parametrize(
    "query",
    [
        "not only ASPIRE but also KST",
        "không chỉ ASPIRE mà còn KST",
        "khong chi ASPIRE ma con KST",
    ],
)
def test_additive_contrast_remains_a_joint_scope(query: str) -> None:
    rag = _live_rag()
    expected = rag.resolve_document_mentions_for_query(query="ASPIRE và KST")
    scope = resolve_document_scope(
        rag,
        query=query,
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert list(scope.document_ids) == expected
    assert scope.must_cover_all is True


@pytest.mark.parametrize(
    "query",
    [
        "đối chiếu ASPIRE và KST",
        "ASPIRE khác với KST thế nào?",
        "ASPIRE giống với KST ở đâu?",
        "phân biệt ASPIRE với KST",
        "differences between ASPIRE and KST",
        "similarities between ASPIRE and KST",
        "ASPIRE against KST",
        "contrast ASPIRE and KST",
        "ASPIRE hay KST tốt hơn?",
    ],
)
def test_comparison_vocabulary_shares_one_joint_scope_policy(query: str) -> None:
    rag = _live_rag()
    expected = rag.resolve_document_mentions_for_query(query="ASPIRE và KST")
    scope = resolve_document_scope(
        rag,
        query=query,
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert list(scope.document_ids) == expected
    assert scope.must_cover_all is True


def test_joint_scope_fails_closed_when_one_operand_is_ambiguous() -> None:
    rag = _live_rag()
    for query in (
        "9router vs ASPIRE",
        "ASPIRE với 9router",
        "bảng của 9router và ASPIRE",
    ):
        scope = resolve_document_scope(
            rag,
            query=query,
            collection_id=None,
            working_state=ConversationWorkingState(active_document_ids=[]),
            previous_messages=[],
        )
        assert scope.document_ids == (), query
        assert scope.source == "ambiguous_current_turn", query
        assert scope.ambiguous_mentions, query


@pytest.mark.parametrize(
    "query",
    [
        "not 9router; use ASPIRE",
        "ASPIRE, not 9router",
        "9router? No, ASPIRE",
        "use ASPIRE instead of 9router",
        "replace 9router with ASPIRE",
    ],
)
def test_rejected_ambiguous_alias_does_not_poison_correction_target(query: str) -> None:
    rag = _live_rag()
    scope = resolve_document_scope(
        rag,
        query=query,
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert len(scope.document_ids) == 1
    assert (rag.get_document(scope.document_ids[0]) or {}).get("filename") == "ASPIRE.pdf"
    assert scope.ambiguous_mentions == ()


def test_joint_scope_fails_closed_when_collection_excludes_one_operand() -> None:
    rag = _live_rag()
    settings = get_settings()
    connection = sqlite3.connect(settings.sqlite_db_path)
    try:
        row = connection.execute(
            """
            SELECT aspire.collection_id
            FROM collection_documents aspire
            JOIN documents da ON da.id = aspire.document_id
            WHERE lower(da.filename) = 'aspire.pdf'
              AND NOT EXISTS (
                  SELECT 1
                  FROM collection_documents excluded
                  JOIN documents de ON de.id = excluded.document_id
                  WHERE excluded.collection_id = aspire.collection_id
                    AND lower(de.filename) = 'demo.md'
              )
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    assert row is not None

    scope = resolve_document_scope(
        rag,
        query="ASPIRE vs demo.md",
        collection_id=str(row[0]),
        working_state=ConversationWorkingState(active_document_ids=[]),
        previous_messages=[],
    )

    assert scope.document_ids == ()
    assert scope.source == "collection_excluded"
    assert scope.collection_removed_ids


def test_active_artifact_reference_treats_catalog_name_as_row_until_explicit_switch() -> None:
    rag = _live_rag()
    aspire_id = rag.resolve_document_mentions_for_query(query="ASPIRE")[0]
    state = ConversationWorkingState(
        active_document_ids=[aspire_id],
        active_topic="ASPIRE",
        active_filenames=["ASPIRE.pdf"],
    )
    for query in (
        "Dòng MSF-SER trong bảng này nghĩa là gì?",
        "MSF-SER baseline ở Table 2 này đạt bao nhiêu?",
        "Trong bảng vừa rồi, MSF-SER dùng modality gì?",
    ):
        scope = resolve_document_scope(
            rag,
            query=query,
            collection_id=None,
            working_state=state,
            previous_messages=[],
        )
        assert list(scope.document_ids) == [aspire_id], query
        assert scope.source == "active_artifact_referent", query

    explicit = resolve_document_scope(
        rag,
        query="mở MSF-SER paper",
        collection_id=None,
        working_state=state,
        previous_messages=[],
    )
    assert (rag.get_document(explicit.document_ids[0]) or {}).get("filename") == "MSF-SER.pdf"


def test_generic_technical_question_is_not_a_filename_token_document_scope() -> None:
    rag = _live_rag()
    for query in (
        "RAG là gì?",
        "LLM là gì?",
        "agent hoạt động thế nào?",
        "memory là gì?",
        "LanceDB là gì?",
    ):
        scope = resolve_document_scope(
            rag,
            query=query,
            collection_id=None,
            working_state=ConversationWorkingState(active_document_ids=[]),
            previous_messages=[],
        )
        assert scope.document_ids == (), query
        assert scope.authoritative is False, query


def test_identifier_led_compound_aliases_keep_optional_separator_modifiers() -> None:
    rag = _live_rag()
    expected = rag.resolve_document_mentions_for_query(query="mamba based fusion")
    assert len(expected) == 1
    for query in (
        "mamba-based fusion",
        "mambabasedfusion",
        "mamba fusion",
        "mambafusion",
    ):
        assert rag.resolve_document_mentions_for_query(query=query) == expected, query


def test_distinctive_contiguous_partial_titles_resolve_without_generic_topic_leak() -> None:
    rag = _live_rag()
    cases = {
        "Robust Audio-Visual Fusion": (
            "Robust_Audio-Visual_Fusion_for_Emotion_Recognition.pdf"
        ),
        "Multimodal Recognition of Valence, Arousal and Dominance": (
            "Multimodal_Recognition_of_Valence_Arousal_and_Domi.pdf"
        ),
    }

    for query, filename in cases.items():
        resolved = rag.resolve_document_mentions_for_query(query=query)
        assert len(resolved) == 1, query
        assert (rag.get_document(resolved[0]) or {}).get("filename") == filename

    # Partial-title support must not promote short topical phrases into paper
    # identities merely because they happen to be unique in today's catalog.
    for query in (
        "multimodal emotion recognition là gì?",
        "speech emotion recognition works how?",
        "audio visual fusion là gì?",
    ):
        assert rag.resolve_document_mentions_for_query(query=query) == [], query


@pytest.mark.parametrize(
    ("query", "active_alias", "explicit_alias", "expected_aliases"),
    [
        (
            "Giờ so sánh paper trước với ASPIRE",
            "ViSEC",
            "ASPIRE",
            ("ViSEC", "ASPIRE"),
        ),
        (
            "So sánh nó với MSF-SER",
            "Mamba fusion",
            "MSF-SER",
            ("Mamba fusion", "MSF-SER"),
        ),
        (
            "Compare that RAG-memory plan with PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB",
            "PROJECT_PLAN_ADDENDUM_RAG_AGENT_MEMORY_VISUAL",
            "PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB",
            (
                "PROJECT_PLAN_ADDENDUM_RAG_AGENT_MEMORY_VISUAL",
                "PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB",
            ),
        ),
    ],
)
def test_joint_scope_composes_singular_context_referent_with_explicit_document(
    query: str,
    active_alias: str,
    explicit_alias: str,
    expected_aliases: tuple[str, str],
) -> None:
    rag = _live_rag()
    active_id = rag.resolve_document_mentions_for_query(query=active_alias)[0]
    expected = [
        rag.resolve_document_mentions_for_query(query=alias)[0]
        for alias in expected_aliases
    ]
    assert rag.resolve_document_mentions_for_query(query=explicit_alias)

    scope = resolve_document_scope(
        rag,
        query=query,
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[active_id]),
        previous_messages=[],
    )

    assert list(scope.document_ids) == expected
    assert scope.must_cover_all is True
    assert scope.authoritative is True


def test_joint_scope_resolves_unique_descriptors_within_recent_document_threads() -> None:
    rag = _live_rag()
    storage_id = rag.resolve_document_mentions_for_query(
        query="PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB"
    )[0]
    mac_id = rag.resolve_document_mentions_for_query(
        query="PROJECT_PLAN_mac_ai_agent"
    )[0]
    demo_id = rag.resolve_document_mentions_for_query(query="demo.md")[0]

    scope = resolve_document_scope(
        rag,
        query="Compare the storage plan and the Mac AI agent plan.",
        collection_id=None,
        working_state=ConversationWorkingState(
            active_document_ids=[mac_id],
            recent_document_threads=(
                DocumentThreadState(document_ids=(storage_id,)),
                DocumentThreadState(document_ids=(demo_id,)),
            ),
        ),
        previous_messages=[],
    )

    assert list(scope.document_ids) == [storage_id, mac_id]
    assert scope.must_cover_all is True


def test_postposed_pronoun_without_joint_connector_does_not_add_stale_document() -> None:
    rag = _live_rag()
    visec_id = rag.resolve_document_mentions_for_query(query="ViSEC")[0]
    aspire_id = rag.resolve_document_mentions_for_query(query="ASPIRE")[0]

    scope = resolve_document_scope(
        rag,
        query="Explain ASPIRE because it is the stronger example.",
        collection_id=None,
        working_state=ConversationWorkingState(active_document_ids=[visec_id]),
        previous_messages=[],
    )

    assert list(scope.document_ids) == [aspire_id]
    assert scope.must_cover_all is False


def test_recent_pair_bounds_ambiguous_alias_and_correction_rejection() -> None:
    rag = _live_rag()
    npm_id = rag.resolve_document_mentions_for_query(query="9router - npm.pdf")[0]
    vietnamese_id = rag.resolve_document_mentions_for_query(
        query="9Router là gì - LLM Gateway cho AI coding tools.pdf"
    )[0]
    state = ConversationWorkingState(
        active_document_ids=[vietnamese_id],
        recent_document_threads=(DocumentThreadState(document_ids=(npm_id,)),),
        referent_document_ids=[npm_id, vietnamese_id],
    )

    pair = resolve_document_scope(
        rag,
        query="Compare those two 9router documents.",
        collection_id=None,
        working_state=state,
        previous_messages=[],
    )
    assert list(pair.document_ids) == [npm_id, vietnamese_id]
    assert pair.must_cover_all is True

    corrected = resolve_document_scope(
        rag,
        query=(
            "Not the npm document; return to the Vietnamese 9Router "
            "document's contribution."
        ),
        collection_id=None,
        working_state=state,
        previous_messages=[],
    )
    assert list(corrected.document_ids) == [vietnamese_id]
    assert corrected.must_cover_all is False
