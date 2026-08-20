from pathlib import Path

from app.catalog.document_card import build_document_card
from app.rag.context import compose_retrieval_context
from app.services.rag_service import _lexical_rerank_boost


def test_document_card_tag_matching_avoids_short_substring_false_positives() -> None:
    card = build_document_card(
        Path("paper.md"),
        "Average arousal is reported in this emotion paper, with no retrieval system discussed.",
    )

    assert "rag" not in card.topic_tags
    assert "valence_arousal" in card.topic_tags


def test_context_composer_deduplicates_and_caps_chunks_per_document() -> None:
    results = [
        {
            "chunk_id": "a",
            "document_id": "doc-1",
            "filename": "paper.md",
            "source_path": "/tmp/paper.md",
            "content": "first relevant chunk",
            "retrieval_channels": ["lancedb"],
        },
        {
            "chunk_id": "b",
            "document_id": "doc-1",
            "filename": "paper.md",
            "source_path": "/tmp/paper.md",
            "content": "second relevant chunk",
            "retrieval_channels": ["sqlite_fts5"],
        },
        {
            "chunk_id": "c",
            "document_id": "doc-1",
            "filename": "paper.md",
            "source_path": "/tmp/paper.md",
            "content": "third chunk should be skipped",
            "retrieval_channels": ["sqlite_fts5"],
        },
    ]

    composed = compose_retrieval_context(results, max_chunks_per_document=2)

    assert composed.stats["source_count"] == 2
    assert composed.sources[0]["source_id"] == "SOURCE 1"
    assert "third chunk should be skipped" not in composed.context_text


def test_context_composer_preserves_structured_table_rows_and_metric_columns() -> None:
    rows = "\n".join(
        f"| Baseline {index} | {70 + index:.1f} | {60 + index:.1f} | {0.40 + index / 100:.2f} |"
        for index in range(30)
    )
    table = (
        "table_type: comparison\n"
        "caption: Table 2: Benchmark results\n"
        "content:\n"
        "| Model | Acc | F1 | CCC |\n"
        "|---|---:|---:|---:|\n"
        f"{rows}"
    )
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "table:2",
                "document_id": "doc-1",
                "artifact_type": "table",
                "table_id": "2",
                "filename": "paper.pdf",
                "content": table,
            }
        ],
        query="benchmark Acc F1 CCC",
        max_chars_per_source=200,
        max_table_chars=700,
    )

    assert "| Model | Acc | F1 | CCC |" in composed.context_text
    assert "\n|---|---:|---:|---:|\n" in composed.context_text
    assert composed.stats["table_source_count"] == 1


def test_long_result_table_preserves_relevant_and_proposed_tail_rows() -> None:
    baseline_rows = "\n".join(
        f"| Baseline-{index} | {0.30 + index / 100:.2f} | {0.20 + index / 100:.2f} |"
        for index in range(35)
    )
    table = (
        "| Model | miF1 | maF1 |\n"
        "|---|---:|---:|\n"
        f"{baseline_rows}\n"
        "| Proposed Methods | | |\n"
        "| Feature-Mamba (Ours) | 0.50 | 0.28 |\n"
        "| Fusion-Gate (Ours) | 0.51 | 0.31 |"
    )

    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "long-table",
                "document_id": "doc-mamba",
                "artifact_type": "table",
                "table_id": "table-2",
                "filename": "FROM_SINGLE_TO_MULTI_LABEL_SER_DATASET_AND_MAMBA_BASED_MODEL.pdf",
                "content": table,
            }
        ],
        query=(
            "Compare results from "
            "FROM_SINGLE_TO_MULTI_LABEL_SER_DATASET_AND_MAMBA_BASED_MODEL"
        ),
        max_chars=900,
        max_table_chars=700,
        min_tables=1,
    )

    projected = composed.sources[0]["content"]
    assert "| Model | miF1 | maF1 |" in projected
    assert "| Feature-Mamba (Ours) | 0.50 | 0.28 |" in projected
    assert "| Fusion-Gate (Ours) | 0.51 | 0.31 |" in projected
    assert "| Baseline-34 |" not in projected


def test_context_composer_reserves_a_table_slot_for_result_intent() -> None:
    results = [
        {
            "chunk_id": f"text-{index}",
            "document_id": "doc-1",
            "filename": "paper.pdf",
            "content": ("architecture explanation " * 80) + str(index),
        }
        for index in range(4)
    ]
    results.append(
        {
            "chunk_id": "table:accuracy-only",
            "document_id": "doc-1",
            "artifact_type": "table",
            "table_id": "accuracy-only",
            "filename": "paper.pdf",
            "content": (
                "table_type: comparison\ncontent:\n"
                "| Model | Acc |\n|---|---:|\n| Proposed | 75.0 |"
            ),
            "score": 0.99,
        }
    )
    results.append(
        {
            "chunk_id": "table:benchmark",
            "document_id": "doc-1",
            "artifact_type": "table",
            "table_id": "benchmark",
            "filename": "paper.pdf",
            "content": (
                "table_type: comparison\n"
                "caption: Benchmark\n"
                "content:\n"
                "| Model | Acc | F1 | CCC |\n"
                "|---|---:|---:|---:|\n"
                "| Proposed | 75.0 | 74.0 | 0.70 |"
            ),
            "score": 0.20,
        }
    )

    composed = compose_retrieval_context(
        results,
        query="benchmark Acc F1 CCC",
        max_sources=2,
        max_chars=1_600,
        max_chars_per_source=900,
        min_tables=1,
    )

    assert composed.sources[0]["table_id"] == "benchmark"
    assert composed.stats["table_source_count"] == 1
    assert "| Model | Acc | F1 | CCC |" in composed.context_text


def test_oversized_source_does_not_hide_later_compact_evidence() -> None:
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "first",
                "document_id": "doc-0",
                "filename": "first.pdf",
                "content": "first evidence",
            },
            {
                "chunk_id": "huge",
                "document_id": "doc-1",
                "filename": "huge.pdf",
                "content": "x" * 2_000,
            },
            {
                "chunk_id": "compact",
                "document_id": "doc-2",
                "filename": "compact.pdf",
                "content": "compact relevant evidence",
            },
        ],
        max_chars=650,
        max_chars_per_source=1_500,
    )

    assert "compact relevant evidence" in composed.context_text


def test_long_text_projection_preserves_late_query_evidence_not_only_prefix() -> None:
    content = (
        ("Background setup and preprocessing details. " * 70)
        + "BetaNet benchmark results report CCC 0.66 on Dataset-X and CCC 0.56 on Dataset-Y. "
        + ("Additional discussion and limitations. " * 30)
    )
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "late-hit",
                "document_id": "doc-beta",
                "filename": "beta.pdf",
                "content": content,
            }
        ],
        query="Compare BetaNet benchmark results and CCC performance.",
        max_chars_per_source=700,
        max_chars=1_200,
    )

    assert "BetaNet benchmark results report CCC 0.66" in composed.context_text
    assert len(composed.sources[0]["content"]) <= 710


def test_parent_expansion_cannot_evict_late_evidence_from_retrieved_child() -> None:
    child = (
        ("Ablation setup and implementation detail. " * 30)
        + "CompactNet reports CCC 0.66 on Corpus-A and CCC 0.56 on Corpus-B. "
        + "TinyNet reaches CCC 0.37 in the same comparison."
    )
    parent = (
        "A dense figure OCR repeats CompactNet TinyNet benchmark comparison "
        + "0.1 0.2 0.3 0.4 accuracy F1 CCC " * 45
    )
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "child-hit",
                "document_id": "doc-compact",
                "filename": "compact.pdf",
                "content": "unused raw child",
                "expanded_content": (
                    "[retrieved chunk / child]\n"
                    + child
                    + "\n\n[parent section context]\n"
                    + parent
                ),
            }
        ],
        query="Compare CompactNet and TinyNet benchmark CCC results.",
        max_chars_per_source=1_050,
        max_chars=1_500,
    )

    projected = composed.sources[0]["content"]
    assert "CompactNet reports CCC 0.66" in projected
    assert "TinyNet reaches CCC 0.37" in projected
    assert len(projected) <= 1_060


def test_required_documents_each_keep_a_context_source_under_global_ranking() -> None:
    results = [
        {
            "chunk_id": f"alpha-{index}",
            "document_id": "doc-alpha",
            "filename": "alpha.pdf",
            "content": ("Alpha benchmark detail. " * 90) + f"rank {index}",
        }
        for index in range(4)
    ] + [
        {
            "chunk_id": "beta-hit",
            "document_id": "doc-beta",
            "filename": "beta.pdf",
            "content": "Beta architecture uses cross-attention and gated fusion.",
        }
    ]

    composed = compose_retrieval_context(
        results,
        query="Compare Alpha and Beta architecture.",
        max_sources=3,
        max_chars=1_800,
        max_chars_per_source=900,
        required_document_ids=["doc-alpha", "doc-beta"],
    )

    assert {source["document_id"] for source in composed.sources} == {
        "doc-alpha",
        "doc-beta",
    }


def test_result_comparison_reserves_best_table_from_each_required_document() -> None:
    results = [
        {
            "chunk_id": "alpha-text",
            "document_id": "doc-alpha",
            "filename": "alpha.pdf",
            "content": "Alpha architecture overview.",
        },
        {
            "chunk_id": "alpha-table",
            "table_id": "alpha-table",
            "artifact_type": "table",
            "document_id": "doc-alpha",
            "filename": "alpha.pdf",
            "caption": "Benchmark results",
            "content": "| Model | F1 |\n|---|---:|\n| Alpha | 0.80 |",
        },
        {
            "chunk_id": "beta-text",
            "document_id": "doc-beta",
            "filename": "beta.pdf",
            "content": "Beta architecture overview.",
        },
        {
            "chunk_id": "beta-table",
            "table_id": "beta-table",
            "artifact_type": "table",
            "document_id": "doc-beta",
            "filename": "beta.pdf",
            "caption": "Experimental results",
            "content": "| Model | F1 |\n|---|---:|\n| Beta | 0.82 |",
        },
    ]

    composed = compose_retrieval_context(
        results,
        query="Compare benchmark F1 results.",
        max_sources=2,
        max_chars=2_400,
        min_tables=1,
        required_document_ids=["doc-alpha", "doc-beta"],
    )

    assert [source.get("table_id") for source in composed.sources] == [
        "alpha-table",
        "beta-table",
    ]


def test_lexical_reranker_boosts_query_matching_dual_channel_result() -> None:
    matching = {
        "filename": "PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB.md",
        "heading_path": ["Storage", "LanceDB"],
        "content": "SQLite FTS5 and LanceDB support local catalog retrieval and direct file read.",
        "retrieval_channels": ["lancedb", "sqlite_fts5"],
        "fts_rank": 1,
    }
    unrelated = {
        "filename": "paper.pdf",
        "heading_path": [],
        "content": "speech emotion recognition benchmark results",
        "retrieval_channels": ["lancedb"],
        "fts_rank": None,
    }

    assert _lexical_rerank_boost("LanceDB storage catalog direct file read", matching) > _lexical_rerank_boost(
        "LanceDB storage catalog direct file read",
        unrelated,
    )
