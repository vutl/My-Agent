from pathlib import Path

import pytest

from app.catalog.document_card import build_document_card
from app.rag.context import compose_retrieval_context, _truncate_table, _result_facets, select_ranked_evidence
from app.services.rag_service import (
    _lexical_rerank_boost,
    _merge_hybrid_chunks,
    _promote_structured_evidence,
)
from app.services.query_rewrite_service import has_enumerated_evidence_intent


def test_document_card_tag_matching_avoids_short_substring_false_positives() -> None:
    card = build_document_card(
        Path("paper.md"),
        "Average arousal is reported in this emotion paper, with no retrieval system discussed.",
    )

    assert "rag" not in card.topic_tags
    assert "valence_arousal" in card.topic_tags


def test_artifact_identity_does_not_prove_benchmark_purpose():
    statistics = {
        "document_id": "manual", "table_id": "population", "artifact_type": "table",
        "caption": "Table 7: Corpus statistics for the training split",
        "content": "| Split | Samples | Speakers |\n|---|---|---|\n| Train | 1200 | 30 |",
    }
    assert "benchmark_results" not in _result_facets(statistics)
    assert "dataset_setup" in _result_facets(statistics)
    statistics["caption"] = "Training-set statistics for benchmark variants; main benchmark protocol"
    assert "benchmark_results" not in _result_facets(statistics)
    # A captionless table is still retrievable, but has unknown semantic role.
    assert "benchmark_results" not in _result_facets({"table_id": "unknown"})
    # Strong caption semantics suffice for an image-backed results table.
    assert "benchmark_results" in _result_facets({
        "table_id": "image", "caption": "Table 9: Experimental results",
        "content": "Image-backed table; cells not yet transcribed.",
    })


@pytest.mark.parametrize("query,expected", [
    ("Compare benchmark results", "main"),
    ("So sánh kết quả hai bài", "main"),
    ("Interpret the ablation performance", "ablation"),
    ("Explain Table 2 benchmark results", "ablation"),
    ("Compare CCC results", "ccc-only"),
])
def test_table_purpose_reservation_preserves_explicit_artifact_and_metric(query, expected):
    sources = [
        {"document_id": "report", "table_id": "ablation", "artifact_type": "table", "score": 0.99,
         "caption": "Table 2. Ablation performance: benchmark results of component variants",
         "content": "| Variant | F1 |\n|---|---|\n| No gate | 0.51 |"},
        {"document_id": "report", "table_id": "main", "artifact_type": "table", "score": 0.1,
         "caption": "Table 8. Comparison with prior methods",
         "content": "| Method | F1 |\n|---|---|\n| ORION | 0.74 |"},
        {"document_id": "report", "table_id": "ccc-only", "artifact_type": "table", "score": 0.09,
         "caption": "Table 11. Experimental results for dimensional prediction",
         "content": "| Method | CCC |\n|---|---|\n| ORION | 0.67 |"},
    ]
    for operation in ("selection", "composition"):
        if operation == "selection":
            selected, _ = select_ranked_evidence(
                sources, limit=1, query=query, required_document_ids=["report"],
                required_facets_by_document={"report": ["benchmark_results"]},
            )
        else:
            selected = compose_retrieval_context(
                sources, max_sources=1, query=query, required_document_ids=["report"],
                required_facets_by_document={"report": ["benchmark_results"]}, min_tables=1,
            ).sources
        assert selected[0]["table_id"] == expected, operation


def test_result_selection_does_not_reserve_high_scoring_dataset_table():
    statistics = {
        "document_id": "report", "table_id": "stats", "artifact_type": "table", "score": 0.99,
        "caption": "Table 3: Dataset statistics",
        "content": "| Dataset | Samples |\n|---|---|\n| Training | 500 |",
    }
    performance = {
        "document_id": "report", "table_id": "scores", "artifact_type": "table", "score": 0.1,
        "caption": "Table 8: Comparison with prior systems",
        "content": "| System | WER |\n|---|---|\n| ORION | 4.2 |",
    }
    selected, _ = select_ranked_evidence(
        [statistics, performance], limit=1, query="Compare the benchmark results",
        required_document_ids=["report"],
    )
    assert selected[0]["table_id"] == "scores"
    composed = compose_retrieval_context(
        [statistics, performance], max_sources=1, query="Compare the benchmark results",
        required_document_ids=["report"], min_tables=1,
    )
    assert composed.sources[0]["table_id"] == "scores"


def test_table_metric_tokens_are_not_substrings_or_neighbor_prose():
    unrelated = {
        "table_id": "dimensions", "caption": "Table 4: Input dimensions",
        "content": "| Feature | Dimension |\n|---|---|\n| F10 | 64 |",
        "expanded_content": "Neighbor paragraph reports benchmark results and accuracy.",
    }
    assert "benchmark_results" not in _result_facets(unrelated)
    for metric in ("F1", "CCC_A", "miF1", "WA", "WER", "Pearson correlation"):
        assert "benchmark_results" in _result_facets({
            "table_id": metric, "content": f"| Model | {metric} |\n|---|---|\n| ORION | 0.72 |",
        })


def test_dataset_facet_reserves_own_table_before_generic_high_ranked_prose():
    prose = {"document_id": "report", "chunk_id": "discussion", "score": 0.99,
             "section_title": "Datasets", "content": "Dataset details inform the evaluation."}
    statistics = {"document_id": "report", "table_id": "population", "score": 0.05,
                  "caption": "Corpus statistics", "artifact_type": "table",
                  "content": "| Dataset | Samples |\n|---|---|\n| Corpus-X | 200 |"}
    performance = {"document_id": "report", "table_id": "scores", "score": 0.9,
                   "caption": "Recognition results", "artifact_type": "table",
                   "content": "| Model | WER |\n|---|---|\n| ORION | 4.2 |"}
    for query in ("Which datasets were evaluated?", "Bài này dùng dữ liệu nào?"):
        selected, _ = select_ranked_evidence(
            [prose, performance, statistics], limit=1, query=query,
            required_document_ids=["report"],
        )
        assert selected[0]["table_id"] == "population"
        composed = compose_retrieval_context(
            [prose, performance, statistics], max_sources=1, query=query,
            required_document_ids=["report"],
        )
        assert composed.sources[0]["table_id"] == "population"
    # A score question still reserves the score table rather than treating
    # any table (or the dataset caption) as benchmark evidence.
    selected, _ = select_ranked_evidence(
        [prose, statistics, performance], limit=1, query="What are the benchmark results?",
        required_document_ids=["report"],
    )
    assert selected[0]["table_id"] == "scores"


def test_table_projection_keeps_late_row_with_its_own_header_and_section():
    first = "\n".join(f"| Baseline-{i} | 60 | 61 |" for i in range(25))
    table = (
        "caption: Multi-task benchmark\n"
        "| Model | Acc | F1 |\n|---|---|---|\n"
        "| I. Recognition | | |\n" + first + "\n"
        "| ORION | 70 | 71 |\n"
        "| Model | CCC A | CCC V |\n"
        "| II. Regression | | |\n"
        "| Baseline-regression | 0.42 | 0.51 |\n"
        "| ORION | 0.63 | 0.74 |"
    )
    projected = _truncate_table(table, 350, query="ORION benchmark results")
    assert "| Model | Acc | F1 |" in projected
    assert "| Model | CCC A | CCC V |" in projected
    assert "| I. Recognition | | |" in projected
    assert "| II. Regression | | |" in projected
    assert "| ORION | 70 | 71 |" in projected
    assert "| ORION | 0.63 | 0.74 |" in projected
    assert projected.index("| Model | CCC A | CCC V |") < projected.index("| ORION | 0.63 | 0.74 |")
    from app.services.evidence_validator import validate_answer_claims
    evidence = [{"document_id": "doc", "filename": "multi-task.pdf", "artifact_type": "table",
                 "table_id": "table", "content": projected}]
    supported = validate_answer_claims(
        answer="ORION reaches CCC A 0.63 and CCC V 0.74.",
        documents=evidence, focus_document_ids=["doc"],
    )
    wrong_schema = validate_answer_claims(
        answer="ORION reaches accuracy 0.63.",
        documents=evidence, focus_document_ids=["doc"],
    )
    assert supported.valid
    assert not wrong_schema.valid


def test_table_projection_preserves_dataset_groups_not_just_matching_model_rows():
    rows = "\n".join(f"| Unrelated-{i} | 40 | 50 |" for i in range(25))
    table = (
        "| Model | UA | WA |\n|---|---|---|\n"
        "| Dataset-A | | |\n" + rows + "\n| Target | 70 | 71 |\n"
        "| Dataset-B | | |\n| Target | 80 | 81 |"
    )
    projected = _truncate_table(table, 250, query="Target evaluation")
    assert "| Dataset-A | | |" in projected
    assert "| Dataset-B | | |" in projected
    assert projected.index("| Dataset-A | | |") < projected.index("| Target | 70 | 71 |")
    assert projected.index("| Dataset-B | | |") < projected.index("| Target | 80 | 81 |")


def test_projection_reaches_second_separate_markdown_table_without_first_schema():
    table = (
        "caption: Summary\n| Item | Price |\n|---|---|\n"
        + "\n".join(f"| Part-{i} | {i} |" for i in range(30))
        + "\n\n### Quality benchmark\n| Method | WER |\n|---|---|\n"
        + "| Target | 4.2 |\n| Baseline | 8.1 |"
    )
    projected = _truncate_table(table, 220, query="Target WER")
    assert "### Quality benchmark" in projected
    assert "| Method | WER |" in projected
    assert "| Target | 4.2 |" in projected


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


def test_context_composer_preserves_exact_technical_anchors_under_budget() -> None:
    """A late exact row must survive when semantic distractors fill the budget."""

    results = [
        {
            "chunk_id": "semantic-1",
            "document_id": "doc-manual",
            "page_number": 20,
            "score": 1.0,
            "content": "General safety and compliance guidance. " * 35,
        },
        {
            "chunk_id": "semantic-2",
            "document_id": "doc-manual",
            "page_number": 19,
            "score": 0.9,
            "content": "General certification and safety guidance. " * 35,
        },
        {
            "chunk_id": "exact-technical-row",
            "document_id": "doc-manual",
            "page_number": 25,
            "score": 0.2,
            "content": "Safety Approval UL Listed and CSA.",
        },
    ]

    composed = compose_retrieval_context(
        results,
        query="What is the safety approval for UL Listed and CSA?",
        max_sources=2,
        max_chars=2_000,
        max_chars_per_source=700,
    )

    assert composed.sources[0]["chunk_id"] == "exact-technical-row"


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


def test_hybrid_frontier_reserves_relevant_table_for_metric_query() -> None:
    """Intro/title chunks must not hide an already retrieved result table."""

    text_rows = [
        type(
            "Chunk",
            (),
            {
                "id": f"text-{index}",
                "text": "Introduction and document overview metrics.",
                "score": 0.01,
                "metadata": {"document_id": "doc-1", "chunk_type": "text"},
                "source": "text",
            },
        )()
        for index in range(5)
    ]
    table = type(
        "Chunk",
        (),
        {
            "id": "table-1",
            "text": "Table 1: Performance metrics | Accuracy | F1 | CCC",
            "score": 0.2,
            "metadata": {
                "document_id": "doc-1",
                "chunk_type": "table",
                "artifact_type": "table",
                "table_id": "table-1",
                "caption": "Performance metrics",
            },
            "source": "table",
        },
    )()

    results = _merge_hybrid_chunks(
        "which metrics were used for performance evaluation?",
        [*text_rows, table],
        [],
        4,
    )

    assert any(item.get("table_id") == "table-1" for item in results)


def test_hybrid_vector_fusion_ranks_each_modality_independently() -> None:
    """A sparse table channel must not be penalized by many text chunks."""

    def chunk(identifier, text, score, chunk_type, **metadata):
        return type(
            "Chunk",
            (),
            {
                "id": identifier,
                "text": text,
                "score": score,
                "metadata": {
                    "document_id": "doc-1",
                    "chunk_type": chunk_type,
                    **metadata,
                },
                "source": chunk_type,
            },
        )()

    text_rows = [
        chunk(f"text-{index}", "generic prose", 0.1 + index / 100, "text")
        for index in range(5)
    ]
    table = chunk(
        "table-1",
        "Table: Indian languages Hindi English Kannada",
        1.3,
        "table",
        artifact_type="table",
        table_id="table-1",
        caption="Dataset languages",
    )

    results = _merge_hybrid_chunks(
        "Which Indian languages were used?",
        [*text_rows, table],
        [],
        2,
        vector_ranked_groups=(text_rows, [table], []),
    )

    assert results[0]["table_id"] == "table-1"


def test_hybrid_list_question_reserves_table_with_exact_items() -> None:
    """List-shaped questions can retain a table even without saying table."""

    text_rows = [
        type(
            "Chunk",
            (),
            {
                "id": f"text-{index}",
                "text": "abstract and experiment prose",
                "score": 0.01,
                "metadata": {"document_id": "doc-1", "chunk_type": "text"},
                "source": "text",
            },
        )()
        for index in range(6)
    ]
    table = type(
        "Chunk",
        (),
        {
            "id": "table-1",
            "text": "Table 1: Languages | Hindi | English | Kannada",
            "score": 1.0,
            "metadata": {
                "document_id": "doc-1",
                "chunk_type": "table",
                "artifact_type": "table",
                "table_id": "table-1",
                "caption": "Dataset languages",
            },
            "source": "table",
        },
    )()

    results = _merge_hybrid_chunks(
        "Which languages did they experiment with?",
        [*text_rows, table],
        [],
        3,
        vector_ranked_groups=(text_rows, [table], []),
    )

    assert any(item.get("table_id") == "table-1" for item in results)


def test_structured_promotion_happens_at_final_cutoff() -> None:
    """A deep candidate frontier must not hide a valid table at visible top-k."""

    ranked = [
        {
            "chunk_id": f"text-{index}",
            "document_id": "doc-1",
            "content": "generic experiment prose",
            "score": 1.0 - index / 100,
        }
        for index in range(10)
    ]
    ranked.append(
        {
            "chunk_id": "table-languages",
            "document_id": "doc-1",
            "artifact_type": "table",
            "table_id": "table-languages",
            "caption": "Dataset languages",
            "content": "Hindi English Kannada",
            "raw_vector_score": 0.1,
            "score": 0.01,
        }
    )

    visible = _promote_structured_evidence(
        ranked,
        query="Which languages did they experiment with?",
        top_k=10,
    )

    assert len(visible) == 10
    assert visible[-1]["table_id"] == "table-languages"


def test_enumerated_evidence_intent_is_generic() -> None:
    assert has_enumerated_evidence_intent("Which languages did they use?")
    assert has_enumerated_evidence_intent("Liệt kê các mô hình được so sánh")
    assert not has_enumerated_evidence_intent("Which paper introduced this method?")


def test_hybrid_frontier_reserves_metric_prose_for_open_ended_detail_query() -> None:
    text_rows = [
        type(
            "Chunk",
            (),
            {
                "id": f"intro-{index}",
                "text": "Introduction and document overview.",
                "score": 0.02,
                "metadata": {"document_id": "doc-1", "chunk_type": "text"},
                "source": "text",
            },
        )()
        for index in range(5)
    ]
    metric_detail = type(
        "Chunk",
        (),
        {
            "id": "metric-detail",
            "text": "Evaluation uses Pearson correlation and macro average F1 metrics.",
            "score": 0.01,
            "metadata": {"document_id": "doc-1", "chunk_type": "text"},
            "source": "text",
        },
    )()

    results = _merge_hybrid_chunks(
        "which metrics were used for performance evaluation?",
        [*text_rows, metric_detail],
        [],
        3,
    )

    assert any(item.get("chunk_id") == "metric-detail" for item in results)


def test_hybrid_frontier_does_not_promote_unrelated_artifact() -> None:
    text = type(
        "Chunk",
        (),
        {
            "id": "text-1",
            "text": "The architecture is described here.",
            "score": 0.1,
            "metadata": {"document_id": "doc-1", "chunk_type": "text"},
            "source": "text",
        },
    )()
    unrelated = type(
        "Chunk",
        (),
        {
            "id": "table-1",
            "text": "Table 1: unrelated scores",
            "score": 0.01,
            "metadata": {
                "document_id": "doc-1",
                "chunk_type": "table",
                "artifact_type": "table",
                "table_id": "table-1",
                "caption": "unrelated scores",
            },
            "source": "table",
        },
    )()
    results = _merge_hybrid_chunks("explain the architecture", [text], [], 1)
    assert results[0]["chunk_id"] == "text-1"


def test_context_deduplicates_visual_rows_by_page_before_budget() -> None:
    results = [
        {
            "chunk_id": "visual-a",
            "document_id": "doc-1",
            "page_number": 6,
            "artifact_type": "page",
            "visual_page": True,
            "retrieval_channels": ["visual_page"],
            "content": "page image A",
        },
        {
            "chunk_id": "visual-b",
            "document_id": "doc-1",
            "page_number": 6,
            "artifact_type": "page",
            "visual_page": True,
            "retrieval_channels": ["visual_page"],
            "content": "page image B",
        },
        {
            "chunk_id": "exact-text",
            "document_id": "doc-1",
            "page_number": 1,
            "content": "Inter-Annotator Agreement was 90.5%.",
        },
    ]
    composed = compose_retrieval_context(
        results,
        query="how much agreement was there?",
        max_sources=2,
        max_chunks_per_document=2,
        min_visual_pages=1,
    )
    assert [item["chunk_id"] for item in composed.sources] == [
        "visual-a",
        "exact-text",
    ]


def test_context_deduplicates_figure_and_page_projection_on_same_page() -> None:
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "figure-1",
                "document_id": "doc-1",
                "page_number": 3,
                "figure_id": "figure-1",
                "artifact_type": "figure",
                "content": "Figure 1 architecture",
                "image_path": "/tmp/figure.png",
            },
            {
                "chunk_id": "page-3",
                "document_id": "doc-1",
                "page_number": 3,
                "artifact_type": "page",
                "visual_page": True,
                "retrieval_channels": ["visual_page"],
                "content": "page pixels",
            },
            {
                "chunk_id": "text-1",
                "document_id": "doc-1",
                "page_number": 1,
                "content": "Text evidence",
            },
        ],
        query="what does the figure show?",
        max_sources=2,
        min_figures=1,
        min_visual_pages=1,
    )
    assert [item["chunk_id"] for item in composed.sources] == ["figure-1", "text-1"]


def test_context_preserves_tagged_complementary_obligations_before_lexical_promotion() -> None:
    results = [
        {
            "chunk_id": "numerator",
            "document_id": "doc-1",
            "page_number": 47,
            "content": "Product development 998 in 2019.",
            "evidence_obligation_ids": ["operand_1"],
        },
        {
            "chunk_id": "denominator",
            "document_id": "doc-1",
            "page_number": 69,
            "content": "Total assets 19,845 at December 31, 2019.",
            "evidence_obligation_ids": ["operand_2"],
        },
        {
            "chunk_id": "distractor",
            "document_id": "doc-1",
            "page_number": 86,
            "content": "Asset accounting policy and 2019 results.",
        },
    ]
    composed = compose_retrieval_context(
        results,
        query="what is R&D to asset ratio for Activision Blizzard in FY 2019?",
        max_sources=2,
        max_chunks_per_document=2,
    )
    assert [item["chunk_id"] for item in composed.sources] == [
        "numerator",
        "denominator",
    ]


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


def test_context_composer_reserves_each_requested_facet_per_document() -> None:
    results = [
        {
            "chunk_id": "alpha-architecture",
            "document_id": "doc-alpha",
            "filename": "alpha.pdf",
            "evidence_facets": ["architecture"],
            "content": "Alpha uses cross-attention.",
        },
        {
            "chunk_id": "alpha-results",
            "document_id": "doc-alpha",
            "filename": "alpha.pdf",
            "table_id": "alpha-results",
            "artifact_type": "table",
            "evidence_facets": ["benchmark_results"],
            "content": "| Model | F1 |\n|---|---:|\n| Alpha | 0.80 |",
        },
        {
            "chunk_id": "beta-architecture",
            "document_id": "doc-beta",
            "filename": "beta.pdf",
            "evidence_facets": ["architecture"],
            "content": "Beta uses gated fusion.",
        },
        {
            "chunk_id": "beta-results",
            "document_id": "doc-beta",
            "filename": "beta.pdf",
            "table_id": "beta-results",
            "artifact_type": "table",
            "evidence_facets": ["benchmark_results"],
            "content": "| Model | F1 |\n|---|---:|\n| Beta | 0.82 |",
        },
    ]

    composed = compose_retrieval_context(
        results,
        query="Compare architectures and results.",
        max_sources=2,
        max_chars=2_600,
        min_tables=1,
        required_document_ids=["doc-alpha", "doc-beta"],
        required_facets_by_document={
            "doc-alpha": ["architecture", "benchmark_results"],
            "doc-beta": ["architecture", "benchmark_results"],
        },
    )

    assert {source["chunk_id"] for source in composed.sources} == {
        "alpha-architecture",
        "alpha-results",
        "beta-architecture",
        "beta-results",
    }
    assert all(
        not row["missing_facets"]
        for row in composed.stats["required_facet_coverage"]
    )


def test_context_composer_does_not_invent_missing_facet_coverage() -> None:
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "alpha-task",
                "document_id": "doc-alpha",
                "filename": "alpha.pdf",
                "evidence_facets": ["task"],
                "content": "Alpha solves a classification task.",
            }
        ],
        required_document_ids=["doc-alpha"],
        required_facets_by_document={"doc-alpha": ["architecture"]},
    )

    assert composed.stats["required_facet_coverage"] == [
        {
            "document_id": "doc-alpha",
            "requested_facets": ["architecture"],
            "covered_facets": [],
            "missing_facets": ["architecture"],
        }
    ]


def test_context_composer_derives_facets_for_focused_documents_without_labels() -> None:
    """Legacy text projections still receive generic task/architecture coverage."""

    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "task",
                "document_id": "doc-alpha",
                "section_title": "Abstract",
                "content": "We propose a model to solve speech emotion recognition.",
            },
            {
                "chunk_id": "architecture",
                "document_id": "doc-alpha",
                "section_title": "Method",
                "content": "The architecture uses an encoder and gated fusion pipeline.",
            },
        ],
        query="What is the task and main architecture?",
        max_sources=2,
        required_document_ids=["doc-alpha"],
    )

    assert [source["chunk_id"] for source in composed.sources] == [
        "task",
        "architecture",
    ]
    coverage = composed.stats["required_facet_coverage"][0]
    assert not coverage["missing_facets"]


def test_context_facet_inference_uses_content_for_unlabelled_result() -> None:
    composed = compose_retrieval_context(
        [
            {
                "chunk_id": "architecture",
                "document_id": "doc-alpha",
                "content": "Our architecture contains an encoder and cross-modal fusion module.",
            },
            {
                "chunk_id": "noise",
                "document_id": "doc-alpha",
                "content": "References and acknowledgements.",
            },
        ],
        query="Explain the architecture.",
        max_sources=1,
        required_document_ids=["doc-alpha"],
    )

    assert composed.sources[0]["chunk_id"] == "architecture"


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


@pytest.mark.parametrize("caption, role", [
    ("Table 2: Results under distribution shift", "main"),
    ("Bảng 3: Kết quả so sánh với các phương pháp khác", "main"),
    ("Table 1: Accuracy (%) of different models", "main"),
    ("Bảng 2: Thống kê bộ dữ liệu", "statistics"),
    ("Table 1: Training-set statistics for benchmark variants", "statistics"),
    ("Table 6: Ablation study", "ablation"),
])
def test_table_role_reads_vietnamese_and_prefers_result_cues(caption, role):
    from app.rag.context import table_result_role

    assert table_result_role({"caption": caption}) == role


def test_table_role_falls_back_to_generic_metric_header():
    from app.rag.context import table_result_role

    table = {"caption": "Table 9", "content": "| Model | mAP | IoU |\n|---|---|---|\n| EM-Net | 0.5 | 0.6 |"}
    assert table_result_role(table) == "main"
    assert table_result_role({"caption": "Table 3", "content": "| Name | City |\n|---|---|\n| EM-Net | Hanoi |"}) == "unknown"


def test_formula_questions_request_the_method_facet():
    from app.rag.paper_facets import extract_query_facets

    for query in ("Write out the total training loss", "Cho mình công thức của loss", "Show the attention equation"):
        assert "training_method" in extract_query_facets(query)
    assert "training_method" not in extract_query_facets("Which datasets were used?")


def test_facet_reservation_prefers_the_passage_about_the_question():
    from app.rag.context import _reserve_required_document_sources

    common = "The ORION model uses training on the corpus."
    setup = {"document_id": "d", "chunk_id": "setup", "section_title": "Experimental setup",
             "content": common + " Training uses Adam with a learning rate and the loss is logged."}
    objective = {"document_id": "d", "chunk_id": "objective", "section_title": "Optimization",
                 "content": common + " We minimize L𝑡𝑜𝑡𝑎𝑙 = L 𝐶𝐸 + λ (1 − CCC) as the total objective."}
    filler = [{"document_id": "d", "chunk_id": f"f{i}", "content": common + " Results are reported."} for i in range(4)]
    _ordered, reserved, _ = _reserve_required_document_sources(
        [setup, *filler, objective], required_document_ids=["d"],
        required_facets_by_document={"d": ["training_method"]}, prefer_tables=False,
        query="Write out the total training loss ORION minimizes",
    )
    assert id(objective) in reserved and id(setup) not in reserved
