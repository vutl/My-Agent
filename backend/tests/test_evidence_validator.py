from app.services.evidence_validator import (
    _extract_line_claims,
    validate_answer_claims,
    validate_retrieval_evidence,
)


def test_multi_document_scope_requires_every_requested_document_independent_of_intent() -> None:
    empty = validate_retrieval_evidence(
        documents=[],
        required_entities=[],
        current_topic=None,
        is_followup=True,
        focus_document_ids=["doc-b", "doc-a"],
        require_all_focus_documents=True,
    )
    assert empty.valid is False
    assert empty.reason == "no_documents"
    assert empty.missing_document_ids == ["doc-a", "doc-b"]

    missing = validate_retrieval_evidence(
        documents=[
            {
                "document_id": "doc-a",
                "filename": "A.pdf",
                "content": "A result table",
            }
        ],
        required_entities=[],
        current_topic=None,
        is_followup=True,
        focus_document_ids=["doc-a", "doc-b"],
        require_all_focus_documents=True,
    )
    assert missing.valid is False
    assert missing.reason == "missing_focus_documents"
    assert missing.missing_document_ids == ["doc-b"]

    complete = validate_retrieval_evidence(
        documents=[
            {"document_id": "doc-a", "filename": "A.pdf", "content": "A"},
            {"document_id": "doc-b", "filename": "B.pdf", "content": "B"},
        ],
        required_entities=[],
        current_topic=None,
        is_followup=True,
        focus_document_ids=["doc-a", "doc-b"],
        require_all_focus_documents=True,
    )
    assert complete.valid is True
    assert complete.missing_document_ids == []


def test_retrieval_rejects_topic_mentioned_only_by_another_paper() -> None:
    result = validate_retrieval_evidence(
        documents=[
            {
                "document_id": "wav2small",
                "filename": "Wav2Small.pdf",
                "content": "Related work compares our method with ASPIRE.",
            }
        ],
        required_entities=["ASPIRE"],
        current_topic="ASPIRE",
        is_followup=False,
        focus_document_ids=None,
    )

    assert result.valid is False
    assert result.reason == "missing_required_entities"
    assert result.missing_entities == ["ASPIRE"]


def test_retrieval_accepts_canonical_focused_document_without_title_in_chunk() -> None:
    result = validate_retrieval_evidence(
        documents=[{"document_id": "aspire-id", "content": "The proposed architecture has three stages."}],
        required_entities=["ASPIRE"],
        current_topic="ASPIRE",
        is_followup=True,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "focused_documents_present"


def test_compact_from_to_slash_comparison_binds_metrics_and_model_versions() -> None:
    documents = [
        {
            "document_id": "visec-id",
            "filename": "ICASSP_2024___ViSEC.pdf",
            "content": """
| Model | UA (%) | WA (%) |
|---|---:|---:|
| ViSEC (Vietnamese) | | |
| Wav2Vec 2.0 (baseline) | 61.74 | 62.37 |
| Pitch-fusion | 72.72 | 71.90 |
""",
        }
    ]
    answer = (
        "Pitch-fusion trên ViSEC tăng từ 61.74/62.37 lên 72.72/71.90 "
        "UA/WA so với Wav2Vec 2.0."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["visec-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.subjects) for claim in result.checked_claims] == [
        ("ua", "61.74", ("WAV2VEC",)),
        ("wa", "62.37", ("WAV2VEC",)),
        ("ua", "72.72", ("PITCH-FUSION",)),
        ("wa", "71.9", ("PITCH-FUSION",)),
    ]


def test_comma_before_metric_value_is_not_a_leading_decimal() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE reports Accuracy 75.86, F1 76.31, and CCC 0.714 / 0.740.",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "ASPIRE đạt 75.86 Accuracy, 76.31 F1, cùng CCC 0.714 arousal "
            "và 0.740 valence."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
        ("ccc", "0.714"),
        ("ccc", "0.74"),
    ]


def test_new_sentence_metric_does_not_inherit_previous_baseline_owner() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | CCC A | CCC V |
|---|---:|---:|
| Hierarchical D-C | 0.717 | 0.660 |
| ASPIRE | 0.714 | 0.740 |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "ASPIRE đạt CCC A 0.714; Hierarchical D-C đạt 0.717. "
            "Tuy nhiên, CCC V 0.740 là kết quả cao nhất của model đề xuất."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    final_claim = result.checked_claims[-1]
    assert (final_claim.metric, final_claim.value, final_claim.subjects) == ("ccc", "0.74", ())


def test_retrieval_rejects_foreign_or_mixed_focus_documents() -> None:
    wrong = validate_retrieval_evidence(
        documents=[{"document_id": "other", "filename": "ASPIRE.pdf", "content": "ASPIRE"}],
        required_entities=["ASPIRE"],
        current_topic="ASPIRE",
        is_followup=True,
        focus_document_ids=["aspire-id"],
    )
    mixed = validate_retrieval_evidence(
        documents=[
            {"document_id": "aspire-id", "content": "ASPIRE"},
            {"document_id": "other", "content": "unrelated"},
        ],
        required_entities=["ASPIRE"],
        current_topic="ASPIRE",
        is_followup=True,
        focus_document_ids=["aspire-id"],
    )

    assert (wrong.valid, wrong.reason) == (False, "focus_document_mismatch")
    assert (mixed.valid, mixed.reason) == (False, "mixed_focus_documents")


def test_qualitative_answer_is_not_blocked_by_numeric_validator() -> None:
    result = validate_answer_claims(
        answer="ASPIRE uses a multi-stage speech emotion recognition pipeline.",
        documents=[],
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []


def test_exact_metric_value_is_supported_and_different_metric_is_not() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE obtains Accuracy = 82.5% on the evaluation set.",
        }
    ]
    supported = validate_answer_claims(
        answer="ASPIRE đạt Acc 82,5%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    unsupported = validate_answer_claims(
        answer="ASPIRE đạt F1 82.5%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert supported.valid is True
    assert supported.reason == "metric_values_supported"
    assert unsupported.valid is False
    assert unsupported.reason == "unsupported_metric_values"
    assert [(claim.metric, claim.value) for claim in unsupported.unsupported_claims] == [("f1", "82.5")]


def test_percent_and_fraction_are_treated_as_equivalent_only_with_percent_conversion() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "The accuracy score is 0.825.",
        }
    ]

    converted = validate_answer_claims(
        answer="ASPIRE có accuracy 82.5%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    wrong_scale = validate_answer_claims(
        answer="ASPIRE có accuracy 82.5.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert converted.valid is True
    assert wrong_scale.valid is False


def test_foreign_paper_cannot_support_active_paper_metric() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "The paper reports qualitative improvements.",
        },
        {
            "document_id": "wav-id",
            "filename": "Wav2Small.pdf",
            "content": "Wav2Small reaches Accuracy 91.7%.",
        },
    ]

    result = validate_answer_claims(
        answer="ASPIRE đạt Accuracy 91.7%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is False
    assert result.foreign_document_ids == ["wav-id"]
    assert result.unsupported_claims[0].metric == "accuracy"


def test_subject_prevents_cross_assignment_inside_multi_paper_scope() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE reports Accuracy 81.0%.",
        },
        {
            "document_id": "wav-id",
            "filename": "Wav2Small.pdf",
            "content": "Wav2Small reports Accuracy 91.7%.",
        },
    ]

    result = validate_answer_claims(
        answer="ASPIRE đạt Accuracy 91.7%.",
        documents=documents,
        focus_document_ids=["aspire-id", "wav-id"],
    )

    assert result.valid is False


def test_answer_must_cover_every_required_document_not_only_retrieval_scope() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaModel.pdf",
            "content": "AlphaModel reports F1 0.80.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaModel.pdf",
            "content": "BetaModel reports F1 0.70.",
        },
    ]

    incomplete = validate_answer_claims(
        answer="AlphaModel đạt F1 0.80.",
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    complete = validate_answer_claims(
        answer="AlphaModel đạt F1 0.80, còn BetaModel đạt F1 0.70.",
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert incomplete.valid is False
    assert incomplete.reason == "missing_answer_documents"
    assert incomplete.covered_document_ids == ["alpha-id"]
    assert incomplete.missing_document_ids == ["beta-id"]
    assert complete.valid is True
    assert complete.covered_document_ids == ["alpha-id", "beta-id"]
    assert complete.missing_document_ids == []


def test_qualitative_multi_document_answer_also_requires_named_coverage() -> None:
    documents = [
        {"document_id": "alpha-id", "filename": "Alpha_Model.pdf", "content": "A"},
        {"document_id": "beta-id", "filename": "Beta-Model.pdf", "content": "B"},
    ]

    incomplete = validate_answer_claims(
        answer="Alpha Model uses an acoustic encoder.",
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
    )
    complete = validate_answer_claims(
        answer="Alpha Model uses an acoustic encoder; Beta Model uses a compact decoder.",
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
    )

    assert incomplete.valid is False
    assert incomplete.missing_document_ids == ["beta-id"]
    assert complete.valid is True


def test_shared_generic_owner_value_requires_per_document_claim_context() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaNet.pdf",
            "content": "| Model | F1 |\n|---|---:|\n| Ours | 0.80 |",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaNet.pdf",
            "content": "| Model | F1 |\n|---|---:|\n| Ours | 0.80 |",
        },
    ]

    ambiguous = validate_answer_claims(
        answer=(
            "AlphaNet uses speech, while BetaNet uses text. "
            "Ours reports F1 0.80."
        ),
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    per_document = validate_answer_claims(
        answer=(
            "AlphaNet — Ours reports F1 0.80.\n"
            "BetaNet — Ours reports F1 0.80."
        ),
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert ambiguous.valid is False
    assert ambiguous.reason == "missing_answer_documents"
    assert ambiguous.covered_document_ids == []
    assert ambiguous.missing_document_ids == ["alpha-id", "beta-id"]
    assert per_document.valid is True
    assert per_document.covered_document_ids == ["alpha-id", "beta-id"]


def test_subjectless_metric_lines_inherit_nearest_document_heading() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaModel reports UA 80% and WA 79%.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "BetaModel reports UA 72% and WA 71%.",
        },
    ]
    result = validate_answer_claims(
        answer=(
            "### AlphaStudy — AlphaModel\nUA 80%; WA 79%.\n"
            "### BetaStudy — BetaModel\nUA 72%; WA 71%."
        ),
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
    )

    assert result.valid is True
    assert result.covered_document_ids == ["alpha-id", "beta-id"]
    assert {claim.subjects[0] for claim in result.checked_claims} == {
        "ALPHASTUDY",
        "BETASTUDY",
    }


def test_document_heading_uses_catalog_alias_not_only_filename_stem() -> None:
    documents = [
        {
            "document_id": "tonal-id",
            "filename": "conference_2024_1127.pdf",
            "metadata": {"catalog_aliases": ["ViSEC", "Pitch-fusion"]},
            "content": "Pitch-fusion reports UA 72.72% and WA 71.90%.",
        },
        {
            "document_id": "other-id",
            "filename": "OtherPaper.pdf",
            "content": "OtherPaper reports UA 70% and WA 69%.",
        },
    ]
    result = validate_answer_claims(
        answer=(
            "### ViSEC — Pitch-fusion\nUA 72.72%; WA 71.90%.\n"
            "### OtherPaper\nUA 70%; WA 69%."
        ),
        documents=documents,
        focus_document_ids=["tonal-id", "other-id"],
        require_all_focus_documents=True,
    )

    assert result.valid is True
    assert result.covered_document_ids == ["other-id", "tonal-id"]


def test_related_work_metric_inside_active_paper_does_not_support_active_model() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "Among prior systems, Wav2Small reports Accuracy 91.7%.",
        }
    ]

    result = validate_answer_claims(
        answer="ASPIRE đạt Accuracy 91.7%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is False


def test_markdown_table_metrics_are_checked_by_column() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc (%) | F1 (%) | CCC |
| --- | ---: | ---: | ---: |
| ASPIRE | 82.5 | 80.4 | 0.71 |
""",
        }
    ]
    answer = """
| Model | Acc (%) | F1 (%) | CCC |
| --- | ---: | ---: | ---: |
| ASPIRE | 82.5 | 80.4 | 0.72 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is False
    assert [(claim.metric, claim.value) for claim in result.unsupported_claims] == [("ccc", "0.72")]


def test_figure_page_year_and_layer_numbers_are_outside_bounded_metric_check() -> None:
    result = validate_answer_claims(
        answer="Figure 2 on page 7 shows a 3-stage model published in 2024.",
        documents=[],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"


def test_bare_percentage_must_exist_in_evidence() -> None:
    documents = [{"document_id": "doc", "content": "The error reduction is 4.2%."}]

    supported = validate_answer_claims(answer="Sai số giảm 4.2%.", documents=documents)
    unsupported = validate_answer_claims(answer="Sai số giảm 5.2%.", documents=documents)

    assert supported.valid is True
    assert unsupported.valid is False


def test_bare_percentage_is_not_supported_by_unrelated_metric_percentage() -> None:
    documents = [{"document_id": "doc", "content": "The model obtains F1 4.2%."}]

    result = validate_answer_claims(answer="Sai số giảm 4.2%.", documents=documents)

    assert result.valid is False


def test_subjectless_proposed_model_claim_cannot_borrow_baseline_table_value() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc (%) |
| --- | ---: |
| BaselineX | 82.5 |
| ASPIRE | 90.0 |
""",
        }
    ]

    baseline_value = validate_answer_claims(
        answer="Mô hình đề xuất đạt Acc 82.5%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    active_value = validate_answer_claims(
        answer="Mô hình đề xuất đạt Acc 90.0%.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert baseline_value.valid is False
    assert active_value.valid is True


def test_uncertainty_value_is_validated_not_silently_ignored() -> None:
    documents = [
        {
            "document_id": "doc",
            "content": "The proposed system obtains F1 = 82.5 ± 0.3%.",
        }
    ]

    supported = validate_answer_claims(
        answer="F1 = 82.5 ± 0.3%.",
        documents=documents,
    )
    unsupported = validate_answer_claims(
        answer="F1 = 82.5 ± 0.4%.",
        documents=documents,
    )

    assert supported.valid is True
    assert unsupported.valid is False
    assert [(claim.metric, claim.value) for claim in unsupported.unsupported_claims] == [("f1", "0.4")]


def test_reference_and_list_numbers_are_not_misread_as_metric_values() -> None:
    documents = [{"document_id": "doc", "content": "Accuracy is 82.5%."}]

    result = validate_answer_claims(
        answer="1. Figure 2 reports Accuracy 82.5%.",
        documents=documents,
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [("accuracy", "82.5")]


def test_trainable_parameter_count_with_magnitude_is_evidence_checked() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE requires only 6.35M trainable parameters.",
        }
    ]

    supported = validate_answer_claims(
        answer="ASPIRE chỉ có 6.35M trainable parameters.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    unsupported = validate_answer_claims(
        answer="ASPIRE chỉ có 8.35M trainable parameters.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert supported.valid is True
    assert supported.checked_claims[0].value == "6350000"
    assert unsupported.valid is False
    assert unsupported.unsupported_claims[0].metric == "parameters"


def test_row_oriented_vietnamese_markdown_metrics_are_all_checked() -> None:
    table = """
| Dataset / split | Metric được nêu | Kết quả ASPIRE | Ghi chú |
|---|---|---:|---|
| IEMOCAP | Accuracy | **75.86%** | controlled |
| IEMOCAP | Macro F1 | **76.31%** | class-balanced |
| MSP-Podcast Test1 | Acc / F1 | **72.10 / 67.58** | in-the-wild |
| MSP-Podcast Test2 | Acc / F1 | **72.53 / 52.93** | noisy |
| IEMOCAP / MSP-Podcast | CCC | **Chưa có số trong evidence hiện tại** | không tự điền |
"""
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": table,
        }
    ]

    result = validate_answer_claims(
        answer=table,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "metric_values_supported"
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
        ("accuracy", "72.1"),
        ("f1", "67.58"),
        ("accuracy", "72.53"),
        ("f1", "52.93"),
    ]


def test_row_oriented_multi_metric_cell_rejects_one_wrong_aligned_value() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Dataset | Metric | Result |
|---|---|---:|
| MSP-Podcast Test1 | Acc / F1 | 72.10 / 67.58 |
""",
        }
    ]
    answer = """
| Dataset | Chỉ số | Kết quả |
|---|---|---:|
| MSP-Podcast Test1 | Acc / F1 | 72.10 / 68.58 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is False
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "72.1"),
        ("f1", "68.58"),
    ]
    assert [(claim.metric, claim.value) for claim in result.unsupported_claims] == [
        ("f1", "68.58")
    ]


def test_generated_table_dataset_column_does_not_become_model_owner() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc | F1 | CCC A | CCC V |
|---|---:|---:|---:|---:|
| ASPIRE | 75.86 | 76.31 | 0.714 | 0.740 |
""",
        }
    ]
    answer = """
| Task | Dataset | Model | Acc | F1 | CCC A | CCC V |
|---|---|---|---:|---:|---:|---:|
| SER | IEMOCAP | ASPIRE | 75.86 | 76.31 | 0.714 | 0.740 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert {claim.metric for claim in result.checked_claims} == {"accuracy", "f1", "ccc"}
    assert all(claim.subjects == ("ASPIRE",) for claim in result.checked_claims)


def test_sentence_cased_hyphenated_model_table_values_are_supported() -> None:
    documents = [
        {
            "document_id": "visec-id",
            "filename": "ICASSP_2024___ViSEC.pdf",
            "content": """
| Model | UA (%) | WA (%) |
|---|---:|---:|
| ViSEC (Vietnamese) | | |
| Wav2Vec 2.0 (baseline) | 61.74 | 62.37 |
| Pitch-fusion | 72.72 | 71.90 |
| ASVP-ESD (Chinese) | | |
| Wav2Vec 2.0 (baseline) | 63.72 | 63.66 |
| Pitch-fusion | 85.40 | 84.77 |
| Thai SER (Thai) | | |
| Wav2Vec 2.0 (baseline) | 76.76 | 77.07 |
| Pitch-fusion | 87.47 | 87.73 |
""",
        }
    ]
    answer = """
| Dataset | Model | UA (%) | WA (%) |
|---|---|---:|---:|
| ViSEC | Wav2Vec 2.0 | 61.74 | 62.37 |
| ViSEC | Pitch-fusion | 72.72 | 71.90 |
| ASVP-ESD | Wav2Vec 2.0 | 63.72 | 63.66 |
| ASVP-ESD | Pitch-fusion | 85.40 | 84.77 |
| Thai SER | Wav2Vec 2.0 | 76.76 | 77.07 |
| Thai SER | Pitch-fusion | 87.47 | 87.73 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["visec-id"],
    )

    assert result.valid is True
    pitch_claims = [
        claim for claim in result.checked_claims if claim.subjects == ("PITCH-FUSION",)
    ]
    assert {(claim.metric, claim.value) for claim in pitch_claims} == {
        ("ua", "72.72"),
        ("wa", "71.9"),
        ("ua", "85.4"),
        ("wa", "84.77"),
        ("ua", "87.47"),
        ("wa", "87.73"),
    }


def test_multidocument_markdown_owner_heading_binds_following_metric_line() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": """
| Model | Accuracy |
|---|---:|
| AlphaNet | 75.86 |
""",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": """
| Model | UA (%) | WA (%) |
|---|---:|---:|
| Encoder 2.0 (baseline) | 61.74 | 62.37 |
| Target-fusion | 72.72 | 71.90 |
""",
        },
    ]
    answer = """
### AlphaStudy
**AlphaNet**
**Accuracy 75.86%**

### BetaStudy
- **Encoder 2.0 (baseline)**
**UA 61.74%, WA 62.37%**
- **Target-fusion**
**UA 72.72%, WA 71.90%**
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert result.valid is True
    baseline_claims = [claim for claim in result.checked_claims if claim.value in {"61.74", "62.37"}]
    assert baseline_claims
    assert all(set(claim.subjects) >= {"BETASTUDY", "BASELINE"} for claim in baseline_claims)


def test_unique_owner_free_measurement_is_supported_but_ambiguous_duplicate_fails() -> None:
    beta_document = {
        "document_id": "beta-id",
        "filename": "BetaStudy.pdf",
        "content": """
| Model | UA (%) | WA (%) |
|---|---:|---:|
| ReferenceNet (baseline) | 61.74 | 62.37 |
| TargetNet | 72.72 | 71.90 |
""",
    }
    unique = validate_answer_claims(
        answer="UA 61.74% and WA 62.37%.",
        documents=[beta_document],
        focus_document_ids=["beta-id"],
        answer_document_ids=["beta-id"],
    )

    assert unique.valid is True
    assert all(claim.subjects == () for claim in unique.checked_claims)

    duplicate = validate_answer_claims(
        answer="UA 61.74%.",
        documents=[
            {
                "document_id": "alpha-id",
                "filename": "AlphaStudy.pdf",
                "content": """
| Model | UA (%) |
|---|---:|
| OtherNet (baseline) | 61.74 |
""",
            },
            beta_document,
        ],
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert duplicate.valid is False
    assert duplicate.unsupported_claims


def test_metric_alias_separators_are_equivalent_for_compact_scientific_labels() -> None:
    documents = [
        {
            "document_id": "dim-id",
            "filename": "DimensionalStudy.pdf",
            "content": """
| Model | Accuracy | Macro F1 | CCC A | CCC V |
|---|---:|---:|---:|---:|
| DimNet | 75.86 | 76.31 | 0.714 | 0.740 |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "DimNet reports Accuracy 75.86%, macro_F1 76.31%, "
            "CCC_A 0.714, and CCC_V 0.740."
        ),
        documents=documents,
        focus_document_ids=["dim-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in result.checked_claims] == [
        ("accuracy", "75.86", ()),
        ("f1", "76.31", ("macro",)),
        ("ccc", "0.714", ("arousal",)),
        ("ccc", "0.74", ("valence",)),
    ]


def test_combined_model_dataset_cell_drops_contextual_dataset_owner() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc | F1 |
|---|---:|---:|
| ASPIRE | 75.86 | 76.31 |
""",
        }
    ]
    answer = """
| Task | Model / dataset | Acc | F1 |
|---|---|---:|---:|
| Categorical SER | ASPIRE trên IEMOCAP | 75.86 | 76.31 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert all(claim.subjects == ("ASPIRE",) for claim in result.checked_claims)


def test_following_comparison_baseline_does_not_own_target_value() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc | F1 | CCC A |
|---|---:|---:|---:|
| ASPIRE | 75.86 | 76.31 | 0.714 |
| AFEA-Net | 75.10 | 75.40 | - |
| PCM | - | - | 0.717 |
""",
        }
    ]
    answer = (
        "- Acc = 75.86, cao hơn AFEA-Net 75.10.\n"
        "- F1 = 76.31, cao hơn AFEA-Net 75.40.\n"
        "- CCC Arousal = 0.714, hơi dưới PCM 0.717."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    target_claims = [claim for claim in result.checked_claims if claim.value in {"75.86", "76.31", "0.714"}]
    assert all(claim.subjects == () for claim in target_claims)


def test_same_clause_following_examples_own_baseline_value() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | CCC A | CCC V |
|---|---:|---:|
| ASPIRE | 0.714 | 0.740 |
| Hierarchical D-C | 0.717 | 0.660 |
| PCM | 0.717 | 0.630 |
""",
        }
    ]
    answer = (
        "Với CCC, ASPIRE đạt CCC A 0.714, thấp nhẹ so với các model đạt "
        "0.717 như Hierarchical D-C và PCM. Nhưng ASPIRE đạt CCC V 0.740."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    baseline = next(claim for claim in result.checked_claims if claim.value == "0.717")
    assert baseline.subjects == ("D-C",)


def test_metric_and_value_in_fragmented_markdown_fail_closed() -> None:
    result = validate_answer_claims(
        answer="**Accuracy được báo cáo**\nGiá trị truy xuất: **75.86**",
        documents=[],
    )

    assert result.valid is False
    assert result.retry_required is True
    assert result.reason == "unparsed_metric_values"
    assert result.checked_claims == []
    assert result.unparsed_signals
    assert result.unparsed_signals[0]["metric"] == "accuracy"
    assert result.unparsed_signals[0]["value"] == "75.86"


def test_metric_sentence_inside_prose_markdown_cell_is_checked() -> None:
    evidence = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE achieves an accuracy of 75.86% on IEMOCAP.",
        }
    ]
    answer = """
| Contribution | Ý nghĩa |
|---|---|
| Multi-signal modeling | ASPIRE đạt **75.86% accuracy trên IEMOCAP**. |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=evidence,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "metric_values_supported"
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86")
    ]


def test_docling_jammed_metric_value_boundary_is_restored() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "On IEMOCAP, ASPIRE achieves an accuracy of75.86%.",
        }
    ]

    result = validate_answer_claims(
        answer="ASPIRE đạt 75.86% accuracy trên IEMOCAP.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "metric_values_supported"
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86")
    ]


def test_detached_metric_period_is_not_parsed_as_leading_decimal() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE reports Acc 75.86 and F1 76.31.",
        }
    ]

    result = validate_answer_claims(
        answer="ASPIRE đạt Acc . 75.86 và F1 76.31 trên IEMOCAP.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
    ]


def test_top_transformer_layer_count_is_not_a_parameter_claim() -> None:
    result = validate_answer_claims(
        answer=(
            "Trong thiết lập within-corpus, mỗi fold lấy 4 partition để train "
            "và 1 partition để test. Với wav2vec2, họ fine-tune LR model bằng "
            "cách removing the top 12 transformer layers, nhằm giữ hiệu năng "
            "nhưng giảm số tham số."
        ),
        documents=[],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []


def test_repeated_metric_header_inside_one_markdown_block_updates_schema() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Modality | Acc. | F1 | P (M) |
|:---|:---|:---|:---|:---|
| I. Categorical Classification | | | | |
| Baseline | A+T | 72.50 | 72.16 | 3.7 |
| ASPIRE | A+T | 75.86 | 76.31 | 6.35 |
| Model | Modality | CCC A | CCC V | P (M) |
| II. Dimensional Regression | | | | |
| Baseline | A | 0.717 | 0.660 | - |
| ASPIRE | A+T | 0.714 | 0.740 | 6.35 |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "ASPIRE reports Acc 75.86 and F1 76.31; "
            "CCC arousal is 0.714 and CCC valence is 0.740."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert result.reason == "metric_values_supported"
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
        ("ccc", "0.714"),
        ("ccc", "0.74"),
    ]


def test_comparison_prose_assigns_each_value_to_its_local_model_subject() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Accuracy |
|---|---:|
| ASPIRE | 75.86 |
| AFEA-Net | 75.40 |
| MER-HAN | 75.10 |
""",
        }
    ]
    answer = (
        "Accuracy comparison: ASPIRE 75.86, "
        "AFEA-Net 75.40, MER-HAN 75.10."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    cross_assigned = validate_answer_claims(
        answer="ASPIRE reports Accuracy 75.40.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [claim.subjects for claim in result.checked_claims] == [
        ("ASPIRE",),
        ("AFEA-NET",),
        ("MER-HAN",),
    ]
    assert cross_assigned.valid is False
    assert cross_assigned.unsupported_claims[0].subjects == ("ASPIRE",)


def test_repeated_metric_labels_pair_each_number_with_nearest_metric() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc | F1 |
|---|---:|---:|
| ASPIRE | 75.86 | 76.31 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="**Acc/F1:** ASPIRE đạt 75.86 Acc và 76.31 F1.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
    ]


def test_single_value_with_many_metric_labels_maps_only_to_nearest_metric() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE has 6.35M trainable parameters.",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "Acc, F1, CCC arousal, and CCC valence are reported; "
            "ASPIRE uses 6.35M trainable parameters."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("parameters", "6350000")
    ]


def test_parameter_millions_abbreviations_in_table_headers_are_scaled() -> None:
    for header in (
        "P (M)",
        "Params(M)",
        "Trainable Params (M).",
        "Efficiency.Params (M)",
    ):
        documents = [
            {
                "document_id": "aspire-id",
                "filename": "ASPIRE.pdf",
                "content": f"""
| Model | {header} |
|---|---:|
| ASPIRE | 6.35 |
""",
            }
        ]

        result = validate_answer_claims(
            answer="ASPIRE uses 6.35M trainable parameters.",
            documents=documents,
            focus_document_ids=["aspire-id"],
        )

        assert result.valid is True, header
        assert result.checked_claims[0].value == "6350000"


def test_spaced_table_decimals_are_normalized_before_parameter_scaling() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Modality | P (M) |
|---|---|---:|
| AFEA-Net | A+T | ≈ 21 . 7 |
| DEER | A | ≈ 0 , 3 |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "AFEA-Net has about 21.7M trainable parameters, while "
            "DEER has about 0.3M trainable parameters."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.subjects) for claim in result.checked_claims] == [
        ("parameters", "21700000", ("AFEA-NET",)),
        ("parameters", "300000", ("DEER",)),
    ]


def test_spaced_decimal_support_does_not_turn_references_or_lists_into_metrics() -> None:
    result = validate_answer_claims(
        answer=(
            "1 . Figure 2 discusses F1; see Table 3 and citation [21 . 7]. "
            "F1 is also named in items 1, 2, 3 without reported values."
        ),
        documents=[],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []


def test_following_possessive_owner_beats_earlier_discourse_subject() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | P (M) |
|---|---:|
| AFEA-Net | 21.7 |
| ASPIRE | 6.35 |
""",
        }
    ]
    answer = (
        "Điểm chính là ASPIRE vẫn gọn. So với AFEA-Net, ASPIRE chỉ dùng "
        "6.35M trainable parameters, thấp hơn 21.7M của AFEA-Net."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    cross_assigned = validate_answer_claims(
        answer="ASPIRE dùng 21.7M trainable parameters.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    contextual_dataset = validate_answer_claims(
        answer="ASPIRE uses 6.35M trainable parameters on IEMOCAP.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert result.valid is True
    assert [(claim.value, claim.subjects) for claim in result.checked_claims] == [
        ("6350000", ("ASPIRE",)),
        ("21700000", ("AFEA-NET",)),
    ]
    assert cross_assigned.valid is False
    assert cross_assigned.unsupported_claims[0].subjects == ("ASPIRE",)
    assert contextual_dataset.valid is True
    assert contextual_dataset.checked_claims[0].subjects == ("ASPIRE",)


def test_structural_counts_near_metric_words_are_not_metric_values() -> None:
    units = (
        "groups",
        "parts",
        "types",
        "categories",
        "metrics",
        "contributions",
        "steps",
        "reasons",
        "nhóm",
        "phần",
        "loại",
        "chỉ số",
        "đóng góp",
        "bước",
        "lý do",
    )

    for unit in units:
        result = validate_answer_claims(
            answer=f"Paper có 3 {unit}; phần sau chỉ thảo luận Acc/F1 định tính.",
            documents=[],
        )
        assert result.valid is True, unit
        assert result.reason == "no_metric_value_claims", unit
        assert result.checked_claims == [], unit


def test_markdown_table_and_figure_numbers_are_excluded_before_metric_pairing() -> None:
    result = validate_answer_claims(
        answer=(
            "Có 3 nhóm metric: Acc/F1. **Table 2:** gồm 2 phần; "
            "xem thêm **Figure 2**. Không có giá trị benchmark trong câu này."
        ),
        documents=[],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []


def test_dataset_context_identifier_does_not_replace_earlier_model_subject() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | Acc |
|---|---:|
| ASPIRE | 75.86 |
""",
        }
    ]
    answers = (
        "ASPIRE được đánh giá trên IEMOCAP, đạt Acc 75.86.",
        "ASPIRE is evaluated on IEMOCAP, reaching Acc 75.86.",
        "ASPIRE runs on the dataset IEMOCAP, with Acc 75.86.",
        "ASPIRE chạy trên dataset IEMOCAP, đạt Acc 75.86.",
    )

    for answer in answers:
        result = validate_answer_claims(
            answer=answer,
            documents=documents,
            focus_document_ids=["aspire-id"],
        )
        assert result.valid is True, answer
        assert result.checked_claims[0].subjects == ("ASPIRE",), answer

    subjectless_context = validate_answer_claims(
        answer="Trên IEMOCAP, Acc đạt 75.86.",
        documents=documents,
        focus_document_ids=["aspire-id"],
    )
    assert subjectless_context.valid is True
    assert subjectless_context.checked_claims[0].subjects == ()


def test_dataset_identifier_remains_subject_when_used_as_explicit_owner() -> None:
    documents = [
        {
            "document_id": "doc",
            "filename": "results.pdf",
            "content": "IEMOCAP reports Accuracy 75.86.",
        }
    ]

    result = validate_answer_claims(
        answer="IEMOCAP reports Acc 75.86.",
        documents=documents,
    )

    assert result.valid is True
    assert result.checked_claims[0].subjects == ("IEMOCAP",)


def test_fail_closed_signal_requires_bounded_metric_number_distance() -> None:
    answer = (
        "Case 3 is discussed qualitatively. "
        + "This paragraph contains no reported numerical result. " * 3
        + "Acc and F1 are mentioned only as metric names."
    )

    result = validate_answer_claims(answer=answer, documents=[])

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []


def test_domain_acronym_near_value_does_not_override_explicit_model_subject() -> None:
    documents = [
        {
            "document_id": "aspire-id",
            "filename": "ASPIRE.pdf",
            "content": """
| Model | P (M) |
|---|---:|
| ASPIRE | 6.35 |
""",
        }
    ]
    exact = validate_answer_claims(
        answer=(
            "ASPIRE thắng mạnh ở categorical SER và model khá gọn với "
            "6.35M trainable params."
        ),
        documents=documents,
        focus_document_ids=["aspire-id"],
    )

    assert exact.valid is True
    assert exact.checked_claims[0].subjects == ("ASPIRE",)

    for domain_acronym in ("SER", "ASR", "SSL", "AV", "AVD", "GT", "LR"):
        result = validate_answer_claims(
            answer=(
                f"ASPIRE uses {domain_acronym} supervision/context while keeping "
                "6.35M trainable parameters."
            ),
            documents=documents,
            focus_document_ids=["aspire-id"],
        )
        assert result.valid is True, domain_acronym
        assert result.checked_claims[0].subjects == ("ASPIRE",), domain_acronym


def test_compact_f1_variants_do_not_leak_into_a_neighbor_metric() -> None:
    documents = [
        {
            "document_id": "fusion-id",
            "filename": "fusion-study.pdf",
            "content": """
| Model | CCC avg | miF1 | maF1 |
|---|---:|---:|---:|
| FusionNet | 0.638 | 0.50 | 0.47 |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "FusionNet: CCC avg 0.638 | FusionNet miF1 0.50 | "
            "FusionNet maF1 0.47."
        ),
        documents=documents,
        focus_document_ids=["fusion-id"],
    )

    assert result.valid is True
    assert [
        (claim.metric, claim.value, claim.qualifiers)
        for claim in result.checked_claims
    ] == [
        ("ccc", "0.638", ("avg",)),
        ("f1", "0.5", ("micro",)),
        ("f1", "0.47", ("macro",)),
    ]


def test_micro_and_macro_f1_aliases_share_a_family_but_not_evidence() -> None:
    documents = [
        {
            "document_id": "benchmark-id",
            "filename": "benchmark.pdf",
            "content": """
| Method | miF1 |
|---|---:|
| ModelX | 0.50 |
""",
        }
    ]

    for spelling in ("miF1", "mi-F1", "micro-F1", "micro F1 score"):
        supported = validate_answer_claims(
            answer=f"ModelX reports {spelling} 0.50.",
            documents=documents,
            focus_document_ids=["benchmark-id"],
        )
        assert supported.valid is True, spelling
        assert supported.checked_claims[0].metric == "f1", spelling
        assert supported.checked_claims[0].qualifiers == ("micro",), spelling

    unsupported = validate_answer_claims(
        answer="ModelX reports macro-F1 0.50.",
        documents=documents,
        focus_document_ids=["benchmark-id"],
    )
    assert unsupported.valid is False
    assert unsupported.unsupported_claims[0].qualifiers == ("macro",)


def test_multi_scope_phrase_does_not_become_a_fake_single_qualifier() -> None:
    documents = [
        {
            "document_id": "benchmark-id",
            "filename": "benchmark.pdf",
            "content": """
| Method | miF1 | Test2.miF1 |
|---|---:|---:|
| ModelX | 0.50 | 0.505 |
""",
        }
    ]

    for answer in (
        "ModelX reports miF1 about 0.50 on both Test1 and Test2.",
        "ModelX đạt miF1 khoảng 0.50 trên cả Test1 và Test2.",
        "ModelX đạt miF1 khoảng 0.50 trên hai test partition.",
    ):
        result = validate_answer_claims(
            answer=answer,
            documents=documents,
            focus_document_ids=["benchmark-id"],
        )
        assert result.valid is True, answer
        assert result.checked_claims[0].qualifiers == ("micro",), answer


def test_threshold_near_a_metric_is_not_parsed_as_that_metric_value() -> None:
    result = validate_answer_claims(
        answer=(
            "Checkpoint selection uses validation miF1 and evaluation applies "
            "a fixed threshold t = 0.45."
        ),
        documents=[],
    )

    assert result.valid is True
    assert result.reason == "no_metric_value_claims"
    assert result.checked_claims == []

    vietnamese = validate_answer_claims(
        answer=(
            "Dùng miF1, maF1, Jaccard và Exact Accuracy; "
            "threshold đánh giá cố định là `0.45`."
        ),
        documents=[],
    )
    assert vietnamese.valid is True
    assert vietnamese.reason == "no_metric_value_claims"


def test_coordinated_dataset_context_does_not_become_result_owner() -> None:
    documents = [
        {
            "document_id": "method-id",
            "filename": "Method.pdf",
            "content": "The proposed model achieves accuracy of 75.86% on IEMOCAP.",
        }
    ]
    answer = (
        "The model is evaluated on IEMOCAP and MSP-Podcast, and achieves "
        "75.86% accuracy on IEMOCAP."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["method-id"],
    )

    assert result.valid is True
    assert result.checked_claims[0].subjects == ()


def test_semantic_dimension_component_headers_match_plain_dimension_headers() -> None:
    documents = [
        {
            "document_id": "vad-id",
            "filename": "VAD.pdf",
            "content": """
| Modality | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE |
|---|---:|---:|---:|---:|
| audio | 0.46 | 0.10 | 0.40 | 0.09 |
""",
        }
    ]
    answer = """
| Modality | Valence model / CCC / MAE | Arousal predictor / CCC / MAE |
|---|---|---|
| audio | 0.46 / 0.10 | 0.40 / 0.09 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["vad-id"],
    )

    assert result.valid is True
    assert len(result.checked_claims) == 4


def test_nested_leading_condition_extracts_only_the_explicit_slice() -> None:
    documents = [
        {
            "document_id": "within-id",
            "filename": "Within.pdf",
            "content": """
| Model | Condition | CCC A | CCC V |
|---|---|---:|---:|
| wav2vec2 | Within | 0.412 | 0.493 |
""",
        }
    ]
    answer = (
        "Với dự đoán thuộc tính trong điều kiện Within, wav2vec2 đạt "
        "CCC 0.412 arousal và 0.493 valence."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["within-id"],
    )

    assert result.valid is True
    assert all("within" in claim.qualifiers for claim in result.checked_claims)


def test_interleaved_metric_qualifiers_bind_to_their_own_values() -> None:
    documents = [
        {
            "document_id": "dimensional-id",
            "filename": "dimensional.pdf",
            "content": """
| Model | CCC V | CCC A | CCC D | CCC avg |
|---|---:|---:|---:|---:|
| DimNet | 0.632 | 0.680 | 0.601 | 0.638 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="DimNet reports CCC V=0.632, A=0.680, D=0.601, avg=0.638.",
        documents=documents,
        focus_document_ids=["dimensional-id"],
    )

    assert result.valid is True
    assert [(claim.value, claim.qualifiers) for claim in result.checked_claims] == [
        ("0.632", ("valence",)),
        ("0.68", ("arousal",)),
        ("0.601", ("dominance",)),
        ("0.638", ("avg",)),
    ]


def test_comparison_matrix_parses_each_multi_metric_cell_with_local_owner() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": """
| Model | CCC V | CCC A | CCC D | CCC avg |
|---|---:|---:|---:|---:|
| AlphaNet | 0.632 | 0.680 | 0.601 | 0.638 |
""",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": """
| Model | miF1 | maF1 | HL | Jac | ExAcc | BiAcc |
|---|---:|---:|---:|---:|---:|---:|
| BetaNet (Ours) | 0.510 | 0.305 | 0.179 | 0.412 | 0.163 | 0.821 |
""",
        },
    ]
    answer = """
| Khía cạnh | AlphaStudy | BetaStudy |
|---|---|---|
| Metric | CCC-V / CCC-A / CCC-D / CCC-avg | miF1 / maF1 / HL / Jac / ExAcc / BiAcc |
| Kết quả | AlphaNet: CCC-V 0.632, CCC-A 0.680, CCC-D 0.601, CCC-avg 0.638 | BetaNet: miF1 0.510, maF1 0.305, HL 0.179, Jac 0.412, ExAcc 0.163, BiAcc 0.821 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert result.valid is True
    assert len(result.checked_claims) == 10
    assert {claim.subjects for claim in result.checked_claims} == {
        ("ALPHANET",),
        ("BETANET",),
    }
    assert {claim.metric for claim in result.checked_claims} == {
        "ccc",
        "f1",
        "hamming_loss",
        "jaccard",
        "exact_accuracy",
        "binary_accuracy",
    }


def test_hierarchical_global_metric_table_supports_section_qualified_claims() -> None:
    documents = [
        {
            "document_id": "whiser-id",
            "filename": "WhiSER.pdf",
            "content": """
| CCC       | LR        | ADA       | DAL       | Within    |
|:----------|:----------|:----------|:----------|:----------|
| Arousal   | Arousal   | Arousal   | Arousal   | Arousal   |
| wavLM     | 0.299     | 0.396     | 0.391     | 0.418     |
| wav2vec2  | 0.301     | 0.391     | 0.384     | 0.412     |
| Valence   | Valence   | Valence   | Valence   | Valence   |
| wavLM     | 0.392     | 0.441     | 0.446     | 0.483     |
| wav2vec2  | 0.395     | 0.457     | 0.461     | 0.493     |
| Dominance | Dominance | Dominance | Dominance | Dominance |
| wavLM     | 0.338     | 0.369     | 0.379     | 0.387     |
| wav2vec2  | 0.326     | 0.371     | 0.376     | 0.391     |
""",
        }
    ]

    result = validate_answer_claims(
        answer=(
            "WHiSER-wavLM reports CCC arousal 0.418, CCC valence 0.483, "
            "and CCC dominance 0.387."
        ),
        documents=documents,
        focus_document_ids=["whiser-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.subjects, claim.qualifiers) for claim in result.checked_claims] == [
        ("ccc", "0.418", ("WAVLM",), ("arousal",)),
        ("ccc", "0.483", ("WAVLM",), ("valence",)),
        ("ccc", "0.387", ("WAVLM",), ("dominance",)),
    ]


def test_hierarchical_metric_value_cannot_cross_section_or_model_owner() -> None:
    documents = [
        {
            "document_id": "hierarchical-id",
            "filename": "hierarchical.pdf",
            "content": """
| CCC     | Within |
|:--------|:-------|
| Arousal | Arousal |
| wavLM   | 0.418 |
| OtherNet | 0.500 |
| Valence | Valence |
| wavLM   | 0.483 |
| OtherNet | 0.600 |

The wavLM summary also repeats CCC 0.418 without naming its dimension.
""",
        }
    ]

    wrong_section = validate_answer_claims(
        answer="wavLM reports CCC valence 0.418.",
        documents=documents,
        focus_document_ids=["hierarchical-id"],
    )
    wrong_model = validate_answer_claims(
        answer="OtherNet reports CCC arousal 0.418.",
        documents=documents,
        focus_document_ids=["hierarchical-id"],
    )

    assert wrong_section.valid is False
    assert wrong_section.unsupported_claims[0].qualifiers == ("valence",)
    assert wrong_model.valid is False
    assert wrong_model.unsupported_claims[0].subjects == ("OTHERNET",)


def test_hierarchical_global_metric_parser_generalizes_to_unrelated_sections() -> None:
    evidence = [
        {
            "document_id": "sensor-id",
            "filename": "sensor-evaluation.pdf",
            "content": """
| Accuracy       | Zero-shot | Fine-tuned |
|:---------------|:----------|:-----------|
| Indoor scenes  | Indoor scenes | Indoor scenes |
| SensorNet      | 0.61      | 0.78       |
| Outdoor scenes | Outdoor scenes | Outdoor scenes |
| SensorNet      | 0.55      | 0.72       |
""",
        }
    ]
    supported_answer = """
| Accuracy       | Zero-shot |
|:---------------|:----------|
| Outdoor scenes | Outdoor scenes |
| SensorNet      | 0.55      |
"""
    cross_assigned_answer = """
| Accuracy      | Zero-shot |
|:--------------|:----------|
| Indoor scenes | Indoor scenes |
| SensorNet     | 0.55      |
"""
    wrong_column_answer = """
| Accuracy       | Fine-tuned |
|:---------------|:-----------|
| Outdoor scenes | Outdoor scenes |
| SensorNet      | 0.55       |
"""

    supported = validate_answer_claims(
        answer=supported_answer,
        documents=evidence,
        focus_document_ids=["sensor-id"],
    )
    cross_assigned = validate_answer_claims(
        answer=cross_assigned_answer,
        documents=evidence,
        focus_document_ids=["sensor-id"],
    )
    wrong_column = validate_answer_claims(
        answer=wrong_column_answer,
        documents=evidence,
        focus_document_ids=["sensor-id"],
    )

    assert supported.valid is True
    assert supported.checked_claims[0].qualifiers == ("outdoor-scenes", "zero-shot")
    assert cross_assigned.valid is False
    assert cross_assigned.unsupported_claims[0].qualifiers == ("indoor-scenes", "zero-shot")
    assert wrong_column.valid is False
    assert wrong_column.unsupported_claims[0].qualifiers == ("outdoor-scenes", "fine-tuned")


def test_postposed_qualifiers_bind_each_value_without_crossing_sections() -> None:
    evidence = [
        {
            "document_id": "sensor-id",
            "filename": "sensor-evaluation.pdf",
            "content": """
| Accuracy       | Zero-shot |
|:---------------|:----------|
| Indoor scenes  | Indoor scenes |
| SensorNet      | 0.61      |
| Outdoor scenes | Outdoor scenes |
| SensorNet      | 0.55      |
""",
        }
    ]

    supported = validate_answer_claims(
        answer=(
            "SensorNet reports Accuracy 0.55 on Outdoor scenes and "
            "0.61 on Indoor scenes."
        ),
        documents=evidence,
        focus_document_ids=["sensor-id"],
    )
    cross_assigned = validate_answer_claims(
        answer="SensorNet reports Accuracy 0.55 for Indoor scenes.",
        documents=evidence,
        focus_document_ids=["sensor-id"],
    )

    assert supported.valid is True
    assert [(claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("0.55", ("outdoor-scenes",)),
        ("0.61", ("indoor-scenes",)),
    ]
    assert cross_assigned.valid is False
    assert cross_assigned.unsupported_claims[0].qualifiers == ("indoor-scenes",)


def test_hierarchical_owner_roles_are_general_and_cannot_borrow_baseline_values() -> None:
    evidence = [
        {
            "document_id": "generic-id",
            "filename": "generic-evaluation.pdf",
            "content": """
| Accuracy       | Held-out |
|:---------------|:---------|
| Urban traffic  | Urban traffic |
| Ours           | 0.80     |
| Baseline       | 0.70     |
| Rural traffic  | Rural traffic |
| Proposed method | 0.76    |
| Reference model | 0.65    |
""",
        }
    ]

    for answer, expected_subject in (
        ("Ours reports Accuracy 0.80 for Urban traffic.", "PROPOSED-MODEL"),
        ("The proposed model reports Accuracy 0.76 for Rural traffic.", "PROPOSED-MODEL"),
        ("Baseline reports Accuracy 0.70 for Urban traffic.", "BASELINE"),
        ("Reference model reports Accuracy 0.65 for Rural traffic.", "BASELINE"),
    ):
        result = validate_answer_claims(
            answer=answer,
            documents=evidence,
            focus_document_ids=["generic-id"],
        )
        assert result.valid is True, answer
        assert result.checked_claims[0].subjects == (expected_subject,), answer

    borrowed = validate_answer_claims(
        answer="The proposed method reports Accuracy 0.70 for Urban traffic.",
        documents=evidence,
        focus_document_ids=["generic-id"],
    )
    assert borrowed.valid is False
    assert borrowed.unsupported_claims[0].subjects == ("PROPOSED-MODEL",)


def test_caption_schema_and_document_identity_support_owned_comparison_rows() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "caption": (
                "Table 4. Results report average F1 (Macro) and UAR (%) "
                "under each evaluation protocol."
            ),
            "content": """
| Model | LR.F1 | LR.UAR | Within.F1 | Within.UAR |
|---|---:|---:|---:|---:|
| AlphaModel | 0.613 | 63.9 | 0.676 | 69.2 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="""
| Paper — Model | Protocol | Metric | Result |
|---|---|---|---:|
| AlphaStudy — AlphaModel | LR | Macro F1 / UAR | 0.613 / 63.9% |
""",
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )

    assert result.valid is True
    assert {(claim.metric, claim.value) for claim in result.supported_claims} == {
        ("f1", "0.613"),
        ("uar", "63.9"),
    }


def test_foreign_document_identity_cannot_borrow_an_owned_metric_row() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaModel reports F1 0.613.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "BetaModel reports F1 0.700.",
        },
    ]

    result = validate_answer_claims(
        answer="""
| Paper — Model | Metric | Result |
|---|---|---:|
| BetaStudy — AlphaModel | F1 | 0.613 |
""",
        documents=evidence,
        focus_document_ids=["alpha-id", "beta-id"],
    )

    assert result.valid is False
    assert result.unsupported_claims[0].subjects == ("BETASTUDY", "ALPHAMODEL")


def test_structural_counts_and_units_are_not_stolen_by_a_nearby_metric() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaModel reports CCC 0.42.",
        }
    ]

    qualitative = validate_answer_claims(
        answer="AlphaStudy evaluates 4 primary emotions using Macro-F1 and UAR.",
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )
    vietnamese_qualitative = validate_answer_claims(
        answer="AlphaStudy phân loại 4 cảm xúc; dùng Macro-F1 và UAR.",
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )
    mixed_units = validate_answer_claims(
        answer=(
            "AlphaModel reports CCC 0.42, runs in 5 ms, and uses "
            "144 FFT bands."
        ),
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )

    assert qualitative.valid is True
    assert qualitative.checked_claims == []
    assert vietnamese_qualitative.valid is True
    assert vietnamese_qualitative.checked_claims == []
    assert mixed_units.valid is True
    assert [(claim.metric, claim.value) for claim in mixed_units.checked_claims] == [
        ("ccc", "0.42")
    ]


def test_metric_vector_is_owned_by_its_label_until_the_next_metric_label() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": """
CCC arousal 0.418.
CCC valence 0.493.
CCC dominance 0.391.
F1 0.676.
UAR 69.2%.
""",
        }
    ]

    result = validate_answer_claims(
        answer="CCC tối đa 0.418/0.493/0.391; F1 0.676, UAR 69.2%.",
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("ccc", "0.418"),
        ("ccc", "0.493"),
        ("ccc", "0.391"),
        ("f1", "0.676"),
        ("uar", "69.2"),
    ]


def test_document_owned_insufficiency_satisfies_only_its_coverage_obligation() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaModel reports F1 0.80.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "The retrieved excerpt discusses setup only.",
        },
    ]
    answer = """
**AlphaStudy.pdf**

AlphaModel reports F1 0.80.

**BetaStudy.pdf**

There is not enough canonical evidence for the requested result, so I will not guess.
"""

    covered = validate_answer_claims(
        answer=answer,
        documents=evidence,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    title_only = validate_answer_claims(
        answer="**AlphaStudy.pdf**\n\nAlphaModel reports F1 0.80.\n\n**BetaStudy.pdf**",
        documents=evidence,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert covered.valid is True
    assert covered.covered_document_ids == ["alpha-id", "beta-id"]
    assert title_only.valid is False
    assert title_only.reason == "missing_answer_documents"
    assert title_only.missing_document_ids == ["beta-id"]


def test_metric_variant_is_not_mistaken_for_a_model_owner() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "caption": "Table 1. Average F1 (Macro) and UAR (%) results.",
            "content": """
| Model | F1 | UAR |
|---|---:|---:|
| AlphaModel | 0.613 | 63.9 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="AlphaModel: Macro-F1 0.613 and UAR 63.9%.",
        documents=evidence,
        focus_document_ids=["alpha-id"],
    )

    assert result.valid is True
    assert all("MACRO-F1" not in claim.subjects for claim in result.checked_claims)


def test_plain_f1_score_label_is_not_mistaken_for_cross_document_owner() -> None:
    evidence = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "Late fusion achieves a weighted F1-score of 78%.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "The excerpt describes architecture without a result value.",
        },
    ]
    answer = """
**AlphaStudy.pdf**

Late fusion reaches 78% weighted F1.

**BetaStudy.pdf**

There is not enough canonical evidence for a result value, so I will not guess.
"""

    result = validate_answer_claims(
        answer=answer,
        documents=evidence,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    wrong = validate_answer_claims(
        answer=answer.replace("78%", "79%"),
        documents=evidence,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert result.valid is True
    assert wrong.valid is False
    assert wrong.reason == "unsupported_metric_values"


def test_composite_metric_headers_align_each_value_and_condition() -> None:
    documents = [
        {
            "document_id": "transfer-id",
            "filename": "transfer-study.pdf",
            "caption": "Average F1 (Macro) and UAR (%) across transfer conditions.",
            "content": """
| Model | LR.F1 | LR.UAR | ADA.F1 | ADA.UAR |
|---|---:|---:|---:|---:|
| EncoderX | 0.613 | 63.9 | 0.621 | 64.2 |
""",
        }
    ]
    answer = """
| Model | LR F1 / UAR | ADA F1 / UAR |
|---|---:|---:|
| EncoderX | 0.613 / 63.9% | 0.621 / 64.2% |
"""
    wrong_condition = answer.replace("0.613 / 63.9%", "0.613 / 64.2%")

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["transfer-id"],
    )
    rejected = validate_answer_claims(
        answer=wrong_condition,
        documents=documents,
        focus_document_ids=["transfer-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.qualifiers) for claim in supported.checked_claims] == [
        ("f1", ("lr",)),
        ("uar", ("lr",)),
        ("f1", ("ada",)),
        ("uar", ("ada",)),
    ]
    assert rejected.valid is False
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in rejected.unsupported_claims] == [
        ("uar", "64.2", ("lr",))
    ]


def test_war_and_uar_remain_distinct_in_wide_benchmark_tables() -> None:
    documents = [
        {
            "document_id": "benchmark-id",
            "filename": "benchmark.pdf",
            "content": """
| Model | IEMOCAP.WAR | IEMOCAP.UAR | MELD.WAR | MELD.UAR |
|---|---:|---:|---:|---:|
| ModelX | 0.7348 | 0.7585 | 0.6257 | 0.5565 |
""",
        }
    ]
    supported = validate_answer_claims(
        answer=(
            "On IEMOCAP, ModelX reports WAR 0.7348 and UAR 0.7585; "
            "on MELD, ModelX reports WAR 0.6257 and UAR 0.5565."
        ),
        documents=documents,
        focus_document_ids=["benchmark-id"],
    )
    swapped = validate_answer_claims(
        answer="On IEMOCAP, ModelX reports WAR 0.7585 and UAR 0.7348.",
        documents=documents,
        focus_document_ids=["benchmark-id"],
    )

    assert supported.valid is True
    assert [claim.metric for claim in supported.checked_claims] == [
        "war",
        "uar",
        "war",
        "uar",
    ]
    assert swapped.valid is False
    assert len(swapped.unsupported_claims) == 2


def test_wide_grouped_table_binds_local_model_row_and_section_conditions() -> None:
    documents = [
        {
            "document_id": "multitarget-id",
            "filename": "multitarget.pdf",
            "content": """
| | Valence.model | Valence.CCC | Valence.MAE | Arousal.model | Arousal.CCC | Arousal.MAE | Dominance.model | Dominance.CCC | Dominance.MAE |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| audio-text | FF-MT | 0.52 | 0.10 | FF-MT | 0.44 | 0.09 | SVR | 0.79 | 0.07 |
| all | FF-MT | 0.70 | 0.08 | FF-MT | 0.57 | 0.08 | XGB | 0.33 | 0.11 |
""",
        }
    ]
    answer = """
### Valence
All: CCC 0.70, MAE 0.08.

### Dominance
Audio-text: CCC 0.79, MAE 0.07.
"""
    wrong_owner_group = answer.replace("CCC 0.79", "CCC 0.52")

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["multitarget-id"],
        answer_document_ids=["multitarget-id"],
    )
    rejected = validate_answer_claims(
        answer=wrong_owner_group,
        documents=documents,
        focus_document_ids=["multitarget-id"],
        answer_document_ids=["multitarget-id"],
    )

    assert supported.valid is True
    assert [(claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("0.7", ("valence",)),
        ("0.08", ("valence",)),
        ("0.79", ("dominance",)),
        ("0.07", ("dominance",)),
    ]
    assert rejected.valid is False
    assert rejected.unsupported_claims[0].value == "0.52"


def test_distributive_shared_metric_value_expands_across_named_dimensions() -> None:
    documents = [
        {
            "document_id": "dimensions-id",
            "filename": "dimensions.pdf",
            "content": """
| | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE |
|---|---:|---:|---:|---:|
| all | 0.70 | 0.08 | 0.57 | 0.08 |
""",
        }
    ]
    supported = validate_answer_claims(
        answer=(
            "Combining all modalities gives valence and arousal CCC 0.70 and "
            "0.57 respectively, both with MAE 0.08."
        ),
        documents=documents,
        focus_document_ids=["dimensions-id"],
        answer_document_ids=["dimensions-id"],
    )
    wrong_shared_value = validate_answer_claims(
        answer=(
            "Combining all modalities gives valence and arousal CCC 0.70 and "
            "0.57 respectively, both with MAE 0.09."
        ),
        documents=documents,
        focus_document_ids=["dimensions-id"],
        answer_document_ids=["dimensions-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in supported.checked_claims[-2:]] == [
        ("mae", "0.08", ("all", "valence")),
        ("mae", "0.08", ("all", "arousal")),
    ]
    assert wrong_shared_value.valid is False
    assert len(wrong_shared_value.unsupported_claims) == 2


def test_discourse_scaffolding_is_not_a_metric_qualifier() -> None:
    documents = [
        {
            "document_id": "fusion-id",
            "filename": "fusion-study.pdf",
            "content": (
                "Late Fusion achieved weighted F1-Score of approximately 78% "
                "on IEMOCAP using audio and text."
            ),
        }
    ]
    supported = validate_answer_claims(
        answer=(
            "Late Fusion has the best weighted F1 in this paper—approximately "
            "78% on IEMOCAP."
        ),
        documents=documents,
        focus_document_ids=["fusion-id"],
        answer_document_ids=["fusion-id"],
    )
    wrong_dataset = validate_answer_claims(
        answer="Late Fusion reaches weighted F1 78% on EMOV.",
        documents=documents,
        focus_document_ids=["fusion-id"],
        answer_document_ids=["fusion-id"],
    )

    assert supported.valid is True
    assert supported.checked_claims[0].qualifiers == ("iemocap",)
    assert wrong_dataset.valid is False


def test_explicit_paper_identity_on_metric_line_overrides_stale_section_context() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaNet reports F1 0.80.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "BetaNet reports F1 0.90.",
        },
    ]
    answer = """
### AlphaStudy
AlphaNet reports F1 0.80.
BetaStudy: BetaNet reports F1 0.90.
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )

    assert result.valid is True
    assert [claim.subjects for claim in result.checked_claims] == [
        ("ALPHASTUDY", "ALPHANET"),
        ("BETASTUDY", "BETANET"),
    ]


def test_multi_metric_table_section_rows_are_dataset_conditions_not_models() -> None:
    documents = [
        {
            "document_id": "sectioned-id",
            "filename": "sectioned.pdf",
            "content": """
| Model | Accuracy | F1 |
|---|---:|---:|
| Dataset North (English) | | |
| ModelX | 0.80 | 0.79 |
| Dataset South (French) | | |
| ModelX | 0.70 | 0.69 |
""",
        }
    ]
    correct = """
| Dataset | Model | Accuracy | F1 |
|---|---|---:|---:|
| Dataset North | ModelX | 0.80 | 0.79 |
| Dataset South | ModelX | 0.70 | 0.69 |
"""
    swapped = correct.replace(
        "Dataset North | ModelX | 0.80 | 0.79",
        "Dataset North | ModelX | 0.70 | 0.69",
    )

    supported = validate_answer_claims(
        answer=correct,
        documents=documents,
        focus_document_ids=["sectioned-id"],
    )
    rejected = validate_answer_claims(
        answer=swapped,
        documents=documents,
        focus_document_ids=["sectioned-id"],
    )

    assert supported.valid is True
    assert rejected.valid is False
    assert len(rejected.unsupported_claims) == 2


def test_semantic_columns_with_inline_metric_bundles_keep_row_and_dimension() -> None:
    documents = [
        {
            "document_id": "multitarget-id",
            "filename": "multitarget.pdf",
            "content": """
| Modality | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE | Dominance.CCC | Dominance.MAE |
|---|---:|---:|---:|---:|---:|---:|
| audio | 0.46 | 0.10 | 0.40 | 0.09 | 0.78 | 0.07 |
| all | 0.70 | 0.08 | 0.57 | 0.08 | 0.33 | 0.11 |
""",
        }
    ]
    answer = """
| Modality | Valence | Arousal | Dominance |
|---|---|---|---|
| audio | CCC 0.46, MAE 0.10 | CCC 0.40, MAE 0.09 | CCC 0.78, MAE 0.07 |
| all | CCC 0.70, MAE 0.08 | CCC 0.57, MAE 0.08 | CCC 0.33, MAE 0.11 |
"""
    cross_assigned = answer.replace(
        "audio | CCC 0.46, MAE 0.10",
        "audio | CCC 0.70, MAE 0.08",
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["multitarget-id"],
        answer_document_ids=["multitarget-id"],
    )
    rejected = validate_answer_claims(
        answer=cross_assigned,
        documents=documents,
        focus_document_ids=["multitarget-id"],
        answer_document_ids=["multitarget-id"],
    )

    assert supported.valid is True
    assert len(supported.checked_claims) == 12
    assert rejected.valid is False


def test_explicit_all_modalities_condition_disambiguates_repeated_values() -> None:
    documents = [
        {
            "document_id": "conditions-id",
            "filename": "conditions.pdf",
            "content": """
| Modality | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE |
|---|---:|---:|---:|---:|
| audio-text | 0.52 | 0.08 | 0.44 | 0.08 |
| all | 0.70 | 0.08 | 0.57 | 0.08 |
""",
        }
    ]
    supported = validate_answer_claims(
        answer=(
            "The study evaluates several combinations. With all modalities, "
            "valence and arousal reach CCC 0.70 and "
            "0.57 respectively, both with MAE 0.08."
        ),
        documents=documents,
        focus_document_ids=["conditions-id"],
        answer_document_ids=["conditions-id"],
    )
    wrong_row = validate_answer_claims(
        answer=(
            "With audio-text, valence and arousal reach CCC 0.70 and 0.57 "
            "respectively, both with MAE 0.08."
        ),
        documents=documents,
        focus_document_ids=["conditions-id"],
        answer_document_ids=["conditions-id"],
    )

    assert supported.valid is True
    assert wrong_row.valid is False


def test_wide_condition_headers_validate_equivalent_long_form_result_table() -> None:
    documents = [
        {
            "document_id": "transfer-id",
            "filename": "transfer.pdf",
            "content": """
| | LR.F1 | LR.UAR | ADA.F1 | ADA.UAR | DAL.F1 | DAL.UAR | Within.F1 | Within.UAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wavLM | 0.613 | 63.9 | 0.621 | 64.2 | 0.627 | 65.3 | 0.676 | 69.2 |
""",
        }
    ]
    answer = """
| Model | Condition | F1 (Macro) | UAR (%) |
|---|---|---:|---:|
| wavLM | LR | 0.613 | 63.9 |
| wavLM | ADA | 0.621 | 64.2 |
| wavLM | DAL | 0.627 | 65.3 |
| wavLM | Within | 0.676 | 69.2 |
"""
    wrong_condition = answer.replace(
        "wavLM | LR | 0.613 | 63.9",
        "wavLM | LR | 0.621 | 64.2",
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["transfer-id"],
    )
    rejected = validate_answer_claims(
        answer=wrong_condition,
        documents=documents,
        focus_document_ids=["transfer-id"],
    )

    assert supported.valid is True
    assert len(supported.checked_claims) == 8
    assert rejected.valid is False


def test_parenthesized_dimension_metric_pairs_bind_locally_in_conjunction() -> None:
    documents = [
        {
            "document_id": "dimensions-id",
            "filename": "dimensions.pdf",
            "content": """
| Modality | Model | Valence.CCC | Arousal.CCC |
|---|---|---:|---:|
| video | FF-MT | 0.59 | 0.47 |
""",
        }
    ]
    correct = (
        "In unimodal results, video with FF-MT is strongest for Valence "
        "(CCC 0.59) and Arousal (CCC 0.47)."
    )
    swapped = correct.replace("CCC 0.59", "CCC 0.47", 1).replace(
        "CCC 0.47).",
        "CCC 0.59).",
    )

    supported = validate_answer_claims(
        answer=correct,
        documents=documents,
        focus_document_ids=["dimensions-id"],
        answer_document_ids=["dimensions-id"],
    )
    rejected = validate_answer_claims(
        answer=swapped,
        documents=documents,
        focus_document_ids=["dimensions-id"],
        answer_document_ids=["dimensions-id"],
    )

    assert supported.valid is True
    assert [(claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("0.59", ("valence",)),
        ("0.47", ("arousal",)),
    ]
    assert rejected.valid is False


def test_explicit_metric_bullet_owner_replaces_previous_local_owner_context() -> None:
    documents = [
        {
            "document_id": "models-id",
            "filename": "models.pdf",
            "content": """
| Model | Test1.miF1 | Test1.maF1 | Test2.miF1 | Test2.maF1 |
|---|---:|---:|---:|---:|
| Fusion-Gate | 0.510 | 0.305 | 0.505 | 0.248 |
| Fusion-Stack | 0.501 | 0.304 | 0.503 | 0.254 |
| MFCC-Mamba | 0.489 | 0.299 | 0.481 | 0.246 |
""",
        }
    ]
    answer = """
- **Fusion-Gate:** Test1 miF1 0.510, maF1 0.305; Test2 miF1 0.505, maF1 0.248.
- **Fusion-Stack:** Test1 miF1 0.501, maF1 0.304; Test2 miF1 0.503, maF1 0.254.
- **MFCC-Mamba:** Test1 miF1 0.489, maF1 0.299; Test2 miF1 0.481, maF1 0.246.
"""
    cross_assigned = answer.replace(
        "Fusion-Stack:** Test1 miF1 0.501",
        "Fusion-Stack:** Test1 miF1 0.510",
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["models-id"],
        answer_document_ids=["models-id"],
    )
    rejected = validate_answer_claims(
        answer=cross_assigned,
        documents=documents,
        focus_document_ids=["models-id"],
        answer_document_ids=["models-id"],
    )

    assert supported.valid is True
    assert all(len(claim.subjects) == 1 for claim in supported.checked_claims)
    assert rejected.valid is False


def test_repeated_unlabelled_value_pair_cycles_only_after_an_aligned_metric_block() -> None:
    documents = [
        {
            "document_id": "tonal-id",
            "filename": "tonal-results.pdf",
            "content": """
| Model | UA (%) | WA (%) |
|---|---:|---:|
| Pitch-Net | 72.72 | 71.90 |
| Acoustic-Base | 61.74 | 62.37 |
""",
        }
    ]
    answer = (
        "Pitch-Net reaches 72.72% UA and 71.90% WA; "
        "Acoustic-Base reaches 61.74%/62.37%."
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["tonal-id"],
    )
    swapped = validate_answer_claims(
        answer=answer.replace("61.74%/62.37%", "62.37%/61.74%"),
        documents=documents,
        focus_document_ids=["tonal-id"],
    )
    unrelated_vector = validate_answer_claims(
        answer="Pitch-Net reports CCC 0.41/0.49/0.39 and F1 0.67.",
        documents=documents,
        focus_document_ids=["tonal-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.value) for claim in supported.checked_claims] == [
        ("ua", "72.72"),
        ("wa", "71.9"),
        ("ua", "61.74"),
        ("wa", "62.37"),
    ]
    assert swapped.valid is False
    assert unrelated_vector.valid is False


def test_structural_counts_and_retrieval_cutoffs_do_not_become_metric_results() -> None:
    for answer in (
        "On LSSED, every experiment runs 10 times before average accuracy is computed.",
        "Trên LSSED, mỗi thí nghiệm chạy 10 lần rồi lấy accuracy trung bình.",
        "The corpus has 3 documents and 20 questions. Metrics include Recall@5.",
    ):
        result = validate_answer_claims(answer=answer, documents=[])
        assert result.valid is True, answer
        assert result.reason == "no_metric_value_claims", answer
        assert result.checked_claims == [], answer


def test_flattened_repeated_metric_groups_infer_only_a_structural_first_group() -> None:
    documents = [
        {
            "document_id": "benchmark-id",
            "filename": "benchmark.pdf",
            "content": """
| Model | Efficiency.Params (M) | miF1 | maF1 | HL | Test2.miF1 | Test2.maF1 | Test2.HL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fusion-Gate | 3.07 | 0.510 | 0.305 | 0.179 | 0.505 | 0.248 | 0.183 |
""",
        }
    ]
    answer = """
| Model | Params (M) | Test1.miF1 | Test1.maF1 | Test1.HL | Test2.miF1 | Test2.maF1 | Test2.HL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fusion-Gate | 3.07 | 0.510 | 0.305 | 0.179 | 0.505 | 0.248 | 0.183 |
"""

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["benchmark-id"],
    )
    cross_assigned = validate_answer_claims(
        answer=answer.replace("0.510 | 0.305", "0.505 | 0.248"),
        documents=documents,
        focus_document_ids=["benchmark-id"],
    )

    assert supported.valid is True
    assert len(supported.checked_claims) == 7
    assert cross_assigned.valid is False


def test_prefix_dimensions_and_postfix_both_bind_to_each_measurement() -> None:
    documents = [
        {
            "document_id": "dimensions-id",
            "filename": "dimensions.pdf",
            "content": """
| Model | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE |
|---|---:|---:|---:|---:|
| Dim-Net | 0.70 | 0.08 | 0.57 | 0.08 |
""",
        }
    ]
    answer = (
        "Dim-Net reports valence CCC 0.70 and arousal CCC 0.57, "
        "with MAE 0.08 for both."
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["dimensions-id"],
    )
    swapped = validate_answer_claims(
        answer=answer.replace("CCC 0.70", "CCC 0.57", 1),
        documents=documents,
        focus_document_ids=["dimensions-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("ccc", "0.7", ("valence",)),
        ("ccc", "0.57", ("arousal",)),
        ("mae", "0.08", ("valence",)),
        ("mae", "0.08", ("arousal",)),
    ]
    assert swapped.valid is False


def test_ocr_jammed_metric_connector_is_repaired_without_relaxing_number_lexing() -> None:
    documents = [
        {
            "document_id": "ocr-id",
            "filename": "ocr-results.pdf",
            "content": "Fusion-Net reports mean CCCof0.610 in the ablation table.",
        }
    ]

    supported = validate_answer_claims(
        answer="Fusion-Net reports mean CCC 0.610.",
        documents=documents,
        focus_document_ids=["ocr-id"],
    )
    wrong = validate_answer_claims(
        answer="Fusion-Net reports mean CCC 0.620.",
        documents=documents,
        focus_document_ids=["ocr-id"],
    )

    assert supported.valid is True
    assert supported.checked_claims[0].qualifiers == ("avg",)
    assert wrong.valid is False


def test_ascii_percentage_range_supports_both_endpoints_but_not_an_invented_one() -> None:
    documents = [
        {
            "document_id": "router-id",
            "filename": "router.pdf",
            "content": "The router can save 20-40% tokens.",
        }
    ]

    supported = validate_answer_claims(
        answer="The router can save 20–40% tokens.",
        documents=documents,
        focus_document_ids=["router-id"],
    )
    wrong = validate_answer_claims(
        answer="The router can save 20–50% tokens.",
        documents=documents,
        focus_document_ids=["router-id"],
    )

    assert supported.valid is True
    assert [claim.value for claim in supported.checked_claims] == ["20", "40"]
    assert wrong.valid is False


def test_caption_dataset_context_and_trailing_dataset_phrase_are_shared_by_metric_bundle() -> None:
    documents = [
        {
            "document_id": "alignment-id",
            "filename": "alignment.pdf",
            "caption": "Comparison based on the IEMOCAP dataset.",
            "content": """
| Input | Model | WA | UA |
|---|---|---:|---:|
| S+T | Align-Net | 0.7239 | 0.7008 |
""",
        }
    ]
    answer = "Align-Net reaches WA 0.7239 and UA 0.7008 on IEMOCAP."

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["alignment-id"],
    )
    cross_assigned = validate_answer_claims(
        answer=answer.replace("UA 0.7008", "UA 0.7239"),
        documents=documents,
        focus_document_ids=["alignment-id"],
    )

    assert supported.valid is True
    assert [claim.qualifiers for claim in supported.checked_claims] == [
        ("iemocap",),
        ("iemocap",),
    ]
    assert cross_assigned.valid is False


def test_all_cardinality_condition_and_dimension_scope_apply_to_coordinated_metrics() -> None:
    documents = [
        {
            "document_id": "fusion-id",
            "filename": "fusion.pdf",
            "content": """
| | Valence.model | Valence.CCC | Valence.MAE | Arousal.model | Arousal.CCC | Arousal.MAE |
|---|---|---:|---:|---|---:|---:|
| all | FF-MT | 0.70 | 0.08 | FF-MT | 0.57 | 0.08 |
""",
        }
    ]
    answer = (
        "With all three modalities, FF-MT obtains valence CCC 0.70 and "
        "MAE 0.08, and arousal CCC 0.57 and MAE 0.08."
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["fusion-id"],
    )
    wrong_dimension = validate_answer_claims(
        answer=answer.replace("arousal CCC 0.57", "arousal CCC 0.70"),
        documents=documents,
        focus_document_ids=["fusion-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.value, claim.subjects, claim.qualifiers) for claim in supported.checked_claims] == [
        ("ccc", "0.7", ("FF-MT",), ("all", "valence")),
        ("mae", "0.08", ("FF-MT",), ("all", "valence")),
        ("ccc", "0.57", ("FF-MT",), ("all", "arousal")),
        ("mae", "0.08", ("FF-MT",), ("all", "arousal")),
    ]
    assert wrong_dimension.valid is False


def test_single_caption_metric_applies_to_qualifier_columns_without_paper_rules() -> None:
    documents = [
        {
            "document_id": "ablation-id",
            "filename": "temporal-ablation.pdf",
            "caption": (
                "Ablation configurations. Evaluation is based on CCC for "
                "activation, pleasantness, and their mean."
            ),
            "content": """
| Model configuration | Activation | Pleasantness | Mean |
|---|---:|---:|---:|
| Without Temporal-A and Temporal-B | 0.438 | 0.401 | 0.420 |
| Only Temporal-B | 0.550 | 0.593 | 0.572 |
| Ours (Temporal-A + Temporal-B) | 0.652 | 0.568 | 0.610 |
""",
        }
    ]
    answer = (
        "Temporal-A + Temporal-B reaches mean CCC 0.610, versus 0.420 "
        "without both and 0.572 with only Temporal-B."
    )

    supported = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["ablation-id"],
    )
    invented = validate_answer_claims(
        answer=answer.replace("0.610", "0.620"),
        documents=documents,
        focus_document_ids=["ablation-id"],
    )

    assert supported.valid is True
    assert [(claim.metric, claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("ccc", "0.61", ("avg",)),
        ("ccc", "0.42", ("avg",)),
        ("ccc", "0.572", ("avg",)),
    ]
    assert invented.valid is False


def test_unparsed_signal_guard_uses_local_pairing_and_ignores_rank_cutoff_vectors() -> None:
    for answer in (
        "The evaluation names Accuracy qualitatively and reports Recall@5/10 as cutoffs.",
        "The evaluation names Accuracy qualitatively and reports Recall at 5/10 as cutoffs.",
        "Có 20 questions. Metrics include Accuracy qualitatively and Recall@5/10.",
    ):
        result = validate_answer_claims(answer=answer, documents=[])
        assert result.valid is True, answer
        assert result.reason == "no_metric_value_claims", answer
        assert result.unparsed_signals == [], answer


def test_estimate_and_uncertainty_remain_distinct_measurements() -> None:
    documents = [
        {
            "document_id": "uncertainty-id",
            "filename": "uncertainty.pdf",
            "caption": "Comparison based on CCC across dimensions.",
            "content": """
| Model | Valence | Arousal |
|---|---:|---:|
| Fusion-Net | 0.617 ± 0.011 | 0.586 ± 0.022 |
""",
        }
    ]

    supported = validate_answer_claims(
        answer="Fusion-Net reports valence CCC 0.617 ± 0.011.",
        documents=documents,
        focus_document_ids=["uncertainty-id"],
    )
    invented_uncertainty = validate_answer_claims(
        answer="Fusion-Net reports valence CCC 0.617 ± 0.099.",
        documents=documents,
        focus_document_ids=["uncertainty-id"],
    )
    uncertainty_as_mean = validate_answer_claims(
        answer="Fusion-Net reports valence CCC 0.011.",
        documents=documents,
        focus_document_ids=["uncertainty-id"],
    )

    assert supported.valid is True
    assert [(claim.value, claim.qualifiers) for claim in supported.checked_claims] == [
        ("0.617", ("valence",)),
        ("0.011", ("valence", "uncertainty")),
    ]
    assert invented_uncertainty.valid is False
    assert uncertainty_as_mean.valid is False


def test_prose_parser_keeps_metric_schema_but_not_owners_across_sentences() -> None:
    claims = _extract_line_claims(
        "Pitch-Net đạt 72.72% UA / 71.90% WA, baseline 61.74% / 62.37%. "
        "Trên corpus khác, kết quả tương ứng là 85.40% / 84.77%, baseline "
        "63.72% / 63.66%."
    )

    assert [(claim.metric, claim.value) for claim in claims] == [
        ("ua", "72.72"),
        ("wa", "71.9"),
        ("ua", "61.74"),
        ("wa", "62.37"),
        ("ua", "85.4"),
        ("wa", "84.77"),
        ("ua", "63.72"),
        ("wa", "63.66"),
    ]
    assert claims[0].subjects == ("PITCH-NET",)
    assert claims[4].subjects == ()
    assert claims[6].subjects == ("BASELINE",)


def test_prose_parser_aligns_open_dimension_and_split_vectors() -> None:
    dimension_claims = _extract_line_claims(
        "Fusion-Net đạt CCC cho V/A/D lần lượt 0.632/0.680/0.601, "
        "trung bình 0.638."
    )
    split_claims = _extract_line_claims(
        "Fusion-Net đạt miF1 lần lượt 0.510 và 0.514 trên Test1/Test2."
    )

    assert [(claim.value, claim.qualifiers) for claim in dimension_claims] == [
        ("0.632", ("valence",)),
        ("0.68", ("arousal",)),
        ("0.601", ("dominance",)),
        ("0.638", ("avg",)),
    ]
    assert [(claim.value, claim.qualifiers) for claim in split_claims] == [
        ("0.51", ("test1", "micro")),
        ("0.514", ("test2", "micro")),
    ]


def test_prose_parser_distinguishes_counts_sizes_and_clause_local_owners() -> None:
    claims = _extract_line_claims(
        "Phân loại 4 lớp; đánh giá bằng UA và WA. "
        "Fusion-Stack (Ours), 3.02M: miF1 0.501. "
        "MemoCMT có WAR 0.649, còn Crab đạt UAR 0.5565."
    )

    assert [(claim.metric, claim.value, claim.subjects) for claim in claims] == [
        ("f1", "0.501", ("PROPOSED-MODEL",)),
        ("war", "0.649", ("MEMOCMT",)),
        ("uar", "0.5565", ("CRAB",)),
    ]


def test_semicolon_starts_a_new_owner_scope_without_breaking_metric_schema() -> None:
    claims = _extract_line_claims(
        "Corpus-A: 72.72 UA / 71.90 WA, baseline 61.74 / 62.37; "
        "Chinese: 85.40 / 84.77."
    )

    assert [(claim.metric, claim.value) for claim in claims] == [
        ("ua", "72.72"),
        ("wa", "71.9"),
        ("ua", "61.74"),
        ("wa", "62.37"),
        ("ua", "85.4"),
        ("wa", "84.77"),
    ]
    assert claims[-1].subjects == ()


def test_document_metric_definition_applies_across_canonical_sources() -> None:
    documents = [
        {
            "document_id": "paper-a",
            "filename": "paper-a.pdf",
            "caption": "Main benchmark results",
            "content": """
| Model | Acc | F1 |
|---|---:|---:|
| Fusion-Net | 75.86 | 76.31 |
""",
        },
        {
            "document_id": "paper-a",
            "filename": "paper-a.pdf",
            "content": "Evaluation metrics are Accuracy (Acc) and Macro F1-score (F1).",
        },
    ]

    macro = validate_answer_claims(
        answer="Fusion-Net reports Macro F1 76.31.",
        documents=documents,
        focus_document_ids=["paper-a"],
    )
    micro = validate_answer_claims(
        answer="Fusion-Net reports Micro F1 76.31.",
        documents=documents,
        focus_document_ids=["paper-a"],
    )

    assert macro.valid is True
    assert micro.valid is False


def test_modality_compositions_normalize_abbreviations_without_paper_rules() -> None:
    documents = [
        {
            "document_id": "fusion-paper",
            "filename": "fusion-paper.pdf",
            "content": """
| Model | Input | WA | UA |
|---|---|---:|---:|
| Fusion-Net | S+T | 0.7239 | 0.7008 |
""",
        }
    ]

    supported = validate_answer_claims(
        answer="""
| Model | Input | WA | UA |
|---|---|---:|---:|
| Fusion-Net | Speech + Text | 0.7239 | 0.7008 |
""",
        documents=documents,
        focus_document_ids=["fusion-paper"],
    )
    wrong_modality = validate_answer_claims(
        answer="""
| Model | Input | WA | UA |
|---|---|---:|---:|
| Fusion-Net | Speech + Video | 0.7239 | 0.7008 |
""",
        documents=documents,
        focus_document_ids=["fusion-paper"],
    )

    assert supported.valid is True
    assert wrong_modality.valid is False


def test_partial_inline_label_uses_complete_composite_header_schema() -> None:
    documents = [
        {
            "document_id": "results-paper",
            "filename": "results-paper.pdf",
            "content": """
| Model | Acc / F1 |
|---|---:|
| Fusion-Net | 75.86 / 76.31 |
""",
        }
    ]
    answer = """
| Model | Acc / F1 |
|---|---:|
| Fusion-Net | **75.86**, F1 **76.31** |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["results-paper"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
    ]


def test_qualitative_metric_after_final_value_does_not_globalize_last_split() -> None:
    documents = [
        {
            "document_id": "split-paper",
            "filename": "split-paper.pdf",
            "content": """
| Model | Test1.miF1 | Test2.miF1 |
|---|---:|---:|
| MulT | 0.508 | 0.526 |
""",
        }
    ]
    answer = (
        "Baseline MulT (Audio-only MFCC) đạt miF1 0.508 trên Test1 và "
        "0.526 trên Test2, tức vẫn cạnh tranh mạnh về miF1."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["split-paper"],
    )

    assert result.valid is True
    assert [(claim.value, set(claim.qualifiers)) for claim in result.checked_claims] == [
        ("0.508", {"test1", "micro"}),
        ("0.526", {"test2", "micro"}),
    ]


def test_hyphenated_mixed_case_model_identity_is_not_split() -> None:
    documents = [
        {
            "document_id": "compact-paper",
            "filename": "compact-paper.pdf",
            "content": "MobileNet-V4-Small has 3.12M parameters.",
        }
    ]

    result = validate_answer_claims(
        answer="MobileNet-V4-Small has 3.12M parameters.",
        documents=documents,
        focus_document_ids=["compact-paper"],
    )

    assert result.valid is True
    assert result.checked_claims[0].subjects == ("MOBILENET-V4-SMALL",)


def test_leading_row_qualifier_does_not_block_unique_model_owner() -> None:
    documents = [
        {
            "document_id": "vad-paper",
            "filename": "vad-paper.pdf",
            "content": """
| Modality | Model | Valence.CCC | Valence.MAE | Arousal.CCC | Arousal.MAE |
|---|---|---:|---:|---:|---:|
| audio-video | FF-MT | 0.67 | 0.08 | 0.54 | 0.09 |
""",
        }
    ]
    answer = (
        "audio-video — FF-MT: Valence CCC 0.67, MAE 0.08; "
        "Arousal CCC 0.54, MAE 0.09."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["vad-paper"],
    )

    assert result.valid is True
    assert [claim.subjects for claim in result.checked_claims] == [
        ("FF-MT",),
        ("FF-MT",),
        ("FF-MT",),
        ("FF-MT",),
    ]


def test_missing_first_metric_label_resolves_only_from_unique_canonical_schema() -> None:
    documents = [
        {
            "document_id": "schema-paper",
            "filename": "schema-paper.pdf",
            "content": """
| Model | Acc | F1 |
|---|---:|---:|
| Fusion-Net | 75.86 | 76.31 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="**75.86**, F1 **76.31**",
        documents=documents,
        focus_document_ids=["schema-paper"],
        answer_document_ids=["schema-paper"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value) for claim in result.checked_claims] == [
        ("accuracy", "75.86"),
        ("f1", "76.31"),
    ]


def test_missing_metric_label_remains_fail_closed_when_value_is_ambiguous() -> None:
    documents = [
        {
            "document_id": "ambiguous-paper",
            "filename": "ambiguous-paper.pdf",
            "content": """
| Model | Acc | F1 |
|---|---:|---:|
| Fusion-Net | 75.86 | 75.86 |
""",
        }
    ]

    result = validate_answer_claims(
        answer="75.86, F1 75.86",
        documents=documents,
        focus_document_ids=["ambiguous-paper"],
    )

    assert result.valid is False


def test_pdf_soft_wrap_preserves_parameter_comparison_ownership() -> None:
    documents = [
        {
            "document_id": "edge-paper",
            "filename": "edge-paper.pdf",
            "content": """
Wav2Small is intended for low-resource hardware, as it has only 72 K

parameters vs 3.12 M parameters for MobileNet-V4-Small.
""",
        }
    ]
    answer = (
        "Wav2Small has 72K parameters, versus 3.12M parameters for "
        "MobileNet-V4-Small."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["edge-paper"],
    )

    assert result.valid is True
    assert [(claim.value, claim.subjects) for claim in result.checked_claims] == [
        ("72000", ("WAV2SMALL",)),
        ("3120000", ("MOBILENET-V4-SMALL",)),
    ]


def test_proposed_component_role_can_be_named_without_paper_specific_alias() -> None:
    documents = [
        {
            "document_id": "teacher-paper",
            "filename": "teacher-paper.pdf",
            "content": (
                "The proposed teacher also sets a new state of the art for "
                "valence on the Wild Speech dataset with a CCC of 0.676."
            ),
        }
    ]

    result = validate_answer_claims(
        answer="Teacher đạt Valence CCC 0.676 trên Wild Speech.",
        documents=documents,
        focus_document_ids=["teacher-paper"],
    )

    assert result.valid is True


def test_deployment_size_and_parameter_bundle_keeps_units_and_owners_distinct() -> None:
    documents = [
        {
            "document_id": "small-paper",
            "filename": "Wav2Small.pdf",
            "content": """
Wav2Small with a size of only 120 KB when quantised for the ONNX runtime

is intended for low-resource hardware, as it has only 72 K

parameters vs 3.12 M parameters for MobileNet-V4-Small.
""",
        }
    ]
    answer = (
        "Bản quantized cho ONNX chỉ khoảng 120 KB, với 72K tham số, "
        "nhỏ hơn 3.12M tham số của MobileNet-V4-Small."
    )

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["small-paper"],
    )

    assert result.valid is True
    assert [(claim.metric, claim.value, claim.subjects) for claim in result.checked_claims] == [
        ("storage_bytes", "120000", ()),
        ("parameters", "72000", ()),
        ("parameters", "3120000", ("MOBILENET-V4-SMALL",)),
    ]


def test_each_quantitative_bullet_owns_its_same_line_metrics() -> None:
    documents = [
        {
            "document_id": "fusion-paper",
            "filename": "fusion-paper.pdf",
            "content": """
| Model | WA | UA |
|---|---:|---:|
| BiAtt-GMU | 0.7188 | 0.6654 |
| BiAtt-TFL | 0.7101 | 0.6627 |
| BiAtt-Concat | 0.7026 | 0.6561 |
| Att-LSTM | 0.6974 | 0.6401 |
""",
        }
    ]
    answer = """
- **BiAtt-GMU — IEMOCAP:** WA 0.7188, UA 0.6654
- **BiAtt-TFL — IEMOCAP:** WA 0.7101, UA 0.6627
- **BiAtt-Concat — IEMOCAP:** WA 0.7026, UA 0.6561
- **Att-LSTM — IEMOCAP:** WA 0.6974, UA 0.6401
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["fusion-paper"],
        answer_document_ids=["fusion-paper"],
    )

    assert result.valid is True
    assert [claim.subjects for claim in result.checked_claims] == [
        ("BIATT-GMU",),
        ("BIATT-GMU",),
        ("BIATT-TFL",),
        ("BIATT-TFL",),
        ("BIATT-CONCAT",),
        ("BIATT-CONCAT",),
        ("ATT-LSTM",),
        ("ATT-LSTM",),
    ]


def test_parameter_count_model_tag_is_not_parsed_as_byte_size() -> None:
    result = validate_answer_claims(
        answer=(
            "Tauri 2 + React UI, FastAPI backend, LangGraph agent, Ollama "
            "with `qwen3.5:4b`, Chroma vectors, and SQLite metadata."
        ),
        documents=[],
    )

    assert result.valid is True
    assert result.checked_claims == []


def test_composite_dataset_split_cell_matches_atomic_evidence_context() -> None:
    documents = [
        {
            "document_id": "comparison-paper",
            "filename": "comparison-paper.pdf",
            "caption": "Comparison based on the IEMOCAP dataset.",
            "content": """
| Input | Model | WA | UA |
|---|---|---:|---:|
| S+T | Fusion-Net | 0.7239 | 0.7008 |
""",
        }
    ]
    answer = """
| Model | Dataset / split | Input | WA | UA |
|---|---|---|---:|---:|
| Fusion-Net | IEMOCAP, 5-fold cross-validation, leave-one-session-out | S+T | 0.7239 | 0.7008 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["comparison-paper"],
        answer_document_ids=["comparison-paper"],
    )

    assert result.valid is True


def test_composite_qualifier_overlap_does_not_hide_conflicting_condition() -> None:
    documents = [
        {
            "document_id": "comparison-paper",
            "filename": "comparison-paper.pdf",
            "content": """
| Dataset / split | Model | F1 |
|---|---|---:|
| IEMOCAP Test-1 | Fusion-Net | 0.80 |
""",
        }
    ]
    answer = """
| Model | Dataset / split | F1 |
|---|---|---:|
| Fusion-Net | IEMOCAP Test-2 | 0.80 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["comparison-paper"],
    )

    assert result.valid is False


def test_row_metric_dimension_matrix_matches_canonical_column_metric_table() -> None:
    documents = [
        {
            "document_id": "vad-paper",
            "filename": "vad-paper.pdf",
            "content": """
| | Valence.model | Valence.CCC | Valence.MAE | Arousal.model | Arousal.CCC | Arousal.MAE | Dominance.model | Dominance.CCC | Dominance.MAE |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| audio-text | FF-MT | 0.52 | 0.10 | FF-MT | 0.44 | 0.09 | SVR | 0.79 | 0.07 |
| all | FF-MT | 0.70 | 0.08 | FF-MT | 0.57 | 0.08 | XGB | 0.33 | 0.11 |
""",
        }
    ]
    answer = """
| Paper | Scheme or modalities | V | A | D |
|---|---|---:|---:|---:|
| VAD paper | all — FF-MT/XGB, CCC | 0.70 | 0.57 | 0.33 |
| VAD paper | all — FF-MT/XGB, MAE | 0.08 | 0.08 | 0.11 |
| VAD paper | audio-text — FF-MT/SVR, CCC | 0.52 | 0.44 | 0.79 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["vad-paper"],
        answer_document_ids=["vad-paper"],
    )

    assert result.valid is True
    assert len(result.checked_claims) == 9
    assert {claim.qualifiers for claim in result.checked_claims[:3]} == {
        ("all", "valence"),
        ("all", "arousal"),
        ("all", "dominance"),
    }


def test_pearson_r_dimension_matrix_matches_parenthesized_r_headers() -> None:
    documents = [
        {
            "document_id": "transfer-paper",
            "filename": "transfer-paper.pdf",
            "caption": "Performance evaluated on the EmoBank dataset.",
            "content": """
| Model | Scheme | V (r) | A (r) | D (r) |
|---|---|---:|---:|---:|
| Ours, SemEval | Zero-shot | 0.715 | 0.319 | 0.308 |
""",
        }
    ]
    answer = """
| Model | Scheme | V | A | D |
|---|---|---:|---:|---:|
| Ours, SemEval | Zero-shot, EmoBank Pearson r | 0.715 | 0.319 | 0.308 |
"""

    result = validate_answer_claims(
        answer=answer,
        documents=documents,
        focus_document_ids=["transfer-paper"],
    )

    assert result.valid is True
    assert [claim.metric for claim in result.checked_claims] == [
        "pearson_r",
        "pearson_r",
        "pearson_r",
    ]
