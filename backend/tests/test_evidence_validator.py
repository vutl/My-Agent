from app.services.evidence_validator import validate_answer_claims, validate_retrieval_evidence


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
    for header in ("P (M)", "Params(M)", "Trainable Params (M)."):
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
