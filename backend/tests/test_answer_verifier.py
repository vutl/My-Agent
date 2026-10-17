from app.services.answer_verifier import (
    apply_corrections, build_verify_prompt, find_unsourced_numbers, parse_corrections,
)

DOCS = [{"document_id": "a", "filename": "report.md", "content": (
    "| Region | Q1 2024 Revenue ($M) | Q2 2024 Revenue ($M) |\n|---|---|---|\n"
    "| North America | 412.5 | 438.2 |\n| Europe | 287.1 | 295.6 |\n")}]


def test_sourced_numbers_are_not_flagged():
    report = find_unsourced_numbers("North America earned 438.2 in Q2 2024 [S1].", DOCS)
    assert report.clean and report.checked >= 1


def test_fabricated_number_is_flagged_with_its_sentence():
    report = find_unsourced_numbers("North America earned 452.0 million in Q2.", DOCS)
    assert [item.value for item in report.flagged] == ["452.0"]
    assert "452.0" in report.flagged[0].sentence


def test_years_citations_and_references_are_not_claims():
    report = find_unsourced_numbers("In 2024 see Table 3 and Figure 2 [S1][2].", DOCS)
    assert report.clean and report.checked == 0


def test_vietnamese_decimal_comma_matches_evidence():
    docs = [{"content": "Tỷ lệ đáp ứng là 64,2%."}]
    assert find_unsourced_numbers("Tỷ lệ đáp ứng đạt 64,2%.", docs).clean
    assert not find_unsourced_numbers("Tỷ lệ đáp ứng đạt 65,9%.", docs).clean


def test_apply_corrections_touches_only_flagged_sentences():
    answer = "North America earned 452.0 in Q2.\n\nEurope earned 295.6 in Q2."
    fixed = apply_corrections(answer, [{"sentence": "North America earned 452.0 in Q2.",
                                        "action": "rewrite",
                                        "rewritten": "North America earned 438.2 in Q2."}])
    assert fixed == "North America earned 438.2 in Q2.\n\nEurope earned 295.6 in Q2."
    removed = apply_corrections(answer, [{"sentence": "North America earned 452.0 in Q2.", "action": "remove"}])
    assert removed == "Europe earned 295.6 in Q2."


def test_unknown_sentence_in_corrections_is_ignored():
    answer = "Europe earned 295.6."
    assert apply_corrections(answer, [{"sentence": "not in answer", "action": "remove"}]) == answer


def test_parse_corrections_tolerates_wrapping_prose():
    assert parse_corrections('{"corrections": [{"sentence": "a", "action": "keep"}]}')[0]["action"] == "keep"
    wrapped = 'Here you go: {"corrections": []} thanks'
    assert parse_corrections(wrapped) == []
    assert parse_corrections("not json") == []


def test_verify_prompt_lists_flagged_sentences_and_numbered_sources():
    report = find_unsourced_numbers("North America earned 452.0 in Q2.", DOCS)
    prompt = build_verify_prompt(report.flagged, DOCS)
    assert "452.0" in prompt and "[S1]" in prompt


def test_list_markers_and_small_counts_are_not_claims():
    assert find_unsourced_numbers("Kết quả gồm các mục sau.\n\n4. Tổng kết.", []).clean
    assert find_unsourced_numbers("Có 4 bước chính.", []).clean
    # Decimals and percentages stay claims even when small.
    assert not find_unsourced_numbers("Tỷ lệ đạt 3,5%.", []).clean


def test_numeric_answer_units_skip_reference_numbers() -> None:
    from app.services.answer_verifier import numeric_answer_units

    answer = (
        "Theo Bảng 2, mô hình dùng semantic và prosody.\n\n"
        "| Model | F1 |\n|---|---|\n| ASPIRE | 80.4 |\n\n"
        "Doanh thu quý 2 là 1.318 tỷ đồng. Xem Figure 4 và Eq. (3)."
    )
    units = numeric_answer_units(answer)
    assert "| ASPIRE | 80.4 |" in units
    assert "Doanh thu quý 2 là 1.318 tỷ đồng." in units
    assert not any("Bảng 2" in unit or "Figure 4" in unit for unit in units)


def test_verify_prompt_keeps_late_table_rows_and_relevant_sources_first():
    from app.services.answer_verifier import UnsourcedNumber

    rows = "\n".join(f"| Model {index} | {60 + index / 10:.2f} |" for index in range(400))
    documents = [
        {"filename": "other.pdf", "content": "unrelated " * 3000},
        {"filename": "paper.pdf", "content": "| Model | WA |\n|---|---|\n" + rows},
    ]
    prompt = build_verify_prompt([UnsourcedNumber(value="", sentence="| Model 399 | 99.90 |")], documents)
    assert "| Model 399 | 99.90 |" in prompt.split("Sources")[1]
    assert prompt.index("paper.pdf") < prompt.index("other.pdf")


def test_mass_removal_verdict_is_ignored():
    answer = "\n".join(f"| M{index} | {70 + index}.1 |" for index in range(4))
    corrections = [{"sentence": f"| M{index} | {70 + index}.1 |", "action": "remove"} for index in range(4)]
    assert apply_corrections(answer, corrections) == answer


def test_verify_prompt_uses_the_expanded_excerpt_the_answer_saw():
    from app.services.answer_verifier import UnsourcedNumber

    documents = [{
        "filename": "CMDM.pdf",
        "content": "Table 3: Model architecture setting.",
        "expanded_content": "Table 3: Model architecture setting.\n\n| L | Ours | 65.4% | 64.8% |",
    }]
    prompt = build_verify_prompt([UnsourcedNumber(value="", sentence="| L | Ours | 65.4% | 64.8% |")], documents)
    assert "| L | Ours | 65.4% | 64.8% |" in prompt.split("Sources")[1]


def test_list_markers_and_diagrams_are_never_units_or_rewritten():
    from app.services.answer_verifier import flagged_answer_units, numeric_answer_units

    answer = (
        "Paper có hai figure chính:\n\n"
        "1. **Figure 1 — Pitch-fusion**: đạt WA 71.9 trên ViSEC.\n"
        "```text\nWaveform -> Wav2Vec 2.0 ---> Cross Attention\n```\n"
        "2. **Figure 2 — pipeline** gồm 3 bước."
    )
    units = numeric_answer_units(answer)
    assert "1." not in units and "2." not in units
    assert not any("Wav2Vec 2.0" in unit for unit in units)
    assert any("71.9" in unit for unit in units)
    corrupted = apply_corrections(answer, [
        {"sentence": "1.", "action": "remove"},
        {"sentence": "2.", "action": "remove"},
    ])
    assert corrupted == answer.strip()
    assert flagged_answer_units("Model 2.0 is new.", ["2.0"]) == ["Model 2.0 is new."]


def test_echoed_list_prefix_still_matches_the_unit():
    answer = "Kết quả:\n\n- ASPIRE đạt F1 91.7% trên IEMOCAP."
    fixed = apply_corrections(answer, [{
        "sentence": "- - ASPIRE đạt F1 91.7% trên IEMOCAP.", "action": "rewrite",
        "rewritten": "ASPIRE đạt F1 76.31% trên IEMOCAP.",
    }])
    assert "76.31%" in fixed and "91.7" not in fixed


def test_full_document_sources_contribute_windows_around_checked_numbers():
    from app.services.answer_verifier import UnsourcedNumber

    text = "filler " * 4000 + "The teacher has 483.9M parameters. " + "filler " * 4000
    prompt = build_verify_prompt(
        [UnsourcedNumber(value="", sentence="Teacher có 483.9M tham số.")],
        [{"filename": "wav2small.pdf", "content": text, "full_document": True}],
    )
    sources = prompt.split("Sources")[1]
    assert "483.9M parameters" in sources and len(sources) < 4_000


def test_whole_documents_reach_the_verifier_after_many_excerpts():
    from app.services.answer_verifier import UnsourcedNumber

    excerpts = [{"filename": f"chunk{index}.pdf", "content": f"unrelated value {index}.5"} for index in range(14)]
    whole = {"filename": "lecture.pdf", "content": "intro " * 3000 + "The triage model uses 4,000 exams.", "full_document": True}
    prompt = build_verify_prompt([UnsourcedNumber(value="", sentence="It was trained on 4,000 exams.")], [*excerpts, whole])
    assert "4,000 exams" in prompt.split("Sources")[1]


def test_model_names_with_numbers_are_names_not_claims():
    from app.services.answer_verifier import numeric_answer_units

    answer = (
        "| CheXNet | DenseNet-121 | ImageNet |\n"
        "- The triage model uses an ImageNet-initialized ResNet-18.\n"
        "GPT-3.5 reached 75.8% accuracy."
    )
    units = numeric_answer_units(answer)
    assert units == ["GPT-3.5 reached 75.8% accuracy."]
    assert find_unsourced_numbers("CheXNet uses DenseNet-121 and COVID-19 data.", []).clean


def test_removal_of_a_line_whose_numbers_are_in_the_sources_is_dropped():
    from app.services.answer_verifier import keep_sourced_removals

    corrections = [
        {"sentence": "North America earned 438.2 in Q2.", "action": "remove"},
        {"sentence": "Europe earned 999.9 in Q2.", "action": "remove"},
        {"sentence": "Europe earned 287.1 in Q2.", "action": "rewrite", "rewritten": "Europe earned 287.1 in Q1."},
    ]
    kept = keep_sourced_removals(corrections, DOCS)
    assert [item["sentence"] for item in kept] == ["Europe earned 999.9 in Q2.", "Europe earned 287.1 in Q2."]


def test_lines_with_only_years_are_not_units_and_years_do_not_drive_windows():
    from app.services.answer_verifier import UnsourcedNumber, numeric_answer_units

    assert numeric_answer_units("No, the two articles do not agree for 2024.") == []
    text = ("In 2024 the economy grew. " * 800) + "| 2024 | 6,0% | 470 |"
    prompt = build_verify_prompt(
        [UnsourcedNumber(value="", sentence="| GDP growth 2024 | 6.0% | US$470 billion |")],
        [{"filename": "vi.docx", "content": text, "full_document": True}],
    )
    assert "| 2024 | 6,0% | 470 |" in prompt.split("Sources")[1]


def test_rewrite_that_only_drops_sourced_values_is_ignored():
    from app.services.answer_verifier import keep_sourced_removals

    corrections = [{"sentence": "| Europe | 287.1 | 295.6 |", "action": "rewrite",
                    "rewritten": "| Europe | Not established by the sources | - |"}]
    assert keep_sourced_removals(corrections, DOCS) == []


def test_long_document_contributes_sections_sharing_rare_query_words_in_order():
    from app.services.rag_service import _query_focused_sections

    blocks = ["Title page of the regional economic update."]
    blocks += [f"Section {index}: the region and the year 2026 in general terms. " * 20 for index in range(60)]
    blocks.insert(30, "In Viet Nam, the economy grew 8.3 percent in the first half of 2026.")
    text = _query_focused_sections(blocks, "How fast did Viet Nam grow in the first half of 2026?", 3_000)
    assert text.startswith("Title page") and "8.3 percent in the first half" in text
    assert len(text) <= 3_200


def _gdp_documents():
    filler_a = "Retail turnover and port traffic were broadly steady across the period. " * 320
    filler_b = "Tourism receipts and rainfall stayed close to seasonal norms. " * 380
    alpha = filler_a + "GDP growth accelerated to 8.2% in the first half of 2026.\n" + filler_a
    beta = filler_b + "GDP grew 8.3 percent in the first half of 2026.\n" + filler_b
    return [
        {"filename": "alpha_outlook.pdf", "content": alpha, "full_document": True},
        {"filename": "beta_update.pdf", "content": beta, "full_document": True},
    ]


_ALPHA_LINE = "Alpha Outlook reports that GDP grew by 8.3% in the first half of 2026."
_BETA_LINE = "Beta Update reports that GDP grew by 8.3% in the first half of 2026."


def test_verify_prompt_shows_the_named_documents_own_value_with_its_label():
    from app.services.answer_verifier import UnsourcedNumber

    prompt = build_verify_prompt([UnsourcedNumber(value="", sentence=_ALPHA_LINE)], _gdp_documents())
    sources = prompt.split("Sources (S1..Sn)", 1)[1]
    assert "8.2% in the first half" in sources
    assert "alpha_outlook.pdf" in sources[: sources.index("8.2% in the first half")].rsplit("[S", 1)[1]
    assert "beta_update.pdf" in sources[: sources.index("8.3 percent")].rsplit("[S", 1)[1]
    assert len(sources) < 24_000
    assert "misattributed" in prompt


def test_misattributed_value_rewrite_survives_the_guard_and_is_applied():
    from app.services.answer_verifier import keep_sourced_removals

    fixed_line = _ALPHA_LINE.replace("8.3%", "8.2%")
    corrections = [{"sentence": _ALPHA_LINE, "action": "rewrite", "rewritten": fixed_line}]
    kept = keep_sourced_removals(corrections, _gdp_documents())
    assert kept == corrections
    answer = f"{_ALPHA_LINE}\n\n{_BETA_LINE}"
    assert apply_corrections(answer, kept) == f"{fixed_line}\n\n{_BETA_LINE}"


def test_unit_without_claim_numbers_does_not_break_the_prompt():
    from app.services.answer_verifier import UnsourcedNumber
    from app.services.numeric_grounding import _number_candidates

    header = "| Source | First-half 2026 growth | Full-year 2026 forecast |"
    prompt = build_verify_prompt([UnsourcedNumber(value="", sentence=header)], _gdp_documents())
    assert header in prompt
    assert _number_candidates("") == frozenset() and _number_candidates("%") == frozenset()


def test_answer_unit_context_gives_section_heading_and_table_header():
    from app.services.answer_verifier import answer_unit_context

    answer = (
        "### alpha_outlook.pdf\n\nGDP grew by 8.3% in the first half.\n\n"
        "| Source | Growth |\n|---|---|\n| Alpha | 8.3% |"
    )
    assert answer_unit_context(answer, "GDP grew by 8.3% in the first half.") == "section: alpha_outlook.pdf"
    row = answer_unit_context(answer, "| Alpha | 8.3% |")
    assert "table header: | Source | Growth |" in row
    assert "table header" not in answer_unit_context(answer, "| Source | Growth |")
    assert answer_unit_context("Plain 8.3% line here.", "Plain 8.3% line here.") == ""


def test_verify_prompt_lists_unit_context_and_question():
    from app.services.answer_verifier import UnsourcedNumber

    prompt = build_verify_prompt(
        [UnsourcedNumber(value="", sentence=_ALPHA_LINE, context="section: alpha_outlook.pdf")],
        _gdp_documents(), question="What did each report say about growth?",
    )
    assert f"- {_ALPHA_LINE}\n  (section: alpha_outlook.pdf)" in prompt
    assert "Question: What did each report say about growth?" in prompt
