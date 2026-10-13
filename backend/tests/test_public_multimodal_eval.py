from __future__ import annotations

import asyncio
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_public_multimodal_loaders_flatten_real_question_units() -> None:
    library = _load_script("public_multimodal_eval_lib")
    mmlong = library.load_mmlong_cases()
    spiqa = library.load_spiqa_test_c_cases()

    assert len(mmlong) == 1_082
    assert len(spiqa) == 493
    assert len({case.case_id for case in mmlong + spiqa}) == 1_575
    assert all(case.question and case.answers[0] for case in mmlong + spiqa)
    assert all(case.source_path and case.source_path.is_file() for case in mmlong)


def test_mmlong_v2_loader_uses_corrected_release_without_changing_v1_default() -> None:
    library = _load_script("public_multimodal_eval_lib")
    v1 = library.load_mmlong_cases()
    v2 = library.load_mmlong_cases(version="v2")

    assert len(v1) == 1_082
    assert len(v2) == 1_071
    assert v2[0].case_id.startswith("mmlongbench-doc-v2:")
    assert all(case.source_path and case.source_path.is_file() for case in v2)
    assert {(case.document_id, case.question) for case in v2}.__len__() == len(v2)


def test_public_multimodal_sampling_is_order_independent_and_balanced() -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = library.load_mmlong_cases()
    forward = library.stable_balanced_sample(cases, limit=12)
    reverse = library.stable_balanced_sample(reversed(cases), limit=12)

    assert [case.case_id for case in forward] == [case.case_id for case in reverse]
    assert len({case.stratum for case in forward}) >= 4
    assert len({case.document_id for case in forward}) >= 4


def test_public_eval_keeps_candidate_and_final_page_recall_distinct() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")

    assert runner._page_recall([1, 2], [2, 3]) == 0.5
    assert runner._page_recall([1, 2, 3], [2, 3]) == 1.0
    assert runner._page_recall([], []) is None


def test_visual_page_reservation_uses_declared_modality_not_provider_overretrieve() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case_type = runner.BenchmarkCase
    plain = case_type(
        suite="mmlongbench-doc",
        case_id="plain",
        document_id="d1",
        title="plain",
        question="When was the complaint filed?",
        answers=("2020-01-01",),
        answer_format="Str",
    )
    visual = case_type(
        suite="mmlongbench-doc",
        case_id="visual",
        document_id="d1",
        title="visual",
        question="What does the diagram show?",
        answers=("a process",),
        answer_format="Str",
        evidence_sources=("Figure",),
    )

    assert runner._case_requires_visual_evidence(plain) is False
    assert runner._case_requires_visual_evidence(visual) is True

    metadata_free_visual = case_type(
        suite="mmlongbench-doc",
        case_id="visual-question",
        document_id="d1",
        title="visual-question",
        question="What is shown in the picture on slide 6?",
        answers=("Not answerable",),
        answer_format="None",
    )
    assert runner._case_requires_visual_evidence(metadata_free_visual) is True


def test_visual_routing_does_not_depend_on_gold_artifact_annotations() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    base = runner.BenchmarkCase(
        suite="spiqa-test-c",
        case_id="annotation-free",
        document_id="d1",
        title="d1",
        question="What metrics were used?",
        answers=("accuracy",),
        answer_format="free_form",
        full_text="# paper\nmetrics were used\n",
    )
    annotated = runner.BenchmarkCase(
        suite=base.suite,
        case_id="annotation-heavy",
        document_id=base.document_id,
        title=base.title,
        question=base.question,
        answers=base.answers,
        answer_format=base.answer_format,
        referred_artifacts=("4-Table1-1.png",),
        evidence_sources=("Table", "Figure"),
        metadata={"table_in_evidence": True, "figure_in_evidence": True},
        full_text=base.full_text,
    )
    assert runner._case_requires_visual_evidence(base) is False
    assert runner._case_requires_visual_evidence(annotated) is False
    assert runner._visual_context_required(base, {"text_chunks": 3}) is False
    assert runner._visual_context_required(annotated, {"text_chunks": 3}) is False


def test_selected_image_backed_table_promotes_pixels_without_promoting_native_table() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    image = Path(__file__).resolve().parent / "_temporary_table_image_for_test.png"
    image.write_bytes(b"test-image")
    try:
        image_table = {
            "artifact_type": "table",
            "table_id": "table-1",
            "metadata": {"image_path": str(image)},
        }
        native_table = {
            "artifact_type": "table",
            "table_id": "table-2",
            "metadata": {"markdown": "| metric | value |"},
        }
        assert runner._selected_image_backed_table([image_table]) is True
        assert runner._selected_image_backed_table([native_table]) is False
        assert runner._has_canonical_visual_input(
            {
                "source_summaries": [
                    {**image_table, "image_path": None},
                ],
                "visual_images_attached": 1,
            },
            require_pixels=True,
        ) is True
    finally:
        image.unlink(missing_ok=True)


def test_selected_image_backed_figure_promotes_pixels_without_gold_annotations() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    image = Path(__file__).resolve().parent / "_temporary_figure_image_for_test.png"
    image.write_bytes(b"test-image")
    try:
        assert runner._selected_image_backed_visual(
            [{"artifact_type": "figure", "image_path": str(image)}]
        ) is True
        assert runner._selected_image_backed_visual(
            [{"artifact_type": "table", "metadata": {"markdown": "|x|"}}]
        ) is False
    finally:
        image.unlink(missing_ok=True)


def test_image_only_pdf_projection_requires_visual_context_for_fact_queries() -> None:
    """Image-only input must not silently use an OCR/table-free text path."""
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case_type = runner.BenchmarkCase
    image_only = case_type(
        suite="mmlongbench-doc",
        case_id="image-only",
        document_id="d1",
        title="image-only",
        question="How many vendors are shown in the table?",
        answers=("7",),
        answer_format="Str",
        source_path=Path("/tmp/document.pdf"),
    )
    assert runner._visual_context_required(image_only, {"text_chunks": 0}) is True
    assert runner._visual_context_required(image_only, {"text_chunks": 3}) is False
    # Plain extracted Markdown without an artifact locator is not a PDF
    # page-image fallback.
    extracted = case_type(
        suite="spiqa-test-c",
        case_id="extracted",
        document_id="d2",
        title="extracted",
        question="How many vendors are shown in the table?",
        answers=("7",),
        answer_format="Str",
        full_text="# extracted\n",
    )
    assert runner._visual_context_required(extracted, {"text_chunks": 0}) is False

    # A Markdown adapter may carry an artifact locator and searchable text;
    # the locator alone must not force a visual abstention for a fact query.
    artifact_only_table = case_type(
        suite="spiqa-test-c",
        case_id="artifact-only-table",
        document_id="d3",
        title="artifact-only-table",
        question="What is the value in the table?",
        answers=("7",),
        answer_format="Str",
        referred_artifacts=("7-Table1-1.png",),
        metadata={"table_in_evidence": True},
    )
    assert runner._visual_context_required(artifact_only_table, {"text_chunks": 3}) is False
    assert runner._visual_context_required(artifact_only_table, {"text_chunks": 0}) is True


def test_partial_ocr_without_native_pages_requires_visual_recovery() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = runner.BenchmarkCase(
        suite="mmlongbench-doc",
        case_id="partial-ocr",
        document_id="d1",
        title="partial-ocr",
        question="How many vendors are above the threshold?",
        answers=("5",),
        answer_format="Int",
        source_path=Path("/tmp/scan.pdf"),
    )
    assert runner._visual_context_required(
        case,
        {
            "canonical_text_chunks": 127,
            "document_readiness": {
                "state": "partial",
                "native_text_pages": 0,
                "ocr_text_pages": 45,
                "visual_only_pages": 0,
                "unresolved_pages": 8,
            },
        },
    ) is True
    assert runner._visual_context_required(
        case,
        {
            "canonical_text_chunks": 40,
            "document_readiness": {
                "state": "ready",
                "native_text_pages": 12,
                "ocr_text_pages": 0,
                "visual_only_pages": 0,
                "unresolved_pages": 0,
            },
        },
    ) is False


def test_structured_native_table_does_not_force_pixel_attachment() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    table_case = runner.BenchmarkCase(
        suite="mmlongbench-doc",
        case_id="native-table",
        document_id="paper.pdf",
        title="paper.pdf",
        question="What is the R&D to asset ratio in FY 2019?",
        answers=("5.03%",),
        answer_format="Str",
        source_path=Path("/tmp/paper.pdf"),
        evidence_sources=("Table",),
        metadata={"table_in_evidence": True},
    )
    assert runner._visual_context_required(table_case, {"text_chunks": 40}) is False
    assert runner._visual_context_required(table_case, {"text_chunks": 0}) is True


def test_public_source_adapter_isolates_each_spiqa_document(tmp_path: Path) -> None:
    """One folder job must not ingest every adapted markdown paper together."""

    runner = _load_script("evaluate_public_multimodal_three_mode")
    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.sources = tmp_path / "sources"
    case = runner.BenchmarkCase(
        suite="spiqa-test-c",
        case_id="case-1",
        document_id="1905.12260",
        title="ImageVec",
        question="What?",
        answers=("answer",),
        answer_format="free_form",
        full_text="# paper\n",
    )
    path, adapter = pipeline._source_for(case)  # noqa: SLF001
    assert adapter == "spiqa_official_extracted_markdown"
    assert path.parent.name == "spiqa-1905_12260"
    assert path.read_text(encoding="utf-8") == "# paper\n"


def test_spiqa_artifact_context_uses_source_prose_not_gold_annotations() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    source = "The benchmark reports precision and recall for each task (Table 1)."
    context = runner._spiqa_artifact_context(
        source,
        filename="3-Table1-1.png",
        caption="Table 1: Benchmark tasks",
    )
    assert context is not None
    assert "precision and recall" in context
    assert runner._spiqa_artifact_context(
        source,
        filename="3-Figure2-1.png",
        caption="Figure 2: unrelated",
    ) is None


def test_ranked_page_projection_preserves_retrieval_order() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    assert runner._ranked_unique_page_numbers(  # noqa: SLF001
        [
            {"page_number": 15},
            {"page_number": 15},
            {"page_number": 4},
            {"page_number": "bad"},
            {"page_number": 1},
        ]
    ) == [15, 4, 1]


def test_visual_attachment_pages_honor_page_hint_without_changing_retrieval_order() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    assert runner._visual_attachment_page_numbers(  # noqa: SLF001
        [15, 6, 4, 1], "what is shown in the picture on slide 6?"
    ) == [6]
    assert runner._visual_attachment_page_numbers(
        [15, 4, 1], "what is shown on slide 6?"
    ) == [15, 4, 1]
    assert runner._preferred_visual_attachment_pages(
        [15, 4, 1],
        [],
        visual_expected=True,
        coverage_enabled=False,
        query="what is shown on slide 6?",
    ) == [6]


def test_visual_gate_keys_accept_production_rag_evidence_dicts() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    case = runner.BenchmarkCase(
        suite="mmlongbench-doc",
        case_id="page-keys",
        document_id="doc",
        title="page-keys",
        question="what is shown on slides 12 and 14?",
        answers=("x",),
        answer_format="Str",
    )
    assert runner._visual_keys(
        case,
        [{"page_number": 12}, {"page_number": 14}],
    ) == ["12", "14"]


def test_visual_attachment_pages_prefer_ranked_visual_hit_for_noncoverage_query() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    visual_hits = [
        {"page_number": 15, "visual_page": True},
        {"page_number": 14, "visual_page": True},
    ]
    assert runner._preferred_visual_attachment_pages(
        [14, 12, 15, 13],
        visual_hits,
        visual_expected=True,
        coverage_enabled=False,
        query="what buildings appear in the first picture?",
    ) == [14]
    assert runner._preferred_visual_attachment_pages(
        [14, 12, 15, 13],
        visual_hits,
        visual_expected=True,
        coverage_enabled=False,
        query="compare the two pictures",
    ) == [14, 15]
    assert runner._preferred_visual_attachment_pages(
        [14],
        [
            {"page_number": 14, "visual_page": True},
            {"page_number": 15, "visual_page": True},
            {"page_number": 9, "visual_page": True},
        ],
        visual_expected=True,
        coverage_enabled=False,
        query="which sources are used?",
    ) == [14, 15]
    assert runner._preferred_visual_attachment_pages(
        [14, 12, 15, 13],
        visual_hits,
        visual_expected=True,
        coverage_enabled=True,
        query="what buildings appear in the first picture?",
    ) == [14, 12, 15, 13]
    assert runner._preferred_visual_attachment_pages(
        [14, 12, 15, 13],
        [],
        visual_expected=True,
        coverage_enabled=True,
        query="what is shown on slides 12 and 14?",
    ) == [12, 14]


def test_visual_attachment_limit_follows_referred_artifacts_without_plural_query() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    assert runner._visual_attachment_limit(  # noqa: SLF001
        "What does the document say?",
        requested_artifact_count=2,
    ) == 4
    assert runner._visual_attachment_limit(
        "What does the document say?",
        requested_artifact_count=8,
    ) == 4
    assert runner._visual_attachment_limit(
        "Compared with 22Q1, how much higher is the increase rate?",
    ) == 4
    assert runner._visual_attachment_limit(
        "How much lower than last year was the result?",
    ) == 4
    assert runner._visual_attachment_limit("which sources are used?") == 2
    assert runner._visual_attachment_limit("How important is visual grounding?") == 2


def test_rendered_page_filename_aliases_share_attachment_identity() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    assert runner._attachment_page_number("page-00015.png") == 15
    assert runner._attachment_page_number("page-15.png") == 15
    assert runner._attachment_page_number("table-2.png") is None


def test_spiqa_artifact_projection_uses_ranked_pages_without_gold_labels() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = SimpleNamespace(
        artifacts=[
            {"file": "6-Table3-1.png"},
            {"file": "6-Table2-1.png"},
            {"file": "4-Figure3-1.png"},
            {"file": "9-Table8-1.png"},
        ]
    )
    assert runner._ranked_spiqa_artifact_files(case, [6, 9, 4]) == [
        "6-Table3-1.png",
        "9-Table8-1.png",
        "4-Figure3-1.png",
        "6-Table2-1.png",
    ]


def test_spiqa_attachment_respects_selected_artifact_before_page_catalog():
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = SimpleNamespace(artifacts=[{"file": "7-Figure1-1.png"},
                                     {"file": "7-Table2-1.png"}], question="What is the result?")
    sources = [{"table_id": "selected-table", "artifact_type": "table", "page_number": 7,
                "image_path": "/isolated/7-Table2-1.png"}]
    assert runner._ranked_spiqa_artifact_files(case, [7], ordered_sources=sources) == [
        "7-Table2-1.png", "7-Figure1-1.png"]
    assert runner._ranked_spiqa_artifact_files(case, [], ordered_sources=sources) == [
        "7-Table2-1.png"]


def test_spiqa_selected_images_keep_distinct_same_page_artifacts_and_scope():
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = SimpleNamespace(artifacts=[{"file": "7-Figure1-1.png"},
                                     {"file": "7-Figure2-1.png"},
                                     {"file": "8-Table1-1.png"}], question="Look at page 7")
    sources = [{"image_path": "/isolated/7-Figure2-1.png", "page_number": 7},
               {"metadata": {"image_path": "/isolated/7-Figure1-1.png"}, "page_number": 7},
               {"image_path": "/isolated/8-Table1-1.png", "page_number": 8}]
    assert runner._ranked_spiqa_artifact_files(case, [7], ordered_sources=sources) == [
        "7-Figure2-1.png", "7-Figure1-1.png"]


def test_multimodal_attachment_text_exposes_unambiguous_page_provenance() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    content = runner._attached_content(  # noqa: SLF001
        "inspect the supplied image",
        [("page-6.png", b"png"), ("table-2.png", b"png")],
    )
    assert isinstance(content, list)
    text = content[0]["text"]
    assert "attachment 1; page 6: page-6.png" in text
    assert "attachment 2: table-2.png" in text


def test_public_aya_eval_mirrors_visual_fail_closed_contract() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    generated = {"raw": '{"answer":"hallucinated"}', "answer": "hallucinated"}
    guarded = runner._apply_visual_sufficiency_contract(  # noqa: SLF001
        question="What does the chart show?",
        mode="aya_pipeline",
        input_meta={"source_summaries": [{"chunk_type": "text"}]},
        generated=generated,
    )
    assert guarded["answer"] == "Not answerable"
    assert guarded["visual_sufficiency_guard"]["applied"] is True

    supported = runner._apply_visual_sufficiency_contract(  # noqa: SLF001
        question="What does the chart show?",
        mode="aya_pipeline",
        input_meta={
            "source_summaries": [
                {"artifact_type": "figure", "retrieval_channels": ["lancedb"]}
            ],
            "visual_images_attached": 1,
        },
        generated=generated,
    )
    assert supported["answer"] == "hallucinated"
    assert supported["visual_sufficiency_guard"]["applied"] is False


def test_image_only_readiness_also_fails_closed_without_page_pixels() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    generated = {"raw": '{"answer":"inferred"}', "answer": "inferred"}
    guarded = runner._apply_visual_sufficiency_contract(  # noqa: SLF001
        question="What is shown in the document?",
        mode="aya_pipeline",
        input_meta={
            "visual_context_expected": True,
            "source_summaries": [{"chunk_type": "text"}],
            "visual_images_attached": 0,
        },
        generated=generated,
    )
    assert guarded["answer"] == "Not answerable"
    assert guarded["visual_sufficiency_guard"]["applied"] is True


def test_readiness_selected_page_pixels_are_visual_evidence_without_visual_hit() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    generated = {"raw": '{"answer":5}', "answer": 5}
    guarded = runner._apply_visual_sufficiency_contract(  # noqa: SLF001
        question="How many vendors are above the threshold?",
        mode="aya_pipeline",
        input_meta={
            "visual_context_expected": True,
            "source_summaries": [{"chunk_type": "text", "page_number": 10}],
            "visual_images_attached": 1,
        },
        generated=generated,
    )
    assert guarded["answer"] == 5
    assert guarded["visual_sufficiency_guard"]["applied"] is False


def test_visual_promotion_gate_rejects_small_perfect_smoke() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    rows = [
        {
            "case_id": "one",
            "status": "ok",
            "profile": "image_only",
            "unanswerable": False,
            "required_keys": ["1"],
            "metrics": {"T": {"recall": 1.0}, "T+V": {"recall": 1.0}},
        }
    ]
    result = runner.visual_promotion_gate(
        rows,
        {"T": {"recall": 1.0}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True},
    )
    assert result["passed"] is False
    assert result["checks"]["minimum_positive_cases"] is False
    assert result["checks"]["visual_gain"] is False
    assert result["scope"] == "isolated_visual_trial"
    assert result["production_eligible"] is False
    assert result["answer_quality_required"] is True
    assert result["checks"]["answer_quality"] is False
    assert result["answer_quality_status"] == "not_attached"


def test_visual_gate_timeout_boundary_returns_without_waiting_for_task() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await runner._bounded_nonblocking(asyncio.sleep(60), 0.01)

    asyncio.run(run())


def test_visual_gate_timeout_report_uses_stage_specific_budget() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    assert runner._timeout_budget_for_stage(  # noqa: SLF001
        "visual_index",
        prepare_timeout_seconds=900,
        visual_timeout_seconds=600,
    ) == 600
    assert runner._timeout_budget_for_stage(  # noqa: SLF001
        "render_pages",
        prepare_timeout_seconds=900,
        visual_timeout_seconds=600,
    ) == 900
    assert runner._timeout_budget_for_stage(  # noqa: SLF001
        "text_search",
        prepare_timeout_seconds=900,
        visual_timeout_seconds=600,
    ) is None


def test_visual_promotion_gate_never_infers_answer_quality_from_retrieval_only() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    rows = []
    for index, profile in enumerate(
        ["native_text"] * 7 + ["mixed_text_image"] * 7 + ["image_only"] * 6
    ):
        rows.append(
            {
                "case_id": f"mmlong-{index}",
                "suite": "mmlongbench-doc",
                "status": "ok",
                "profile": profile,
                "unanswerable": False,
                "required_keys": [str(index)],
                "metrics": {
                    "T": {"recall": 0.5},
                    "T+V": {"recall": 1.0},
                },
            }
        )
    rows.extend(
        {
            "case_id": f"spiqa-{index}",
            "suite": "spiqa-test-c",
            "status": "ok",
            "profile": "figure",
            "unanswerable": False,
            "required_keys": [f"s{index}"],
            "metrics": {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        }
        for index in range(5)
    )
    result = runner.visual_promotion_gate(
        rows,
        {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True},
    )
    assert result["checks"]["minimum_positive_cases"] is True
    assert result["checks"]["public_suite_frontier"] is True
    assert result["checks"]["answer_quality"] is False
    assert result["passed"] is False

    quality = {
        "status": "complete",
        "paired_cases": 20,
        "promotion_safe": True,
    }
    promoted = runner.visual_promotion_gate(
        rows,
        {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True},
        answer_quality=quality,
        selected_case_ids=[row["case_id"] for row in rows],
        production_text_baseline=True,
    )
    assert promoted["checks"]["answer_quality"] is True
    assert promoted["answer_quality_status"] == "complete"
    assert promoted["passed"] is True

    incomplete = runner.visual_promotion_gate(
        rows, {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True}, answer_quality=quality,
        selected_case_ids=[row["case_id"] for row in rows] + ["not_attempted"],
    )
    assert incomplete["checks"]["complete_frontier"] is False
    assert incomplete["passed"] is False
    disabled_ocr = runner.visual_promotion_gate(
        rows, {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True}, answer_quality=quality,
        production_text_baseline=False,
    )
    assert disabled_ocr["checks"]["production_text_baseline"] is False
    assert disabled_ocr["passed"] is False


def test_visual_promotion_gate_requires_no_fused_text_regression() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    rows = [
        {
            "case_id": "regressed",
            "status": "ok",
            "profile": "native_text",
            "unanswerable": False,
            "required_keys": ["1"],
            "metrics": {"T": {"recall": 1.0}, "T+V": {"recall": 0.0}},
        }
    ]
    result = runner.visual_promotion_gate(
        rows,
        {"T": {"recall": 1.0}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True},
        minimum_positive_cases=1,
    )
    assert result["checks"]["no_text_regression"] is False
    assert result["regressions"] == ["regressed"]


def test_visual_promotion_gate_requires_pdf_format_frontier() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    rows = [
        {
            "case_id": f"spiqa-{index}",
            "suite": "spiqa-test-c",
            "status": "ok",
            "profile": profile,
            "unanswerable": False,
            "required_keys": [str(index)],
            "metrics": {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        }
        for index, profile in enumerate(
            ["figure"] * 5 + ["figure+table"] * 2 + ["table"] * 13
        )
    ]
    result = runner.visual_promotion_gate(
        rows,
        {"T": {"recall": 0.5}, "T+V": {"recall": 1.0}},
        worker_health={"ok": True},
    )
    assert result["passed"] is False
    assert result["checks"]["public_suite_frontier"] is False
    assert result["checks"]["pdf_format_frontier"] is False
    assert result["observed_pdf_profiles"] == []


def test_mmlong_profile_sampling_is_document_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = [
        library.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id=f"case-{index}",
            document_id=document,
            title=document,
            question="q",
            answers=("a",),
            answer_format="Str",
        )
        for index, (document, profile) in enumerate(
            [
                ("native-a", "native_text"),
                ("native-a", "native_text"),
                ("mixed-a", "mixed_text_image"),
                ("image-a", "image_only"),
                ("image-b", "image_only"),
            ]
        )
    ]
    profile_by_document = {
        "native-a": "native_text",
        "mixed-a": "mixed_text_image",
        "image-a": "image_only",
        "image-b": "image_only",
    }
    monkeypatch.setattr(
        library,
        "mmlong_document_profile",
        lambda case: profile_by_document[case.document_id],
    )

    selected = library.stable_mmlong_profile_sample(cases, per_profile=1)
    assert {library.mmlong_document_profile(case) for case in selected} == {
        "native_text",
        "mixed_text_image",
        "image_only",
    }
    assert len({case.document_id for case in selected}) == 3


def test_stable_mmlong_case_sample_reuses_documents_deterministically() -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = [
        library.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id=f"case-{document}-{index}",
            document_id=document,
            title=document,
            question=f"question {index}",
            answers=("answer",),
            answer_format="Str",
            source_path=Path(f"/tmp/{document}.pdf"),
            evidence_pages=(index + 1,),
        )
        for document in ("native-a", "native-b")
        for index in range(3)
    ]
    selected = library.stable_mmlong_case_sample(
        cases,
        per_profile=1,
        cases_per_document=2,
    )
    again = library.stable_mmlong_case_sample(
        cases,
        per_profile=1,
        cases_per_document=2,
    )
    assert len(selected) == 2
    assert len({case.document_id for case in selected}) == 1
    assert [case.case_id for case in selected] == [case.case_id for case in again]


def test_stable_mmlong_answer_sample_covers_profiles_without_gold_peeking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = [
        library.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id=f"{profile}:{document}:{index}",
            document_id=document,
            title=document,
            question="question",
            answers=("answer",),
            answer_format="Str",
            source_path=Path(f"/tmp/{document}.pdf"),
        )
        for profile, document in (
            ("native_text", "native-a"),
            ("mixed_text_image", "mixed-a"),
            ("image_only", "image-a"),
        )
        for index in range(3)
    ]
    profile_by_document = {
        "native-a": "native_text",
        "mixed-a": "mixed_text_image",
        "image-a": "image_only",
    }
    monkeypatch.setattr(
        library,
        "mmlong_document_profile",
        lambda case: profile_by_document[case.document_id],
    )
    selected = library.stable_mmlong_answer_sample(
        reversed(cases), limit=6, cases_per_document=2
    )
    assert len(selected) == 6
    assert {library.mmlong_document_profile(case) for case in selected} == {
        "native_text",
        "mixed_text_image",
        "image_only",
    }
    assert Counter(case.document_id for case in selected) == {
        "native-a": 2,
        "mixed-a": 2,
        "image-a": 2,
    }


def test_stable_spiqa_case_sample_is_stratified_and_bounds_document_reuse() -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = []
    for stratum in ("figure", "table", "text"):
        for document_index in range(2):
            document_id = f"{stratum}-doc-{document_index}"
            for question_index in range(3):
                cases.append(
                    library.BenchmarkCase(
                        suite="spiqa-test-c",
                        case_id=f"{stratum}:{document_index}:{question_index}",
                        document_id=document_id,
                        title=document_id,
                        question="question",
                        answers=("answer",),
                        answer_format="free_form",
                        metadata={
                            "figure_in_evidence": stratum == "figure",
                            "table_in_evidence": stratum == "table",
                        },
                    )
                )

    selected = library.stable_spiqa_case_sample(
        cases,
        limit=12,
        cases_per_document=2,
    )
    repeated = library.stable_spiqa_case_sample(
        reversed(cases),
        limit=12,
        cases_per_document=2,
    )

    assert len(selected) == 12
    assert [case.case_id for case in selected] == [case.case_id for case in repeated]
    assert {case.stratum for case in selected} == {"figure", "table", "text"}
    counts = Counter((case.stratum, case.document_id) for case in selected)
    assert max(counts.values()) <= 2


def test_eval_prompt_contract_cannot_accept_or_leak_reference_answer() -> None:
    library = _load_script("public_multimodal_eval_lib")
    prompt = library.build_eval_prompt(
        question="Which model wins?",
        context="Table: Baseline 70; Proposed 80.",
        context_label="gold_evidence",
        answer_format="Str",
    )

    assert "reference" not in prompt.lower()
    assert "hidden-gold-answer" not in prompt
    assert "Which model wins?" in prompt
    assert "Expected answer format: Str" in prompt
    assert library.parse_answer_payload('{"answer":"Proposed"}') == "Proposed"
    assert library.parse_answer_payload('{"answer":["A","B"]}') == '["A", "B"]'


def test_eval_prompt_visual_contract_is_generic_and_page_safe() -> None:
    library = _load_script("public_multimodal_eval_lib")
    prompt = library.build_eval_prompt(
        question="What does the figure show?",
        context="page 3 image attached",
        context_label="aya_production_retrieval",
        answer_format="Str",
        visual_context=True,
    )
    assert "Visual-evidence rule" in prompt
    assert "Do not transfer visual or numeric values from another page" in prompt
    assert "return exactly: Not answerable" in prompt


def test_eval_prompt_preserves_atomic_lists_and_numeric_details() -> None:
    library = _load_script("public_multimodal_eval_lib")
    list_prompt = library.build_eval_prompt(
        question="Which stages are shown on slides 12 and 14?",
        context="slide 12: Interphase, Prophase; slide 14: Metaphase, Anaphase",
        context_label="aya_production_retrieval",
        answer_format="List",
    )
    assert "separate items" in list_prompt
    assert "do not group" in list_prompt
    freeform_prompt = library.build_eval_prompt(
        question="Did the annotators agree and how much?",
        context="Fleiss' Kappa 0.64-0.68; coreference 90.5%",
        context_label="aya_production_retrieval",
        answer_format="None",
    )
    assert "numeric value or range" in freeform_prompt


def test_bounded_context_labels_truncation_without_claiming_full() -> None:
    library = _load_script("public_multimodal_eval_lib")
    bounded = library.bounded_context("a" * 10_000, max_chars=1_000, label="full")

    assert bounded.truncated is True
    assert bounded.original_chars == 10_000
    assert bounded.included_chars <= 1_000
    assert "TRUNCATED" in bounded.text


def test_token_bounded_context_reports_windowed_long_context() -> None:
    library = _load_script("public_multimodal_eval_lib")
    bounded = library.bounded_context_tokens(
        " ".join(f"token-{index}" for index in range(100)),
        max_tokens=20,
        label="full_extracted_document",
    )

    assert bounded.truncated is True
    assert bounded.original_tokens == 100
    assert bounded.included_tokens <= 20
    assert "WINDOWED" in bounded.text


def test_spiqa_zip_reader_rejects_traversal_and_returns_exact_member() -> None:
    library = _load_script("public_multimodal_eval_lib")
    case = next(case for case in library.load_spiqa_test_c_cases() if case.referred_artifacts)
    name = case.referred_artifacts[0]
    images = library.read_spiqa_images(case, [name])

    assert images[0][0] == name
    assert images[0][1].startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsafe SPIQA image"):
        library.read_spiqa_images(case, ["../secret.png"])


def test_spiqa_reader_preserves_native_pixels_and_never_falls_back(tmp_path, monkeypatch) -> None:
    from io import BytesIO
    from zipfile import ZipFile
    from PIL import Image

    library = _load_script("public_multimodal_eval_lib")
    native = tmp_path / "SPIQA_testC_Images.zip"
    thumbnails = tmp_path / "SPIQA_testC_Images_224px.zip"
    name = "1-Table1-1.png"
    case = library.BenchmarkCase(
        suite="spiqa", case_id="synthetic", document_id="paper", title="paper",
        question="Which categories are listed?", answers=(), answer_format="None",
    )
    pixels = BytesIO()
    Image.new("RGB", (1024, 320), "white").save(pixels, format="PNG")
    with ZipFile(native, "w") as archive:
        archive.writestr(f"test-C/paper/{name}", pixels.getvalue())
    with ZipFile(thumbnails, "w") as archive:
        archive.writestr(f"test-C/paper/{name}", b"thumbnail")
    monkeypatch.setattr(library, "SPIQA_TEST_C_IMAGES", native)

    result = library.read_spiqa_images(case, [name])
    assert result == [(name, pixels.getvalue())]
    assert Image.open(BytesIO(result[0][1])).size == (1024, 320)
    provenance = library.supplied_image_provenance(*result[0])
    assert provenance["width"] == 1024 and provenance["height"] == 320
    assert provenance["bytes"] == len(pixels.getvalue())
    native.unlink()
    with pytest.raises(FileNotFoundError, match="224px thumbnails are not an equivalent input"):
        library.read_spiqa_images(case, [name])


def test_spiqa_default_archive_is_original_resolution() -> None:
    library = _load_script("public_multimodal_eval_lib")
    assert library.SPIQA_TEST_C_IMAGES.name == "SPIQA_testC_Images.zip"


def test_official_mmlong_scorer_wrapper_matches_upstream_contract() -> None:
    library = _load_script("public_multimodal_eval_lib")

    assert library.official_mmlong_score(gold="18.29%", prediction="18.29", answer_format="Float") == 1.0
    assert library.official_mmlong_score(gold="42", prediction="42.0", answer_format="Int") == 1.0
    assert library.official_mmlong_score(gold="alpha", prediction="unrelated", answer_format="Str") == 0.0
    assert library.official_mmlong_score(
        gold="['A', 'B']", prediction='["B", "A"]', answer_format="List"
    ) == 1.0


def test_three_mode_answer_quality_comparison_is_paired_and_non_promotional() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    selected = [
        runner.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id="mmlong:case-1",
            document_id="doc-1",
            title="Doc",
            question="Which value?",
            answers=("42",),
            answer_format="Int",
        )
    ]
    rows = [
        {
            "status": "ok",
            "case_id": "mmlong:case-1",
            "suite": "mmlongbench-doc",
            "stratum": "table",
            "mode": "gold_evidence",
            "score": {"official_mmlong_score": 1.0},
        },
        {
            "status": "ok",
            "case_id": "mmlong:case-1",
            "suite": "mmlongbench-doc",
            "stratum": "table",
            "mode": "full_extracted_document",
            "score": {"official_mmlong_score": 0.5},
        },
        {
            "status": "ok",
            "case_id": "mmlong:case-1",
            "suite": "mmlongbench-doc",
            "stratum": "table",
            "mode": "aya_pipeline",
            "score": {"official_mmlong_score": 0.75},
        },
    ]

    result = runner._answer_quality_comparison(  # noqa: SLF001
        selected, rows, ("gold_evidence", "full_extracted_document", "aya_pipeline")
    )

    assert result["status"] == "complete"
    assert result["paired_cases"] == 1
    assert result["overall"]["aya_vs_full_delta"] == 0.25
    assert result["overall"]["aya_vs_gold_delta"] == -0.25
    assert result["by_document_profile"]["extracted_adapter"]["cases"] == 1
    assert result["promotion_decision"] == "not_evaluated"


def test_three_mode_answer_quality_comparison_marks_missing_mode_partial() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    selected = [
        runner.BenchmarkCase(
            suite="spiqa-test-c",
            case_id="spiqa:case-1",
            document_id="doc-1",
            title="Doc",
            question="What?",
            answers=("answer",),
            answer_format="free_form",
        )
    ]
    result = runner._answer_quality_comparison(  # noqa: SLF001
        selected,
        [
            {
                "status": "ok",
                "case_id": "spiqa:case-1",
                "suite": "spiqa-test-c",
                "stratum": "figure",
                "mode": "aya_pipeline",
                "score": {"token_f1": 1.0},
            }
        ],
        ("gold_evidence", "full_extracted_document", "aya_pipeline"),
    )

    assert result["status"] == "partial"
    assert result["paired_cases"] == 0
    assert result["incomplete_cases"] == 1
    assert result["overall"]["status"] == "unscored"


def test_public_eval_output_and_runtime_paths_fail_closed(tmp_path: Path) -> None:
    library = _load_script("public_multimodal_eval_lib")
    runner = _load_script("evaluate_public_multimodal_three_mode")

    with pytest.raises(ValueError, match="under"):
        library.ensure_public_output_path(tmp_path / "report.json")
    with pytest.raises(ValueError, match="under"):
        runner._safe_runtime_root(tmp_path / "runtime")  # noqa: SLF001


def test_public_evaluator_uses_bounded_active_page_projection() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")

    class FakeRag:
        def __init__(self, count: int) -> None:
            self.count = count
            self.requested: list[tuple[list[str], int]] = []

        def document_page_counts(self, document_ids):
            return {document_ids[0]: self.count}

        def list_page_coverage(self, *, document_ids, page_limit):
            self.requested.append((document_ids, page_limit))
            return [{"document_id": document_ids[0], "page_number": index} for index in range(1, page_limit + 1)]

    policy = SimpleNamespace(aggregation_page_limit=64)
    rag = FakeRag(40)
    pages, expected = runner._bounded_exhaustive_page_projection(  # noqa: SLF001
        rag,
        document_id="doc-1",
        coverage_policy=policy,
    )
    assert expected == 40
    assert len(pages) == 40
    assert rag.requested == [(["doc-1"], 40)]

    long_rag = FakeRag(120)
    pages, expected = runner._bounded_exhaustive_page_projection(  # noqa: SLF001
        long_rag,
        document_id="doc-1",
        coverage_policy=policy,
    )
    assert expected == 120
    assert pages == []
    assert long_rag.requested == []


def test_visual_gate_text_arm_uses_page_coverage_for_document_census() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")

    class FakeRag:
        async def search_hybrid(self, **_kwargs):
            return {
                "results": [
                    {"document_id": "doc-1", "page_number": 2, "chunk_id": "c2"}
                ]
            }

        def document_page_counts(self, _document_ids):
            return {"doc-1": 4}

        def list_page_coverage(self, *, document_ids, page_limit):
            return [
                {
                    "document_id": document_ids[0],
                    "page_number": page,
                    "chunk_id": f"page:{page}",
                }
                for page in range(1, page_limit + 1)
            ]

    class FakePipeline:
        rag = FakeRag()
        vector_index = SimpleNamespace(retrieval_store=object())
        embeddings = object()

    case = SimpleNamespace(
        question="How many pages in this document contain charts?",
        evidence_sources=(),
        referred_artifacts=(),
    )
    import asyncio

    results, _latency = asyncio.run(
        runner._text_search(FakePipeline(), case, "doc-1", top_k=3)  # noqa: SLF001
    )
    assert [item["page_number"] for item in results] == [2, 1, 3, 4]


def test_visual_runner_can_disable_ocr_without_changing_default(tmp_path: Path) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    settings = SimpleNamespace(
        embedding_model="embeddinggemma:300m",
        embedding_query_prefix="",
        embedding_document_prefix="",
        ollama_host="http://127.0.0.1:11434",
        request_timeout_seconds=1.0,
    )
    visual = runner.IsolatedAyaPipeline(
        runtime_root=tmp_path / "visual",
        settings=settings,
        enable_pdf_ocr=False,
    )
    assert visual.indexing.enable_pdf_ocr is False
    assert runner.IsolatedAyaPipeline(
        runtime_root=tmp_path / "default",
        settings=settings,
    ).indexing.enable_pdf_ocr is True


def test_public_pipeline_visual_index_timeout_is_optional_and_retryable(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")

    class SlowProvider:
        name = "fake_visual"

        def __init__(self) -> None:
            self.aborted = False

        async def index_pages(self, _records):
            await asyncio.sleep(0.05)
            return {"ok": True, "status": "complete"}

        def abort(self) -> None:
            self.aborted = True

    provider = SlowProvider()
    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.visual_provider = provider
    pipeline.visual_timeout_seconds = 0.001
    pipeline._visual_prepared = {}
    pipeline.root = tmp_path
    pipeline._source_for = lambda _case: (tmp_path / "paper.pdf", "test")
    monkeypatch.setattr(
        runner,
        "render_pdf_page_records",
        lambda **_kwargs: [],
    )
    case = runner.BenchmarkCase(
        suite="mmlongbench-doc",
        case_id="timeout",
        document_id="d1",
        title="timeout",
        question="What is shown?",
        answers=("x",),
        answer_format="Str",
        source_path=tmp_path / "paper.pdf",
    )

    status = asyncio.run(
        pipeline._prepare_visual_pages(  # noqa: SLF001
            case,
            document_id="d1",
            revision_id="r1",
        )
    )
    assert status["status"] == "failed"
    assert status["error_type"] == "TimeoutError"
    assert status["timeout_seconds"] == 0.001
    assert provider.aborted is True


def test_public_pipeline_preflight_reports_embedding_contract() -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")

    class FakeEmbeddings:
        model = "embeddinggemma:300m"

        async def embed_query(self, _query: str):
            return [0.1, 0.2, 0.3]

    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.embeddings = FakeEmbeddings()
    import asyncio

    result = asyncio.run(pipeline.preflight_local_search())
    assert result["provider"] == "ollama"
    assert result["model"] == "embeddinggemma:300m"
    assert result["dimension"] == 3
    assert result["latency_ms"] >= 0


def test_public_pipeline_resolves_legacy_document_by_raw_content_hash(tmp_path: Path) -> None:
    from app.db.sqlite import connect, init_db

    runner = _load_script("evaluate_public_multimodal_three_mode")
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.pdf"
    # The resolver only needs bytes; a small non-PDF fixture keeps this test
    # independent of Docling while exercising the legacy hash lookup.
    source.write_bytes(b"legacy-source-bytes")
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO indexed_folders(id, folder_path, recursive, file_types, created_at, updated_at)
            VALUES ('folder-1', ?, 0, 'pdf', '2026-01-01', '2026-01-01')
            """,
            (str(tmp_path),),
        )
        connection.execute(
            """
            INSERT INTO documents(
                id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count, raw_content_hash,
                active_revision_id, updated_at
            ) VALUES ('doc-1', 'folder-1', '/old/location/paper.pdf', 'paper.pdf',
                      'pdf', ?, '2026-01-01', '2026-01-01', 0, ?, 'rev-1', '2026-01-01')
            """,
            (
                __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
                __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
            ),
        )
    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.db_path = db_path
    row = pipeline._find_document_by_content(source)  # noqa: SLF001
    assert row is not None
    assert row["id"] == "doc-1"
    assert row["active_revision_id"] == "rev-1"


def test_three_mode_checkpoint_retries_failed_rows_and_is_atomic(tmp_path: Path) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    selected = [
        runner.BenchmarkCase(
            suite="spiqa-test-c",
            case_id="case-1",
            document_id="doc-1",
            title="Doc",
            question="What?",
            answers=("answer",),
            answer_format="free_form",
        )
    ]
    fingerprint = runner._evaluation_config_fingerprint(  # noqa: SLF001
        model="cx/gpt-5.6-sol",
        modes=("gold_evidence", "aya_pipeline"),
        mmlong_cases=0,
        spiqa_cases=1,
        top_k=8,
        max_context_chars=9_000,
        max_full_images=4,
        max_context_tokens=96_000,
        output_reserve_tokens=16_000,
        mmlong_version="v2",
        visual_pages=False,
        enable_pdf_ocr=True,
        mmlong_cases_per_document=2,
        spiqa_cases_per_document=2,
        selected_case_ids=["case-1"],
    )
    path = tmp_path / "run.partial"
    good = {
        "status": "ok",
        "case_id": "case-1",
        "mode": "gold_evidence",
        "answer": "answer",
    }
    failed = {
        "status": "failed",
        "case_id": "case-1",
        "mode": "aya_pipeline",
        "error": "provider timeout",
    }
    runner._write_three_mode_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        model="cx/gpt-5.6-sol",
        modes=("gold_evidence", "aya_pipeline"),
        selected=selected,
        rows=[good, failed],
    )
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = runner._load_three_mode_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        selected_case_ids={"case-1"},
    )
    assert [runner._result_key(row) for row in loaded] == [("case-1", "gold_evidence")]  # noqa: SLF001


def test_three_mode_checkpoint_retries_successful_aya_row_with_visual_failure(
    tmp_path: Path,
) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    selected = [
        runner.BenchmarkCase(
            suite="mmlongbench-doc",
            case_id="case-visual",
            document_id="doc-visual",
            title="Doc",
            question="What is shown?",
            answers=("chart",),
            answer_format="free_form",
        )
    ]
    fingerprint = runner._evaluation_config_fingerprint(  # noqa: SLF001
        model="cx/gpt-5.6-sol",
        modes=("aya_pipeline",),
        mmlong_cases=1,
        spiqa_cases=0,
        top_k=8,
        max_context_chars=9_000,
        max_full_images=4,
        max_context_tokens=96_000,
        output_reserve_tokens=16_000,
        mmlong_version="v2",
        visual_pages=True,
        enable_pdf_ocr=True,
        mmlong_cases_per_document=2,
        spiqa_cases_per_document=2,
        selected_case_ids=["case-visual"],
    )
    path = tmp_path / "visual.partial"
    runner._write_three_mode_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        model="cx/gpt-5.6-sol",
        modes=("aya_pipeline",),
        selected=selected,
        rows=[
            {
                "status": "ok",
                "case_id": "case-visual",
                "mode": "aya_pipeline",
                "input": {
                    "visual_index_status": {
                        "status": "failed",
                        "error": "worker crashed",
                    }
                },
            }
        ],
    )
    loaded = runner._load_three_mode_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        selected_case_ids={"case-visual"},
    )
    assert loaded == []


def test_visual_gate_checkpoint_keeps_contract_and_current_stage(tmp_path: Path) -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    path = tmp_path / "visual.partial"
    rows = [
        {
            "case_id": "case-1",
            "status": "preparing",
            "stage": "visual_index",
        }
    ]

    runner._write_checkpoint(  # noqa: SLF001
        path,
        rows=rows,
        checkpoint_contract={"device": "cpu", "render_max_pixels": 2_000_000},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evaluator_fingerprint"] == runner.VISUAL_EVALUATOR_FINGERPRINT
    assert payload["checkpoint_contract"]["device"] == "cpu"
    assert payload["rows"] == rows
    assert payload["rows"][0]["stage"] == "visual_index"


def test_visual_checkpoint_replace_failure_preserves_previous_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    path = tmp_path / "run.partial"
    contract = {"case_ids": ["one"], "top_k": 10}
    runner._write_checkpoint(path, rows=[], checkpoint_contract=contract)
    original = path.read_bytes()

    def fail_replace(*_args: object) -> None:
        raise OSError("simulated crash before atomic replace")

    monkeypatch.setattr(_load_script("evaluation_checkpoint").os, "replace", fail_replace)
    with pytest.raises(OSError):
        runner._write_checkpoint(
            path, rows=[{"case_id": "one", "status": "ok"}],
            checkpoint_contract=contract,
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_visual_checkpoint_resume_rejects_changed_config_and_duplicates(tmp_path: Path) -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    path = tmp_path / "run.partial"
    contract = {"case_ids": ["one", "two"], "top_k": 10}
    good = {"case_id": "one", "status": "ok"}
    failed = {"case_id": "two", "status": "failed"}
    runner._write_checkpoint(path, rows=[good, failed], checkpoint_contract=contract)
    kwargs = {"checkpoint_contract": contract, "selected_ids": ["one", "two"]}
    assert runner._load_checkpoint(path, **kwargs) == ([good], "matched")
    assert runner._load_checkpoint(
        path, checkpoint_contract={**contract, "top_k": 5}, selected_ids=["one", "two"],
    ) == ([], "contract_changed")
    runner._write_checkpoint(path, rows=[good, good], checkpoint_contract=contract)
    assert runner._load_checkpoint(path, **kwargs) == ([], "invalid_or_duplicate_rows")


def test_public_retrieval_checkpoint_reuses_only_identical_inputs_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script("evaluate_public_retrieval_only")
    library = _load_script("public_multimodal_eval_lib")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source bytes v1")
    case = library.BenchmarkCase(
        suite="mmlongbench-doc", case_id="case:1", document_id="doc:1",
        title="Untuned source", question="What does the source say?",
        answers=("native data",), answer_format="Str", source_path=source,
        evidence_pages=(1,), evidence_text=("native data",),
    )
    retrieval_calls: list[int] = []
    class Pipeline:
        def __init__(self, **kwargs: object) -> None:
            self.indexing = SimpleNamespace(materialization_policy=lambda _: {
                "enable_pdf_ocr": kwargs["enable_pdf_ocr"],
            })
            self.processing = SimpleNamespace(stage_version="test-stage")
            self.rerank_enabled = False
            self.rerank_mode = "cross_encoder"
            self.rerank_max_candidates = 20
            self.rerank_cross_encoder_path = None

        async def preflight_local_search(self) -> dict:
            return {"ok": True}

        async def retrieve(self, _case: object, *, top_k: int) -> dict:
            retrieval_calls.append(top_k)
            return {"context": "native data", "retrieved_pages": [1], "retrieval_ms": 2.0}

    monkeypatch.setattr(runner, "IsolatedAyaPipeline", Pipeline)
    monkeypatch.setattr(runner, "load_mmlong_cases", lambda **_: [case])
    monkeypatch.setattr(runner, "load_spiqa_test_c_cases", lambda: [])
    monkeypatch.setattr(runner, "mmlong_document_profile", lambda _: "native_text")
    args = SimpleNamespace(
        mmlong_cases=1, spiqa_cases=0, mmlong_per_profile=0,
        mmlong_version="v2", case_ids_file=None, disable_pdf_ocr=False,
        runtime_root=tmp_path / "runtime", output=tmp_path / "result.json",
        resume=True, top_k=10,
    )
    fresh = asyncio.run(runner.run(args))
    assert fresh["score_valid"] is True
    assert fresh["resume_status"] == "missing"
    assert retrieval_calls == [10]
    identical = asyncio.run(runner.run(args))
    assert identical["resume_status"] == "matched"
    assert identical["resumed_rows"] == 1
    assert retrieval_calls == [10]

    args.disable_pdf_ocr = True
    changed = asyncio.run(runner.run(args))
    assert changed["resume_status"] == "contract_changed"
    assert retrieval_calls == [10, 10]
    args.top_k = 5
    changed = asyncio.run(runner.run(args))
    assert changed["resume_status"] == "contract_changed"
    assert retrieval_calls == [10, 10, 5]
    source.write_bytes(b"image-only change with identical annotation IDs")
    changed = asyncio.run(runner.run(args))
    assert changed["resume_status"] == "contract_changed"
    assert retrieval_calls == [10, 10, 5, 5]
    # Legacy reports with the same case ID but no proof of configuration cannot
    # be reused as successes under a new code version.
    args.output.with_suffix(".json.partial").write_text(
        json.dumps({"rows": changed["rows"]}), encoding="utf-8",
    )
    assert asyncio.run(runner.run(args))["resume_status"] == "evaluator_changed"
    assert retrieval_calls == [10, 10, 5, 5, 5]


def test_public_checkpoint_fingerprints_all_artifact_pixels_not_only_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script("evaluate_public_retrieval_only")
    library = _load_script("public_multimodal_eval_lib")
    case = library.BenchmarkCase(
        suite="spiqa-test-c", case_id="spiqa:1", document_id="doc:1", title="Source",
        question="What is shown?", answers=("answer",), answer_format="Str",
        artifacts=({"file": "figure.png", "caption": "same caption"},
                   {"file": "other.png", "caption": "another figure"}),
        referred_artifacts=("figure.png",),
    )
    calls: list[list[str]] = []
    payload = b"original pixels"
    def images(_case: object, names: list[str]) -> list[tuple[str, bytes]]:
        calls.append(names)
        return [("figure.png", b"unchanged"), ("other.png", payload)]
    monkeypatch.setattr(runner, "read_spiqa_images", images)
    settings = SimpleNamespace(embedding_model="local", embedding_query_prefix="q:",
                               embedding_document_prefix="d:")
    pipeline = SimpleNamespace(
        indexing=SimpleNamespace(materialization_policy=lambda _: {}),
        processing=SimpleNamespace(stage_version="test-stage"), rerank_enabled=False,
        rerank_mode="cross_encoder", rerank_max_candidates=20, rerank_cross_encoder_path=None,
    )
    args = SimpleNamespace(runtime_root=tmp_path, top_k=10, mmlong_version="v2")
    before = runner._checkpoint_contract(args, pipeline, settings, [case])
    payload = b"changed non-gold artifact pixels"
    after = runner._checkpoint_contract(args, pipeline, settings, [case])
    assert before["source_annotations_sha256"] != after["source_annotations_sha256"]
    assert calls == [["figure.png", "other.png"], ["figure.png", "other.png"]]


def test_three_mode_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    path = tmp_path / "run.partial"
    path.write_text(
        json.dumps(
            {
                "schema_version": runner.CHECKPOINT_SCHEMA_VERSION,
                "config_fingerprint": "old",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config mismatch"):
        runner._load_three_mode_checkpoint(  # noqa: SLF001
            path,
            config_fingerprint="new",
            selected_case_ids=set(),
        )


def test_three_mode_final_report_write_is_atomic(tmp_path: Path) -> None:
    runner = _load_script("evaluate_public_multimodal_three_mode")
    output = tmp_path / "nested" / "report.json"

    runner._write_json_atomic(output, {"status": "provider_unavailable"})  # noqa: SLF001

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "provider_unavailable"
    assert not list(output.parent.glob("*.tmp"))


def test_public_retrieval_reports_artifact_support_separately_from_ocr() -> None:
    runner = _load_script("evaluate_public_retrieval_only")
    assert runner._artifact_recall({"table.png": True, "figure.png": False}) == 0.5  # noqa: SLF001
    assert runner._artifact_recall({"annotation_present_but_name_unresolved": True}) is None  # noqa: SLF001
    recall, hits = runner._candidate_artifact_recall(  # noqa: SLF001
        ["table.png", "figure.png"],
        ["other.png", "table.png"],
    )
    assert recall == 0.5
    assert hits == {"table.png": True, "figure.png": False}


def test_spiqa_adapter_markdown_has_source_but_not_gold_answer() -> None:
    library = _load_script("public_multimodal_eval_lib")
    case = library.load_spiqa_test_c_cases()[0]

    assert case.title in case.full_text
    assert "## Abstract" in case.full_text
    # The canonical answer is scored out-of-band. It is not deliberately
    # appended to the document adapter as an answer field.
    assert '"free_form_answer"' not in case.full_text
    assert '"highlighted_evidence"' not in case.full_text


def test_spiqa_visual_adapter_indexes_all_artifacts_without_gold_selection(
    tmp_path: Path,
) -> None:
    import sqlite3

    from app.db.sqlite import init_db
    from app.services.indexing_service import IndexingService

    library = _load_script("public_multimodal_eval_lib")
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = next(
        item
        for item in library.load_spiqa_test_c_cases()
        if len(item.artifacts) > len(item.referred_artifacts) >= 1
    )
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text(case.full_text, encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.root = tmp_path
    pipeline.db_path = db_path

    assert pipeline._sync_spiqa_artifacts(case, document.id) is True  # noqa: SLF001
    assert pipeline._sync_spiqa_artifacts(case, document.id) is False  # noqa: SLF001
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT caption, image_path, visual_summary, metadata_json FROM document_figures WHERE document_id = ?",
            (document.id,),
        ).fetchall()
        tables = connection.execute(
            "SELECT caption, metadata_json FROM document_tables WHERE document_id = ?",
            (document.id,),
        ).fetchall()
        counts = connection.execute(
            "SELECT figure_count, table_count FROM documents WHERE id = ?",
            (document.id,),
        ).fetchone()
        active_revision = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (document.id,),
        ).fetchone()[0]
        figure_revisions = {
            row[0]
            for row in connection.execute(
                "SELECT revision_id FROM document_figures WHERE document_id = ?",
                (document.id,),
            ).fetchall()
        }
        table_revisions = {
            row[0]
            for row in connection.execute(
                "SELECT revision_id FROM document_tables WHERE document_id = ?",
                (document.id,),
            ).fetchall()
        }

    assert len(rows) == len(case.artifacts)
    assert len(rows) > len(case.referred_artifacts)
    assert all(Path(row[1]).is_file() for row in rows)
    assert all("spiqa_official_artifact_adapter" in row[3] for row in rows)
    assert figure_revisions == {active_revision}
    assert table_revisions == {active_revision}
    expected_table_count = sum(
        "table" in Path(str(item["file"])).stem.lower()
        for item in case.artifacts
    )
    assert len(tables) == expected_table_count
    assert all("spiqa_official_artifact_adapter" in row[1] for row in tables)
    assert counts == (len(case.artifacts), expected_table_count)

    figure_id = connection = None
    with sqlite3.connect(db_path) as connection:
        figure_id = connection.execute(
            "SELECT id FROM document_figures WHERE document_id = ? ORDER BY figure_index LIMIT 1",
            (document.id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE document_figures SET visual_summary = ?, metadata_json = json_set(metadata_json, '$.vision_model', ?) WHERE id = ?",
            ("A durable visual description", "cx/gpt-5.6-sol", figure_id),
        )

    assert pipeline._sync_spiqa_artifacts(case, document.id) is False  # noqa: SLF001
    with sqlite3.connect(db_path) as connection:
        summary, metadata_json = connection.execute(
            "SELECT visual_summary, metadata_json FROM document_figures WHERE id = ?",
            (figure_id,),
        ).fetchone()
    assert summary == "A durable visual description"
    assert json.loads(metadata_json)["vision_model"] == "cx/gpt-5.6-sol"

    # A byte-identical extracted text can have a new attachment policy/bytes.
    # Materialize into a staged revision without stealing old artifact IDs.
    replacement = IndexingService(
        db_path, canonical_artifact_fingerprint="changed-authored-attachments"
    ).index_file(source_path=str(source), defer_activation=True)
    assert replacement.id == document.id
    assert replacement.revision_id != active_revision
    assert pipeline._sync_spiqa_artifacts(  # noqa: SLF001
        case, document.id, revision_id=replacement.revision_id
    ) is True
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (document.id,)
        ).fetchone()[0] == active_revision
        old_ids = {row[0] for row in connection.execute(
            "SELECT id FROM document_figures WHERE document_id = ? AND revision_id = ?",
            (document.id, active_revision),
        )}
        new_ids = {row[0] for row in connection.execute(
            "SELECT id FROM document_figures WHERE document_id = ? AND revision_id = ?",
            (document.id, replacement.revision_id),
        )}
        assert len(old_ids) == len(new_ids) == len(case.artifacts)
        assert old_ids.isdisjoint(new_ids)
        old_summary, old_path = connection.execute(
            "SELECT visual_summary, image_path FROM document_figures WHERE id = ?", (figure_id,)
        ).fetchone()
        new_summary, new_path = connection.execute(
            "SELECT visual_summary, image_path FROM document_figures WHERE revision_id = ? AND figure_index = 0",
            (replacement.revision_id,),
        ).fetchone()
        assert old_summary == new_summary == "A durable visual description"
        assert old_path != new_path
        assert Path(old_path).read_bytes() == Path(new_path).read_bytes()
    assert pipeline._sync_spiqa_artifacts(  # noqa: SLF001
        case, document.id, revision_id=replacement.revision_id
    ) is False


@pytest.mark.parametrize("change", ["pixels", "caption", "membership"])
def test_spiqa_active_artifact_change_fails_before_mutating_snapshot(
    tmp_path: Path, monkeypatch, change: str,
) -> None:
    from dataclasses import replace
    from app.db.sqlite import init_db, connect
    from app.services.indexing_service import IndexingService

    library = _load_script("public_multimodal_eval_lib")
    runner = _load_script("evaluate_public_multimodal_three_mode")
    case = library.BenchmarkCase(
        suite="spiqa-test-c", case_id="synthetic", document_id="paper",
        title="paper", question="Explain the diagram", answers=(),
        answer_format="None", full_text="# Immutable source",
        artifacts=({"file": "1-Figure1-1.png", "caption": "Architecture"},),
    )
    payload = b"original-image-bytes"
    monkeypatch.setattr(runner, "read_spiqa_images", lambda case, names: [(n, payload) for n in names])
    db_path = tmp_path / "app.db"
    source = tmp_path / "source.md"
    source.write_text(case.full_text)
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    pipeline = object.__new__(runner.IsolatedAyaPipeline)
    pipeline.root = tmp_path
    pipeline.db_path = db_path
    assert pipeline._sync_spiqa_artifacts(case, document.id)
    with connect(db_path) as c:
        before = [dict(r) for r in c.execute("SELECT * FROM document_figures")]
    path = Path(before[0]["image_path"])
    if change == "pixels":
        payload = b"different-image-bytes"
    elif change == "caption":
        case = replace(case, artifacts=({"file": "1-Figure1-1.png", "caption": "New architecture"},))
    else:
        case = replace(case, artifacts=(*case.artifacts, {"file": "2-Figure2-1.png", "caption": "New image"}))
    with pytest.raises(RuntimeError, match="requires_staged_revision"):
        pipeline._sync_spiqa_artifacts(case, document.id)
    assert path.read_bytes() == b"original-image-bytes"
    with connect(db_path) as c:
        assert [dict(r) for r in c.execute("SELECT * FROM document_figures")] == before


def _visual_report(case_id: str, *, suite: str, profile: str) -> dict:
    metrics = {
        "scored": True,
        "recall": 1.0,
        "hit_at_1": True,
        "hit_at_3": True,
        "hit_at_5": True,
        "hit_at_10": True,
        "mrr": 1.0,
        "ndcg": 1.0,
    }
    return {
        "schema_version": "visual-retrieval-gate-v1",
        "promotion_contract_version": "visual-arbitrary-format-v2",
        "metric_contract_version": "visual-evidence-scope-v2",
        "model": "vidore/colSmol-256M",
        "model_revision": "model-rev",
        "model_fingerprint": "same-model",
        "evaluator_fingerprint": "test-evaluator-v1",
        "embedding_model_T": "embeddinggemma:300m",
        "no_llm_calls": True,
        "pdf_ocr_enabled": True,
        "evaluation_configuration": {
            "render_scale": 1.25,
            "device": "auto",
            "local_files_only": True,
            "stage_version": "foundation-v3",
        },
        "cases": 1,
        "selected_cases": 1,
        "score_valid": True,
        "suite_complete": True,
        "selection": {"case_ids": [case_id]},
        "runtime_root": f"/tmp/{case_id}",
        "visual_index_size_bytes": 10,
        "worker_health": {"ok": True},
        "rows": [
            {
                "case_id": case_id,
                "suite": suite,
                "profile": profile,
                "status": "ok",
                "unanswerable": False,
                "required_keys": ["page:1"],
                "metrics": {"T": metrics, "V": metrics, "T+V": metrics},
                "latency_ms": {"T": 10.0, "V": 5.0, "T+V": 16.0, "fusion": 1.0},
            }
        ],
    }


def _promotion_ready_visual_report() -> dict:
    runner = _load_script("evaluate_visual_retrieval_gate")
    rows = []
    for profile in ("native_text", "mixed_text_image", "image_only", "table", "figure"):
        suite = "spiqa-test-c" if profile in {"table", "figure"} else "mmlongbench-doc"
        for index in range(5):
            row = _visual_report(f"{profile}:{index}", suite=suite, profile=profile)["rows"][0]
            row.update({
                "required_keys": ["1", "2"],
                "text_page_required_keys": ["1", "2"],
                "rankings": {"T": ["1"], "T_artifact": ["1"], "T_page": ["1"],
                             "V": ["1", "2"], "T+V": ["1", "2"]},
                "metrics": {"T": runner._ranking_metrics(["1"], {"1", "2"}),
                            "V": runner._ranking_metrics(["1", "2"], {"1", "2"}),
                            "T+V": runner._ranking_metrics(["1", "2"], {"1", "2"})},
            })
            rows.append(row)
    report = _visual_report("unused", suite="spiqa-test-c", profile="figure")
    report.update({
        "rows": rows, "cases": len(rows), "selected_cases": len(rows),
        "selection": {"case_ids": [row["case_id"] for row in rows]},
        "aggregate": runner._aggregate_metrics(rows),
        "answer_quality": {"status": "complete", "paired_cases": len(rows),
                           "promotion_safe": True,
                           "overall": {"aya_vs_full_delta": 0.1},
                           "per_case": [{"case_id": row["case_id"], "status": "ok",
                                         "delta_aya_vs_full": 0.1} for row in rows]},
    })
    return report


@pytest.mark.parametrize("postprocessor", ["attach", "recompute"])
@pytest.mark.parametrize("violation", ["ocr_disabled", "incomplete", "posthoc", "legacy"])
def test_visual_postprocessors_cannot_launder_diagnostic_reports(
    postprocessor: str, violation: str,
) -> None:
    report = _promotion_ready_visual_report()

    def process(value: dict) -> dict:
        if postprocessor == "recompute":
            return _load_script("recompute_visual_gate_metrics").recompute(value)
        answer = {"status": "complete", "model": "cx/gpt-5.6-sol",
                  "selection": value["selection"], "answer_quality": value["answer_quality"]}
        return _load_script("attach_answer_quality_to_visual_gate").attach(value, answer)

    assert process(report)["promotion"]["passed"] is True
    if violation == "ocr_disabled":
        report["pdf_ocr_enabled"] = False
    elif violation == "incomplete":
        report["selection"]["case_ids"].append("not_attempted")
        # Even incorrect green completion flags must not hide missing rows.
        if postprocessor == "attach":
            with pytest.raises(ValueError, match="scored answer rows"):
                process(report)
            return
    elif violation == "posthoc":
        report["posthoc_selection"] = True
    else:
        report.pop("score_valid")
        report.pop("suite_complete")
    assert process(report)["promotion"]["passed"] is False


def test_visual_merge_preserves_unattempted_frontier_and_filtered_diagnostic_status(tmp_path: Path) -> None:
    merger = _load_script("merge_visual_gate_reports")
    report = _promotion_ready_visual_report()
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    sources = []
    for path, rows in zip(paths, [report["rows"][:10], report["rows"][10:]]):
        source = {**report, "rows": rows, "cases": len(rows),
                  "selection": {"case_ids": [row["case_id"] for row in rows]}}
        sources.append(source)
        path.write_text(json.dumps(source), encoding="utf-8")
    merged = merger.merge_reports(paths)
    gate = _load_script("evaluate_visual_retrieval_gate")
    # Merge intentionally does not attest answer quality itself.
    merged["answer_quality"] = report["answer_quality"]
    assert gate.visual_report_promotion_gate(merged)["passed"] is True

    sources[1]["selection"]["case_ids"].append("not_attempted")
    paths[1].write_text(json.dumps(sources[1]), encoding="utf-8")
    missing = merger.merge_reports(paths)
    missing["answer_quality"] = report["answer_quality"]
    assert "not_attempted" in missing["selection"]["case_ids"]
    assert gate.visual_report_promotion_gate(missing)["passed"] is False

    # Explicit posthoc filtering remains useful for diagnosis but cannot erase
    # the source frontier and turn the remaining successes into acceptance.
    filtered = merger.merge_reports(paths, include_case_ids=set(report["selection"]["case_ids"]))
    filtered["answer_quality"] = report["answer_quality"]
    assert filtered["posthoc_selection"] is True
    assert gate.visual_report_promotion_gate(filtered)["passed"] is False


def test_visual_gate_report_merge_requires_compatible_contract_and_recomputes_metrics(
    tmp_path: Path,
) -> None:
    merger = _load_script("merge_visual_gate_reports")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_visual_report("mmlong:1", suite="mmlongbench-doc", profile="native_text")),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_visual_report("spiqa:1", suite="spiqa-test-c", profile="figure")),
        encoding="utf-8",
    )

    merged = merger.merge_reports([first, second])
    assert merged["cases"] == 2
    assert merged["aggregate"]["T"]["positive_cases"] == 2
    assert merged["aggregate"]["V"]["p50_ms"] == 5.0
    # Fusion is a measured stage, not an unreported/null part of T+V.
    assert merged["aggregate"]["T+V"]["p50_ms"] == 16.0
    assert merged["promotion"]["checks"]["public_suite_frontier"] is True
    assert merged["promotion"]["passed"] is False

    incompatible = json.loads(second.read_text(encoding="utf-8"))
    incompatible["model_fingerprint"] = "different-model"
    second.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(merger.MergeContractError):
        merger.merge_reports([first, second])

    stale = json.loads(first.read_text(encoding="utf-8"))
    stale.pop("evaluator_fingerprint")
    first.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(merger.MergeContractError, match="evaluator_fingerprint"):
        merger.merge_reports([first, second])


def test_visual_gate_report_merge_can_select_a_disjoint_frontier_without_retrieval(
    tmp_path: Path,
) -> None:
    merger = _load_script("merge_visual_gate_reports")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                **_visual_report("keep", suite="mmlongbench-doc", profile="native_text"),
                "rows": [
                    _visual_report("keep", suite="mmlongbench-doc", profile="native_text")["rows"][0],
                    _visual_report("drop", suite="mmlongbench-doc", profile="native_text")["rows"][0],
                ],
                "cases": 2,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_visual_report("also-keep", suite="spiqa-test-c", profile="figure")),
        encoding="utf-8",
    )

    merged = merger.merge_reports(
        [first, second], include_case_ids={"keep", "also-keep"}
    )
    assert merged["selection"]["case_ids"] == ["keep", "also-keep"]
    assert merged["cases"] == 2
    assert merged["aggregate"]["T"]["positive_cases"] == 2


def test_visual_gate_metric_migration_does_not_change_rankings(tmp_path: Path) -> None:
    migration = _load_script("recompute_visual_gate_metrics")
    report = _visual_report("spiqa:1", suite="spiqa-test-c", profile="table")
    report["rows"][0]["rankings"] = {
        "T": ["1"],
        "T_artifact": ["page-1.png"],
        "T_page": ["1"],
        "V": ["page-1.png"],
        "T+V": ["page-1.png"],
    }
    report["rows"][0]["required_keys"] = ["page-1.png"]
    before = json.loads(json.dumps(report["rows"][0]["rankings"]))
    migrated = migration.recompute(report)
    row = migrated["rows"][0]
    assert row["rankings"] == before
    assert row["text_metric_scope"] == "artifact"
    assert row["metrics"]["T"]["recall"] == 1.0
    assert row["metrics"]["T_page"]["scored"] is False


def test_visual_gate_scores_text_on_page_and_visual_on_artifact_identity() -> None:
    runner = _load_script("evaluate_visual_retrieval_gate")
    library = _load_script("public_multimodal_eval_lib")
    case = library.BenchmarkCase(
        suite="spiqa-test-c",
        case_id="spiqa:test:0",
        document_id="paper",
        title="paper",
        question="Which table?",
        answers=("answer",),
        answer_format="free_form",
        referred_artifacts=("7-Table1-1.png",),
    )
    results = [{"id": "table-runtime-id", "page_number": 7}]
    kwargs = {
        "identity_to_file": {"table-runtime-id": "7-Table1-1.png"},
        "path_to_file": {},
        "file_to_page": {"7-Table1-1.png": "7"},
    }
    assert runner._text_keys(case, results, **kwargs, key_mode="page") == ["7"]
    assert runner._text_keys(case, results, **kwargs, key_mode="artifact") == [
        "7-Table1-1.png"
    ]
