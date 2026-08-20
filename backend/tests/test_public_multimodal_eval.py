from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
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


def test_public_multimodal_sampling_is_order_independent_and_balanced() -> None:
    library = _load_script("public_multimodal_eval_lib")
    cases = library.load_mmlong_cases()
    forward = library.stable_balanced_sample(cases, limit=12)
    reverse = library.stable_balanced_sample(reversed(cases), limit=12)

    assert [case.case_id for case in forward] == [case.case_id for case in reverse]
    assert len({case.stratum for case in forward}) >= 4
    assert len({case.document_id for case in forward}) >= 4


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


def test_bounded_context_labels_truncation_without_claiming_full() -> None:
    library = _load_script("public_multimodal_eval_lib")
    bounded = library.bounded_context("a" * 10_000, max_chars=1_000, label="full")

    assert bounded.truncated is True
    assert bounded.original_chars == 10_000
    assert bounded.included_chars <= 1_000
    assert "TRUNCATED" in bounded.text


def test_spiqa_zip_reader_rejects_traversal_and_returns_exact_member() -> None:
    library = _load_script("public_multimodal_eval_lib")
    case = next(case for case in library.load_spiqa_test_c_cases() if case.referred_artifacts)
    name = case.referred_artifacts[0]
    images = library.read_spiqa_images(case, [name])

    assert images[0][0] == name
    assert images[0][1].startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsafe SPIQA image"):
        library.read_spiqa_images(case, ["../secret.png"])


def test_official_mmlong_scorer_wrapper_matches_upstream_contract() -> None:
    library = _load_script("public_multimodal_eval_lib")

    assert library.official_mmlong_score(gold="18.29%", prediction="18.29", answer_format="Float") == 1.0
    assert library.official_mmlong_score(gold="42", prediction="42.0", answer_format="Int") == 1.0
    assert library.official_mmlong_score(gold="alpha", prediction="unrelated", answer_format="Str") == 0.0
    assert library.official_mmlong_score(
        gold="['A', 'B']", prediction='["B", "A"]', answer_format="List"
    ) == 1.0


def test_public_eval_output_and_runtime_paths_fail_closed(tmp_path: Path) -> None:
    library = _load_script("public_multimodal_eval_lib")
    runner = _load_script("evaluate_public_multimodal_three_mode")

    with pytest.raises(ValueError, match="under"):
        library.ensure_public_output_path(tmp_path / "report.json")
    with pytest.raises(ValueError, match="under"):
        runner._safe_runtime_root(tmp_path / "runtime")  # noqa: SLF001


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
            "SELECT caption, image_path, metadata_json FROM document_figures WHERE document_id = ?",
            (document.id,),
        ).fetchall()

    assert len(rows) == len(case.artifacts)
    assert len(rows) > len(case.referred_artifacts)
    assert all(Path(row[1]).is_file() for row in rows)
    assert all("spiqa_official_artifact_adapter" in row[2] for row in rows)
