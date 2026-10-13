from __future__ import annotations

import importlib.util
import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script() -> ModuleType:
    path = SCRIPTS / "semantic_audit_retrieval_eval_v22.py"
    spec = importlib.util.spec_from_file_location("semantic_audit_retrieval_eval_v22", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_audit_fingerprint_includes_suite_model_batch_and_prompt() -> None:
    runner = _load_script()
    base = runner._audit_config_fingerprint(  # noqa: SLF001
        suite_sha256="suite-a",
        model="cx/gpt-5.6-sol",
        expected_cases=60,
        batch_size=8,
    )
    assert base == runner._audit_config_fingerprint(  # noqa: SLF001
        suite_sha256="suite-a",
        model="cx/gpt-5.6-sol",
        expected_cases=60,
        batch_size=8,
    )
    assert base != runner._audit_config_fingerprint(  # noqa: SLF001
        suite_sha256="suite-b",
        model="cx/gpt-5.6-sol",
        expected_cases=60,
        batch_size=8,
    )
    assert base != runner._audit_config_fingerprint(  # noqa: SLF001
        suite_sha256="suite-a",
        model="cx/gpt-5.6-sol",
        expected_cases=60,
        batch_size=4,
    )


def test_semantic_audit_checkpoint_loads_successes_and_retries_failures(tmp_path: Path) -> None:
    runner = _load_script()
    path = tmp_path / "audit.partial"
    fingerprint = runner._audit_config_fingerprint(  # noqa: SLF001
        suite_sha256="suite-a",
        model="cx/gpt-5.6-sol",
        expected_cases=2,
        batch_size=1,
    )
    runner._write_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        suite_sha256="suite-a",
        model="cx/gpt-5.6-sol",
        expected_cases=2,
        batch_size=1,
        reviews=[{"case_id": "case-1", "status": "accept"}],
        failed_batches=[
            {"batch_index": 1, "case_ids": ["case-2"], "error_type": "TimeoutError"}
        ],
    )

    reviews, failures = runner._load_checkpoint(  # noqa: SLF001
        path,
        config_fingerprint=fingerprint,
        expected_case_ids={"case-1", "case-2"},
    )
    assert reviews == [{"case_id": "case-1", "status": "accept"}]
    assert failures[0]["batch_index"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_semantic_audit_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    runner = _load_script()
    path = tmp_path / "audit.partial"
    path.write_text(
        json.dumps(
            {
                "schema_version": runner.CHECKPOINT_SCHEMA_VERSION,
                "config_fingerprint": "old",
                "reviews": [],
                "failed_batches": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config mismatch"):
        runner._load_checkpoint(  # noqa: SLF001
            path,
            config_fingerprint="new",
            expected_case_ids=set(),
        )


def test_semantic_audit_resume_retries_only_failed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_script()
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": f"case-{index}",
                    "curation_status": "proposal",
                    "category": "text",
                    "language": "en",
                    "raw_query": "What?",
                    "evidence_groups": [],
                }
            )
            for index in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        suite=suite,
        output=output,
        checkpoint=None,
        model="cx/gpt-5.6-sol",
        expected_cases=2,
        batch_size=1,
        concurrency=1,
        timeout=1.0,
        resume=True,
    )

    class FakeClient:
        async def health(self):
            return {"reachable": True, "models": [args.model]}

    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: SimpleNamespace(openai_api_base="http://test/v1", openai_api_key="test"),
    )
    monkeypatch.setattr(runner, "OpenAICompatibleClient", lambda **_kwargs: FakeClient())
    calls: list[str] = []
    failed_once = {"case-2": False}

    async def fake_review_batch(*, rows, **_kwargs):
        case_id = str(rows[0]["case_id"])
        calls.append(case_id)
        if case_id == "case-2" and not failed_once[case_id]:
            failed_once[case_id] = True
            raise TimeoutError("simulated timeout")
        return {"cases": [{"case_id": case_id, "status": "accept", "evidence": []}]}

    monkeypatch.setattr(runner, "_review_batch", fake_review_batch)

    first = asyncio.run(runner.run(args))
    assert first["status"] == "partial"
    assert first["reviewed_cases"] == 1
    assert first["failed_batches"][0]["case_ids"] == ["case-2"]
    assert calls == ["case-1", "case-2"]

    second = asyncio.run(runner.run(args))
    assert second["status"] == "complete"
    assert second["reviewed_cases"] == 2
    assert second["failed_batches"] == []
    assert calls == ["case-1", "case-2", "case-2"]

    args.resume = False
    fresh = asyncio.run(runner.run(args))
    assert fresh["status"] == "complete"
    assert calls == ["case-1", "case-2", "case-2", "case-1", "case-2"]
