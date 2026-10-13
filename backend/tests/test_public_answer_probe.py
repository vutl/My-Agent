from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
probe = importlib.import_module("evaluate_public_answer_probe")
freeze = importlib.import_module("freeze_evaluation_code")
from public_multimodal_eval_lib import BenchmarkCase


def make_case(n, stratum="table", answer="answer"):
    return BenchmarkCase(suite="spiqa-test-c", case_id=f"case-{n}", document_id=f"doc-{n}",
                         title="title", question="question", answers=(answer,),
                         answer_format="free_form", metadata={
                             "figure_in_evidence": stratum == "figure",
                             "table_in_evidence": stratum == "table"})


def test_probe_selection_is_stratified_and_answer_independent():
    cases = [make_case(i, stratum="table" if i % 2 else "figure") for i in range(12)]
    ids = [case.case_id for case in probe.select_cases(cases, 4)]
    changed = [make_case(i, stratum="table" if i % 2 else "figure", answer="different") for i in range(12)]
    assert ids == [case.case_id for case in probe.select_cases(changed, 4)]
    assert {case.stratum for case in probe.select_cases(cases, 4)} == {"table", "figure"}
    with pytest.raises(ValueError):
        probe.select_cases(cases, 13)


def test_probe_stops_after_first_provider_error_and_keeps_attempt(tmp_path, monkeypatch):
    cases = [make_case(1), make_case(2)]
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps([c.case_id for c in cases]))
    args = SimpleNamespace(model="cx/gpt-5.6-sol", case_ids_file=manifest, limit=2,
                           runtime_root=tmp_path / "runtime", output=tmp_path / "report.json",
                           max_context_chars=60000, max_context_tokens=16000, max_full_images=4,
                           timeout=30, seed=1, top_k=6, resume=False)
    settings = SimpleNamespace(llm_provider="openai_compatible", openai_api_base="http://local/v1",
                               openai_api_key="not-printed", embedding_model="embeddinggemma:300m",
                               embedding_query_prefix="q", embedding_document_prefix="d")

    class Pipeline:
        def __init__(self, **kwargs):
            self.root = tmp_path / "runtime"
            self.rerank_enabled = False
            self.rerank_mode = "embedding"
            self.indexing = SimpleNamespace(materialization_policy=lambda ext: {})

        async def preflight_local_search(self):
            return {}

        async def retrieve(self, case, **kwargs):
            return {"context": "content", "images": [("image.png", b"pixels")]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def health(self):
            return {"reachable": True, "models": [args.model]}

    calls = []

    async def fail(**kwargs):
        calls.append(kwargs["model"])
        raise RuntimeError("quota exhausted")

    monkeypatch.setattr(probe, "load_mmlong_cases", lambda **kwargs: [])
    monkeypatch.setattr(probe, "load_spiqa_test_c_cases", lambda: cases)
    monkeypatch.setattr(probe, "get_settings", lambda: settings)
    monkeypatch.setattr(probe, "source_fingerprint", lambda selected: "source")
    monkeypatch.setattr(probe, "fingerprint_files", lambda *args: "code")
    monkeypatch.setattr(probe, "_safe_runtime_root", lambda root: root)
    monkeypatch.setattr(probe, "IsolatedAyaPipeline", Pipeline)
    monkeypatch.setattr(probe, "OpenAICompatibleClient", Client)
    monkeypatch.setattr(probe, "_generate", fail)
    monkeypatch.setattr(probe, "_mode_input", lambda *args, **kwargs:
                        ("content", [], {"context_label": "label"}))
    result = asyncio.run(probe.run(args))
    assert calls == [args.model]
    assert result["complete"] is False
    assert result["inflight"] is not None
    assert result["error"] == {"type": "RuntimeError", "stage": "generation"}
    assert json.loads(args.output.read_text())["completed_generations"] == 0
    args.resume = True
    with pytest.raises(ValueError, match="failed provider"):
        asyncio.run(probe.run(args))
    assert len(calls) == 1


def test_probe_requires_approved_model_before_any_request(tmp_path):
    args = SimpleNamespace(model="unapproved")
    with pytest.raises(ValueError, match="Unapproved"):
        asyncio.run(probe.run(args))


def test_code_snapshot_excludes_secrets_and_production_data(tmp_path):
    project, target = tmp_path / "project", tmp_path / "snapshot"
    (project / "backend" / "app").mkdir(parents=True)
    (project / "backend" / "app" / "module.py").write_text("value = 1\n")
    (project / "backend" / ".env").write_text("SECRET=not-for-copy\n")
    (project / "scripts").mkdir()
    (project / "scripts" / "runner.py").write_text("value = 2\n")
    (project / "data" / "retrieval_eval" / "public").mkdir(parents=True)
    (project / "data" / "sqlite").mkdir()
    (project / "data" / "sqlite" / "app.db").write_bytes(b"private")
    report = freeze.freeze_code(project, target)
    assert not (target / "backend" / ".env").exists()
    assert not (target / "data" / "sqlite").exists()
    assert (target / "data" / "retrieval_eval" / "public").resolve() == (
        project / "data" / "retrieval_eval" / "public").resolve()
    assert report["production_data_linked"] is False
    with pytest.raises(FileExistsError):
        freeze.freeze_code(project, target)
