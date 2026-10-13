from __future__ import annotations

import importlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
checkpoint = importlib.import_module("agent_evaluation_checkpoint")
evaluator = importlib.import_module("evaluate_agent_conversations")
cloner = importlib.import_module("clone_agent_evaluation_runtime")
completeness = importlib.import_module("audit_agent_answer_completeness")


def test_partial_answer_is_reviewed_even_when_internal_validator_passes():
    row = {"answer": "Insufficient evidence is available for the removed results.",
           "table_ids": ["table-b"], "answer_validation": {"valid": True, "fallback_used": False}}
    assert completeness.structural_findings(row["answer"]) == []
    assert completeness.guarded_answer_findings(row)[0]["reason"] == "partial_answer_with_retrieved_artifacts"
    # This flags review, not proof of failure: retrieved tables may genuinely
    # be irrelevant to the user's question.
    assert completeness.guarded_answer_findings({**row, "table_ids": []}) == []
    assert completeness.guarded_answer_findings({**row, "answer": "Đã có bảng phù hợp."}) == []


def test_answer_fallback_is_not_hidden_by_clean_markdown():
    row = {"answer": "No evidence available.", "answer_validation": {"fallback_used": True}}
    assert completeness.guarded_answer_findings(row) == [{"reason": "answer_fallback_requires_review"}]


def case(n=1, group="g"):
    return {"case_id": f"{group}-{n}", "conversation_group": group, "turn_index": n}


def fake_get(path):
    if path.startswith("/agent/runs/"):
        return {"status": "completed", "conversation_id": "conversation", "final_answer": "answer"}
    return [{"id": "message", "content": "answer", "role": "assistant"}]


def finished(tmp_path, *, ok=True):
    path = tmp_path / "report.partial"
    cp = checkpoint.AgentCheckpoint(path, contract={"code": "v1"}, cases=[case()],
                                    resume=False, get_json=fake_get)
    cp.begin(case(), None)
    cp.started({"run_id": "run", "conversation_id": "conversation"})
    cp.finish({**case(), "run_id": "run", "ok": ok, "elapsed_ms": 123,
               "client_first_delta_ms": 32}, conversation_id="conversation",
              answer="answer", completed=True)
    return cp


def test_resume_preserves_measured_timings_and_quality_failures(tmp_path):
    old = finished(tmp_path, ok=False)
    resumed = checkpoint.AgentCheckpoint(old.path, contract=old.contract, cases=[case()],
                                         resume=True, get_json=fake_get)
    assert resumed.rows == old.rows
    assert resumed.resumed_cases == 1
    assert resumed.rows[0]["ok"] is False
    assert resumed.rows[0]["client_first_delta_ms"] == 32


@pytest.mark.parametrize("change", ["contract", "history", "answer", "status", "missing_binding"])
def test_resume_rejects_stale_contract_or_server_history(tmp_path, change):
    old = finished(tmp_path)
    contract = old.contract if change != "contract" else {"code": "v2"}
    if change == "missing_binding":
        payload = json.loads(old.path.read_text())
        payload["histories"] = {}
        old.path.write_text(json.dumps(payload))

    def get(path):
        value = fake_get(path)
        if change == "history" and isinstance(value, list):
            return []
        if isinstance(value, dict):
            if change == "answer":
                value["final_answer"] = "different answer"
            if change == "status":
                value["status"] = "running"
        return value

    with pytest.raises(checkpoint.AgentCheckpointError):
        checkpoint.AgentCheckpoint(old.path, contract=contract, cases=[case()],
                                    resume=True, get_json=get)


def test_interruption_records_inflight_and_forbids_silent_replay(tmp_path):
    cp = checkpoint.AgentCheckpoint(tmp_path / "partial", contract={}, cases=[case()],
                                    resume=False, get_json=fake_get)
    cp.begin(case(), None)
    cp.started({"run_id": "run", "conversation_id": "conversation"})
    assert json.loads(cp.path.read_text())["inflight"]["run_id"] == "run"
    with pytest.raises(checkpoint.AgentCheckpointError, match="in flight"):
        checkpoint.AgentCheckpoint(cp.path, contract={}, cases=[case()],
                                    resume=True, get_json=fake_get)


def test_persisted_answer_mismatch_keeps_turn_inflight(tmp_path):
    cp = checkpoint.AgentCheckpoint(tmp_path / "partial", contract={}, cases=[case()],
                                    resume=False, get_json=fake_get)
    cp.begin(case(), None)
    with pytest.raises(checkpoint.AgentCheckpointError, match="identically"):
        cp.finish({**case(), "run_id": "run"}, conversation_id="conversation",
                  answer="wrong", completed=True)
    assert cp.rows == []
    assert json.loads(cp.path.read_text())["inflight"] is not None


def test_followup_selection_includes_context_not_unrelated_groups():
    cases = [case(1), case(2), case(3), case(1, "other")]
    assert evaluator._select_with_context(cases, {"g-2"}) == cases[:2]
    with pytest.raises(ValueError, match="Unknown"):
        evaluator._select_with_context(cases, {"missing"})


def create_source(root):
    (root / "sqlite").mkdir(parents=True)
    db = root / "sqlite" / "app.db"
    with sqlite3.connect(db) as c:
        c.executescript("CREATE TABLE documents(id TEXT); INSERT INTO documents VALUES('doc');"
                        "CREATE TABLE conversations(id TEXT PRIMARY KEY,title TEXT,created_at TEXT,"
                        "updated_at TEXT,metadata_json TEXT);"
                        "CREATE TABLE document_processing_jobs(id TEXT);"
                        "INSERT INTO document_processing_jobs VALUES('pending');")
    (root / "lancedb").mkdir()
    (root / "artifacts").mkdir()
    return db


def test_clone_preserves_source_and_refuses_overwrite(tmp_path):
    source, target = tmp_path / "source", tmp_path / "clone"
    source_db = create_source(source)
    cloner.clone_runtime(source, target)
    with sqlite3.connect(source_db) as c:
        assert c.execute("SELECT count(*) FROM document_processing_jobs").fetchone()[0] == 1
    with sqlite3.connect(target / "sqlite" / "app.db") as c:
        assert c.execute("SELECT count(*) FROM document_processing_jobs").fetchone()[0] == 0
    with pytest.raises(FileExistsError):
        cloner.clone_runtime(source, target)


def test_runtime_binding_is_verified_and_marker_cleaned(tmp_path):
    root, project = tmp_path / "clone", tmp_path / "project"
    db = create_source(root)

    def visible_marker(path):
        assert path == "/chat/conversations"
        with sqlite3.connect(db) as c:
            return [{"id": row[0]} for row in c.execute("SELECT id FROM conversations")]

    checkpoint.verify_runtime_binding(root, project, visible_marker)
    with pytest.raises(checkpoint.AgentCheckpointError, match="not serving"):
        checkpoint.verify_runtime_binding(root, project, lambda path: [])
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM conversations").fetchone()[0] == 0


def test_fingerprint_ignores_chat_but_includes_images(tmp_path):
    db = create_source(tmp_path / "clone")
    root = db.parent.parent
    original = checkpoint.runtime_fingerprint(root)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO conversations(id) VALUES('new-chat')")
    assert checkpoint.runtime_fingerprint(root) == original
    (root / "artifacts" / "image.png").write_bytes(b"new-image")
    assert checkpoint.runtime_fingerprint(root) != original


def test_production_alias_refused_before_marker_write(tmp_path):
    project = tmp_path / "project"
    create_source(project / "data")
    with pytest.raises(checkpoint.AgentCheckpointError, match="isolated"):
        checkpoint.verify_runtime_binding(project / "data", project, lambda path: [])


def test_completeness_diagnostic_detects_empty_tables_and_dangling_sections():
    answer = "| Method | F1 |\n|---|---|\n\n### Conclusion"
    reasons = {item["reason"] for item in completeness.structural_findings(answer)}
    assert reasons == {"markdown_table_without_data_rows", "dangling_final_heading"}


def test_completeness_diagnostic_accepts_rows_and_ignores_code_examples():
    answer = "| Method | F1 |\n|---|---|\n| M | 0.5 |\n\n### Conclusion\nSupported."
    assert completeness.structural_findings(answer) == []
    assert completeness.structural_findings("```md\n| A | B |\n|---|---|\n### Heading\n```") == []


def test_completeness_diagnostic_surfaces_empty_result_rows_for_review():
    answer = "| Model | Results |\n|---|---|\n| ORION ||\n| VEGA | F1 0.5 |"
    findings = completeness.structural_findings(answer)
    assert [item["reason"] for item in findings] == ["markdown_table_empty_data_row"]
    assert "ORION" in findings[0]["excerpt"]
    assert completeness.structural_findings("```md\n" + answer + "\n```") == []


def test_completeness_diagnostic_requires_answers_and_preserves_source(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"results": [{"case_id": "case"}]}))
    with pytest.raises(ValueError, match="include-answers"):
        completeness.audit(path)
    path.write_text(json.dumps({"results": [{"case_id": "case", "answer": "### Empty"}]}))
    original = path.read_bytes()
    report = completeness.audit(path)
    assert path.read_bytes() == original
    assert report["affected_cases"] == 1
    assert report["semantic_quality_verified"] is False
