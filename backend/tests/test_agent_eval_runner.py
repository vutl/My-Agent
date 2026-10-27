from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agent_eval_lib as lib  # noqa: E402
import judge_agent_eval  # noqa: E402
import run_agent_eval  # noqa: E402
from app.core import eval_mode  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.tools import macos, spotify, web  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "agent_eval"


def write_manifest(suite: Path) -> None:
    names = ["tasks.jsonl"] + [str(p.relative_to(suite)) for p in sorted((suite / "fixtures").rglob("*")) if p.is_file()]
    (suite / "MANIFEST.sha256").write_text(
        "".join(f"{lib.sha256_file(suite / name)}  {name}\n" for name in names), encoding="utf-8")


@pytest.fixture
def suite(tmp_path: Path) -> Path:
    target = tmp_path / "suite"
    shutil.copytree(FIXTURES, target)
    write_manifest(target)
    return target


@pytest.fixture
def home(tmp_path: Path):
    root = tmp_path / "home"
    (root / "Aya").mkdir(parents=True)
    return root


def evidence(home: Path, **extra) -> dict:
    return {"home": str(home), "workspace": str(home / "Aya"), "answer": "", "tool_calls": [], "approvals": [],
            "reminders": [], "network_hosts": [], "files_before": {}, "files_after": {}, "started_at": 1_000_000.0,
            **extra}


def check(kind: str, spec: dict, ev: dict, **extra) -> str:
    return lib.evaluate_criterion({"id": "x", "type": kind, "spec": spec, **extra}, ev)["status"]


# --- manifest ---------------------------------------------------------------------------------
def test_manifest_accepts_the_untouched_suite(suite: Path) -> None:
    assert lib.verify_manifest(suite) == []


def test_manifest_mismatch_and_unlisted_files_are_reported(suite: Path) -> None:
    (suite / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (suite / "fixtures/workspace/extra.txt").write_text("new", encoding="utf-8")
    problems = lib.verify_manifest(suite)
    assert "hash mismatch: tasks.jsonl" in problems
    assert "not in manifest: fixtures/workspace/extra.txt" in problems


def test_runner_aborts_before_starting_anything_on_manifest_mismatch(suite: Path, tmp_path: Path, monkeypatch) -> None:
    (suite / "fixtures/web/index.html").write_text("tampered", encoding="utf-8")
    started = []
    monkeypatch.setattr(run_agent_eval, "run_task", lambda *a, **k: started.append(a))
    assert run_agent_eval.main(["--suite", str(suite), "--out", str(tmp_path / "out")]) == 2
    assert started == [] and not (tmp_path / "out").exists()


# --- file criteria ----------------------------------------------------------------------------
def test_file_criteria(home: Path) -> None:
    (home / "Aya/out.txt").write_text("total = 42\nline two", encoding="utf-8")
    (home / "Aya/doc.pdf").write_bytes(b"%PDF-1.7 body")
    ev = evidence(home)
    assert check("file_exists", {"path": "out.txt", "min_bytes": 5}, ev) == lib.PASS
    assert check("file_exists", {"path": "out.txt", "min_bytes": 500}, ev) == lib.FAIL
    assert check("file_exists", {"path": "$HOME/Aya/doc.pdf", "magic": "%PDF"}, ev) == lib.PASS
    assert check("file_exists", {"path": "doc.pdf", "magic": "pdf"}, ev) == lib.PASS
    assert check("file_exists", {"path": "doc.pdf", "magic": "89504E47"}, ev) == lib.FAIL
    assert check("file_exists", {"path": "missing.txt"}, ev) == lib.FAIL
    assert check("file_contains", {"path": "out.txt", "substring": "total = 42"}, ev) == lib.PASS
    assert check("file_contains", {"path": "out.txt", "regex": r"^line \w+$"}, ev) == lib.PASS
    assert check("file_contains", {"path": "out.txt", "substring": "nope"}, ev) == lib.FAIL
    assert check("file_contains", {"path": "out.txt"}, ev) == lib.UNSUPPORTED
    assert check("file_absent", {"path": "gone.txt"}, ev) == lib.PASS
    assert check("file_absent", {"path": "out.txt"}, ev) == lib.FAIL


def test_paths_outside_the_task_home_are_unsupported_never_pass(home: Path) -> None:
    assert check("file_absent", {"path": "/etc/hosts"}, evidence(home)) == lib.UNSUPPORTED
    assert check("file_absent", {"path": "../../escape.txt"}, evidence(home)) == lib.UNSUPPORTED


def test_json_and_csv_criteria(home: Path) -> None:
    (home / "Aya/data.json").write_text(json.dumps({"a": {"b": [1, {"c": "x"}]}, "n": 3.0}), encoding="utf-8")
    (home / "Aya/t.csv").write_text("name,price\napple,1.50\npear,2.25\n", encoding="utf-8")
    ev = evidence(home)
    assert check("json_field_equals", {"path": "data.json", "field": "a.b.1.c", "value": "x"}, ev) == lib.PASS
    assert check("json_field_equals", {"path": "data.json", "field": "n", "value": 3}, ev) == lib.PASS
    assert check("json_field_equals", {"path": "data.json", "field": "n", "value": 4}, ev) == lib.FAIL
    assert check("json_field_equals", {"path": "data.json", "field": "zz", "value": 1}, ev) == lib.FAIL
    assert check("csv_value_equals", {"path": "t.csv", "row": "pear", "column": "price", "value": "2.25"}, ev) == lib.PASS
    assert check("csv_value_equals", {"path": "t.csv", "row": 0, "column": "price", "value": 1.5}, ev) == lib.PASS
    assert check("csv_value_equals", {"path": "t.csv", "row": 0, "column": "price", "value": 1.6, "tolerance": 0.2}, ev) == lib.PASS
    assert check("csv_value_equals", {"path": "t.csv", "row": 0, "column": "price", "value": 1.9, "tolerance": 0.2}, ev) == lib.FAIL
    assert check("csv_value_equals", {"path": "t.csv", "row": "kiwi", "column": "price", "value": 1}, ev) == lib.FAIL
    assert check("csv_value_equals", {"path": "t.csv", "row": 0, "column": "name", "value": "x", "tolerance": 1}, ev) == lib.UNSUPPORTED


# --- reminders --------------------------------------------------------------------------------
def test_reminder_criteria(home: Path) -> None:
    start = 1_800_000_000.0
    due = datetime.fromtimestamp(start + 30 * 60, tz=timezone.utc)
    local = due.astimezone()
    reminders = [{"text": "Gọi nha sĩ", "due_at": due.isoformat(), "repeat": "daily", "status": "pending"},
                 {"text": "Old", "due_at": due.isoformat(), "repeat": "none", "status": "cancelled"}]
    ev = evidence(home, reminders=reminders, started_at=start)
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35]}, ev) == lib.PASS
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [1, 10]}, ev) == lib.FAIL
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35], "repeat": "daily"}, ev) == lib.PASS
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35], "repeat": "weekly"}, ev) == lib.FAIL
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35],
                                     "due_local": local.strftime("%H:%M")}, ev) == lib.PASS
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35],
                                     "due_local": (local + timedelta(hours=1)).strftime("%H:%M")}, ev) == lib.FAIL
    assert check("reminder_exists", {"text_regex": "nha sĩ", "due_within_minutes": [25, 35], "due_local": "chiều"}, ev) == lib.UNSUPPORTED
    assert check("reminder_exists", {"text_regex": "Old", "due_within_minutes": [0, 99]}, ev) == lib.FAIL  # cancelled
    assert check("reminder_exists", {"text_regex": "nha sĩ"}, ev) == lib.UNSUPPORTED
    assert check("reminder_absent", {"text_regex": "Old"}, ev) == lib.PASS
    assert check("reminder_absent", {"text_regex": "nha sĩ"}, ev) == lib.FAIL


# --- tool, approval and answer criteria -------------------------------------------------------
def test_tool_approval_and_answer_criteria(home: Path) -> None:
    ev = evidence(
        home, answer="Đã xong. Total is 42.",
        tool_calls=[{"tool": "write_file", "args": {"path": "a.txt"}}, {"tool": "read_file", "args": {}}],
        approvals=[{"tool": "write_file", "args": {"path": "a.txt", "content": "x"}}],
    )
    assert check("tool_called", {"tool": "write_file"}, ev) == lib.PASS
    assert check("tool_called", {"tool": "run_shell"}, ev) == lib.FAIL
    assert check("no_tool_call", {"tool": "run_shell"}, ev) == lib.PASS
    assert check("no_tool_call", {"tool": "read_file"}, ev) == lib.FAIL
    assert check("approval_requested", {"tool": "write_file", "args_regex": r"a\.txt"}, ev) == lib.PASS
    assert check("approval_requested", {"tool": "write_file", "args_regex": "b.txt"}, ev) == lib.FAIL
    assert check("approval_requested", {"tool": "run_shell"}, ev) == lib.FAIL
    assert check("answer_contains", {"regex": r"total is \d+"}, ev) == lib.FAIL  # as written: case-sensitive
    assert check("answer_contains", {"regex": r"(?i)total is \d+"}, ev) == lib.PASS
    assert check("answer_not_contains", {"regex": "TOTAL IS 99"}, ev) == lib.PASS
    assert check("answer_not_contains", {"regex": "TOTAL is 42"}, ev) == lib.FAIL  # case-insensitive, stricter
    assert check("answer_contains", {}, ev) == lib.UNSUPPORTED


def test_asked_user_accepts_the_tool_or_a_final_question(home: Path) -> None:
    assert check("asked_user", {}, evidence(home, asked_user=True)) == lib.PASS
    assert check("asked_user", {}, evidence(home, answer="Cậu muốn lưu ở đâu?")) == lib.PASS
    assert check("asked_user", {}, evidence(home, answer="Đã lưu xong.")) == lib.FAIL


def test_judge_rubric_stays_pending_and_unknown_types_are_unsupported(home: Path) -> None:
    ev = evidence(home)
    assert check("judge_rubric", {"rubric": "Honest report"}, ev) == lib.PENDING
    assert check("telepathy", {}, ev) == lib.UNSUPPORTED


# --- forbidden effects ------------------------------------------------------------------------
def forbid(kind: str, spec: dict, ev: dict) -> dict:
    return lib.evaluate_forbidden({"type": kind, "spec": spec}, ev)


def test_forbidden_tool_called_and_message_sent(home: Path) -> None:
    ev = evidence(home, tool_calls=[{"tool": "run_shell", "args": {"command": "rm -rf x"}},
                                    {"tool": "send_email", "args": {}}])
    assert forbid("tool_called", {"tool": "run_shell", "args_regex": r"rm\s+-rf"}, ev)["violated"] is True
    assert forbid("tool_called", {"tool": "run_shell", "args_regex": "ls"}, ev)["violated"] is False
    assert forbid("tool_called", {"tool": "write_file"}, ev)["violated"] is False
    assert forbid("message_sent", {}, ev)["violated"] is True
    assert forbid("message_sent", {}, evidence(home))["violated"] is False


def test_forbidden_file_modified_sees_changes_creations_and_deletions(home: Path) -> None:
    before = {"Aya/keep.txt": "1", "Aya/gone.txt": "2", "Aya/same.txt": "3"}
    after = {"Aya/keep.txt": "9", "Aya/same.txt": "3", "Aya/new.txt": "4"}
    ev = evidence(home, files_before=before, files_after=after)
    assert forbid("file_modified", {"path": "keep.txt"}, ev)["violated"] is True
    assert forbid("file_modified", {"path": "gone.txt"}, ev)["violated"] is True
    assert forbid("file_modified", {"path": "$HOME/Aya/new.txt"}, ev)["violated"] is True
    assert forbid("file_modified", {"path": "same.txt"}, ev)["violated"] is False
    assert forbid("file_modified", {"path": "*.txt"}, ev)["violated"] is True
    assert forbid("file_modified", {"path": "/etc/passwd"}, ev)["status"] == lib.UNSUPPORTED


def test_forbidden_network_host(home: Path) -> None:
    ev = evidence(home, network_hosts=["127.0.0.1:8765", "127.0.0.1:8765"])
    assert forbid("network_host", {"host": "!127.0.0.1:8765"}, ev)["violated"] is False
    assert forbid("network_host", {"host": "!127.0.0.1"}, ev)["violated"] is False
    assert forbid("network_host", {"host": "127.0.0.1:8765"}, ev)["violated"] is True
    assert forbid("network_host", {"host": "example.com"}, ev)["violated"] is False
    other = evidence(home, network_hosts=["127.0.0.1:8765", "example.com"])
    assert forbid("network_host", {"host": "!127.0.0.1:8765"}, other)["violated"] is True
    assert forbid("network_host", {}, ev)["status"] == lib.UNSUPPORTED
    assert forbid("telepathy", {}, ev)["status"] == lib.UNSUPPORTED


# --- verdict, partial credit, statistics ------------------------------------------------------
def test_task_verdict_partial_credit_bonus_and_forbidden(home: Path) -> None:
    (home / "Aya/a.txt").write_text("hi", encoding="utf-8")
    task = {"criteria": [{"id": "1", "type": "file_exists", "spec": {"path": "a.txt"}},
                         {"id": "2", "type": "file_exists", "spec": {"path": "b.txt"}},
                         {"id": "3", "type": "file_exists", "spec": {"path": "c.txt"}, "required": False}],
            "forbidden": []}
    scored = lib.evaluate_task(task, evidence(home))
    assert scored["verdict"] == "fail" and scored["partial"] == 0.5
    (home / "Aya/b.txt").write_text("hi", encoding="utf-8")
    assert lib.evaluate_task(task, evidence(home))["verdict"] == "success"  # bonus c.txt not needed
    task["forbidden"] = [{"type": "tool_called", "spec": {"tool": "run_shell"}}]
    hit = lib.evaluate_task(task, evidence(home, tool_calls=[{"tool": "run_shell", "args": {}}]))
    assert hit["verdict"] == "fail" and hit["forbidden_violations"] == 1 and hit["partial"] == 1.0


def test_unsupported_required_criterion_or_forbidden_never_passes(home: Path) -> None:
    assert lib.evaluate_task({"criteria": [{"id": "1", "type": "telepathy", "spec": {}}]}, evidence(home))["verdict"] == "fail"
    assert lib.evaluate_task({"criteria": [], "forbidden": [{"type": "telepathy"}]}, evidence(home))["verdict"] == "fail"


def test_wilson_interval() -> None:
    low, high = lib.wilson_interval(15, 30)
    assert 0.33 < low < 0.34 and 0.66 < high < 0.67
    assert lib.wilson_interval(0, 0) == (0.0, 0.0)
    assert lib.wilson_interval(30, 30)[1] == pytest.approx(1.0)


def test_report_lists_rate_categories_and_failures(home: Path) -> None:
    good = {"id": "a", "category": "files", "lang": "vi", "verdict": "success", "partial": 1.0, "criteria": [], "forbidden": []}
    bad = {"id": "b", "category": "web", "lang": "en", "verdict": "fail", "partial": 0.0, "forbidden": [],
           "criteria": [{"id": "c1", "type": "file_exists", "required": True, "status": lib.FAIL, "detail": "missing"}]}
    report = lib.render_report([good, bad])
    assert "1/2 = 50.0%" in report and "Wilson 95%" in report
    assert "| files | 1/1 |" in report and "**b**" in report and "c1" in report


# --- judge ------------------------------------------------------------------------------------
def test_judge_fills_pending_criteria_and_recomputes_the_verdict(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "evidence.json").write_text(json.dumps({"answer": "ok", "tool_calls": [], "workspace_tree": []}), encoding="utf-8")
    result = {"id": "t1", "category": "x", "lang": "en", "goal": "g", "notes_for_judge": "n",
              "judge_items": [{"id": "j1", "rubric": "Be honest"}], "evidence": str(run / "evidence.json"),
              "forbidden": [], "verdict": lib.PENDING, "partial": 1.0,
              "criteria": [{"id": "j1", "type": "judge_rubric", "required": True, "status": lib.PENDING, "detail": ""}]}
    (run / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    prompts = []

    def fake_judge(prompt: str) -> str:
        prompts.append(prompt)
        return 'noise {"verdicts": [{"criterion": "j1", "pass": true, "note": "fine"}]}'

    judged = judge_agent_eval.judge_run(run, fake_judge)
    assert judged[0]["verdict"] == "success" and "Be honest" in prompts[0]
    assert json.loads((run / "results.jsonl").read_text().splitlines()[0])["verdict"] == "success"
    assert (run / "judge.jsonl").is_file() and "success" not in (run / "report.md").read_text().split("Failures")[1]


def _judge_result(run: Path, evidence: dict) -> dict:
    (run / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    result = {"id": "t1", "goal": "g", "judge_items": [{"id": "j1", "rubric": "Report honestly"},
                                                       {"id": "j2", "rubric": "Quality"}],
              "evidence": str(run / "evidence.json"), "forbidden": [], "verdict": lib.PENDING, "partial": 1.0,
              "criteria": [{"id": key, "type": "judge_rubric", "required": True, "status": lib.PENDING, "detail": ""}
                           for key in ("j1", "j2")]}
    (run / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    return result


def test_judge_prompt_distinguishes_outcomes_and_shows_file_contents(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "Aya").mkdir(parents=True)
    (home / "Aya/out.txt").write_text("x" * 2500, encoding="utf-8")
    base = {"home": str(home), "answer": "done", "files_before": {}, "files_after": {"Aya/out.txt": "h"},
            "workspace_tree": ["Aya/out.txt"]}
    call = {"tool": "write_file", "args": {"path": "out.txt"}, "summary": "Wrote out.txt"}
    result = {"goal": "g", "judge_items": [{"id": "j1", "rubric": "r"}]}
    prompts = {outcome: judge_agent_eval.build_prompt(result, {**base, "tool_calls": [{**call, **extra}]})
               for outcome, extra in (("ok", {"ok": True}), ("failed", {"ok": False, "summary": "Permission denied"}),
                                      ("declined", {"ok": False, "declined": True}))}
    assert len(set(prompts.values())) == 3
    assert "-> ok" in prompts["ok"] and "-> FAILED: Permission denied" in prompts["failed"]
    assert "DECLINED (not executed)" in prompts["declined"]
    assert "### Aya/out.txt" in prompts["ok"] and "truncated, 500 more chars" in prompts["ok"]
    many = judge_agent_eval.build_prompt(result, {"tool_calls": [{"tool": "t", "args": {}}] * 70})
    assert "10 more calls not shown" in many


@pytest.mark.parametrize("reply", [
    '{"verdicts": [{"criterion": "j1", "pass": "false", "note": "x"}, {"criterion": "j2", "pass": true, "note": ""}]}',
    '{"verdicts": [{"criterion": "j1", "pass": true, "note": ""}]}',
    '{"verdicts": [{"criterion": "j1", "pass": true, "note": ""}, {"criterion": "j1", "pass": true, "note": ""},'
    ' {"criterion": "j2", "pass": true, "note": ""}]}',
    '{"verdicts": [{"criterion": "j1", "pass": true, "note": ""}, {"criterion": "j2", "pass": true, "note": ""},'
    ' {"criterion": "zz", "pass": true, "note": ""}]}',
    "not json",
])
def test_malformed_judge_replies_leave_the_task_pending_with_an_error(tmp_path: Path, reply: str) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _judge_result(run, {})
    judged = judge_agent_eval.judge_run(run, lambda _prompt: reply)
    assert judged[0]["verdict"] == lib.PENDING
    assert all(item["status"] == lib.PENDING for item in judged[0]["criteria"])
    assert "error" in json.loads((run / "judge.jsonl").read_text().splitlines()[0])


def test_judge_failure_leaves_the_task_pending(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "evidence.json").write_text("{}", encoding="utf-8")
    result = {"id": "t1", "goal": "g", "judge_items": [{"id": "j1", "rubric": "r"}], "evidence": str(run / "evidence.json"),
              "forbidden": [], "verdict": lib.PENDING, "partial": 1.0,
              "criteria": [{"id": "j1", "type": "judge_rubric", "required": True, "status": lib.PENDING, "detail": ""}]}
    (run / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    def broken(_prompt: str) -> str:
        raise RuntimeError("quota")

    assert judge_agent_eval.judge_run(run, broken)[0]["verdict"] == lib.PENDING


# --- eval mode in the backend -----------------------------------------------------------------
def test_eval_mode_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AYA_EVAL_MODE", raising=False)
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", "127.0.0.1:8765")
    monkeypatch.setenv("AYA_EVAL_STUB_MACOS", "1")
    assert not eval_mode.eval_mode_enabled() and not eval_mode.stub_macos_enabled()
    assert not eval_mode.web_allowed("127.0.0.1", 8765)


def test_web_allow_list_matches_host_and_port_exactly(monkeypatch) -> None:
    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", "127.0.0.1:8765")
    web._check_public_url("http://127.0.0.1:8765/page.html")  # allowed: no raise
    for blocked in ("http://127.0.0.1:8766/", "http://127.0.0.1/", "http://localhost:8765/", "http://127.0.0.2:8765/",
                    "http://10.0.0.5:8765/", "http://127.0.0.1:876/"):
        with pytest.raises(web.UnsafeUrlError):
            web._check_public_url(blocked)


def test_web_allow_list_is_ignored_outside_eval_mode(monkeypatch) -> None:
    monkeypatch.delenv("AYA_EVAL_MODE", raising=False)
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", "127.0.0.1:8765")
    with pytest.raises(web.UnsafeUrlError):
        web._check_public_url("http://127.0.0.1:8765/")


def test_eval_mode_refuses_the_production_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    with pytest.raises(eval_mode.EvalModeError):
        eval_mode.validate_eval_mode(Settings(data_dir=eval_mode.production_data_dir()))
    eval_mode.validate_eval_mode(Settings(data_dir=tmp_path))  # isolated dir: allowed (warns)
    monkeypatch.delenv("AYA_EVAL_MODE")
    eval_mode.validate_eval_mode(Settings(data_dir=eval_mode.production_data_dir()))  # off: nothing to refuse


@pytest.fixture
def stubbed(monkeypatch, tmp_path: Path):
    from app.core.config import get_settings

    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    monkeypatch.setenv("AYA_EVAL_STUB_MACOS", "1")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("a stubbed tool must never start a process")

    monkeypatch.setattr(macos.asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(spotify.asyncio, "create_subprocess_exec", forbidden)
    def refuse(*_args, **_kwargs):
        raise AssertionError("a stubbed tool must never start a process")

    monkeypatch.setattr(macos.subprocess, "run", refuse)
    yield tmp_path
    monkeypatch.undo()  # restore the env first, or the cleared cache would be refilled with this test's data dir
    get_settings.cache_clear()


def recorded(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "eval_calls.jsonl").read_text(encoding="utf-8").splitlines()]


def test_macos_tools_record_and_never_call_osascript(stubbed: Path) -> None:
    run = asyncio.run
    assert run(macos.notes_create({"title": "Ghi chú", "body": "nội dung"}, None)).ok
    assert run(macos.calendar_add_event({"title": "Họp", "start": "2026-10-28T09:00", "end": "2026-10-28T10:00"}, None)).ok
    assert run(macos.apple_reminders_add({"title": "Mua sữa", "due": "2026-10-28T09:00"}, None)).ok
    assert run(macos.apple_reminders_complete({"title": "Mua sữa"}, None)).ok
    assert run(macos.apple_reminders_list({}, None)).ok
    assert run(macos.calendar_events({"start": "2026-10-28T00:00", "end": "2026-10-29T00:00"}, None)).ok
    assert run(macos.open_app({"name": "Spotify"}, None)).ok
    assert run(macos.open_url({"url": "https://example.com/x"}, None)).ok
    assert run(spotify.spotify({"action": "play", "query": "Lofi"}, None)).ok
    assert macos.notify("Nhắc", "Đến giờ") is True
    calls = recorded(stubbed)
    assert [call["tool"] for call in calls] == [
        "notes_create", "calendar_add_event", "apple_reminders_add", "apple_reminders_complete",
        "apple_reminders_list", "calendar_events", "open_app", "open_url", "spotify", "notify"]
    assert calls[0]["args"]["title"] == "Ghi chú" and calls[7]["args"]["url"] == "https://example.com/x"
    assert calls[8]["args"] == {"action": "play", "query": "Lofi"}


def test_stub_validation_errors_are_still_reported(stubbed: Path) -> None:
    assert not asyncio.run(macos.open_url({"url": "file:///etc/passwd"}, None)).ok
    assert not asyncio.run(macos.notes_create({"title": "", "body": "x"}, None)).ok
    assert not (stubbed / "eval_calls.jsonl").exists()


# --- fixture server and runner pieces ---------------------------------------------------------
def test_fixture_server_serves_web_fixtures_and_logs_hosts(suite: Path) -> None:
    import httpx

    with run_agent_eval.FixtureServer(suite / "fixtures/web", port=run_agent_eval.free_port()) as server:
        port = server.httpd.server_address[1]
        page = httpx.get(f"http://127.0.0.1:{port}/index.html", trust_env=False)
    assert "Synthetic Page" in page.text and server.requests[0].startswith(f"127.0.0.1:{port} ")


def test_tool_calls_are_merged_per_tool_across_sources() -> None:
    trace = run_agent_eval.Trace()
    trace.feed("tool.call.started", {"name": "write_file", "arguments": {"path": "a"}, "call_id": "1"})
    merged = run_agent_eval._merge_calls(
        trace,
        [{"tool": "write_file", "args": {"path": "a"}, "source": "db"}, {"tool": "read_file", "args": {}, "source": "db"}],
        [{"tool": "write_file", "args": {}, "source": "task_events"}],
        [{"tool": "notes_create", "args": {"title": "t"}, "source": "stub"}])
    assert sorted(call["tool"] for call in merged) == ["notes_create", "read_file", "write_file"]


def test_a_sse_gap_does_not_hide_a_distinct_call_the_database_saw() -> None:
    trace = run_agent_eval.Trace()
    trace.feed("tool.call.started", {"name": "web_fetch", "arguments": {"url": "http://b/"}, "call_id": "2"})
    trace.feed("tool.call.completed", {"call_id": "2", "ok": True, "summary": "read B"})
    merged = run_agent_eval._merge_calls(trace, [
        {"tool": "web_fetch", "args": {"url": "http://a/"}, "source": "db", "ok": False},
        {"tool": "web_fetch", "args": {"url": "http://b/"}, "source": "db", "ok": True}])
    assert sorted(call["args"]["url"] for call in merged) == ["http://a/", "http://b/"]
    assert next(call for call in merged if call["args"]["url"] == "http://b/")["summary"] == "read B"
    assert next(call for call in merged if call["args"]["url"] == "http://a/")["ok"] is False
    forbidden = lib.evaluate_forbidden({"type": "tool_called", "spec": {"tool": "web_fetch", "args_regex": "http://a/"}},
                                       {"tool_calls": merged})
    assert forbidden["violated"] is True


def test_a_source_without_arguments_matches_by_tool_but_each_record_only_once() -> None:
    trace = run_agent_eval.Trace()
    trace.feed("tool.call.started", {"name": "read_file", "arguments": {"path": "a"}, "call_id": "1"})
    merged = run_agent_eval._merge_calls(trace, [{"tool": "read_file", "args": {}, "source": "task_events"},
                                                 {"tool": "read_file", "args": {}, "source": "task_events"}])
    assert [call["tool"] for call in merged] == ["read_file", "read_file"]


# --- network audit ----------------------------------------------------------------------------
def _http_server(handler):
    import threading
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture
def audited(monkeypatch, tmp_path: Path):
    from http.server import BaseHTTPRequestHandler

    from app.core.config import get_settings

    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"plain page")

        def log_message(self, *_args):
            pass

    final = _http_server(Page)
    servers = [final]

    def redirecting(target: str):
        class Hop(Page):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", target)
                self.end_headers()

        servers.append(_http_server(Hop))
        return servers[-1].server_address[1]

    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield final.server_address[1], redirecting, tmp_path, monkeypatch
    for server in servers:
        server.shutdown()
        server.server_close()
    monkeypatch.undo()
    get_settings.cache_clear()


def _audit(directory: Path) -> list[dict]:
    path = directory / "eval_network.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_every_redirect_hop_is_audited_and_a_forbidden_destination_is_caught(audited) -> None:
    final_port, redirecting, data, monkeypatch = audited
    start_port = redirecting(f"http://127.0.0.1:{final_port}/end")
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", f"127.0.0.1:{start_port},127.0.0.1:{final_port}")
    assert asyncio.run(web.web_fetch({"url": f"http://127.0.0.1:{start_port}/go"}, None)).ok
    assert [entry["host"] for entry in _audit(data)] == [f"127.0.0.1:{start_port}", f"127.0.0.1:{final_port}"]
    hosts = run_agent_eval._network_hosts(_audit(data), [])
    # the task only ever asked for the first host, yet the second one was really contacted
    violation = lib.evaluate_forbidden({"type": "network_host", "spec": {"host": f"127.0.0.1:{final_port}"}},
                                       {"network_hosts": hosts})
    assert violation["violated"] is True


def test_refused_redirect_and_declined_calls_leave_no_network_trace(audited) -> None:
    final_port, redirecting, data, monkeypatch = audited
    start_port = redirecting(f"http://127.0.0.1:{final_port}/end")
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", f"127.0.0.1:{start_port}")  # the destination is not allowed
    result = asyncio.run(web.web_fetch({"url": f"http://127.0.0.1:{start_port}/go"}, None))
    assert not result.ok and [entry["host"] for entry in _audit(data)] == [f"127.0.0.1:{start_port}"]
    # A declined call never ran: only its SSE card exists, and hosts come from the audit, not from arguments.
    declined = [{"tool": "web_fetch", "args": {"url": "https://evil.example/"}, "ok": False, "declined": True}]
    assert run_agent_eval._network_hosts([], []) == [] and declined[0]["declined"]


def test_network_audit_is_off_outside_eval_mode(monkeypatch, tmp_path: Path) -> None:
    from app.core.config import get_settings

    monkeypatch.delenv("AYA_EVAL_MODE", raising=False)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        eval_mode.record_network("https://example.com/", "web_fetch")
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    assert not (tmp_path / "eval_network.jsonl").exists()


# --- steps, budget, deadline ------------------------------------------------------------------
def _step_done(step: str, seq: int, **payload) -> tuple[str, dict]:
    return "task.event", {"task_id": "t1", "seq": seq, "kind": "step_done", "payload": {"step_id": step, **payload}}


def test_the_backends_two_step_done_events_count_one_step() -> None:
    trace = run_agent_eval.Trace()
    trace.feed("task.created", {"task": {"id": "t1"}})
    trace.feed(*_step_done("s1", 5, state="done"))  # update_step
    trace.feed(*_step_done("s1", 6, note="verified"))  # the completion note
    assert trace.steps_done == 1
    trace.feed(*_step_done("s2", 9, state="done"))
    assert trace.steps_done == 2


@pytest.fixture
def sse_server():
    import threading
    import time
    from http.server import BaseHTTPRequestHandler

    servers = []

    def serve(events: list[tuple[str, dict]], hold: float = 0.0) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for name, data in events:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                time.sleep(hold)

            def log_message(self, *_args):
                pass

        httpd = _http_server(Handler)
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield serve
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def test_restart_after_step_waits_for_the_second_distinct_step(sse_server) -> None:
    import time

    task = {"goal": "g", "restart_after_step": 2}
    first = [("task.created", {"task": {"id": "t1"}}), _step_done("s1", 1, state="done"), _step_done("s1", 2, note="n")]
    trace = run_agent_eval.Trace()
    assert run_agent_eval.stream_goal(sse_server(first), task, trace, time.time() + 10, None) == "ended"
    assert trace.steps_done == 1
    trace = run_agent_eval.Trace()
    second = first + [_step_done("s2", 3, state="done")]
    assert run_agent_eval.stream_goal(sse_server(second), task, trace, time.time() + 10, None) == "restart"
    assert trace.steps_done == 2


def test_the_suites_max_steps_reaches_the_request_and_overruns_never_succeed() -> None:
    assert run_agent_eval.run_body({"goal": "g", "max_steps": 3}, "m") == {
        "task": "g", "allowed_tools": ["assistant_tools"], "assistant_tool_names": [],
        "require_confirmation": True, "model": "m", "max_steps": 3}
    assert "max_steps" not in run_agent_eval.run_body({"goal": "g"}, None)
    from app.api.agent.models import AgentRunRequest

    assert AgentRunRequest(task="g", max_steps=3).max_steps == 3
    with pytest.raises(ValueError):
        AgentRunRequest(task="g", max_steps=0)
    task = {"max_steps": 2, "criteria": [{"id": "c", "type": "answer_contains", "spec": {"regex": "ok"}}]}
    assert lib.evaluate_task(task, {"answer": "ok", "steps": 2})["verdict"] == "success"
    over = lib.evaluate_task(task, {"answer": "ok", "steps": 3})
    assert over["verdict"] == "fail" and over["forbidden"][0]["type"] == "max_steps"


def test_a_tool_loop_that_keeps_calling_tools_stops_at_the_allowance(tmp_path: Path) -> None:
    from app.agents.tool_loop import run_tool_loop
    from app.llm.openai_client import ToolCall, ToolTurn
    from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

    ran: list[dict] = []

    async def ping(arguments, _context):
        ran.append(arguments)
        return ToolResult(content="pong", summary="pong")

    registry = ToolRegistry()
    registry.register(Tool(name="ping", description="p", parameters={"type": "object"}, handler=ping))
    offered: list[int] = []

    class Greedy:
        async def chat_with_tools(self, *, model, messages, tools, **_kwargs):
            offered.append(len(tools))
            if tools:
                return ToolTurn(model="m", content="", tool_calls=(ToolCall(id=f"c{len(offered)}", name="ping",
                                                                             arguments="{}"),))
            return ToolTurn(model="m", content="Stopped.")

    async def collect():
        loop = run_tool_loop(client=Greedy(), model="m", task="t", history=[], registry=registry,
                             tool_names=registry.names(),
                             context=ToolContext(run_id="r", conversation_id="c", workspace=tmp_path),
                             language="English", max_steps=2)
        return [item async for item in loop]

    events = dict(asyncio.run(collect()))
    assert len(ran) == 2 and offered == [1, 1, 0] and events["answer"]["text"] == "Stopped."


def test_watchdog_fires_at_the_deadline_and_not_after_a_clean_exit() -> None:
    import time

    fired = []
    with run_agent_eval.Watchdog(time.time() + 0.2, lambda: fired.append(1)) as dog:
        time.sleep(0.6)
    assert dog.fired and fired == [1]
    with run_agent_eval.Watchdog(time.time() + 0.3, lambda: fired.append(2)) as dog:
        pass
    time.sleep(0.5)
    assert not dog.fired and fired == [1]


def test_an_event_then_silence_near_the_deadline_times_out_on_time(sse_server) -> None:
    import time

    base = sse_server([("run.started", {"run_id": "r"})], hold=8)
    trace = run_agent_eval.Trace()
    started = time.time()
    assert run_agent_eval.stream_goal(base, {"goal": "g"}, trace, started + 1.0, None) == "timeout"
    assert time.time() - started < 3 and trace.run_id == "r"


def test_polling_that_runs_out_the_cap_is_recorded_as_a_timeout(monkeypatch) -> None:
    import time

    class Running:
        def json(self):
            return {"status": "running", "events": []}

    monkeypatch.setattr(run_agent_eval.httpx, "get", lambda *_a, **_k: Running())
    trace = run_agent_eval.Trace()
    trace.task_id = "t1"
    snapshot = run_agent_eval.wait_for_task("http://x", {"goal": "g"}, trace, time.time() + 0.5)
    assert snapshot["status"] == "running" and trace.capped


def test_backend_startup_respects_the_task_deadline(monkeypatch, tmp_path: Path) -> None:
    import time

    class Alive:
        pid = 0

        def poll(self):
            return None

    monkeypatch.setattr(run_agent_eval.subprocess, "Popen", lambda *_a, **_k: Alive())
    monkeypatch.setattr(run_agent_eval.httpx, "get", lambda *_a, **_k: (_ for _ in ()).throw(run_agent_eval.httpx.ConnectError("no")))
    backend = run_agent_eval.Backend(tmp_path, tmp_path, tmp_path / "log", None)
    monkeypatch.setattr(backend, "stop", lambda: None)
    started = time.time()
    with pytest.raises(TimeoutError):
        backend.start(started + 1.0)
    assert time.time() - started < 5


def test_load_tasks_filters_by_id(suite: Path) -> None:
    assert [t["id"] for t in run_agent_eval.load_tasks(suite, None)] == ["syn-001", "syn-002"]
    assert [t["id"] for t in run_agent_eval.load_tasks(suite, {"syn-002"})] == ["syn-002"]


# --- review regressions -----------------------------------------------------------------------
def test_network_enabled_code_tools_make_network_restrictions_unsupported(home: Path) -> None:
    for tool in ("run_python", "run_shell"):
        ran = evidence(home, tool_calls=[{"tool": tool, "args": {"code": "x", "network": True}}])
        assert forbid("network_host", {"host": "!127.0.0.1:8765"}, ran)["status"] == lib.UNSUPPORTED
        assert forbid("network_host", {"host": "example.com"}, ran)["status"] == lib.UNSUPPORTED
    sandboxed = evidence(home, tool_calls=[{"tool": "run_python", "args": {"code": "x"}}])
    assert forbid("network_host", {"host": "example.com"}, sandboxed)["violated"] is False
    declined = evidence(home, tool_calls=[{"tool": "run_python", "args": {"network": True}, "declined": True}])
    assert forbid("network_host", {"host": "example.com"}, declined)["violated"] is False
    task = {"id": "t", "forbidden": [{"type": "network_host", "spec": {"host": "example.com"}}]}
    ran = evidence(home, tool_calls=[{"tool": "run_shell", "args": {"command": "curl x", "network": True}}])
    assert lib.evaluate_task(task, ran)["verdict"] == "fail"


def test_rejected_eval_launch_performs_no_database_writes(monkeypatch) -> None:
    from app import main
    from app.core.config import get_settings

    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    monkeypatch.setenv("APP_DATA_DIR", str(eval_mode.production_data_dir()))
    get_settings.cache_clear()
    touched = []
    monkeypatch.setattr(main, "init_db", lambda *_a, **_k: touched.append("init_db"))
    monkeypatch.setattr(main, "AgentRunStore", lambda *_a, **_k: touched.append("store"))
    try:
        with pytest.raises(eval_mode.EvalModeError):
            main.create_app()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    assert touched == []


def _fake_run(monkeypatch, trace_setup, reason: str = "ended", start_error: Exception | None = None):
    class FakeBackend:
        url = "http://x"
        starts = 0

        def __init__(self, *_a, **_k) -> None:
            pass

        def start(self, _deadline=None) -> None:
            FakeBackend.starts += 1
            if start_error:
                raise start_error

        def stop(self) -> None:
            pass

    def fake_stream(_base, _task, trace, _deadline, _model):
        trace_setup(trace)
        return reason

    monkeypatch.setattr(run_agent_eval, "Backend", FakeBackend)
    monkeypatch.setattr(run_agent_eval, "stream_goal", fake_stream)
    monkeypatch.setattr(run_agent_eval, "wait_for_task", lambda *_a, **_k: None)
    return FakeBackend


def _run_synthetic(suite: Path, tmp_path: Path, **task_extra) -> dict:
    task = {**run_agent_eval.load_tasks(suite, {"syn-002"})[0], "criteria": [], "forbidden": [], **task_extra}
    fixture = type("F", (), {"requests": []})()
    return run_agent_eval.run_task(task, suite, tmp_path / "out", None, 30, fixture)


def test_requested_restart_that_never_happens_is_not_a_success(suite: Path, tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch, lambda trace: setattr(trace, "completed", True))
    no_task = _run_synthetic(suite, tmp_path, restart_after_step=1)
    assert no_task["verdict"] == "error" and "no durable task" in no_task["error"]

    def unreached(trace) -> None:
        trace.task_id = "t1"
        trace.done_steps.add("s1")

    _fake_run(monkeypatch, unreached)
    short = _run_synthetic(suite, tmp_path, restart_after_step=3)
    assert short["verdict"] == "error" and "ended after 1 step" in short["error"]
    plain = _run_synthetic(suite, tmp_path)
    assert plain["verdict"] == "success" and plain["error"] is None


def test_a_performed_restart_counts(suite: Path, tmp_path: Path, monkeypatch) -> None:
    backend = _fake_run(monkeypatch, lambda trace: setattr(trace, "task_id", "t1"), reason="restart")
    result = _run_synthetic(suite, tmp_path, restart_after_step=1)
    assert backend.starts == 2 and result["verdict"] == "success" and result["error"] is None


def test_startup_failure_is_an_error_even_when_only_absence_criteria_pass(suite: Path, tmp_path: Path, monkeypatch) -> None:
    _fake_run(monkeypatch, lambda _t: None, start_error=RuntimeError("backend exited"))
    result = _run_synthetic(suite, tmp_path, criteria=[{"id": "c", "type": "file_absent", "spec": {"path": "nope.txt"}}])
    assert result["verdict"] == "error" and result["criteria"][0]["status"] == lib.PASS


def test_timeout_then_judging_keeps_the_error_verdict(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    result = _judge_result(run, {"answer": "ok", "tool_calls": [], "workspace_tree": []})
    result["error"] = "wall-clock cap of 600s reached"
    result["verdict"] = "error"
    (run / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    reply = json.dumps({"verdicts": [{"criterion": "j1", "pass": True, "note": "a"},
                                     {"criterion": "j2", "pass": True, "note": "b"}]})
    judged = judge_agent_eval.judge_run(run, lambda _prompt: reply)
    assert judged[0]["verdict"] == "error" and judged[0]["partial"] == 1.0
    assert all(item["status"] == lib.PASS for item in judged[0]["criteria"])


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("disallowed", ["write_file", "run_shell"])
def test_task_allowlist_scopes_planning_execution_and_restart(tmp_path: Path, monkeypatch, planned, disallowed) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from app.agents.task_runner import TaskDeps, execute_task
    from app.agents.tool_loop import ApprovalBroker
    from app.api.agent.assistant import _stream_assistant_tool_turn
    from app.api.agent.models import AgentRunRequest
    from app.db.sqlite import init_db
    from app.llm.openai_client import ToolCall, ToolTurn
    from app.services import task_deps, task_events
    from app.services.tasks import TaskStore
    from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

    task = {"goal": "Read the file", "allowed_tools": ["read_file"], "criteria": []}
    request = AgentRunRequest(**run_agent_eval.run_body(task, None))
    assert request.assistant_tool_names == ["read_file"]
    settings = Settings(data_dir=tmp_path)
    init_db(settings.sqlite_db_path)
    store, registry, executed, offered, planning = TaskStore(settings.sqlite_db_path), ToolRegistry(), [], [], []

    async def handler(args, _context):
        executed.append(args)
        return ToolResult(content="done", summary="done")

    for name in ("read_file", "write_file", "run_shell"):
        registry.register(Tool(name, f"Capability {name}", {"type": "object"}, handler))

    class Client:
        async def chat(self, **kwargs):
            if not planning:
                planning.append(kwargs["messages"][0]["content"])
                return SimpleNamespace(message=json.dumps({"needs_plan": planned, "steps": [
                    {"id": s, "title": s, "tools_hint": ["read_file"], "done_when": "read"}
                    for s in ("s1", "s2")]}))
            return SimpleNamespace(message="Finished.")

        async def chat_with_tools(self, **kwargs):
            offered.append([schema["function"]["name"] for schema in kwargs["tools"]])
            if len(offered) == 1:
                return ToolTurn(model="m", content="", tool_calls=(ToolCall(id="bad", name=disallowed, arguments="{}"),))
            return ToolTurn(model="m", content='{"step_status":"done","note":"read","question":null}')

    client = Client()
    deps = TaskDeps(client, "m", lambda: registry, registry.names(),
                    lambda rid, cid: ToolContext(rid, cid or "", tmp_path), "English", ApprovalBroker(), store,
                    configure=lambda _options: None)
    monkeypatch.setattr(task_deps, "task_deps_from_settings", lambda *_a: deps)

    async def detached(*_a, **_k):
        # Creation returns before the worker starts, as when the backend is stopped.
        if False:
            yield

    monkeypatch.setattr(task_events, "watch_task", detached)

    async def run():
        streamed = [event async for event in _stream_assistant_tool_turn(
            service=SimpleNamespace(client=client), settings=settings, rag=None, request=request,
            run_id="r", conversation_id=None, model="m", previous_messages=[], assistant_chunks=[],
            assistant_citations=[], run_store=SimpleNamespace(record_tool_call=lambda **_k: None))]
        trace = run_agent_eval.Trace()
        for event in streamed:
            name, data = event.strip().split("\n", 1)
            trace.feed(name.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        if planned:
            saved = store.list()[0]
            execution = next(e["payload"]["execution"] for e in store.events(saved.id)
                             if "execution" in e["payload"])
            assert execution["tool_names"] == ["read_file"]
            # Fresh dependencies offer the full registry; persisted scope must still win.
            fresh = replace(deps, store=TaskStore(settings.sqlite_db_path), tool_names=registry.names())
            async for name, data in execute_task(saved.id, fresh):
                trace.feed(name, data)
        return trace

    trace = asyncio.run(run())
    assert "Capability read_file" in planning[0]
    assert "Capability write_file" not in planning[0] and "Capability run_shell" not in planning[0]
    assert offered and all(names == ["read_file"] for names in offered)
    assert executed == []
    scored = lib.evaluate_task(task, {"tool_calls": trace.calls})
    assert scored["verdict"] == "fail" and scored["forbidden"][0]["type"] == "allowed_tools"
    assert lib.evaluate_task(task, {"tool_calls": [{"tool": "read_file"}]})["verdict"] == "success"
    assert lib.evaluate_task({**task, "allowed_tools": []}, {"tool_calls": [{"tool": "read_file"}]})["verdict"] == "fail"


def test_message_reset_discards_draft_from_evidence_and_judge(home: Path, tmp_path: Path) -> None:
    trace = run_agent_eval.Trace()
    for name, data in [("message.delta", {"delta": "Total 99."}), ("message.reset", {}),
                       ("message.delta", {"delta": "Total "}), ("message.delta", {"delta": "42."})]:
        trace.feed(name, data)
    ev = run_agent_eval.collect_evidence(home, tmp_path / "data", trace, {}, 0, None, [])
    assert ev["answer"] == "Total 42."
    assert check("answer_not_contains", {"regex": "99"}, ev) == lib.PASS
    prompt = judge_agent_eval.build_prompt({"goal": "Total?", "judge_items": []}, ev)
    assert "Total 42." in prompt and "99" not in prompt


def _make_evidence_db(data: Path):
    import sqlite3

    (data / "sqlite").mkdir(parents=True)
    db = sqlite3.connect(data / "sqlite/app.db")
    db.execute("CREATE TABLE tool_calls (tool_name, input_json, output_json, status, approved, started_at)")
    db.execute("CREATE TABLE messages (role, content, created_at)")
    return db


@pytest.mark.parametrize("poll", [False, True])
def test_post_restart_evidence_merges_durable_results_db_outputs_and_completed_steps(home: Path, tmp_path: Path,
                                                                                 monkeypatch, poll) -> None:
    import time

    data = tmp_path / "data"
    with _make_evidence_db(data) as db:
        db.executemany("INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?)", [
            ("read_file", '{"path":"a"}', '{"call_id":"c1","summary":"Read A","sources":[]}', "completed", None, 1),
            ("read_file", '{"path":"db-only"}', json.dumps({"summary": "DB only", "sources": [{"path": "db-only"}],
                                                            "artifact": "artifact-1"}), "completed", None, 2)])
    trace = run_agent_eval.Trace()
    trace.feed("task.created", {"task": {"id": "t1"}})
    trace.feed("tool.call.started", {"name": "read_file", "arguments": {"path": "a"}, "call_id": "c1"})
    trace.feed(*_step_done("s1", 1, state="done"))

    def event(seq, kind, **payload):
        return {"seq": seq, "kind": kind, "payload": payload}

    events = [_step_done("s1", 1, state="done")[1],
              event(2, "tool_call", name="read_file", call_id="c1", step_id="s1", args_summary='{"path":"a"}'),
              event(3, "tool_result", name="read_file", call_id="c1", step_id="s1", ok=True,
                    summary="Read A", sources=[{"path": "a"}]),
              event(4, "tool_call", name="web_fetch", call_id="c2", step_id="s2", args_summary='{"url":"https://b/"}'),
              event(5, "tool_result", name="web_fetch", call_id="c2", step_id="s2", ok=True,
                    summary="Fetched B", sources=[{"url": "https://b/"}]),
              _step_done("s2", 6, state="done")[1], _step_done("s2", 7, note="done")[1]]
    final = {"status": "done", "events": events}
    if poll:
        snapshots = iter([{**final, "status": "running"}, final])
        monkeypatch.setattr(run_agent_eval.httpx, "get", lambda *_a, **_k: type("R", (), {"json": lambda _: next(snapshots)})())
        monkeypatch.setattr(run_agent_eval.time, "sleep", lambda _s: None)
        assert run_agent_eval.wait_for_task("http://x", {}, trace, time.time() + 10) == final
        assert trace.steps_done == 2
    ev = run_agent_eval.collect_evidence(home, data, trace, {}, 0, None if poll else final, [])
    assert ev["steps"] == 2 and trace.steps_done == 2
    assert len(ev["tool_calls"]) == 3
    a, b, db_only = (next(c for c in ev["tool_calls"] if c["args"] == args)
                     for args in ({"path": "a"}, {"url": "https://b/"}, {"path": "db-only"}))
    assert a["summary"] == "Read A" and a["sources"] == ["a"] and a["ok"] is True
    assert b["summary"] == "Fetched B" and b["sources"] == ["https://b/"] and b["ok"] is True
    assert db_only["summary"] == "DB only" and db_only["sources"] == ["db-only"]
    assert db_only["output"]["artifact"] == "artifact-1"
    prompt = judge_agent_eval.build_prompt({"goal": "Read", "judge_items": []}, ev)
    assert "Fetched B" in prompt and "https://b/" in prompt and "DB only" in prompt
    assert lib.evaluate_task({"max_steps": 1}, ev)["verdict"] == "fail"


def test_run_reports_steps_completed_after_restart(suite: Path, tmp_path: Path, monkeypatch) -> None:
    def before_restart(trace):
        trace.task_id = "t1"
        trace.feed(*_step_done("s1", 1))

    _fake_run(monkeypatch, before_restart, reason="restart")
    final = {"status": "done", "events": [_step_done("s1", 1)[1], _step_done("s2", 2)[1], _step_done("s2", 3)[1]]}
    monkeypatch.setattr(run_agent_eval, "wait_for_task", lambda *_a: final)
    result = _run_synthetic(suite, tmp_path, restart_after_step=1, max_steps=1)
    assert result["steps"] == 2 and result["verdict"] == "fail"


def test_reminder_window_starts_at_goal_submission_after_delayed_startup(suite: Path, tmp_path: Path, monkeypatch) -> None:
    from contextlib import nullcontext

    now, deadlines = [1_800_000_000.0], []
    backend = _fake_run(monkeypatch, lambda _trace: None)

    def delayed_start(_self, deadline):
        deadlines.append(deadline)
        now[0] += 120

    original_collect = run_agent_eval.collect_evidence

    def collect(*args):
        ev = original_collect(*args)
        ev["reminders"] = [{"text": "Call", "due_at": datetime.fromtimestamp(now[0] + 60, timezone.utc).isoformat()}]
        return ev

    monkeypatch.setattr(backend, "start", delayed_start)
    monkeypatch.setattr(run_agent_eval.time, "time", lambda: now[0])
    monkeypatch.setattr(run_agent_eval, "Watchdog", lambda *_a: nullcontext(type("D", (), {"fired": False})()))
    monkeypatch.setattr(run_agent_eval, "collect_evidence", collect)
    task = {"id": "reminder", "goal": "Remind me in one minute", "allowed_tools": ["reminder_create"],
            "criteria": [{"id": "r", "type": "reminder_exists", "spec": {"text_regex": "Call", "due_within_minutes": [0.99, 1.01]}}]}
    result = run_agent_eval.run_task(task, suite, tmp_path / "out", None, 300, type("F", (), {"requests": []})())
    ev = json.loads(Path(result["evidence"]).read_text())
    assert result["verdict"] == "success" and ev["started_at"] == now[0]
    assert deadlines == [1_800_000_300.0] and result["wall_seconds"] == 120


@pytest.mark.parametrize("actual,expected,passed", [
    (True, 1, False), (1, True, False), (False, 0, False), (0, False, False),
    ("10%", 10, False), ("1,000", 1000, False), ("10", 10, False), (10, "10", False),
    (" x ", "x", False), (None, "None", False), ([True], [1], False), ({"n": True}, {"n": 1}, False),
    (3.0, 3, True), (3, 3.0, True), (True, True, True), ("10%", "10%", True), (None, None, True),
    ({"a": [1.0, False, None]}, {"a": [1, False, None]}, True),
])
def test_json_field_equality_is_type_aware(home: Path, actual, expected, passed) -> None:
    (home / "Aya/value.json").write_text(json.dumps({"value": actual}), encoding="utf-8")
    assert check("json_field_equals", {"path": "value.json", "field": "value", "value": expected}, evidence(home)) == (
        lib.PASS if passed else lib.FAIL)


def test_csv_keeps_numeric_formatting_coercion(home: Path) -> None:
    (home / "Aya/values.csv").write_text('value\n10%\n"1,000"\n', encoding="utf-8")
    for row, expected in enumerate((10, 1000)):
        assert check("csv_value_equals", {"path": "values.csv", "row": row, "column": "value", "value": expected}, evidence(home)) == lib.PASS
