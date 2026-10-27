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


def test_network_hosts_ignore_calls_the_ssrf_guard_refused() -> None:
    calls = [{"tool": "web_fetch", "args": {"url": "http://127.0.0.1:8765/x"}, "ok": True},
             {"tool": "web_fetch", "args": {"url": "http://169.254.169.254/"}, "ok": False,
              "summary": "Không mở http://169.254.169.254/: địa chỉ không an toàn"},
             {"tool": "web_fetch", "args": {"url": "https://example.com/"}, "ok": False, "summary": "Không mở được"},
             {"tool": "web_search", "args": {"query": "q"}}]
    assert run_agent_eval._network_hosts(calls, []) == ["127.0.0.1:8765", "example.com", "web-search"]


def test_load_tasks_filters_by_id(suite: Path) -> None:
    assert [t["id"] for t in run_agent_eval.load_tasks(suite, None)] == ["syn-001", "syn-002"]
    assert [t["id"] for t in run_agent_eval.load_tasks(suite, {"syn-002"})] == ["syn-002"]
