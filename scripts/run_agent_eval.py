#!/usr/bin/env python3
"""Run a sealed agent suite against isolated Aya backends and score it.

Per task: a fresh HOME with the suite's ``fixtures/workspace`` copied to ``$HOME/Aya``, a fresh
``APP_DATA_DIR``, and a backend on a free port in eval mode (fixture web server allowed, macOS
tools stubbed, TTS off). The goal goes to ``/agent/run/stream``; approvals are answered per the
task's ``approval_policy`` (default approve); ``restart_after_step`` kills and restarts the backend
after that step and waits for the task to finish through ``/tasks``.

Usage:
  backend/.venv/bin/python scripts/run_agent_eval.py --suite DIR --out DIR [--only id,id] [--model M]

Outputs in --out: ``results.jsonl``, ``report.md`` and ``tasks/<id>/`` (home, data, backend.log,
sse.jsonl, evidence.json). Grade ``judge_rubric`` criteria afterwards with judge_agent_eval.py.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval_lib import evaluate_task, render_report, sha256_file, verify_manifest  # noqa: E402

WEB_HOST, WEB_PORT = "127.0.0.1", 8765
BACKEND_PYTHON = ROOT / "backend/.venv/bin/python"
STARTUP_SECONDS = 150
TERMINAL = {"done", "failed", "cancelled"}


class FixtureServer:
    """Static server for ``fixtures/web``; remembers which Host headers asked for what."""

    def __init__(self, directory: Path, port: int = WEB_PORT) -> None:
        self.requests: list[str] = []
        outer = self

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                outer.requests.append(f"{self.headers.get('Host', '')} {self.path}")

        self.httpd = ThreadingHTTPServer((WEB_HOST, port), partial(Handler, directory=str(directory)))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FixtureServer":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Backend:
    """One isolated uvicorn process; ``start`` can be called again after ``stop`` (same dirs, same port)."""

    def __init__(self, home: Path, data: Path, log: Path, model: str | None) -> None:
        self.home, self.data, self.log, self.model = home, data, log, model
        self.port = free_port()
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> dict[str, str]:
        env = {**os.environ, "NO_PROXY": "*", "PYTHONPATH": str(ROOT / "backend"), "HOME": str(self.home),
               "APP_DATA_DIR": str(self.data), "ASSISTANT_WORKSPACE": str(self.home / "Aya"),
               "TTS_ENABLED": "false", "TELEGRAM_BOT_TOKEN": "", "AYA_EVAL_MODE": "1",
               "AYA_EVAL_WEB_ALLOW": f"{WEB_HOST}:{WEB_PORT}", "AYA_EVAL_STUB_MACOS": "1",
               "LIGHTRAG_ENABLED": "false"}
        if self.model:
            env["DEFAULT_MODEL"] = self.model
        return env

    def start(self, deadline: float | None = None) -> None:
        """Start and wait for /health, never past the task's absolute ``deadline``."""
        handle = self.log.open("ab")
        self.process = subprocess.Popen(
            [str(BACKEND_PYTHON), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.port),
             "--app-dir", str(ROOT / "backend")],
            cwd=ROOT, env=self.env(), stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        limit = time.time() + STARTUP_SECONDS
        capped = deadline is not None and deadline < limit
        limit = deadline if capped else limit
        while time.time() < limit:
            if self.process is None or self.process.poll() is not None:  # exited, or the watchdog stopped it
                raise RuntimeError(f"backend exited; see {self.log}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2, trust_env=False).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        if capped:
            raise TimeoutError("wall-clock cap reached while the backend was starting")
        raise RuntimeError(f"backend did not come up in {STARTUP_SECONDS}s; see {self.log}")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=wait)
                return
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue


class Watchdog:
    """Runs ``on_expire`` once at the absolute ``deadline`` unless the block exits first."""

    def __init__(self, deadline: float, on_expire) -> None:
        self.deadline, self.on_expire, self.fired = deadline, on_expire, False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        if not self._done.wait(max(0.0, self.deadline - time.time())):
            self.fired = True
            self.on_expire()

    def __enter__(self) -> "Watchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._done.set()
        self._thread.join(timeout=15)


def iter_sse(response: httpx.Response):
    event, data = "message", []
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            try:
                yield event, json.loads("\n".join(data))
            except ValueError:
                pass
            event, data = "message", []


def snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()}


def _decide(base: str, run_id: str, call_id: str, approved: bool) -> None:
    try:
        httpx.post(f"{base}/agent/runs/{run_id}/tool-calls/{call_id}/decision", json={"approved": approved},
                   timeout=20, trust_env=False)
    except httpx.HTTPError:
        pass


class Trace:
    """Everything one task run observed over SSE."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.run_id = self.task_id = None
        self.answer: list[str] = []
        self.calls: list[dict] = []
        self.approvals: dict[str, dict] = {}
        self.asked_user = False
        self.done_steps: set[str] = set()
        self.failed: str | None = None
        self.completed = False
        self.capped = False

    @property
    def steps_done(self) -> int:
        """Distinct completed steps: the backend emits two ``step_done`` events per step."""
        return len(self.done_steps)

    def feed(self, name: str, data: dict) -> None:
        self.events.append((name, data))
        self.run_id = self.run_id or data.get("run_id")
        if name == "task.created":
            self.task_id = data["task"]["id"]
        elif name == "message.delta":
            self.answer.append(str(data.get("delta") or ""))
        elif name == "tool.call.started":
            self.calls.append({"tool": data.get("name"), "args": data.get("arguments"), "call_id": data.get("call_id"),
                               "source": "sse"})
        elif name == "tool.call.completed":
            for call in self.calls:
                if call.get("call_id") == data.get("call_id"):
                    call.update(ok=bool(data.get("ok")), summary=str(data.get("summary") or "")[:500],
                                declined=bool(data.get("declined")),
                                sources=[str(s.get("url") or s.get("path") or s.get("title"))
                                         for s in (data.get("sources") or [])][:5])
        elif name == "ask.user":
            self.asked_user = True
        elif name == "task.event" and data.get("kind") == "step_done":
            self.done_steps.add(str((data.get("payload") or {}).get("step_id") or f"seq-{data.get('seq')}"))
        elif name == "task.event" and data.get("kind") == "ask":
            self.asked_user = True
        elif name == "run.failed":
            self.failed = str(data.get("error") or "run failed")
        elif name == "run.completed":
            self.completed = True


def _record_approval(trace: Trace, card: dict) -> None:
    trace.approvals.setdefault(str(card.get("call_id")), {
        "tool": card.get("name"), "args": card.get("arguments"), "call_id": card.get("call_id")})


def run_body(task: dict, model: str | None) -> dict:
    """The request for one goal; the suite's ``max_steps`` becomes the backend's step allowance."""
    body = {"task": task["goal"], "allowed_tools": ["assistant_tools"], "require_confirmation": True}
    if model:
        body["model"] = model
    max_steps = task.get("max_steps")
    if isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0:
        body["max_steps"] = max_steps
    return body


_END = object()


def _pump(base: str, body: dict, events: queue.Queue, stop: threading.Event) -> None:
    """Follow the stream on a thread so the caller can give up at an absolute deadline mid-read."""
    try:
        with httpx.Client(timeout=httpx.Timeout(15, read=None), trust_env=False) as client, client.stream(
                "POST", f"{base}/agent/run/stream", json=body) as response:
            response.raise_for_status()
            for item in iter_sse(response):
                events.put(item)
                if stop.is_set():
                    return
    except httpx.HTTPError as exc:
        events.put(exc)
    finally:
        events.put(_END)


def stream_goal(base: str, task: dict, trace: Trace, deadline: float, model: str | None) -> str:
    """Send the goal and follow the stream. Returns why it stopped: ended | restart | timeout."""
    policy = task.get("approval_policy") or "approve"
    restart_after = task.get("restart_after_step")
    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_pump, args=(base, run_body(task, model), events, stop), daemon=True).start()
    try:
        while True:
            try:
                item = events.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                return "timeout"
            if item is _END:
                return "ended"
            if isinstance(item, httpx.HTTPError):
                trace.failed = trace.failed or f"stream error: {type(item).__name__}"
                continue
            name, data = item
            trace.feed(name, data)
            if name == "tool.call.approval_required":
                _record_approval(trace, data)
                _decide(base, str(data.get("run_id") or trace.run_id), str(data["call_id"]), policy != "deny")
            elif name == "task.event" and data.get("kind") == "approval" and data["payload"].get("decision") is None:
                _record_approval(trace, data["payload"])
                _decide(base, str(data["payload"]["run_id"]), str(data["payload"]["call_id"]), policy != "deny")
            if restart_after and trace.task_id and trace.steps_done >= int(restart_after):
                return "restart"
            if time.time() > deadline:
                return "timeout"
    finally:
        stop.set()


def wait_for_task(base: str, task: dict, trace: Trace, deadline: float) -> dict | None:
    """Poll /tasks until the task is terminal or parked on the user; answers approvals on the way."""
    policy = task.get("approval_policy") or "approve"
    snapshot_ = None
    while trace.task_id:
        remaining = deadline - time.time()
        if remaining <= 0:
            trace.capped = True  # polling ran out the wall-clock cap with the task still running
            break
        try:
            snapshot_ = httpx.get(f"{base}/tasks/{trace.task_id}", timeout=max(1.0, min(15.0, remaining)),
                                  trust_env=False).json()
        except (httpx.HTTPError, ValueError):
            time.sleep(min(1, max(0.0, deadline - time.time())))
            continue
        for event in snapshot_.get("events", []):
            payload = event.get("payload") or {}
            if event["kind"] == "approval":
                _record_approval(trace, payload)
                if payload.get("decision") is None:
                    _decide(base, str(payload["run_id"]), str(payload["call_id"]), policy != "deny")
            elif event["kind"] == "ask":
                trace.asked_user = True
        if snapshot_["status"] in TERMINAL or snapshot_["status"] == "waiting_user":
            break
        time.sleep(min(2, max(0.0, deadline - time.time())))
    return snapshot_


def _json(text: str | None):
    try:
        return json.loads(text) if text else None
    except ValueError:
        return {"raw": text}


def _canonical(args) -> str:
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)


def _match_call(merged: list[dict], claimed: set[int], call: dict) -> int | None:
    """Index of the unclaimed record this call describes: same tool and arguments; a side that
    carries no arguments (a summary-only source) matches any unclaimed call of the tool."""
    same = [index for index, other in enumerate(merged) if index not in claimed and other["tool"] == call["tool"]]
    wanted = _canonical(call.get("args"))
    for index in same:
        if _canonical(merged[index].get("args")) == wanted:
            return index
    for index in same:
        if not call.get("args") or not merged[index].get("args"):
            return index
    return None


def _merge_calls(trace: Trace, *sources: list[dict]) -> list[dict]:
    """One record per invocation: identity is (tool, arguments), each source claims a record at most
    once, outcomes fill in from whichever source has them, unmatched records are kept."""
    merged = [dict(call) for call in trace.calls]
    for extra in sources:
        claimed: set[int] = set()
        for call in extra:
            index = _match_call(merged, claimed, call)
            if index is None:
                merged.append(dict(call))
                claimed.add(len(merged) - 1)
                continue
            claimed.add(index)
            for key, value in call.items():
                if key != "source" and merged[index].get(key) in (None, "", {}):
                    merged[index][key] = value
    return merged


def _network_hosts(audit: list[dict], fixture_requests: list[str]) -> list[str]:
    """Hosts the backend actually contacted (every redirect hop, from the eval-mode audit) plus the
    Host headers the fixture server saw. Declined or refused calls never reach the audit."""
    return [request.split(" ")[0].lower() for request in fixture_requests if request] + [
        str(entry["host"]).lower() for entry in audit if entry.get("host")]


def collect_evidence(home: Path, data: Path, trace: Trace, before: dict, started: float,
                     task_snapshot: dict | None, fixture_requests: list[str]) -> dict:
    db_calls, reminders, answers = [], [], []
    database = data / "sqlite" / "app.db"
    if database.is_file():
        with closing(sqlite3.connect(database)) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT tool_name, input_json, status, approved FROM tool_calls ORDER BY started_at"):
                args = _json(row["input_json"])
                db_calls.append({"tool": row["tool_name"], "args": args if isinstance(args, dict) else {}, "source": "db",
                                 "ok": row["status"] == "completed", "declined": row["approved"] == 0})
            try:
                reminders = [dict(row) for row in db.execute("SELECT text, due_at, repeat, status FROM reminders")]
            except sqlite3.OperationalError:
                pass
            answers = [row[0] for row in db.execute(
                "SELECT content FROM messages WHERE role = 'assistant' ORDER BY created_at")]
    task_calls = []
    for event in (task_snapshot or {}).get("events", []):
        payload = event.get("payload") or {}
        if event["kind"] == "tool_call":
            args = _json(payload.get("args_summary"))
            task_calls.append({"tool": payload.get("name"), "args": args if isinstance(args, dict) else {},
                               "source": "task_events"})
        elif event["kind"] == "approval":
            _record_approval(trace, payload)
    stub_calls = []
    calls_file = data / "eval_calls.jsonl"
    if calls_file.is_file():
        stub_calls = [{"tool": entry["tool"], "args": entry.get("args") or {}, "source": "stub"}
                      for entry in map(json.loads, calls_file.read_text(encoding="utf-8").splitlines())
                      if entry.get("tool") != "notify"]
    audit = []
    audit_file = data / "eval_network.jsonl"
    if audit_file.is_file():
        audit = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = _merge_calls(trace, db_calls, task_calls, stub_calls)
    answer = "".join(trace.answer).strip() or (answers[-1] if answers else "")
    after = snapshot(home)
    return {
        "answer": answer, "tool_calls": calls, "approvals": list(trace.approvals.values()),
        "asked_user": trace.asked_user, "reminders": reminders, "started_at": started, "home": str(home),
        "workspace": str(home / "Aya"), "files_before": before, "files_after": after,
        "network_hosts": _network_hosts(audit, fixture_requests), "network_audit": audit, "steps": trace.steps_done,
        "workspace_tree": sorted(name for name in after if name.startswith("Aya/")),
    }


def run_task(task: dict, suite: Path, out: Path, model: str | None, timeout: float, fixture: FixtureServer) -> dict:
    task_dir = out / "tasks" / task["id"]
    if task_dir.exists():
        shutil.rmtree(task_dir)
    home, data = task_dir / "home", task_dir / "data"
    home.mkdir(parents=True)
    data.mkdir()
    workspace_fixture = suite / "fixtures" / "workspace"
    if workspace_fixture.is_dir():
        shutil.copytree(workspace_fixture, home / "Aya")
    else:
        (home / "Aya").mkdir()
    before = snapshot(home)
    backend = Backend(home, data, task_dir / "backend.log", model)
    trace, error, note, task_snapshot = Trace(), None, None, None
    started = time.time()
    deadline = started + timeout
    fixture.requests.clear()
    cap_error = f"wall-clock cap of {timeout:.0f}s reached"
    try:
        # One absolute deadline covers startup, streaming, restart and polling; the watchdog kills the
        # backend when it passes so no blocking call can outlive it.
        with Watchdog(deadline, backend.stop) as watchdog:
            try:
                backend.start(deadline)
                reason = stream_goal(backend.url, task, trace, deadline, model)
                if reason == "restart":
                    backend.stop()
                    backend.start(deadline)
                elif task.get("restart_after_step") and not trace.task_id:
                    note = "restart_after_step ignored: the run did not create a task"
                if reason != "timeout":
                    task_snapshot = wait_for_task(backend.url, task, trace, deadline)
                if reason == "timeout" or trace.capped:
                    error = cap_error
                elif trace.failed and not trace.completed:
                    error = trace.failed
            except TimeoutError:
                error = cap_error
            except Exception as exc:  # noqa: BLE001 - one broken task must not stop the suite
                error = f"{type(exc).__name__}: {exc}"[:300]
            if watchdog.fired:
                error = cap_error
    finally:
        backend.stop()
    evidence = collect_evidence(home, data, trace, before, started, task_snapshot, list(fixture.requests))
    (task_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    (task_dir / "sse.jsonl").write_text(
        "\n".join(json.dumps({"event": name, "data": data_}, ensure_ascii=False) for name, data_ in trace.events),
        encoding="utf-8")
    scored = evaluate_task(task, evidence)
    if error and scored["verdict"] != "success":
        scored["verdict"] = "error"
    return {
        "id": task["id"], "lang": task.get("lang"), "category": task.get("category"), **scored,
        "error": error, "note": note, "wall_seconds": round(time.time() - started, 1),
        "tool_call_count": len(evidence["tool_calls"]), "steps": trace.steps_done,
        "answer_excerpt": evidence["answer"][:400],
        "judge_items": [{"id": item.get("id"), "rubric": (item.get("spec") or {}).get("rubric")}
                        for item in task.get("criteria") or [] if item.get("type") == "judge_rubric"],
        "goal": task["goal"], "notes_for_judge": task.get("notes_for_judge"),
        "evidence": str(task_dir / "evidence.json"),
    }


def load_tasks(suite: Path, only: set[str] | None) -> list[dict]:
    tasks = [json.loads(line) for line in (suite / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [task for task in tasks if not only or task["id"] in only]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--model")
    parser.add_argument("--task-timeout", type=float, default=600, help="wall-clock cap per task, seconds")
    args = parser.parse_args(argv)
    suite, out = args.suite.expanduser().resolve(), args.out.expanduser().resolve()
    problems = verify_manifest(suite)
    if problems:
        print("MANIFEST check failed, aborting:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    tasks = load_tasks(suite, set(args.only.split(",")) if args.only else None)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    kept = {}
    if args.only and results_path.is_file():  # a partial rerun replaces only the reran tasks
        kept = {item["id"]: item for item in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())}
    web_dir = suite / "fixtures" / "web"
    if not web_dir.is_dir():  # never write into the suite
        web_dir = out / "empty-web"
        web_dir.mkdir(exist_ok=True)
    with FixtureServer(web_dir) as fixture:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task['id']} …", flush=True)
            result = run_task(task, suite, out, args.model, args.task_timeout, fixture)
            kept[result["id"]] = result
            print(f"    {result['verdict']} partial={result['partial']:.0%} {result['wall_seconds']}s", flush=True)
            results_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept.values()),
                                    encoding="utf-8")
            (out / "report.md").write_text(render_report(list(kept.values())), encoding="utf-8")
    print(f"done: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
