"""Shared plumbing for the agent harness: config, CLI binaries, the two Codex
accounts with usage-limit failover, and Claude calls.

Every model call goes straight to the vendor CLI (``codex exec`` or
``claude -p``); nothing here goes through 9router.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import tomllib
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
STATE_DIR = REPO_ROOT / ".harness"
STATE_FILE = STATE_DIR / "state.json"
CALL_LOG = STATE_DIR / "calls.jsonl"
DEFAULT_COOLDOWN_SECONDS = 20 * 60

_RATE_LIMIT_RE = re.compile(
    r"usage limit|rate[\s_-]?limit|too many requests|\b429\b|session limit|quota",
    re.IGNORECASE,
)
_RESET_RE = re.compile(
    r"(?:reset|resets|try again)\s+(?:after|in)\s+(?:(\d+)\s*h(?:ours?)?\s*)?(?:(\d+)\s*m(?:in(?:utes?)?)?)?",
    re.IGNORECASE,
)
_lock = threading.Lock()


class QuotaExhausted(RuntimeError):
    """Every account for an engine is cooling down."""


@dataclass
class CallResult:
    ok: bool
    text: str
    role: str
    model: str
    account: str | None
    seconds: float
    log_path: Path | None
    rate_limited: bool = False


def load_config() -> dict[str, Any]:
    with open(HARNESS_DIR / "harness.toml", "rb") as handle:
        return tomllib.load(handle)


def role_config(role: str) -> dict[str, Any]:
    roles = load_config().get("roles", {})
    if role not in roles:
        raise KeyError(f"unknown role {role!r}; known: {', '.join(sorted(roles))}")
    return dict(roles[role])


def _latest(pattern: str) -> str | None:
    matches = sorted(glob.glob(os.path.expanduser(pattern)), key=os.path.getmtime)
    return matches[-1] if matches else None


def binary(name: str) -> str:
    configured = str(load_config().get("binaries", {}).get(name, "auto"))
    if configured != "auto":
        return os.path.expanduser(configured)
    found = shutil.which(name)
    if found:
        return found
    bundled = {
        "codex": "~/.vscode/extensions/openai.chatgpt-*/bin/macos-aarch64/codex",
        "claude": "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude",
    }.get(name)
    path = _latest(bundled) if bundled else None
    if not path:
        raise FileNotFoundError(f"{name} CLI not found on PATH or in VS Code extensions")
    return path


def accounts() -> list[dict[str, str]]:
    """Configured Codex accounts that are logged in."""
    result = []
    for account in load_config().get("accounts", []):
        home = os.path.expanduser(account["codex_home"])
        if Path(home, "auth.json").exists():
            result.append({"name": account["name"], "codex_home": home})
    return result


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _update_state(mutator) -> dict[str, Any]:
    STATE_DIR.mkdir(exist_ok=True)
    with _lock, open(STATE_DIR / "state.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_state()
        mutator(state)
        STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        return state


def pick_account() -> dict[str, str]:
    """Least recently used account that is not cooling down; reserves it."""
    chosen: dict[str, str] = {}
    candidates = accounts()
    if not candidates:
        raise QuotaExhausted("no logged-in Codex account available")

    def reserve(state: dict[str, Any]) -> None:
        now = time.time()
        usable = [
            account for account in candidates
            if state.get(account["name"], {}).get("cooldown_until", 0) <= now
        ]
        if not usable:
            return
        usable.sort(key=lambda account: state.get(account["name"], {}).get("last_used", 0))
        chosen.update(usable[0])
        state.setdefault(chosen["name"], {})["last_used"] = now

    state = _update_state(reserve)
    if not chosen:
        wake = min(state.get(account["name"], {}).get("cooldown_until", 0) for account in candidates)
        raise QuotaExhausted(f"all Codex accounts cooling down until {time.ctime(wake)}")
    return chosen


_RESET_AT_RE = re.compile(
    r"try again at\s+([A-Z][a-z]{2})[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}),?\s+(\d{1,2}):(\d{2})\s*([AP]M)",
)


def _cooldown_seconds(text: str) -> int:
    """Seconds until the account's limit resets: "reset after 28m",
    "try again in 2h 5m" or "try again at Oct 18th, 2026 3:42 AM" (local time)."""
    match = _RESET_RE.search(text or "")
    if match and (match.group(1) or match.group(2)):
        return int(match.group(1) or 0) * 3600 + int(match.group(2) or 0) * 60 + 30
    at = _RESET_AT_RE.search(text or "")
    if at:
        month, day, year, hour, minute, half = at.groups()
        stamp = time.strptime(f"{month} {day} {year} {hour}:{minute} {half}", "%b %d %Y %I:%M %p")
        return max(60, int(time.mktime(stamp) - time.time()) + 30)
    return DEFAULT_COOLDOWN_SECONDS


def mark_cooldown(account_name: str, text: str) -> None:
    seconds = _cooldown_seconds(text)

    def apply(state: dict[str, Any]) -> None:
        state.setdefault(account_name, {})["cooldown_until"] = time.time() + seconds

    _update_state(apply)


def _log_call(entry: dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    with _lock, open(CALL_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_codex(
    role: str,
    prompt: str,
    *,
    cwd: Path = REPO_ROOT,
    log_dir: Path | None = None,
    schema: Path | None = None,
    timeout: int = 3600,
    wait_for_quota: bool = True,
    add_dirs: list[Path] | None = None,
) -> CallResult:
    """``codex exec`` for ``role``; on a usage limit, retry on the other account."""
    config = role_config(role)
    log_dir = Path(log_dir or STATE_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            # A rate-limited account is cooling down, so the next pick is the other one.
            account = pick_account()
        except QuotaExhausted:
            if not wait_for_quota:
                raise
            time.sleep(120)
            continue
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = log_dir / f"{role}-{stamp}-{account['name']}.last.md"
        log_path = log_dir / f"{role}-{stamp}-{account['name']}.log"
        command = [
            binary("codex"), "exec",
            "-m", config["model"],
            "-s", config.get("sandbox", "read-only"),
            "-c", f'model_reasoning_effort="{config.get("effort", "medium")}"',
            "-C", str(cwd),
            "-o", str(out_file),
            "--skip-git-repo-check",
        ]
        if config.get("sandbox", "read-only") == "read-only":
            command.append("--ephemeral")
        if schema:
            command += ["--output-schema", str(schema)]
        for directory in add_dirs or []:
            command += ["--add-dir", str(directory)]
        command.append("-")
        started = time.time()
        env = {**os.environ, "CODEX_HOME": account["codex_home"]}
        try:
            completed = subprocess.run(
                command, input=prompt, text=True, capture_output=True, env=env, timeout=timeout,
            )
            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = f"TIMEOUT after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
            code = -1
        log_path.write_text(output, encoding="utf-8")
        text = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
        seconds = round(time.time() - started, 1)
        limited = code != 0 and bool(_RATE_LIMIT_RE.search(output))
        _log_call({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "engine": "codex",
            "model": config["model"], "account": account["name"], "seconds": seconds,
            "ok": code == 0 and bool(text.strip()), "rate_limited": limited,
        })
        if limited:
            mark_cooldown(account["name"], output)
            continue
        return CallResult(
            ok=code == 0 and bool(text.strip()), text=text, role=role, model=config["model"],
            account=account["name"], seconds=seconds, log_path=log_path,
        )


def run_claude(
    role: str,
    prompt: str,
    *,
    cwd: Path = REPO_ROOT,
    log_dir: Path | None = None,
    timeout: int = 3600,
) -> CallResult:
    """``claude -p`` for ``role`` (Claude has one account: no failover)."""
    config = role_config(role)
    log_dir = Path(log_dir or STATE_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{role}-{stamp}-claude.log"
    command = [
        binary("claude"), "-p",
        "--model", config["model"],
        "--permission-mode", config.get("permission_mode", "plan"),
        "--output-format", "text",
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True, cwd=cwd, timeout=timeout,
        )
        text, err, code = completed.stdout or "", completed.stderr or "", completed.returncode
    except subprocess.TimeoutExpired as exc:
        text, err, code = str(exc.stdout or ""), f"TIMEOUT after {timeout}s", -1
    log_path.write_text(f"{err}\n---\n{text}", encoding="utf-8")
    seconds = round(time.time() - started, 1)
    limited = code != 0 and bool(_RATE_LIMIT_RE.search(err + text))
    _log_call({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "engine": "claude",
        "model": config["model"], "account": "claude", "seconds": seconds,
        "ok": code == 0 and bool(text.strip()), "rate_limited": limited,
    })
    if limited:
        raise QuotaExhausted(f"Claude quota: {(err or text)[:200]}")
    return CallResult(
        ok=code == 0 and bool(text.strip()), text=text, role=role, model=config["model"],
        account="claude", seconds=seconds, log_path=log_path, rate_limited=limited,
    )


def run_role(role: str, prompt: str, **kwargs: Any) -> CallResult:
    engine = role_config(role).get("engine", "codex")
    if engine == "claude":
        for key in ("schema", "wait_for_quota", "add_dirs"):
            kwargs.pop(key, None)
        return run_claude(role, prompt, **kwargs)
    return run_codex(role, prompt, **kwargs)
