"""Code tools: Python and shell commands inside a macOS Seatbelt sandbox.

The profile denies by default: processes may read the system and the home
folder except secrets, write only the workspace and a per-run temp folder,
and have no network unless the user approved network for that call. Runs are
argv lists (no shell), time-limited, with capped output and a scrubbed
environment (no API keys). Python runs automatically; shell commands run
automatically only when the program is on a read-only allow-list.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
from typing import Any
import uuid

from app.tools import policy
from app.tools.registry import Tool, ToolContext, ToolResult, ToolSource

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_TIMEOUT_SECONDS = 90
_MAX_OUTPUT = 20_000
_SAFE_PROGRAMS = {
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "find", "pwd", "echo", "date", "cal", "du", "df",
    "file", "stat", "sort", "uniq", "cut", "tr", "jq", "basename", "dirname", "realpath", "which", "uname",
    "sw_vers", "mdfind", "awk", "sed", "diff", "comm", "column", "shasum", "md5", "tree",
}
_UNSAFE_FLAGS = {"-delete", "-exec", "-execdir", "-ok", "-i", "--in-place"}
_ENV_KEEP = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL", "TMPDIR"}


def _sbpl(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def sandbox_profile(*, workspace: Path, scratch: Path, allow_network: bool) -> str:
    denied_reads = "\n".join(f'(deny file-read* (subpath "{_sbpl(root)}"))' for root in policy.secret_roots())
    return f"""(version 1)
(deny default)
(allow process-exec process-fork signal sysctl-read mach-lookup ipc-posix-shm file-read-metadata)
(allow file-read*)
{denied_reads}
(deny file-read* (regex #"/\\.env(\\.[^/]*)?$"))
(allow file-write* (subpath "{_sbpl(workspace)}") (subpath "{_sbpl(scratch)}")
  (literal "/dev/null") (literal "/dev/dtracehelper") (literal "/dev/tty"))
{"(allow network*)" if allow_network else "(deny network*)"}
"""


def _environment(scratch: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _ENV_KEEP}
    env.update({"TMPDIR": str(scratch), "MPLCONFIGDIR": str(scratch / "mpl"), "MPLBACKEND": "Agg",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
    return env


async def run_sandboxed(argv: list[str], *, workspace: Path, allow_network: bool = False,
                        timeout: float = _TIMEOUT_SECONDS) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    scratch = Path(tempfile.mkdtemp(prefix="aya-run-")).resolve()
    profile = scratch / "profile.sb"
    profile.write_text(sandbox_profile(workspace=workspace, scratch=scratch, allow_network=allow_network))
    before = {path: path.stat().st_mtime for path in workspace.rglob("*") if path.is_file()}
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        SANDBOX_EXEC, "-f", str(profile), *argv, cwd=str(workspace), env=_environment(scratch),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        timed_out = False
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        timed_out = True
    changed = sorted(
        str(path) for path in workspace.rglob("*")
        if path.is_file() and before.get(path) != path.stat().st_mtime and ".aya_runs" not in path.parts
    )
    return {
        "exit_code": process.returncode, "timed_out": timed_out, "seconds": round(time.perf_counter() - started, 2),
        "stdout": stdout.decode("utf-8", errors="replace")[-_MAX_OUTPUT:],
        "stderr": stderr.decode("utf-8", errors="replace")[-_MAX_OUTPUT // 2:],
        "changed_files": changed[:40],
    }


def _result(outcome: dict[str, Any], what: str) -> ToolResult:
    lines = [f"exit code {outcome['exit_code']}" + (" (timed out)" if outcome["timed_out"] else "")]
    if outcome["stdout"].strip():
        lines.append("stdout:\n" + outcome["stdout"])
    if outcome["stderr"].strip():
        lines.append("stderr:\n" + outcome["stderr"])
    if outcome["changed_files"]:
        lines.append("files created or changed:\n" + "\n".join(outcome["changed_files"]))
    ok = outcome["exit_code"] == 0 and not outcome["timed_out"]
    return ToolResult(
        content="\n".join(lines), ok=ok, untrusted=True, data=outcome,
        summary=f"{what} — {'xong' if ok else 'lỗi'} sau {outcome['seconds']}s"
        + (f", {len(outcome['changed_files'])} file" if outcome["changed_files"] else ""),
        sources=[ToolSource(title=Path(path).name, path=path) for path in outcome["changed_files"][:8]],
    )


async def run_python(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    code = str(arguments.get("code") or "")
    if not code.strip():
        return ToolResult(content="Empty code.", summary="Thiếu code", ok=False)
    workspace = context.workspace.resolve() if context.workspace.exists() else context.workspace
    runs = workspace / ".aya_runs"
    runs.mkdir(parents=True, exist_ok=True)
    script = runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.py"
    script.write_text(code, encoding="utf-8")
    outcome = await run_sandboxed([sys.executable, "-I", str(script)], workspace=workspace,
                                  allow_network=bool(arguments.get("network")))
    return _result(outcome, "Chạy Python")


def _shell_is_safe(argv: list[str]) -> bool:
    if not argv or Path(argv[0]).name not in _SAFE_PROGRAMS:
        return False
    if Path(argv[0]).name == "sed" and any(arg.startswith("-i") for arg in argv[1:]):
        return False
    return not any(arg in _UNSAFE_FLAGS for arg in argv[1:])


def _shell_argv(arguments: dict[str, Any]) -> list[str]:
    try:
        return shlex.split(str(arguments.get("command") or ""))
    except ValueError:
        return []


async def run_shell(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    argv = _shell_argv(arguments)
    if not argv:
        return ToolResult(content="Empty or unparsable command.", summary="Lệnh không hợp lệ", ok=False)
    outcome = await run_sandboxed(argv, workspace=context.workspace, allow_network=bool(arguments.get("network")))
    return _result(outcome, f"Chạy `{' '.join(argv)[:60]}`")


RUN_PYTHON = Tool(
    name="run_python",
    description=(
        "Run a Python 3 script in a sandbox (pandas, numpy, matplotlib, openpyxl available). Working folder is "
        "the workspace; it may read the user's files (not secrets) and write only inside the workspace. No network "
        "unless network=true (then the user approves). Save charts/files into the workspace and print results."
    ),
    parameters={"type": "object", "properties": {"code": {"type": "string"}, "network": {"type": "boolean"}},
                "required": ["code"]},
    # Sandboxed, no network, writes only the workspace: nothing to exfiltrate
    # with, so untrusted context alone does not require approval.
    handler=run_python, label="Chạy Python",
    needs_approval=lambda arguments, _context: bool(arguments.get("network")),
)
RUN_SHELL = Tool(
    name="run_shell",
    description=(
        "Run one command (no pipes or shell syntax) in the same sandbox. Read-only commands (ls, cat, grep, find, "
        "du, …) run directly; anything else, or network=true, waits for the user's approval."
    ),
    parameters={"type": "object", "properties": {"command": {"type": "string"}, "network": {"type": "boolean"}},
                "required": ["command"]},
    handler=run_shell, label="Chạy lệnh",
    needs_approval=lambda arguments, _context: bool(arguments.get("network")) or not _shell_is_safe(_shell_argv(arguments)),
)

CODE_TOOLS = (RUN_PYTHON, RUN_SHELL)
