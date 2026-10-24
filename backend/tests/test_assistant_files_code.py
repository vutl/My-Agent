from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys

import pytest

from app.tools import code, files, policy
from app.tools.registry import ToolContext


@pytest.fixture(scope="module")
def seatbelt_available():
    if sys.platform != "darwin":
        pytest.skip("Seatbelt sandbox is macOS-only")
    # Darwin can itself be sandboxed (CI or an agent worktree). Probe the OS
    # facility before testing the application policy, without running user code.
    result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True, text=True, timeout=5,
    )
    if "sandbox_apply: Operation not permitted" in result.stderr:
        pytest.skip("Host sandbox denies nested Seatbelt sandbox_apply")
    assert result.returncode == 0, result.stderr


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    fake = (tmp_path / "home").resolve()
    (fake / ".ssh").mkdir(parents=True)
    (fake / ".ssh" / "id_ed25519").write_text("PRIVATE")
    (fake / "Library" / "Cookies").mkdir(parents=True)
    (fake / "Documents").mkdir()
    (fake / "Documents" / "notes.md").write_text("# Notes\nbudget 2026: 42 triệu\n")
    (fake / "project" ).mkdir()
    (fake / "project" / ".env").write_text("OPENAI_API_KEY=sk-secret")
    monkeypatch.setenv("HOME", str(fake))
    return fake


def _context(home: Path) -> ToolContext:
    return ToolContext(run_id="r", conversation_id="c", workspace=home / "Aya")


def run(coro):
    return asyncio.run(coro)


def test_secrets_are_never_readable(home) -> None:
    context = _context(home)
    for path in ("~/.ssh/id_ed25519", "~/Library/Cookies", "~/project/.env"):
        result = run(files.read_file({"path": path}, context))
        assert not result.ok and "private" in result.content
    listing = run(files.list_dir({"path": "~"}, context))
    assert ".ssh" not in listing.content and "Library" not in listing.content


def test_symlink_into_secrets_is_refused(home) -> None:
    context = _context(home)
    context.workspace.mkdir(parents=True)
    (context.workspace / "innocent.txt").symlink_to(home / ".ssh" / "id_ed25519")
    assert not run(files.read_file({"path": "innocent.txt"}, context)).ok


def test_read_and_search_user_files(home) -> None:
    context = _context(home)
    result = run(files.read_file({"path": "~/Documents/notes.md"}, context))
    assert result.ok and "42 triệu" in result.content and result.untrusted
    found = run(files.search_files({"pattern": "budget", "path": "~"}, context))
    assert "notes.md" in found.content and ".env" not in found.content


def test_writes_inside_workspace_run_directly_outside_need_approval(home) -> None:
    context = _context(home)
    assert not files.WRITE_FILE.requires_approval({"path": "report.md"}, context)
    assert files.WRITE_FILE.requires_approval({"path": "~/Documents/report.md"}, context)
    assert files.MOVE_PATH.requires_approval({"source": "a.txt", "destination": "~/Desktop/a.txt"}, context)
    assert files.TRASH_PATH.requires_approval({"path": "report.md"}, context)
    assert run(files.write_file({"path": "report.md", "content": "hi"}, context)).ok
    assert (context.workspace / "report.md").read_text() == "hi"
    assert not run(files.write_file({"path": "report.md", "content": "x"}, context)).ok  # no silent overwrite
    assert not run(files.write_file({"path": "/etc/aya.txt", "content": "x"}, context)).ok  # outside home


def test_edit_requires_a_unique_match_and_trash_is_reversible(home) -> None:
    context = _context(home)
    run(files.write_file({"path": "a.txt", "content": "one two one"}, context))
    assert not run(files.edit_file({"path": "a.txt", "old_text": "one", "new_text": "1"}, context)).ok
    assert run(files.edit_file({"path": "a.txt", "old_text": "two", "new_text": "2"}, context)).ok
    assert run(files.trash_path({"path": "a.txt"}, context)).ok
    assert (home / ".Trash" / "a.txt").read_text() == "one 2 one"


def test_shell_allow_list() -> None:
    assert code._shell_is_safe(["ls", "-la"])
    assert code._shell_is_safe(["grep", "-r", "x", "."])
    assert not code._shell_is_safe(["rm", "-rf", "~"])
    assert not code._shell_is_safe(["find", ".", "-delete"])
    assert not code._shell_is_safe(["sed", "-i", "", "s/a/b/", "f"])
    assert not code._shell_is_safe(["curl", "https://x"])
    assert code.RUN_SHELL.requires_approval({"command": "rm -rf ~/Documents"}, None)
    assert not code.RUN_SHELL.requires_approval({"command": "ls -la"}, None)
    assert code.RUN_PYTHON.requires_approval({"code": "1", "network": True}, None)


@pytest.mark.usefixtures("seatbelt_available")
def test_sandbox_blocks_escape_secrets_and_network_but_runs_analysis(home) -> None:
    context = _context(home)
    script = f"""
import os, socket
open("inside.txt", "w").write("ok")
results = []
for path in [{str(home / 'escape.txt')!r}, "/tmp/aya-escape.txt"]:
    try:
        open(path, "w"); results.append("ESCAPED " + path)
    except OSError:
        results.append("write-blocked")
try:
    open({str(home / '.ssh' / 'id_ed25519')!r}).read(); results.append("SECRET READ")
except OSError:
    results.append("secret-blocked")
try:
    open({str(home / 'project' / '.env')!r}).read(); results.append("ENV READ")
except OSError:
    results.append("env-blocked")
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3); results.append("NETWORK")
except OSError:
    results.append("network-blocked")
print(open({str(home / 'Documents' / 'notes.md')!r}).read().splitlines()[1])
print(" ".join(results))
print("APIKEY" if os.environ.get("OPENAI_API_KEY") else "no-key")
"""
    result = run(code.run_python({"code": script}, context))
    assert result.ok, result.content
    out = result.data["stdout"]
    assert "budget 2026" in out
    assert "ESCAPED" not in out and "SECRET READ" not in out and "ENV READ" not in out and "NETWORK" not in out
    assert out.count("write-blocked") == 2 and "secret-blocked" in out and "env-blocked" in out
    assert "network-blocked" in out and "no-key" in out
    assert (context.workspace / "inside.txt").read_text() == "ok"
    assert not (home / "escape.txt").exists()


@pytest.mark.usefixtures("seatbelt_available")
def test_sandbox_times_out(home) -> None:
    outcome = run(code.run_sandboxed([sys.executable, "-c", "import time; time.sleep(30)"],
                                     workspace=home / "Aya", timeout=1))
    assert outcome["timed_out"]


@pytest.mark.usefixtures("seatbelt_available")
def test_python_chart_lands_in_the_workspace(home) -> None:
    context = _context(home)
    script = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [2, 4, 1]); plt.savefig('chart.png'); print('saved')\n"
    )
    result = run(code.run_python({"code": script}, context))
    assert result.ok, result.content
    assert (context.workspace / "chart.png").exists()
    assert any(source.path.endswith("chart.png") for source in result.sources)
