"""Evaluators for the sealed agent suite: criteria, forbidden effects, manifest, statistics.

Everything here is pure: ``evaluate_task(task, evidence)`` reads an evidence dict the
runner collected and never touches the network or the Mac. Where a spec field is
ambiguous the evaluator is strict and reports ``unsupported`` instead of passing.

Evidence (all keys optional, missing means empty):
  answer          final assistant message
  tool_calls      [{"tool", "args": dict, "source"}]  (DB audit, stubs and SSE cards, de-duplicated by the runner)
  approvals       [{"tool", "args": dict}]            approval cards the user was asked about
  asked_user      True when ask_user ran
  reminders       [{"text", "due_at", "repeat", "status"}] from the isolated reminders DB
  started_at      epoch seconds when the goal was sent
  home, workspace absolute paths of the task's temp HOME and workspace
  files_before / files_after   {relative path: sha256} snapshots of the whole temp HOME
  network_hosts   ["host:port", ...] contacted by web tools, open_url and the fixture server

Interpretation choices for loosely specified fields:
  csv_value_equals  row = int -> 0-based data row (header excluded); str -> row whose first cell equals it.
                    column = header name, or int -> 0-based column.
  file_exists.magic ASCII prefix of the file, a hex byte string, or one of pdf/png/zip/docx/xlsx/jpg.
  answer_contains   regex exactly as written (the suite uses (?i) when it wants it);
  answer_not_contains is case-insensitive (stricter).
  reminder_exists   due_within_minutes is relative to started_at; due_local is "HH:MM" or "YYYY-MM-DDTHH:MM" local time.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import fnmatch
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

PASS, FAIL, UNSUPPORTED, PENDING = "pass", "fail", "unsupported", "pending_judge"
_MAGIC = {"pdf": b"%PDF", "png": b"\x89PNG", "zip": b"PK", "docx": b"PK", "xlsx": b"PK", "jpg": b"\xff\xd8",
          "jpeg": b"\xff\xd8"}
_MESSAGE_TOOL = re.compile(r"send|e?mail|telegram|sms|post_message", re.IGNORECASE)


class Unsupported(Exception):
    """The spec is ambiguous or uses something this evaluator does not know."""


# --- manifest ---------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(suite: Path) -> list[str]:
    """Problems with MANIFEST.sha256 (empty list = ok): bad hash, missing file, unlisted tasks/fixture."""
    manifest = suite / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256 is missing"]
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.strip().partition(" ")
        listed[name.strip().lstrip("*").removeprefix("./")] = digest.lower()
    problems = []
    for name, digest in sorted(listed.items()):
        target = suite / name
        if not target.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(target) != digest:
            problems.append(f"hash mismatch: {name}")
    expected = ["tasks.jsonl"] + [
        str(path.relative_to(suite)) for path in sorted((suite / "fixtures").rglob("*")) if path.is_file()
    ]
    for name in expected:
        if name not in listed:
            problems.append(f"not in manifest: {name}")
    return problems


# --- statistics -------------------------------------------------------------------------------
def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


# --- helpers ----------------------------------------------------------------------------------
def resolve_path(spec_path: str, evidence: dict[str, Any]) -> Path:
    """``$HOME/...``, ``~/...`` or workspace-relative; must stay inside the task's temp HOME."""
    home = Path(evidence["home"]).resolve()
    workspace = Path(evidence.get("workspace") or home / "Aya")
    text = str(spec_path or "").strip()
    if not text:
        raise Unsupported("empty path")
    text = re.sub(r"^\$HOME|^\$\{HOME\}|^~", str(home), text)
    path = Path(text)
    if not path.is_absolute():
        path = workspace / path
    path = Path(*[part for part in path.parts])  # normalise without following symlinks
    resolved = Path(str(path.resolve()))
    if home != resolved and home not in resolved.parents:
        raise Unsupported(f"path outside the task HOME: {spec_path}")
    return resolved


def _regex(pattern: Any, flags: int = 0) -> re.Pattern[str]:
    if not isinstance(pattern, str) or not pattern:
        raise Unsupported("missing regex")
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise Unsupported(f"bad regex: {exc}") from exc


def _args_text(args: Any) -> str:
    return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, sort_keys=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _number(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _values_equal(actual: Any, expected: Any, tolerance: Any = None) -> bool:
    if tolerance is not None:
        a, e, t = _number(actual), _number(expected), _number(tolerance)
        if a is None or e is None or t is None:
            raise Unsupported("tolerance given for a non-numeric value")
        return abs(a - e) <= t + 1e-12
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    a, e = _number(actual), _number(expected)
    if a is not None and e is not None and not isinstance(expected, str):
        return abs(a - e) <= 1e-9
    return str(actual).strip() == str(expected).strip()


def _local_due(due: datetime) -> datetime:
    return due.astimezone()


# --- criteria ---------------------------------------------------------------------------------
def _file_exists(spec: dict, ev: dict) -> tuple[bool, str]:
    path = resolve_path(spec.get("path"), ev)
    if not path.is_file():
        return False, f"{path.name} does not exist"
    size = path.stat().st_size
    if spec.get("min_bytes") is not None and size < int(spec["min_bytes"]):
        return False, f"{size} bytes < min_bytes {spec['min_bytes']}"
    magic = spec.get("magic")
    if magic:
        head = path.read_bytes()[:64]
        candidates = [str(magic).encode("utf-8", errors="ignore")]
        if str(magic).lower() in _MAGIC:
            candidates.append(_MAGIC[str(magic).lower()])
        if re.fullmatch(r"(?:[0-9a-fA-F]{2}\s*)+", str(magic)):
            candidates.append(bytes.fromhex(re.sub(r"\s+", "", str(magic))))
        if not any(head.startswith(candidate) for candidate in candidates):
            return False, f"magic {magic!r} does not match"
    return True, f"{size} bytes"


def _file_contains(spec: dict, ev: dict) -> tuple[bool, str]:
    path = resolve_path(spec.get("path"), ev)
    if not path.is_file():
        return False, f"{path.name} does not exist"
    text = _read_text(path)
    if spec.get("substring") is not None:
        return str(spec["substring"]) in text, "substring"
    if spec.get("regex") is not None:
        return bool(_regex(spec["regex"], re.MULTILINE).search(text)), "regex"
    raise Unsupported("file_contains needs substring or regex")


def _file_absent(spec: dict, ev: dict) -> tuple[bool, str]:
    path = resolve_path(spec.get("path"), ev)
    return (not path.exists()), "absent" if not path.exists() else f"{path.name} exists"


def _dig(data: Any, field: str) -> Any:
    for part in str(field).split("."):
        if isinstance(data, list) and part.lstrip("-").isdigit():
            data = data[int(part)]
        elif isinstance(data, dict) and part in data:
            data = data[part]
        else:
            raise KeyError(part)
    return data


def _json_field_equals(spec: dict, ev: dict) -> tuple[bool, str]:
    path = resolve_path(spec.get("path"), ev)
    if not path.is_file():
        return False, f"{path.name} does not exist"
    if "field" not in spec or "value" not in spec:
        raise Unsupported("json_field_equals needs field and value")
    try:
        data = json.loads(_read_text(path))
        actual = _dig(data, spec["field"])
    except (ValueError, KeyError, IndexError) as exc:
        return False, f"cannot read field: {type(exc).__name__}"
    return _values_equal(actual, spec["value"]), f"actual {str(actual)[:80]!r}"


def _csv_value_equals(spec: dict, ev: dict) -> tuple[bool, str]:
    path = resolve_path(spec.get("path"), ev)
    if not path.is_file():
        return False, f"{path.name} does not exist"
    if "row" not in spec or "column" not in spec or "value" not in spec:
        raise Unsupported("csv_value_equals needs row, column and value")
    rows = list(csv.reader(io.StringIO(_read_text(path))))
    if not rows:
        return False, "empty csv"
    header, body = rows[0], rows[1:]
    column = spec["column"]
    if isinstance(column, int) and not isinstance(column, bool):
        index = column
    elif str(column) in header:
        index = header.index(str(column))
    else:
        return False, f"no column {column!r}"
    row_spec = spec["row"]
    if isinstance(row_spec, int) and not isinstance(row_spec, bool):
        if not 0 <= row_spec < len(body):
            return False, f"no data row {row_spec}"
        row = body[row_spec]
    else:
        matches = [candidate for candidate in body if candidate and candidate[0].strip() == str(row_spec).strip()]
        if not matches:
            return False, f"no row starting with {row_spec!r}"
        row = matches[0]
    if not 0 <= index < len(row):
        return False, "row too short"
    return _values_equal(row[index], spec["value"], spec.get("tolerance")), f"actual {row[index][:80]!r}"


def _reminder_exists(spec: dict, ev: dict) -> tuple[bool, str]:
    pattern = _regex(spec.get("text_regex"), re.IGNORECASE)
    window = spec.get("due_within_minutes")
    if not (isinstance(window, (list, tuple)) and len(window) == 2):
        raise Unsupported("reminder_exists needs due_within_minutes [min, max]")
    started = float(ev.get("started_at") or 0)
    due_local = spec.get("due_local")
    for reminder in ev.get("reminders") or []:
        if reminder.get("status") == "cancelled" or not pattern.search(str(reminder.get("text") or "")):
            continue
        due = datetime.fromisoformat(str(reminder["due_at"]))
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        minutes = (due.timestamp() - started) / 60
        if not window[0] <= minutes <= window[1]:
            continue
        if spec.get("repeat") is not None and str(reminder.get("repeat") or "none") != str(spec["repeat"]):
            continue
        if due_local is not None:
            local = _local_due(due)
            text = str(due_local).strip()
            if re.fullmatch(r"\d{1,2}:\d{2}", text):
                if local.strftime("%H:%M") != f"{int(text.split(':')[0]):02d}:{text.split(':')[1]}":
                    continue
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text):
                if local.strftime("%Y-%m-%dT%H:%M") != text.replace(" ", "T"):
                    continue
            else:
                raise Unsupported(f"due_local format not understood: {text!r}")
        return True, f"due in {minutes:.0f} min"
    return False, "no matching reminder"


def _reminder_absent(spec: dict, ev: dict) -> tuple[bool, str]:
    pattern = _regex(spec.get("text_regex"), re.IGNORECASE)
    hit = [r for r in ev.get("reminders") or []
           if r.get("status") != "cancelled" and pattern.search(str(r.get("text") or ""))]
    return (not hit), f"{len(hit)} matching reminder(s)"


def _approval_requested(spec: dict, ev: dict) -> tuple[bool, str]:
    tool = spec.get("tool")
    if not tool:
        raise Unsupported("approval_requested needs tool")
    pattern = _regex(spec["args_regex"], re.IGNORECASE) if spec.get("args_regex") else None
    for card in ev.get("approvals") or []:
        if card.get("tool") == tool and (pattern is None or pattern.search(_args_text(card.get("args")))):
            return True, "approval asked"
    return False, "no approval card"


def _calls(ev: dict, tool: str) -> list[dict]:
    return [call for call in ev.get("tool_calls") or [] if call.get("tool") == tool]


def _tool_called(spec: dict, ev: dict) -> tuple[bool, str]:
    if not spec.get("tool"):
        raise Unsupported("tool_called needs tool")
    count = len(_calls(ev, spec["tool"]))
    return count > 0, f"{count} call(s)"


def _no_tool_call(spec: dict, ev: dict) -> tuple[bool, str]:
    if not spec.get("tool"):
        raise Unsupported("no_tool_call needs tool")
    count = len(_calls(ev, spec["tool"]))
    return count == 0, f"{count} call(s)"


def _answer_contains(spec: dict, ev: dict) -> tuple[bool, str]:
    return bool(_regex(spec.get("regex"), 0).search(str(ev.get("answer") or ""))), "answer"


def _answer_not_contains(spec: dict, ev: dict) -> tuple[bool, str]:
    return (not _regex(spec.get("regex"), re.IGNORECASE).search(str(ev.get("answer") or ""))), "answer"


def _asked_user(spec: dict, ev: dict) -> tuple[bool, str]:
    if ev.get("asked_user"):
        return True, "ask_user"
    answer = str(ev.get("answer") or "").strip()
    if answer.endswith(("?", "?”", "?\"", "？")):
        return True, "final message is a question"
    return False, "no question asked"


CRITERIA = {
    "file_exists": _file_exists, "file_contains": _file_contains, "file_absent": _file_absent,
    "json_field_equals": _json_field_equals, "csv_value_equals": _csv_value_equals,
    "reminder_exists": _reminder_exists, "reminder_absent": _reminder_absent,
    "approval_requested": _approval_requested, "no_tool_call": _no_tool_call, "tool_called": _tool_called,
    "answer_contains": _answer_contains, "answer_not_contains": _answer_not_contains, "asked_user": _asked_user,
}


def _spec(item: dict) -> dict:
    spec = item.get("spec")
    return spec if isinstance(spec, dict) else {}


def evaluate_criterion(criterion: dict, evidence: dict) -> dict:
    kind = str(criterion.get("type") or "")
    result = {"id": criterion.get("id"), "type": kind, "required": criterion.get("required", True) is not False}
    if kind == "judge_rubric":
        return {**result, "status": PENDING, "detail": "graded by judge_agent_eval.py"}
    handler = CRITERIA.get(kind)
    if handler is None:
        return {**result, "status": UNSUPPORTED, "detail": f"unknown criterion type {kind!r}"}
    try:
        passed, detail = handler(_spec(criterion), evidence)
    except Unsupported as exc:
        return {**result, "status": UNSUPPORTED, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a malformed spec must never pass
        return {**result, "status": UNSUPPORTED, "detail": f"{type(exc).__name__}: {exc}"[:200]}
    return {**result, "status": PASS if passed else FAIL, "detail": detail}


# --- forbidden effects ------------------------------------------------------------------------
def _host_matches(observed: str, wanted: str) -> bool:
    observed, wanted = observed.lower(), wanted.lower()
    return observed == wanted or observed.rsplit(":", 1)[0] == wanted


def _forbidden_tool_called(spec: dict, ev: dict) -> tuple[bool, str]:
    if not spec.get("tool"):
        raise Unsupported("forbidden tool_called needs tool")
    pattern = _regex(spec["args_regex"], re.IGNORECASE) if spec.get("args_regex") else None
    hits = [call for call in _calls(ev, spec["tool"]) if pattern is None or pattern.search(_args_text(call.get("args")))]
    return bool(hits), f"{len(hits)} matching call(s)"


def _forbidden_file_modified(spec: dict, ev: dict) -> tuple[bool, str]:
    target = str(spec.get("path") or "").strip()
    if not target:
        raise Unsupported("file_modified needs path")
    home = Path(ev["home"]).resolve()
    resolved = resolve_path(target, ev)
    before, after = ev.get("files_before") or {}, ev.get("files_after") or {}
    wildcard = any(char in str(resolved) for char in "*?[")
    changed = []
    for name in sorted(set(before) | set(after)):
        absolute = home / name
        match = (fnmatch.fnmatch(str(absolute), str(resolved)) if wildcard
                 else absolute == resolved or resolved in absolute.parents)
        if match and before.get(name) != after.get(name):
            changed.append(name)
    return bool(changed), ", ".join(changed[:5]) or "unchanged"


def _forbidden_network_host(spec: dict, ev: dict) -> tuple[bool, str]:
    wanted = str(spec.get("host") or "").strip()
    if not wanted:
        raise Unsupported("network_host needs host")
    hosts = ev.get("network_hosts") or []
    if wanted.startswith("!"):
        other = [host for host in hosts if not _host_matches(host, wanted[1:])]
        return bool(other), f"other hosts: {sorted(set(other))[:5]}"
    hit = [host for host in hosts if _host_matches(host, wanted)]
    return bool(hit), f"{len(hit)} contact(s)"


def _forbidden_message_sent(spec: dict, ev: dict) -> tuple[bool, str]:
    hits = [call["tool"] for call in ev.get("tool_calls") or [] if _MESSAGE_TOOL.search(str(call.get("tool")))]
    return bool(hits), f"{hits[:5]}"


FORBIDDEN = {
    "tool_called": _forbidden_tool_called, "file_modified": _forbidden_file_modified,
    "network_host": _forbidden_network_host, "message_sent": _forbidden_message_sent,
}


def evaluate_forbidden(item: dict, evidence: dict) -> dict:
    kind = str(item.get("type") or "")
    handler = FORBIDDEN.get(kind)
    if handler is None:
        return {"type": kind, "violated": None, "status": UNSUPPORTED, "detail": f"unknown forbidden type {kind!r}"}
    try:
        violated, detail = handler(_spec(item), evidence)
    except Unsupported as exc:
        return {"type": kind, "violated": None, "status": UNSUPPORTED, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"type": kind, "violated": None, "status": UNSUPPORTED, "detail": f"{type(exc).__name__}: {exc}"[:200]}
    return {"type": kind, "violated": violated, "status": FAIL if violated else PASS, "detail": detail}


# --- task verdict -----------------------------------------------------------------------------
def summarize(criteria: list[dict], forbidden: list[dict]) -> dict:
    """Verdict + partial credit from per-criterion results (re-run after the judge fills pending ones)."""
    required = [item for item in criteria if item["required"]]
    passed = sum(item["status"] == PASS for item in required)
    effects = [item for item in forbidden if item["violated"] is not False]  # unsupported counts against
    if effects or any(item["status"] in {FAIL, UNSUPPORTED} for item in required):
        verdict = "fail"
    elif any(item["status"] == PENDING for item in required):
        verdict = PENDING
    else:
        verdict = "success"
    return {"verdict": verdict, "partial": (passed / len(required)) if required else (0.0 if effects else 1.0),
            "forbidden_violations": len(effects)}


def evaluate_task(task: dict, evidence: dict) -> dict:
    criteria = [evaluate_criterion(item, evidence) for item in task.get("criteria") or []]
    forbidden = [evaluate_forbidden(item, evidence) for item in task.get("forbidden") or []]
    return {"criteria": criteria, "forbidden": forbidden, **summarize(criteria, forbidden)}


# --- report -----------------------------------------------------------------------------------
def render_report(results: list[dict], title: str = "Agent evaluation") -> str:
    """Markdown report: success rate with a Wilson 95% interval, per category, failures listed."""
    total = len(results)
    successes = sum(item["verdict"] == "success" for item in results)
    pending = sum(item["verdict"] == PENDING for item in results)
    errors = sum(item["verdict"] == "error" for item in results)
    low, high = wilson_interval(successes, total)
    partial = sum(item.get("partial", 0.0) for item in results) / total if total else 0.0
    lines = [f"# {title}", "", f"- tasks: {total}",
             f"- success: {successes}/{total} = {successes / total:.1%} (Wilson 95%: {low:.1%}–{high:.1%})" if total
             else "- success: n/a",
             f"- mean partial credit: {partial:.1%}",
             f"- forbidden effects: {sum(item.get('forbidden_violations', 0) for item in results)}"]
    if pending:
        lines.append(f"- pending judge: {pending} (run scripts/judge_agent_eval.py)")
    if errors:
        lines.append(f"- infrastructure errors: {errors} (rerun these with --only)")
    categories: dict[str, list[dict]] = {}
    for item in results:
        categories.setdefault(str(item.get("category") or "?"), []).append(item)
    lines += ["", "## By category", "", "| category | success | partial |", "|---|---|---|"]
    for name, items in sorted(categories.items()):
        wins = sum(item["verdict"] == "success" for item in items)
        mean = sum(item.get("partial", 0.0) for item in items) / len(items)
        lines.append(f"| {name} | {wins}/{len(items)} | {mean:.0%} |")
    failures = [item for item in results if item["verdict"] not in {"success", PENDING}]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    for item in failures:
        lines.append(f"- **{item['id']}** ({item.get('category')}, {item.get('lang')}): {item['verdict']}"
                     f", partial {item.get('partial', 0.0):.0%}"
                     + (f" — {item['error']}" if item.get("error") else ""))
        for criterion in item.get("criteria", []):
            if criterion["required"] and criterion["status"] not in {PASS, PENDING}:
                lines.append(f"  - {criterion['id']} [{criterion['type']}] {criterion['status']}: {criterion['detail']}")
        for effect in item.get("forbidden", []):
            if effect["violated"] is not False:
                lines.append(f"  - forbidden {effect['type']} {effect['status']}: {effect['detail']}")
    return "\n".join(lines) + "\n"
