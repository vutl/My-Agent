#!/usr/bin/env python3
"""Replay evaluation conversations with debug traces and summarise, per turn,
what the agent decided: route and router outputs, document scope, retrieval
focus, the sources in the answer prompt, and the answer's opening.

  backend/.venv/bin/python scripts/trace_turns.py --dataset <questions.jsonl> \
      --base-url http://127.0.0.1:7791 --conversations vague-ref-var-01 vague-ref-var-08 \
      --output <trace.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_agent_conversations import _run_turn  # noqa: E402


def _event_dicts(events: list) -> list[dict]:
    return [event.__dict__ if hasattr(event, "__dict__") else event for event in events]


def _first(events: list[dict], name: str) -> dict:
    return next((event.get("data") or {} for event in events if event.get("event") == name), {})


def _trace(base_url: str, run_id: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url}/agent/runs/{run_id}/debug-trace", timeout=30) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001 - a missing trace is reported, not fatal
        return {"error": str(exc)}


def _prompt_sources(trace: dict) -> list[str]:
    text = json.dumps(trace, ensure_ascii=False)
    marker = text.find("final_prompt")
    prompt = text[marker:] if marker >= 0 else ""
    import re

    return list(dict.fromkeys(re.findall(r"\[(S\d+|SOURCE \d+)\]\\nfile: ([^\\]+)", prompt)))[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--conversations", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = []
    for name in args.conversations:
        conversation_id = None
        for turn in sorted((row for row in rows if row["conversation"] == name), key=lambda row: row["turn"]):
            result = _run_turn(
                base_url=args.base_url,
                case={"case_id": f"{name}-t{turn['turn']}", "message": turn["message"],
                      "expected_document_ids": [], "expected_route": "file_qa"},
                conversation_id=conversation_id, model=None, timeout=600, debug_trace=True,
            )
            events = _event_dicts(result["events"])
            started = _first(events, "run.started")
            conversation_id = conversation_id or started.get("conversation_id")
            decided = _first(events, "agent.route.decided")
            trace = _trace(args.base_url, started.get("run_id", ""))
            entry = {
                "conversation": name, "turn": turn["turn"], "message": turn["message"],
                "expected": turn.get("expected_document") or turn.get("expected_documents"),
                "route": decided.get("route"), "reason": decided.get("reason"),
                "library_document_ids": decided.get("library_document_ids"),
                "referent_document_ids": decided.get("referent_document_ids"),
                "active_before": (decided.get("working_state") or {}).get("active_filenames"),
                "retrieval_focus": _first(events, "retrieval.started").get("focus_document_ids"),
                "rewrite": {key: _first(events, "query.rewritten").get(key)
                            for key in ("current_topic", "required_entities", "focus_document_ids")},
                "prompt_sources": _prompt_sources(trace),
                # The trace is size-capped; answer.citations lists every prompt
                # source (the last event is the final answer's prompt).
                "citations": [
                    (item.get("id"), item.get("filename"), item.get("page_number"), item.get("whole_document"))
                    for item in ([event.get("data") or {} for event in events if event.get("event") == "answer.citations"]
                                 or [{}])[-1].get("citations", [])
                ],
                "resets": [(event.get("data") or {}).get("reason") for event in events if event.get("event") == "message.reset"],
                "failed": _first(events, "run.failed").get("error"),
                "answer_head": str(result.get("answer") or "")[:400],
            }
            report.append(entry)
            print(json.dumps({k: entry[k] for k in ("conversation", "turn", "reason", "referent_document_ids",
                                                     "active_before", "retrieval_focus")}, ensure_ascii=False))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
