#!/usr/bin/env python3
"""Quick audit: persona, RAG, tools, latency."""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7777"


def parse_sse_stream(resp) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    deltas: list[str] = []
    current_event = "message"
    for raw_line in resp:
        line = raw_line.decode("utf-8").rstrip("\n")
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            payload = json.loads(line.split(":", 1)[1].strip())
            events.append({"event": current_event, "payload": payload})
            if current_event == "message.delta":
                deltas.append(payload.get("delta") or "")
    return events, deltas


def post_sse(path: str, body: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    first_token_ms = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        events, deltas = parse_sse_stream(resp)
        for ev in events:
            if ev["event"] == "message.delta" and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "answer": "".join(deltas),
        "total_ms": round(total_ms, 1),
        "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
        "events": events,
    }


def post_json(path: str, body: dict, timeout: float = 90.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    out["_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def summarize(events: list[dict], answer: str, total_ms: float, first_token_ms) -> dict:
    route = tools_started = tools_done = []
    timings = []
    docs = 0
    failed = None
    visual_merged = None
    for ev in events:
            name = ev["event"]
            p = ev["payload"]
            if name == "agent.route.decided":
                route = [p.get("route"), p.get("selected_tools"), p.get("reason")]
            if name == "tool.started":
                tools_started.append(p.get("tool_name"))
            if name == "tool.completed":
                tools_done.append(p.get("tool_name"))
            if name == "retrieval.completed":
                docs = len(p.get("documents") or [])
            if name == "timing":
                timings.append({k: p.get(k) for k in ("stage", "ms", "tokens_per_second") if p.get(k) is not None})
            if name == "run.failed":
                failed = p.get("error")
    return {
        "total_ms": total_ms,
        "first_token_ms": first_token_ms,
        "route": route,
        "tools_started": tools_started,
        "tools_done": tools_done,
        "retrieved_docs": docs,
        "timings": timings,
        "failed": failed,
        "answer_preview": answer[:600],
        "answer_len": len(answer),
        "has_figure_ref": any(x in answer.lower() for x in ("figure", "hình", "sơ đồ", "image")),
        "aya_markers": {
            "vietnamese": any(c in answer for c in "àáạảãăâđêôơư"),
            "robot_rag_open": any(x in answer.lower() for x in ("dựa trên tài liệu", "theo ngữ cảnh được cung cấp")),
            "emoji": any(x in answer for x in ("😄", "✨", "~", "(≧▽≦)")),
        },
    }


def main() -> None:
    report = {}

    # 1 casual chat
    c = post_json("/chat", {"message": "ê bro hôm nay sao rồi?", "model": "cx/gpt-5.5"})
    report["chat_casual"] = {"ms": c["_ms"], "answer": c.get("message", "")}

    # 2 agent casual
    a1 = post_sse("/agent/run/stream", {"task": "ê bro hôm nay sao rồi?", "model": "cx/gpt-5.5", "mode": "research"})
    report["agent_casual"] = summarize(a1["events"], a1["answer"], a1["total_ms"], a1["first_token_ms"])

    # 3 agent paper
    a2 = post_sse("/agent/run/stream", {"task": "Giải thích MSF-SER pipeline chính gồm những gì?", "model": "cx/gpt-5.5", "mode": "file_qa"})
    report["agent_paper"] = summarize(a2["events"], a2["answer"], a2["total_ms"], a2["first_token_ms"])

    # 4 figure
    a3 = post_sse("/agent/run/stream", {"task": "Trong MSF-SER có figure architecture nào? Mô tả.", "model": "cx/gpt-5.5", "mode": "file_qa"})
    report["agent_figure"] = summarize(a3["events"], a3["answer"], a3["total_ms"], a3["first_token_ms"])

    # 5 hybrid
    try:
        h = post_json("/rag/search-hybrid", {"query": "MSF-SER multimodal speech emotion", "top_k": 5})
        report["rag_hybrid"] = {
            "ms": h["_ms"],
            "count": len(h.get("results") or []),
            "files": [r.get("filename") for r in (h.get("results") or [])[:3]],
            "has_figure": any(r.get("figure_id") or r.get("image_path") for r in (h.get("results") or [])),
        }
    except Exception as exc:
        report["rag_hybrid"] = {"error": str(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
