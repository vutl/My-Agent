"""Measure the router-bypass decision on a probe set (gate G2 of plan P1.10).

Each probe is sent as the first turn of a fresh conversation to an isolated
backend running with ``AYA_ROUTER_BYPASS=1`` (and eval mode with stubbed macOS
tools). The stream is read until the decision is known: a library handoff
(``message.reset`` with reason ``handoff`` followed by the router path), or the
end of the bypassed turn. The decision comes from the run's routing telemetry.

    backend/.venv/bin/python scripts/run_router_probe.py --suite ~/.aya/sealed/router-probe-v1 \
        --base-url http://127.0.0.1:7796 --out data/retrieval_eval/router-probe/run-20261028
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

ALLOWED_TOOLS = ["search_local_docs", "retrieve_visual_assets", "assistant_tools"]


def verify_manifest(suite: Path) -> None:
    for line in (suite / "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        actual = hashlib.sha256((suite / name.strip().lstrip("*")).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"MANIFEST mismatch: {name}")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (centre - spread, centre + spread)


def run_probe(base_url: str, probe: dict, timeout: float) -> dict:
    body = json.dumps({
        "task": probe["message"], "mode": "auto", "temperature": 0.2,
        "allowed_tools": ALLOWED_TOOLS, "retrieval_mode": "auto",
    }).encode()
    request = urllib.request.Request(f"{base_url}/agent/run/stream", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter()
    run_id, routes, handoff, first_delta_ms, error = None, [], False, None, None
    event = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event: "):
                    event = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                run_id = run_id or data.get("run_id")
                if event == "agent.route.decided":
                    routes.append({"route": data.get("route"), "reason": data.get("reason")})
                    if handoff or data.get("reason") != "router_bypass":
                        break  # The router path has taken over: the decision is known.
                elif event == "message.reset" and data.get("reason") == "handoff":
                    handoff = True
                elif event == "message.delta" and first_delta_ms is None and not handoff:
                    first_delta_ms = round((time.perf_counter() - started) * 1000)
                elif event in {"run.completed", "run.failed", "tool.approval.required"}:
                    break
    except Exception as exc:  # noqa: BLE001 - recorded per probe
        error = f"{type(exc).__name__}: {exc}"
    run = {}
    if run_id:
        time.sleep(1)  # Let the cancelled/finished run settle its status.
        try:
            with urllib.request.urlopen(f"{base_url}/agent/runs/{run_id}", timeout=30) as response:
                run = json.loads(response.read()) or {}
        except Exception:  # noqa: BLE001
            pass
    metadata = run.get("metadata") or {}
    if run.get("status") == "failed" and not error:
        error = f"run failed: {str(run.get('error_message') or '')[:200]}"
    bypassed = bool(metadata.get("router_bypass")) or any(r["reason"] == "router_bypass" for r in routes)
    # The handoff name comes from the run's telemetry; a bare reset only says one happened.
    decision = (metadata.get("handoff") or ("handoff_unknown" if handoff else None)
                or ("no_library" if bypassed else "router_path"))
    return {"id": probe["id"], "expect": probe["expect"], "lang": probe.get("lang"), "decision": decision,
            "library_signal": metadata.get("library_signal"), "routes": routes,
            "first_delta_ms": first_delta_ms, "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "error": error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:7796")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--pause", type=float, default=0, help="seconds between probes (gateway quota)")
    args = parser.parse_args()
    suite = args.suite.expanduser()
    verify_manifest(suite)
    probes = [json.loads(line) for line in (suite / "probes.jsonl").read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    # Probes that errored (quota, gateway) are run again; their old lines are dropped.
    kept = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    kept = [row for row in kept if not row.get("error")]
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    done = {row["id"] for row in kept}
    with results_path.open("a") as handle:
        for probe in probes:
            if probe["id"] in done:
                continue
            if args.pause:
                time.sleep(args.pause)
            result = run_probe(args.base_url, probe, args.timeout)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(result["id"], result["expect"], result["decision"], result["error"] or "", flush=True)
    results = [json.loads(line) for line in results_path.read_text().splitlines()]
    lines = ["# Router probe results", ""]
    errored = [r for r in results if r.get("error")]
    if errored:
        lines.append(f"- errored (excluded): {len(errored)}")
    results = [r for r in results if not r.get("error")]
    for expect, wanted in (("ask_library", {"ask_library", "router_path"}), ("no_library", {"no_library"})):
        rows = [r for r in results if r["expect"] == expect]
        ok = sum(r["decision"] in wanted for r in rows)
        low, high = wilson(ok, len(rows))
        lines.append(f"- {expect}: {ok}/{len(rows)} ({ok / max(1, len(rows)):.0%}, Wilson 95% {low:.0%}–{high:.0%})")
        lines.extend(f"  - miss {r['id']} ({r['lang']}): {r['decision']}, signal={r['library_signal']}, error={r['error']}"
                     for r in rows if r["decision"] not in wanted)
    by_path = {}
    for r in results:
        by_path.setdefault(r["decision"], []).append(r)
    lines.append("")
    lines.append("Decisions: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_path.items())))
    bypass_first = sorted(r["first_delta_ms"] for r in results if r["decision"] == "no_library" and r["first_delta_ms"])
    if bypass_first:
        lines.append(f"Bypassed first text: median {bypass_first[len(bypass_first) // 2]} ms, "
                     f"max {bypass_first[-1]} ms (n={len(bypass_first)})")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
