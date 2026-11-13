"""R2's deterministic interim report; the entry point is the R3 contract."""
from collections import defaultdict
import re


def _plain(value):
    return re.sub(r"\[([WS]\d+)\]", "", str(value))


async def write_report(run_id, store, deps) -> str:
    run = store.get(run_id)
    brief = run["brief"]
    lines = ["# " + _plain(brief.get("objective") or run["question"])]
    if brief.get("assumptions"):
        lines += ["", "## Assumptions", *[f"- {_plain(x)}" for x in brief["assumptions"]]]
    sources = {s["id"]: s for s in store.sources(run_id) if s["status"] == "read" and s["label"]}
    for subq in store.subquestions(run_id):
        lines += ["", "## " + _plain(subq["text"])]
        cards = [c for c in store.cards(run_id, subq["id"]) if c["source_id"] in sources]
        lines += [f"- {_plain(c['claim'])} [{sources[c['source_id']]['label']}]" for c in cards] or ["No verified evidence found."]
    if brief.get("further_directions"):
        lines += ["", "## Further directions", *[f"- {_plain(x)}" for x in brief["further_directions"]]]
    grouped = defaultdict(list)
    for source in sources.values():
        grouped[source["kind"]].append(source)
    if grouped:
        lines += ["", "## Sources"]
    for kind, items in sorted(grouped.items()):
        lines += ["", "### " + kind]
        for source in items:
            lines.append(f"- [{source['label']}] {_plain(source['title'])} — {_plain(source['url'])} (accessed {source['fetched_at'][:10]})")
    return "\n".join(lines)
