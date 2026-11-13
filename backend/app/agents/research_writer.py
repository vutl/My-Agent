"""R2's deterministic interim report; the entry point is the R3 contract."""
from collections import defaultdict
import re


LABELS = {
    "vi": {"assumptions": "Giả định", "none": "Chưa tìm được bằng chứng đã kiểm chứng.",
           "further": "Hướng tìm hiểu tiếp", "sources": "Nguồn", "accessed": "truy cập",
           "web": "Trang web", "pdf": "PDF", "paper": "Bài báo khoa học", "wiki": "Wikipedia",
           "library": "Thư viện của cậu"},
    "en": {"assumptions": "Assumptions", "none": "No verified evidence found.",
           "further": "Further directions", "sources": "Sources", "accessed": "accessed",
           "web": "Web pages", "pdf": "PDFs", "paper": "Papers", "wiki": "Wikipedia", "library": "Your library"},
}


def _plain(value):
    return re.sub(r"\[([WS]\d+)\]", "", str(value))


async def write_report(run_id, store, deps) -> str:
    run = store.get(run_id)
    brief = run["brief"]
    text = LABELS["vi" if str(run.get("language") or "vi").casefold().startswith("vi") else "en"]
    lines = ["# " + _plain(brief.get("objective") or run["question"])]
    if brief.get("assumptions"):
        lines += ["", "## " + text["assumptions"], *[f"- {_plain(x)}" for x in brief["assumptions"]]]
    sources = {s["id"]: s for s in store.sources(run_id) if s["status"] == "read" and s["label"]}
    for subq in store.subquestions(run_id):
        lines += ["", "## " + _plain(subq["text"])]
        cards = [c for c in store.cards(run_id, subq["id"]) if c["source_id"] in sources]
        lines += [f"- {_plain(c['claim'])} [{sources[c['source_id']]['label']}]" for c in cards] or [text["none"]]
    if brief.get("further_directions"):
        lines += ["", "## " + text["further"], *[f"- {_plain(x)}" for x in brief["further_directions"]]]
    grouped = defaultdict(list)
    for source in sources.values():
        grouped[source["kind"]].append(source)
    if grouped:
        lines += ["", "## " + text["sources"]]
    for kind, items in sorted(grouped.items()):
        lines += ["", "### " + text.get(kind, kind)]
        for source in items:
            lines.append(f"- [{source['label']}] {_plain(source['title'])} — {_plain(source['url'])} ({text['accessed']} {source['fetched_at'][:10]})")
    return "\n".join(lines)
