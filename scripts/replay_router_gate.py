"""Replay saved evaluation conversations through the router-bypass gate (no LLM).

For each question set and its library, report the turns the gate would let
skip the intent router and, among them, the turns that expected documents.
Entity-style names are looked up in the library text, as the live gate does;
document-id links and the results-ask lookup are not replayed, so the count of
bypassed document turns is an upper bound.

    PYTHONPATH=.:backend backend/.venv/bin/python scripts/replay_router_gate.py \
        --set evals/aya_acceptance_v1.jsonl:data/sqlite/app.db ...
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from app.api.agent.retrieval import _is_strong_name, _named_query_terms
from app.api.agent.router_gate import library_signal


def library_titles(db_path: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT d.id, d.title, c.title_guess, d.filename "
            "FROM documents d LEFT JOIN document_cards c ON c.document_id = d.id"
        ).fetchall()
    finally:
        connection.close()
    titles = []
    for document_id, title, guess, filename in rows:
        title = str(title or guess or "").strip()
        filename = str(filename or "").strip()
        titles.append((document_id, f"{title} ({filename})" if title and filename else title or filename))
    return titles


def mentions_named_terms(db_path: Path, task: str) -> bool:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for term in [term for term in _named_query_terms(task) if _is_strong_name(term)][:5]:
            if connection.execute(
                "SELECT 1 FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.revision_id = d.active_revision_id AND c.content LIKE ? LIMIT 1",
                (f"%{term}%",),
            ).fetchone():
                return True
    finally:
        connection.close()
    return False


def expects_documents(row: dict) -> bool:
    return bool(row.get("expected_document") or row.get("expected_documents"))


def replay(questions: Path, db_path: Path) -> dict:
    titles = library_titles(db_path)
    conversations: dict[str, list[dict]] = defaultdict(list)
    for line in questions.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            conversations[str(row["conversation"])].append(row)
    empty_state = SimpleNamespace(active_document_ids=[], active_topic=None, active_filenames=[],
                                  conversation_documents=[])
    empty_scope = SimpleNamespace(document_ids=(), mentions=(), ambiguous_mentions=(),
                                  collection_removed_ids=())
    turns = bypassed = 0
    risky: list[dict] = []
    for rows in conversations.values():
        has_documents = False
        for row in sorted(rows, key=lambda item: int(item["turn"])):
            turns += 1
            if has_documents:
                continue  # The conversation already holds documents: the old path.
            request = SimpleNamespace(task=row["message"], mode="auto", collection_id=None,
                                      allowed_tools=["assistant_tools", "search_local_docs"])
            signal = library_signal(
                request=request, loaded_working_state=empty_state, scope_resolution=empty_scope,
                recent_document_ids=[], library_titles=titles, named_document_ids=[],
                asks_library_results=False, supports_tools=True,
                named_terms_in_library=mentions_named_terms(db_path, row["message"]),
            )
            if signal is None:
                bypassed += 1
                if expects_documents(row):
                    risky.append({"conversation": row["conversation"], "turn": row["turn"],
                                  "message": row["message"]})
            if expects_documents(row):
                has_documents = True
    return {"set": str(questions), "library": len(titles), "turns": turns,
            "bypassed": bypassed, "bypassed_document_turns": risky}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", required=True, help="questions.jsonl:app.db")
    args = parser.parse_args()
    for spec in args.set:
        questions, db_path = spec.split(":", 1)
        print(json.dumps(replay(Path(questions), Path(db_path)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
