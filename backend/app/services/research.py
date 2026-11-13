"""Durable research checkpoints and evidence, with extracted text kept outside SQLite."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse
import uuid

from app.db.sqlite import connect
from app.services.tasks import TaskStore, utc_now
from app.tools.search_providers import canonical_url

DEPTHS = {
    "quick": {"subq": (3, 4), "rounds": 1, "pages": 15, "lead_calls": 10, "worker_calls": 40, "seconds": 600},
    "standard": {"subq": (5, 8), "rounds": 2, "pages": 40, "lead_calls": 20, "worker_calls": 120, "seconds": 1500},
    "deep": {"subq": (8, 12), "rounds": 4, "pages": 100, "lead_calls": 40, "worker_calls": 300, "seconds": 3600},
}
STAGES = {"scope", "clarify", "plan", "awaiting_approval", "research", "synthesis", "done"}
ANGLES = {"background", "state_of_the_art", "numbers", "opposing_views", "recent_news", "local_context", "how_to"}
HINTS = {"web", "news", "academic", "library", "official", "forums"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
    question TEXT NOT NULL, language TEXT NOT NULL DEFAULT 'vi', depth TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'scope', stage_detail TEXT,
    brief_json TEXT NOT NULL DEFAULT '{}', round INTEGER NOT NULL DEFAULT 0,
    budget_json TEXT NOT NULL, usage_json TEXT NOT NULL,
    stop_reason TEXT, auto_approve INTEGER NOT NULL DEFAULT 0, user_finish INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_subquestions (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES research_runs(task_id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, text TEXT NOT NULL, angle TEXT NOT NULL, source_hints_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', origin TEXT NOT NULL, round_added INTEGER NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_research_subq ON research_subquestions(task_id, idx);
CREATE TABLE IF NOT EXISTS research_queries (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES research_runs(task_id) ON DELETE CASCADE,
    subq_id TEXT NOT NULL REFERENCES research_subquestions(id) ON DELETE CASCADE,
    round INTEGER NOT NULL, query TEXT NOT NULL, providers_json TEXT NOT NULL, hits_json TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(task_id, subq_id, round, query)
);
CREATE TABLE IF NOT EXISTS research_sources (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES research_runs(task_id) ON DELETE CASCADE,
    canonical_url TEXT NOT NULL, url TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL,
    domain TEXT NOT NULL, published TEXT, status TEXT NOT NULL DEFAULT 'found', label TEXT,
    links_json TEXT NOT NULL DEFAULT '[]', text_path TEXT, chars INTEGER NOT NULL DEFAULT 0, error TEXT, fetched_at TEXT,
    UNIQUE(task_id, canonical_url), UNIQUE(task_id, label)
);
CREATE TABLE IF NOT EXISTS research_cards (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES research_runs(task_id) ON DELETE CASCADE,
    subq_id TEXT NOT NULL REFERENCES research_subquestions(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
    round INTEGER NOT NULL, claim TEXT NOT NULL, quote TEXT NOT NULL, numbers_json TEXT NOT NULL,
    locator TEXT NOT NULL, stance TEXT NOT NULL, confidence TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE(task_id, source_id, quote)
);
CREATE TABLE IF NOT EXISTS research_card_subquestions (
    card_id TEXT NOT NULL REFERENCES research_cards(id) ON DELETE CASCADE,
    subq_id TEXT NOT NULL REFERENCES research_subquestions(id) ON DELETE CASCADE,
    PRIMARY KEY(card_id, subq_id)
);
CREATE INDEX IF NOT EXISTS idx_research_card_subq ON research_card_subquestions(subq_id, card_id);
INSERT OR IGNORE INTO research_card_subquestions(card_id, subq_id)
    SELECT id, subq_id FROM research_cards;
"""


def _decode(row):
    if row is None:
        return None
    value = dict(row)
    for key in list(value):
        if key.endswith("_json"):
            value[key[:-5]] = json.loads(value.pop(key))
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def budget_ratio(run, budget=None):
    budget = budget or run["budget"]
    return max(run["usage"].get(key, 0) / budget[key] for key in ("lead_calls", "worker_calls", "pages", "seconds"))


class ResearchStore:
    def __init__(self, db_path: Path, data_dir: Path | None = None):
        self.db_path = Path(db_path)
        self.root = (Path(data_dir) if data_dir is not None else self.db_path.parent.parent) / "research"

    def create(self, task_id, question, depth="standard", auto_approve=False):
        if depth not in DEPTHS:
            raise ValueError("unknown depth")
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO research_runs(task_id,question,depth,budget_json,usage_json,auto_approve,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)", (task_id, question, depth, _json(DEPTHS[depth]),
                _json(dict(lead_calls=0, worker_calls=0, pages=0, seconds=0, cards_rejected=0)), int(auto_approve), now, now))
        return self.get(task_id)

    def get(self, task_id):
        with connect(self.db_path) as connection:
            return _decode(connection.execute("SELECT * FROM research_runs WHERE task_id=?", (task_id,)).fetchone())

    def update(self, task_id, **values):
        allowed = {"language", "depth", "stage", "stage_detail", "brief", "round", "budget", "stop_reason", "user_finish"}
        if values.keys() - allowed or "stage" in values and values["stage"] not in STAGES:
            raise ValueError("invalid research update")
        columns = {key + "_json" if key in {"brief", "budget"} else key:
                   _json(value) if key in {"brief", "budget"} else value for key, value in values.items()}
        columns["updated_at"] = utc_now()
        with connect(self.db_path) as connection:
            connection.execute("UPDATE research_runs SET " + ",".join(f"{key}=?" for key in columns) + " WHERE task_id=?",
                               (*columns.values(), task_id))

    def add_usage(self, task_id, **increments):
        if increments.keys() - {"lead_calls", "worker_calls", "pages", "seconds", "cards_rejected"}:
            raise ValueError("unknown usage counter")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in increments.values()):
            raise ValueError("usage must be finite and non-negative")
        with connect(self.db_path) as connection:
            for key, value in increments.items():
                connection.execute("UPDATE research_runs SET usage_json=json_set(usage_json, ?, "
                    "COALESCE(json_extract(usage_json, ?),0)+?), updated_at=? WHERE task_id=?",
                    (f"$.{key}", f"$.{key}", value, utc_now(), task_id))

    def subquestions(self, task_id):
        with connect(self.db_path) as connection:
            return [_decode(r) for r in connection.execute(
                "SELECT * FROM research_subquestions WHERE task_id=? ORDER BY idx", (task_id,))]

    def replace_subquestions(self, task_id, questions, origin="plan"):
        if origin not in {"plan", "user", "expansion", "conflict"}:
            raise ValueError("invalid origin")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = {_decode(r)["text"]: _decode(r) for r in connection.execute(
                "SELECT * FROM research_subquestions WHERE task_id=?", (task_id,))}
            connection.execute("DELETE FROM research_subquestions WHERE task_id=?", (task_id,))
            for idx, q in enumerate(questions):
                previous = old.get(q["text"])
                self._insert_subq(connection, task_id, idx, q,
                    previous["origin"] if previous and previous["angle"] == q.get("angle", "background") else origin, 0)

    @staticmethod
    def _insert_subq(connection, task_id, idx, q, origin, round_added):
        text = q["text"].strip()
        angle = q.get("angle", "background")
        if origin not in {"plan", "user", "expansion", "conflict"}:
            raise ValueError("invalid origin")
        if not 5 <= len(text) <= 300 or angle not in ANGLES:
            raise ValueError("invalid subquestion")
        hints = [x for x in q.get("source_hints", ["web"]) if x in HINTS]
        connection.execute("INSERT INTO research_subquestions(id,task_id,idx,text,angle,source_hints_json,origin,round_added) "
            "VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), task_id, idx, text, angle, _json(hints), origin, round_added))

    def add_subquestion(self, task_id, question, origin, round_added):
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            idx = connection.execute("SELECT COUNT(*) FROM research_subquestions WHERE task_id=?", (task_id,)).fetchone()[0]
            self._insert_subq(connection, task_id, idx, question, origin, round_added)

    def checkpoint(self, subq_id, *, status=None, state=None):
        if status is not None and status not in {"pending", "active", "covered", "exhausted"}:
            raise ValueError("invalid subquestion status")
        with connect(self.db_path) as connection:
            if status is not None:
                connection.execute("UPDATE research_subquestions SET status=? WHERE id=?", (status, subq_id))
            if state is not None:
                connection.execute("UPDATE research_subquestions SET state_json=? WHERE id=?", (_json(state), subq_id))

    def queries(self, task_id, subq_id, round):
        with connect(self.db_path) as connection:
            return [_decode(r) for r in connection.execute(
                "SELECT * FROM research_queries WHERE task_id=? AND subq_id=? AND round=? ORDER BY created_at,id",
                (task_id, subq_id, round))]

    def record_query(self, task_id, subq_id, round, query, providers, hits):
        with connect(self.db_path) as connection:
            connection.execute("INSERT OR IGNORE INTO research_queries VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), task_id, subq_id, round, query, _json(providers), _json(hits), utc_now()))

    def source(self, task_id, hit):
        url = hit["url"]
        kind = hit.get("kind", "web")
        if kind not in {"web", "pdf", "paper", "wiki", "library"}:
            raise ValueError("invalid source kind")
        key = url if kind == "library" else canonical_url(url)
        if not key:
            raise ValueError("invalid source URL")
        with connect(self.db_path) as connection:
            connection.execute("INSERT OR IGNORE INTO research_sources(id,task_id,canonical_url,url,title,kind,domain,published) "
                "VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), task_id, key, url, (hit.get("title") or url)[:500], kind,
                "library" if kind == "library" else urlparse(url).netloc.casefold().removeprefix("www."), hit.get("published")))
            return _decode(connection.execute("SELECT * FROM research_sources WHERE task_id=? AND canonical_url=?",
                                              (task_id, key)).fetchone())

    def sources(self, task_id):
        with connect(self.db_path) as connection:
            return [_decode(r) for r in connection.execute("SELECT * FROM research_sources WHERE task_id=? ORDER BY rowid", (task_id,))]

    def get_source(self, source_id):
        with connect(self.db_path) as connection:
            return _decode(connection.execute("SELECT * FROM research_sources WHERE id=?", (source_id,)).fetchone())

    def mark_source(self, source_id, *, text=None, error=None, title=None, kind=None, published=None, links=None):
        source = self.get_source(source_id)
        path = self.root / source["task_id"] / f"{source_id}.txt"
        readable = text is not None and len(text.strip()) >= 200
        if readable:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{uuid.uuid4()}.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            label = connection.execute("SELECT label FROM research_sources WHERE id=?", (source_id,)).fetchone()[0]
            if readable and not label:
                prefix = "S" if source["kind"] == "library" else "W"
                number = connection.execute("SELECT COALESCE(MAX(CAST(SUBSTR(label,2) AS INTEGER)),0)+1 "
                    "FROM research_sources WHERE task_id=? AND label LIKE ?", (source["task_id"], prefix + "%")).fetchone()[0]
                label = f"{prefix}{number}"
            connection.execute("UPDATE research_sources SET status=?,label=?,text_path=?,chars=?,error=?,fetched_at=?,"
                "title=?,kind=?,published=?,links_json=? WHERE id=?", ("read" if readable else "unreadable", label,
                str(path) if readable else None, len(text or ""), error, utc_now(), (title or source["title"])[:500],
                kind or source["kind"], published or source["published"], _json(links or []), source_id))
        return self.get_source(source_id)

    def text(self, source):
        return Path(source["text_path"]).read_text(encoding="utf-8") if source["status"] == "read" else ""

    def merge_library_chunks(self, source_id, chunks):
        """Append distinct retrieved passages with their document/page headers.

        The runner holds its per-source lock across this read/merge/write.
        """
        source = self.get_source(source_id)
        if source["kind"] != "library":
            raise ValueError("not a library source")
        existing = [part.strip() for part in re.split(r"(?m)(?=^Document: .*?; page [^\n]+\n)",
                                                     self.text(source)) if part.strip()]
        seen = {" ".join(part.split()) for part in existing}
        for chunk in chunks:
            key = " ".join(chunk.split())
            if key and key not in seen:
                existing.append(chunk.strip())
                seen.add(key)
        text = "\n\n".join(existing)
        if text == self.text(source):
            return source
        return self.mark_source(source_id, text=text)

    def add_card(self, task_id, subq_id, source_id, round, card):
        """Count a quote once globally, but retain every subquestion association."""
        if card["stance"] not in {"supports", "contradicts", "neutral"} or card["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("invalid card")
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute("SELECT 1 FROM research_sources WHERE id=? AND task_id=? AND status='read'",
                                      (source_id, task_id)).fetchone():
                return False
            if not connection.execute("SELECT 1 FROM research_subquestions WHERE id=? AND task_id=?",
                                      (subq_id, task_id)).fetchone():
                raise ValueError("subquestion does not belong to run")
            added = connection.execute("INSERT OR IGNORE INTO research_cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), task_id, subq_id, source_id, round, card["claim"], card["quote"],
                 _json(card.get("numbers", [])), card.get("locator", ""), card["stance"], card["confidence"], utc_now())).rowcount > 0
            card_id = connection.execute("SELECT id FROM research_cards WHERE task_id=? AND source_id=? AND quote=?",
                                         (task_id, source_id, card["quote"])).fetchone()[0]
            connection.execute("INSERT OR IGNORE INTO research_card_subquestions VALUES (?,?)", (card_id, subq_id))
            return added

    def cards(self, task_id, subq_id=None):
        """Return unique cards with all associations, or this question's evidence."""
        with connect(self.db_path) as connection:
            cards = [_decode(r) for r in connection.execute("SELECT c.*,s.label,s.domain, "
                "(SELECT json_group_array(a.subq_id) FROM research_card_subquestions a WHERE a.card_id=c.id) AS subq_ids_json "
                "FROM research_cards c "
                "JOIN research_sources s ON s.id=c.source_id WHERE c.task_id=?" +
                (" AND EXISTS (SELECT 1 FROM research_card_subquestions a WHERE a.card_id=c.id AND a.subq_id=?)" if subq_id else "") + " ORDER BY c.created_at,c.id",
                (task_id, subq_id) if subq_id else (task_id,))]
        if subq_id:
            for card in cards:
                card["subq_id"] = subq_id
        return cards

    def counts(self, task_id):
        sources = self.sources(task_id)
        return dict(sources_found=len(sources), sources_read=sum(x["status"] == "read" for x in sources),
            unreadable=sum(x["status"] == "unreadable" for x in sources), cards=len(self.cards(task_id)),
            cards_rejected=self.get(task_id)["usage"].get("cards_rejected", 0))

    def delete(self, task_id):
        # Resolve the folder from the stored ID, never an unvalidated API path.
        run = self.get(task_id)
        if run is None:
            return
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM research_runs WHERE task_id=?", (task_id,))
        shutil.rmtree(self.root / run["task_id"], ignore_errors=True)


def start_research(*, question, settings, conversation_id=None, depth="standard", auto_approve=False,
                   collection_id=None):
    store = TaskStore(settings.sqlite_db_path)
    budget = DEPTHS[depth]
    task = store.create(question, conversation_id, kind="research",
        budget=dict(max_seconds=budget["seconds"], max_steps=1000, max_tool_calls=1000))
    ResearchStore(settings.sqlite_db_path, settings.data_dir).create(task.id, question, depth, auto_approve)
    store.append_event(task.id, "note", {"execution": {"collection_id": collection_id,
        "model": settings.research_lead_model, "run_id": task.id}})
    store.transition(task.id, "running")
    return task.id
