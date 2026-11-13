"""Research lifecycle and evidence inspection; page text is never an API response."""
import json
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.db.sqlite import connect
from app.services.research import ANGLES, DEPTHS, HINTS, ResearchStore, _json, start_research
from app.services.tasks import _append_event, utc_now

router = APIRouter(prefix="/research", tags=["research"])
Config = Annotated[Settings, Depends(get_settings)]
Depth = Literal["quick", "standard", "deep"]


class CreateResearch(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    depth: Depth = "standard"
    auto_approve: bool = False
    collection_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value):
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class PlanQuestion(BaseModel):
    text: str = Field(min_length=5, max_length=300)
    angle: str = "background"
    source_hints: list[str] | None = None

    @field_validator("source_hints")
    @classmethod
    def valid_hints(cls, value):
        if value is not None and (not value or set(value) - HINTS):
            raise ValueError("unknown source hint")
        return value

    @field_validator("angle")
    @classmethod
    def valid_angle(cls, value):
        if value not in ANGLES:
            raise ValueError("unknown angle")
        return value

    @field_validator("text")
    @classmethod
    def valid_text(cls, value):
        if len(value.strip()) < 5:
            raise ValueError("subquestion too short")
        return value.strip()


class ApprovePlan(BaseModel):
    action: Literal["start", "cancel"]
    subquestions: list[PlanQuestion] | None = Field(default=None, min_length=1, max_length=12)
    depth: Depth | None = None


def _store(settings, task_id):
    store = ResearchStore(settings.sqlite_db_path, settings.data_dir)
    if store.get(task_id) is None:
        raise HTTPException(404, "Research run not found")
    return store


@router.post("", status_code=201)
def create(body: CreateResearch, settings: Config):
    try:
        task_id = start_research(settings=settings, **body.model_dump())
    except (ValueError, KeyError, sqlite3.IntegrityError):
        raise HTTPException(422, "Invalid research request") from None
    return {"task_id": task_id}


@router.get("/{task_id}")
def get(task_id: str, settings: Config):
    store = _store(settings, task_id)
    run = store.get(task_id)
    return {**{key: run[key] for key in ("stage", "stage_detail", "depth", "brief", "round", "usage", "budget", "stop_reason")},
            "subquestions": [{k: q[k] for k in ("id", "idx", "text", "angle", "source_hints", "status", "origin", "round_added")}
                             for q in store.subquestions(task_id)], "counts": store.counts(task_id)}


@router.get("/{task_id}/sources")
def sources(task_id: str, settings: Config):
    return [{k: s[k] for k in ("id", "label", "title", "url", "domain", "kind", "status", "published")}
            for s in _store(settings, task_id).sources(task_id)]


@router.get("/{task_id}/cards")
def cards(task_id: str, settings: Config, subq_id: str | None = None):
    return _store(settings, task_id).cards(task_id, subq_id)


@router.post("/{task_id}/plan")
def plan(task_id: str, body: ApprovePlan, settings: Config):
    store = _store(settings, task_id)
    # Edits, approval answer and the transition share the lock. A competing
    # cancel/approval cannot wake a worker with only half the edited plan.
    with connect(store.db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = connection.execute("SELECT * FROM research_runs WHERE task_id=?", (task_id,)).fetchone()
        task = connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        ask = connection.execute("SELECT * FROM task_events WHERE task_id=? AND kind='ask' "
            "AND json_extract(payload_json,'$.step_id')='research_plan' ORDER BY seq DESC LIMIT 1", (task_id,)).fetchone()
        if run["stage"] != "awaiting_approval" or task["status"] != "waiting_user" or not ask:
            raise HTTPException(409, "Research plan is not awaiting approval")
        depth = body.depth or run["depth"]
        brief = json.loads(run["brief_json"])
        old = list(connection.execute("SELECT * FROM research_subquestions WHERE task_id=? ORDER BY idx", (task_id,)))
        questions = [q.model_dump() for q in body.subquestions] if body.subquestions is not None else None
        if questions is not None:
            if len(questions) > DEPTHS[depth]["subq"][1]:
                raise HTTPException(422, "Too many subquestions for this depth")
            # Below the depth's minimum is allowed: the user may narrow the plan.
            connection.execute("DELETE FROM research_subquestions WHERE task_id=?", (task_id,))
            for idx, q in enumerate(questions):
                previous = next((x for x in old if x["text"] == q["text"] and x["angle"] == q["angle"]), None)
                # An edited question keeps the hints of the question it replaced
                # (same position, same angle) unless the user gave new ones.
                inherited = previous or (old[idx] if idx < len(old) and old[idx]["angle"] == q["angle"] else None)
                if q["source_hints"] is None:
                    q["source_hints"] = json.loads(inherited["source_hints_json"]) if inherited else ["web"]
                store._insert_subq(connection, task_id, idx, q, previous["origin"] if previous else "user", 0)
            brief["subquestions"] = questions
        now = utc_now()
        connection.execute("UPDATE research_runs SET depth=?,budget_json=?,brief_json=?,updated_at=? WHERE task_id=?",
                           (depth, _json(DEPTHS[depth]), _json(brief), now, task_id))
        budget = dict(max_seconds=DEPTHS[depth]["seconds"], max_steps=1000, max_tool_calls=1000)
        connection.execute("UPDATE tasks SET budget_json=?,status='running',updated_at=? WHERE id=?", (_json(budget), now, task_id))
        _append_event(connection, task_id, "answer", {"ask_seq": ask["seq"], "text": "Huỷ" if body.action == "cancel" else "Bắt đầu"}, now)
        _append_event(connection, task_id, "status", {"from": "waiting_user", "to": "running"}, now)
    from app.services.task_wakeup import TASK_WORKER_WAKEUP
    TASK_WORKER_WAKEUP.wake()
    return {"task_id": task_id}


@router.post("/{task_id}/finish")
def finish(task_id: str, settings: Config):
    store = _store(settings, task_id)
    # Only a run that is gathering evidence can be cut short; an armed flag in
    # an earlier stage would end the research with no evidence once approved.
    if store.get(task_id)["stage"] != "research":
        raise HTTPException(409, "Research is not gathering evidence")
    store.update(task_id, user_finish=True)
    from app.services.task_wakeup import TASK_WORKER_WAKEUP
    TASK_WORKER_WAKEUP.wake()
    return {"task_id": task_id}
