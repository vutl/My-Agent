"""Ingest-time structured numeric facts from prose chunks.

Tables already carry their row/column structure; prose does not, so the
answer-time numeric grounding could only check that a number exists, not who
it belongs to or under which condition. This offline, resumable builder asks
an approved 9router model to restate each numeric statement of a chunk as
``(entity, attribute, conditions, value, unit)`` with a verbatim quote.

A fact is stored only when its quote occurs verbatim in the chunk and its
value occurs in the quote, so the model can restructure evidence but never
add a number. Facts are bound to the chunk's active revision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import re
import uuid
from typing import Any

from app.core.config import APPROVED_9ROUTER_MODELS
from app.db.sqlite import connect

PROMPT_VERSION = "facts-v2-source-spans"
_NUMBER_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,  ]\d+)*")
_MAX_BATCH_CHARS = 6_000
_MAX_BATCH_CHUNKS = 6

_SYSTEM_PROMPT = """You restate numeric statements from document excerpts as structured facts.
Return JSON only: {"facts": [{"chunk": <int>, "entity": str, "attribute": str,
"conditions": [str, ...], "value": str, "unit": str|null, "quote": str}]}.

Rules:
- One fact per stated number that measures something (amounts, rates, scores,
  prices, counts, durations). Skip years used as dates, section/table/figure
  numbers, citations and list markers.
- entity: who/what the number belongs to, written as in the text (a region,
  team, drug group, product, model, school...).
- attribute: what is measured (revenue, hazard ratio, goals conceded, phí vận chuyển...).
- conditions: every qualifier the number depends on, written as in the text:
  time ("Q2 2024", "năm 2023"), place, subgroup, dataset/split, and inclusion or
  exclusion phrases ("excluding taxes", "chưa gồm VAT", "without FPN").
- value: the number exactly as printed (keep separators and decimal commas).
- quote: the shortest verbatim span of the excerpt containing the value and
  enough words to show entity and attribute. Copy it character for character.
- Never compute, convert, round or infer a number that is not printed.
- Keep the document's language for all strings."""


@dataclass
class FactDraft:
    chunk_id: str
    entity: str
    attribute: str
    conditions: list[str]
    value_text: str
    unit: str | None
    quote: str
    quote_start: int = -1
    quote_end: int = -1
    validation_status: str = "unverified"


@dataclass
class ExtractionResult:
    facts: list[FactDraft] = field(default_factory=list)
    rejected: int = 0


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def validate_fact(raw: dict[str, Any], chunk_text: str) -> FactDraft | None:
    """Validate literal anchors; this is not a semantic entailment proof.

    Multi-value or repeated quotes remain unverified proposals. The runtime
    fact layer can only strengthen, never replace, canonical-source checks.
    """
    if any(not isinstance(raw.get(key), str) for key in ("quote", "value", "entity", "attribute")):
        return None
    if raw.get("unit") is not None and not isinstance(raw["unit"], str):
        return None
    quote = _normalize_space(raw.get("quote"))
    value = str(raw.get("value") or "").strip()
    entity = _normalize_space(raw.get("entity"))
    attribute = _normalize_space(raw.get("attribute"))
    if not (quote and value and entity and attribute):
        return None
    if quote not in _normalize_space(chunk_text):
        return None
    if not _NUMBER_RE.fullmatch(value) or value not in {m.group(0) for m in _NUMBER_RE.finditer(quote)}:
        return None
    raw_conditions = raw.get("conditions") or []
    if not isinstance(raw_conditions, list) or len(raw_conditions) > 12:
        return None
    if any(not isinstance(item, str) or not _normalize_space(item) for item in raw_conditions):
        return None
    conditions = [_normalize_space(item) for item in raw_conditions]
    unit = _normalize_space(raw.get("unit")) or None
    def anchored(label):
        return re.search(rf"(?<!\w){re.escape(label)}(?!\w)", quote, re.I) is not None
    # Literal grounding is necessary. Unknown/reworded labels must not be
    # promoted by a quote belonging to another entity or condition.
    if not all(anchored(label) for label in [entity, attribute, *conditions, *([unit] if unit else [])]):
        return None
    tokens = list(re.finditer(r"\S+", chunk_text))
    normal_positions = []
    for index, token in enumerate(tokens):
        if index:
            normal_positions.append(token.start())
        normal_positions.extend(range(token.start(), token.end()))
    normal = _normalize_space(chunk_text)
    position = normal.find(quote)
    # Ambiguous/multi-value quotes remain proposals. Literal presence alone
    # does not establish which of several measurements the fact describes.
    status = "source_bound" if len(list(_NUMBER_RE.finditer(quote))) == 1 and normal.count(quote) == 1 else "unverified"
    start, end = normal_positions[position], normal_positions[position + len(quote) - 1] + 1
    return FactDraft("", entity, attribute, conditions, value, unit, quote, start, end, status)


def numeric_chunks(db_path: Path, document_id: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Active-revision prose chunks that state at least one number."""
    with connect(db_path) as connection:
        document = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if document is None:
            return None, []
        revision = document["active_revision_id"]
        rows = connection.execute(
            """
            SELECT id, content FROM chunks
            WHERE document_id = ? AND chunk_type = 'text'
              AND (revision_id = ? OR (? IS NULL AND revision_id IS NULL))
            ORDER BY chunk_index ASC
            """,
            (document_id, revision, revision),
        ).fetchall()
    chunks = [
        {"chunk_id": str(row["id"]), "content": str(row["content"] or "")}
        for row in rows
        if _NUMBER_RE.search(str(row["content"] or ""))
    ]
    return revision, chunks


def _batches(chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for chunk in chunks:
        length = len(chunk["content"])
        if current and (size + length > _MAX_BATCH_CHARS or len(current) >= _MAX_BATCH_CHUNKS):
            batches.append(current)
            current, size = [], 0
        current.append(chunk)
        size += length
    if current:
        batches.append(current)
    return batches


def _chunk_fingerprint(chunks: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _build_fingerprint(document_id: str, revision: str | None, chunks: list[dict[str, Any]], model: str) -> str:
    contract = [document_id, revision, _chunk_fingerprint(chunks), model, PROMPT_VERSION,
                _SYSTEM_PROMPT, _MAX_BATCH_CHARS, _MAX_BATCH_CHUNKS,
                inspect.getsource(validate_fact), inspect.getsource(_normalize_space), _NUMBER_RE.pattern]
    return hashlib.sha256(json.dumps(contract, ensure_ascii=False).encode()).hexdigest()


async def _complete_json(client: Any, *, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    parts: list[str] = []
    async for chunk in client.stream_chat(
        model=model,
        messages=messages,
        temperature=0.0,
        num_predict=4000,
        response_format={"type": "json_object"},
    ):
        content = str(getattr(chunk, "content", "") or "")
        if content:
            parts.append(content)
    text = "".join(parts).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    payload = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(payload, dict):
        raise ValueError("fact extraction returned a non-object JSON value")
    return payload


async def extract_facts(
    chunks: list[dict[str, Any]], *, client: Any, model: str,
) -> ExtractionResult:
    """Extract and validate facts for in-memory chunks (no persistence)."""
    if model not in APPROVED_9ROUTER_MODELS:
        raise ValueError(f"Fact extraction model must be an approved 9router model, got {model!r}")
    result = ExtractionResult()
    for batch in _batches(chunks):
        excerpt = "\n\n".join(
            json.dumps({"chunk": index, "content": chunk["content"]}, ensure_ascii=False)
            for index, chunk in enumerate(batch)
        )
        payload = await _complete_json(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts (JSON lines):\n{excerpt}"},
            ],
        )
        for raw in payload.get("facts") or []:
            if not isinstance(raw, dict):
                result.rejected += 1
                continue
            index = raw.get("chunk")
            if not isinstance(index, int) or isinstance(index, bool):
                result.rejected += 1
                continue
            if not 0 <= index < len(batch):
                result.rejected += 1
                continue
            fact = validate_fact(raw, batch[index]["content"])
            if fact is None:
                result.rejected += 1
                continue
            fact.chunk_id = batch[index]["chunk_id"]
            result.facts.append(fact)
    return result


class FactStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def job_status(self, document_id: str, revision_id: str | None) -> str | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT status FROM document_fact_jobs WHERE document_id = ? AND revision_id = ?",
                (document_id, revision_id or ""),
            ).fetchone()
        return str(row["status"]) if row else None

    def replace(
        self, document_id: str, revision_id: str | None, result: ExtractionResult, *, model: str,
        build_id: str | None = None, lease_owner: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT active_revision_id FROM documents WHERE id=?", (document_id,)).fetchone()
            if current is None or current[0] != revision_id:
                raise RuntimeError("Fact activation refused: active revision changed")
            if build_id:
                build = connection.execute("SELECT * FROM document_fact_builds WHERE id=?", (build_id,)).fetchone()
                rows = connection.execute(
                    "SELECT id, content FROM chunks WHERE document_id=? AND chunk_type='text' "
                    "AND COALESCE(revision_id,'')=COALESCE(?, '') ORDER BY chunk_index", (document_id, revision_id),
                ).fetchall()
                active_chunks = [{"chunk_id": str(r[0]), "content": str(r[1] or "")} for r in rows if _NUMBER_RE.search(str(r[1] or ""))]
                if (build is None or build["lease_owner"] != lease_owner
                    or build["lease_expires_at"] <= now
                    or build["source_hash"] != _chunk_fingerprint(active_chunks)):
                    raise RuntimeError("Fact activation refused: source or lease changed")
                count = connection.execute("SELECT COUNT(*) FROM document_fact_batches WHERE build_id=?", (build_id,)).fetchone()[0]
                if count != build["expected_batches"]:
                    raise RuntimeError("Fact activation refused: incomplete batch projection")
            status = "partial" if result.rejected or any(f.validation_status != "source_bound" for f in result.facts) else "complete"
            connection.execute(
                "DELETE FROM document_facts WHERE document_id = ? AND COALESCE(revision_id, '') = ?",
                (document_id, revision_id or ""),
            )
            for fact in result.facts:
                identity = hashlib.sha256(
                    json.dumps([document_id, revision_id, fact.chunk_id, fact.quote, fact.value_text,
                                fact.entity, fact.attribute, fact.conditions, fact.unit], ensure_ascii=False).encode()
                ).hexdigest()[:32]
                connection.execute(
                    """
                    INSERT OR REPLACE INTO document_facts
                    (id, document_id, revision_id, chunk_id, entity, attribute, conditions_json,
                     value_text, unit, quote, model, prompt_version, created_at,
                     quote_start, quote_end, source_hash, validation_status, build_fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (identity, document_id, revision_id, fact.chunk_id, fact.entity, fact.attribute,
                     json.dumps(fact.conditions, ensure_ascii=False), fact.value_text, fact.unit,
                     fact.quote, model, PROMPT_VERSION, now, fact.quote_start, fact.quote_end,
                     build["source_hash"] if build_id else None, fact.validation_status, build_id),
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO document_fact_jobs
                (document_id, revision_id, status, model, prompt_version, fact_count,
                 rejected_count, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (document_id, revision_id or "", status, model, PROMPT_VERSION, len(result.facts),
                 result.rejected, now),
            )
            if build_id:
                connection.execute("UPDATE document_fact_builds SET status=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?",
                                   (status, now, build_id))

    def mark_failed(self, document_id: str, revision_id: str | None, error: str, *, model: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO document_fact_jobs
                (document_id, revision_id, status, model, prompt_version, error, updated_at)
                VALUES (?, ?, 'failed', ?, ?, ?, ?)
                """,
                (document_id, revision_id or "", model, PROMPT_VERSION, error[:500],
                 datetime.now(UTC).isoformat()),
            )

    def facts_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Active-revision facts by chunk id, ready for numeric grounding."""
        ids = [str(chunk_id) for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT f.document_id, f.revision_id, f.chunk_id, f.entity, f.attribute, f.conditions_json, f.value_text, f.unit, f.quote,
                       f.quote_start, f.quote_end, f.validation_status, f.prompt_version, c.content
                       ,f.source_hash,f.build_fingerprint
                FROM document_facts f
                JOIN documents d ON d.id = f.document_id
                JOIN chunks c ON c.id = f.chunk_id AND c.document_id=f.document_id
                WHERE f.chunk_id IN ({placeholders})
                  AND COALESCE(f.revision_id, '') = COALESCE(d.active_revision_id, '')
                  AND COALESCE(c.revision_id, '') = COALESCE(f.revision_id, '')
                  AND f.prompt_version=? AND f.validation_status='source_bound'
                """,
                [*ids, PROMPT_VERSION],
            ).fetchall()
        facts: dict[str, list[dict[str, Any]]] = {}
        current_hashes: dict[str, str] = {}
        for row in rows:
            if row["build_fingerprint"]:
                document_id = str(row["document_id"])
                if document_id not in current_hashes:
                    _, chunks = numeric_chunks(self.db_path, document_id)
                    current_hashes[document_id] = _chunk_fingerprint(chunks)
                if row["source_hash"] != current_hashes[document_id]:
                    continue
            raw = {"entity": row["entity"], "attribute": row["attribute"],
                   "conditions": json.loads(row["conditions_json"] or "[]"), "value": row["value_text"],
                   "unit": row["unit"], "quote": row["quote"]}
            draft = validate_fact(raw, str(row["content"] or ""))
            if draft is None or draft.validation_status != "source_bound" or (draft.quote_start, draft.quote_end) != (row["quote_start"], row["quote_end"]):
                continue
            facts.setdefault(str(row["chunk_id"]), []).append({
                "entity": row["entity"],
                "attribute": row["attribute"],
                "conditions": json.loads(row["conditions_json"] or "[]"),
                "value": row["value_text"],
                "unit": row["unit"],
                "quote": row["quote"],
                "quote_start": row["quote_start"], "quote_end": row["quote_end"],
                "validation_status": row["validation_status"],
            })
        return facts


async def build_document_facts(
    db_path: Path, document_id: str, *, client: Any, model: str, force: bool = False,
    lease_seconds: float = 90, retry_delay_seconds: float = 30,
) -> dict[str, Any]:
    """Checkpoint each batch; publish only after source and lease validation.

    Retry is explicit, never a model fallback. Expired leases are recoverable;
    completed batches are reused only for the identical source/model/prompt.
    """
    if model not in APPROVED_9ROUTER_MODELS:
        raise ValueError(f"Fact extraction model must be an approved 9router model, got {model!r}")
    if lease_seconds <= 0 or retry_delay_seconds < 0:
        raise ValueError("Invalid fact build lease/retry duration")
    store = FactStore(db_path)
    revision, chunks = numeric_chunks(db_path, document_id)
    batches = _batches(chunks)
    build_id = _build_fingerprint(document_id, revision, chunks, model)
    owner = uuid.uuid4().hex
    now = datetime.now(UTC)
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone() is None:
            return {"document_id": document_id, "status": "missing"}
        build = connection.execute("SELECT * FROM document_fact_builds WHERE id=?", (build_id,)).fetchone()
        if build and build["lease_owner"] and build["lease_expires_at"] > now.isoformat():
            return {"document_id": document_id, "status": "busy", "build_id": build_id}
        if build and not force:
            if build["status"] in {"complete", "partial"}:
                return {"document_id": document_id, "status": "skipped", "build_id": build_id}
            if build["next_retry_at"] and build["next_retry_at"] > now.isoformat():
                return {"document_id": document_id, "status": "retry_wait", "build_id": build_id,
                        "next_retry_at": build["next_retry_at"]}
        connection.execute(
            "INSERT OR IGNORE INTO document_fact_builds "
            "(id,document_id,revision_id,source_hash,model,prompt_version,expected_batches,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (build_id, document_id, revision, _chunk_fingerprint(chunks), model, PROMPT_VERSION,
             len(batches), now.isoformat(), now.isoformat()),
        )
        if force:
            connection.execute("DELETE FROM document_fact_batches WHERE build_id=?", (build_id,))
        connection.execute(
            "UPDATE document_fact_builds SET status='running',lease_owner=?,lease_expires_at=?,"
            "heartbeat_at=?,attempts=attempts+1,next_retry_at=NULL,error=NULL,updated_at=? WHERE id=?",
            (owner, (now + timedelta(seconds=lease_seconds)).isoformat(), now.isoformat(), now.isoformat(), build_id),
        )

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(max(0.01, lease_seconds / 3))
            tick = datetime.now(UTC)
            with connect(db_path) as connection:
                changed = connection.execute(
                    "UPDATE document_fact_builds SET heartbeat_at=?,lease_expires_at=? WHERE id=? AND lease_owner=?",
                    (tick.isoformat(), (tick + timedelta(seconds=lease_seconds)).isoformat(), build_id, owner),
                ).rowcount
            if changed != 1:
                raise RuntimeError("Fact build lease lost")

    pulse = asyncio.create_task(heartbeat())
    result = ExtractionResult()
    try:
        for index, batch in enumerate(batches):
            input_hash = _chunk_fingerprint(batch)
            with connect(db_path) as connection:
                cached = connection.execute(
                    "SELECT input_hash,result_json,rejected_count FROM document_fact_batches WHERE build_id=? AND batch_index=?",
                    (build_id, index),
                ).fetchone()
            if cached:
                if cached["input_hash"] != input_hash:
                    raise RuntimeError("Fact checkpoint fingerprint mismatch")
                extracted = ExtractionResult([FactDraft(**fact) for fact in json.loads(cached["result_json"])], cached["rejected_count"])
            else:
                extracted = await extract_facts(batch, client=client, model=model)
                tick = datetime.now(UTC)
                with connect(db_path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    lease = connection.execute("SELECT lease_owner,lease_expires_at FROM document_fact_builds WHERE id=?", (build_id,)).fetchone()
                    if lease is None or lease["lease_owner"] != owner or lease["lease_expires_at"] <= tick.isoformat():
                        raise RuntimeError("Fact checkpoint refused: lease lost")
                    connection.execute(
                        "INSERT INTO document_fact_batches (build_id,batch_index,input_hash,result_json,rejected_count,completed_at) VALUES (?,?,?,?,?,?)",
                        (build_id, index, input_hash, json.dumps([asdict(f) for f in extracted.facts], ensure_ascii=False), extracted.rejected, tick.isoformat()),
                    )
            result.facts.extend(extracted.facts)
            result.rejected += extracted.rejected
            if pulse.done():
                pulse.result()
        store.replace(document_id, revision, result, model=model, build_id=build_id, lease_owner=owner)
    except BaseException as exc:  # cancellation/provider/quota errors preserve previous active facts
        with connect(db_path) as connection:
            tick = datetime.now(UTC)
            attempts = connection.execute("SELECT attempts FROM document_fact_builds WHERE id=?", (build_id,)).fetchone()[0]
            delay = min(3600, retry_delay_seconds * 2 ** min(attempts - 1, 7))
            connection.execute(
                "UPDATE document_fact_builds SET status='failed',error=?,next_retry_at=?,lease_owner=NULL,"
                "lease_expires_at=NULL,updated_at=? WHERE id=? AND lease_owner=?",
                (type(exc).__name__, (tick + timedelta(seconds=delay)).isoformat(), tick.isoformat(), build_id, owner),
            )
        store.mark_failed(document_id, revision, f"{type(exc).__name__}: {exc}", model=model)
        raise
    finally:
        pulse.cancel()
        try:
            await pulse
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # lease loss was checked above; never mask the provider error
    return {
        "document_id": document_id,
        "status": "partial" if result.rejected or any(f.validation_status != "source_bound" for f in result.facts) else "complete",
        "build_id": build_id,
        "chunks": len(chunks),
        "facts": len(result.facts),
        "rejected": result.rejected,
    }
