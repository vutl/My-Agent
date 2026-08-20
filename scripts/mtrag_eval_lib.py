"""Shared, evaluation-only adapters for the official MTRAG corpora.

This module deliberately has no imports from Aya's application services. Its
SQLite index is a lexical benchmark baseline and may only live below
``data/retrieval_eval/public/indexes``; it never opens the production app DB,
LanceDB, or LightRAG directories.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Iterator
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = PROJECT_ROOT / "data" / "retrieval_eval" / "public"
MTRAG_ROOT = PUBLIC_ROOT / "raw" / "github" / "mtrag"
INDEX_ROOT = PUBLIC_ROOT / "indexes"
DEFAULT_INDEX_PATH = INDEX_ROOT / "mtrag-fts-v1.sqlite"
INDEX_SCHEMA_VERSION = 1
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9][A-Za-zÀ-ỹĐđ0-9_'-]*", re.UNICODE)
_QUERY_MARKER_RE = re.compile(r"\|(?:user|agent)\|\s*:\s*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "his", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "should", "so",
    "that", "the", "their", "them", "there", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "user", "agent",
}


def ensure_isolated_index_path(path: Path, *, public_root: Path = PUBLIC_ROOT) -> Path:
    resolved = path.expanduser().resolve()
    allowed_root = (public_root / "indexes").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(
            f"MTRAG evaluation index must be below {allowed_root}; got {resolved}"
        )
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_zip_path(domain: str, *, mtrag_root: Path = MTRAG_ROOT) -> Path:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported MTRAG domain: {domain}")
    return mtrag_root / "corpora" / "passage_level" / f"{domain}.jsonl.zip"


def iter_corpus_passages(
    domain: str,
    *,
    mtrag_root: Path = MTRAG_ROOT,
) -> Iterator[dict[str, Any]]:
    archive_path = corpus_zip_path(domain, mtrag_root=mtrag_root)
    with ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if len(members) != 1:
            raise ValueError(f"Expected one JSONL in {archive_path}, got {members}")
        with archive.open(members[0]) as raw:
            for line_number, raw_line in enumerate(raw, start=1):
                if not raw_line.strip():
                    continue
                item = json.loads(raw_line.decode("utf-8"))
                passage_id = str(item.get("_id") or item.get("id") or "").strip()
                text = str(item.get("text") or "").strip()
                if not passage_id:
                    raise ValueError(
                        f"{archive_path}:{line_number} lacks passage id"
                    )
                yield {
                    "passage_id": passage_id,
                    "domain": domain,
                    "title": str(item.get("title") or "").strip(),
                    "text": text,
                }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE manifest (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE passages (
            rowid INTEGER PRIMARY KEY,
            passage_id TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX idx_mtrag_passages_domain ON passages(domain);
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            passage_id UNINDEXED,
            domain UNINDEXED,
            title,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def build_fts_index(
    output_path: Path,
    *,
    domains: Iterable[str] = DOMAINS,
    mtrag_root: Path = MTRAG_ROOT,
    public_root: Path = PUBLIC_ROOT,
    batch_size: int = 2_000,
) -> dict[str, Any]:
    """Atomically build a full, isolated lexical index from pinned ZIP files."""
    output_path = ensure_isolated_index_path(output_path, public_root=public_root)
    selected_domains = tuple(dict.fromkeys(domains))
    if not selected_domains or any(domain not in DOMAINS for domain in selected_domains):
        raise ValueError(f"domains must be a non-empty subset of {DOMAINS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_suffix(output_path.suffix + ".staging")
    if staging_path.exists():
        staging_path.unlink()

    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    skipped_empty: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    try:
        with closing(sqlite3.connect(staging_path)) as connection:
            _create_schema(connection)
            for domain in selected_domains:
                archive_path = corpus_zip_path(domain, mtrag_root=mtrag_root)
                source_hashes[domain] = sha256_file(archive_path)
                batch: list[tuple[str, str, str, str]] = []
                count = 0
                source_count = 0
                empty_count = 0
                for passage in iter_corpus_passages(domain, mtrag_root=mtrag_root):
                    source_count += 1
                    if not passage["text"]:
                        empty_count += 1
                        continue
                    batch.append(
                        (
                            passage["passage_id"],
                            domain,
                            passage["title"],
                            passage["text"],
                        )
                    )
                    if len(batch) >= batch_size:
                        _insert_passages(connection, batch)
                        count += len(batch)
                        batch.clear()
                if batch:
                    _insert_passages(connection, batch)
                    count += len(batch)
                counts[domain] = count
                source_counts[domain] = source_count
                skipped_empty[domain] = empty_count
                connection.commit()

            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "engine": "sqlite_fts5_bm25",
                "evaluation_only": True,
                "production_corpus_modified": False,
                "domains": list(selected_domains),
                "counts": counts,
                "source_counts": source_counts,
                "skipped_empty": skipped_empty,
                "source_sha256": source_hashes,
            }
            connection.executemany(
                "INSERT INTO manifest(key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(INDEX_SCHEMA_VERSION)),
                    ("manifest_json", json.dumps(manifest, sort_keys=True)),
                ],
            )
            connection.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize')")
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"MTRAG index integrity check failed: {integrity}")
        staging_path.replace(output_path)
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise

    return {
        "ok": True,
        "index": str(output_path),
        "engine": "sqlite_fts5_bm25",
        "counts": counts,
        "source_counts": source_counts,
        "skipped_empty": skipped_empty,
        "total_passages": sum(counts.values()),
        "source_sha256": source_hashes,
        "production_corpus_modified": False,
    }


def _insert_passages(
    connection: sqlite3.Connection,
    batch: list[tuple[str, str, str, str]],
) -> None:
    connection.executemany(
        "INSERT INTO passages(passage_id, domain, title, text) VALUES (?, ?, ?, ?)",
        batch,
    )
    connection.executemany(
        "INSERT INTO passages_fts(passage_id, domain, title, text) VALUES (?, ?, ?, ?)",
        batch,
    )


def read_index_manifest(index_path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(index_path)) as connection:
        row = connection.execute(
            "SELECT value FROM manifest WHERE key = 'manifest_json'"
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing manifest in MTRAG index: {index_path}")
    return json.loads(row[0])


def fts_query(text: str, *, max_terms: int = 64) -> str:
    normalized = _QUERY_MARKER_RE.sub(" ", text)
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(normalized.lower()):
        term = match.group(0).strip("_'-")
        if len(term) < 2 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    # Recent terms matter most for multi-turn queries. FTS OR keeps the query
    # robust when a conversational turn includes older, now-irrelevant topics.
    terms = terms[-max_terms:]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def search_fts(
    connection: sqlite3.Connection,
    *,
    domain: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    expression = fts_query(query)
    if not expression:
        return []
    rows = connection.execute(
        """
        SELECT passage_id, domain, title,
               snippet(passages_fts, 3, '[', ']', ' … ', 24) AS snippet,
               bm25(passages_fts, 0.0, 0.0, 2.0, 1.0) AS score
        FROM passages_fts
        WHERE passages_fts MATCH ? AND domain = ?
        ORDER BY score ASC
        LIMIT ?
        """,
        (expression, domain, top_k),
    ).fetchall()
    return [
        {
            "passage_id": str(row[0]),
            "domain": str(row[1]),
            "title": str(row[2]),
            "snippet": str(row[3]),
            "score": -float(row[4]),
        }
        for row in rows
    ]


def fetch_passages(
    connection: sqlite3.Connection,
    passage_ids: Iterable[str],
) -> dict[str, dict[str, str]]:
    """Fetch canonical candidate text without changing caller ranking order."""
    ordered_ids = list(dict.fromkeys(str(value) for value in passage_ids if value))
    passages: dict[str, dict[str, str]] = {}
    for offset in range(0, len(ordered_ids), 800):
        batch = ordered_ids[offset : offset + 800]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT passage_id, domain, title, text FROM passages "
            f"WHERE passage_id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            passages[str(row[0])] = {
                "passage_id": str(row[0]),
                "domain": str(row[1]),
                "title": str(row[2]),
                "text": str(row[3]),
            }
    return passages


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]],
    *,
    rank_constant: int = 60,
) -> list[str]:
    """Fuse ranked ID lists with deterministic first-seen tie-breaking."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking in rankings:
        per_ranking_seen: set[str] = set()
        for rank, raw_id in enumerate(ranking, start=1):
            item_id = str(raw_id)
            if not item_id or item_id in per_ranking_seen:
                continue
            per_ranking_seen.add(item_id)
            if item_id not in first_seen:
                first_seen[item_id] = seen_counter
                seen_counter += 1
            scores[item_id] += 1.0 / (rank_constant + rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))


def domain_from_collection(collection: Any) -> str | None:
    normalized = str(collection or "").lower()
    if "clapnq" in normalized:
        return "clapnq"
    if "cloud" in normalized:
        return "cloud"
    if "fiqa" in normalized:
        return "fiqa"
    if "govt" in normalized or "government" in normalized:
        return "govt"
    return None


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    return dict(qrels)


def load_human_retrieval_cases(
    *,
    query_mode: str,
    domains: Iterable[str] = DOMAINS,
    mtrag_root: Path = MTRAG_ROOT,
) -> list[dict[str, Any]]:
    if query_mode not in {"lastturn", "rewrite", "questions"}:
        raise ValueError("Human MTRAG query_mode must be lastturn, rewrite, or questions")
    cases: list[dict[str, Any]] = []
    for domain in domains:
        base = mtrag_root / "mtrag-human" / "retrieval_tasks" / domain
        qrels = load_qrels(base / "qrels" / "dev.tsv")
        query_path = base / f"{domain}_{query_mode}.jsonl"
        for item in _read_jsonl(query_path):
            query_id = str(item["_id"])
            relevant = qrels.get(query_id)
            if not relevant:
                continue
            cases.append(
                {
                    "query_id": query_id,
                    "domain": domain,
                    "query": str(item["text"]),
                    "relevant_passage_ids": sorted(relevant),
                    "suite": "mtrag-human",
                    "query_mode": query_mode,
                }
            )
    return cases


def load_un_retrieval_cases(
    *,
    query_mode: str,
    domains: Iterable[str] = DOMAINS,
    mtrag_root: Path = MTRAG_ROOT,
) -> list[dict[str, Any]]:
    if query_mode not in {"lastturn", "questions", "conversation"}:
        raise ValueError("MTRAG-UN query_mode must be lastturn, questions, or conversation")
    selected_domains = set(domains)
    qrels_by_domain = {
        domain: load_qrels(
            mtrag_root / "mtragun-human" / "retrieval_tasks" / "qrels" / f"{domain}.tsv"
        )
        for domain in selected_domains
    }
    tasks_path = mtrag_root / "mtragun-human" / "generation_tasks" / "reference.jsonl"
    cases: list[dict[str, Any]] = []
    for item in _read_jsonl(tasks_path):
        domain = domain_from_collection(item.get("Collection"))
        if domain not in selected_domains:
            continue
        query_id = str(item.get("task_id") or "")
        relevant = qrels_by_domain[domain].get(query_id)
        if not relevant:
            continue
        messages = item.get("input") or []
        if query_mode == "lastturn":
            query = next(
                (str(message.get("text") or "") for message in reversed(messages)
                 if message.get("speaker") == "user"),
                "",
            )
        elif query_mode == "questions":
            query = "\n".join(
                str(message.get("text") or "")
                for message in messages
                if message.get("speaker") == "user"
            )
        else:
            query = "\n".join(
                f"|{message.get('speaker')}|: {message.get('text') or ''}"
                for message in messages
            )
        cases.append(
            {
                "query_id": query_id,
                "domain": domain,
                "query": query,
                "relevant_passage_ids": sorted(relevant),
                "suite": "mtrag-un",
                "query_mode": query_mode,
                "answerability": item.get("answerability"),
                "question_type": item.get("Question Type"),
            }
        )
    return cases


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def retrieval_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    *,
    cutoffs: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        top = ranked_ids[:cutoff]
        relevant_at_k = sum(passage_id in relevant_ids for passage_id in top)
        result[f"hit@{cutoff}"] = float(relevant_at_k > 0)
        result[f"recall@{cutoff}"] = relevant_at_k / max(1, len(relevant_ids))
        reciprocal_rank = next(
            (1.0 / rank for rank, passage_id in enumerate(top, start=1)
             if passage_id in relevant_ids),
            0.0,
        )
        result[f"mrr@{cutoff}"] = reciprocal_rank
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, passage_id in enumerate(top, start=1)
            if passage_id in relevant_ids
        )
        ideal_hits = min(len(relevant_ids), cutoff)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        result[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
    return result


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {
        key: round(sum(float(row[key]) for row in rows) / len(rows), 6)
        for key in keys
    }
