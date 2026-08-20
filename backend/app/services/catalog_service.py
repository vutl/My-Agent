from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
import json
import mimetypes
from pathlib import Path
import uuid

from app.db.sqlite import connect
from app.rag.parsers import parse_document, supported_file_type
from app.rag.retriever import build_fts_query


READ_ACCESS_LEVELS = {"read_only", "read_index", "read_write"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CatalogService:
    db_path: Path

    def resolve_file(
        self,
        *,
        filename_or_query: str,
        base_folder: str | None = None,
        allow_fuzzy: bool = True,
        max_candidates: int = 10,
    ) -> dict:
        query_path = Path(filename_or_query).expanduser()
        if query_path.is_absolute() and query_path.exists() and query_path.is_file():
            return {
                "status": "single_match",
                "candidates": [_candidate_from_path(query_path.resolve(), 1.0, "absolute path match")],
            }

        candidates: list[dict] = []
        if base_folder:
            root = Path(base_folder).expanduser().resolve()
            if root.exists() and root.is_dir():
                candidates.extend(
                    _folder_candidates(
                        root,
                        filename_or_query,
                        allow_fuzzy=allow_fuzzy,
                        max_candidates=max_candidates,
                    )
                )

        with connect(self.db_path) as connection:
            candidates.extend(
                self._registered_file_candidates(
                    connection,
                    filename_or_query,
                    allow_fuzzy=allow_fuzzy,
                    max_candidates=max_candidates,
                )
            )

        deduped = _dedupe_candidates(candidates)[:max_candidates]
        if not deduped:
            return {"status": "not_found", "candidates": []}
        if len(deduped) == 1 or deduped[0]["confidence"] >= 0.96:
            return {"status": "single_match", "candidates": [deduped[0]]}
        return {"status": "multiple_matches", "candidates": deduped}

    def read_file_direct(self, *, source_path: str, max_tokens: int = 6000) -> dict:
        path = Path(source_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {source_path}")
        if supported_file_type(path) is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if not self._can_read_path(path):
            raise ValueError("File is not inside an approved folder. Scan or index its folder first.")

        parsed = parse_document(path)
        words = parsed.text.split()
        truncated = len(words) > max_tokens
        content = " ".join(words[:max_tokens]) if truncated else parsed.text
        return {
            "source_path": str(path),
            "filename": path.name,
            "file_type": supported_file_type(path),
            "parser_name": parsed.parser_name,
            "parser_version": parsed.parser_version,
            "page_count": parsed.page_count,
            "token_estimate": len(words),
            "truncated": truncated,
            "content": content,
        }

    def scan_folder(
        self,
        *,
        folder_path: str,
        recursive: bool = False,
    ) -> dict:
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        now = utc_now()
        approved_folder_id = self._upsert_approved_folder(root, recursive, now)
        paths = root.rglob("*") if recursive else root.iterdir()
        files: list[dict] = []
        unsupported = 0

        with connect(self.db_path) as connection:
            for path in paths:
                if not path.is_file():
                    continue
                file_type = supported_file_type(path)
                if file_type is None:
                    unsupported += 1
                files.append(self._upsert_file_row(connection, path, file_type, now))

        return {
            "folder_id": approved_folder_id,
            "folder_path": str(root),
            "recursive": recursive,
            "file_count": len(files),
            "supported_files": sum(1 for item in files if item["is_supported"]),
            "unsupported_files": unsupported,
            "files": files,
        }

    def search(self, *, query: str, folder_path: str | None = None, top_k: int = 20) -> dict:
        fts_query = build_fts_query(query)
        if not fts_query:
            return {
                "query": query,
                "fts_query": "",
                "results": [],
            }

        folder_filter = str(Path(folder_path).expanduser().resolve()) if folder_path else None
        with connect(self.db_path) as connection:
            file_rows = self._search_files(connection, fts_query, folder_filter, top_k)
            card_rows = self._search_document_cards(connection, fts_query, folder_filter, top_k)

        merged = _merge_catalog_results(file_rows, card_rows, top_k)
        return {
            "query": query,
            "fts_query": fts_query,
            "retrieval_channels": ["fts_files", "fts_document_cards"],
            "results": merged,
        }

    def list_collections(self) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, name, type, scope_type, scope_id, created_at, updated_at, metadata_json
                FROM collections
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [_decode_metadata(dict(row)) for row in rows]

    def _can_read_path(self, path: Path) -> bool:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT folder_path, access_level, recursive
                FROM approved_folders
                """
            ).fetchall()

        for row in rows:
            if row["access_level"] not in READ_ACCESS_LEVELS:
                continue
            root = Path(row["folder_path"]).expanduser().resolve()
            if path.parent == root:
                return True
            if row["recursive"] and _is_relative_to(path, root):
                return True
        return False

    def _upsert_approved_folder(self, root: Path, recursive: bool, now: str) -> str:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM approved_folders WHERE folder_path = ?",
                (str(root),),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE approved_folders
                    SET access_level = 'read_only', recursive = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (int(recursive), now, existing["id"]),
                )
                return existing["id"]

            folder_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO approved_folders (
                    id, folder_path, access_level, recursive, watch_enabled,
                    created_at, updated_at, metadata_json
                )
                VALUES (?, ?, 'read_only', ?, 0, ?, ?, ?)
                """,
                (
                    folder_id,
                    str(root),
                    int(recursive),
                    now,
                    now,
                    json.dumps({"source": "catalog_scan"}),
                ),
            )
            return folder_id

    def _upsert_file_row(self, connection, path: Path, file_type: str | None, now: str) -> dict:
        stat = path.stat()
        source_path = str(path)
        mime_type, _ = mimetypes.guess_type(source_path)
        row = {
            "source_path": source_path,
            "filename": path.name,
            "extension": path.suffix.lower().lstrip(".") or None,
            "mime_type": mime_type,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "parent_folder": str(path.parent),
            "approved": True,
            "last_seen_at": now,
            "is_supported": file_type is not None,
            "risk_level": "normal",
            "metadata_json": json.dumps({"source": "catalog_scan"}),
        }
        existing = connection.execute(
            "SELECT id FROM files WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if existing:
            file_id = existing["id"]
            connection.execute(
                """
                UPDATE files
                SET filename = ?, extension = ?, mime_type = ?, size_bytes = ?,
                    created_at = ?, modified_at = ?, parent_folder = ?, approved = 1,
                    last_seen_at = ?, is_supported = ?, risk_level = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    row["filename"],
                    row["extension"],
                    row["mime_type"],
                    row["size_bytes"],
                    row["created_at"],
                    row["modified_at"],
                    row["parent_folder"],
                    row["last_seen_at"],
                    int(row["is_supported"]),
                    row["risk_level"],
                    row["metadata_json"],
                    file_id,
                ),
            )
        else:
            file_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO files (
                    id, source_path, filename, extension, mime_type, size_bytes,
                    created_at, modified_at, parent_folder, approved, last_seen_at,
                    is_supported, risk_level, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    row["source_path"],
                    row["filename"],
                    row["extension"],
                    row["mime_type"],
                    row["size_bytes"],
                    row["created_at"],
                    row["modified_at"],
                    row["parent_folder"],
                    row["last_seen_at"],
                    int(row["is_supported"]),
                    row["risk_level"],
                    row["metadata_json"],
                ),
            )

        connection.execute("DELETE FROM fts_files WHERE file_id = ?", (file_id,))
        connection.execute(
            "INSERT INTO fts_files (file_id, filename, source_path, extension) VALUES (?, ?, ?, ?)",
            (file_id, path.name, source_path, row["extension"] or ""),
        )

        row["id"] = file_id
        row["approved"] = bool(row["approved"])
        return row

    def _search_files(self, connection, fts_query: str, folder_path: str | None, top_k: int) -> list[dict]:
        where_folder = "AND files.parent_folder = ?" if folder_path else ""
        params: tuple = (fts_query, folder_path, top_k) if folder_path else (fts_query, top_k)
        rows = connection.execute(
            f"""
            SELECT
                files.id AS file_id,
                files.source_path,
                files.filename,
                files.extension,
                files.size_bytes,
                files.modified_at,
                files.is_supported,
                bm25(fts_files) AS score
            FROM fts_files
            JOIN files ON files.id = fts_files.file_id
            WHERE fts_files MATCH ?
            {where_folder}
            ORDER BY score
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [{**dict(row), "source": "file"} for row in rows]

    def _registered_file_candidates(
        self,
        connection,
        filename_or_query: str,
        *,
        allow_fuzzy: bool,
        max_candidates: int,
    ) -> list[dict]:
        exact_rows = connection.execute(
            """
            SELECT id AS file_id, source_path, filename, extension, size_bytes, modified_at
            FROM files
            WHERE filename = ? OR source_path = ?
            LIMIT ?
            """,
            (filename_or_query, filename_or_query, max_candidates),
        ).fetchall()
        candidates = [
            _candidate_from_row(dict(row), 0.98, "registered exact match")
            for row in exact_rows
        ]
        if candidates or not allow_fuzzy:
            return candidates

        rows = connection.execute(
            """
            SELECT id AS file_id, source_path, filename, extension, size_bytes, modified_at
            FROM files
            ORDER BY last_seen_at DESC
            LIMIT 200
            """
        ).fetchall()
        scored = [
            _candidate_from_row(
                dict(row),
                _name_similarity(filename_or_query, row["filename"]),
                "registered fuzzy filename match",
            )
            for row in rows
        ]
        ranked = sorted(scored, key=lambda row: row["confidence"], reverse=True)
        return [item for item in ranked if item["confidence"] >= 0.45][:max_candidates]

    def _search_document_cards(
        self,
        connection,
        fts_query: str,
        folder_path: str | None,
        top_k: int,
    ) -> list[dict]:
        where_folder = "AND files.parent_folder = ?" if folder_path else ""
        params: tuple = (fts_query, folder_path, top_k) if folder_path else (fts_query, top_k)
        rows = connection.execute(
            f"""
            SELECT
                document_cards.document_id,
                documents.file_id,
                documents.source_path,
                documents.filename,
                document_cards.title_guess,
                document_cards.doc_type,
                document_cards.language,
                document_cards.short_summary,
                document_cards.topic_tags_json,
                document_cards.project_tags_json,
                document_cards.keywords_json,
                document_cards.importance_score,
                bm25(fts_document_cards) AS score
            FROM fts_document_cards
            JOIN document_cards ON document_cards.document_id = fts_document_cards.document_id
            JOIN documents ON documents.id = document_cards.document_id
            LEFT JOIN files ON files.id = documents.file_id
            WHERE fts_document_cards MATCH ?
            {where_folder}
            ORDER BY score
            LIMIT ?
            """,
            params,
        ).fetchall()
        results: list[dict] = []
        for row in rows:
            item = dict(row)
            item["topic_tags"] = json.loads(item.pop("topic_tags_json") or "[]")
            item["project_tags"] = json.loads(item.pop("project_tags_json") or "[]")
            item["keywords"] = json.loads(item.pop("keywords_json") or "[]")
            item["source"] = "document_card"
            results.append(item)
        return results


def _merge_catalog_results(file_rows: list[dict], card_rows: list[dict], top_k: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for rank, row in enumerate(card_rows, start=1):
        key = row.get("document_id") or row.get("file_id")
        merged[key] = {
            **row,
            "rank_score": _rrf(rank) + 0.15 + float(row.get("importance_score") or 0) * 0.05,
        }
    for rank, row in enumerate(file_rows, start=1):
        key = row.get("file_id")
        current = merged.get(key)
        score = _rrf(rank) + 0.05
        if current:
            current["rank_score"] += score
            current["file_match"] = row
        else:
            merged[key] = {**row, "rank_score": score}

    return sorted(merged.values(), key=lambda item: item["rank_score"], reverse=True)[:top_k]


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _decode_metadata(row: dict) -> dict:
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    return row


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _folder_candidates(
    root: Path,
    filename_or_query: str,
    *,
    allow_fuzzy: bool,
    max_candidates: int,
) -> list[dict]:
    exact = root / filename_or_query
    if exact.exists() and exact.is_file():
        return [_candidate_from_path(exact.resolve(), 0.98, "exact filename match in base folder")]

    candidates: list[dict] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.name == filename_or_query:
            candidates.append(_candidate_from_path(path.resolve(), 0.98, "exact filename match in base folder"))
        elif allow_fuzzy:
            confidence = _name_similarity(filename_or_query, path.name)
            if confidence >= 0.45:
                candidates.append(_candidate_from_path(path.resolve(), confidence, "fuzzy filename match in base folder"))
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)[:max_candidates]


def _candidate_from_path(path: Path, confidence: float, reason: str) -> dict:
    stat = path.stat()
    return {
        "file_id": None,
        "filename": path.name,
        "source_path": str(path),
        "extension": path.suffix.lower().lstrip("."),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "confidence": round(confidence, 4),
        "reason": reason,
        "is_supported": supported_file_type(path) is not None,
    }


def _candidate_from_row(row: dict, confidence: float, reason: str) -> dict:
    return {
        "file_id": row.get("file_id"),
        "filename": row["filename"],
        "source_path": row["source_path"],
        "extension": row.get("extension"),
        "size_bytes": row.get("size_bytes"),
        "modified_at": row.get("modified_at"),
        "confidence": round(confidence, 4),
        "reason": reason,
        "is_supported": True,
    }


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    by_path: dict[str, dict] = {}
    for candidate in candidates:
        current = by_path.get(candidate["source_path"])
        if current is None or candidate["confidence"] > current["confidence"]:
            by_path[candidate["source_path"]] = candidate
    return sorted(by_path.values(), key=lambda item: item["confidence"], reverse=True)


def _name_similarity(query: str, filename: str) -> float:
    query_norm = query.lower().strip()
    filename_norm = filename.lower().strip()
    if query_norm in filename_norm:
        return 0.82
    return SequenceMatcher(None, query_norm, filename_norm).ratio()
