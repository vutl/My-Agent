#!/usr/bin/env python3
"""Prepare, validate, and publish retrieval-evaluation fixtures.

Proposal generation is deterministic and structural. It never promotes its
own labels: only an explicitly audited reviewed file can become dev-v2.1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402


# A proposal revision is intentionally distinct from an audited rewrite-policy
# fingerprint.  Bump it whenever the deterministic proposal builder changes,
# so a reviewer cannot accidentally attest to a different generated suite.
PROPOSAL_REVISION = "dev-v2.1-proposal-r3"


def _norm(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(_norm(value).encode()).hexdigest()


def _sha_case_ids(rows: Iterable[dict[str, Any]]) -> str:
    case_ids = "\n".join(str(row.get("case_id") or "") for row in rows) + "\n"
    return hashlib.sha256(case_ids.encode()).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"proposal file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object JSON at {path}:{line_number}")
        rows.append(row)
    return rows


def _review_packet_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make the semantic review decision explicit without approving a case.

    This packet deliberately repeats the locator and its source excerpt rather
    than resolving it to a runtime UUID.  A reviewer can judge whether the
    question is actually answered by the labelled span/artifact after an
    isolated re-index, and the decision remains portable across UUID changes.
    """

    evidence: list[dict[str, Any]] = []
    for locator in row.get("evidence_groups") or []:
        if not isinstance(locator, dict):
            continue
        evidence.append(
            {
                "group_id": locator.get("group_id"),
                "kind": locator.get("kind"),
                "document": locator.get("document"),
                "page_number": locator.get("page_number"),
                "section": locator.get("section"),
                "artifact_index": locator.get("artifact_index"),
                "source_excerpt": locator.get("quote") or locator.get("caption") or "",
                "locator": locator,
            }
        )
    return {
        "case_id": row.get("case_id"),
        "review_status": "pending",
        "case": {
            "category": row.get("category"),
            "language": row.get("language"),
            "raw_query": row.get("raw_query"),
            "standalone_full": row.get("standalone_full"),
            "standalone_anchor": row.get("standalone_anchor"),
            "conversation_context": row.get("conversation_context"),
            "working_state": row.get("working_state"),
            "rewrite_provenance": row.get("rewrite_provenance"),
        },
        "evidence_to_review": evidence,
        "attestation_template": {
            "decision": "accept_or_edit_or_reject",
            "language_reviewed": False,
            "qrels_reviewed": False,
            "rewrite_reviewed": False,
            "reviewer": "",
            "reviewed_at": "",
            "notes": "Explain every qrel/query change; do not derive rewrites from gold evidence.",
        },
    }


def _load_manual_audit(path: Path, proposal: list[dict[str, Any]], proposal_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"manual audit manifest does not exist: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid manual audit JSON: {path}") from exc
    if not isinstance(audit, dict):
        raise RuntimeError("manual audit manifest must be a JSON object")
    if str(audit.get("proposal_sha256") or "") != _sha_file(proposal_path):
        raise RuntimeError("manual audit manifest targets a different proposal checksum")
    if str(audit.get("case_ids_sha256") or "") != _sha_case_ids(proposal):
        raise RuntimeError("manual audit manifest targets a different ordered case set")
    if int(audit.get("reviewed_case_count") or 0) != len(proposal):
        raise RuntimeError("manual audit manifest does not attest every proposal case")
    if not str(audit.get("reviewer") or "").strip() or not str(audit.get("reviewed_at") or "").strip():
        raise RuntimeError("manual audit manifest needs reviewer and reviewed_at")
    overrides = audit.get("text_locator_overrides") or {}
    if not isinstance(overrides, dict):
        raise RuntimeError("manual audit text_locator_overrides must be an object")
    known = {str(row.get("case_id") or "") for row in proposal}
    unknown = set(overrides) - known
    if unknown:
        raise RuntimeError(f"manual audit references unknown case IDs: {sorted(unknown)[:3]}")
    return audit


def _load_semantic_audit(path: Path, proposal: list[dict[str, Any]], proposal_path: Path) -> dict[str, Any]:
    """Verify an independent model attestation without calling it human review."""
    if not path.is_file():
        raise RuntimeError(f"semantic audit report does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid semantic audit JSON: {path}") from exc
    if not isinstance(report, dict) or report.get("promotion_eligible") is not True:
        raise RuntimeError("semantic audit is not promotion-eligible")
    if str(report.get("suite_sha256") or "") != _sha_file(proposal_path):
        raise RuntimeError("semantic audit targets a different proposal checksum")
    counts = report.get("counts") or {}
    if int(counts.get("accept") or 0) != len(proposal) or int(counts.get("reject") or 0) or int(counts.get("uncertain") or 0):
        raise RuntimeError("semantic audit does not accept every proposal case")
    reviews = report.get("reviews") or []
    expected = [str(row.get("case_id") or "") for row in proposal]
    reviewed = [str(item.get("case_id") or "") for item in reviews if isinstance(item, dict)]
    if sorted(reviewed) != sorted(expected) or len(reviewed) != len(set(reviewed)):
        raise RuntimeError("semantic audit case coverage is incomplete or duplicated")
    model = str(report.get("model") or "").strip()
    if not model:
        raise RuntimeError("semantic audit report has no model identity")
    return report


def apply_semantic_audit(*, proposal: list[dict[str, Any]], proposal_path: Path, report_path: Path) -> list[dict[str, Any]]:
    """Publish a model-attested copy; never edits the proposal in place."""
    report = _load_semantic_audit(report_path, proposal, proposal_path)
    report_sha = _sha_file(report_path)
    reviewer = f"{report['model']} (independent semantic audit)"
    reviewed_at = str(report.get("created_at") or "")
    if not reviewed_at:
        raise RuntimeError("semantic audit report has no created_at")
    reviewed: list[dict[str, Any]] = []
    for source in proposal:
        row = json.loads(json.dumps(source))
        row["curation_status"] = "audited"
        row["language_reviewed"] = True
        row["qrels_reviewed"] = True
        row["audit"] = {
            "status": "approved",
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "attestation_type": "independent_model",
            "report_sha256": report_sha,
            "notes": "All cases accepted by the independent Sol semantic audit; no human edits or qrel-derived rewrites were applied.",
        }
        reviewed.append(row)
    return reviewed


def _manual_text_locator(
    connection: sqlite3.Connection,
    *,
    current: dict[str, Any],
    chunk_index: int,
) -> dict[str, Any]:
    """Replace a reviewed text qrel by portable source content, never UUID."""

    document = current.get("document") or {}
    filename = str(document.get("filename") or "")
    content_hash = str(document.get("content_sha256") or "")
    row = connection.execute(
        """
        SELECT c.page_number, c.content, c.metadata_json
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.filename = ? AND d.content_hash = ? AND c.chunk_type = 'text' AND c.chunk_index = ?
        """,
        (filename, content_hash, int(chunk_index)),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"manual audit locator does not resolve: {filename!r} chunk_index={chunk_index}"
        )
    quote = str(row["content"] or "")[:320].strip()
    return {
        "group_id": current.get("group_id"),
        "kind": "text",
        "document": document,
        "page_number": row["page_number"],
        "section": json.loads(row["metadata_json"] or "{}").get("section_title"),
        "quote": quote,
        "quote_sha256": _sha_text(quote),
    }


def apply_manual_audit(
    connection: sqlite3.Connection,
    *,
    proposal: list[dict[str, Any]],
    proposal_path: Path,
    audit_path: Path,
) -> list[dict[str, Any]]:
    """Apply a human attestation and its explicit qrel corrections.

    The audit manifest is immutable with respect to a proposal checksum and
    ordered case IDs.  It may point to a current parser chunk only to rebuild a
    *portable* quote/page/section locator in the reviewed suite; runtime UUIDs
    never leave this function.
    """

    audit = _load_manual_audit(audit_path, proposal, proposal_path)
    overrides = dict(audit.get("text_locator_overrides") or {})
    reviewed: list[dict[str, Any]] = []
    for source in proposal:
        row = json.loads(json.dumps(source))
        case_id = str(row["case_id"])
        override = overrides.get(case_id)
        if override is not None:
            if not isinstance(override, dict):
                raise RuntimeError(f"manual audit override must be an object: {case_id}")
            replacements = override.get("groups") or {}
            if not isinstance(replacements, dict):
                raise RuntimeError(f"manual audit groups must be an object: {case_id}")
            groups = list(row.get("evidence_groups") or [])
            for index, group in enumerate(groups):
                group_id = str(group.get("group_id") or "")
                replacement = replacements.get(group_id)
                if replacement is None:
                    continue
                if not isinstance(replacement, dict) or "chunk_index" not in replacement:
                    raise RuntimeError(f"manual audit replacement must name chunk_index: {case_id}/{group_id}")
                groups[index] = _manual_text_locator(
                    connection,
                    current=group,
                    chunk_index=int(replacement["chunk_index"]),
                )
            row["evidence_groups"] = groups
        row["curation_status"] = "audited"
        row["language_reviewed"] = True
        row["qrels_reviewed"] = True
        row["rewrite_fingerprint"] = str(audit.get("rewrite_fingerprint") or "")
        row["audit"] = {
            "status": "approved",
            "reviewer": str(audit["reviewer"]),
            "reviewed_at": str(audit["reviewed_at"]),
            "notes": str((override or {}).get("notes") or audit.get("notes") or "semantic review complete"),
        }
        reviewed.append(row)
    return reviewed


def _document_refs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT id, source_path, filename, content_hash FROM documents ORDER BY filename"
    ).fetchall()
    refs: list[dict[str, Any]] = []
    for row in rows:
        source = Path(str(row["source_path"] or ""))
        ref = {"id": str(row["id"]), "filename": str(row["filename"]), "content_sha256": str(row["content_hash"])}
        if source.is_file():
            ref["source_sha256"] = _sha_file(source)
        refs.append(ref)
    if len(refs) < 21:
        raise RuntimeError(f"need at least 21 indexed documents, found {len(refs)}")
    return refs


def _public_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in ref.items() if key != "id"}


def _chunk_for_role(connection: sqlite3.Connection, document_id: str, role: str, *, skip: set[str] | None = None) -> sqlite3.Row:
    rows = connection.execute(
        "SELECT id, page_number, content, metadata_json, chunk_index FROM chunks WHERE document_id = ? AND chunk_type = 'text' ORDER BY chunk_index ASC",
        (document_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"document has no text chunks: {document_id}")
    preferred = {
        "task": (
            ("abstract", 8), ("overview", 7), ("purpose", 7),
            ("why this addendum", 7), ("goal", 6), ("introduction", 6),
            ("we present", 6), ("we propose", 6), ("this paper", 5),
            ("this work", 5), ("conclusion", 1),
        ),
        "architecture": (
            ("proposed method", 9), ("proposed model", 9),
            ("proposed framework", 9), ("architecture", 8), ("pipeline", 7),
            ("fusion", 7), ("framework", 6), ("approach", 6),
            ("method", 5), ("design", 5), ("model", 3),
        ),
    }[role]
    excluded = skip or set()

    def score(row: sqlite3.Row) -> tuple[int, int, int, int]:
        raw_section = str(json.loads(row["metadata_json"] or "{}").get("section_title") or "")
        section = _norm(raw_section)
        # PDF extraction may split an all-caps heading as `M ETHODOLOGY` or
        # `C ONCLUSION`. Compact only this explicit OCR artifact; never merge
        # ordinary title-cased words such as `A Model`.
        compact_section = _norm(re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", raw_section))
        content = _norm(str(row["content"] or ""))
        section_score = max(
            (weight for term, weight in preferred if term in section or term in compact_section),
            default=0,
        )
        # Documents without well-formed headings (HTML exports, notes) still
        # need a substantive source span.  This is content-quality selection,
        # not answer-token overlap: audit remains mandatory before publication.
        content_score = sum(weight for term, weight in preferred if term in content[:1200])
        # Extraction can emit table cells as ordinary text chunks.  Treat a
        # metric-heavy pseudo-heading as a poor narrative source for a task or
        # design qrel; it remains available for an explicit table qrel.
        table_markers = (" f1 ", " uar ", " wa ", " ua ", " ccc ", " mae ", " accuracy ", " table ")
        table_marker_count = sum(marker in f" {section} {content[:600]} " for marker in table_markers)
        table_penalty = 18 if table_marker_count >= 3 else 0
        alphabetic = sum(char.isalpha() for char in content)
        semantic_score = (section_score * 3) + content_score - table_penalty
        return (semantic_score, section_score, min(alphabetic, 600), -int(row["chunk_index"] or 0))

    candidates = [row for row in rows if str(row["id"]) not in excluded] or rows
    return max(candidates, key=score)


def _text_locator(ref: dict[str, Any], row: sqlite3.Row, group_id: str) -> dict[str, Any]:
    quote = str(row["content"] or "")[:320].strip()
    return {
        "group_id": group_id,
        "kind": "text",
        "document": _public_ref(ref),
        "page_number": row["page_number"],
        "section": json.loads(row["metadata_json"] or "{}").get("section_title"),
        "quote": quote,
        "quote_sha256": _sha_text(quote),
    }


def _artifact_rows(connection: sqlite3.Connection, *, kind: str, limit: int) -> list[sqlite3.Row]:
    table = "document_tables" if kind == "table" else "document_figures"
    index_column = "table_index" if kind == "table" else "figure_index"
    rows = connection.execute(
        f"SELECT id, document_id, page_number, {index_column} AS artifact_index, caption, metadata_json FROM {table} ORDER BY document_id, artifact_index"
    ).fetchall()
    by_document: dict[str, deque[sqlite3.Row]] = defaultdict(deque)
    for row in rows:
        caption = str(row["caption"] or "")
        # A runtime artifact index is not a paper-facing Figure/Table number.
        # Only evaluate a numbered-artifact query when its extracted caption
        # explicitly supplies a matching human-facing label.  This excludes
        # logos, visual fallbacks, and captionless HTML fragments without a
        # per-document exception list.
        if _caption_artifact_label(caption, kind) is None:
            continue
        if kind == "figure":
            quality = json.loads(row["metadata_json"] or "{}").get("quality_status")
            if quality not in {None, "accepted"}:
                continue
        by_document[str(row["document_id"])].append(row)
    selected: list[sqlite3.Row] = []
    while by_document and len(selected) < limit:
        for document_id in list(sorted(by_document)):
            if by_document[document_id]:
                selected.append(by_document[document_id].popleft())
                if len(selected) == limit:
                    break
            if not by_document[document_id]:
                del by_document[document_id]
    if len(selected) != limit:
        raise RuntimeError(f"need {limit} {kind} artifacts, found {len(selected)}")
    return selected


def _caption_artifact_label(caption: str, kind: str) -> str | None:
    pattern = r"\b(?:table)\s*([0-9]+|[ivxlcdm]+)\b" if kind == "table" else r"\b(?:figure|fig\.?)\s*([0-9]+|[ivxlcdm]+)\b"
    match = re.search(pattern, str(caption or ""), flags=re.IGNORECASE)
    if match is None:
        return None
    return f"{'Table' if kind == 'table' else 'Figure'} {match.group(1)}"


def _base(
    *,
    case_id: str,
    category: str,
    language: str,
    raw_query: str,
    full: str,
    anchor: str,
    entities: list[str],
    context: list[dict[str, str]],
    state: dict[str, Any],
    scope: str,
    focus: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    abstain: bool = False,
) -> dict[str, Any]:
    sources: list[dict[str, str]] = [{"source": "raw_query", "text": raw_query}]
    sources.extend({"source": "conversation", "text": item["content"]} for item in context)
    if state:
        sources.append({"source": "working_state", "text": json.dumps(state, ensure_ascii=False)})
    sources.extend({"source": "catalog", "text": item} for item in entities)
    return {
        "case_id": case_id,
        "category": category,
        "language": language,
        "raw_query": raw_query,
        "enriched_query": full,
        "standalone_query": full,
        "standalone_full": full,
        "standalone_anchor": anchor,
        "required_entities": entities,
        "conversation_context": context,
        "working_state": state,
        "rewrite_provenance": {"full": sources, "anchor": sources},
        "scope": scope,
        "focus_documents": focus,
        "evidence_groups": groups,
        "forbidden_documents": [],
        "expected_abstention": abstain,
        "rewrite_fingerprint": PROPOSAL_REVISION,
        "curation_status": "review_required",
        "language_reviewed": False,
        "qrels_reviewed": False,
        "audit": {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "",
        },
    }


def _artifact_cases(connection: sqlite3.Connection, refs: dict[str, dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    label = "Table" if kind == "table" else "Figure"
    category = "table_numeric" if kind == "table" else "figure_page"
    rows: list[dict[str, Any]] = []
    for index, artifact_row in enumerate(_artifact_rows(connection, kind=kind, limit=10), start=1):
        ref = refs[str(artifact_row["document_id"])]
        language, contextual = ("vi", True) if index % 2 else ("en", False)
        artifact = _caption_artifact_label(str(artifact_row["caption"] or ""), kind)
        if artifact is None:  # Defensive: _artifact_rows already enforces this.
            raise RuntimeError(f"selected {kind} artifact has no paper-facing label")
        if language == "vi":
            direct = f"Trong {ref['filename']}, {artifact} báo cáo hoặc minh hoạ điều gì?"
            followup = "Bảng đó nói gì?" if kind == "table" else "Hình đó minh hoạ điều gì?"
        else:
            direct = f"In {ref['filename']}, what does {artifact} report or illustrate?"
            followup = "What does that table report?" if kind == "table" else "What does that figure illustrate?"
        raw = followup if contextual else direct
        context = ([{"role": "user", "content": f"I am reading {ref['filename']}."}, {"role": "assistant", "content": f"The current artifact is {artifact}."}] if contextual else [])
        state = {"active_document": ref["filename"], "active_artifact": artifact} if contextual else {}
        locator = {
            "group_id": "artifact-1",
            "kind": kind,
            "document": _public_ref(ref),
            "page_number": artifact_row["page_number"],
            "artifact_index": artifact_row["artifact_index"],
            "caption": str(artifact_row["caption"] or ""),
            "caption_sha256": _sha_text(str(artifact_row["caption"] or "")),
        }
        rows.append(_base(case_id=f"dev-v2.1-{kind}-{index:02d}", category=category, language=language, raw_query=raw, full=f"{ref['filename']} {artifact} {raw}", anchor=f"{ref['filename']} {artifact}", entities=[ref["filename"]], context=context, state=state, scope="focused", focus=[_public_ref(ref)], groups=[locator]))
    return rows


def _text_cases(connection: sqlite3.Connection, refs: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    if category == "text":
        # Include one one-chunk document in the one-span category.  A
        # single-chunk document cannot honestly support an intra-document
        # multi-hop qrel, but must still have two independent qrels overall
        # (this one plus its cross-document comparison).  This is based on
        # document structure, never a filename exception.
        singleton_refs = [ref for ref in refs if _text_chunk_count(connection, ref["id"]) == 1]
        selected_refs = [*refs[:9], *singleton_refs[:1]]
        if len(selected_refs) != 10:
            raise RuntimeError("need nine multi-chunk documents plus one singleton text document")
    else:
        text_document_ids = {
            ref["id"]
            for ref in [*refs[:9], *[item for item in refs if _text_chunk_count(connection, item["id"]) == 1][:1]]
        }
        eligible = [
            ref
            for ref in refs
            if ref["id"] not in text_document_ids and _text_chunk_count(connection, ref["id"]) >= 2
        ]
        # The remaining document that does not become an intra-document case
        # still needs two independent qrels.  Select the only omission from
        # a document that has a usable numbered artifact, so its artifact and
        # cross-document qrels provide that coverage.  This derives solely
        # from indexed structure; it is not a paper-specific mapping.
        omissions = [ref for ref in eligible if _numbered_artifact_count(connection, ref["id"]) > 0]
        if not omissions:
            raise RuntimeError("cannot preserve document coverage: no artifact-backed intra-document omission")
        omitted = max(omissions, key=lambda ref: (_numbered_artifact_count(connection, ref["id"]), ref["filename"]))
        selected_refs = [ref for ref in eligible if ref["id"] != omitted["id"]][:10]
        if len(selected_refs) < 10:
            raise RuntimeError("need ten documents with independent task/design spans")
    rows: list[dict[str, Any]] = []
    for index, ref in enumerate(selected_refs, start=1):
        language, contextual = ("vi", True) if index % 2 else ("en", False)
        roles = ("task",) if category == "text" else ("task", "architecture")
        used: set[str] = set()
        groups: list[dict[str, Any]] = []
        for group_index, role in enumerate(roles, start=1):
            chunk = _chunk_for_role(connection, ref["id"], role, skip=used)
            used.add(str(chunk["id"]))
            groups.append(_text_locator(ref, chunk, f"{role}-{group_index}"))
        if language == "vi":
            direct = f"{ref['filename']} nói về điều gì?" if category == "text" else f"{ref['filename']} có mục đích gì và được tổ chức hoặc thiết kế ra sao?"
            followup = "Tài liệu đó nói về gì?" if category == "text" else "Mục đích và thiết kế của tài liệu đó ra sao?"
            intent = "mục đích" if category == "text" else "mục đích thiết kế"
        else:
            direct = f"What is {ref['filename']} about?" if category == "text" else f"What is the purpose and main design of {ref['filename']}?"
            followup = "What is that document about?" if category == "text" else "What are that document's purpose and main design?"
            intent = "purpose" if category == "text" else "purpose design"
        raw = followup if contextual else direct
        context = [{"role": "user", "content": f"I am reading {ref['filename']}."}] if contextual else []
        state = {"active_document": ref["filename"]} if contextual else {}
        rows.append(_base(case_id=f"dev-v2.1-{category}-{index:02d}", category=category, language=language, raw_query=raw, full=f"{ref['filename']} {raw}", anchor=f"{ref['filename']} {intent}", entities=[ref["filename"]], context=context, state=state, scope="focused" if contextual else "unscoped", focus=[_public_ref(ref)] if contextual else [], groups=groups))
    return rows


def _text_chunk_count(connection: sqlite3.Connection, document_id: str) -> int:
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ? AND chunk_type = 'text'",
            (document_id,),
        ).fetchone()[0]
    )


def _numbered_artifact_count(connection: sqlite3.Connection, document_id: str) -> int:
    count = 0
    for kind, table, index_column in (
        ("table", "document_tables", "table_index"),
        ("figure", "document_figures", "figure_index"),
    ):
        rows = connection.execute(
            f"SELECT caption, metadata_json FROM {table} WHERE document_id = ? ORDER BY {index_column}",
            (document_id,),
        ).fetchall()
        for row in rows:
            if _caption_artifact_label(str(row["caption"] or ""), kind) is None:
                continue
            if kind == "figure":
                quality = json.loads(row["metadata_json"] or "{}").get("quality_status")
                if quality not in {None, "accepted"}:
                    continue
            count += 1
    return count


def _comparison_cases(connection: sqlite3.Connection, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(10):
        left_index, right_index = ((19, 20) if index == 9 else (index * 2, index * 2 + 1))
        left, right = refs[left_index % len(refs)], refs[right_index % len(refs)]
        language, contextual = ("vi", True) if (index + 1) % 2 else ("en", False)
        if language == "vi":
            direct, followup, intent = f"So sánh mục đích của {left['filename']} và {right['filename']}.", "So sánh mục đích của hai tài liệu đó.", "mục đích"
        else:
            direct, followup, intent = f"Compare the purpose of {left['filename']} and {right['filename']}.", "Compare the purpose of those two documents.", "purpose"
        raw = followup if contextual else direct
        context = ([{"role": "user", "content": f"I am comparing {left['filename']} with {right['filename']}."}] if contextual else [])
        state = {"active_documents": [left["filename"], right["filename"]]} if contextual else {}
        rows.append(_base(case_id=f"dev-v2.1-cross-document-compare-{index + 1:02d}", category="cross_document_compare", language=language, raw_query=raw, full=f"{left['filename']} {right['filename']} {raw}", anchor=f"{left['filename']} {right['filename']} {intent}", entities=[left["filename"], right["filename"]], context=context, state=state, scope="focused", focus=[_public_ref(left), _public_ref(right)], groups=[_text_locator(left, _chunk_for_role(connection, left["id"], "task"), "left-task"), _text_locator(right, _chunk_for_role(connection, right["id"], "task"), "right-task")]))
    return rows


def _negative_cases() -> list[dict[str, Any]]:
    prompts = [("vi", "Có paper nào dự báo thời tiết trên Sao Hoả không?"), ("en", "Does the library contain a stock-price table for 2035?"), ("vi", "Bài nào đo tốc độ tàu vũ trụ bằng dữ liệu trong thư viện?"), ("en", "Which paper explains alien DNA in a figure?"), ("vi", "Theo các tài liệu này ai thắng Oscar năm 2040?"), ("en", "Which document studies cooking recipes from satellite images?"), ("vi", "Có bảng nào về tỷ giá Bitcoin năm 1990 không?"), ("en", "Which paper predicts the next lunar eclipse from this corpus?"), ("vi", "Có tài liệu nào nói về luật giao thông trên Sao Kim không?"), ("en", "Does any paper benchmark interstellar crop yields?")]
    rows: list[dict[str, Any]] = []
    for index, (language, query) in enumerate(prompts, start=1):
        contextual = index % 2 == 1
        context = [{"role": "user", "content": "Please keep searching only within the local paper library."}] if contextual else []
        rows.append(_base(case_id=f"dev-v2.1-negative-{index:02d}", category="negative_unanswerable", language=language, raw_query=query, full=query, anchor=query, entities=[], context=context, state={"scope": "local_library"} if contextual else {}, scope="unscoped", focus=[], groups=[], abstain=True))
    return rows


def build_proposal(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    all_refs = _document_refs(connection)
    refs = {str(ref["id"]): ref for ref in all_refs}
    rows = [*_text_cases(connection, all_refs, "text"), *_artifact_cases(connection, refs, "table"), *_artifact_cases(connection, refs, "figure"), *_text_cases(connection, all_refs, "intra_document_multihop"), *_comparison_cases(connection, all_refs), *_negative_cases()]
    if len(rows) != 60:
        raise RuntimeError(f"proposal must contain exactly 60 cases, found {len(rows)}")
    return rows


def validate_audited_suite(path: Path, db_path: Path) -> dict[str, Any]:
    from evaluate_dual_query_retrieval import _validate_suite_shape, load_cases, resolve_case

    cases = load_cases(path, require_audited=True)
    _validate_suite_shape(cases, expected=60)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        resolved = [resolve_case(connection, case) for case in cases]
    finally:
        connection.close()
    return {"suite": str(path), "cases": len(cases), "qrels_resolved": len(resolved), "status": "valid"}


def preflight_proposal(path: Path, db_path: Path) -> dict[str, Any]:
    """Resolve a structural proposal without treating it as audited evidence."""

    from evaluate_dual_query_retrieval import _validate_suite_shape, load_cases, resolve_case

    cases = load_cases(path, require_audited=False)
    _validate_suite_shape(cases, expected=60)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        resolved = [resolve_case(connection, case) for case in cases]
    finally:
        connection.close()
    return {
        "proposal": str(path),
        "cases": len(cases),
        "qrels_resolved": len(resolved),
        "status": "preflight_only_review_required",
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("propose", "review-packet", "preflight", "apply-manual-audit", "apply-semantic-audit", "validate", "publish-reviewed"))
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--proposal-output", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.proposed.jsonl")
    parser.add_argument("--review-output", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.review-packet.jsonl")
    parser.add_argument("--audit-manifest", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.manual-audit.json")
    parser.add_argument("--semantic-report", type=Path, default=None)
    parser.add_argument("--reviewed-output", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.reviewed.jsonl")
    parser.add_argument("--reviewed-input", type=Path, default=None)
    parser.add_argument("--suite-output", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.jsonl")
    args = parser.parse_args()
    db_path = args.db_path or get_settings().sqlite_db_path
    if args.command == "propose":
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            proposal = build_proposal(connection)
        finally:
            connection.close()
        _write_jsonl(args.proposal_output, proposal)
        print(json.dumps({"proposal": str(args.proposal_output), "cases": len(proposal), "status": "review_required"}, ensure_ascii=False))
        return 0
    if args.command == "review-packet":
        proposal = _read_jsonl(args.proposal_output)
        _write_jsonl(args.review_output, (_review_packet_row(row) for row in proposal))
        print(json.dumps({"proposal": str(args.proposal_output), "review_packet": str(args.review_output), "cases": len(proposal), "status": "pending_human_audit"}, ensure_ascii=False))
        return 0
    if args.command == "preflight":
        print(json.dumps(preflight_proposal(args.proposal_output, db_path), ensure_ascii=False, indent=2))
        return 0
    if args.command == "apply-manual-audit":
        proposal = _read_jsonl(args.proposal_output)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            reviewed = apply_manual_audit(
                connection,
                proposal=proposal,
                proposal_path=args.proposal_output,
                audit_path=args.audit_manifest,
            )
        finally:
            connection.close()
        _write_jsonl(args.reviewed_output, reviewed)
        print(json.dumps({"reviewed": str(args.reviewed_output), "cases": len(reviewed), "status": "audited_pending_publish"}, ensure_ascii=False))
        return 0
    if args.command == "apply-semantic-audit":
        if args.semantic_report is None:
            parser.error("--semantic-report is required for apply-semantic-audit")
        proposal = _read_jsonl(args.proposal_output)
        reviewed = apply_semantic_audit(
            proposal=proposal,
            proposal_path=args.proposal_output,
            report_path=args.semantic_report,
        )
        _write_jsonl(args.reviewed_output, reviewed)
        print(json.dumps({"reviewed": str(args.reviewed_output), "cases": len(reviewed), "status": "model_attested_pending_publish"}, ensure_ascii=False))
        return 0
    source = args.reviewed_input or args.suite_output
    result = validate_audited_suite(source, db_path)
    if args.command == "publish-reviewed":
        if args.reviewed_input is None:
            parser.error("--reviewed-input is required for publish-reviewed")
        if args.reviewed_input.resolve() == args.proposal_output.resolve():
            parser.error("--reviewed-input must be a separately audited copy, never the generated proposal")
        args.suite_output.parent.mkdir(parents=True, exist_ok=True)
        args.suite_output.write_text(args.reviewed_input.read_text(encoding="utf-8"), encoding="utf-8")
        result["published_to"] = str(args.suite_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
