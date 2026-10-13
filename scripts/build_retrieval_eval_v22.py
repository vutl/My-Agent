#!/usr/bin/env python3
"""Create a more diverse, explicitly cross-balanced Gate-C proposal.

This is a proposal builder only.  It reuses portable qrels from the audited
v2.1 source, but rewrites the *questions and conversational conditions* so the
experiment is not perfectly confounded by language and contextuality.  The
result remains review-required and is never promoted automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

FINGERPRINT = "dev-v2.2-proposal-r1"
MATRIX_A = (
    ("vi", True),
    ("vi", True),
    ("vi", False),
    ("vi", False),
    ("vi", False),
    ("en", True),
    ("en", True),
    ("en", True),
    ("en", False),
    ("en", False),
)
MATRIX_B = (
    ("vi", True),
    ("vi", True),
    ("vi", True),
    ("vi", False),
    ("vi", False),
    ("en", True),
    ("en", True),
    ("en", False),
    ("en", False),
    ("en", False),
)


def _normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _document_names(row: dict[str, Any]) -> list[str]:
    names = [str(value) for value in row.get("required_entities") or [] if str(value).strip()]
    if names:
        return names
    return [
        str((group.get("document") or {}).get("filename") or "")
        for group in row.get("evidence_groups") or []
        if str((group.get("document") or {}).get("filename") or "").strip()
    ]


def _artifact_label(row: dict[str, Any]) -> str:
    text = str(row.get("standalone_anchor") or row.get("standalone_full") or "")
    match = re.search(r"\b((?:table|figure|fig\.?)\s*[0-9ivxlcdm]+)\b", text, re.I)
    return match.group(1).replace("fig.", "Figure").replace("fig", "Figure") if match else "artifact"


def _focus_documents(row: dict[str, Any], names: list[str], *, contextual: bool) -> list[dict[str, str]]:
    """Materialize catalog scope from the conversationally named documents.

    The source v2.1 proposal sometimes left this field empty even when the
    working state named an active paper.  That made the evaluator silently
    widen a focused turn to the whole corpus.  Only filename/hash identity is
    copied here; no qrel quote, caption or answer text enters the rewrite.
    """

    if not contextual or not names:
        return []
    by_filename = {
        str((group.get("document") or {}).get("filename") or ""): group.get("document") or {}
        for group in row.get("evidence_groups") or []
        if str((group.get("document") or {}).get("filename") or "").strip()
    }
    result: list[dict[str, str]] = []
    for name in names:
        document = by_filename.get(name)
        if not document:
            continue
        result.append(
            {
                "filename": name,
                "content_sha256": str(document.get("content_sha256") or ""),
                "source_sha256": str(document.get("source_sha256") or ""),
            }
        )
    return result


def _context(language: str, names: list[str], *, artifact: str | None = None) -> list[dict[str, str]]:
    if len(names) == 1:
        if language == "vi":
            content = f"Mình đang đọc {names[0]}."
        else:
            content = f"I am reading {names[0]}."
        if artifact:
            content += f" {artifact} is the current artifact."
    else:
        if language == "vi":
            content = f"Mình đang đối chiếu {names[0]} với {names[1]}."
        else:
            content = f"I am comparing {names[0]} with {names[1]}."
    return [{"role": "user", "content": content}]


def _state(names: list[str], *, artifact: str | None = None) -> dict[str, Any]:
    if len(names) == 1:
        value: dict[str, Any] = {"active_document": names[0]}
        if artifact:
            value["active_artifact"] = artifact
        return value
    return {"active_documents": names}


def _question(category: str, language: str, names: list[str], index: int, contextual: bool, artifact: str | None) -> str:
    vi = language == "vi"
    if category == "text":
        if vi:
            return (
                ("Bài này tập trung vào vấn đề nào?", "Nghiên cứu này muốn giải quyết bài toán gì?", "Đóng góp chính của bài là gì?")[index % 3]
                if contextual
                else f"{names[0]} nghiên cứu chủ đề nào?"
            )
        return (
            ("What problem does this paper study?", "What question is this research trying to solve?", "What is the paper's main contribution?")[index % 3]
            if contextual
            else f"What topic does {names[0]} study?"
        )
    if category == "intra_document_multihop":
        if vi:
            return (
                ("Mục tiêu và kiến trúc chính của bài này là gì?", "Bài này giải quyết bài toán gì và mô hình được thiết kế ra sao?", "Nghiên cứu này đề xuất gì và cấu trúc của nó thế nào?")[index % 3]
                if contextual
                else f"{names[0]} có mục tiêu gì và kiến trúc chính ra sao?"
            )
        return (
            ("What are this paper's goal and main architecture?", "What problem does it solve and how is the model designed?", "What does the study propose and what structure does it use?")[index % 3]
            if contextual
            else f"What are the goal and main architecture of {names[0]}?"
        )
    if category == "cross_document_compare":
        if vi:
            return (
                ("Hai bài này khác nhau thế nào về mục tiêu?", "Hai tài liệu này cùng giải quyết bài toán gì và khác nhau ở đâu?", "So sánh trọng tâm nghiên cứu của hai bài này.")[index % 3]
                if contextual
                else f"So sánh mục tiêu của {names[0]} và {names[1]}."
            )
        return (
            ("How do these two papers differ in purpose?", "What problem does each document address, and how are they different?", "Compare the research focus of these two papers.")[index % 3]
            if contextual
            else f"Compare the purpose of {names[0]} and {names[1]}."
        )
    if category == "table_numeric":
        label = artifact or "Table"
        if vi:
            return ((f"{label} này báo cáo thông tin gì?", f"Nội dung chính của {label} này là gì?", f"{label} này trình bày những gì?")[index % 3] if contextual else f"Trong {names[0]}, {label} báo cáo thông tin gì?")
        return ((f"What information does this {label} report?", f"What is the main content of this {label}?", f"What does this {label} present?")[index % 3] if contextual else f"What information does {label} report in {names[0]}?")
    if category == "figure_page":
        label = artifact or "Figure"
        if vi:
            return ((f"{label} này minh họa thành phần nào?", f"Quy trình trong {label} này diễn ra ra sao?", f"Thông tin chính của {label} này là gì?")[index % 3] if contextual else f"{label} trong {names[0]} minh họa điều gì?")
        return ((f"What component does this {label} illustrate?", f"How does the process in this {label} work?", f"What is the main information in this {label}?")[index % 3] if contextual else f"What does the {label} in {names[0]} illustrate?")
    return ""


def _provenance(raw: str, context: list[dict[str, str]], state: dict[str, Any], names: list[str]) -> dict[str, list[dict[str, str]]]:
    entries: list[dict[str, str]] = [{"source": "raw_query", "text": raw}]
    entries.extend({"source": "conversation", "text": item["content"]} for item in context)
    if state:
        entries.append({"source": "working_state", "text": json.dumps(state, ensure_ascii=False)})
    entries.extend({"source": "catalog", "text": name} for name in names)
    return {"full": entries, "anchor": entries}


def _transform(row: dict[str, Any], category_index: int, index: int) -> dict[str, Any]:
    names = _document_names(row)
    if not names and row.get("category") != "negative_unanswerable":
        raise ValueError(f"{row.get('case_id')}: no document names")
    matrix = MATRIX_A if category_index % 2 == 0 else MATRIX_B
    language, contextual = matrix[index]
    category = str(row.get("category") or "")
    artifact = _artifact_label(row) if category in {"table_numeric", "figure_page"} else None
    raw = _question(category, language, names, index, contextual, artifact)
    if category == "negative_unanswerable":
        raw = (
            ("Có tài liệu nào trong thư viện nói về dự báo thời tiết trên Sao Hoả không?", "Bài nào đo tốc độ tàu vũ trụ trong thư viện?", "Có bảng nào về tỷ giá Bitcoin năm 1990 không?")[index % 3]
            if language == "vi"
            else ("Does the library contain a paper about Martian weather?", "Which paper measures spacecraft speed?", "Does any table report Bitcoin exchange rates from 1990?")[index % 3]
        )
    if contextual and names:
        context = _context(language, names, artifact=artifact)
        state = _state(names, artifact=artifact)
    elif contextual and category == "negative_unanswerable":
        context = [{"role": "user", "content": "Mình chỉ muốn tìm trong thư viện tài liệu nội bộ." if language == "vi" else "Please search only within the local document library."}]
        state = {"scope": "local_library"}
    else:
        context = []
        state = {}
    full = " ".join([*names, raw]).strip() if contextual else raw
    anchor_terms = {
        "text": "chủ đề" if language == "vi" else "topic",
        "intra_document_multihop": "mục tiêu kiến trúc" if language == "vi" else "goal architecture",
        "cross_document_compare": "mục tiêu" if language == "vi" else "purpose",
        "table_numeric": artifact or "Table",
        "figure_page": artifact or "Figure",
        "negative_unanswerable": "",
    }.get(category, "")
    anchor = " ".join([*names, anchor_terms]).strip()
    if category == "negative_unanswerable":
        full = raw
        anchor = raw
    transformed = dict(row)
    transformed.update(
        {
            "case_id": str(row.get("case_id") or "").replace("dev-v2.1-", "dev-v2.2-", 1),
            "language": language,
            "raw_query": raw,
            "enriched_query": full,
            "standalone_query": full,
            "standalone_full": full,
            "standalone_anchor": anchor,
            "conversation_context": context,
            "working_state": state,
            "rewrite_provenance": _provenance(raw, context, state, names),
            "scope": "focused" if contextual and names else "unscoped",
            "focus_documents": _focus_documents(row, names, contextual=contextual),
            "rewrite_fingerprint": FINGERPRINT,
            "curation_status": "review_required",
            "language_reviewed": False,
            "qrels_reviewed": False,
            "audit": {"status": "pending", "reviewer": None, "reviewed_at": None, "notes": "Independent semantic audit required for v2.2."},
        }
    )
    return transformed


def build_proposal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 60:
        raise ValueError(f"expected 60 source cases, found {len(rows)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category") or ""), []).append(row)
    output: list[dict[str, Any]] = []
    for category_index, category in enumerate(sorted(grouped)):
        category_rows = grouped[category]
        if len(category_rows) != 10:
            raise ValueError(f"category {category} has {len(category_rows)} cases, expected 10")
        output.extend(_transform(row, category_index, index) for index, row in enumerate(category_rows))
    return output


def validate_proposal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = Counter((row.get("language"), bool(row.get("conversation_context"))) for row in rows)
    category_matrix: dict[str, Counter] = {}
    duplicate_templates: dict[str, int] = {}
    for category in sorted({str(row.get("category") or "") for row in rows}):
        category_rows = [row for row in rows if row.get("category") == category]
        category_matrix[category] = Counter((row.get("language"), bool(row.get("conversation_context"))) for row in category_rows)
        duplicate_templates[category] = Counter(_normalize_query(str(row.get("raw_query") or "")) for row in category_rows).most_common(1)[0][1]
    if matrix != Counter({("vi", True): 15, ("vi", False): 15, ("en", True): 15, ("en", False): 15}):
        raise ValueError(f"language/context matrix invalid: {matrix}")
    if any(min(values.values()) < 2 for values in category_matrix.values()):
        raise ValueError(f"category matrix has a missing cell: {category_matrix}")
    if any(value > 4 for value in duplicate_templates.values()):
        raise ValueError(f"template dominance remains: {duplicate_templates}")
    return {
        "cases": len(rows),
        "language_context_matrix": {f"{language}:{'contextual' if contextual else 'standalone'}": count for (language, contextual), count in matrix.items()},
        "category_matrix": {
            key: {f"{language}:{'contextual' if contextual else 'standalone'}": count for (language, contextual), count in value.items()}
            for key, value in category_matrix.items()
        },
        "max_duplicate_raw_by_category": duplicate_templates,
        "status": "review_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.1.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing proposal: {args.output}; use --force explicitly")
    proposal = build_proposal(_read_jsonl(args.input))
    diagnostics = validate_proposal(proposal)
    _write_jsonl(args.output, proposal)
    print(json.dumps({"proposal": str(args.output), **diagnostics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
