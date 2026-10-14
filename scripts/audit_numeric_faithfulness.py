#!/usr/bin/env python3
"""Independent numeric-faithfulness audit of saved Aya answers.

For every answer in a report (Codex conversational reports with
``--include-answers`` or ``evaluate_aya_acceptance.py`` reports), collect the
numbers it prints, pull the passages and tables of the library that contain
those numbers (plain SQLite lookup, active revisions only), and ask a judge
model that did not write the answer (default ``cx/gpt-5.6-terra``) to list
numeric claims that the evidence does not support or attributes differently.

Values derived by arithmetic from the evidence (totals, differences) count as
supported. Answers without numbers are skipped.

Usage:
  backend/.venv/bin/python scripts/audit_numeric_faithfulness.py \
      --report data/retrieval_eval/internal/codex-conversational-dev-v1-20261014.json \
      --db-path data/retrieval_eval/internal/runtime/e2e-foundation-20261005/sqlite/app.db
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")
_PROMPT = """You audit the numbers in an assistant's answer about research papers.

Evidence (passages and tables from the papers that contain these numbers):
{evidence}

Answer:
<<<
{answer}
>>>

List every numeric claim in the answer that the evidence does not support: a value
that does not appear, or appears for a different model, dataset, metric, split or
condition than the answer says. Values correctly computed from evidence values
(sums, differences, averages, rounding) are supported. Table/figure/section numbers,
years and citation numbers are not claims.

Return JSON only: {{"unsupported": [{{"claim": "<short quote>", "why": "<reason>"}}]}}"""


def _numbers(answer: str) -> list[str]:
    values = []
    for raw in _NUMBER.findall(answer):
        value = raw.rstrip("%")
        digits = re.sub(r"\D", "", value)
        if len(digits) < 2 or re.fullmatch(r"(19|20)\d\d", value):
            continue
        values.append(value)
    return list(dict.fromkeys(values))


def _evidence(
    connection: sqlite3.Connection, values: list[str], document_ids: list[str], budget: int = 24_000,
) -> str:
    """Tables and number-bearing passages of the documents the turn used.

    Without document ids (older reports) the lookup falls back to the whole
    library, which is weaker: it can miss values printed differently.
    """
    blocks: list[str] = []
    used = 0

    def add(block: str) -> bool:
        nonlocal used
        if used + len(block) > budget:
            return False
        blocks.append(block)
        used += len(block)
        return True

    scope = ""
    params: list[Any] = []
    if document_ids:
        scope = f" AND d.id IN ({','.join('?' for _ in document_ids)})"
        params = list(document_ids)
        for filename, caption, markdown in connection.execute(
            "SELECT d.filename, t.caption, t.markdown FROM document_tables t JOIN documents d ON d.id = t.document_id "
            "WHERE COALESCE(t.revision_id, '') = COALESCE(d.active_revision_id, '')" + scope,
            params,
        ).fetchall():
            if not add(f"[{filename} · table] {caption or ''}\n{str(markdown)[:3_000]}"):
                return "\n\n".join(blocks)
        for filename, caption, summary in connection.execute(
            "SELECT d.filename, f.caption, f.visual_summary FROM document_figures f JOIN documents d ON d.id = f.document_id "
            "WHERE f.visual_summary IS NOT NULL AND COALESCE(f.revision_id, '') = COALESCE(d.active_revision_id, '')" + scope,
            params,
        ).fetchall():
            if not any(value in str(summary) for value in values):
                continue
            if not add(f"[{filename} · figure description] {caption or ''}\n{str(summary)[:2_000]}"):
                return "\n\n".join(blocks)
    seen: set[str] = set()
    for value in values:
        for variant in {value, value.replace(",", ""), value.replace(",", ".")}:
            rows = connection.execute(
                "SELECT d.filename, c.content, c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.content LIKE ? AND c.revision_id = d.active_revision_id" + scope + " LIMIT 4",
                [f"%{variant}%", *params],
            ).fetchall()
            for filename, content, key in rows:
                if key in seen:
                    continue
                seen.add(key)
                if not add(f"[{filename} · text]\n{str(content)[:2_000]}"):
                    return "\n\n".join(blocks)
    return "\n\n".join(blocks)


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("results") or report.get("rows") or []
    out = []
    for row in rows:
        answer = str(row.get("answer") or "")
        ident = row.get("case_id") or f"{row.get('conversation')}-t{row.get('turn')}"
        document_ids = list(dict.fromkeys(
            [*(row.get("retrieved_document_ids") or []), *(row.get("focus_document_ids") or [])]
        ))
        out.append({"id": ident, "answer": answer, "document_ids": document_ids})
    return out


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    client = get_llm_client(
        provider="openai_compatible", ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base, openai_api_key=settings.openai_api_key,
        timeout_seconds=180,
    )
    connection = sqlite3.connect(args.db_path)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    gate = asyncio.Semaphore(args.concurrency)

    async def audit(row: dict[str, Any]) -> dict[str, Any]:
        values = _numbers(row["answer"])
        if not values:
            return {"id": row["id"], "status": "no_numbers", "unsupported": []}
        evidence = _evidence(connection, values, row.get("document_ids") or [])
        prompt = _PROMPT.format(evidence=evidence or "(no passage contains these numbers)", answer=row["answer"][:10_000])
        for attempt in range(3):
            try:
                async with gate:
                    completion = await client.chat(
                        model=args.model, temperature=0.0, num_predict=1_200,
                        messages=[{"role": "user", "content": prompt}],
                    )
                text = getattr(completion, "message", "") or ""
                match = re.search(r"\{.*\}", text, re.DOTALL)
                data = json.loads(match.group(0)) if match else {}
                unsupported = [item for item in data.get("unsupported") or [] if isinstance(item, dict)]
                return {"id": row["id"], "status": "audited", "numbers": len(values), "unsupported": unsupported}
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    return {"id": row["id"], "status": "error", "error": str(exc)[:200], "unsupported": []}
                await asyncio.sleep(20 * (attempt + 1))
        return {"id": row["id"], "status": "error", "unsupported": []}

    results = await asyncio.gather(*[audit(row) for row in _rows(report)])
    audited = [item for item in results if item["status"] == "audited"]
    flagged = [item for item in audited if item["unsupported"]]
    for item in flagged:
        print(f"[{item['id']}] " + "; ".join(f"{u.get('claim')!r}: {u.get('why')}" for u in item["unsupported"])[:600])
    print({"answers": len(results), "with_numbers": len(audited), "clean": len(audited) - len(flagged),
           "flagged": len(flagged), "errors": sum(1 for item in results if item["status"] == "error")})
    output = args.output or args.report.with_name(args.report.stem + "-numeric-audit.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--model", default="cx/gpt-5.6-terra")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
