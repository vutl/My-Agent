# 12 — Release record: Aya RAG v1 (RC1)

Status: **release candidate accepted for quality** (2026-10-22). Production
rollout (starting the RC on the real `data/`) waits for the user's go-ahead.

## Candidate

| Item | Value |
|---|---|
| Release SHA | `3219b2bc` (regression on `e7fec5ce`; the only later change moves the title refresh from import to server startup and adds the drill script) |
| Rollback baseline | `7fcc5bd9` (2026-10-19, before the ingest / catalog / migration changes) |
| Answer model | `cx/gpt-6.1-sol` (9router) |
| Router model | `cx/gpt-6-luna` |
| Ingest vision model | `cx/gpt-5.6-sol` (`backend/.env` `VISION_MODEL`; changing it re-indexes the library) |
| Embeddings | Ollama `embeddinggemma:300m` |
| Processor contract | `foundation-v6-scanned-table-ocr` |
| Page OCR cache | `pdf-page-ocr-v3` (Apple Vision on macOS, RapidOCR fallback) |
| Title rules | `titles-v3` (re-titled once at server startup) |
| Backend tests | 1511 pass |

## Quality evidence

Sealed sets: corpus and questions written by an independent agent, unseen by
the head agent and the code before the run, run once, judged by Sonnet from
references; an independent agent re-judged a **sample** (28–29 turns per set)
against the raw documents.

| Set | Correct | Correct+partial | Fabricated | False abstention | Unanswerable declined |
|---|---|---|---|---|---|
| v4 (`1cb91b08`, 20 files, 95 turns) | 81/95 = 85.3% (CI 77–91) | 94.7% | 0 | 4/95 | 11/12 |
| v5 (`fd814dbd`, 20 files, 104 turns) | 90/104 = 86.5% (CI 79–92) | 94.2% | 0 | 4/104 | 12/13 |

Audits: no answer judged correct was wrong in the samples; no unsupported
number found; the judge is strict (2–5 partials per set miss only points the
question did not ask). The sets are dev sets from now on.

Regression on `e7fec5ce`:

| Suite | Result | What it measures |
|---|---|---|
| Acceptance v1 (60 turns) | 60/60 structural; **58/60 correct, 2 partial on content** (Sonnet from must_mention), 0 fabricated | structural = focus document, no fallback, attachment; content = judged answer |
| Vague-reference variants (47 turns) | 46/47 correct, 0 fabricated, 0 false abstention | resolution of "nó / bài kia / the first paper" |

Raw files (copies in git under `evals/`): `evals/acceptance-rc-e7fec5ce/`,
`evals/sealed/v1/variants/`, `evals/sealed/v4`, `evals/sealed/v5` (raw documents stay in `data/retrieval_eval/sealed/`, gitignored).

## Rollback drill

`scripts/release_rollback_drill.sh 3219b2bc 7fcc5bd9 <work>` on APFS clones of
`data/sqlite/app.db`, `data/lancedb`, `data/artifacts` (never the live files).
Evidence: `evals/release/rc1-rollback-drill-3219b2bc.jsonl`.

| Step | Result |
|---|---|
| Backup checksum | `00c91a2fe88806f8` |
| Release on a copy: health, 21 documents, smoke "KST là viết tắt của gì?" → cites KST.pdf | ok |
| Code-only rollback (baseline code on the release-migrated data) | ok |
| Restore backup (checksum identical) + baseline | ok |

Startup migrations of this release are forward-compatible: they only rewrite
titles and add a `title_rules` metadata key, which older code ignores.

Incident found by the drill: before `3219b2bc`, importing `app.main` (every
pytest run) ran the title refresh against the configured library, i.e. the
production DB. On 2026-10-22 04:54 it stamped `title_rules: titles-v3` into
the 21 production documents' metadata; **no title changed** (checked against
`data/sqlite/backups/app-before-search-refresh-20261014.db`). Fixed and
guarded by a test. Note: `create_app()` still runs `init_db` (idempotent
schema migrations) on import, as before this release.

## Rollout (needs user approval)

1. Stop :7777; back up `data/sqlite/app.db`, `data/lancedb`, `data/artifacts`.
2. Start `3219b2bc` on `data/`; check `/health`, the catalog titles and one
   question with citations in the desktop app.
3. Rollback: stop, check out `7fcc5bd9`, restore the backup if needed, start.

## Maintenance backlog (not release blockers)

- Within-document misses on long or scanned documents (v5-08 visitor counts,
  v5-15 slide 16, v5-17 scanned appendix, v5-33 second rule, v4-24/28 class).
- Documents sharing a number or name answered for one only ("Nghị định 85"
  2017 vs 2024; two README files).
- Switched-document label still flips on some runs (vague-ref var-07 t2).
- Acceptance partials: exact term naming (c04 t4), dataset context (c10 t1).
- Card language for scanned PDFs (no text at card time); `_phrase_document_count`
  tokenizes Vietnamese poorly.
- 9router listens on all interfaces (`*:20128`); uclab Codex workspace out of credits.
