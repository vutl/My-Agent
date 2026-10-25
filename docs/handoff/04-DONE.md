# 04 — Already done (updated 2026-10-10)

## 2026-10-25 — P1.5 capability sheet and routing

- Grouped all first-party tools, with registry-derived cached vi/en summaries,
  label fallback for unknown groups and approval guidance. The description-only
  registry needs no library searcher; chat adds the app's library summary.
- Chat system messages and agent tool loops share capability descriptions;
  file_qa/research system and document prompts do not receive the sheet.
- Strengthened action-versus-ability routing without keyword dispatch; added
  registry, language, cache, allowlist, prompt-isolation and offline SSE tests.
- Review follow-up: restrict greeting shortcuts to complete social greetings;
  fake-LLM regressions cover prefixed actions with tools allowed/disallowed,
  plus standalone greetings that still skip the model. The public-corpus guard
  verifies actual model calls instead of assuming a fixed shortcut count.
- Required backend command after review: **1687 passed, 3 host-capability skips**,
  zero failures. Graphify AST update and diff whitespace check passed.

## 2026-10-25 — Voice Plan A

- Further cleanup audit: suppress code fences nested in lists and blockquotes,
  match complete inline backtick delimiters, retain nested link labels and drop
  balanced URL targets (including escaped parentheses). Added 16 regressions,
  including an API check that code-only input never reaches translation/audio.
- Follow-up audit: anchor configured sidecar paths before changing cwd, keep
  the venv interpreter symlink, and parse fenced code without leaking incomplete
  opening labels or dropping prose after inline code. Added offline regressions
  for path handling, fence boundaries and runtime HF environment flags.
- Added the isolated TTS package, online idempotent installer, offline engine
  startup, policy-approved reference clones and per-engine synthesis locks.
- Registered speech status/voices/WAV proxy, configured-router Japanese
  translation, settings and lifespan-only process supervision with restart limits.
- Added fake-engine/transport/process tests, isolated test DB initialization
  from production data, and capability-check Seatbelt tests on restricted hosts.
- Required checks after further audit: backend 1626 passed / 3 host-capability skips; sidecar 41
  passed; no failures. Graphify AST update and diff whitespace check passed.
- PCM WAV encoding uses stdlib `wave` because the required backend test venv
  lacks `soundfile`; engine audio processing still uses the dedicated TTS env.

## RAG release candidate — 2026-10-22 (Claude)

- Sealed v4 85.3% and v5 86.5% (independent corpora/questions, one run each),
  0 fabricated, ≤4.2% false abstention; acceptance 60/60, vague-ref 45/47.
- Router decides loose/unresolved/shared names, vague and ambiguous references
  (asks when "the other one" is ambiguous), unlisted documents and versions;
  catalog titles hardened; ingest robust to stuck vision calls, interrupted
  revisions and restarts; LLM calls retry empty completions.

## Sealed v3 and its fixes — 2026-10-21 (Claude)

- Sealed v3 run once: 81.6% (CI 72–88), 1 fabricated, 13.8% false abstention.
- Router checks loose catalog name matches; scanned-page tables OCR'd by
  region; "show me" is not an image request; unlisted documents named as
  missing; ingest vision deadline; interrupted-revision retry.

## RAG on GPT-6, citations, OCR, routing — 2026-10-19 (Claude)

- Model router resolves vague document references; named document sets and
  one-sided comparisons are covered whole; per-document hybrid priority
  passages per sub-question lead long documents.
- [S#] citations (answer, verifier, UI chips, history); Apple Vision OCR with
  table rows for scans; offline end-to-end stream tests.

## Verifier and comparison fixes — 2026-10-16 (Claude)

- Post-verifier: whole documents ranked first among its sources; only stated
  values are checked (not years, counts, model names like ResNet-18); a
  removal or number-dropping rewrite of a line whose values are all in the
  sources is ignored.
- Title acronyms are aliases even when they spell app words ("RAG").
- English questions get an explicit answer language, restated after sources.

## Misspelled names, real titles, formulas, post-verify — 2026-10-14 (Claude)

- Sticky focus survives dataset/metric names; entities switch papers only when
  they resolve to exactly one other document.
- Real catalog titles (`app/catalog/titles.py`, backfill script, applied to
  production with backup); fuzzy + description name resolution; router may
  return `library_document`.
- Wrong table numbers yield to the described table, with a correction note.
- KaTeX formulas; Unicode math letters normalized for search; formula facet.
- `ANSWER_VERIFY_MODE=post`: stream first, one Sol call fixes only flagged
  lines. Live: 6 legacy false flags all kept, 0 errors.
- Quota retry (same model) for 9router round-robin accounts.

---
Older entries (before 2026-10-13): `docs/archive/handoff-history/04-DONE-before-2026-10-13.md`.
