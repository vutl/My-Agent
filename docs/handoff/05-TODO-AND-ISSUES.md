# 05 — TODO và vấn đề tồn đọng (2026-10-10)

## Assistant phase — 2026-10-22 (see 13-ASSISTANT-TOOLS-PLAN.md)

- [ ] Codebase cleanup: split backend/app/api/agent.py (harness run
  split-agent), ChatPage/types (harness run frontend-chat), group scripts/,
  retire the legacy evidence_validator path, refresh CODEBASE_MAP.md.
- [ ] Stream tool-call deltas so tool cards appear while the model writes.
- [ ] Junk appended to function arguments by the model/gateway
  (`…)"} regretting? ma`) — investigate 9router argument translation.
- [ ] Loopback token for the backend + `aya doctor`.
- [ ] Phase 3: reminders/heartbeat, Telegram (owner only), daily-brief skill.
- [ ] Phase 4: skills (research, paper-digest, file-organizer, data-analyst),
  TTS voice with lip sync from audio, VRM/own Live2D model for Aya.
- [ ] Deploy the assistant + Live2D build to production (:7777/:5173).

## RAG v1 RC1 — 2026-10-22

- [x] Regression on the final SHA, acceptance content-judged, rollback drill
  (`12-RELEASE-RAG-V1.md`).
- [x] Production rollout of `3219b2bc` on `data/` — 2026-10-22 20:28 (backup `data/backups/pre-rc1-20261022-202756`).
- [ ] Maintenance backlog (not blockers): see `12-RELEASE-RAG-V1.md` §Maintenance.
- [ ] Next phase: OpenClaw-style tools on the router → evidence → answer →
  verify pipeline.

## After sealed v3 — 2026-10-21

- [ ] Re-measure v3 (dev) after `072984ec`; then a fresh sealed v4 written by
  an independent model for the next out-of-sample number.
- [ ] Remaining v3 failures not yet addressed: v3-09 "which of my files come
  from EIA" (library inventory by publisher), v3-04/74 single facts missed in
  long native PDFs, v3-59 image-only slide content, v3-56/70 required details.
- [x] Catalog titles (generic, legacy-encoded, section names) — `ae92f69f`.
- [ ] Catalog: wrong card language (English LIGO papers tagged vi),
  `_phrase_document_count` tokenizes with [A-Za-z0-9] so Vietnamese words with
  diacritics break.
- [x] Startup backfill no longer activates an empty placeholder for an
  interrupted ingest — `0f267f61`.
- [ ] 9router listens on *:20128 (all interfaces) — bind to localhost.

## Sealed evaluations — 2026-10-16

- [ ] Sealed v2 50.0% (gate 85%). Rerun of failing conversations after the
  generic fixes pending; scanned tables (OCR diacritics "Quàng Ninh", split
  columns) and FOMC Table 1 retrieval are the next suspects.

- [ ] Scanned PDFs keep the filename as title (OCR headers are garbled), so
  description matches favour titled documents (sealed v1 s22). Derive a
  title from OCR text robustly or use the router's view of content.
- [ ] Hybrid fusion vs pure vector: doc hit@1 0.850 vs 0.917 on sealed v1
  (card routing / channel mix suspected; FTS pruning and FTS weight were
  measured and rejected).
- [x] Re-login 9router account the uclab account (401) — done 2026-10-16.
- [x] v1 comparisons s33/s34/s40/s41 — correct on rerun7 (2026-10-16).

---
Older entries (before 2026-10-13): `docs/archive/handoff-history/05-TODO-AND-ISSUES-before-2026-10-13.md`.
