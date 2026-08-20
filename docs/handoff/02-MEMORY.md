# 02 — Conversation memory (L0 / L1 / L2 / L3)

Updated **2026-07-23 (UTC+7)**.

## Invariants

- SQLite raw messages are the source of truth; a summary never replaces L0.
- The foreground answer never waits for an LLM summary.
- An old stable summary is paired with durable completed turns after its cursor,
  so a quick next message does not lose the previous turn.
- Answer/router/summary mặc định dùng **9router `cx/gpt-5.6-sol`**. Quota/provider errors are
  surfaced or persisted for same-model retry; there is no heuristic/local-model
  summary and no silent model switch.
- Text stored in SQLite is full text. Prompt limits use episode/turn boundaries
  and visible omission markers, not an unmarked character slice.

## L0 — immutable event/episode log

- `messages`: every user/assistant message, full text and timestamp.
- `message_search`: FTS5 index synchronized by insert/update/delete triggers;
  migration backfills old messages idempotently.
- `HistoricalConversationSearch` retrieves complete user/assistant episodes,
  not isolated facts. It uses safe FTS tokens plus LIKE/accent-insensitive
  overlap fallback.
- Normal queries search the current conversation. Cross-thread search is only
  enabled by an explicit history cue such as “hôm qua”, “lần trước”, “previous
  conversation”, or an explicit router/UI decision.
- Episodes overlapping the raw-recent window are excluded atomically, avoiding
  a duplicated or half-visible exchange.

Code: `backend/app/services/long_term_memory.py`.

## L1 — synchronous working state

- `conversations.metadata_json.working`
- Active document IDs, topic, filenames, answer intent, and recent document
  thread.
- Updated only after a grounded successful retrieval answer.
- Casual detours do not erase active paper focus.

Code: `backend/app/services/conversation_state.py`.

## L2 — durable sleep-time consolidation

Durable tables:

- `conversation_memory_turns`: complete paired exchanges with monotonic
  `turn_seq`, message IDs, focus snapshot, and completion time.
- `conversation_memory_jobs`: `dirty_through_seq`, `summary_through_seq`,
  status, retry cursor/error, and next attempt time.
- `conversation_memory_l3_outbox`: L3 operations committed atomically with the
  L2 summary and acknowledged only after materialization succeeds.
- `memory_operation_receipts`: per-source/per-operation replay protection.
- `conversations.summary`: stable cumulative summary.
- `metadata_json.memory.recent_beats`: three sentence-safe convenience notes;
  user budget 800 chars and assistant budget 1,200 chars. Beats are not the fold
  source.

Foreground completion synchronously records the full turn + dirty cursor, then
returns. Production consolidation waits for a **12-second idle window**, so fast
bursts coalesce. There is one task per conversation and at most one GPT fold in
the whole process.

Foreground priority is global: when any answer is queued/running, active folds
are canceled safely and their job remains pending. Same-conversation foreground
turns serialize; different conversations may answer concurrently.

Prompt protection:

- Stable summary budget: 12,000 chars / prompt asks for at most ~800 words.
- Pending-turn block: about 16,000 chars, newest whole turns first, with an
  explicit count when older pending turns do not fit.
- Fold input: chronological prefix around 48,000 chars; the summary cursor only
  advances through the last included turn, then continues in another batch.
- Raw recent context: up to 12 messages / about 7,200 chars, sentence-safe.

Failure/restart behavior:

- Provider/quota failure persists `last_error`, exponential retry time and the
  same dirty cursor; it does not fabricate a local summary.
- Startup repairs interrupted `running` jobs, completed-agent crash gaps, and
  direct-chat adjacent user→assistant pairs missed before memory enqueue.
- Foreground/shutdown cancellation preserves provider retry backoff.
- Shutdown drains for a bounded time; remaining work stays durable.
- Historical conversations created before this migration are `dormant`: they
  remain searchable/prompt-visible but do not trigger a bulk GPT summarization
  storm. A new completed turn wakes that conversation and folds its backlog.

Code: `backend/app/services/conversation_memory.py`,
`backend/app/services/conversation_runtime.py`.

## L3 — typed long-term memory

`memory_items` stores append-only versions of:

- `semantic`: stable user/project facts and preferences.
- `episodic`: decisions/events worth recalling.
- `procedural`: rules Aya should consistently follow.

Each item includes logical key, scope (`user` or one conversation), confidence,
validity interval, source conversation/turn, and `supersedes_id`. Updating a key
closes the prior version; forgetting sets `valid_to` instead of deleting audit
history. Procedural items are always considered; semantic/episodic items require
query relevance.

The L2 GPT call may return strictly validated `memory_ops`. Invalid scope/kind,
malformed keys, low-integrity payloads and password/API-key/token content are
discarded. Paper/assistant claims must never become global user facts. Applying
the same operation after a retry is idempotent by source cursor and operation
fingerprint. Callback failures replay from the durable outbox without another
GPT call and remain visible in health.

Code: `backend/app/services/long_term_memory.py`.

## Prompt pack

```text
L1 active working state
+ relevant L3 procedural/semantic/episodic items
+ stable L2 rolling summary
+ recent folded beats
+ full durable turns after summary_through_seq
+ relevant historical L0 episodes
+ raw recent messages
+ retrieved paper/table/figure evidence (when RAG runs)
+ current user turn
```

The router receives the stable summary separately from bounded recent/pending,
L3, and historical notes, avoiding duplicate summary text.

## Live migration snapshot (2026-07-23 10:40 +07)

- 555 raw messages and 555 FTS rows after smoke-test cleanup.
- 269 durable paired turns across 118 conversation jobs.
- 17 jobs idle/caught up.
- 101 legacy jobs dormant (198 historical unfurled turns, **zero startup GPT
  calls** until those threads are used again).
- L3 starts empty and is populated only by future validated consolidation ops.

## Evaluation coverage

- Burst larger than the old three-beat window.
- Coalescing/global max-inflight.
- Foreground cancel/resume across different conversations.
- Failure/backoff with no summary/model fallback.
- Restart/interrupted-job recovery and completed-run crash-gap recovery.
- Bounded multi-batch cursor progression.
- Same-thread vs explicit cross-thread L0 retrieval.
- Temporal L3 supersede/forget/provenance/idempotency.
- Same-conversation serialization and cancellation cleanup.
