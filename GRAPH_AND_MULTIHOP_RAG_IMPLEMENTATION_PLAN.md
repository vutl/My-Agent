# Graph & Multi-Hop RAG — Delta Implementation Plan

> **Project:** My Agent / Aya
> **Version:** 2.1
> **Updated:** 2026-07-31
> **Purpose:** Record the implemented graph/multi-hop delta and the evaluation
> gates that still remain. This is not a standalone or green-field blueprint.

## Status legend

- `[x]` Already implemented in the repository.
- `[~]` Partially implemented or locally verified, but not complete.
- `[ ]` Still required.

## Delivery snapshot

- Live provider-free provenance backfill: **21/21 documents**, **335/382**
  LightRAG chunks safely mapped to **931** parent links; **47** ambiguous chunks
  were rejected. Exact/normalized mappings produced 709 positive-overlap links;
  fuzzy top-1 mappings remain explicitly marked with zero overlap.
- Provider-free live bridge smoke resolved a graph entity to scoped canonical
  parent passages with graph descriptions excluded from answer evidence.
- Backend: **344 passed**; compileall and desktop production build passed.
- These checks establish implementation safety, not retrieval quality on a
  representative benchmark. The frozen 60-question held-out evaluation below
  is intentionally still open.

## 1. Current baseline — do not reimplement

- [x] Token-aware structural parent/child chunks are stored in SQLite/LanceDB.
  A child row links through `chunks.parent_chunk_id`; parent text is currently in
  `chunks.metadata_json.parent_content`.
- [x] `backend/app/lightrag/query.py` queries the installed LightRAG API through
  `aquery_data(..., only_need_context=True)`.
- [x] `backend/app/lightrag/bridge.py` converts LightRAG chunks, entities and
  relationships into the shared retrieval-result shape.
- [x] LightRAG provenance is resolved to canonical documents using document ID
  or an unambiguous source path. Ambiguous basename/multi-paper provenance is
  dropped. L1 focus is a filter and must never be used to relabel evidence.
- [x] A supplied `collection_id` is applied to LightRAG as well as local
  retrieval. An explicitly empty collection is an empty scope, not permission
  to search the full corpus.
- [x] Known multi-paper comparisons create one canonical document branch per
  paper and execute branches with `asyncio.gather`.
- [x] Retrieval evidence is validated before answering and may trigger one
  bounded second retrieval pass.
- [x] Quantitative paper answers are buffered, validated against evidence, and
  only then emitted as `message.delta` chunks. Ordinary answers may stream
  normally after retrieval.
- [x] The cross-encoder implementation is local PyTorch/Transformers, disabled
  by default, and requires an explicit local model path. There is no ONNX
  production reranker in the current repository.

The implementation below must extend these paths rather than add fictional
methods such as `search_entities()` or query columns such as `parent_index` and
`parent_text`, which do not exist.

## 2. Target data flow

```text
user query + L0/L1/L2 context
        |
        v
query rewrite + canonical paper/collection scope
        |
        v
deterministic staged engine policy (no extra router-model call)
        |
        +--> focused/direct/table query --> scoped FTS + LanceDB + parent expansion
        |
        +--> discovery/cross-document --> LightRAG aquery_data
                                                |
                         graph source_id(s) ----+
                                                v
                             durable source-to-parent provenance
                                                |
                                                v
                                 canonical parent/table/figure evidence
        |
        v
coverage/evidence validation
        |
        +--> sufficient --> answer
        |
        +--> evidence gap --> bounded hop 2, at most 3 scoped subqueries
                                      |
                                      v
                         merge + dedupe + validate again
                                      |
                                      v
                            answer or grounded refusal
```

The graph is a navigation layer. Entity names, relationship descriptions and
graph paths may propose candidates, but the final answer must be grounded in a
canonical parent passage, table, figure or source chunk with valid provenance.

## 3. Workstream A — durable graph-to-parent provenance

### 3.1 Required storage contract

- [x] Add an idempotent SQLite table for the many-to-many mapping:

```text
lightrag_chunk_parent_provenance
  lightrag_chunk_id
  document_id
  parent_chunk_id
  overlap_chars
  content_hash / parent_content_hash
  canonical_method / mapping_method / mapping_score
  document_char_start / document_char_end
  mapped_at

PRIMARY KEY (lightrag_chunk_id, parent_chunk_id)
FOREIGN KEY document_id -> documents.id ON DELETE CASCADE
INDEX (document_id, parent_chunk_id)
```

This is a new schema. It must not be confused with existing `chunks` columns.
A LightRAG chunk may overlap more than one parent, so the mapping is not
necessarily one-to-one. `parent_chunk_id` is currently a logical group ID, not
a guaranteed `chunks.id`, so it must not be declared as a foreign key to
`chunks(id)` unless parent passages are normalized into their own table first.

### 3.2 Mapping lifecycle

- [x] After a canonical LightRAG ingest reaches `processed`, collect the
  document-owned LightRAG chunks and align their normalized text spans with the
  parent units used to construct the ingest text.
- [x] Persist exact/fingerprint matches first. Persist overlap matches only when
  ownership is canonical and the match clears an explicit confidence threshold.
- [x] Never infer a parent from a `-chunk-NNN` suffix, active-paper focus, a
  duplicated filename, or a hardcoded paper/figure mapping.
- [x] Split aggregated graph `source_id` fields on LightRAG's `<SEP>` separator,
  resolve every source independently, and preserve valid multiple parents.
- [x] Rebuild mappings after re-ingest and remove them when a document is
  deleted or reindexed.
- [x] Expose diagnostics for mapped, ambiguous, stale and unmapped source IDs.
  Ambiguous rows are dropped from graph grounding rather than guessed.

### 3.3 Runtime parent grounding

- [x] Keep `aquery_data` as the LightRAG query API.
- [x] For each returned entity/relation, resolve its `source_id` values through
  the durable mapping, then fetch one child row with the matching
  `(document_id, parent_chunk_id)` and read `metadata_json.parent_content`.
- [x] Deduplicate by `(document_id, parent_chunk_id)` and retain graph anchors
  only as navigation metadata.
- [x] Apply L1 and collection scope before a parent can enter the composed
  context. If provenance does not prove ownership, exclude the candidate.
- [x] Add per-document candidate quotas so a high-degree paper cannot crowd all
  evidence from the other compared papers.

Runtime retrieval must not parse
`data/lightrag/graph_chunk_entity_relation.graphml` directly and must not query
the exported Obsidian vault. Those are inspection/export artifacts, not the
application's retrieval API.

### 3.4 Generic graph-hub suppression

- [x] Suppress structural labels such as `Figure 2`, `Table 1`, `Section 3`,
  page/equation labels, and relations whose endpoints are both structural.
- [x] Suppress or strongly demote short, high-source-cardinality hubs such as
  generic `Model`, `Accuracy`, `IEEE` or `Journal` nodes.
- [x] Keep specific shared concepts when their provenance is resolvable; do not
  delete graph nodes globally and do not encode paper-specific exceptions.
- [x] Record suppression reason/source cardinality in retrieval diagnostics so
  tuning is auditable.

### 3.5 Implementation files

- `backend/app/db/sqlite.py` — schema and idempotent migration.
- `backend/app/lightrag/ingest.py` — mapping build/invalidation lifecycle.
- `backend/app/lightrag/provenance.py` — mapping/alignment/resolution logic.
- `backend/app/lightrag/bridge.py` — graph candidate suppression, parent
  grounding and diagnostics.
- `backend/app/rag/context.py` — only if parent-level dedupe or per-document
  quotas cannot be expressed by the bridge.

## 4. Workstream B — deterministic staged engine policy

The engine selector is retrieval policy, not a second LLM intent call.

| Condition after rewrite/scope resolution | `auto` path |
|---|---|
| Explicit FTS request | Scoped local hybrid |
| Focused result-table/benchmark request | Scoped local hybrid/table chunks |
| One canonical focused paper, direct QA | Scoped local hybrid + parent expansion |
| No focused paper, corpus discovery | LightRAG |
| `compare` / `infer_structure` or at least two focused papers | LightRAG, with per-document compare branches where applicable |
| Explicit `legacy`, `lightrag` or `dual` configuration | Honor for diagnostics/manual control |

- [x] Make `auto` the configured default and keep this policy deterministic in
  `backend/app/api/agent.py`.
- [x] Preserve existing one-document-per-branch comparison and parallel
  execution.
- [x] Treat multi-hop as an evidence-conditioned orchestration layer over these
  engines, not as a fourth independent index.
- [x] Emit selected-engine and policy-reason diagnostics for every retrieval.

### Fail-closed scope and fallback rules

1. A selected collection with zero documents, or an empty canonical
   collection/focus intersection, returns no corpus evidence; it never widens
   scope.
2. L1 focus filters proven provenance; it never manufactures provenance.
3. An uninitialized optional LightRAG store may use the same scoped local
   retrieval path.
4. A valid LightRAG response that becomes empty after safe scope/provenance
   filtering may use the same scoped local path and must report that reason.
5. Provider, quota, authentication, model-policy, corruption and provenance
   integrity errors propagate. They are not caught by a blanket
   `except Exception` and are not hidden by local/model fallback.

## 5. Workstream C — evidence-conditioned two-hop retrieval

### 5.1 Existing behavior

- [x] Hop 1 uses the rewritten query.
- [x] Known comparisons use deterministic per-document branches without
  another model call.
- [x] A second pass can already be requested by evidence validation or missing
  document/visual coverage.

### 5.2 Required adaptive behavior

- [x] Derive missing facets from the user request and hop-1 evidence:
  architecture, method/training, dataset/setup, benchmark results, ablation,
  limitations and visual evidence.
- [x] Extract only specific, provenance-backed graph anchors from hop 1.
- [x] Build hop-2 subqueries deterministically from the original task, sticky
  topic, required entities, missing documents/facets and safe graph anchors.
  The default path makes no additional planner-LLM call.
- [x] Execute subqueries in the same hop concurrently, with each branch carrying
  its canonical document and collection scope.
- [x] Merge hop 1 and hop 2 by canonical evidence identity. Retain hop 1 if hop 2
  does not improve validation/coverage.
- [ ] Keep an optional LLM subquery planner outside the default path until it
  has a separate call/token/latency evaluation and an explicit setting.

### 5.3 Hard bounds

- Maximum retrieval hops: **2 total**.
- Maximum hop-2 subqueries: **3 total**.
- Maximum concurrent work: only the bounded branches in the active hop.
- Deadline: each hop-2 branch currently uses
  `AGENTIC_RETRIEVAL_HOP_TIMEOUT_SECONDS` (45 seconds by default) through
  `asyncio.wait_for`, and timeout/provider failures propagate. An enclosing
  absolute request deadline with a measured generation reserve is still an
  evaluation/rollout task; it is not claimed as implemented here.
- No progress: stop when a completed hop adds no new canonical evidence IDs and
  does not improve requested-facet/document coverage. There is never a hop 3.
- Model errors: only configured transient timeout/connection retries with the
  exact same approved model are allowed. Quota/status/policy errors stop the
  operation and surface to the user.

## 6. SSE and answer-generation contract

No new incompatible SSE protocol is required:

```text
tool.started
  { run_id, conversation_id, tool_name, input? }

retrieval.retrying
  { run_id, conversation_id, query, sub_queries?, hop?, max_hops?,
    parallel?, reason, reasons?, missing_entities,
    previous_focus_document_ids }

retrieval.completed
  { run_id, tool_call_id, conversation_id, query, documents, ... }

message.delta
  { delta }
```

- [x] Add hop/subquery metadata only through fields accepted by
  `apps/desktop/src/lib/types.ts`.
- [x] Emit one `retrieval.retrying` event before hop 2 and preserve existing
  `tool.started`, `tool.completed`, `retrieval.completed` and timing events.
- [x] Retrieval finishes before answer token generation, so graph/multi-hop work
  increases time-to-first-token; this plan makes no “zero interruption” claim.
- [x] Quantitative/evidence-critical answers may be fully buffered for claim
  validation and then emitted in bounded `message.delta` chunks.

Implementation files:

- `backend/app/api/agent.py`
- `backend/app/core/events.py`
- `apps/desktop/src/lib/types.ts`
- `apps/desktop/src/lib/sse.ts`

## 7. Privacy and model-failure boundary

### Local by default

- Raw corpus and extracted artifacts at rest.
- SQLite/FTS, LanceDB vectors and LightRAG stores at rest.
- Ollama embeddings and local lexical/vector retrieval.
- Cross-encoder reranking only when explicitly enabled with a local model path.

### Sent through 9router to the configured upstream model when used

- Document text supplied to LightRAG ingest/entity-relation extraction.
- User/retrieval queries used by LightRAG query processing.
- Final-answer prompts containing selected passages plus required chat/memory
  context.
- Figure/page crops used by VLM enrichment.
- Background L2 summarization/folding inputs when that configured worker runs.

Therefore this system is **local-first, not 100% local/offline**. The local
`:20128` process is a gateway; it does not imply local inference.

Approved models remain:

- `cx/gpt-5.6-sol` (default)
- `cx/gpt-5.6-terra`
- `cx/gpt-5.6-luna`

No quota/auth/status failure may silently switch to another approved model,
Ollama, or another provider. A transient timeout may retry only the exact same
configured model within its bounded retry policy.

## 8. Verification

### 8.1 Unit and integration tests

- [x] `backend/tests/test_lightrag_bridge.py`
  - split/dedupe `<SEP>` source IDs;
  - ambiguous provenance is never relabelled by focus;
  - structural/high-cardinality hubs are suppressed;
  - graph candidates resolve to the correct canonical parent;
  - stale/unmapped provenance is dropped.
- [x] `backend/tests/test_lightrag_ingest.py`
  - mapping is created only after processed ingest;
  - re-ingest/delete invalidates old mappings;
  - ingest/provider failures do not leave mappings marked valid.
- [x] `backend/tests/test_retrieval_agent_service.py`
  - deterministic facet/subquery planning;
  - maximum two hops and three hop-2 queries;
  - no-progress stop;
  - per-document scopes survive parallel execution.
- [x] `backend/tests/test_agent_retrieval_focus.py`
  - deterministic `auto` routing;
  - L1 and collection intersection is fail-closed;
  - explicit empty collection never widens;
  - LightRAG provider/quota failure is not hidden by fallback.
- [x] Agent integration behavior is covered in
  `backend/tests/test_agent_retrieval_focus.py` rather than a duplicate
  `test_agent_graph.py`:
  - graph-parent evidence reaches the answer context;
  - hop-2 SSE payload matches the desktop contract;
  - hop 1 remains selected when hop 2 adds no canonical evidence;
  - per-branch timeout and provider errors stop the run.
- [x] Existing numeric-claim tests continue proving that unsupported
  metrics are removed/refused before `message.delta`.
- [x] Complete backend suite: **344 passed**; `compileall` and desktop
  production build passed on 2026-07-31.

### 8.2 Held-out evaluation requirements

The former eight-query smoke set is useful for development only and must not be
used as production evidence.

- [ ] Freeze a held-out set before threshold/routing tuning. Split by document
  or paper family, not by paraphrased query, to prevent source leakage.
- [ ] Include at least 60 independently labelled questions covering:
  single-paper direct QA, cross-document relation, true two-hop synthesis,
  Vietnamese/English queries, hard-negative generic hubs, figure/table asks and
  unanswerable questions.
- [ ] Record gold canonical documents, parent passages, required facets and
  valid supporting paths. Keep adjudication blind to the retrieval engine.
- [ ] Compare the same frozen set across:
  1. scoped local hybrid;
  2. current LightRAG bridge;
  3. graph plus durable parent grounding;
  4. graph plus bounded adaptive hop 2.
- [ ] Report retrieval Hit@k/MRR/nDCG only where those labels are meaningful,
  plus provenance precision/coverage, cross-document coverage, path validity,
  answer grounding, unsupported-claim/no-answer behavior, P50/P95 retrieval,
  first validated token, total latency, upstream calls and tokens.
- [ ] Acceptance requires zero scope leakage, no ambiguous provenance promoted
  to evidence, no silent model fallback, and a measured cross-document/grounding
  gain without an unreviewed latency/cost regression.

## 9. Delivery checklist

- [x] Durable `source_id -> canonical parent` mapping.
- [x] Graph candidate grounding in parent passages.
- [x] Generic structural/high-cardinality hub suppression.
- [x] Deterministic staged `auto` engine policy.
- [x] Evidence-conditioned, no-extra-LLM default hop-2 planning.
- [x] Per-branch deadline propagation and canonical no-progress execution tests.
- [ ] Enclosing absolute request deadline with measured generation reserve.
- [x] Full backend suite and desktop build.
- [ ] Frozen held-out cross-document/multi-hop evaluation.
- [x] Update handoff status after implementation and verification.
