# Retrieval Evaluation Gate v2.1

This directory is the versioned, retrieval-only evaluation contract for the
original-query / frozen-standalone-query RRF experiment. It measures retrieval
only: the runner does not create rewrites, instantiate an answer model, call
9router, or mutate any production index.

## Status

- `dev-v2.1.jsonl` is retained as an audited **diagnostic only** suite. It is
  now rejected for promotion because language and contextuality are perfectly
  confounded (all VI cases are contextual and all EN cases are standalone),
  and several categories are dominated by one repeated template.
- `dev-v2.2.jsonl` is the current audited replacement. It keeps the same
  six-category/60-case coverage and portable qrels, balances the four
  language/context cells at 15 cases each, and varies question forms. Its
  audit attestations and rewrite fingerprint are frozen; use this suite for
  Gate-C measurement. The `*.proposed*.jsonl` files remain review-only inputs.
- `dev-v2.jsonl` and `retrieval-v2-dev.json` are retained for audit only. They
  are `invalid_for_promotion`: a gold caption leaked into an enriched query,
  typed evidence serialization lost table/figure IDs, and final-stage reranking
  was not independently measured.
- `dev-v2.1.proposed.jsonl` is a **review-required structural proposal**, not a
  benchmark and not promotion evidence. It has the intended 6×10 / VI-EN /
  contextual matrix and stable document/artifact locators, but a reviewer must
  verify each question and qrel semantically.
- `dev-v2.1.review-packet.jsonl` is a read-only audit worksheet generated from
  that proposal. It exposes the exact source span/caption and its portable
  locator; it does not contain runtime UUIDs and cannot approve a case.
- `dev-v2.2.jsonl` is the currently published audited suite and is the
  evaluator default. A future reviewed replacement must pass the same
  validation before publication; the evaluator refuses non-audited suites by
  default.
- `shadow-v2` is deprecated and cannot be opened. Build a new sealed
  `shadow-v3` from genuine public MMLongBench-Doc/SPIQA question–evidence pairs
  ingested through an isolated production pipeline; do not synthesize pairings
  or cycle categories.
- `shadow-v3.manifest.json` and `shadow-v3.questions.jsonl` are now a sealed,
  answer-free frontier: 24 cases across six public format/modality buckets and
  24 distinct documents excluded from the visible Gate-E frontier. The
  manifest stores only case IDs and source/content fingerprints; `opened` is
  `false`. It must not be treated as a score until the RC freeze and explicit
  one-time shadow run.

The detailed invalidation record is in `invalidated_reports.json`.

## Public visual input fidelity

SPIQA test-C ingestion and all three answer modes now read the pinned official
`SPIQA_testC_Images.zip` (original-resolution images), not its 224px archive.
Missing native images fail closed; thumbnails are never a silent substitute.
The source contract hashes every authored artifact, not only referred/gold
images. Image-byte changes create isolated staged revisions; the answer probe
also records the actual supplied width/height, byte count and SHA-256.

Historical 224px reports are retained as thumbnail-condition diagnostics. They
do not establish original-resolution OCR/VLM quality and must not be pooled
with native-image runs or cited as native-document acceptance. This is an
input-condition correction, not a measured improvement in retrieval ranking.
The official dataset card distinguishes both archives:
https://huggingface.co/datasets/google/spiqa

## Case contract

Every audited case contains:

- raw user query, full frozen standalone rewrite, and diagnostic anchor rewrite;
- conversation context plus working state for follow-ups;
- provenance for every rewrite source (`raw_query`, `conversation`,
  `working_state`, or `catalog`);
- source/content-hash document locators, and text/table/figure/page qrels that
  resolve to fresh runtime IDs; and
- an explicit, uniform rewrite-policy fingerprint, plus reviewer attestations
  `language_reviewed: true` and `qrels_reviewed: true`.

`validate` fails closed for bad language/category/contextual balance, duplicate
cases, weak 21-document coverage, unresolved qrels, leaked qrel caption/page/
artifact facts, unsupported exact entities, or inconsistent rewrite fingerprints.
One evidence group may declare equivalent same-document locators under
`alternatives`; any resolved alternative is a hit. Separate documents in a
comparison remain separate required groups.

## Workflow

```bash
# 1. Produce a structural proposal. This never overwrites an audited suite.
./backend/.venv/bin/python scripts/build_retrieval_eval_v2.py propose

# Gate-C v2.2 proposal: orthogonal language/context matrix and varied wording.
./backend/.venv/bin/python scripts/build_retrieval_eval_v22.py

# Preflight v2.2 locators without treating them as semantic approval.
./backend/.venv/bin/python scripts/build_retrieval_eval_v2.py preflight \
  --proposal-output evals/retrieval_v2/dev-v2.2.proposed.jsonl

# 2. Resolve every locator against the current index. This proves only that
#    qrels are portable and reachable; it is not semantic approval.
./backend/.venv/bin/python scripts/build_retrieval_eval_v2.py preflight

# 3. Export a review packet. A reviewer verifies every question, source span,
#    artifact caption and rewrite against this evidence, then copies the
#    proposal to a separately named reviewed file. The generated proposal can
#    never be published as its own review.
./backend/.venv/bin/python scripts/build_retrieval_eval_v2.py review-packet

# 4. In the separately reviewed copy, set curation_status to "audited" and
#    audit.status to "approved". Every case needs language_reviewed and
#    qrels_reviewed, a named reviewer, timestamp, notes, and one common frozen
#    rewrite fingerprint. Do not derive rewrites from caption/page/answer text.

# 5. Validate and publish the reviewed file (use a versioned output name).
./backend/.venv/bin/python scripts/build_retrieval_eval_v2.py publish-reviewed \
  --reviewed-input /absolute/path/to/dev-v2.2.reviewed.jsonl \
  --suite-output evals/retrieval_v2/dev-v2.3.jsonl

# 4. Run A0/A1/B-full/B-anchor on the audited v2.2 suite. This calls only local EmbeddingGemma through
#    Ollama and reads the existing SQLite/LanceDB index.
./backend/.venv/bin/python scripts/evaluate_dual_query_retrieval.py \
  --suite evals/retrieval_v2/dev-v2.2.jsonl \
  --expected-cases 60 \
  --expected-rewrite-fingerprint <frozen-policy-fingerprint> \
  --output data/retrieval_eval/internal/retrieval-v2.2-dev.json

# Build (but do not open) the public shadow manifest after the RC frontier is
# frozen. This command never emits answers/qrels into the tracked files.
./backend/.venv/bin/python scripts/build_shadow_v3_manifest.py

# Validate the sealed manifest and ignored private qrels without opening it.
./backend/.venv/bin/python scripts/validate_shadow_v3_manifest.py

# Production-policy diagnostic (after an isolated DB/Lance projection exists).
# This measures per-document compare branches and facet-parallel multi-hop;
# it never calls Sol and remains invalid until the proposal is semantically
# audited.
./backend/.venv/bin/python scripts/evaluate_agent_policy_retrieval.py \
  --suite evals/retrieval_v2/dev-v2.2.proposed.jsonl --allow-draft \
  --db-path /path/to/isolated/app.db \
  --lancedb-path /path/to/isolated/lancedb \
  --output data/retrieval_eval/internal/agent-policy-v2.2-diagnostic.json
```

`--allow-draft` exists only for contract/debug smoke tests and marks its output
as unsuitable for promotion. It must not be used to report Aya quality.

## Arms and metrics

- **A0:** production control — one `standalone_full` query, existing document
  candidate behavior.
- **A1:** A0 plus candidate-document union.
- **B-full:** raw query + `standalone_full`, with generic raw follow-ups removed
  only when they add no exact/content-bearing signal.
- **B-anchor:** raw query + `standalone_anchor`; diagnostic only, never a
  production treatment.

All arms keep the same external candidate depth (top 30), then apply the
production reranker exactly once to that pool for final top 10. Reports include
Candidate Recall@30, Document Recall@5, Evidence Recall@10, Hit@k, MRR/nDCG,
modality recall, multi-document coverage, scope leaks, per-stage timing,
embedding/rerank calls, and failure taxonomy.

No rewrite word-count rule is used. Full versus anchor is reported as a paired
ablation by length bucket, contextuality, exact-anchor signal, category, and
raw-query informativeness.
