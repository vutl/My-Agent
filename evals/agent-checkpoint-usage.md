# Durable agent HTTP/SSE evaluation

Use an isolated data root, never the production database, for current-code
startup migrations and regression. Runtime data/reports stay ignored by Git.

Create a new clone (existing targets are rejected):

```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/clone_agent_evaluation_runtime.py \
  --runtime-root data/retrieval_eval/internal/runtime/e2e-foundation-20261005
```

Start the backend with the clone's absolute path in `APP_DATA_DIR` and
`LIGHTRAG_ENABLED=false VISUAL_RETRIEVAL_ENABLED=false DUAL_QUERY_RRF_ENABLED=false`.
Port 7790 avoids production 7777. Frontend may use
`VITE_BACKEND_URL=http://127.0.0.1:7790 npm run dev`.

Run the visible suite with durable checkpoints:

```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/evaluate_agent_conversations.py \
  --dataset data/retrieval_eval/conversational-dev-v1.jsonl \
  --base-url http://127.0.0.1:7790 --model cx/gpt-5.6-sol \
  --runtime-root data/retrieval_eval/internal/runtime/e2e-foundation-20261005 \
  --include-answers \
  --output data/retrieval_eval/internal/agent-dev20-foundation-20261005.json
```

Each completed turn is written atomically to `<output>.partial`. Client timings
come from the original SSE exchange, never a reconstruction from server logs.
Failed quality assertions are retained. Provider/run failures stop the suite
without selecting another model. Selecting a follow-up with `--case-id` includes
its actual prior turns automatically.

Add `--resume` only after the old evaluator process has terminated. The command
verifies its exact case/code/index/image/options contract and HTTP-persisted
conversation/run history before reusing rows. A temporary empty conversation
marker proves that the declared clone is the actual backend data root; no
model request is made for this verification.

An interrupted in-flight turn is recorded and rejected on resume, even if it
might subsequently have completed server-side. Do not erase this audit trail
or turn uncertain timings into a pass. Inspect that run and explicitly replay
the affected conversation from its start into a new report. Automatic recovery
of an in-flight turn is not implemented. Completed unaffected checkpoints can
resume without new Sol calls.

Do not run two evaluators with the same output path. Do not modify backend
code or canonical projections during a measured run. Completion requires all
requested cases and no transport/index/code-contract error. A green visible
regression is not a blind shadow result or general arbitrary-file acceptance.
