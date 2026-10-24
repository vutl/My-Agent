# 10 — Agent harness: who does what (2026-10-17)

How development work on Aya is split between models. Aya itself runs on
9router `cx/*` models (AGENTS.md). The harness calls the Codex and Claude CLIs
directly; the only exception (user, 2026-10-25) is the `*_ag` roles, which run
`claude -p` through 9router's Anthropic-compatible endpoint with Antigravity's
`ag/claude-opus-5-5*` / `ag/claude-sonnet-5-5*` models. Those spare the Codex
accounts (shared with Aya) and the Claude subscription; any gateway failure
hands the call to the role's fallback.

## Roles

| Role | Engine / model | Sandbox | Used for |
|---|---|---|---|
| planner | Claude `fable` (head session or `claude -p`) | plan mode (read-only) | turn a goal into a plan: files, steps, tests, acceptance |
| plan_review | Codex `gpt-6-astra`, effort high | read-only | attack the plan: missing cases, hardcoding, wrong files; verdict approve/revise |
| executor | Codex `gpt-6.1-sol`, effort high | workspace-write, own git worktree | implement an approved plan, run tests |
| coder | Codex `gpt-6-luna` (or Claude `sonnet` / `haiku`) | workspace-write, own worktree | bounded edits: tests, refactors, scripts |
| reviewer | Codex `gpt-6-astra`, effort high | read-only | review the diff against the plan; blocking findings go back to the executor |
| judge | Codex `gpt-6-astra`, effort medium | read-only | grade eval answers (independent of Aya's gpt-5.6-sol) |
| scout | Claude `haiku` | read-only | cheap lookups across files |
| planner_ag, executor_ag, coder_ag, reviewer_ag, judge_ag, scout_ag | `claude -p` via 9router, `ag/claude-opus-5-5(-high)` / `ag/claude-sonnet-5-5(-high)` | as the matching role | same jobs, Antigravity quota; fallback to planner_codex / executor / coder / reviewer / judge / scout |

The head agent (Claude in the user's session) owns the goal, picks the role
for each piece of work, merges branches and reports. Nothing merges to `main`
without the head agent reading the diff and the test result.

## Accounts and quota

- Two ChatGPT accounts, one `CODEX_HOME` each: `~/.codex-vutlpro` (vutlpro; `~/.codex` belongs to the IDE)
  and `~/.codex-uclab` (vltran). One-time login for the second:
  `CODEX_HOME=~/.codex-uclab codex login` (browser flow on localhost:1455;
  the uclab Team workspace blocks `--device-auth`).
- `tools/harness/cx.py` picks the account used least recently that is not
  cooling down; on a usage-limit / 429 reply it records the reset time
  ("reset after 28m") and retries on the other account. State:
  `.harness/state.json`; every call is logged to `.harness/calls.jsonl`.
- Claude quota (Fable/Sonnet/Haiku) is the scarcer one: planning and the head
  role only. Judging, plan review, review and bulk execution go to Codex.
- The same two ChatGPT accounts back 9router (Aya's answers and ingestion):
  every harness call spends Aya's quota too. Keep Codex calls to plan review,
  review, judging and execution; run Aya evals in batches with quota waits.

## Pipeline (`tools/harness/pipeline.py`)

```text
task.md -> plan (fable) -> plan_review (astra) -> [revise ≤2] -> approved plan
        -> execute (6.1-sol, worktree harness/<run>) -> tests
        -> review (astra) -> [fix ≤2 by executor] -> report + branch
```

Artifacts per run in `.harness/runs/<run>/`: `task.md`, `plan.md`,
`plan_review.json`, `execute.log`, `tests.log`, `review.json`, `report.md`.
The branch `harness/<run>` is left for the head agent to merge.

Batch jobs use the same account rotation, e.g. eval judging:
`python tools/harness/judge.py --questions … --answers … --out …` shards the
conversations and runs `judge` calls in parallel (one per account at a time).

## Skills and MCP shared by Claude and Codex

- One source of truth: `.agents/skills/<name>/SKILL.md` (agentskills.io
  format, read by Codex). `tools/harness/sync_skills.sh` symlinks every skill
  into `.claude/skills/` (Claude) and `.codex/skills/`.
- Project skills: `aya-runtime` (services, ports, 9router), `aya-eval`
  (suites, runs, judging, gates), `agent-harness` (how to dispatch roles).
- MCP for both (`.mcp.json` for Claude, `.codex/config.toml` for Codex):
  graphify (existing), context7 (current library docs), playwright (drive the
  desktop UI in a browser).

## Rules every role gets

- Read AGENTS.md and docs/handoff/00-STATUS.md first; graphify before grep.
- Fix generically; never hardcode a paper, figure number or test case.
- Never fabricate numbers; keep sticky focus; Aya models stay cx/gpt-5.6-*.
- Tests: `cd backend && PYTHONPATH=.. .venv/bin/python -m pytest -q` must pass.
- After code changes: `graphify update .`.

## Antigravity through 9router (2026-10-25)

- Config: `[gateways.9router]` in `tools/harness/harness.toml` (base URL, token
  read from `NINEROUTER_API_KEY` or `OPENAI_API_KEY` in `backend/.env`); a role
  opts in with `gateway = "9router"`.
- Verified: Claude Code CLI with `ANTHROPIC_BASE_URL=http://127.0.0.1:20128`
  runs tools (Read) on `ag/claude-sonnet-4-6`.
- 2026-10-25: Antigravity answers 404 "Requested entity was not found" for every
  `ag/claude-*-5-5*` id (9router 0.5.109, account the Antigravity account) while
  `ag/claude-sonnet-4-6` and `ag/gemini-*` work: the `_ag` roles fall back until
  the account can use 5.5.

