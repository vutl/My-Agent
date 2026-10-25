# 07 — Execution: work packages and who builds them

The head agent (Claude in the user's session) owns the plan, writes each work
package as an approved plan, dispatches it through `tools/harness/pipeline.py`,
reads every diff, runs the gates and merges. Roles (`tools/harness/harness.toml`):

| Work | Role (fallback) |
|---|---|
| Plan a package | head agent, or `planner_ag` (Opus 5.5 via Antigravity) → `planner_codex` |
| Attack the plan | `plan_review` (gpt-6-astra) |
| Multi-file implementation + tests | `executor` (gpt-6.1-sol) or `executor_ag` (Opus 5.5) |
| Bounded edits, tests, fixtures | `coder` (gpt-6-luna) or `coder_ag` (Sonnet 5.5) |
| Review the diff | `reviewer` (astra) or `reviewer_ag` |
| Judge eval runs | `judge_ag` → `judge` |

Two packages run in parallel when they touch different files (one per Codex
account, a third on Antigravity). Each package names its test command
(`pipeline.py --tests`). Codex sandboxes have no network: the head agent does
installs, model downloads and live smoke tests.

## P1 packages

| # | Package | Main files | Parallel with |
|---|---|---|---|
| P1.1 | Task tables, service, API, SSE events | `db/sqlite.py`, `services/tasks.py`, `api/tasks.py` | P1.5 |
| P1.2 | Plan/execute/replan around the tool loop | `agents/task_runner.py`, `agents/tool_loop.py` | after P1.1 |
| P1.3 | Background runner + resume + notifications | `main.py` lifespan, `services/tasks.py` | after P1.2 |
| P1.4 | Budgets, rate limits, argument validation/repair | `agents/`, `tools/registry.py` | after P1.2 |
| P1.5 | Capability sheet in every route; router fix for tool intents | `agents/graph.py`, `services/tool_decision_service.py` | P1.1 |
| P1.6 | `ask_user`, `remember`/`recall`, `notify`, `read_artifact`, artifact store | `tools/` | after P1.1 |
| P1.7 | Task card + "Việc đang làm" on the stage; streamed tool-call deltas | `apps/desktop` | after P1.1 API |
| P1.8 | Agent suite v1 fixtures + runner + judge | `evals/agent/`, `scripts/` | from the start (author ≠ implementer, sealed) |

## Later phases

P2: one package per tool group in the order of 02-TOOLS.md, each with its safety
row from 04-SAFETY.md. P3: skill loader, then one package per skill. P4: one
package per item of 05-PRESENCE.md. P5: audit/undo, token + doctor, packaging.

## Rollout

Isolated stack (:7790/:5174) first; gate; backup `data/backups/pre-<phase>-<date>`;
rollback drill (`scripts/release_rollback_drill.sh`); production restart; smoke;
update `docs/handoff/00-STATUS.md`, `04-DONE.md`, `05-TODO-AND-ISSUES.md`.
