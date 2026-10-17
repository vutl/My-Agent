---
name: agent-harness
description: Dispatch development work to the right model through the repo's agent harness (Fable plans, gpt-6-astra reviews plans and diffs and judges evals, gpt-6.1-sol executes in a git worktree, gpt-6-luna / Sonnet / Haiku code and scout) using two ChatGPT accounts with usage-limit failover. Use when splitting a task between agents, running plan→execute→review, batch-judging, or checking Codex account quota.
---

# Agent harness

Design and roles: `docs/handoff/10-AGENT-HARNESS.md`. Config: `tools/harness/harness.toml`.
All calls go straight to `codex exec` / `claude -p`; never through 9router
(9router is Aya's runtime gateway only).

## Commands

```bash
python tools/harness/cx.py status                       # accounts, cooldowns, recent calls
python tools/harness/cx.py run <role> --prompt-file p.md [--schema s.json] [--cwd dir]
python tools/harness/pipeline.py --task task.md         # plan → review → execute → test → review
python tools/harness/pipeline.py --task task.md --plan approved.md --executor coder
python tools/harness/judge.py --set-dir <dir> --answers a.jsonl --out judged.jsonl
tools/harness/sync_skills.sh                            # link .agents/skills into Claude and Codex
```

## Choosing a role

| Work | Role |
|---|---|
| Turn a goal into a plan | planner (Fable); `planner_codex` when Claude quota is out |
| Attack a plan before coding | plan_review (gpt-6-astra) |
| Multi-file change with tests | executor (gpt-6.1-sol, worktree `harness/<run>`) |
| Bounded edit, tests, scripts | coder (gpt-6-luna) or coder_claude (Sonnet) |
| Review a diff | reviewer (gpt-6-astra) |
| Grade eval answers | judge (gpt-6-astra) |
| Find files / facts cheaply | scout (Haiku) |

## Rules

- The head agent reads every diff and test result before merging
  `harness/<run>` into main; nothing merges itself.
- Give each role the repo rules: AGENTS.md, generic fixes only, no fabricated
  numbers, Aya models stay `cx/gpt-5.6-*`, tests must pass.
- Quota: Codex accounts rotate least-recently-used and cool down on usage
  limits (`.harness/state.json`, log `.harness/calls.jsonl`). Claude quota is
  scarcer: keep it for planning and the head role.
- Second account login (once): `CODEX_HOME=~/.codex-uclab codex login --device-auth`
  (codex binary: `python -c "import sys; sys.path.insert(0,'tools/harness'); import core; print(core.binary('codex'))"`).
