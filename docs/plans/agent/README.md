# Aya as an agent that does everything — plan set

Written 2026-10-25 after RAG v1, the assistant tool loop, reminders/macOS
integrations, the Live2D stage and the voice sidecar shipped. These files are
the working plan: agents (Codex/Claude through the harness) implement them one
work package at a time; the head agent keeps them current.

| File | What it decides |
|---|---|
| [00-ROADMAP.md](00-ROADMAP.md) | What "does everything" means, the gap from today, phases and their gates |
| [01-AGENT-CORE.md](01-AGENT-CORE.md) | Tasks, planning, background runs, sub-agents, context, capability awareness |
| [02-TOOLS.md](02-TOOLS.md) | Tool catalog: what exists, what to add, in which order, with which permission |
| [03-SKILLS-AND-WORKFLOWS.md](03-SKILLS-AND-WORKFLOWS.md) | Skills (packaged procedures), triggers and scheduled workflows |
| [04-SAFETY.md](04-SAFETY.md) | Threat model, permissions, approvals, audit, undo |
| [05-PRESENCE.md](05-PRESENCE.md) | Avatar and voice: what is done, what is next (emotion, interruptions, voice input) |
| [06-EVAL.md](06-EVAL.md) | How we know it works: agent task suite, metrics, gates |
| [07-EXECUTION.md](07-EXECUTION.md) | Work packages, who builds each (harness roles), order, rollout |
| [08-DEEP-RESEARCH.md](08-DEEP-RESEARCH.md) | Deep research mode (ChatGPT/Gemini/Claude level): plan, parallel workers, evidence cards, verified cited report, sources as a collection |

Rules that hold across all files (from AGENTS.md and past lessons):

- Aya's models stay as approved: answers/verification `cx/gpt-6.1-sol`, router
  `cx/gpt-6-luna`; `VISION_MODEL` stays `cx/gpt-5.6-sol`. No local Ollama for
  answers or routing.
- Document questions keep the measured RAG pipeline; no regression on its gates.
- Tool output is untrusted data; side effects after untrusted content need approval.
- Generic code only: no hardcoded per-question/per-paper behaviour, no fabricated numbers.
- Every phase ends with a measured gate; production rollouts get a backup and a
  rollback drill.
