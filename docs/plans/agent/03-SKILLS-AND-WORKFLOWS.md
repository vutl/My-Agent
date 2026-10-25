# 03 — Skills and workflows

A **skill** is a packaged procedure Aya follows for a kind of job; a
**workflow** is a skill (or goal) that runs on a trigger.

## Skill format

`skills/<name>/SKILL.md` (first-party, in the repo) and `~/Aya/skills/<name>/SKILL.md`
(user skills, enabled only after the user reads and approves them):

```markdown
---
name: research
description: One line used to pick the skill.
when: phrases/intents that should trigger it
tools: [web_search, web_fetch, make_document]
budget: {steps: 25, minutes: 15}
output: cited brief in the chat + optional docx in ~/Aya/reports
---
Steps, checks, output format, what to ask the user, what never to do.
```

- Loader `services/skills.py`: parse, validate tools against the registry, keep a
  hash; a changed user skill must be re-approved.
- Selection: the capability sheet lists skills; the planner picks one (or the user
  names it: "/research …"); the skill text becomes the task's plan template and
  its tool list restricts the task.

## First-party skills (P3)

| Skill | Job | Acceptance task |
|---|---|---|
| research | question → search → read 3–6 sources → cited brief (+ docx) | brief with ≥ 3 distinct cited sources, no uncited numbers |
| daily-brief | morning: weather, calendar, reminders, unread important mail, followed topics → spoken + Telegram | runs on schedule 3 days; items match the sources |
| paper-digest | new document in the library → summary, key numbers (grounded), questions | digest numbers all traced to [S#] |
| inbox-triage | summarise unread mail, propose replies (drafts only), flag urgent | drafts created, nothing sent |
| meeting-prep | next calendar event → attendees, related mail/docs, agenda draft | prep note in Notes/workspace |
| file-organizer | tidy a folder by rules, plan shown, approval per batch, undo log | moves reversible, plan approved |
| data-analyst | CSV/Excel → pandas → chart + findings | chart file + stated numbers match the data |
| writer | draft/edit documents in the user's style (from memory) | docx produced, user edits remembered |

## Triggers and workflows

- Time (`schedule_task`, cron-like, local timezone) — daily brief, weekly review.
- Folder watch (`~/Aya/inbox`): new file → paper-digest / file-organizer.
- Mail rule: new mail matching a rule → inbox-triage for that thread.
- Calendar: 15 min before an event → meeting-prep.
- Each workflow is a task with `parent=workflow id`; runs are listed on the stage
  ("Việc đang làm") and in a Workflows page (enable/disable, last runs, logs).

## Learning

After a task the user corrects, Aya proposes a memory ("next time, put reports in
~/Aya/reports") or a skill edit; both need the user's OK.
