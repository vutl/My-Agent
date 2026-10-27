# 01 — Agent core

Today: `api/agent/assistant.py` → `agents/tool_loop.run_tool_loop` runs one
turn: the model calls tools until it answers or hits the step cap. That is
enough for "set a reminder" but not for "plan my week", "research X and write a
report", or anything that outlives one HTTP stream.

## 1. Tasks (persistent goals)

- SQLite tables (additive migration in `db/sqlite.py`):
  - `tasks(id, conversation_id, parent_id, goal, status[planning|running|waiting_user|waiting_approval|paused|done|failed|cancelled], plan_json, summary, budget_json, usage_json, lease_until, created_at, updated_at)`
  - `task_events(id, task_id, seq, kind[plan|plan_edit|step_started|step_done|tool_call|tool_result|note|ask|answer|approval|error|budget|status], payload_json, created_at)`
- `services/tasks.py`: create, transition (guarded state machine), append event,
  load for resume. Tool calls keep using `tool_calls` and reference `task_id`.
- API: `GET /tasks`, `GET /tasks/{id}` (+ all ordered events), `POST /tasks`,
  `POST /tasks/{id}/cancel|pause|resume` (resume requires paused).
- Chat SSE contract (emission is added in a later package):
  - `ask.user {question, options, reason}` ends a tool turn. Without a task the
    question is the answer text. Inside a task step the runner commits an `ask`
    event `{step_id, question, options, reason}` with `running → waiting_user`
    (the step stays `running`) and the stream ends with the question as the answer.
  - The next message in the same conversation is triaged only against the latest
    task, and only if it is `waiting_user` with an unanswered `ask`. The router
    model returns `{"resume": bool}`; if that fails, a question without options
    resumes (free text) and one with options resumes only on a matching option. A
    clearly new request follows normal routing and leaves the task waiting.
  - Resume commits `answer {text, ask_seq}` with `waiting_user → running`; the worker
    continues the same task from its persisted state: the current step is retried
    with `user_questions_and_answers` and the earlier `tool_result` events (artifact
    handles included), finished steps are kept, and tools that already ran are not
    re-run. The chat stream attaches like to any running task. `answer` events
    carrying `ask_seq` are the user's reply, never a report.
  - `task.created {task}` after creation.
  - `task.updated {task}` after a persisted task change.
  - `task.event {task_id, seq, kind, payload}` after an event is committed;
    `seq` increases within each task and is the cursor for replay/deduplication.
  - `task` is `Task.to_dict()`: parsed `plan`, `budget`, `usage`, nullable
    `conversation_id`, `parent_id`, `summary`, `lease_until`, and UTC timestamps.
- P1.1 provides storage/API only. Defaults: 20 steps, 40 tool calls, 900 seconds;
  positive integer budget overrides apply per key. `add_usage` reports strictly
  exceeding a limit and leaves state changes to the later runner. Claims lease
  up to 20 running tasks atomically; waiting/planning/terminal tasks are excluded.
- UI: a task card in the chat (goal, plan checklist with live step state,
  artifacts, cancel/pause) and a "Việc đang làm" list on Aya's stage.

## 2. Plan → execute → replan

- Planning prompt (cx/gpt-6.1-sol): for goals that need ≥ 3 steps or ≥ 2 tools,
  produce a short checklist (`[{id, title, tools_hint, done_when}]`), shown to
  the user before risky steps. Small requests skip planning (today's behaviour).
- Executor: runs one plan step at a time through the existing tool loop with
  the step's context (goal, plan, step, notes so far), records events, marks
  the step done when its `done_when` holds (model judgement + tool evidence).
- Replan: on failure, missing information or a new finding, the model may edit
  the remaining steps (recorded as an event); hard cap on replans.
- Final report: what was done, artifacts (files, links, events), what was not
  done and why. Never claim an action without a matching tool event.

## 3. Background runs that survive restarts

- A task runner in the lifespan (lease pattern like the reminder worker): picks
  `running` tasks without a live lease, resumes from the last event. Each step
  is idempotent or guarded (writes go through tools that record what they did).
- The chat stream attaches to a running task; closing the UI does not stop it.
- On completion: notification (macOS + in-app + Telegram if configured), a
  message in the conversation, and Aya's end reaction.

## 4. Budgets and limits

- Per task: max steps, max tool calls, max wall time, max model tokens
  (estimated), max spend of 9router calls; defaults by task size; the user can
  raise them on the card. Hitting a budget → `waiting_user` with a summary.
- Per tool: rate limits (e.g. web fetch per minute) and concurrency caps.

## 5. Sub-agents for parallel work

- `spawn_subtask(goal, tools, budget)` for fan-out work (read 5 sources, try 3
  approaches). Children run with a restricted tool set and return a compact
  result (≤ 2k tokens) to the parent; their events nest under the parent task.
- Model: cx/gpt-6.1-sol; summarisation of long tool output with cx/gpt-6-luna.

## 6. Context management

- Tool results over 6,000 characters (after the source labels are rendered) are stored
  at `<workspace>/.aya_runs/artifacts/<run_id>/<call_id>.txt`; the model gets the
  handle, the first 1,500 characters, the length and one summary line, and reads
  ranges with `read_artifact(handle, offset, length<=6000)` (traversal/symlinks are
  rejected). Handles are recorded on `tool_call` completion cards and `tool_result`
  task events, so a resumed step can read them. If storage fails the result, receipts
  and control data (`ask_user`) are kept and a separate `artifact_error` is reported.
- The task context = goal + plan + step notes + last N events + relevant memory;
  the full transcript stays in `task_events`.

## 7. Capability awareness

- The system prompt of every route (chat, RAG, agent) gets a short, generated
  capability sheet from the registry and the skill list (names + one line), so
  "cậu làm được gì?" lists real tools and skills.
- Router: a request that needs a tool must route to `agent` even when phrased
  as small talk ("mở nhạc đi", "nhắc tớ…").

## 8. Interaction primitives

- `ask_user(question, options?)`: structured question card; the task waits in
  `waiting_user`; the answer resumes it.
- `remember(fact, scope, key?, kind?)` / `recall(query, limit?)`: thin tools over
  long-term memory (L3) with provenance; generated keys carry a digest of the whole
  fact; sensitive values are refused.
- `forget(key, scope)`: always asks approval with a preview of the exact active
  versions; the approval is bound to those version ids and a stale set asks again.
- `notify(text, title?, channel?)`: explicit app/macOS/Telegram notifications, at
  most five per task (durable) or per turn. macOS counts as delivered only when
  osascript succeeds.

## 9. Known issues to fix in this phase

- Junk appended to function arguments by the model/gateway: validate JSON
  arguments against the schema, repair or re-ask once with the validation error.
- Stream tool-call deltas so a long first tool call shows its card immediately.

## Tests

State machine transitions; resume after simulated crash (kill mid-step, restart,
finish); budget stops; plan edit events; ask_user round trip; capability sheet
contents match the registry; no route answers "I can't" for a registered tool.
