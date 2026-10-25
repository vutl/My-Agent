import type { ChatMessage, Task, TaskEvent, TaskStatus, TaskStep, TaskStepState } from "./types";

/** Per-task event list kept in the UI; older events are dropped. */
export const MAX_TASK_EVENTS = 200;

export const TASK_STATUSES: TaskStatus[] = [
  "planning",
  "running",
  "waiting_user",
  "waiting_approval",
  "paused",
  "done",
  "failed",
  "cancelled",
];
const STEP_STATES: TaskStepState[] = ["pending", "running", "done", "skipped", "failed"];

const STATUS_LABEL: Record<TaskStatus, string> = {
  planning: "Đang lên kế hoạch",
  running: "Đang làm",
  waiting_user: "Chờ cậu trả lời",
  waiting_approval: "Chờ cậu cho phép",
  paused: "Tạm dừng",
  done: "Xong",
  failed: "Gặp lỗi",
  cancelled: "Đã huỷ",
};

export function statusLabel(status: TaskStatus): string {
  return STATUS_LABEL[status] ?? status;
}

export function isTerminal(status: TaskStatus): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

/** Tasks the "Việc đang làm" list shows. */
export const ACTIVE_TASK_STATUSES: TaskStatus[] = ["running", "waiting_user", "waiting_approval", "paused"];

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function count(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function normalizeStep(raw: unknown, index: number): TaskStep {
  const step = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const state = STEP_STATES.includes(step.state as TaskStepState) ? (step.state as TaskStepState) : "pending";
  return {
    id: text(step.id) || String(index + 1),
    title: text(step.title),
    tools_hint: Array.isArray(step.tools_hint) ? step.tools_hint.filter((item) => typeof item === "string") : [],
    done_when: text(step.done_when),
    state,
  };
}

/** Accept whatever the backend sent as a task; missing fields get neutral
 * defaults so a partial payload never breaks rendering. Null without an id. */
export function normalizeTask(raw: unknown): Task | null {
  if (!raw || typeof raw !== "object") return null;
  const task = raw as Record<string, unknown>;
  const id = text(task.id);
  if (!id) return null;
  const usage = (task.usage && typeof task.usage === "object" ? task.usage : {}) as Record<string, unknown>;
  const budget = (task.budget && typeof task.budget === "object" ? task.budget : {}) as Record<string, unknown>;
  return {
    id,
    conversation_id: text(task.conversation_id),
    goal: text(task.goal),
    status: TASK_STATUSES.includes(task.status as TaskStatus) ? (task.status as TaskStatus) : "running",
    plan: Array.isArray(task.plan) ? task.plan.map(normalizeStep) : [],
    usage: { steps: count(usage.steps), tool_calls: count(usage.tool_calls), seconds: count(usage.seconds) },
    budget: {
      max_steps: count(budget.max_steps),
      max_tool_calls: count(budget.max_tool_calls),
      max_seconds: count(budget.max_seconds),
    },
    summary: text(task.summary),
    created_at: text(task.created_at),
    updated_at: text(task.updated_at),
  };
}

/** A list endpoint may answer with a bare array or `{tasks: [...]}`. */
export function normalizeTaskList(raw: unknown): Task[] {
  const items = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object" && Array.isArray((raw as { tasks?: unknown }).tasks)
      ? (raw as { tasks: unknown[] }).tasks
      : [];
  return items.map(normalizeTask).filter((task): task is Task => task !== null);
}

export function normalizeTaskEvent(raw: unknown): TaskEvent | null {
  if (!raw || typeof raw !== "object") return null;
  const event = raw as Record<string, unknown>;
  if (typeof event.kind !== "string" || typeof event.seq !== "number") return null;
  return { task_id: text(event.task_id), seq: event.seq, kind: event.kind, payload: event.payload };
}

export function formatSeconds(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} giây`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return rest ? `${minutes} phút ${rest} giây` : `${minutes} phút`;
}

/** The step number the progress line points at: the running step, otherwise
 * the next one to do (the last one when everything is finished). */
export function currentStepNumber(task: Task): number {
  const total = task.plan.length;
  const running = task.plan.findIndex((step) => step.state === "running");
  if (running >= 0) return running + 1;
  const finished = task.plan.filter((step) => step.state !== "pending").length;
  return Math.min(total, finished + (isTerminal(task.status) ? 0 : 1));
}

/** "Bước 2/3 · 4 lần dùng công cụ · 18 giây" */
export function progressText(task: Task): string {
  const parts: string[] = [];
  if (task.plan.length > 0) parts.push(`Bước ${currentStepNumber(task)}/${task.plan.length}`);
  parts.push(`${task.usage.tool_calls} lần dùng công cụ`);
  parts.push(formatSeconds(task.usage.seconds));
  return parts.join(" · ");
}

/** Short progress for the stage list: "2/3 bước". */
export function stepProgressText(task: Task): string {
  if (task.plan.length === 0) return "";
  const done = task.plan.filter((step) => step.state === "done" || step.state === "skipped").length;
  return `${done}/${task.plan.length} bước`;
}

/** Append an event; duplicates (same seq, e.g. after a stream reconnect) are
 * dropped, order follows seq and the list keeps the newest MAX_TASK_EVENTS. */
export function appendTaskEvent(events: TaskEvent[] | undefined, event: TaskEvent, max = MAX_TASK_EVENTS): TaskEvent[] {
  const current = events ?? [];
  if (current.some((item) => item.seq === event.seq && item.task_id === event.task_id)) return current;
  const next = [...current, event];
  if (current.length && current[current.length - 1].seq > event.seq) next.sort((a, b) => a.seq - b.seq);
  return next.length > max ? next.slice(next.length - max) : next;
}

export function lastSeq(events: TaskEvent[] | undefined): number {
  return events && events.length ? events[events.length - 1].seq : 0;
}

function payloadText(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return "";
  const record = payload as Record<string, unknown>;
  for (const key of ["note", "text", "summary", "message", "result", "reason"]) {
    if (typeof record[key] === "string" && record[key]) return record[key] as string;
  }
  return "";
}

function oneLine(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

/** The latest note written while working: from `note` / `step_done` events. */
export function latestNote(events: TaskEvent[] | undefined): string {
  if (!events) return "";
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.kind !== "note" && event.kind !== "step_done") continue;
    const line = oneLine(payloadText(event.payload));
    if (line) return line;
  }
  return "";
}

/** Budget warnings and errors, newest last, without repeats. */
export function budgetNotices(events: TaskEvent[] | undefined, limit = 3): string[] {
  const seen: string[] = [];
  for (const event of events ?? []) {
    if (event.kind !== "budget") continue;
    const line = oneLine(payloadText(event.payload));
    if (line && !seen.includes(line)) seen.push(line);
  }
  return seen.slice(-limit);
}

export function latestError(events: TaskEvent[] | undefined): string {
  if (!events) return "";
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].kind !== "error") continue;
    const line = oneLine(payloadText(events[index].payload));
    if (line) return line;
  }
  return "";
}

/** Message reducers: apply a task snapshot / event to the message that owns it. */
export function withTask(message: ChatMessage, task: Task): ChatMessage {
  return { ...message, task };
}

export function withTaskEvent(message: ChatMessage, event: TaskEvent): ChatMessage {
  if (message.task && event.task_id && message.task.id !== event.task_id) return message;
  const events = appendTaskEvent(message.taskEvents, event);
  return events === message.taskEvents ? message : { ...message, taskEvents: events };
}

/** The assistant message a stored task belongs to when a conversation is
 * reopened: the first reply written after the task was created, else the last. */
export function messageIndexForTask(messages: ChatMessage[], task: Task): number {
  const created = Date.parse(task.created_at);
  let last = -1;
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    last = index;
    const at = message.createdAt ? Date.parse(message.createdAt) : NaN;
    if (!Number.isNaN(created) && !Number.isNaN(at) && at >= created) return index;
  }
  return last;
}

export function openAsk(messages: ChatMessage[]): { question: string } | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const ask = messages[index].ask;
    if (ask && !ask.answered) return { question: ask.question };
  }
  return null;
}

/** Window event other parts of the app (stage list, chat) use to share a task
 * snapshot without refetching. */
export const TASK_UPDATED_EVENT = "aya:task-updated";

export interface TaskUpdatedDetail {
  task: Task;
  source?: string;
}

export function publishTask(task: Task, source?: string): void {
  window.dispatchEvent(new CustomEvent<TaskUpdatedDetail>(TASK_UPDATED_EVENT, { detail: { task, source } }));
}
