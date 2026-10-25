import { getTask } from "./api";
import { streamTask } from "./sse";
import { isTerminal, normalizeTask, normalizeTaskEvent } from "./tasks";
import type { Task, TaskEvent } from "./types";

const POLL_MS = 3_000;

export interface TaskFollowHandlers {
  onTask: (task: Task) => void;
  onEvent: (event: TaskEvent) => void;
  /** Highest event seq already known, so a stream resumes where it left off. */
  afterSeq: () => number;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { window.clearTimeout(timer); resolve(); }, { once: true });
  });
}

/** Keep a task card live until the task ends: follow `/tasks/{id}/stream`
 * when the backend has it, otherwise poll `GET /tasks/{id}` every 3 s. */
export async function followTask(taskId: string, signal: AbortSignal, handlers: TaskFollowHandlers): Promise<void> {
  try {
    const fresh = await getTask(taskId);
    if (signal.aborted) return;
    if (fresh) {
      handlers.onTask(fresh);
      if (isTerminal(fresh.status)) return;
    }
  } catch {
    // keep going: the stream or the poll below will retry
  }

  try {
    for await (const event of streamTask(taskId, handlers.afterSeq(), signal)) {
      if (signal.aborted) return;
      if (event.event === "task.created" || event.event === "task.updated") {
        const task = normalizeTask(event.data.task);
        if (!task) continue;
        handlers.onTask(task);
        if (isTerminal(task.status)) return;
      } else if (event.event === "task.event") {
        const next = normalizeTaskEvent(event.data);
        if (next) handlers.onEvent(next);
      }
    }
  } catch {
    // No stream endpoint yet (404) or the stream failed: fall back to polling.
    if (signal.aborted) return;
  }

  while (!signal.aborted) {
    await sleep(POLL_MS, signal);
    if (signal.aborted) return;
    try {
      const task = await getTask(taskId);
      if (task) {
        handlers.onTask(task);
        if (isTerminal(task.status)) return;
      }
    } catch {
      // backend briefly unreachable: try again on the next tick
    }
  }
}
