export type TaskStatus =
  | "planning"
  | "running"
  | "waiting_user"
  | "waiting_approval"
  | "paused"
  | "done"
  | "failed"
  | "cancelled";

export type TaskStepState = "pending" | "running" | "done" | "skipped" | "failed";

export interface TaskStep {
  id: string;
  title: string;
  tools_hint: string[];
  done_when: string;
  state: TaskStepState;
}

export interface Task {
  id: string;
  conversation_id: string;
  goal: string;
  status: TaskStatus;
  plan: TaskStep[];
  usage: { steps: number; tool_calls: number; seconds: number };
  budget: { max_steps: number; max_tool_calls: number; max_seconds: number };
  summary: string;
  created_at: string;
  updated_at: string;
}

export type TaskEventKind =
  | "plan"
  | "plan_edit"
  | "step_started"
  | "step_done"
  | "tool_call"
  | "tool_result"
  | "note"
  | "ask"
  | "answer"
  | "approval"
  | "error"
  | "budget"
  | "status";

export interface TaskEvent {
  task_id: string;
  seq: number;
  // Kinds the app does not know yet arrive as plain strings and are ignored.
  kind: TaskEventKind | (string & {});
  payload: unknown;
}

/** A question Aya asked the user while working on a task. */
export interface AskUserCard {
  question: string;
  options: string[];
  reason?: string;
  taskId?: string;
  answered?: string;
}
