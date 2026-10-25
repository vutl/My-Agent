import { useEffect, useState } from "react";
import { cancelTask, getTask, pauseTask, resumeTask } from "../../lib/api";
import {
  budgetNotices,
  isTerminal,
  latestError,
  latestNote,
  progressText,
  publishTask,
  statusLabel,
} from "../../lib/tasks";
import type { Task, TaskEvent, TaskStep, ToolCallCard } from "../../lib/types";
import { ToolCalls } from "./ToolCalls";

const STEP_ICON: Record<TaskStep["state"], string> = {
  pending: "○",
  running: "",
  done: "✓",
  failed: "⚠",
  skipped: "—",
};

type TaskAction = "pause" | "resume" | "cancel";

interface TaskCardProps {
  task: Task;
  events?: TaskEvent[];
  /** Tool cards that belong to this task (matched by task id). */
  calls?: ToolCallCard[];
}

export function TaskCard({ task, events, calls = [] }: TaskCardProps) {
  const [pending, setPending] = useState<TaskAction | null>(null);
  const [error, setError] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const terminal = isTerminal(task.status);

  // A finished task folds away; the header and the progress line stay.
  useEffect(() => {
    if (task.status === "done") setCollapsed(true);
  }, [task.status]);

  const run = async (action: TaskAction) => {
    setPending(action);
    setError("");
    try {
      const request = action === "pause" ? pauseTask : action === "resume" ? resumeTask : cancelTask;
      let updated = await request(task.id);
      if (!updated) updated = await getTask(task.id);
      if (updated) publishTask(updated);
    } catch (failure) {
      setError(failure instanceof Error && failure.message ? failure.message : "Không thực hiện được");
    } finally {
      setPending(null);
    }
  };

  const note = terminal ? "" : latestNote(events);
  const notices = budgetNotices(events);
  const failure = task.status === "failed" ? latestError(events) : "";
  const taskLevelCalls = calls.filter((call) => !call.stepId || !task.plan.some((step) => step.id === call.stepId));

  return (
    <section className={`task-card task-card--${task.status}`} data-task-id={task.id} aria-label="Việc Aya đang làm">
      <header className="task-card__head">
        <span className="task-card__goal" title={task.goal}>{task.goal || "Việc đang làm"}</span>
        <span className={`task-chip task-chip--${task.status}`}>{statusLabel(task.status)}</span>
        {terminal ? (
          <button
            type="button"
            className="task-card__toggle"
            aria-expanded={!collapsed}
            title={collapsed ? "Mở rộng" : "Thu gọn"}
            onClick={() => setCollapsed((value) => !value)}
          >
            {collapsed ? "▸" : "▾"}
          </button>
        ) : null}
      </header>
      <p className="task-card__progress">{progressText(task)}</p>
      {collapsed ? null : (
        <>
          {task.plan.length > 0 ? (
            <ol className="task-card__plan">
              {task.plan.map((step) => {
                const stepCalls = calls.filter((call) => call.stepId === step.id);
                return (
                  <li key={step.id} className={`task-step task-step--${step.state}`}>
                    <span className="task-step__icon" aria-hidden="true">{STEP_ICON[step.state]}</span>
                    <span className="task-step__title">{step.title}</span>
                    {stepCalls.length > 0 ? (
                      <div className="task-step__calls"><ToolCalls calls={stepCalls} /></div>
                    ) : null}
                  </li>
                );
              })}
            </ol>
          ) : null}
          {taskLevelCalls.length > 0 ? <ToolCalls calls={taskLevelCalls} /> : null}
          {note ? <p className="task-card__note" title={note}>{note}</p> : null}
          {notices.map((notice) => (
            <p key={notice} className="task-card__notice">⚠ {notice}</p>
          ))}
          {failure ? <p className="task-card__notice">⚠ {failure}</p> : null}
          {terminal && task.summary ? <p className="task-card__summary">{task.summary}</p> : null}
        </>
      )}
      {error ? <p className="task-card__notice">⚠ {error}</p> : null}
      {terminal ? null : (
        <div className="task-card__actions">
          {task.status === "paused" ? (
            <button type="button" disabled={pending !== null} onClick={() => void run("resume")}>
              Tiếp tục
            </button>
          ) : (
            <button type="button" disabled={pending !== null} onClick={() => void run("pause")}>
              Tạm dừng
            </button>
          )}
          <button type="button" className="task-card__cancel" disabled={pending !== null} onClick={() => void run("cancel")}>
            Huỷ
          </button>
        </div>
      )}
    </section>
  );
}
