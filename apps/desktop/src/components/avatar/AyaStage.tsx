import { useEffect, useState } from "react";
import { API_BASE_URL, listTasks } from "../../lib/api";
import { speech } from "../../lib/speech";
import {
  ACTIVE_TASK_STATUSES,
  TASK_UPDATED_EVENT,
  isTerminal,
  statusLabel,
  stepProgressText,
  type TaskUpdatedDetail,
} from "../../lib/tasks";
import type { AvatarState, Task } from "../../lib/types";
import { VoicePicker } from "../voice/VoicePicker";
import { Live2DAvatar } from "./Live2DAvatar";

interface Reminder {
  id: string;
  text: string;
  due_at: string;
  repeat: string;
}

interface AyaStageProps {
  state: AvatarState;
  activity: string;
  onCollapse: () => void;
  /** A task of the "Việc đang làm" list was picked. */
  onOpenTask?: (task: Task) => void;
}

const IDLE_LINES: Record<AvatarState, string> = {
  idle: "Cậu cần gì cứ gọi tôi nha~",
  thinking: "Để tôi nghĩ chút…",
  tool: "Tôi đang làm việc…",
  waiting: "Cậu cho phép thao tác này không?",
  talking: "Đang trả lời cậu nè~",
  happy: "Xong rồi nè~ ✨",
  confused: "Ơ… có gì đó không ổn",
};

/** Aya's stage: the Live2D character, a speech bubble with what she is doing
 * and the next reminders. */
export function AyaStage({ state, activity, onCollapse, onOpenTask }: AyaStageProps) {
  const reminders = useUpcomingReminders();
  const tasks = useActiveTasks();
  const [permission, requestPermission] = useNotificationPermission();
  const [japanese, setJapanese] = useState("");
  useEffect(() => speech.subscribe(() => {
    setJapanese(speech.speaking && speech.spokenVoiceId?.startsWith("ja:") ? speech.spokenText : "");
  }), []);
  const line = activity || IDLE_LINES[state];
  return (
    <aside className="aya-stage" aria-label="Aya">
      <header className="aya-stage__header">
        <div>
          <strong>Aya</strong>
          <span className={`aya-stage__dot aya-stage__dot--${state}`} aria-hidden="true" />
        </div>
        <button type="button" className="aya-stage__collapse" onClick={onCollapse} title="Ẩn Aya" aria-label="Ẩn Aya">
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <path d="M8 5l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </header>
      <div className="aya-stage__figure">
        <Live2DAvatar state={state} fill framing={{ zoom: 1.62, offsetY: -0.13 }} />
      </div>
      <div className={`aya-stage__bubble aya-stage__bubble--${state}`} role="status" aria-live="polite">
        {line}
        {japanese && <span className="aya-stage__spoken" lang="ja">{japanese}</span>}
      </div>
      <VoicePicker />
      {tasks.length > 0 && (
        <section className="aya-stage__tasks" aria-label="Việc đang làm">
          <h3>Việc đang làm</h3>
          <ul>
            {tasks.slice(0, 4).map((task) => (
              <li key={task.id}>
                <button type="button" onClick={() => onOpenTask?.(task)} title={task.goal}>
                  <span className="aya-stage__task-goal">{task.goal || "Việc đang làm"}</span>
                  <span className="aya-stage__task-meta">
                    <span className={`task-chip task-chip--${task.status}`}>{statusLabel(task.status)}</span>
                    {stepProgressText(task) && <span>{stepProgressText(task)}</span>}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      <section className="aya-stage__reminders" aria-label="Lời nhắc sắp tới">
        <h3>Lời nhắc sắp tới</h3>
        {permission === "default" && (
          <button type="button" className="aya-stage__notify" onClick={requestPermission}>
            🔔 Bật thông báo trên macOS
          </button>
        )}
        {permission === "denied" && (
          <p className="aya-stage__notify-hint">
            Trình duyệt đang chặn thông báo của Aya. Bấm biểu tượng bên trái thanh địa chỉ → Thông báo → Cho phép.
          </p>
        )}
        {reminders.length === 0 ? (
          <p className="aya-stage__empty">Chưa có lời nhắc nào. Thử “nhắc tớ uống nước lúc 3h chiều” nhé.</p>
        ) : (
          <ul>
            {reminders.slice(0, 4).map((item) => (
              <li key={item.id}>
                <time dateTime={item.due_at}>{formatDue(item.due_at)}</time>
                <span>{item.text}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}

type Permission = NotificationPermission | "unsupported";

/** Browser notification permission; it must be asked from a click, a prompt
 * on page load is silently blocked by Chrome. */
function useNotificationPermission(): [Permission, () => void] {
  const current = (): Permission => ("Notification" in window ? Notification.permission : "unsupported");
  const [permission, setPermission] = useState<Permission>(current);
  useEffect(() => {
    // The user may change it from the address bar while the app is open.
    const timer = window.setInterval(() => setPermission(current()), 5_000);
    return () => window.clearInterval(timer);
  }, []);
  const request = () => {
    if (!("Notification" in window)) return;
    void Notification.requestPermission().then((result) => {
      setPermission(result);
      if (result === "granted") new Notification("Aya", { body: "Thông báo đã bật — lời nhắc sẽ hiện ở đây nè~" });
    });
  };
  return [permission, request];
}

/** Tasks Aya is working on (or that wait for the user), polled every 5 s while
 * there are any and every 30 s otherwise; nothing is fetched while the window
 * is hidden. Task updates seen elsewhere in the app apply immediately. */
function useActiveTasks(): Task[] {
  const [items, setItems] = useState<Task[]>([]);
  const hasAny = items.length > 0;
  useEffect(() => {
    let stopped = false;
    async function load() {
      if (document.hidden) return;
      const results = await Promise.allSettled(
        ACTIVE_TASK_STATUSES.map((status) => listTasks({ status, limit: 20 })),
      );
      if (stopped || results.every((result) => result.status === "rejected")) return;
      const byId = new Map<string, Task>();
      for (const result of results) {
        if (result.status !== "fulfilled") continue;
        for (const task of result.value) {
          if (ACTIVE_TASK_STATUSES.includes(task.status)) byId.set(task.id, task);
        }
      }
      setItems([...byId.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at)));
    }
    void load();
    const timer = window.setInterval(() => void load(), hasAny ? 5_000 : 30_000);
    const onVisible = () => {
      if (!document.hidden) void load();
    };
    const onTaskUpdated = (event: Event) => {
      const task = (event as CustomEvent<TaskUpdatedDetail>).detail?.task;
      if (!task) return;
      setItems((current) => {
        const rest = current.filter((item) => item.id !== task.id);
        return isTerminal(task.status) || !ACTIVE_TASK_STATUSES.includes(task.status)
          ? rest
          : [task, ...rest].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
      });
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener(TASK_UPDATED_EVENT, onTaskUpdated);
    return () => {
      stopped = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener(TASK_UPDATED_EVENT, onTaskUpdated);
    };
  }, [hasAny]);
  return items;
}

function useUpcomingReminders(): Reminder[] {
  const [items, setItems] = useState<Reminder[]>([]);
  useEffect(() => {
    let stopped = false;
    async function load() {
      try {
        const response = await fetch(`${API_BASE_URL}/assistant/reminders`, { signal: AbortSignal.timeout(10_000) });
        if (response.ok && !stopped) setItems((await response.json()).reminders ?? []);
      } catch {
        // backend offline
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 30_000);
    // A new reminder or a fired one changes the list right away.
    const refresh = () => void load();
    window.addEventListener("aya:notification", refresh);
    window.addEventListener("aya:reminders-changed", refresh);
    return () => {
      stopped = true;
      window.clearInterval(timer);
      window.removeEventListener("aya:notification", refresh);
      window.removeEventListener("aya:reminders-changed", refresh);
    };
  }, []);
  return items;
}

function formatDue(value: string): string {
  const date = new Date(value);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  const time = date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
  return sameDay ? `${time} hôm nay` : `${time} ${date.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" })}`;
}
