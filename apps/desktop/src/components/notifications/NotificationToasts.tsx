import { useEffect, useState } from "react";
import { API_BASE_URL } from "../../lib/api";

interface AssistantNotification {
  id: string;
  kind: string;
  title: string;
  body: string;
  created_at: string;
  conversation_id?: string | null;
}

const POLL_MS = 15_000;
// Closed toasts stay closed even if telling the backend failed; the ids are
// re-sent on every poll until it accepts them.
const DISMISSED_KEY = "aya.dismissed_notifications";

function readDismissed(): string[] {
  try {
    const saved = JSON.parse(localStorage.getItem(DISMISSED_KEY) ?? "[]");
    return Array.isArray(saved) ? saved.filter((id) => typeof id === "string") : [];
  } catch {
    return [];
  }
}

function writeDismissed(ids: string[]) {
  try {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify(ids.slice(-200)));
  } catch {
    // storage unavailable: the backend read mark is the only record
  }
}

async function markRead(ids: string[]): Promise<boolean> {
  if (ids.length === 0) return true;
  try {
    const response = await fetch(`${API_BASE_URL}/assistant/notifications/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/** Reminders and other assistant notifications: a toast that stays until the
 * user closes it (only then is it marked read, so a missed one comes back
 * after a reload), a system notification when the user allowed them, and an
 * "aya:notification" window event the chat uses to react. */
export function NotificationToasts() {
  const [items, setItems] = useState<AssistantNotification[]>([]);

  useEffect(() => {
    let stopped = false;
    const seen = new Set<string>();
    async function poll() {
      const dismissed = readDismissed();
      // Retry read marks the backend has not taken yet.
      if (dismissed.length > 0 && (await markRead(dismissed))) writeDismissed([]);
      try {
        const response = await fetch(`${API_BASE_URL}/assistant/notifications`);
        if (!response.ok) return;
        const fresh: AssistantNotification[] = (await response.json()).notifications ?? [];
        const closed = new Set(readDismissed());
        const unseen = fresh.filter((item) => !seen.has(item.id) && !closed.has(item.id));
        if (stopped || unseen.length === 0) return;
        unseen.forEach((item) => seen.add(item.id));
        setItems((current) => [...current, ...unseen]);
        for (const item of unseen) {
          if ("Notification" in window && Notification.permission === "granted") {
            // Lands in macOS Notification Center (as the browser) and stays
            // until clicked; clicking brings Aya to the front.
            const system = new Notification(item.title, { body: item.body, tag: item.id, requireInteraction: true });
            system.onclick = () => {
              window.focus();
              system.close();
            };
          }
          window.dispatchEvent(new CustomEvent("aya:notification", { detail: item }));
        }
      } catch {
        // Backend offline: try again on the next tick.
      }
    }
    void poll();
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);

  function dismiss(ids: string[]) {
    setItems((current) => current.filter((old) => !ids.includes(old.id)));
    writeDismissed([...readDismissed(), ...ids]);
    void markRead(ids).then((ok) => {
      if (ok) writeDismissed(readDismissed().filter((id) => !ids.includes(id)));
    });
  }

  if (items.length === 0) return null;
  return (
    <div className="aya-toasts" role="status" aria-live="polite">
      {items.length > 1 && (
        <button type="button" className="aya-toasts__clear" onClick={() => dismiss(items.map((item) => item.id))}>
          Đóng tất cả ({items.length})
        </button>
      )}
      {items.map((item) => (
        <div key={item.id} className="aya-toast">
          <span className="aya-toast__icon" aria-hidden="true">⏰</span>
          <div>
            <strong>{item.title}</strong>
            <p>{item.body}</p>
          </div>
          <button type="button" className="aya-toast__close" aria-label="Đóng" title="Đóng" onClick={() => dismiss([item.id])}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
