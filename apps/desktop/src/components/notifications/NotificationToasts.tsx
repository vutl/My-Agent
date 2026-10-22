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

/** Reminders and other assistant notifications: a toast that stays until the
 * user closes it (only then is it marked read, so a missed one comes back
 * after a reload), a system notification when allowed, and an
 * "aya:notification" window event the chat uses to react. */
export function NotificationToasts() {
  const [items, setItems] = useState<AssistantNotification[]>([]);

  useEffect(() => {
    let stopped = false;
    const seen = new Set<string>();
    async function poll() {
      try {
        const response = await fetch(`${API_BASE_URL}/assistant/notifications`);
        if (!response.ok) return;
        const fresh: AssistantNotification[] = (await response.json()).notifications ?? [];
        if (stopped || fresh.length === 0) return;
        const unseen = fresh.filter((item) => !seen.has(item.id));
        if (unseen.length === 0) return;
        unseen.forEach((item) => seen.add(item.id));
        setItems((current) => [...current, ...unseen]);
        for (const item of unseen) {
          if ("Notification" in window && Notification.permission === "granted") {
            new Notification(item.title, { body: item.body });
          }
          window.dispatchEvent(new CustomEvent("aya:notification", { detail: item }));
        }
      } catch {
        // Backend offline: try again on the next tick.
      }
    }
    if ("Notification" in window && Notification.permission === "default") {
      void Notification.requestPermission();
    }
    void poll();
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);

  function dismiss(id: string) {
    setItems((current) => current.filter((old) => old.id !== id));
    void fetch(`${API_BASE_URL}/assistant/notifications/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [id] }),
    }).catch(() => undefined);
  }

  if (items.length === 0) return null;
  return (
    <div className="aya-toasts" role="status" aria-live="polite">
      {items.map((item) => (
        <div key={item.id} className="aya-toast">
          <span className="aya-toast__icon" aria-hidden="true">⏰</span>
          <div>
            <strong>{item.title}</strong>
            <p>{item.body}</p>
          </div>
          <button type="button" aria-label="Đóng" onClick={() => dismiss(item.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
