import { useEffect, useState } from "react";
import { API_BASE_URL } from "../../lib/api";

interface AssistantNotification {
  id: string;
  kind: string;
  title: string;
  body: string;
  created_at: string;
}

const POLL_MS = 15_000;
const SHOW_MS = 12_000;

/** Reminders and other assistant notifications: a toast in the app and a
 * system notification when the user has allowed them. */
export function NotificationToasts() {
  const [items, setItems] = useState<AssistantNotification[]>([]);

  useEffect(() => {
    let stopped = false;
    async function poll() {
      try {
        const response = await fetch(`${API_BASE_URL}/assistant/notifications`);
        if (!response.ok) return;
        const fresh: AssistantNotification[] = (await response.json()).notifications ?? [];
        if (stopped || fresh.length === 0) return;
        setItems((current) => [...current, ...fresh.filter((item) => !current.some((old) => old.id === item.id))]);
        for (const item of fresh) {
          if ("Notification" in window && Notification.permission === "granted") {
            new Notification(item.title, { body: item.body });
          }
          window.setTimeout(() => setItems((current) => current.filter((old) => old.id !== item.id)), SHOW_MS);
        }
        await fetch(`${API_BASE_URL}/assistant/notifications/read`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids: fresh.map((item) => item.id) }),
        });
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
          <button type="button" aria-label="Đóng" onClick={() => setItems((current) => current.filter((old) => old.id !== item.id))}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
