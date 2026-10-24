import { useEffect, useState } from "react";

import { ChatPage } from "./routes/ChatPage";
import { FilesPage } from "./routes/FilesPage";
import { Sidebar } from "./components/layout/Sidebar";
import { Topbar } from "./components/layout/Topbar";
import { NotificationToasts } from "./components/notifications/NotificationToasts";

export type AppRoute = "chat" | "library";

// Whether Aya's stage is hidden; remembered across reloads and chat events,
// and kept in step between open windows.
const STAGE_COLLAPSED_STORAGE_KEY = "aya.stage_collapsed";

function useStageCollapsed(): [boolean, (collapsed: boolean) => void] {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STAGE_COLLAPSED_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === STAGE_COLLAPSED_STORAGE_KEY) setCollapsed(event.newValue === "1");
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);
  const update = (next: boolean) => {
    setCollapsed(next);
    try {
      localStorage.setItem(STAGE_COLLAPSED_STORAGE_KEY, next ? "1" : "0");
    } catch {
      // storage unavailable: only this session remembers
    }
  };
  return [collapsed, update];
}

function App() {
  const [route, setRoute] = useState<AppRoute>("chat");
  const [model, setModel] = useState("cx/gpt-6.1-sol");
  const [historySlot, setHistorySlot] = useState<HTMLElement | null>(null);
  const [stageCollapsed, setStageCollapsed] = useStageCollapsed();

  return (
    <div className="app-shell">
      <Sidebar activeRoute={route} onRouteChange={setRoute} historySlotRef={setHistorySlot} />
      <main className="main-shell">
        <Topbar
          route={route}
          model={model}
          onModelChange={setModel}
          stageCollapsed={stageCollapsed}
          onStageToggle={() => setStageCollapsed(!stageCollapsed)}
        />
        <div className="route-pane" hidden={route !== "chat"}>
          <ChatPage
            model={model}
            historySlot={historySlot}
            onOpenChat={() => setRoute("chat")}
            stageCollapsed={stageCollapsed}
            onStageCollapsedChange={setStageCollapsed}
          />
        </div>
        <div className="route-pane" hidden={route !== "library"}>
          <FilesPage />
        </div>
      </main>
      <NotificationToasts />
    </div>
  );
}

export default App;
