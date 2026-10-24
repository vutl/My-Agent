import { useSyncExternalStore } from "react";
import type { AppRoute } from "../../App";
import { speech } from "../../lib/speech";
import { ModelSelect } from "./ModelSelect";

interface TopbarProps {
  route: AppRoute;
  model: string;
  onModelChange: (model: string) => void;
  stageCollapsed: boolean;
  onStageToggle: () => void;
}

const routeNames: Record<AppRoute, string> = {
  chat: "Chat",
  library: "Library",
};

export function Topbar({ route, model, onModelChange, stageCollapsed, onStageToggle }: TopbarProps) {
  const voice = useSyncExternalStore(speech.subscribe, () => speech.settings);
  return (
    <header className="topbar">
      <h1>{routeNames[route]}</h1>
      <div className="topbar-actions">
        {route === "chat" && (
          <>
            <button
              type="button"
              className={stageCollapsed ? "stage-toggle" : "stage-toggle stage-toggle--on"}
              onClick={onStageToggle}
              aria-pressed={!stageCollapsed}
              title={stageCollapsed ? "Hiện Aya" : "Ẩn Aya"}
            >
              <svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true">
                <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
                <path d="M12.5 3.5v13" stroke="currentColor" strokeWidth="1.6" />
                {!stageCollapsed && <rect x="12.5" y="3.5" width="5" height="13" rx="1" fill="currentColor" opacity="0.35" />}
              </svg>
              {stageCollapsed ? "Hiện Aya" : "Ẩn Aya"}
            </button>
            <button
              type="button"
              className={`stage-toggle voice-toggle${voice.enabled ? " stage-toggle--on" : ""}`}
              onClick={() => void speech.setEnabled(!voice.enabled)}
              aria-pressed={voice.enabled}
              title={voice.enabled ? "Tắt giọng Aya" : "Bật giọng Aya"}
              aria-label={voice.enabled ? "Tắt giọng Aya" : "Bật giọng Aya"}
            >
              <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                <path d="M3 7h4l4-4v14l-4-4H3z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                {voice.enabled
                  ? <path d="M14 6c2 2 2 6 0 8m2-10c3 3 3 9 0 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                  : <path d="m14 8 4 4m0-4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />}
              </svg>
            </button>
          </>
        )}
        <ModelSelect model={model} onModelChange={onModelChange} />
      </div>
    </header>
  );
}
