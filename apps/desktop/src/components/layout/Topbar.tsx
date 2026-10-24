import type { AppRoute } from "../../App";
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
  return (
    <header className="topbar">
      <h1>{routeNames[route]}</h1>
      <div className="topbar-actions">
        {route === "chat" && (
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
        )}
        <ModelSelect model={model} onModelChange={onModelChange} />
      </div>
    </header>
  );
}
