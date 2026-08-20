import type { AppRoute } from "../../App";
import { ModelSelect } from "./ModelSelect";

interface TopbarProps {
  route: AppRoute;
  model: string;
  onModelChange: (model: string) => void;
}

const routeNames: Record<AppRoute, string> = {
  chat: "Chat",
  library: "Library",
};

export function Topbar({ route, model, onModelChange }: TopbarProps) {
  return (
    <header className="topbar">
      <h1>{routeNames[route]}</h1>
      <ModelSelect model={model} onModelChange={onModelChange} />
    </header>
  );
}
