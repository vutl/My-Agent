import { useState } from "react";

import { ChatPage } from "./routes/ChatPage";
import { FilesPage } from "./routes/FilesPage";
import { Sidebar } from "./components/layout/Sidebar";
import { Topbar } from "./components/layout/Topbar";

export type AppRoute = "chat" | "library";

function App() {
  const [route, setRoute] = useState<AppRoute>("chat");
  const [model, setModel] = useState("cx/gpt-5.6-sol");

  return (
    <div className="app-shell">
      <Sidebar activeRoute={route} onRouteChange={setRoute} />
      <main className="main-shell">
        <Topbar route={route} model={model} onModelChange={setModel} />
        <div className="route-pane" hidden={route !== "chat"}>
          <ChatPage model={model} />
        </div>
        <div className="route-pane" hidden={route !== "library"}>
          <FilesPage />
        </div>
      </main>
    </div>
  );
}

export default App;
