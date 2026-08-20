import { API_BASE_URL } from "../lib/api";

export function SettingsPage() {
  return (
    <section className="settings-page">
      <div className="settings-row">
        <label>
          <span>Backend URL</span>
          <input readOnly value={API_BASE_URL} />
        </label>
      </div>
      <div className="settings-row">
        <label>
          <span>Ollama Host</span>
          <input readOnly value="http://localhost:11434" />
        </label>
      </div>
    </section>
  );
}
