import { useEffect, useState, useSyncExternalStore } from "react";
import { getSpeechStatus, getSpeechVoices, speech, type SpeechStatus, type SpeechVoice } from "../../lib/speech";

const GROUPS = [
  { kind: "vi", label: "Tiếng Việt" },
  { kind: "vi-anime", label: "Tiếng Việt · giọng anime" },
  { kind: "ja", label: "Tiếng Nhật (VOICEVOX)" },
] as const;

export function VoicePicker() {
  const settings = useSyncExternalStore(speech.subscribe, () => speech.settings);
  const [status, setStatus] = useState<SpeechStatus>({ state: "off", detail: null });
  const [voices, setVoices] = useState<SpeechVoice[]>([]);
  const [error, setError] = useState("");

  useEffect(() => speech.subscribe((event) => {
    if (event.type === "error") setError(event.detail ?? "");
    if (event.type === "start" || event.type === "settings") setError("");
  }), []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    async function poll() {
      controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 10_000);
      let ready = false;
      try {
        const next = await getSpeechStatus(controller.signal);
        if (stopped) return;
        setStatus(next);
        if (next.state === "ready") {
          const catalog = await getSpeechVoices(controller.signal);
          if (stopped) return;
          setVoices(catalog.voices);
          ready = true;
          if (!catalog.voices.some((voice) => voice.id === speech.settings.voiceId)) {
            speech.updateSettings({ voiceId: catalog.default_voice_id });
          }
        }
      } catch {
        if (!stopped) setStatus({ state: "off", detail: "Giọng nói chưa được cài hoặc đang ngoại tuyến" });
      } finally {
        window.clearTimeout(timeout);
        if (!stopped) timer = setTimeout(() => void poll(), ready ? 60_000 : 5_000);
      }
    }
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, []);

  const selected = voices.find((voice) => voice.id === settings.voiceId);
  // Compact by default so Aya keeps the stage; speed/pitch open on demand.
  const [expanded, setExpanded] = useState(() => {
    try { return localStorage.getItem("aya.voice_picker_open") === "1"; } catch { return false; }
  });
  const toggleExpanded = () => {
    setExpanded((open) => {
      try { localStorage.setItem("aya.voice_picker_open", open ? "0" : "1"); } catch { /* session only */ }
      return !open;
    });
  };
  if (status.state !== "ready" || !voices.length) {
    const message = status.state === "starting" ? "Giọng đang khởi động…"
      : status.state === "error" ? status.detail || "Giọng nói gặp lỗi"
      : status.detail || "Giọng nói chưa được cài";
    return <section className="voice-picker" aria-label="Giọng Aya"><p className="voice-picker__hint" role="status">{message}</p></section>;
  }

  return (
    <section className={`voice-picker${expanded ? " voice-picker--open" : ""}`} aria-label="Giọng Aya">
      <div className="voice-picker__choice">
        <select id="aya-voice" aria-label="Giọng Aya" value={settings.voiceId ?? ""} onChange={(event) => {
          speech.stop();
          speech.updateSettings({ voiceId: event.target.value });
        }} title={selected?.description}>
          {GROUPS.map((group) => (
            <optgroup key={group.kind} label={group.label}>
              {voices.filter((voice) => voice.kind === group.kind)
                .sort((a, b) => Number(b.gender === "female") - Number(a.gender === "female") || a.name.localeCompare(b.name))
                .map((voice) => <option key={voice.id} value={voice.id}>{voice.name}{voice.region ? ` · ${voice.region}` : ""}</option>)}
            </optgroup>
          ))}
        </select>
        <button type="button" title="Nghe thử giọng Aya" aria-label="Nghe thử giọng Aya" onClick={() => {
          void speech.setEnabled(true);
          speech.preview();
        }}>▶</button>
        <button type="button" className="voice-picker__more" aria-expanded={expanded}
          title={expanded ? "Thu gọn" : "Tốc độ, cao độ"} aria-label={expanded ? "Thu gọn" : "Tốc độ, cao độ"}
          onClick={toggleExpanded}>{expanded ? "▴" : "▾"}</button>
      </div>
      {expanded && (<>
      <label className="voice-picker__slider">
        <span>Tốc độ <output>{settings.speed.toFixed(1)}×</output></span>
        <input type="range" min="0.8" max="1.3" step="0.1" value={settings.speed}
          onChange={(event) => speech.updateSettings({ speed: Number(event.target.value) })} />
      </label>
      <label className="voice-picker__slider">
        <span>Cao độ <output>{settings.pitch > 0 ? "+" : ""}{settings.pitch}</output></span>
        <input type="range" min="-2" max="5" step="1" value={settings.pitch}
          onChange={(event) => speech.updateSettings({ pitch: Number(event.target.value) })} />
      </label>
      </>)}
      {selected?.credit && (
        <p className="voice-picker__credit">
          {selected.terms_url && /^https?:\/\//i.test(selected.terms_url)
            ? <a href={selected.terms_url} target="_blank" rel="noopener noreferrer">{selected.credit}</a>
            : selected.credit}
        </p>
      )}
      {error && <p className="voice-picker__hint" role="status">{error}</p>}
    </section>
  );
}
