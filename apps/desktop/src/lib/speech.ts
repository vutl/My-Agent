import { API_BASE_URL } from "./api";

export interface SpeechSettings {
  enabled: boolean;
  voiceId: string | null;
  speed: number;
  pitch: number;
}

export interface SpeechVoice {
  id: string;
  name: string;
  kind: "vi" | "vi-anime" | "ja";
  language: string;
  gender: string;
  region: string | null;
  description: string;
  credit: string | null;
  terms_url: string | null;
}

export interface SpeechStatus {
  state: "off" | "starting" | "ready" | "error";
  detail: string | null;
}

export interface SpeechVoices {
  voices: SpeechVoice[];
  default_voice_id: string;
}

export async function getSpeechStatus(signal?: AbortSignal): Promise<SpeechStatus> {
  const response = await fetch(`${API_BASE_URL}/speech/status`, { signal });
  if (!response.ok) throw new Error("Giọng nói chưa được cài hoặc đang ngoại tuyến");
  return response.json();
}

export async function getSpeechVoices(signal?: AbortSignal): Promise<SpeechVoices> {
  const response = await fetch(`${API_BASE_URL}/speech/voices`, { signal });
  if (!response.ok) throw new Error("Chưa tải được danh sách giọng nói");
  return response.json();
}

const DEFAULTS: SpeechSettings = { enabled: false, voiceId: null, speed: 1, pitch: 0 };
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

function loadSettings(): SpeechSettings {
  try {
    const saved = JSON.parse(localStorage.getItem("aya.voice") ?? "null");
    return {
      enabled: saved?.enabled === true,
      voiceId: typeof saved?.voiceId === "string" && saved.voiceId ? saved.voiceId : null,
      speed: Number.isFinite(saved?.speed) ? clamp(saved.speed, 0.7, 1.5) : 1,
      pitch: Number.isFinite(saved?.pitch) ? clamp(saved.pitch, -4, 6) : 0,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

/** Pure incremental segmenter. Closed code fences are discarded; an open fence
 * (including a delimiter split across deltas) stays in the remainder. */
export function segmentSpeech(text: string, flush = false): { segments: string[]; remainder: string } {
  const segments: string[] = [];
  while (text) {
    if (text.startsWith("```")) {
      const close = text.indexOf("```", 3);
      if (close < 0) break;
      text = text.slice(close + 3);
      continue;
    }
    const fence = text.indexOf("```");
    const partialFence = text.match(/`{1,2}$/)?.[0].length ?? 0;
    const available = fence >= 0 ? fence : text.length - partialFence;
    const prefix = text.slice(0, available);
    let cut = 0;
    for (let index = 0; index < Math.min(prefix.length, 160); index++) {
      if (index >= 19 && /[.!?…。！？\n]/.test(prefix[index])) {
        cut = index + 1;
        break;
      }
    }
    if (!cut && prefix.length >= 160) {
      const head = prefix.slice(0, 160);
      cut = Math.max(head.lastIndexOf(",") + 1, head.lastIndexOf(" ") + 1) || 160;
      // Do not split a UTF-16 surrogate pair.
      if (/[\uD800-\uDBFF]/.test(text[cut - 1])) cut--;
    }
    if (!cut && (flush || fence >= 0)) cut = available;
    if (!cut) break;
    const segment = text.slice(0, cut).trim();
    if (segment) segments.push(segment);
    text = text.slice(cut);
  }
  return { segments, remainder: text };
}

type SpeechEvent = { type: "settings" | "start" | "end" | "change" | "error"; detail?: string };
interface Segment {
  messageId: string;
  text: string;
  settings: SpeechSettings;
  state: "pending" | "loading" | "ready" | "skip";
  audio?: AudioBuffer;
  spokenText?: string;
}

class SpeechEngine {
  settings = loadSettings();
  speaking = false;
  spokenText = "";
  spokenVoiceId: string | null = null;
  private listeners = new Set<(event: SpeechEvent) => void>();
  private messageId: string | null = null;
  private pendingText = "";
  private flushTimer: ReturnType<typeof setTimeout> | undefined;
  private queue: Segment[] = [];
  private requests = new Set<AbortController>();
  private inFlight = 0;
  private generation = 0;
  private reportedError = false;
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private samples: Uint8Array<ArrayBuffer> | null = null;
  private source: AudioBufferSourceNode | null = null;
  private mouthLevel = 0;
  private levelTime = 0;

  constructor() {
    if (typeof window !== "undefined") {
      const unlock = () => { if (this.settings.enabled) this.resume(); };
      window.addEventListener("pointerdown", unlock, { passive: true });
      window.addEventListener("keydown", unlock);
    }
  }

  subscribe = (listener: (event: SpeechEvent) => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private emit(event: SpeechEvent) {
    this.listeners.forEach((listener) => listener(event));
  }

  updateSettings(update: Partial<SpeechSettings>) {
    this.settings = {
      ...this.settings, ...update,
      speed: Number.isFinite(update.speed) ? clamp(update.speed!, 0.7, 1.5) : this.settings.speed,
      pitch: Number.isFinite(update.pitch) ? clamp(update.pitch!, -4, 6) : this.settings.pitch,
    };
    try { localStorage.setItem("aya.voice", JSON.stringify(this.settings)); } catch { /* session only */ }
    if (!this.settings.enabled) this.stop();
    this.emit({ type: "settings" });
  }

  async setEnabled(enabled: boolean) {
    this.updateSettings({ enabled });
    if (!enabled) return;
    this.resume();
    if (!this.settings.voiceId) {
      try {
        const catalog = await getSpeechVoices(AbortSignal.timeout(10_000));
        if (this.settings.enabled && !this.settings.voiceId) {
          this.updateSettings({ voiceId: catalog.default_voice_id });
        }
      } catch { /* picker polls until the backend becomes ready */ }
    }
  }

  private getContext(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext();
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 512;
      this.samples = new Uint8Array(this.analyser.fftSize);
      this.analyser.connect(this.context.destination);
      this.context.onstatechange = () => this.playNext();
    }
    return this.context;
  }

  resume() {
    try {
      void this.getContext().resume().then(() => this.playNext()).catch(() => this.error());
    } catch { this.error(); }
  }

  push(messageId: string, delta: string) {
    if (!this.settings.enabled || !this.settings.voiceId || !delta) return;
    if (this.messageId !== messageId) {
      this.stop();
      this.messageId = messageId;
    }
    this.pendingText += delta;
    this.flush(false);
    clearTimeout(this.flushTimer);
    this.flushTimer = setTimeout(() => this.flush(true), 700);
  }

  finish(messageId: string) {
    if (!this.settings.enabled || this.messageId !== messageId) return;
    clearTimeout(this.flushTimer);
    this.flush(true);
    // An unfinished code fence must never be sent, even at end of stream.
    this.pendingText = "";
  }

  reset(messageId: string) {
    if (this.messageId === messageId) this.stop();
  }

  stop() {
    this.generation++;
    clearTimeout(this.flushTimer);
    this.pendingText = "";
    this.messageId = null;
    this.queue = [];
    this.requests.forEach((controller) => controller.abort());
    if (this.source) {
      this.source.onended = null;
      try { this.source.stop(); } catch { /* already stopped */ }
      this.source.disconnect();
      this.source = null;
    }
    const wasSpeaking = this.speaking;
    this.speaking = false;
    this.spokenText = "";
    this.spokenVoiceId = null;
    this.mouthLevel = 0;
    this.reportedError = false;
    if (wasSpeaking) this.emit({ type: "end" });
    this.emit({ type: "change" });
  }

  preview() {
    if (!this.settings.enabled) return;
    this.stop();
    this.resume();
    const id = `preview:${Date.now()}`;
    this.push(id, "Chào cậu~ Tôi là Aya đây!");
    this.finish(id);
  }

  level(): number {
    if (!this.speaking || !this.analyser || !this.samples) return 0;
    this.analyser.getByteTimeDomainData(this.samples);
    let squares = 0;
    for (const sample of this.samples) squares += ((sample - 128) / 128) ** 2;
    const target = Math.min(1, Math.sqrt(squares / this.samples.length) * 5);
    const now = performance.now();
    const elapsed = Math.min(100, now - this.levelTime);
    this.levelTime = now;
    const timeConstant = target > this.mouthLevel ? 25 : 140;
    this.mouthLevel += (target - this.mouthLevel) * (1 - Math.exp(-elapsed / timeConstant));
    return this.mouthLevel;
  }

  private flush(force: boolean) {
    const result = segmentSpeech(this.pendingText, force);
    this.pendingText = result.remainder;
    for (const text of result.segments) {
      this.queue.push({ messageId: this.messageId!, text, settings: { ...this.settings }, state: "pending" });
    }
    this.pump();
  }

  private pump() {
    while (this.inFlight < 2) {
      const segment = this.queue.find((item) => item.state === "pending");
      if (!segment) break;
      segment.state = "loading";
      void this.synthesize(segment, this.generation);
    }
    this.playNext();
  }

  private async synthesize(segment: Segment, generation: number) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);
    this.requests.add(controller);
    this.inFlight++;
    try {
      const response = await fetch(`${API_BASE_URL}/speech/synthesize`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: segment.text, voice_id: segment.settings.voiceId,
          speed: segment.settings.speed, pitch: segment.settings.pitch }),
        signal: controller.signal,
      });
      if (generation !== this.generation) return;
      if (response.status === 204) {
        segment.state = "skip";
      } else {
        if (!response.ok) throw new Error("synthesis failed");
        const wav = await response.arrayBuffer();
        if (generation !== this.generation) return;
        segment.audio = await this.getContext().decodeAudioData(wav);
        if (generation !== this.generation) return;
        try { segment.spokenText = decodeURIComponent(response.headers.get("X-Aya-Spoken-Text") ?? ""); } catch { /* malformed header */ }
        segment.state = "ready";
      }
    } catch {
      if (generation === this.generation) {
        segment.state = "skip";
        this.error();
      }
    } finally {
      clearTimeout(timeout);
      this.requests.delete(controller);
      this.inFlight--;
      this.pump();
    }
  }

  private playNext() {
    if (this.source) return;
    while (this.queue[0]?.state === "skip") this.queue.shift();
    const segment = this.queue[0];
    if (segment?.state !== "ready" || !segment.audio || this.context?.state !== "running") return;
    this.queue.shift();
    try {
      const source = this.context.createBufferSource();
      source.buffer = segment.audio;
      source.connect(this.analyser!);
      const generation = this.generation;
      source.onended = () => {
        source.disconnect();
        if (generation !== this.generation) return;
        this.source = null;
        this.speaking = false;
        this.spokenText = "";
        this.spokenVoiceId = null;
        this.mouthLevel = 0;
        this.emit({ type: "end" });
        this.pump();
      };
      source.start();
      this.source = source;
      this.speaking = true;
      this.spokenText = segment.spokenText ?? "";
      this.spokenVoiceId = segment.settings.voiceId;
      this.levelTime = performance.now();
      this.emit({ type: "start" });
    } catch {
      this.error();
      this.playNext();
    }
  }

  private error() {
    if (this.reportedError) return;
    this.reportedError = true;
    this.emit({ type: "error", detail: "Tôi chưa phát được giọng nói. Cậu thử lại sau nhé." });
  }
}

export const speech = new SpeechEngine();
