import { useEffect, useRef, useState } from "react";
import * as PIXI from "pixi.js";
import { Live2DModel } from "pixi-live2d-display/cubism4";
import type { AvatarState } from "../../lib/types";
import { speech } from "../../lib/speech";
import { AyaAvatar, type AvatarProps } from "./AyaAvatar";

// pixi-live2d-display drives its animation from PIXI's ticker.
(window as unknown as { PIXI: typeof PIXI }).PIXI = PIXI;
Live2DModel.registerTicker(PIXI.Ticker);

type ModelName = "haru" | "mao";

interface ModelSpec {
  url: string;
  expressions: Record<AvatarState, string>;
  // Idle life: short gestures and long "special" motions (TapBody indices),
  // and the expressions for dozing off, waking up and being poked on the head.
  gestures: number[];
  specials: number[];
  sleepy: string;
  startled: string;
  shy: string;
  // Bust framing: model scale relative to the canvas height and vertical offset.
  zoom: number;
  offsetY: number;
}

const MODELS: Record<ModelName, ModelSpec> = {
  haru: {
    url: "/live2d/haru/Haru.model3.json",
    expressions: {
      idle: "F01", thinking: "F04", tool: "F07", waiting: "F06", talking: "F01", happy: "F05", confused: "F03",
    },
    gestures: [0, 1, 2, 3],
    specials: [],
    sleepy: "F05",
    startled: "F06",
    shy: "F08",
    zoom: 3.4,
    offsetY: -0.02,
  },
  mao: {
    url: "/live2d/mao/Mao.model3.json",
    expressions: {
      idle: "exp_01", thinking: "exp_05", tool: "exp_08", waiting: "exp_07", talking: "exp_01", happy: "exp_02",
      confused: "exp_06",
    },
    gestures: [0, 1, 2],
    specials: [3, 4, 5],
    sleepy: "exp_03",
    startled: "exp_07",
    shy: "exp_06",
    zoom: 2.2,
    offsetY: -0.22,
  },
};

// The stage view the user zooms and pans: zoom 1..MAX_ZOOM, pan as fractions
// of the canvas. Mao ships one 2048px texture, so past ~3× she turns soft.
const MAX_ZOOM = 3;
const VIEW_KEY = "aya.stage_view";
interface View {
  zoom: number;
  x: number;
  y: number;
}

// Where Aya looks (focus -1..1) in each state.
const GAZE: Partial<Record<AvatarState, [number, number]>> = {
  thinking: [0.55, 0.6],
  tool: [-0.25, -0.35],
  waiting: [0, 0],
};

interface Live2DAvatarProps extends AvatarProps {
  model?: ModelName;
  // Canvas size; the stage shows Aya larger and taller than the avatar slot.
  width?: number;
  height?: number;
  // Override the model's default bust framing (scale vs canvas height, top offset).
  framing?: { zoom: number; offsetY: number };
  // Fill the parent element and follow its size (the stage); otherwise fixed width/height.
  fill?: boolean;
}

/** Aya as a Live2D character; falls back to the SVG drawing if WebGL or the
 * model cannot load. Same props as every avatar renderer. */
export function Live2DAvatar({ state, size = 200, model = "mao", width: widthProp, height: heightProp, framing, fill = false }: Live2DAvatarProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const modelRef = useRef<Live2DModel | null>(null);
  const stateRef = useRef<AvatarState>(state);
  const [failed, setFailed] = useState(false);
  const viewRef = useRef<View>(loadView());
  const layoutRef = useRef<(() => void) | null>(null);
  const [zoomed, setZoomed] = useState(viewRef.current.zoom > 1);
  stateRef.current = state;

  function updateView(next: View) {
    viewRef.current = clampView(next);
    try {
      localStorage.setItem(VIEW_KEY, JSON.stringify(viewRef.current));
    } catch {
      // storage unavailable: the view just is not remembered
    }
    layoutRef.current?.();
    setZoomed(viewRef.current.zoom > 1.001);
  }

  // Zoom about a point of the canvas (CSS px), keeping that point still.
  function zoomAt(factor: number, cx?: number, cy?: number) {
    const box = hostRef.current?.getBoundingClientRect();
    if (!box?.width || !box.height) return;
    const px = cx ?? box.width / 2;
    const py = cy ?? box.height * 0.3;
    const view = viewRef.current;
    const zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom * factor));
    const x = (px - (zoom * (px - view.x * box.width)) / view.zoom) / box.width;
    const y = (py - (zoom * (py - view.y * box.height)) / view.zoom) / box.height;
    updateView({ zoom, x, y });
  }
  const width = widthProp ?? size;
  const height = heightProp ?? Math.round(size * 1.18);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || failed) return;
    let disposed = false;
    // PIXI owns its canvas: reusing one canvas across mounts (React StrictMode
    // remounts) hands the second app a lost WebGL context.
    let app: PIXI.Application;
    try {
      app = new PIXI.Application({
        width, height, backgroundAlpha: 0, antialias: true, autoDensity: true,
        // Full device density: on a Retina screen Aya renders 1:1 with its pixels.
        resolution: Math.min(window.devicePixelRatio || 1, 3),
      });
    } catch {
      setFailed(true);
      return;
    }
    host.appendChild(app.view);
    // Fill mode: the canvas follows the host's box (window resize, reminders
    // list growing, sidebar changes), not only the window.
    const observer = fill
      ? new ResizeObserver(([entry]) => {
          const { width: w, height: h } = entry.contentRect;
          if (w > 0 && h > 0) app.renderer.resize(Math.round(w), Math.round(h));
        })
      : null;
    observer?.observe(host);
    const spec = MODELS[model];
    const zoom = framing?.zoom ?? spec.zoom;
    const offsetY = framing?.offsetY ?? spec.offsetY;
    Live2DModel.from(spec.url, { autoInteract: true })
      .then((loaded: Live2DModel) => {
        if (disposed) {
          loaded.destroy();
          return;
        }
        loaded.textures.forEach((texture) => {
          texture.baseTexture.anisotropicLevel = 16;
        });
        const baseWidth = loaded.width;
        const baseHeight = loaded.height;
        loaded.anchor.set(0.5, 0);
        // Fill mode fits what is actually drawn (the model's canvas has empty
        // margins and the staff sticks out to one side) into the stage, then
        // applies the user's zoom and pan; otherwise the bust framing scales
        // from the canvas height.
        const content = fill ? drawnBox(loaded) : null;
        const layout = () => {
          const { width: w, height: h } = app.renderer.screen;
          if (!w || !h) return;
          if (content) {
            const scale = Math.min((h * 0.96) / (content.height * baseHeight), (w * 0.94) / (content.width * baseWidth));
            const x0 = w / 2 - (content.x + content.width / 2 - 0.5) * baseWidth * scale;
            const y0 = (h - content.height * baseHeight * scale) / 2 - content.y * baseHeight * scale;
            const view = viewRef.current;
            loaded.scale.set(scale * view.zoom);
            loaded.x = view.x * w + view.zoom * x0;
            loaded.y = view.y * h + view.zoom * y0;
            return;
          }
          const scale = (h * zoom) / baseHeight;
          loaded.scale.set(scale);
          loaded.x = w / 2;
          loaded.y = h * offsetY;
        };
        layout();
        layoutRef.current = layout;
        app.renderer.on("resize", layout);
        app.stage.addChild(loaded);
        modelRef.current = loaded;
        // Runs every frame after motions and the focus controller have set their
        // parameters: reading gaze, mouth and a small nod while speaking.
        const core = loaded.internalModel.coreModel as unknown as {
          setParameterValueById: (id: string, value: number) => void;
          addParameterValueById?: (id: string, value: number, weight?: number) => void;
        };
        const focusController = (loaded.internalModel as unknown as {
          focusController: { focus: (x: number, y: number, instant?: boolean) => void };
        }).focusController;
        let frame = 0;
        let readTarget: [number, number] | null = null;
        // The mouth parameters come from the model's own LipSync group (Mao: the
        // vowel "ParamA"; Haru: "ParamMouthOpenY"). Writing a parameter the model
        // does not have is silently ignored, which left Mao's mouth shut.
        const lipSync = (loaded.internalModel.motionManager as unknown as { lipSyncIds?: string[] }).lipSyncIds;
        const mouthIds = lipSync && lipSync.length > 0 ? lipSync : ["ParamMouthOpenY"];
        const parameterIds = new Set(
          ((core as unknown as { getModel?: () => { parameters?: { ids?: string[] } } }).getModel?.()?.parameters?.ids) ?? [],
        );
        // Models with vowel shapes get a changing secondary vowel so speech is not one "a-a-a".
        const vowels = ["ParamI", "ParamU", "ParamE", "ParamO"].filter((id) => parameterIds.has(id) && !mouthIds.includes(id));
        let vowel = 0;
        let vowelUntil = 0;
        loaded.internalModel.on("beforeModelUpdate", () => {
          const reading = speech.busy || stateRef.current === "talking";
          if (reading) {
            // Look at the answer being written, then follow the voice down it.
            if (frame++ % 6 === 0) readTarget = readingFocus(loaded, host);
            if (readTarget) focusController.focus(readTarget[0], readTarget[1]);
          } else {
            readTarget = null;
          }
          if (!reading) return;
          const time = performance.now() / 1000;
          // Audio level when voice plays; closed while the next sentence is being
          // synthesized; the syllable rhythm when only text streams (voice off).
          const open = speech.speaking ? speech.level() * 1.6
            : speech.busy ? 0
            : Math.max(0, Math.sin(time * 13) * 0.6 + Math.sin(time * 7.3) * 0.4);
          const mouth = Math.min(1, open);
          for (const id of mouthIds) core.setParameterValueById(id, mouth);
          if (vowels.length > 0) {
            const now = performance.now();
            if (now > vowelUntil) {
              vowel = Math.floor(Math.random() * vowels.length);
              vowelUntil = now + 90 + Math.random() * 140;
            }
            vowels.forEach((id, index) => core.setParameterValueById(id, index === vowel ? mouth * 0.45 : 0));
          }
          if (speech.speaking) core.addParameterValueById?.("ParamAngleY", -speech.level() * 6);
        });
        applyState(loaded, stateRef.current, spec);
        // Poke reactions: head → shy blush, body → a little gesture.
        loaded.on("hit", (areas: string[]) => {
          if (stateRef.current !== "idle") return;
          if (areas.includes("Head")) {
            void loaded.expression(spec.shy);
            window.setTimeout(() => void loaded.expression(spec.expressions.idle), 2500);
          } else {
            void loaded.motion("TapBody", pick(spec.gestures));
          }
        });
      })
      .catch(() => !disposed && setFailed(true));
    return () => {
      disposed = true;
      observer?.disconnect();
      layoutRef.current = null;
      modelRef.current = null;
      app.destroy(true, { children: true });
    };
  }, [model, fill ? 0 : width, fill ? 0 : height, failed, fill, framing?.zoom, framing?.offsetY]);

  useEffect(() => {
    if (modelRef.current) applyState(modelRef.current, state, MODELS[model]);
  }, [state, model]);

  // Idle life: while nothing happens Aya gestures now and then, casts a spell
  // after a longer while and dozes off after a few minutes; any mouse or key
  // activity wakes her up (startled for a moment).
  useEffect(() => {
    if (state !== "idle") return;
    const spec = MODELS[model];
    let idleSince = Date.now();
    let nextGesture = idleSince + randomBetween(20_000, 40_000);
    let nextSpecial = idleSince + randomBetween(90_000, 150_000);
    let sleeping = false;
    const wake = () => {
      idleSince = Date.now();
      nextGesture = idleSince + randomBetween(20_000, 40_000);
      nextSpecial = idleSince + randomBetween(90_000, 150_000);
      if (sleeping && modelRef.current) {
        sleeping = false;
        const current = modelRef.current;
        void current.expression(spec.startled);
        window.setTimeout(() => void current.expression(spec.expressions.idle), 1400);
      }
    };
    const tick = window.setInterval(() => {
      const current = modelRef.current;
      if (!current || sleeping) return;
      const now = Date.now();
      if (now - idleSince > 180_000) {
        sleeping = true;
        void current.expression(spec.sleepy);
      } else if (spec.specials.length && now >= nextSpecial) {
        nextSpecial = now + randomBetween(120_000, 200_000);
        nextGesture = now + randomBetween(25_000, 45_000);
        void current.motion("TapBody", pick(spec.specials));
      } else if (now >= nextGesture) {
        nextGesture = now + randomBetween(20_000, 40_000);
        void current.motion("TapBody", pick(spec.gestures));
      }
    }, 2_000);
    const events = ["pointermove", "keydown", "pointerdown"] as const;
    events.forEach((name) => window.addEventListener(name, wake, { passive: true }));
    return () => {
      window.clearInterval(tick);
      events.forEach((name) => window.removeEventListener(name, wake));
    };
  }, [state, model]);

  // Stage view controls: wheel zooms about the cursor, drag pans while zoomed,
  // double-click goes back to the whole of Aya.
  useEffect(() => {
    const host = hostRef.current;
    if (!fill || !host || failed) return;
    let drag: { id: number; x: number; y: number } | null = null;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const box = host.getBoundingClientRect();
      zoomAt(Math.exp(-event.deltaY * 0.0015), event.clientX - box.left, event.clientY - box.top);
    };
    const onDown = (event: PointerEvent) => {
      if (event.button === 0 && viewRef.current.zoom > 1) drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    };
    const onMove = (event: PointerEvent) => {
      if (!drag || event.pointerId !== drag.id) return;
      const box = host.getBoundingClientRect();
      const view = viewRef.current;
      updateView({ ...view, x: view.x + (event.clientX - drag.x) / box.width, y: view.y + (event.clientY - drag.y) / box.height });
      drag = { ...drag, x: event.clientX, y: event.clientY };
    };
    const onUp = () => {
      drag = null;
    };
    const onDoubleClick = () => updateView({ zoom: 1, x: 0, y: 0 });
    host.addEventListener("wheel", onWheel, { passive: false });
    host.addEventListener("pointerdown", onDown);
    host.addEventListener("dblclick", onDoubleClick);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      host.removeEventListener("wheel", onWheel);
      host.removeEventListener("pointerdown", onDown);
      host.removeEventListener("dblclick", onDoubleClick);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    // zoomAt/updateView only touch refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fill, failed]);

  if (failed) return <AyaAvatar state={state} size={Math.min(size, 160)} />;
  if (fill) {
    return (
      <div className="aya-live2d-frame">
        <div
          ref={hostRef}
          className={`aya-live2d aya-live2d--${state}${zoomed ? " is-zoomed" : ""}`}
          style={{ width: "100%", height: "100%" }}
          role="img"
          aria-label={`Aya (${state})`}
          title="Cuộn để phóng to/thu nhỏ · kéo để di chuyển · nhấp đúp để xem toàn thân"
        />
        <div className="aya-live2d-zoom" role="group" aria-label="Phóng to Aya">
          <button type="button" onClick={() => zoomAt(1.25)} aria-label="Phóng to" title="Phóng to">+</button>
          <button type="button" onClick={() => zoomAt(0.8)} aria-label="Thu nhỏ" title="Thu nhỏ">−</button>
          {zoomed && (
            <button type="button" onClick={() => updateView({ zoom: 1, x: 0, y: 0 })} aria-label="Xem toàn thân" title="Xem toàn thân">
              <svg viewBox="0 0 20 20" width="13" height="13" aria-hidden="true">
                <path d="M4 8V4h4M16 8V4h-4M4 12v4h4M16 12v4h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>
      </div>
    );
  }
  return (
    <div
      ref={hostRef}
      className={`aya-live2d aya-live2d--${state}`}
      style={fill ? { width: "100%", height: "100%" } : { width, height }}
      role="img"
      aria-label={`Aya (${state})`}
    />
  );
}

function clampView(view: View): View {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, Number(view.zoom) || 1));
  const clamp = (value: number) => Math.min(0, Math.max(1 - zoom, Number(value) || 0));
  return { zoom, x: clamp(view.x), y: clamp(view.y) };
}

function loadView(): View {
  try {
    const saved = JSON.parse(localStorage.getItem(VIEW_KEY) ?? "null");
    if (saved && typeof saved === "object") return clampView(saved as View);
  } catch {
    // nothing remembered
  }
  return { zoom: 1, x: 0, y: 0 };
}

interface CoreModel {
  getDrawableCount(): number;
  getDrawableOpacity(index: number): number;
}

/** The box the visible drawables cover, as fractions of the model's canvas. */
function drawnBox(loaded: Live2DModel): { x: number; y: number; width: number; height: number } | null {
  const internal = loaded.internalModel as unknown as {
    coreModel: CoreModel;
    originalWidth: number;
    originalHeight: number;
    getDrawableVertices(index: number): Float32Array;
  };
  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
  try {
    const count = internal.coreModel.getDrawableCount();
    for (let index = 0; index < count; index++) {
      if (internal.coreModel.getDrawableOpacity(index) < 0.05) continue;
      const vertices = internal.getDrawableVertices(index);
      for (let i = 0; i < vertices.length; i += 2) {
        left = Math.min(left, vertices[i]);
        right = Math.max(right, vertices[i]);
        top = Math.min(top, vertices[i + 1]);
        bottom = Math.max(bottom, vertices[i + 1]);
      }
    }
  } catch {
    return null;
  }
  if (!(right > left && bottom > top)) return null;
  const { originalWidth: ow, originalHeight: oh } = internal;
  return { x: left / ow, y: top / oh, width: (right - left) / ow, height: (bottom - top) / oh };
}

/** Focus (-1..1, y up) that points Aya's head at the line of the latest answer
 * being written — or, once written, at the part the voice has reached. */
function readingFocus(model: Live2DModel, host: HTMLElement): [number, number] | null {
  const bubbles = document.querySelectorAll<HTMLElement>(".message-row.assistant .message-bubble");
  const bubble = bubbles[bubbles.length - 1];
  if (!bubble) return null;
  const box = bubble.getBoundingClientRect();
  if (!box.height) return null;
  const writing = bubble.querySelector(".markdown-block.streaming") !== null;
  const along = writing || !speech.busy ? 1 : speech.progress();
  const y = Math.min(window.innerHeight - 40, Math.max(40, box.top + along * box.height));
  const x = box.left + Math.min(box.width, 420) / 2;
  const canvas = host.getBoundingClientRect();
  const bounds = model.getBounds();
  const headX = canvas.left + bounds.x + bounds.width / 2;
  const headY = canvas.top + bounds.y + bounds.height * 0.16;
  const clamp = (value: number) => Math.max(-1, Math.min(1, value));
  return [clamp((x - headX) / 500), clamp((headY - y) / 350)];
}

function pick(values: number[]): number {
  return values[Math.floor(Math.random() * values.length)] ?? 0;
}

function randomBetween(low: number, high: number): number {
  return low + Math.random() * (high - low);
}

function applyState(model: Live2DModel, state: AvatarState, spec: ModelSpec) {
  void model.expression(spec.expressions[state]);
  const gaze = GAZE[state];
  const focus = (model.internalModel as unknown as {
    focusController: { focus: (x: number, y: number, instant?: boolean) => void };
  }).focusController;
  if (gaze) focus.focus(gaze[0], gaze[1]);
  if (state === "happy") void model.motion("TapBody", pick(spec.gestures));
}
