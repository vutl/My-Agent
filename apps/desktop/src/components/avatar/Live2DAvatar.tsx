import { useEffect, useRef, useState } from "react";
import * as PIXI from "pixi.js";
import { Live2DModel } from "pixi-live2d-display/cubism4";
import type { AvatarState } from "../../lib/types";
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
  stateRef.current = state;
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
        resolution: Math.min(window.devicePixelRatio || 1, 2),
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
        const baseWidth = loaded.width;
        const baseHeight = loaded.height;
        loaded.anchor.set(0.5, 0);
        // Framing follows the canvas: scale from its height, capped so a narrow
        // column never cuts more than a little of the sides, top offset kept
        // proportional to the drawn model.
        const layout = () => {
          const { width: w, height: h } = app.renderer.screen;
          if (!w || !h) return;
          const scale = Math.min((h * zoom) / baseHeight, (w * 1.5) / baseWidth);
          loaded.scale.set(scale);
          loaded.x = w / 2;
          loaded.y = (offsetY * scale * baseHeight) / zoom;
        };
        layout();
        app.renderer.on("resize", layout);
        app.stage.addChild(loaded);
        modelRef.current = loaded;
        // Lip sync while Aya streams her answer: open the mouth in a syllable
        // rhythm after motions have set their parameters for this frame.
        loaded.internalModel.on("beforeModelUpdate", () => {
          if (stateRef.current !== "talking") return;
          const time = performance.now() / 1000;
          const open = Math.max(0, Math.sin(time * 13) * 0.6 + Math.sin(time * 7.3) * 0.4);
          (loaded.internalModel.coreModel as unknown as {
            setParameterValueById: (id: string, value: number) => void;
          }).setParameterValueById("ParamMouthOpenY", Math.min(1, open));
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

  if (failed) return <AyaAvatar state={state} size={Math.min(size, 160)} />;
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
