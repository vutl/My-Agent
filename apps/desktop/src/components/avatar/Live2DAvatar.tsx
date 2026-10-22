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
    zoom: 3.4,
    offsetY: -0.02,
  },
  mao: {
    url: "/live2d/mao/Mao.model3.json",
    expressions: {
      idle: "exp_01", thinking: "exp_05", tool: "exp_08", waiting: "exp_07", talking: "exp_01", happy: "exp_02",
      confused: "exp_06",
    },
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
}

/** Aya as a Live2D character; falls back to the SVG drawing if WebGL or the
 * model cannot load. Same props as every avatar renderer. */
export function Live2DAvatar({ state, size = 200, model = "mao" }: Live2DAvatarProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const modelRef = useRef<Live2DModel | null>(null);
  const stateRef = useRef<AvatarState>(state);
  const [failed, setFailed] = useState(false);
  stateRef.current = state;
  const width = size;
  const height = Math.round(size * 1.18);

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
    const spec = MODELS[model];
    Live2DModel.from(spec.url, { autoInteract: true })
      .then((loaded: Live2DModel) => {
        if (disposed) {
          loaded.destroy();
          return;
        }
        const scale = (height * spec.zoom) / loaded.height;
        loaded.scale.set(scale);
        loaded.anchor.set(0.5, 0);
        loaded.x = width / 2;
        loaded.y = height * spec.offsetY;
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
      })
      .catch(() => !disposed && setFailed(true));
    return () => {
      disposed = true;
      modelRef.current = null;
      app.destroy(true, { children: true });
    };
  }, [model, width, height, failed]);

  useEffect(() => {
    if (modelRef.current) applyState(modelRef.current, state, MODELS[model]);
  }, [state, model]);

  if (failed) return <AyaAvatar state={state} size={Math.min(size, 160)} />;
  return (
    <div
      ref={hostRef}
      className={`aya-live2d aya-live2d--${state}`}
      style={{ width, height }}
      role="img"
      aria-label={`Aya (${state})`}
    />
  );
}

function applyState(model: Live2DModel, state: AvatarState, spec: ModelSpec) {
  void model.expression(spec.expressions[state]);
  const gaze = GAZE[state];
  const focus = (model.internalModel as unknown as {
    focusController: { focus: (x: number, y: number, instant?: boolean) => void };
  }).focusController;
  if (gaze) focus.focus(gaze[0], gaze[1]);
  if (state === "happy") void model.motion("TapBody");
}
