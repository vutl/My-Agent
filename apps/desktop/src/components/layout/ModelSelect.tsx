import { useEffect, useId, useRef, useState } from "react";

export interface ModelOption {
  label: string;
  value: string;
  hint?: string;
  group: "cloud" | "other";
}

const MODEL_OPTIONS: ModelOption[] = [
  { label: "GPT-5.6 Sol", value: "cx/gpt-5.6-sol", hint: "9router", group: "cloud" },
  { label: "GPT-5.6 Terra", value: "cx/gpt-5.6-terra", hint: "9router", group: "cloud" },
  { label: "GPT-5.6 Luna", value: "cx/gpt-5.6-luna", hint: "9router", group: "cloud" },
];

interface ModelSelectProps {
  model: string;
  onModelChange: (model: string) => void;
}

function displayLabel(model: string): string {
  const preset = MODEL_OPTIONS.find((option) => option.value === model);
  if (preset) return preset.label;
  if (model.startsWith("cx/")) return model.replace("cx/", "Codex · ");
  if (model.startsWith("cc/")) return model.replace("cc/", "Claude · ");
  return model || "Custom model";
}

function providerBadge(model: string): string | null {
  const preset = MODEL_OPTIONS.find((option) => option.value === model);
  if (preset?.hint) return preset.hint;
  return model ? "9router" : null;
}

export function ModelSelect({ model, onModelChange }: ModelSelectProps) {
  const [open, setOpen] = useState(false);
  const [customMode, setCustomMode] = useState(
    () => !MODEL_OPTIONS.some((option) => option.value === model)
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();

  const isPreset = MODEL_OPTIONS.some((option) => option.value === model);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  function choosePreset(value: string) {
    setCustomMode(false);
    onModelChange(value);
    setOpen(false);
  }

  function enableCustom() {
    setCustomMode(true);
    setOpen(false);
  }

  const badge = providerBadge(model);

  return (
    <div className="model-select" ref={rootRef}>
      <span className="model-select-label">Model</span>

      <div className="model-select-control">
        <button
          type="button"
          className="model-select-trigger"
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-controls={listId}
          onClick={() => setOpen((current) => !current)}
        >
          <span className="model-select-trigger-text">
            <span className="model-select-trigger-name">{displayLabel(model)}</span>
            {badge ? <span className="model-select-trigger-badge">{badge}</span> : null}
          </span>
          <svg className="model-select-chevron" width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        {open ? (
          <div className="model-select-menu" id={listId} role="listbox" aria-label="Choose model">
            <div className="model-select-group">
              <span className="model-select-group-label">Cloud</span>
              {MODEL_OPTIONS.filter((option) => option.group === "cloud").map((option) => (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={model === option.value}
                  className={model === option.value ? "model-select-option active" : "model-select-option"}
                  onClick={() => choosePreset(option.value)}
                >
                  <span className="model-select-option-label">{option.label}</span>
                  <span className="model-select-option-hint">{option.hint}</span>
                </button>
              ))}
            </div>

            <div className="model-select-divider" />

            <button
              type="button"
              role="option"
              aria-selected={customMode || !isPreset}
              className={customMode || !isPreset ? "model-select-option active" : "model-select-option"}
              onClick={enableCustom}
            >
              <span className="model-select-option-label">Custom model</span>
              <span className="model-select-option-hint">9router · manual</span>
            </button>
          </div>
        ) : null}

        {(customMode || !isPreset) && (
          <input
            className="model-select-custom"
            value={model}
            onChange={(event) => onModelChange(event.target.value)}
            placeholder="cx/gpt-5.6-sol"
            spellCheck={false}
          />
        )}
      </div>
    </div>
  );
}
