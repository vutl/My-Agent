import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

export type AgentReasoningMode = "auto" | "fast" | "smart";

const REASONING_OPTIONS: Array<{
  value: AgentReasoningMode;
  label: string;
  hint: string;
  description: string;
}> = [
  {
    value: "auto",
    label: "Tự động",
    hint: "Adaptive",
    description: "Chỉ search lại khi so sánh paper hoặc hỏi hình.",
  },
  {
    value: "fast",
    label: "Nhanh",
    hint: "Single-pass",
    description: "Một lượt retrieval, phản hồi nhanh nhất.",
  },
  {
    value: "smart",
    label: "Sâu",
    hint: "Thorough",
    description: "Search lại khi thiếu evidence hoặc thiếu figure.",
  },
];

interface ComposerProps {
  value: string;
  disabled: boolean;
  streaming?: boolean;
  agentReasoning: AgentReasoningMode;
  debugTrace: boolean;
  debugTraceAvailable: boolean;
  onAgentReasoningChange: (value: AgentReasoningMode) => void;
  onDebugTraceChange: (value: boolean) => void;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel?: () => void;
}

export function Composer({
  value,
  disabled,
  streaming = false,
  agentReasoning,
  debugTrace,
  debugTraceAvailable,
  onAgentReasoningChange,
  onDebugTraceChange,
  onChange,
  onSubmit,
  onCancel,
}: ComposerProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const menuId = useId();
  const activeOption =
    REASONING_OPTIONS.find((option) => option.value === agentReasoning) ?? REASONING_OPTIONS[0];
  const inputDisabled = disabled || streaming;

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    const nextHeight = Math.min(textarea.scrollHeight, 140);
    textarea.style.height = `${nextHeight}px`;
  }, [value]);

  useEffect(() => {
    if (!menuOpen) return;

    function handlePointerDown(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }

    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onSubmit();
  }

  function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  }

  function selectMode(next: AgentReasoningMode) {
    onAgentReasoningChange(next);
    setMenuOpen(false);
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer-inner">
        <div className={`composer-card${menuOpen ? " menu-open" : ""}`}>
          <div className="composer-body">
            <div className="composer-retrieval" ref={menuRef}>
              <button
                type="button"
                className="composer-retrieval-trigger"
                disabled={inputDisabled}
                aria-haspopup="listbox"
                aria-expanded={menuOpen}
                aria-controls={menuId}
                title={`Retrieval: ${activeOption.label}`}
                onClick={() => setMenuOpen((open) => !open)}
              >
                <RetrievalIcon />
                <span className="composer-retrieval-value">{activeOption.label}</span>
                <ChevronIcon open={menuOpen} />
              </button>

              {menuOpen && (
                <div className="composer-retrieval-menu" id={menuId} role="listbox" aria-label="Chế độ retrieval">
                  {REASONING_OPTIONS.map((option) => {
                    const selected = option.value === agentReasoning;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        role="option"
                        aria-selected={selected}
                        className={`composer-retrieval-option${selected ? " selected" : ""}`}
                        onClick={() => selectMode(option.value)}
                      >
                        <span className="composer-retrieval-option-head">
                          <span className="composer-retrieval-option-label">{option.label}</span>
                          <span className="composer-retrieval-option-hint">{option.hint}</span>
                        </span>
                        <span className="composer-retrieval-option-desc">{option.description}</span>
                      </button>
                    );
                  })}
                  <button
                    type="button"
                    role="switch"
                    aria-checked={debugTrace}
                    className={`composer-debug-toggle${debugTrace ? " selected" : ""}`}
                    disabled={!debugTraceAvailable || inputDisabled}
                    title={
                      debugTraceAvailable
                        ? "Lưu prompt/draft đã redact cho run này"
                        : "Bật AGENT_DEBUG_TRACE_ENABLED trên backend để dùng"
                    }
                    onClick={() => onDebugTraceChange(!debugTrace)}
                  >
                    <span>
                      <strong>Debug trace</strong>
                      <small>Prompt + draft đã redact, tự xoá</small>
                    </span>
                    <span className="composer-debug-switch" aria-hidden="true" />
                  </button>
                </div>
              )}
            </div>

            <textarea
              ref={textareaRef}
              className="composer-input"
              value={value}
              disabled={inputDisabled}
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={handleTextareaKeyDown}
              placeholder="Nhắn Aya…"
              rows={1}
            />

            {streaming ? (
              <button
                className="send-button composer-send cancel-button"
                onClick={onCancel}
                type="button"
                aria-label="Dừng phản hồi"
                title="Dừng phản hồi"
              >
                <span className="cancel-button-mark" aria-hidden="true" />
              </button>
            ) : (
              <button
                className="send-button composer-send"
                disabled={disabled || value.trim().length === 0}
                type="submit"
                aria-label="Gửi"
              >
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M15.5 9L3 15.5L5.5 9L3 2.5L15.5 9Z"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth="1"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>
        <p className="composer-hint">
          {streaming ? "Có thể dừng phản hồi đang chạy" : "Enter gửi · Shift+Enter xuống dòng"}
        </p>
      </div>
    </form>
  );
}

function RetrievalIcon() {
  return (
    <svg
      className="composer-retrieval-icon"
      width="15"
      height="15"
      viewBox="0 0 15 15"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M2.5 4.5h10M2.5 7.5h7M2.5 10.5h4"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <circle cx="11.5" cy="10.5" r="2" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      className={`composer-retrieval-chevron${open ? " open" : ""}`}
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      aria-hidden="true"
    >
      <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}
