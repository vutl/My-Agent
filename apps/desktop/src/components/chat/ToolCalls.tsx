import { useState } from "react";
import { decideToolCall } from "../../lib/sse";
import type { ToolCallCard } from "../../lib/types";

const ICON: Record<ToolCallCard["status"], string> = {
  running: "⏳",
  waiting: "✋",
  done: "✓",
  failed: "⚠",
  declined: "✕",
};

function argumentText(value: unknown): string {
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 1) return String(entries[0][1]);
    return JSON.stringify(value, null, 2);
  }
  return String(value ?? "");
}

export function ToolCalls({ calls }: { calls: ToolCallCard[] }) {
  if (!calls.length) return null;
  return (
    <div className="tool-calls">
      {calls.map((call) => (
        <ToolCallItem key={call.callId} call={call} />
      ))}
    </div>
  );
}

function ToolCallItem({ call }: { call: ToolCallCard }) {
  const [sending, setSending] = useState(false);
  const decide = async (approved: boolean) => {
    setSending(true);
    try {
      await decideToolCall(call.runId, call.callId, approved);
    } finally {
      setSending(false);
    }
  };
  const detail = call.summary || argumentText(call.arguments);
  return (
    <div className={`tool-call tool-call--${call.status}`}>
      <div className="tool-call__head">
        <span className="tool-call__icon" aria-hidden="true">{ICON[call.status]}</span>
        <span className="tool-call__label">{call.label || call.name}</span>
        <span className="tool-call__summary" title={detail}>{detail}</span>
        {typeof call.ms === "number" && call.status !== "waiting" ? (
          <span className="tool-call__ms">{(call.ms / 1000).toFixed(1)}s</span>
        ) : null}
      </div>
      {call.status === "waiting" ? (
        <div className="tool-call__approval">
          <span>Aya muốn làm việc này — cậu cho phép không?</span>
          <pre className="tool-call__args">{argumentText(call.arguments)}</pre>
          <div className="tool-call__buttons">
            <button type="button" className="tool-call__approve" disabled={sending} onClick={() => void decide(true)}>
              Cho phép
            </button>
            <button type="button" disabled={sending} onClick={() => void decide(false)}>
              Từ chối
            </button>
          </div>
        </div>
      ) : null}
      {call.sources && call.sources.length > 0 && call.status === "done" ? (
        <div className="tool-call__sources">
          {call.sources.slice(0, 6).map((source) =>
            source.url ? (
              <a key={source.id} href={source.url} target="_blank" rel="noopener noreferrer" title={source.url}>
                [{source.id}] {source.title}
              </a>
            ) : null,
          )}
        </div>
      ) : null}
    </div>
  );
}
