import { memo, useMemo, useState } from "react";

import { buildCollapsedTraceSummary, buildTraceTree, flattenSpans, formatMs, type TraceSpan } from "../../lib/traceBuilder";
import type { AgentDebugTraceResponse, TraceEvent } from "../../lib/types";

interface AgentTracePanelProps {
  events: TraceEvent[];
  status: string;
  streamDeltaCount: number;
  getDebugOutput: () => string;
  debugTrace?: AgentDebugTraceResponse | null;
}

export const AgentTracePanel = memo(function AgentTracePanel({
  events,
  status,
  streamDeltaCount,
  getDebugOutput,
  debugTrace,
}: AgentTracePanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { root, stats, maxMs } = useMemo(
    () => (expanded ? buildTraceTree(events) : buildCollapsedTraceSummary(events, streamDeltaCount)),
    [events, expanded, streamDeltaCount]
  );
  const rows = useMemo(() => (expanded && root ? flattenSpans(root) : []), [expanded, root]);
  const selected = useMemo(
    () => rows.find((row) => row.span.id === selectedId)?.span ?? root,
    [rows, selectedId, root]
  );

  const headline =
    stats.route || stats.totalMs
      ? `${stats.route ?? "agent"} · ${formatMs(stats.totalMs)} · ${expanded ? stats.spanCount : events.length} steps`
      : streamDeltaCount > 0
        ? `streaming · ${streamDeltaCount} chunks`
        : "No trace yet";

  const displayStats = expanded ? stats : { ...stats, deltaCount: streamDeltaCount };

  return (
    <aside className={expanded ? "trace-panel expanded" : "trace-panel"}>
      <button className="trace-toggle" onClick={() => setExpanded((value) => !value)} type="button">
        <span className="trace-toggle-label">
          <span className={`trace-status-dot ${displayStats.status}`} aria-hidden="true" />
          {expanded ? "Hide Trace" : "Trace"}
        </span>
        <span className="trace-toggle-meta">{headline}</span>
        {!expanded && streamDeltaCount > 0 ? (
          <span className="status-pill">{streamDeltaCount} tok</span>
        ) : null}
        <span className="status-pill">{status}</span>
      </button>

      {expanded && root ? (
        <div className="trace-body">
          <div className="trace-summary-bar">
            <div className="trace-summary-item">
              <span>Route</span>
              <strong>{stats.route ?? "—"}</strong>
            </div>
            <div className="trace-summary-item">
              <span>Total</span>
              <strong>{formatMs(stats.totalMs)}</strong>
            </div>
            <div className="trace-summary-item">
              <span>Spans</span>
              <strong>{stats.spanCount}</strong>
            </div>
            <div className="trace-summary-item">
              <span>Stream</span>
              <strong>{streamDeltaCount} chunks</strong>
            </div>
            <div className="trace-summary-item">
              <span>Model</span>
              <strong>{stats.model ?? "—"}</strong>
            </div>
          </div>

          <div className="trace-layout">
            <div className="trace-tree" role="tree" aria-label="Agent trace">
              {rows.map(({ span, depth }) => (
                <TraceSpanRow
                  key={span.id}
                  depth={depth}
                  maxMs={maxMs}
                  selected={selected?.id === span.id}
                  span={span}
                  onSelect={() => setSelectedId(span.id)}
                />
              ))}
            </div>

            <div className="trace-detail">
              {selected ? (
                <TraceSpanDetail debugTrace={debugTrace} getDebugOutput={getDebugOutput} span={selected} />
              ) : (
                <div className="trace-empty">Select a span</div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
});

function TraceSpanRow({
  span,
  depth,
  maxMs,
  selected,
  onSelect,
}: {
  span: TraceSpan;
  depth: number;
  maxMs: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const barWidth = span.durationMs && maxMs > 0 ? Math.max(6, Math.round((span.durationMs / maxMs) * 100)) : 0;

  return (
    <button
      className={selected ? "trace-span-row selected" : "trace-span-row"}
      onClick={onSelect}
      style={{ paddingLeft: `${12 + depth * 16}px` }}
      type="button"
    >
      <span className={`trace-span-status ${span.status}`} aria-hidden="true" />
      <span className="trace-span-main">
        <span className="trace-span-name">{span.name}</span>
        <span className="trace-span-summary">{span.summary}</span>
      </span>
      {barWidth > 0 ? (
        <span className="trace-span-bar-wrap" aria-hidden="true">
          <span className="trace-span-bar" style={{ width: `${barWidth}%` }} />
        </span>
      ) : (
        <span className="trace-span-bar-wrap" />
      )}
      <span className="trace-span-duration">{formatMs(span.durationMs)}</span>
    </button>
  );
}

function TraceSpanDetail({
  span,
  getDebugOutput,
  debugTrace,
}: {
  span: TraceSpan;
  getDebugOutput: () => string;
  debugTrace?: AgentDebugTraceResponse | null;
}) {
  const [tab, setTab] = useState<"summary" | "input" | "output" | "raw" | "debug">("summary");
  const output = getDebugOutput();
  const tabs = debugTrace
    ? (["summary", "input", "output", "raw", "debug"] as const)
    : (["summary", "input", "output", "raw"] as const);

  return (
    <div className="trace-detail-panel">
      <div className="trace-detail-header">
        <div>
          <strong>{span.name}</strong>
          <span className="trace-detail-kind">{span.kind}</span>
        </div>
        <span className="trace-detail-duration">{formatMs(span.durationMs)}</span>
      </div>

      <div className="trace-detail-tabs">
        {tabs.map((item) => (
          <button
            key={item}
            className={tab === item ? "trace-detail-tab active" : "trace-detail-tab"}
            onClick={() => setTab(item)}
            type="button"
          >
            {item}
          </button>
        ))}
      </div>

      <div className="trace-detail-content">
        {tab === "summary" ? (
          <div className="trace-detail-summary">
            <p>{span.summary || "—"}</p>
            <dl>
              <dt>Status</dt>
              <dd>{span.status}</dd>
              <dt>Started</dt>
              <dd>{span.startedAt}</dd>
              {span.children.length > 0 ? (
                <>
                  <dt>Children</dt>
                  <dd>{span.children.length}</dd>
                </>
              ) : null}
            </dl>
            {span.kind === "generation" && output ? (
              <>
                <h4>Live stream preview</h4>
                <pre className="trace-preview">{truncate(output, 1200)}</pre>
              </>
            ) : null}
            {span.kind === "retrieval" && span.output ? (
              <RetrievalPreview output={span.output} />
            ) : null}
          </div>
        ) : null}
        {tab === "input" ? <JsonBlock value={span.input} emptyLabel="No input captured" /> : null}
        {tab === "output" ? <JsonBlock value={span.output} emptyLabel="No output captured" /> : null}
        {tab === "raw" ? <JsonBlock value={span} emptyLabel="No data" /> : null}
        {tab === "debug" ? <JsonBlock value={debugTrace} emptyLabel="Debug trace was not captured" /> : null}
      </div>
    </div>
  );
}

function RetrievalPreview({ output }: { output: unknown }) {
  const data = output as Record<string, unknown>;
  const documents = Array.isArray(data.documents) ? data.documents : [];
  return (
    <div className="trace-retrieval-preview">
      <h4>Sources ({documents.length})</h4>
      <ul>
        {documents.slice(0, 8).map((document, index) => {
          const item = document as Record<string, unknown>;
          const page =
            item.page_number === null || item.page_number === undefined ? "" : ` · p${item.page_number}`;
          return (
            <li key={String(item.chunk_id ?? index)}>
              <strong>{stringValue(item.source_id) || `SOURCE ${index + 1}`}</strong>
              <span>
                {stringValue(item.filename)}
                {page}
              </span>
              <small>{truncate(stringValue(item.content), 140)}</small>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function JsonBlock({ value, emptyLabel }: { value: unknown; emptyLabel: string }) {
  if (value === undefined || value === null) {
    return <div className="trace-empty">{emptyLabel}</div>;
  }
  return <pre className="trace-json">{JSON.stringify(value, null, 2)}</pre>;
}

function truncate(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}…`;
}

function stringValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}
