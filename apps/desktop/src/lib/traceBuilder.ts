import type { TraceEvent } from "./types";

export type SpanStatus = "running" | "success" | "error" | "skipped";

export interface TraceSpan {
  id: string;
  name: string;
  kind: string;
  status: SpanStatus;
  durationMs: number | null;
  startedAt: string;
  summary: string;
  input?: unknown;
  output?: unknown;
  children: TraceSpan[];
}

export interface TraceStats {
  totalMs: number | null;
  spanCount: number;
  route: string | null;
  status: "running" | "success" | "error" | "idle";
  deltaCount: number;
  model: string | null;
}

let spanCounter = 0;

function nextId(): string {
  spanCounter += 1;
  return `span-${spanCounter}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
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

function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) {
    return "—";
  }
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

function collectTimings(events: TraceEvent[]): Map<string, number> {
  const timings = new Map<string, number>();
  for (const event of events) {
    if (event.event !== "timing") {
      continue;
    }
    const data = asRecord(event.data);
    const stage = stringValue(data.stage);
    if (stage) {
      timings.set(stage, Number(data.ms));
    }
  }
  return timings;
}

function countSpans(span: TraceSpan): number {
  return 1 + span.children.reduce((total, child) => total + countSpans(child), 0);
}

function maxDuration(span: TraceSpan): number {
  const own = span.durationMs ?? 0;
  return span.children.reduce((max, child) => Math.max(max, maxDuration(child)), own);
}

export function buildTraceTree(events: TraceEvent[]): { root: TraceSpan | null; stats: TraceStats; maxMs: number } {
  spanCounter = 0;
  const emptyStats: TraceStats = {
    totalMs: null,
    spanCount: 0,
    route: null,
    status: "idle",
    deltaCount: 0,
    model: null,
  };

  if (events.length === 0) {
    return { root: null, stats: emptyStats, maxMs: 0 };
  }

  const timings = collectTimings(events);
  const deltaCount = events.filter((event) => event.event === "message.delta").length;
  const startedAt = events[0]?.timestamp ?? "";
  const startedMs = events[0]?.at ?? null;
  const endedMs = events[events.length - 1]?.at ?? null;
  const wallMs = startedMs !== null && endedMs !== null ? Math.max(0, endedMs - startedMs) : null;

  const generationMs = timings.get("generation_total") ?? null;
  const totalMs = wallMs ?? generationMs;

  let route: string | null = null;
  let model: string | null = null;
  let runStatus: TraceStats["status"] = "running";

  const root: TraceSpan = {
    id: nextId(),
    name: "Agent Run",
    kind: "run",
    status: "running",
    durationMs: totalMs,
    startedAt,
    summary: "",
    children: [],
  };

  let retrievalSpan: TraceSpan | null = null;
  let graphSpan: TraceSpan | null = null;
  let generationSpan: TraceSpan | null = null;
  const openTools = new Map<string, TraceSpan>();

  for (const event of events) {
    const data = asRecord(event.data);

    switch (event.event) {
      case "run.started": {
        model = stringValue(data.model) || null;
        root.input = data;
        root.summary = model ? `model · ${model}` : "agent run";
        break;
      }
      case "agent.route.decided": {
        route = stringValue(data.route) || null;
        root.children.push({
          id: nextId(),
          name: "Route",
          kind: "route",
          status: "success",
          durationMs: timings.get("router") ?? null,
          startedAt: event.timestamp,
          summary: `${stringValue(data.route)} · ${(data.selected_tools as string[] | undefined)?.join(", ") || "no tools"}`,
          input: data,
          children: [],
        });
        break;
      }
      case "query.rewritten": {
        root.children.push({
          id: nextId(),
          name: "Query Rewrite",
          kind: "rewrite",
          status: "success",
          durationMs: timings.get("rewrite") ?? null,
          startedAt: event.timestamp,
          summary: [
            stringValue(data.standalone_query),
            data.is_followup ? "follow-up" : "direct",
            data.answer_intent ? `intent=${stringValue(data.answer_intent)}` : "",
            (data.focus_document_ids as string[] | undefined)?.length
              ? `focus=${(data.focus_document_ids as string[]).length} doc`
              : "",
          ]
            .filter(Boolean)
            .join(" · "),
          input: { original_query: data.original_query },
          output: data,
          children: [],
        });
        break;
      }
      case "retrieval.skipped": {
        root.children.push({
          id: nextId(),
          name: "Retrieval",
          kind: "retrieval",
          status: "skipped",
          durationMs: null,
          startedAt: event.timestamp,
          summary: stringValue(data.reason) || "skipped",
          output: data,
          children: [],
        });
        break;
      }
      case "retrieval.started": {
        retrievalSpan = {
          id: nextId(),
          name: "Retrieval",
          kind: "retrieval",
          status: "running",
          durationMs: timings.get("retrieval") ?? null,
          startedAt: event.timestamp,
          summary: stringValue(data.query) || "hybrid search",
          input: data,
          children: [],
        };
        root.children.push(retrievalSpan);
        break;
      }
      case "tool.started": {
        const toolName = stringValue(data.tool_name) || "tool";
        const toolSpan: TraceSpan = {
          id: nextId(),
          name: toolName,
          kind: "tool",
          status: "running",
          durationMs: null,
          startedAt: event.timestamp,
          summary: "running",
          input: data.input ?? data,
          children: [],
        };
        openTools.set(toolName, toolSpan);
        (retrievalSpan ?? root).children.push(toolSpan);
        break;
      }
      case "retrieval.retrying": {
        const retrySpan: TraceSpan = {
          id: nextId(),
          name: "Retry",
          kind: "retry",
          status: "running",
          durationMs: timings.get("retrieval.retry") ?? null,
          startedAt: event.timestamp,
          summary: stringValue(data.reason) || "retry",
          input: data,
          children: [],
        };
        (retrievalSpan ?? root).children.push(retrySpan);
        break;
      }
      case "retrieval.completed": {
        const documents = Array.isArray(data.documents) ? data.documents : [];
        const validation = asRecord(data.evidence_validation);
        if (retrievalSpan) {
          retrievalSpan.status = "success";
          retrievalSpan.durationMs = timings.get("retrieval.retry") ?? timings.get("retrieval") ?? retrievalSpan.durationMs;
          retrievalSpan.output = data;
          retrievalSpan.summary = [
            `${documents.length} sources`,
            stringValue(data.retrieval_mode) || "hybrid",
            validation.valid === false ? `invalid: ${stringValue(validation.reason)}` : "valid",
            data.retry_performed ? "retried" : "",
          ]
            .filter(Boolean)
            .join(" · ");
        } else {
          root.children.push({
            id: nextId(),
            name: "Retrieval",
            kind: "retrieval",
            status: "success",
            durationMs: timings.get("retrieval") ?? null,
            startedAt: event.timestamp,
            summary: `${documents.length} sources`,
            output: data,
            children: [],
          });
        }
        break;
      }
      case "tool.completed": {
        const toolName = stringValue(data.tool_name) || "tool";
        const toolSpan = openTools.get(toolName);
        if (toolSpan) {
          toolSpan.status = data.fallback ? "skipped" : "success";
          toolSpan.output = data;
          toolSpan.summary = [
            data.result_count !== undefined ? `${stringValue(data.result_count)} hits` : "done",
            data.fallback ? "fallback" : "",
          ]
            .filter(Boolean)
            .join(" · ");
          openTools.delete(toolName);
        }
        break;
      }
      case "tool.fallback.started": {
        root.children.push({
          id: nextId(),
          name: "RAG Fallback",
          kind: "fallback",
          status: "running",
          durationMs: timings.get("retrieval.fallback") ?? null,
          startedAt: event.timestamp,
          summary: stringValue(data.reason) || "fallback",
          input: data,
          children: [],
        });
        break;
      }
      case "agent.event": {
        const graphEvent = stringValue(data.event);
        if (graphEvent === "graph.started" && !graphSpan) {
          graphSpan = {
            id: nextId(),
            name: "LangGraph",
            kind: "graph",
            status: "running",
            durationMs: timings.get("graph") ?? null,
            startedAt: event.timestamp,
            summary: "router → planner → final_prompt",
            children: [],
          };
          root.children.push(graphSpan);
        }
        if (graphEvent.endsWith(".started") || graphEvent.endsWith(".completed")) {
          if (!graphSpan) {
            graphSpan = {
              id: nextId(),
              name: "LangGraph",
              kind: "graph",
              status: "running",
              durationMs: timings.get("graph") ?? null,
              startedAt: event.timestamp,
              summary: "graph nodes",
              children: [],
            };
            root.children.push(graphSpan);
          }
          const nodeName = graphEvent.replace(/\.(started|completed)$/, "");
          if (graphEvent.endsWith(".started")) {
            graphSpan.children.push({
              id: nextId(),
              name: nodeName,
              kind: "node",
              status: "running",
              durationMs: null,
              startedAt: event.timestamp,
              summary: "running",
              input: data,
              children: [],
            });
          } else {
            const existing = [...graphSpan.children].reverse().find((child) => child.name === nodeName);
            if (existing) {
              existing.status = "success";
              existing.output = data;
              if (nodeName === "planner" && Array.isArray(data.plan)) {
                existing.summary = `${data.plan.length} steps`;
              } else if (nodeName === "router") {
                existing.summary = `${stringValue(data.route)} · ${stringValue(data.mode)}`;
              } else if (nodeName === "final_prompt") {
                existing.summary = "prompt composed";
              } else {
                existing.summary = "done";
              }
            } else {
              graphSpan.children.push({
                id: nextId(),
                name: nodeName,
                kind: "node",
                status: "success",
                durationMs: null,
                startedAt: event.timestamp,
                summary: "done",
                output: data,
                children: [],
              });
            }
          }
        }
        if (graphEvent === "graph.completed" && graphSpan) {
          graphSpan.status = "success";
          graphSpan.durationMs = timings.get("graph") ?? graphSpan.durationMs;
          graphSpan.output = data;
          graphSpan.summary = `${stringValue(data.route)} · ${timings.get("graph") ? formatMs(timings.get("graph")) : "done"}`;
        }
        break;
      }
      case "planner.completed": {
        if (!graphSpan) {
          graphSpan = {
            id: nextId(),
            name: "LangGraph",
            kind: "graph",
            status: "success",
            durationMs: timings.get("graph") ?? null,
            startedAt: event.timestamp,
            summary: "planner",
            children: [],
          };
          root.children.push(graphSpan);
        }
        graphSpan.children.push({
          id: nextId(),
          name: "planner",
          kind: "node",
          status: "success",
          durationMs: null,
          startedAt: event.timestamp,
          summary: `${(data.plan as string[] | undefined)?.length ?? 0} steps`,
          output: data,
          children: [],
        });
        break;
      }
      case "timing": {
        const stage = stringValue(data.stage);
        const isFirstVisibleToken = stage === "first_token" || stage === "first_validated_token";
        if (isFirstVisibleToken && !generationSpan) {
          generationSpan = {
            id: nextId(),
            name: "Generation",
            kind: "generation",
            status: "running",
            durationMs: timings.get("generation_total") ?? null,
            startedAt: event.timestamp,
            summary: `${stage === "first_validated_token" ? "first validated token" : "first token"} ${formatMs(Number(data.ms))}`,
            children: [],
          };
          root.children.push(generationSpan);
        }
        if (isFirstVisibleToken && generationSpan) {
          generationSpan.children.push({
            id: nextId(),
            name: stage === "first_validated_token" ? "First Validated Token" : "First Token",
            kind: "timing",
            status: "success",
            durationMs: Number(data.ms),
            startedAt: event.timestamp,
            summary: formatMs(Number(data.ms)),
            output: data,
            children: [],
          });
        }
        break;
      }
      case "message.finished": {
        if (!generationSpan) {
          generationSpan = {
            id: nextId(),
            name: "Generation",
            kind: "generation",
            status: "success",
            durationMs: timings.get("generation_total") ?? null,
            startedAt: event.timestamp,
            summary: "completed",
            children: [],
          };
          root.children.push(generationSpan);
        }
        generationSpan.status = "success";
        generationSpan.durationMs = timings.get("generation_total") ?? generationSpan.durationMs;
        generationSpan.output = data;
        const metrics = asRecord(data.metrics);
        generationSpan.summary = [
          stringValue(data.finish_reason) || "stop",
          data.eval_count !== undefined ? `${stringValue(data.eval_count)} tokens` : "",
          metrics.tokens_per_second ? `${stringValue(metrics.tokens_per_second)} tok/s` : "",
          data.truncated ? "truncated" : "",
        ]
          .filter(Boolean)
          .join(" · ");
        break;
      }
      case "run.completed": {
        root.status = "success";
        runStatus = "success";
        root.durationMs = totalMs;
        break;
      }
      case "run.failed": {
        root.status = "error";
        runStatus = "error";
        root.output = data;
        root.summary = [root.summary, stringValue(data.error)].filter(Boolean).join(" · ");
        break;
      }
      default:
        break;
    }
  }

  if (retrievalSpan?.status === "running") {
    retrievalSpan.status = "success";
  }
  if (graphSpan?.status === "running") {
    graphSpan.status = "success";
  }
  if (generationSpan?.status === "running") {
    generationSpan.status = "success";
  }
  if (root.status === "running" && runStatus !== "error") {
    root.status = events.some((event) => event.event === "run.completed") ? "success" : "running";
  }

  const stats: TraceStats = {
    totalMs,
    spanCount: countSpans(root),
    route,
    status:
      runStatus === "error"
        ? "error"
        : events.some((event) => event.event === "run.completed")
          ? "success"
          : "running",
    deltaCount,
    model,
  };

  return { root, stats, maxMs: maxDuration(root) };
}

export function flattenSpans(span: TraceSpan, depth = 0): Array<{ span: TraceSpan; depth: number }> {
  return [{ span, depth }, ...span.children.flatMap((child) => flattenSpans(child, depth + 1))];
}

export function buildCollapsedTraceSummary(
  events: TraceEvent[],
  streamDeltaCount: number
): { root: TraceSpan | null; stats: TraceStats; maxMs: number } {
  if (events.length === 0) {
    return {
      root: null,
      stats: {
        totalMs: null,
        spanCount: 0,
        route: null,
        status: streamDeltaCount > 0 ? "running" : "idle",
        deltaCount: streamDeltaCount,
        model: null,
      },
      maxMs: 0,
    };
  }

  const startedMs = events[0]?.at ?? null;
  const endedMs = events[events.length - 1]?.at ?? null;
  const routeEvent = [...events].reverse().find((event) => event.event === "agent.route.decided");
  const route = routeEvent ? stringValue(asRecord(routeEvent.data).route) || null : null;
  const runStarted = events.find((event) => event.event === "run.started");
  const model = runStarted ? stringValue(asRecord(runStarted.data).model) || null : null;
  const failed = events.some((event) => event.event === "run.failed");
  const completed = events.some((event) => event.event === "run.completed");

  return {
    root: null,
    stats: {
      totalMs: startedMs !== null && endedMs !== null ? Math.max(0, endedMs - startedMs) : null,
      spanCount: events.filter((event) => event.event !== "timing").length,
      route,
      status: failed ? "error" : completed ? "success" : "running",
      deltaCount: streamDeltaCount,
      model,
    },
    maxMs: 0,
  };
}

export { formatMs };
