import { API_BASE_URL } from "./api";
import type { AgentStreamEvent, ChatStreamEvent } from "./types";

export interface StreamChatInput {
  conversationId: string | null;
  message: string;
  model: string;
  temperature: number;
  signal?: AbortSignal;
}

export async function* streamChat(input: StreamChatInput): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: input.conversationId,
      message: input.message,
      model: input.model,
      temperature: input.temperature
    }),
    signal: input.signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  yield* readEventStream<ChatStreamEvent>(response.body);
}

export interface StreamAgentInput {
  conversationId: string | null;
  task: string;
  mode: string;
  model: string;
  temperature: number;
  collectionId?: string | null;
  retrievalMode?: "auto" | "hybrid" | "fts";
  agentReasoning?: "auto" | "fast" | "smart";
  debugTrace?: boolean;
  signal?: AbortSignal;
}

export async function* streamAgent(input: StreamAgentInput): AsyncGenerator<AgentStreamEvent> {
  const response = await fetch(`${API_BASE_URL}/agent/run/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: input.conversationId,
      task: input.task,
      mode: input.mode,
      model: input.model,
      temperature: input.temperature,
      allowed_tools: ["search_local_docs", "retrieve_visual_assets"],
      require_confirmation: true,
      collection_id: input.collectionId || null,
      retrieval_mode: input.retrievalMode ?? "auto",
      agent_reasoning: input.agentReasoning ?? "auto",
      debug_trace: input.debugTrace ?? false
    }),
    signal: input.signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  yield* readEventStream<AgentStreamEvent>(response.body);
}

async function* readEventStream<TEvent>(body: ReadableStream<Uint8Array>): AsyncGenerator<TEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const event = parseEvent(part);
      if (event) {
        for (const visibleEvent of splitVisibleMessageDelta(event)) {
          yield visibleEvent as TEvent;
          // A validated/table answer can already be complete when the browser
          // receives it, so several SSE deltas may be decoded in one JavaScript
          // turn.  Without yielding to the renderer, React sees every state
          // update but the user only sees the final one.  Pace only visible
          // message output; trace/control events remain unthrottled.
          if (visibleEvent.event === "message.delta") {
            await waitForVisiblePaint();
          }
        }
      }
    }
  }

  const trailing = parseEvent(buffer);
  if (trailing) {
    for (const visibleEvent of splitVisibleMessageDelta(trailing)) {
      yield visibleEvent as TEvent;
      if (visibleEvent.event === "message.delta") {
        await waitForVisiblePaint();
      }
    }
  }
}

const MAX_VISIBLE_DELTA_CHARS = 64;

function splitVisibleMessageDelta(
  event: ChatStreamEvent | AgentStreamEvent
): Array<ChatStreamEvent | AgentStreamEvent> {
  if (event.event !== "message.delta") return [event];
  const characters = Array.from(event.data.delta);
  if (characters.length <= MAX_VISIBLE_DELTA_CHARS) return [event];

  const events: Array<ChatStreamEvent | AgentStreamEvent> = [];
  for (let offset = 0; offset < characters.length; offset += MAX_VISIBLE_DELTA_CHARS) {
    events.push({
      event: "message.delta",
      data: { delta: characters.slice(offset, offset + MAX_VISIBLE_DELTA_CHARS).join("") },
    });
  }
  return events;
}

function waitForVisiblePaint(): Promise<void> {
  if (
    typeof document !== "undefined" &&
    document.visibilityState === "visible" &&
    typeof requestAnimationFrame === "function"
  ) {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  return Promise.resolve();
}

function parseEvent(raw: string): ChatStreamEvent | AgentStreamEvent | null {
  const lines = raw.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event: "));
  const dataLines = lines.filter((line) => line.startsWith("data: "));

  if (!eventLine || dataLines.length === 0) {
    return null;
  }

  return {
    event: eventLine.slice("event: ".length),
    data: JSON.parse(dataLines.map((line) => line.slice("data: ".length)).join("\n"))
  } as ChatStreamEvent;
}
