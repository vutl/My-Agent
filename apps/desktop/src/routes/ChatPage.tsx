import { memo, useEffect, useMemo, useRef, useState, type RefObject } from "react";

import { AgentTracePanel } from "../components/agent/AgentTracePanel";
import { ChatWindow } from "../components/chat/ChatWindow";
import { Composer, type AgentReasoningMode } from "../components/chat/Composer";
import { getAgentDebugTrace, getHealth, listConversations, listMessages } from "../lib/api";
import { streamAgent } from "../lib/sse";
import type {
  ChatMessage,
  AgentDebugTraceResponse,
  Conversation,
  HealthDependency,
  HealthResponse,
  RetrievedDocument,
  StoredMessage,
  TraceEvent,
} from "../lib/types";

interface ChatPageProps {
  model: string;
}

const AGENT_REASONING_STORAGE_KEY = "aya.agent_reasoning";
const ACTIVE_CONVERSATION_STORAGE_KEY = "aya.active_conversation";
const NEW_CONVERSATION_STORAGE_VALUE = "__new__";
const DEBUG_TRACE_STORAGE_KEY = "aya.debug_trace";

function loadAgentReasoningMode(): AgentReasoningMode {
  const stored = localStorage.getItem(AGENT_REASONING_STORAGE_KEY);
  if (stored === "auto" || stored === "fast" || stored === "smart") {
    return stored;
  }
  return "auto";
}

function loadDebugTraceEnabled(): boolean {
  return localStorage.getItem(DEBUG_TRACE_STORAGE_KEY) === "true";
}

export function ChatPage({ model }: ChatPageProps) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("checking backend");
  const [historyError, setHistoryError] = useState("");
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [streamDeltaCount, setStreamDeltaCount] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingConversation, setIsLoadingConversation] = useState(true);
  const [agentReasoning, setAgentReasoning] = useState<AgentReasoningMode>(() => loadAgentReasoningMode());
  const [debugTrace, setDebugTrace] = useState(() => loadDebugTraceEnabled());
  const [debugTraceAvailable, setDebugTraceAvailable] = useState(false);
  const [debugTracePayload, setDebugTracePayload] = useState<AgentDebugTraceResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const currentSourcesRef = useRef<RetrievedDocument[]>([]);
  const deltaCountRef = useRef(0);
  const debugOutputRef = useRef("");
  const mountedRef = useRef(true);
  const conversationLoadVersionRef = useRef(0);
  const idleStatusRef = useRef("backend online · 9router unverified");

  const canSend = useMemo(
    () => input.trim().length > 0 && !isStreaming && !isLoadingConversation,
    [input, isLoadingConversation, isStreaming]
  );

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;

    async function loadInitialState() {
      const [healthResult, conversationResult] = await Promise.allSettled([
        getHealth(),
        listConversations(),
      ]);
      if (cancelled) return;

      if (healthResult.status === "fulfilled") {
        const nextStatus = describeHealth(healthResult.value);
        const traceAvailable = Boolean(healthResult.value.agent_debug_trace?.enabled);
        setDebugTraceAvailable(traceAvailable);
        if (!traceAvailable) setDebugTrace(false);
        idleStatusRef.current = nextStatus;
        setStatus(nextStatus);
      } else {
        setStatus(errorMessage(healthResult.reason, "backend offline"));
      }

      if (conversationResult.status === "rejected") {
        setHistoryError(errorMessage(conversationResult.reason, "Không tải được lịch sử chat"));
        setIsLoadingConversation(false);
        return;
      }

      const available = conversationResult.value;
      setConversations(available);
      const savedId = localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
      const selected = savedId === NEW_CONVERSATION_STORAGE_VALUE
        ? undefined
        : available.find((conversation) => conversation.id === savedId) ?? available[0];
      if (!selected) {
        setIsLoadingConversation(false);
        return;
      }

      setConversationId(selected.id);
      localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, selected.id);
      try {
        const storedMessages = await listMessages(selected.id);
        if (cancelled) return;
        setMessages(storedMessages.map(toChatMessage));
      } catch (error) {
        if (cancelled) return;
        setHistoryError(errorMessage(error, "Không tải được tin nhắn"));
      } finally {
        if (!cancelled) setIsLoadingConversation(false);
      }
    }

    void loadInitialState();
    return () => {
      cancelled = true;
      mountedRef.current = false;
      conversationLoadVersionRef.current += 1;
      abortRef.current?.abort();
    };
    // Initial hydration is intentionally run once. Later model changes should not overwrite run errors.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pushTraceEvent(event: TraceEvent) {
    if (mountedRef.current) setTraceEvents((current) => [...current, event]);
  }

  async function loadConversationHistory(selectedId: string) {
    if (isStreaming || isLoadingConversation || selectedId === conversationId) return;
    const loadVersion = conversationLoadVersionRef.current + 1;
    conversationLoadVersionRef.current = loadVersion;
    setConversationId(selectedId);
    setMessages([]);
    setTraceEvents([]);
    setDebugTracePayload(null);
    setStreamDeltaCount(0);
    setHistoryError("");
    setIsLoadingConversation(true);
    localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, selectedId);

    try {
      const storedMessages = await listMessages(selectedId);
      if (!mountedRef.current || conversationLoadVersionRef.current !== loadVersion) return;
      setMessages(storedMessages.map(toChatMessage));
    } catch (error) {
      if (!mountedRef.current || conversationLoadVersionRef.current !== loadVersion) return;
      setHistoryError(errorMessage(error, "Không tải được tin nhắn"));
    } finally {
      if (mountedRef.current && conversationLoadVersionRef.current === loadVersion) {
        setIsLoadingConversation(false);
      }
    }
  }

  function startNewConversation() {
    if (isStreaming || isLoadingConversation) return;
    conversationLoadVersionRef.current += 1;
    setConversationId(null);
    setMessages([]);
    setInput("");
    setTraceEvents([]);
    setDebugTracePayload(null);
    setStreamDeltaCount(0);
    setHistoryError("");
    setIsLoadingConversation(false);
    localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, NEW_CONVERSATION_STORAGE_VALUE);
  }

  async function refreshConversationList() {
    try {
      const available = await listConversations();
      if (mountedRef.current) setConversations(available);
    } catch (error) {
      if (mountedRef.current) {
        setHistoryError(errorMessage(error, "Không refresh được lịch sử chat"));
      }
    }
  }

  function cancelStream() {
    if (!abortRef.current || abortRef.current.signal.aborted) return;
    setStatus("đang dừng phản hồi");
    abortRef.current.abort();
  }

  async function sendMessage() {
    const content = input.trim();
    if (!content || !canSend) return;

    const controller = new AbortController();
    abortRef.current = controller;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      model,
      pending: true,
    };

    setInput("");
    setIsStreaming(true);
    setStatus("streaming");
    setTraceEvents([]);
    setDebugTracePayload(null);
    setStreamDeltaCount(0);
    deltaCountRef.current = 0;
    debugOutputRef.current = "";
    currentSourcesRef.current = [];
    setMessages((current) => [...current, userMessage, assistantMessage]);

    let activeConversationId = conversationId;
    let activeRunId: string | null = null;
    try {
      let receivedDelta = false;
      let streamError = "";
      let runCompleted = false;
      const stream = streamAgent({
        conversationId,
        task: content,
        mode: "auto",
        model,
        temperature: 0.2,
        collectionId: null,
        retrievalMode: "auto",
        agentReasoning,
        debugTrace,
        signal: controller.signal,
      });

      for await (const event of stream) {
        if (controller.signal.aborted) break;
        if (event.event !== "message.delta") {
          pushTraceEvent({
            event: event.event,
            data: event.data,
            timestamp: new Date().toLocaleTimeString(),
            at: Date.now(),
          });
        }

        if (event.event === "run.started") {
          activeRunId = event.data.run_id;
          activeConversationId = event.data.conversation_id;
          setConversationId(activeConversationId);
          localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, activeConversationId);
        }

        if (event.event === "message.delta") {
          receivedDelta = true;
          deltaCountRef.current += 1;
          debugOutputRef.current += event.data.delta;
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + event.data.delta }
                : message
            )
          );
          if (deltaCountRef.current % 12 === 0) {
            setStreamDeltaCount(deltaCountRef.current);
          }
        }

        if (event.event === "timing" && event.data.stage === "first_validated_token") {
          const streamingMode = String(event.data.streaming_mode || "");
          if (event.data.direct_canonical_table) {
            setStatus("đang hiển thị bảng đã kiểm chứng");
          } else if (streamingMode === "buffered_validation") {
            setStatus("đang hiển thị câu trả lời đã kiểm chứng");
          } else if (streamingMode === "validated_blocks") {
            setStatus("đang stream từng phần đã kiểm chứng");
          } else if (streamingMode === "validated_paper_sections") {
            setStatus("đang stream từng paper đã kiểm chứng");
          }
        }

        if (event.event === "evidence.paper.ready") {
          setStatus("đã chuẩn bị evidence theo paper");
        }

        if (event.event === "retrieval.completed") {
          currentSourcesRef.current = event.data.documents;
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, sources: event.data.documents } : message
            )
          );
        }

        if (event.event === "run.failed") {
          streamError = event.data.error || "Agent run failed";
          setStatus(`failed · ${streamError}`);
        }

        if (event.event === "run.completed") {
          runCompleted = true;
        }
      }

      if (!mountedRef.current) return;
      setStreamDeltaCount(deltaCountRef.current);

      if (debugTrace && activeRunId && runCompleted) {
        try {
          setDebugTracePayload(await getAgentDebugTrace(activeRunId));
        } catch (error) {
          setDebugTracePayload({
            run_id: activeRunId,
            schema_version: 1,
            size_bytes: 0,
            redaction_count: 0,
            truncated: false,
            created_at: "",
            updated_at: "",
            expires_at: "",
            payload: { error: errorMessage(error, "Không tải được debug trace") },
          });
        }
      }

      if (controller.signal.aborted) {
        finishAssistant(assistantId, "Đã dừng phản hồi.");
        setStatus("đã dừng");
      } else if (streamError) {
        finishAssistant(assistantId, `Không hoàn tất: ${streamError}`);
        setStatus(`failed · ${streamError}`);
      } else if (!runCompleted) {
        const message = "Kết nối stream kết thúc trước khi agent báo hoàn tất.";
        finishAssistant(assistantId, message);
        setStatus("stream interrupted");
      } else if (!receivedDelta) {
        const message = "Agent báo hoàn tất nhưng không trả nội dung.";
        finishAssistant(assistantId, message);
        setStatus("completed without output");
      } else {
        finishAssistant(assistantId);
        setStatus(idleStatusRef.current);
      }
    } catch (error) {
      if (!mountedRef.current) return;
      if (controller.signal.aborted || isAbortError(error)) {
        finishAssistant(assistantId, "Đã dừng phản hồi.");
        setStatus("đã dừng");
      } else {
        const message = errorMessage(error, "unknown error");
        finishAssistant(assistantId, `Request failed: ${message}`);
        setStatus(`failed · ${message}`);
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      if (mountedRef.current) {
        setIsStreaming(false);
        if (activeConversationId) void refreshConversationList();
      }
    }
  }

  function finishAssistant(assistantId: string, failureNote?: string) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== assistantId) return message;
        const failureContent = failureNote ? `_${failureNote}_` : "";
        return {
          ...message,
          content: failureNote
            ? message.content
              ? `${message.content}\n\n${failureContent}`
              : failureContent
            : message.content,
          sources: currentSourcesRef.current,
          pending: false,
        };
      })
    );
  }

  return (
    <div className="chat-page">
      <section className="chat-surface">
        <ChatWindow messages={messages} isStreaming={isStreaming || isLoadingConversation} />
        <Composer
          value={input}
          disabled={isLoadingConversation}
          streaming={isStreaming}
          agentReasoning={agentReasoning}
          debugTrace={debugTrace}
          debugTraceAvailable={debugTraceAvailable}
          onAgentReasoningChange={(value) => {
            setAgentReasoning(value);
            localStorage.setItem(AGENT_REASONING_STORAGE_KEY, value);
          }}
          onDebugTraceChange={(value) => {
            setDebugTrace(value);
            localStorage.setItem(DEBUG_TRACE_STORAGE_KEY, String(value));
          }}
          onChange={setInput}
          onSubmit={sendMessage}
          onCancel={cancelStream}
        />
      </section>
      <aside className="companion-panel" aria-label="Conversation history and assistant status">
        <div className="companion-content">
          <div className="companion-card companion-card--compact">
            <div className="companion-avatar" aria-hidden="true">A</div>
            <div>
              <h2>Aya</h2>
              <p>Local-first assistant</p>
            </div>
          </div>
          <section className="conversation-history" aria-label="Lịch sử hội thoại">
            <div className="conversation-history-header">
              <span>Hội thoại</span>
              <button
                type="button"
                className="conversation-new-button"
                disabled={isStreaming || isLoadingConversation}
                onClick={startNewConversation}
              >
                Mới
              </button>
            </div>
            {historyError ? <p className="conversation-history-error">{historyError}</p> : null}
            <div className="conversation-list">
              {isLoadingConversation && conversations.length === 0 ? (
                <div className="conversation-history-empty">Đang tải…</div>
              ) : conversations.length === 0 ? (
                <div className="conversation-history-empty">Chưa có hội thoại</div>
              ) : (
                conversations.map((conversation) => (
                  <button
                    type="button"
                    className={conversation.id === conversationId ? "conversation-item active" : "conversation-item"}
                    disabled={isStreaming || isLoadingConversation}
                    key={conversation.id}
                    onClick={() => void loadConversationHistory(conversation.id)}
                    title={conversation.title}
                  >
                    <span>{conversation.title || "New chat"}</span>
                    <small>{formatConversationDate(conversation.updated_at)}</small>
                  </button>
                ))
              )}
            </div>
          </section>
        </div>
        <div className="companion-status">
          <span className="companion-status-label">Status</span>
          <span className="companion-status-value" title={status}>{status}</span>
        </div>
      </aside>
      <MemoTracePanel
        debugOutputRef={debugOutputRef}
        debugTracePayload={debugTracePayload}
        events={traceEvents}
        status={status}
        streamDeltaCount={streamDeltaCount}
      />
    </div>
  );
}

function toChatMessage(message: StoredMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    model: message.model,
    createdAt: message.created_at,
    sources: message.sources,
  };
}

function describeHealth(health: HealthResponse): string {
  const dependencies = [health.gateway, health.llm, health.router].filter(
    (dependency): dependency is HealthDependency => Boolean(dependency)
  );
  const states = dependencies.map(dependencyReachable).filter((state): state is boolean => state !== undefined);
  if (states.some((state) => state === false)) return "9router unavailable";
  if (states.some((state) => state === true)) {
    return health.ollama.reachable ? "ready" : "answer ready · RAG/Ollama offline";
  }
  if (health.status && !["ok", "ready", "healthy"].includes(health.status.toLowerCase())) {
    return `backend ${health.status}`;
  }
  return "backend online · 9router unverified";
}

function dependencyReachable(dependency: HealthDependency): boolean | undefined {
  if (typeof dependency.reachable === "boolean") return dependency.reachable;
  if (typeof dependency.ok === "boolean") return dependency.ok;
  if (!dependency.status) return undefined;
  const status = dependency.status.toLowerCase();
  if (["ok", "ready", "healthy", "online", "reachable"].includes(status)) return true;
  if (["error", "failed", "offline", "unavailable", "unreachable"].includes(status)) return false;
  return undefined;
}

function formatConversationDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { day: "2-digit", month: "2-digit" });
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

const MemoTracePanel = memo(function MemoTracePanel({
  events,
  status,
  streamDeltaCount,
  debugOutputRef,
  debugTracePayload,
}: {
  events: TraceEvent[];
  status: string;
  streamDeltaCount: number;
  debugOutputRef: RefObject<string>;
  debugTracePayload: AgentDebugTraceResponse | null;
}) {
  return (
    <AgentTracePanel
      events={events}
      getDebugOutput={() => debugOutputRef.current}
      debugTrace={debugTracePayload}
      status={status}
      streamDeltaCount={streamDeltaCount}
    />
  );
});
