import { memo, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";

import { AgentTracePanel } from "../components/agent/AgentTracePanel";
import { AyaStage } from "../components/avatar/AyaStage";
import { ChatWindow } from "../components/chat/ChatWindow";
import { Composer, type AgentReasoningMode } from "../components/chat/Composer";
import { deleteConversation, getAgentDebugTrace, getHealth, listConversations, listMessages } from "../lib/api";
import { streamAgent } from "../lib/sse";
import type {
  AvatarState,
  ChatMessage,
  ToolCallCard,
  AgentDebugTraceResponse,
  Conversation,
  HealthDependency,
  HealthResponse,
  Citation,
  RetrievedDocument,
  StoredMessage,
  TraceEvent,
} from "../lib/types";

interface ChatPageProps {
  model: string;
  // Sidebar element the conversation history renders into.
  historySlot?: HTMLElement | null;
  // Aya's stage hidden or shown; owned by the app shell (top bar toggle).
  stageCollapsed: boolean;
  onStageCollapsedChange: (collapsed: boolean) => void;
  onOpenChat?: () => void;
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

export function ChatPage({ model, historySlot, onOpenChat, stageCollapsed, onStageCollapsedChange }: ChatPageProps) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("checking backend");
  const [historyError, setHistoryError] = useState("");
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [streamDeltaCount, setStreamDeltaCount] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [avatarState, setAvatarState] = useState<AvatarState>("idle");
  // What Aya is doing right now, shown in her speech bubble.
  const [activity, setActivity] = useState("");
  const [isLoadingConversation, setIsLoadingConversation] = useState(true);
  const [agentReasoning, setAgentReasoning] = useState<AgentReasoningMode>(() => loadAgentReasoningMode());
  const [debugTrace, setDebugTrace] = useState(() => loadDebugTraceEnabled());
  const [debugTraceAvailable, setDebugTraceAvailable] = useState(false);
  const [debugTracePayload, setDebugTracePayload] = useState<AgentDebugTraceResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const currentSourcesRef = useRef<RetrievedDocument[]>([]);
  const currentCitationsRef = useRef<Citation[]>([]);
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

  // A reminder fired: Aya reacts, and the open chat shows the reminder message
  // the backend just added to it.
  useEffect(() => {
    function onNotification(event: Event) {
      const detail = (event as CustomEvent<{ conversation_id?: string | null }>).detail;
      if (!isStreaming) {
        setAvatarState("happy");
        window.setTimeout(() => mountedRef.current && setAvatarState("idle"), 4000);
      }
      if (detail?.conversation_id && detail.conversation_id === conversationId && !isStreaming) {
        void listMessages(conversationId)
          .then((stored) => mountedRef.current && setMessages(stored.map(toChatMessage)))
          .catch(() => undefined);
      }
    }
    window.addEventListener("aya:notification", onNotification);
    return () => window.removeEventListener("aya:notification", onNotification);
  }, [conversationId, isStreaming]);

  async function loadConversationHistory(selectedId: string) {
    onOpenChat?.();
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
    onOpenChat?.();
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

  async function removeConversation(id: string, title: string) {
    if (!window.confirm(`Xóa “${title || "New chat"}”? Tin nhắn và những gì Aya ghi nhớ từ cuộc này sẽ mất, không hoàn tác được.`)) {
      return;
    }
    try {
      await deleteConversation(id);
    } catch (error) {
      setHistoryError(errorMessage(error, "Không xóa được cuộc trò chuyện"));
      return;
    }
    setConversations((current) => current.filter((item) => item.id !== id));
    if (id === conversationId) startNewConversation();
    void refreshConversationList();
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
    setAvatarState("thinking");
    setStatus("streaming");
    setTraceEvents([]);
    setDebugTracePayload(null);
    setStreamDeltaCount(0);
    deltaCountRef.current = 0;
    debugOutputRef.current = "";
    currentSourcesRef.current = [];
    currentCitationsRef.current = [];
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
          if (!receivedDelta) {
            setAvatarState("talking");
            setActivity("");
          }
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

        if (
          event.event === "tool.call.started" ||
          event.event === "tool.call.approval_required" ||
          event.event === "tool.call.completed"
        ) {
          const data = event.data;
          const status: ToolCallCard["status"] =
            event.event === "tool.call.started"
              ? "running"
              : event.event === "tool.call.approval_required"
                ? "waiting"
                : data.declined
                  ? "declined"
                  : data.ok
                    ? "done"
                    : "failed";
          const card: ToolCallCard = {
            callId: data.call_id,
            runId: data.run_id,
            name: data.name,
            label: data.label,
            arguments: data.arguments,
            status,
            summary: data.summary,
            sources: data.sources,
            ms: data.ms,
          };
          setMessages((current) =>
            current.map((message) => {
              if (message.id !== assistantId) return message;
              const calls = message.toolCalls ?? [];
              const known = calls.some((item) => item.callId === card.callId);
              return {
                ...message,
                toolCalls: known
                  ? calls.map((item) => (item.callId === card.callId ? { ...item, ...card, summary: card.summary ?? item.summary } : item))
                  : [...calls, card],
              };
            })
          );
          if (event.event === "tool.call.started") {
            setActivity(`${data.label || data.name}…`);
          } else if (event.event === "tool.call.approval_required") {
            setStatus("Aya đang chờ cậu cho phép một thao tác");
            setActivity(`Cậu cho phép “${data.label}” không?`);
            setAvatarState("waiting");
          } else if (event.event === "tool.call.completed") {
            setStatus("streaming");
            setActivity(data.summary || "");
            if (!data.ok && !data.declined) setAvatarState("confused");
            if (data.ok && /reminder/.test(data.name)) window.dispatchEvent(new Event("aya:reminders-changed"));
          }
        }

        if (event.event === "avatar.state") {
          setAvatarState(event.data.state);
        }

        if (event.event === "answer.citations") {
          // A later event (full-document re-answer) replaces the earlier mapping.
          currentCitationsRef.current = event.data.citations;
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, citations: event.data.citations } : message
            )
          );
        }

        if (event.event === "message.reset") {
          // The backend replaced an already-shown opening with a validated
          // answer; drop it so unverified text never sits beside the result.
          debugOutputRef.current = "";
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, content: "" } : message
            )
          );
          setStatus(
            event.data.reason === "full_document"
              ? "Aya đang đọc toàn văn paper để trả lời kỹ hơn~"
              : "đã thay bằng câu trả lời đã kiểm chứng"
          );
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
          } else if (streamingMode === "number_free_prefix") {
            setStatus("đang viết; số liệu sẽ hiện sau khi đối chiếu tài liệu");
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
          setAvatarState("confused");
          streamError = event.data.error || "Agent run failed";
          setStatus(`failed · ${streamError}`);
        }

        if (event.event === "run.completed") {
          setAvatarState("happy");
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
        // Happy or confused for a moment, then back to idle.
        setActivity("");
        window.setTimeout(() => mountedRef.current && setAvatarState("idle"), 2600);
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
          citations: currentCitationsRef.current,
          pending: false,
        };
      })
    );
  }

  return (
    <div className={stageCollapsed ? "chat-page chat-page--stage-collapsed" : "chat-page"}>
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
      {stageCollapsed ? (
        <button type="button" className="aya-stage-reopen" onClick={() => onStageCollapsedChange(false)} title="Hiện Aya">
          <span aria-hidden="true">‹</span> Aya
          <span className={`aya-stage__dot aya-stage__dot--${avatarState}`} aria-hidden="true" />
        </button>
      ) : (
        <AyaStage state={avatarState} activity={activity} onCollapse={() => onStageCollapsedChange(true)} />
      )}
      {historySlot ? createPortal(
          <section className="conversation-history" aria-label="Lịch sử hội thoại">
          <button
            type="button"
            className="conversation-new-button"
            disabled={isStreaming || isLoadingConversation}
            onClick={startNewConversation}
          >
            <span aria-hidden="true">＋</span> Cuộc trò chuyện mới
          </button>
          <div className="conversation-history-header">Gần đây</div>
          {historyError ? <p className="conversation-history-error">{historyError}</p> : null}
          <div className="conversation-list">
            {isLoadingConversation && conversations.length === 0 ? (
              <div className="conversation-history-empty">Đang tải…</div>
            ) : conversations.length === 0 ? (
              <div className="conversation-history-empty">Chưa có hội thoại</div>
            ) : (
              conversations.map((conversation) => (
                <div className="conversation-row" key={conversation.id}>
                  <button
                    type="button"
                    className={conversation.id === conversationId ? "conversation-item active" : "conversation-item"}
                    disabled={isStreaming || isLoadingConversation}
                    onClick={() => void loadConversationHistory(conversation.id)}
                    title={conversation.title}
                  >
                    <span>{conversation.title || "New chat"}</span>
                    <small>{formatConversationDate(conversation.updated_at)}</small>
                  </button>
                  <button
                    type="button"
                    className="conversation-delete"
                    aria-label={`Xóa “${conversation.title || "New chat"}”`}
                    title="Xóa cuộc trò chuyện"
                    disabled={isStreaming || isLoadingConversation}
                    onClick={() => void removeConversation(conversation.id, conversation.title)}
                  >
                    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                      <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"
                        fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </div>
              ))
            )}
          </div>
        </section>,
        historySlot,
      ) : null}
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
    citations: message.citations,
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
  if (dependency.completion_authorized === false) return false;
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
