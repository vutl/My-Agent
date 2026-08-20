import { useEffect, useMemo, useRef, useState } from "react";

import { AgentTracePanel } from "../components/agent/AgentTracePanel";
import { MarkdownBlock } from "../components/chat/MarkdownBlock";
import { listCollections } from "../lib/api";
import { streamAgent } from "../lib/sse";
import type { CatalogCollection, RetrievedDocument, TraceEvent } from "../lib/types";

interface AgentPageProps {
  model: string;
}

export function AgentPage({ model }: AgentPageProps) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [task, setTask] = useState("");
  const [answer, setAnswer] = useState("");
  const [plan, setPlan] = useState<string[]>([]);
  const [sources, setSources] = useState<RetrievedDocument[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState("ready");
  const [isRunning, setIsRunning] = useState(false);
  const [collections, setCollections] = useState<CatalogCollection[]>([]);
  const [collectionId, setCollectionId] = useState("");
  const [streamDeltaCount, setStreamDeltaCount] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const answerRef = useRef("");
  const deltaCountRef = useRef(0);

  const canRun = useMemo(() => task.trim().length > 0 && !isRunning, [task, isRunning]);

  useEffect(() => {
    listCollections()
      .then(setCollections)
      .catch((error) => setStatus(error instanceof Error ? error.message : "failed"));
  }, []);

  async function runAgent() {
    const content = task.trim();
    if (!content || !canRun) {
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setAnswer("");
    setPlan([]);
    setSources([]);
    setEvents([]);
    setStreamDeltaCount(0);
    deltaCountRef.current = 0;
    answerRef.current = "";
    setRunId(null);
    setIsRunning(true);
    setStatus("running");

    try {
      let receivedDelta = false;
      for await (const event of streamAgent({
        conversationId,
        task: content,
        mode: "research",
        model,
        temperature: 0.2,
        collectionId: collectionId || null,
        retrievalMode: "auto",
        signal: controller.signal
      })) {
        if (event.event !== "message.delta") {
          setEvents((current) => [
            ...current,
            {
              event: event.event,
              data: event.data,
              timestamp: new Date().toLocaleTimeString(),
              at: Date.now()
            }
          ]);
        }

        if (event.event === "run.started") {
          setConversationId(event.data.conversation_id);
          setRunId(event.data.run_id);
        }

        if (event.event === "planner.completed") {
          setPlan(event.data.plan);
        }

        if (event.event === "retrieval.completed") {
          setSources(event.data.documents);
        }

        if (event.event === "message.delta") {
          receivedDelta = true;
          deltaCountRef.current += 1;
          answerRef.current += event.data.delta;
          setAnswer(answerRef.current);
          if (deltaCountRef.current % 12 === 0) {
            setStreamDeltaCount(deltaCountRef.current);
          }
        }

        if (event.event === "run.failed") {
          setStatus(event.data.error);
        }

        if (event.event === "run.completed") {
          setStatus("ready");
        }
      }
      setStreamDeltaCount(deltaCountRef.current);
      if (!receivedDelta) {
        setAnswer("Không nhận được output từ backend. Kiểm tra backend và Ollama đang chạy rồi gửi lại.");
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "failed");
      setAnswer(`Request failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="workspace-grid">
      <section className="agent-surface">
        <div className="agent-task-panel">
          <label>
            <span>Collection</span>
            <select
              value={collectionId}
              disabled={isRunning}
              onChange={(event) => setCollectionId(event.target.value)}
            >
              <option value="">All indexed documents</option>
              {collections.map((collection) => (
                <option key={collection.id} value={collection.id}>
                  {collection.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Task</span>
            <textarea
              value={task}
              disabled={isRunning}
              onChange={(event) => setTask(event.target.value)}
              placeholder="Ask the agent to plan or analyze a task"
              rows={5}
            />
          </label>
          <button className="send-button agent-run-button" disabled={!canRun} onClick={runAgent} type="button">
            Run
          </button>
        </div>

        <section className="agent-result-grid">
          <div className="agent-plan">
            <h2>Plan</h2>
            {plan.length === 0 ? (
              <div className="trace-empty">Waiting</div>
            ) : (
              plan.map((step, index) => (
                <div className="trace-item" key={`${step}-${index}`}>
                  {index + 1}. {step}
                </div>
              ))
            )}
            <h2>Sources</h2>
            {sources.length === 0 ? (
              <div className="trace-empty">No local sources</div>
            ) : (
              sources.map((source, index) => (
                <div className="trace-item" key={source.chunk_id}>
                  {source.citation_label || `SOURCE ${index + 1}`}: {source.filename}
                  {typeof source.page_number === "number" ? ` · page ${source.page_number}` : ""}
                  {typeof source.chunk_index === "number" ? ` · chunk ${source.chunk_index + 1}` : ""}
                </div>
              ))
            )}
            {runId ? <div className="trace-empty">Run {runId.slice(0, 8)}</div> : null}
          </div>
          <div className="agent-answer">
            <h2>Answer</h2>
            <div className="message-bubble">
              {answer ? <MarkdownBlock content={answer} sources={sources} /> : "No answer yet."}
            </div>
          </div>
        </section>
      </section>
      <AgentTracePanel
        events={events}
        getDebugOutput={() => answerRef.current}
        status={status}
        streamDeltaCount={streamDeltaCount}
      />
    </div>
  );
}
