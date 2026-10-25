import type {
  AgentDebugTraceResponse,
  Conversation,
  CatalogCollection,
  CatalogSearchResponse,
  DirectReadResponse,
  HealthResponse,
  HybridSearchResponse,
  IndexedDocument,
  IndexedFolder,
  IndexFolderResult,
  IndexSelectedFilesResult,
  ResolveFileResponse,
  RetrievedDocument,
  ScanFolderResult,
  StoredMessage,
  Task,
  VectorIndexAllResult,
  LightRAGInsertAllResult,
  LightRAGStatus
} from "./types";

import { normalizeTask, normalizeTaskList } from "./tasks";

export const API_BASE_URL =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:7777";

async function readJson<T>(path: string): Promise<T> {
  // Reads time out: a hung backend must not leave requests pending forever.
  const response = await fetch(`${API_BASE_URL}${path}`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function writeJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(formatApiError(error, response.status, response.statusText));
  }
  return response.json() as Promise<T>;
}

function formatApiError(raw: string, status: number, statusText: string): string {
  if (!raw) {
    return `${status} ${statusText}`;
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    return raw;
  }
  return raw;
}

export function getHealth(): Promise<HealthResponse> {
  return readJson<HealthResponse>("/health");
}

export function getAgentDebugTrace(runId: string): Promise<AgentDebugTraceResponse> {
  return readJson<AgentDebugTraceResponse>(`/agent/runs/${runId}/debug-trace`);
}

export function listConversations(): Promise<Conversation[]> {
  return readJson<Conversation[]>("/chat/conversations");
}

export function listMessages(conversationId: string): Promise<StoredMessage[]> {
  return readJson<StoredMessage[]>(`/chat/conversations/${conversationId}/messages`);
}

/** Delete a conversation (messages, runs, memory learned from it). */
export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
  if (response.status === 409) throw new Error("Aya đang trả lời trong cuộc trò chuyện này");
  if (!response.ok && response.status !== 404) throw new Error(`${response.status} ${response.statusText}`);
}

async function taskRequest(path: string, method: "GET" | "POST"): Promise<unknown> {
  // Polled from the chat and the stage: a hung backend must not pile requests up.
  const response = await fetch(`${API_BASE_URL}${path}`, { method, signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json().catch(() => null);
}

export async function listTasks(input: { status?: string; limit?: number } = {}): Promise<Task[]> {
  const query = new URLSearchParams();
  if (input.status) query.set("status", input.status);
  if (input.limit) query.set("limit", String(input.limit));
  const suffix = query.toString() ? `?${query}` : "";
  return normalizeTaskList(await taskRequest(`/tasks${suffix}`, "GET"));
}

export async function getTask(taskId: string): Promise<Task | null> {
  return normalizeTask(await taskRequest(`/tasks/${encodeURIComponent(taskId)}`, "GET"));
}

// The actions answer with the updated task; when they do not, the caller refetches.
export async function cancelTask(taskId: string): Promise<Task | null> {
  return normalizeTask(await taskRequest(`/tasks/${encodeURIComponent(taskId)}/cancel`, "POST"));
}

export async function pauseTask(taskId: string): Promise<Task | null> {
  return normalizeTask(await taskRequest(`/tasks/${encodeURIComponent(taskId)}/pause`, "POST"));
}

export async function resumeTask(taskId: string): Promise<Task | null> {
  return normalizeTask(await taskRequest(`/tasks/${encodeURIComponent(taskId)}/resume`, "POST"));
}

export function indexFolder(input: {
  folder_path: string;
  recursive: boolean;
  file_types: string[];
}): Promise<IndexFolderResult> {
  return writeJson<IndexFolderResult>("/files/index-folder", input);
}

export function indexFile(input: {
  source_path: string;
  collection_name?: string | null;
  collection_type?: string;
  scope_type?: string;
  scope_id?: string | null;
}): Promise<{ document: IndexedDocument }> {
  return writeJson<{ document: IndexedDocument }>("/rag/index-file", {
    collection_type: "manual",
    scope_type: "global",
    scope_id: null,
    ...input
  });
}

export function scanCatalogFolder(input: {
  folder_path: string;
  recursive: boolean;
  mode?: "shallow";
}): Promise<ScanFolderResult> {
  return writeJson<ScanFolderResult>("/catalog/scan-folder", {
    ...input,
    mode: input.mode ?? "shallow"
  });
}

export function listIndexedFolders(): Promise<IndexedFolder[]> {
  return readJson<IndexedFolder[]>("/files/indexed-folders");
}

export function listDocuments(): Promise<IndexedDocument[]> {
  return readJson<IndexedDocument[]>("/rag/documents");
}

export function listCollections(): Promise<CatalogCollection[]> {
  return readJson<CatalogCollection[]>("/catalog/collections");
}

export function searchRag(query: string, topK = 5): Promise<RetrievedDocument[]> {
  return writeJson<RetrievedDocument[]>("/rag/search", { query, top_k: topK });
}

export function searchCatalog(
  query: string,
  folderPath: string | null,
  topK = 20
): Promise<CatalogSearchResponse> {
  return writeJson<CatalogSearchResponse>("/catalog/search", {
    query,
    folder_path: folderPath || null,
    top_k: topK
  });
}

export function resolveFile(input: {
  filename_or_query: string;
  base_folder?: string | null;
  allow_fuzzy?: boolean;
  max_candidates?: number;
}): Promise<ResolveFileResponse> {
  return writeJson<ResolveFileResponse>("/files/resolve", {
    allow_fuzzy: true,
    max_candidates: 10,
    ...input
  });
}

export function readFileDirect(input: {
  source_path: string;
  max_tokens?: number;
}): Promise<DirectReadResponse> {
  return writeJson<DirectReadResponse>("/files/read", {
    source_path: input.source_path,
    mode: "transient",
    max_tokens: input.max_tokens ?? 1600
  });
}

export function indexSelectedFiles(input: {
  files: string[];
  collection_name: string;
  collection_type?: string;
  scope_type?: string;
  scope_id?: string | null;
}): Promise<IndexSelectedFilesResult> {
  return writeJson<IndexSelectedFilesResult>("/rag/index-selected-files", {
    collection_type: "manual",
    scope_type: "global",
    scope_id: null,
    ...input
  });
}

export function vectorIndexAll(limit?: number): Promise<VectorIndexAllResult> {
  return writeJson<VectorIndexAllResult>(
    "/rag/vector/index-all",
    typeof limit === "number" ? { limit } : {}
  );
}

export function lightragInsertAll(limit?: number): Promise<LightRAGInsertAllResult> {
  return writeJson<LightRAGInsertAllResult>(
    "/rag/lightrag/insert-all",
    typeof limit === "number" ? { limit } : {}
  );
}

export function getLightRAGStatus(): Promise<LightRAGStatus> {
  return readJson<LightRAGStatus>("/rag/lightrag/status");
}

export function searchHybrid(input: {
  query: string;
  top_k?: number;
  collection_id?: string | null;
}): Promise<HybridSearchResponse> {
  return writeJson<HybridSearchResponse>("/rag/search-hybrid", {
    query: input.query,
    top_k: input.top_k ?? 8,
    collection_id: input.collection_id || null
  });
}
