import type { RetrievedDocument } from "./documents";

export type ChatRole = "user" | "assistant";

export interface Citation {
  id: string;
  document_id?: string | null;
  filename?: string | null;
  title?: string | null;
  page_number?: number | null;
  source_path?: string | null;
  caption?: string | null;
  whole_document?: boolean;
  url?: string | null;
}

export type ToolCallStatus = "running" | "waiting" | "done" | "failed" | "declined";

export interface ToolCallSource {
  id: string;
  title: string;
  url?: string;
  path?: string;
}

export interface ToolCallCard {
  callId: string;
  runId: string;
  name: string;
  label: string;
  arguments: unknown;
  status: ToolCallStatus;
  summary?: string;
  sources?: ToolCallSource[];
  ms?: number;
}

export type AvatarState = "idle" | "thinking" | "tool" | "waiting" | "talking" | "happy" | "confused";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  model?: string | null;
  createdAt?: string;
  pending?: boolean;
  sources?: RetrievedDocument[];
  citations?: Citation[];
  toolCalls?: ToolCallCard[];
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: ChatRole;
  content: string;
  model: string | null;
  created_at: string;
  sources: RetrievedDocument[];
  citations?: Citation[];
}

