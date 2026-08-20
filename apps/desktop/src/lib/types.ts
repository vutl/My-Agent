export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  model?: string | null;
  createdAt?: string;
  pending?: boolean;
  sources?: RetrievedDocument[];
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
}

export interface HealthResponse {
  status: string;
  default_model: string;
  router_model?: string;
  llm?: HealthDependency;
  router?: HealthDependency;
  gateway?: HealthDependency;
  ollama: {
    reachable: boolean;
    host: string;
    models: string[];
    error?: string;
  };
  agent_debug_trace?: {
    enabled: boolean;
    max_bytes: number;
    retention_hours: number;
    max_runs: number;
    redacted: boolean;
  };
}

export interface AgentDebugTraceResponse {
  run_id: string;
  schema_version: number;
  size_bytes: number;
  redaction_count: number;
  truncated: boolean;
  created_at: string;
  updated_at: string;
  expires_at: string;
  payload: Record<string, unknown>;
}

export interface HealthDependency {
  provider?: string;
  reachable?: boolean;
  ok?: boolean;
  status?: string;
  model?: string;
  configured_model?: string;
  base_url?: string;
  model_available?: boolean;
  error?: string;
}

export interface IndexedFolder {
  id: string;
  folder_path: string;
  recursive: boolean;
  file_types: string[];
  created_at: string;
  updated_at: string;
}

export interface IndexedDocument {
  id: string;
  folder_id: string;
  source_path: string;
  filename: string;
  file_type: string;
  modified_at: string;
  indexed_at: string;
  chunk_count: number;
  table_count?: number;
  figure_count?: number;
  index_status?: string;
  parser_name?: string | null;
  page_count?: number | null;
}

export interface CatalogCollection {
  id: string;
  name: string;
  type: string;
  scope_type: string;
  scope_id: string | null;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface RetrievedDocument {
  chunk_id: string;
  document_id: string;
  source_path: string;
  filename: string;
  content: string;
  citation_label?: string;
  score: number;
  chunk_index?: number;
  page_number?: number | null;
  heading_path?: string[];
  token_count?: number | null;
  char_count?: number | null;
  retrieval_channels?: string[];
  vector_rank?: number | null;
  fts_rank?: number | null;
  source_id?: string;
  chunk_type?: string | null;
  artifact_type?: string | null;
  caption?: string | null;
  image_path?: string | null;
  image_url?: string | null;
  table_id?: string | null;
  table_index?: number | null;
  figure_id?: string | null;
  figure_index?: number | null;
  figure_label?: string | null;
  figure_number?: number | null;
  quality_status?: string | null;
  asset_kind?: string | null;
  is_content?: boolean | null;
  is_complete?: boolean | null;
  figure_type?: string | null;
}

export interface IndexFolderResult {
  folder_id: string;
  folder_path: string;
  scanned_files: number;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  documents: Array<{
    id: string;
    source_path: string;
    filename: string;
    file_type: string;
    chunk_count: number;
  }>;
}

export interface CatalogFile {
  id: string;
  source_path: string;
  filename: string;
  extension?: string | null;
  size_bytes?: number | null;
  modified_at?: string | null;
  approved: boolean;
  is_supported: boolean;
}

export interface ScanFolderResult {
  folder_id: string;
  folder_path: string;
  recursive: boolean;
  file_count: number;
  supported_files: number;
  unsupported_files: number;
  files: CatalogFile[];
}

export interface CatalogSearchResult {
  source: "file" | "document_card";
  file_id?: string;
  document_id?: string;
  source_path: string;
  filename: string;
  extension?: string | null;
  title_guess?: string | null;
  short_summary?: string | null;
  rank_score?: number;
  is_supported?: boolean;
}

export interface CatalogSearchResponse {
  query: string;
  fts_query: string;
  retrieval_channels?: string[];
  results: CatalogSearchResult[];
}

export interface FileCandidate {
  file_id: string | null;
  filename: string;
  source_path: string;
  extension?: string | null;
  size_bytes?: number | null;
  modified_at?: string | null;
  confidence: number;
  reason: string;
  is_supported: boolean;
}

export interface ResolveFileResponse {
  status: "single_match" | "multiple_matches" | "not_found";
  candidates: FileCandidate[];
}

export interface DirectReadResponse {
  source_path: string;
  filename: string;
  file_type: string;
  parser_name: string;
  parser_version: string;
  page_count: number | null;
  token_estimate: number;
  truncated: boolean;
  content: string;
}

export interface IndexSelectedFilesResult {
  collection_id: string;
  collection_name: string;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  documents: IndexFolderResult["documents"];
  errors: Array<{ source_path: string; error: string }>;
}

export interface VectorIndexAllResult {
  documents: number;
  indexed_documents: number;
  skipped_documents?: number;
  failed_documents: number;
}

export interface LightRAGInsertAllResult {
  total: number;
  inserted: number;
  skipped: number;
  failed: number;
  deleted_stale?: number;
  prune_skipped_reason?: string | null;
  unready_document_ids?: string[];
  results: Array<{
    document_id: string;
    ok: boolean;
    track_id?: string;
    source_path?: string;
    char_count?: number;
    skipped?: boolean;
    reason?: string | null;
    error?: string;
  }>;
}

export interface LightRAGStatus {
  enabled: boolean;
  retrieval_engine: string;
  llm_model: string;
  llm_api_base: string;
  embedding_model: string;
  working_dir: string;
  status_counts: Record<string, number>;
  documents: Array<Record<string, unknown>>;
}

export interface HybridSearchResponse {
  query: string;
  collection_id?: string | null;
  selected_document_ids: string[];
  retrieval_channels: string[];
  results: RetrievedDocument[];
  context?: {
    sources: RetrievedDocument[];
    stats: {
      source_count: number;
      character_count: number;
      document_count: number;
    };
  };
}

export interface TraceEvent {
  event: string;
  data: unknown;
  timestamp: string;
  at?: number;
}

export type ChatStreamEvent =
  | { event: "message.started"; data: { conversation_id: string; model: string } }
  | { event: "message.delta"; data: { delta: string } }
  | { event: "message.completed"; data: { conversation_id: string } }
  | { event: "message.failed"; data: { error: string } };

export type AgentStreamEvent =
  | {
      event: "run.started";
      data: {
        run_id: string;
        conversation_id: string;
        user_message_id: string;
        mode: string;
        model: string;
        allowed_tools: string[];
        require_confirmation: boolean;
        collection_id?: string | null;
        retrieval_mode?: string;
        debug_trace_enabled?: boolean;
      };
    }
  | {
      event: "query.rewritten";
      data: {
        run_id: string;
        conversation_id: string;
        original_query: string;
        standalone_query: string;
        is_followup: boolean;
        current_topic?: string | null;
        required_entities: string[];
        use_last_sources: boolean;
        answer_intent?: string;
        answer_depth?: string;
        focus_document_ids: string[];
        rewrite_used: boolean;
        diagnostics?: Record<string, unknown>;
      };
    }
  | {
      event: "agent.route.decided";
      data: {
        run_id: string;
        conversation_id: string;
        route: string;
        selected_tools: string[];
        reason: string;
        confidence: string;
        max_tool_rounds: number;
        needs_fallback: boolean;
        use_local_retrieval: boolean;
      };
    }
  | {
      event: "tool.started";
      data: {
        run_id: string;
        conversation_id: string;
        tool_name: string;
        input?: Record<string, unknown>;
      };
    }
  | {
      event: "tool.completed";
      data: {
        run_id: string;
        conversation_id: string;
        tool_name: string;
        tool_call_id?: string;
        status: string;
        result_count?: number;
        fallback?: boolean;
      };
    }
  | {
      event: "tool.fallback.started";
      data: {
        run_id: string;
        conversation_id: string;
        tool_name: string;
        reason: string;
      };
    }
  | {
      event: "retrieval.skipped";
      data: {
        run_id: string;
        conversation_id: string;
        reason: string;
      };
    }
  | { event: "planner.started"; data: { conversation_id: string } }
  | { event: "agent.event"; data: Record<string, unknown> }
  | {
      event: "retrieval.started";
      data: {
        run_id: string;
        conversation_id: string;
        query: string;
        original_task?: string;
        focus_document_ids?: string[];
        tool_name: string;
        cache_candidate?: boolean;
      };
    }
  | {
      event: "retrieval.retrying";
      data: {
        run_id: string;
        conversation_id: string;
        query: string;
        sub_queries?: string[];
        hop?: number;
        max_hops?: number;
        parallel?: boolean;
        reason: string;
        reasons?: string[];
        missing_entities: string[];
        previous_focus_document_ids: string[];
      };
    }
  | {
      event: "retrieval.completed";
      data: {
        run_id: string;
        tool_call_id: string;
        conversation_id: string;
        query: string;
        original_task?: string;
        focus_document_ids?: string[];
        documents: RetrievedDocument[];
        retrieval_mode?: string;
        context_stats?: {
          source_count: number;
          character_count: number;
          document_count: number;
        };
        query_rewrite?: {
          standalone_query: string;
          is_followup: boolean;
          current_topic?: string | null;
          required_entities: string[];
          rewrite_used: boolean;
          answer_intent?: string;
          answer_depth?: string;
        };
        evidence_validation?: {
          valid: boolean;
          retry_required: boolean;
          reason: string;
          required_entities: string[];
          matched_entities: string[];
          missing_entities: string[];
        };
        retry_performed?: boolean;
      };
    }
  | {
      event: "evidence.card.coverage";
      data: {
        run_id: string;
        conversation_id: string;
        mode: string;
        documents: Array<{
          document_id: string;
          requested_facets: string[];
          covered_facets: string[];
          missing_facets: string[];
          stale: boolean;
          status: string;
        }>;
      };
    }
  | {
      event: "evidence.paper.ready";
      data: {
        run_id: string;
        conversation_id: string;
        document_id: string;
        requested_facets: string[];
        covered_facets: string[];
        missing_facets: string[];
        stale: boolean;
        status: string;
      };
    }
  | {
      event: "answer.paper.validated";
      data: {
        run_id: string;
        conversation_id: string;
        document_id?: string | null;
        section: string;
        valid: boolean;
        reason: string;
        fallback_used: boolean;
      };
    }
  | {
      event: "planner.completed";
      data: {
        run_id: string;
        conversation_id: string;
        route: string;
        mode: string;
        plan: string[];
        selected_tools: string[];
      };
    }
  | { event: "message.delta"; data: { delta: string } }
  | {
      event: "message.finished";
      data: {
        run_id: string;
        conversation_id: string;
        finish_reason?: string | null;
        eval_count?: number | null;
        metrics?: Record<string, number | string | null>;
        truncated: boolean;
      };
    }
  | {
      event: "timing";
      data: {
        run_id: string;
        conversation_id: string;
        stage: string;
        ms: number;
        [key: string]: unknown;
      };
    }
  | { event: "run.completed"; data: { run_id: string; conversation_id: string } }
  | { event: "run.failed"; data: { run_id: string; conversation_id: string; error: string } };
