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
  completion_status?: "unknown" | "ok" | "failed" | string;
  completion_authorized?: boolean | null;
  completion_error?: string;
  error?: string;
}

