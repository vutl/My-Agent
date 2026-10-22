import type { AvatarState, Citation, ToolCallSource } from "./chat";
import type { RetrievedDocument } from "./documents";

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
      event: "tool.call.started" | "tool.call.approval_required" | "tool.call.completed";
      data: {
        run_id: string;
        conversation_id: string;
        call_id: string;
        name: string;
        label: string;
        arguments: unknown;
        ok?: boolean;
        summary?: string;
        declined?: boolean;
        sources?: ToolCallSource[];
        ms?: number;
      };
    }
  | {
      event: "avatar.state";
      data: { run_id: string; state: AvatarState; tool?: string };
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
  | { event: "answer.citations"; data: { run_id?: string; conversation_id?: string; citations: Citation[] } }
  | { event: "message.reset"; data: { run_id?: string; conversation_id?: string; reason?: string } }
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
