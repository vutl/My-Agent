"""Opt-in, redacted milestone recorder for agent-run diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import time
from typing import Any

from app.core.redaction import redact_and_bound
from app.services.agent_run_store import AgentRunStore


logger = logging.getLogger(__name__)


@dataclass
class DebugTraceRecorder:
    store: AgentRunStore
    run_id: str
    enabled: bool
    max_bytes: int = 65_536
    retention_hours: int = 72
    max_runs: int = 25
    exact_secrets: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled and not self.payload:
            self.payload = {
                "schema_version": 1,
                "capture": {"redacted": True, "max_bytes": self.max_bytes},
                "scope_resolution": {},
                "generations": [],
                "outcome": {"status": "running", "error": None},
            }

    def record_scope(
        self,
        *,
        loaded_working_state: dict[str, Any],
        effective_working_state: dict[str, Any],
        resolution: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self.payload["scope_resolution"] = {
            "working_state_loaded": loaded_working_state,
            "working_state_effective": effective_working_state,
            "decision": resolution,
        }
        self.flush()

    def record_route(self, route: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.payload["route"] = route
        self.flush()

    def record_rewrite(self, rewrite: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.payload["rewrite"] = rewrite
        self.flush()

    def record_retrieval(
        self,
        *,
        focus_document_ids: list[str],
        retrieved_document_ids: list[str],
        validation: dict[str, Any],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.payload["retrieval"] = {
            "focus_document_ids": focus_document_ids,
            "retrieved_document_ids": retrieved_document_ids,
            "validation": validation,
            "diagnostics": diagnostics or {},
        }
        self.flush()

    def start_generation(
        self,
        *,
        phase: str,
        route: str,
        model: str,
        temperature: float,
        prompt: str,
    ) -> int | None:
        if not self.enabled:
            return None
        generations = self.payload.setdefault("generations", [])
        generations.append(
            {
                "phase": phase,
                "route": route,
                "input": {
                    "model": model,
                    "temperature": temperature,
                    "final_prompt": prompt,
                },
                "attempts": [],
                "selected_output": None,
            }
        )
        self.flush()
        return len(generations) - 1

    def finish_attempt(
        self,
        generation_index: int | None,
        *,
        kind: str,
        draft: str,
        started_at: float,
        validation: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        if not self.enabled or generation_index is None:
            return
        generation = self.payload["generations"][generation_index]
        generation.setdefault("attempts", []).append(
            {
                "kind": kind,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "finish_reason": finish_reason,
                "draft": draft,
                "validation": validation or {},
            }
        )
        self.flush()

    def annotate_last_attempt(
        self,
        generation_index: int | None,
        *,
        validation: dict[str, Any],
    ) -> None:
        if not self.enabled or generation_index is None:
            return
        attempts = self.payload["generations"][generation_index].get("attempts") or []
        if attempts:
            attempts[-1]["validation"] = validation
            self.flush()

    def select_output(self, generation_index: int | None, selected: str) -> None:
        if not self.enabled or generation_index is None:
            return
        self.payload["generations"][generation_index]["selected_output"] = selected
        self.flush()

    def record_direct_execution(self, *, phase: str, kind: str) -> None:
        if not self.enabled:
            return
        self.payload.setdefault("generations", []).append(
            {
                "phase": phase,
                "execution_kind": kind,
                "attempts": [],
                "selected_output": kind,
            }
        )
        self.flush()

    def outcome(self, status: str, error: str | None = None) -> None:
        if not self.enabled:
            return
        self.payload["outcome"] = {
            "status": status,
            "error": (error or "")[:1024] or None,
            "at": datetime.now(UTC).isoformat(),
        }
        self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            sanitized, count, truncated, size_bytes = redact_and_bound(
                self.payload,
                max_bytes=self.max_bytes,
                exact_secrets=self.exact_secrets,
            )
            self.store.upsert_debug_trace(
                run_id=self.run_id,
                payload=sanitized,
                size_bytes=size_bytes,
                redaction_count=count,
                truncated=truncated,
                retention_hours=self.retention_hours,
                max_runs=self.max_runs,
                max_bytes=self.max_bytes,
            )
        except Exception as exc:  # Trace failure must never fail the chat run.
            logger.warning(
                "debug_trace_snapshot_dropped run_id=%s error_type=%s",
                self.run_id,
                type(exc).__name__,
            )
