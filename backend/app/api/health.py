import asyncio

from fastapi import APIRouter, Request

from app.core.config import (
    APPROVED_9ROUTER_MODELS,
    DEFAULT_9ROUTER_MODEL,
    get_settings,
)
from app.db.sqlite import connect
from app.core.network import request_client_is_loopback
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAICompatibleClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    settings = get_settings()
    ollama = OllamaClient(settings.ollama_host, settings.request_timeout_seconds)
    gateway_client = OpenAICompatibleClient(
        settings.openai_api_base,
        settings.openai_api_key,
        settings.request_timeout_seconds,
    )
    ollama_status, raw_gateway_status = await asyncio.gather(
        ollama.health(),
        gateway_client.health(),
    )
    gateway_status = _gateway_status(
        raw_gateway_status,
        model=settings.default_model,
        base_url=settings.openai_api_base,
        provider=settings.llm_provider,
    )

    return {
        "status": "ok" if gateway_status["reachable"] else "degraded",
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
        },
        "ollama": ollama_status,
        "gateway": gateway_status,
        "default_model": settings.default_model,
        "router_model": settings.router_model,
        "model_policy": {
            "default_approved_model": DEFAULT_9ROUTER_MODEL,
            "approved_models": sorted(APPROVED_9ROUTER_MODELS),
            "answer": settings.default_model,
            "router": settings.router_model,
            "vision": settings.vision_model,
            "lightrag": settings.lightrag_llm_model,
            "model_fallback_enabled": False,
        },
        "memory": _memory_status(settings.sqlite_db_path),
        "agent_debug_trace": {
            "enabled": settings.agent_debug_trace_enabled
            and request_client_is_loopback(request),
            "max_bytes": settings.agent_debug_trace_max_bytes,
            "retention_hours": settings.agent_debug_trace_retention_hours,
            "max_runs": settings.agent_debug_trace_max_runs,
            "redacted": True,
        },
        "paper_evidence_cards": {
            **_paper_evidence_status(settings.sqlite_db_path),
            "runtime_enabled": settings.paper_evidence_cards_enabled,
            "build_enabled": settings.paper_evidence_card_build_enabled,
            "section_streaming_enabled": settings.paper_section_streaming_enabled,
            "model": settings.paper_evidence_card_model,
            "schema_version": settings.paper_evidence_card_schema_version,
            "prompt_version": settings.paper_evidence_card_prompt_version,
            "max_concurrency": settings.paper_evidence_card_max_concurrency,
        },
        "vision": {
            "provider": (
                "9router"
                if settings.vision_provider == "openai_compatible"
                else settings.vision_provider
            ),
            "model": settings.vision_model,
            "reachable": (
                gateway_status["reachable"]
                if settings.vision_provider == "openai_compatible"
                else bool(ollama_status.get("reachable"))
            ),
        },
    }


def _paper_evidence_status(db_path) -> dict:
    with connect(db_path) as connection:
        document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        rows = connection.execute(
            "SELECT status, COUNT(*) count FROM paper_evidence_cards GROUP BY status"
        ).fetchall()
        job_rows = connection.execute(
            "SELECT status, COUNT(*) count FROM paper_evidence_build_jobs GROUP BY status"
        ).fetchall()
    cards = {str(row["status"]): int(row["count"]) for row in rows}
    return {
        "documents": document_count,
        "cards": sum(cards.values()),
        "card_status": cards,
        "job_status": {
            str(row["status"]): int(row["count"]) for row in job_rows
        },
    }


def _gateway_status(
    raw: dict,
    *,
    model: str,
    base_url: str,
    provider: str,
) -> dict:
    models = [str(item) for item in raw.get("models") or []]
    provider_name = "9router" if provider == "openai_compatible" else provider
    model_available = model in models if models else bool(raw.get("reachable"))
    reachable = bool(raw.get("reachable")) and model_available
    result = {
        "provider": provider_name,
        "reachable": reachable,
        "model": model,
        "base_url": base_url,
        "model_available": model_available,
    }
    if raw.get("error"):
        result["error"] = str(raw["error"])
    elif raw.get("reachable") and not model_available:
        result["error"] = f"Configured model is not advertised by the gateway: {model}"
    return result


def _memory_status(db_path) -> dict:
    try:
        with connect(db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN dirty_through_seq > summary_through_seq
                                  AND status != 'dormant' THEN 1 ELSE 0 END)
                        AS pending_conversations,
                    SUM(CASE WHEN status = 'dormant' THEN 1 ELSE 0 END)
                        AS dormant_conversations,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END)
                        AS running_conversations,
                    SUM(CASE WHEN dirty_through_seq > summary_through_seq
                                  AND last_error IS NOT NULL THEN 1 ELSE 0 END)
                        AS retrying_conversations
                FROM conversation_memory_jobs
                """
            ).fetchone()
            error_row = connection.execute(
                """
                SELECT last_error, next_attempt_at, updated_at
                FROM conversation_memory_jobs
                WHERE last_error IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
            l3_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS pending_operations,
                    COUNT(DISTINCT conversation_id) AS pending_conversations,
                    SUM(CASE WHEN last_error IS NOT NULL THEN 1 ELSE 0 END)
                        AS retrying_operations
                FROM conversation_memory_l3_outbox
                WHERE status = 'pending'
                """
            ).fetchone()
            l3_error_row = connection.execute(
                """
                SELECT last_error, next_attempt_at, updated_at,
                       conversation_id, source_turn_seq
                FROM conversation_memory_l3_outbox
                WHERE status = 'pending' AND last_error IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
    except Exception as exc:
        return {"status": "unavailable", "error": " ".join(str(exc).split())[:300]}

    result = {
        "status": "ok",
        "pending_conversations": int(row["pending_conversations"] or 0),
        "dormant_conversations": int(row["dormant_conversations"] or 0),
        "running_conversations": int(row["running_conversations"] or 0),
        "retrying_conversations": int(row["retrying_conversations"] or 0),
        "l3_outbox": {
            "pending_operations": int(l3_row["pending_operations"] or 0),
            "pending_conversations": int(l3_row["pending_conversations"] or 0),
            "retrying_operations": int(l3_row["retrying_operations"] or 0),
        },
    }
    if error_row is not None:
        result["last_error"] = str(error_row["last_error"] or "")[:500]
        result["next_attempt_at"] = error_row["next_attempt_at"]
        result["last_error_at"] = error_row["updated_at"]
    if l3_error_row is not None:
        result["l3_outbox"].update(
            {
                "last_error": str(l3_error_row["last_error"] or "")[:500],
                "next_attempt_at": l3_error_row["next_attempt_at"],
                "last_error_at": l3_error_row["updated_at"],
                "conversation_id": l3_error_row["conversation_id"],
                "source_turn_seq": int(l3_error_row["source_turn_seq"]),
            }
        )
    return result
