from pathlib import Path
from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from app.core.config import Settings, get_settings
from app.core.network import request_client_is_loopback
from app.services.agent_run_store import AgentRunStore
from .deps import get_agent_run_store
from .deps import get_agent_service
from .deps import get_chat_history
from .deps import get_conversation_memory_store
from .deps import get_conversation_runtime_gate
from .deps import get_conversation_state_store
from .deps import get_historical_conversation_search
from .deps import get_intent_router
from .deps import get_long_term_memory_store
from .deps import get_query_rewrite_service
from .deps import get_rag_service
from .models import AgentRunRequest
from .models import ToolCallDecision
from .stream import run_agent_stream


router = APIRouter(prefix="/agent", tags=["agent"])


router.post("/run/stream")(run_agent_stream)


@router.post("/runs/{run_id}/tool-calls/{call_id}/decision")
def decide_tool_call(run_id: str, call_id: str, decision: ToolCallDecision,
                     settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Approve or deny a tool call that is waiting in a running agent turn."""
    from app.agents import tool_loop

    from app.services.tasks import TaskStore

    if not (TaskStore(settings.sqlite_db_path).decide_approval(run_id, call_id, decision.approved)
            or tool_loop.APPROVALS.decide(run_id, call_id, decision.approved)):
        raise HTTPException(status_code=404, detail="No pending approval for this tool call")
    return {"run_id": run_id, "call_id": call_id, "approved": decision.approved}


@router.get("/workspace-file")
def workspace_file(path: str, settings: Annotated[Settings, Depends(get_settings)]):
    """Serve an image Aya created in her workspace (charts shown in chat)."""
    from fastapi.responses import FileResponse

    workspace = Path(settings.assistant_workspace).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    allowed = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
               ".gif": "image/gif", ".webp": "image/webp"}
    try:
        target.relative_to(workspace)
    except ValueError:
        raise HTTPException(status_code=403, detail="Only files inside Aya's workspace are served") from None
    if target.suffix.lower() not in allowed or not target.is_file():
        raise HTTPException(status_code=404, detail="Not an image in the workspace")
    return FileResponse(target, media_type=allowed[target.suffix.lower()])


@router.get("/runs/{run_id}")
async def get_agent_run(
    run_id: str,
    run_store: Annotated[AgentRunStore, Depends(get_agent_run_store)],
) -> dict:
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


@router.get("/runs/{run_id}/debug-trace")
async def get_agent_run_debug_trace(
    run_id: str,
    http_request: Request,
    response: Response,
    run_store: Annotated[AgentRunStore, Depends(get_agent_run_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    if (
        not settings.agent_debug_trace_enabled
        or not request_client_is_loopback(http_request)
    ):
        raise HTTPException(status_code=404, detail="Agent debug trace not found")
    trace = run_store.get_debug_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Agent debug trace not found")
    response.headers["Cache-Control"] = "no-store"
    return trace

