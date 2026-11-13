"""Build the same restartable tool environment for chat planning and the worker."""

from pathlib import Path

from app.agents.task_runner import TaskDeps
from app.agents.tool_loop import APPROVALS
from app.core.config import Settings
from app.services.tasks import TaskStore
from app.tools.registry import ToolContext


def task_deps_from_settings(settings: Settings, conversation_id: str | None) -> TaskDeps:
    # Import the patchable providers used by the ordinary chat route. Retrieval
    # providers are lazy: a file-only background task must not load the RAG stack.
    from app.api.agent import deps
    from app import tools
    from app.services.agent_run_store import AgentRunStore

    options = {}

    async def library_searcher(query, top_k):
        rag = deps.get_rag_service(settings)
        result = await rag.search_hybrid(
            query=query, top_k=top_k, collection_id=options.get("collection_id"),
            retrieval_store=deps._retrieval_store(settings), embeddings=deps._embedding_provider(settings),
            rerank=settings.rerank_enabled, rerank_mode=settings.rerank_mode,
            cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
        )
        return [item for item in result.get("results") or [] if not item.get("figure_id")]

    def registry_factory():
        return tools.build_registry(library_searcher=library_searcher)

    audit_store = AgentRunStore(settings.sqlite_db_path)

    def audit(**kwargs):
        # Tasks created outside chat need not have an agent_run row.
        if audit_store.get_run(kwargs["run_id"]):
            audit_store.record_tool_call(**kwargs)

    return TaskDeps(
        client=deps.get_agent_service(settings).client, model=settings.default_model,
        registry_factory=registry_factory, tool_names=registry_factory().names(),
        context_factory=lambda run_id, cid: ToolContext(
            run_id=run_id, conversation_id=cid or "", workspace=Path(settings.assistant_workspace).expanduser(),
            settings=settings,
        ),
        language="Vietnamese", approvals=APPROVALS, store=TaskStore(settings.sqlite_db_path), audit=audit,
        configure=options.update, library_searcher=library_searcher,
    )
