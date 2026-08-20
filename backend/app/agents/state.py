from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict):
    run_id: NotRequired[str]
    conversation_id: str
    user_message_id: NotRequired[str]
    user_task: str
    resolved_task: NotRequired[str]
    conversation_context: NotRequired[str]
    answer_intent: NotRequired[str]
    answer_depth: NotRequired[str]
    answer_style: NotRequired[str]
    mode: str
    model: str
    route: str
    local_context_required: NotRequired[bool]
    tool_decision: NotRequired[dict[str, Any]]
    plan: list[str]
    selected_tools: list[str]
    retrieved_docs: list[dict]
    paper_evidence_context: NotRequired[str]
    evidence_sufficiency_context: NotRequired[str]
    paper_evidence_coverage: NotRequired[list[dict[str, Any]]]
    paper_section_streaming: NotRequired[bool]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_results: NotRequired[list[dict[str, Any]]]
    needs_fallback: NotRequired[bool]
    final_prompt: str
    error: str | None
