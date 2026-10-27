from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    conversation_id: str | None = None
    task: str = Field(min_length=1)
    mode: str = "research"
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    allowed_tools: list[str] = Field(default_factory=list)
    # Individual assistant tools; allowed_tools above selects routes such as assistant_tools.
    # None preserves the ordinary registry; an empty list permits no tool calls.
    assistant_tool_names: list[str] | None = None
    require_confirmation: bool = True
    collection_id: str | None = None
    retrieval_mode: str = "auto"
    agent_reasoning: str = "auto"
    debug_trace: bool = False
    max_steps: int | None = Field(default=None, ge=1, le=100)


class ToolCallDecision(BaseModel):
    approved: bool

