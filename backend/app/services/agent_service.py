from dataclasses import dataclass
from typing import AsyncIterator

from app.agents.graph import _pick_system_prompt, build_agent_graph, generation_max_tokens
from app.agents.state import AgentState
from app.llm.ollama_client import OllamaClient


@dataclass(frozen=True)
class AgentGraphResult:
    route: str
    mode: str
    plan: list[str]
    final_prompt: str
    selected_tools: list[str]
    retrieved_docs: list[dict]
    tool_decision: dict | None = None


@dataclass(frozen=True)
class AgentGraphStreamEvent:
    event: str
    payload: dict
    result: AgentGraphResult | None = None


@dataclass(frozen=True)
class AnswerStreamChunk:
    content: str
    done: bool = False
    finish_reason: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class AgentService:
    client: OllamaClient
    default_model: str

    async def run_graph(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_message_id: str,
        task: str,
        mode: str,
        model: str | None,
        temperature: float,
        resolved_task: str | None = None,
        conversation_context: str = "",
        answer_intent: str = "direct_answer",
        answer_depth: str = "normal",
        answer_style: str = "natural_technical",
        retrieved_docs: list[dict] | None = None,
        tool_decision: dict | None = None,
        paper_evidence_context: str = "",
        evidence_sufficiency_context: str = "",
        paper_evidence_coverage: list[dict] | None = None,
        paper_section_streaming: bool = False,
    ) -> AgentGraphResult:
        selected_model = model or self.default_model
        graph = build_agent_graph(self.client, temperature=temperature)
        initial_state = _initial_state(
            run_id=run_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            task=task,
            resolved_task=resolved_task,
            conversation_context=conversation_context,
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_style=answer_style,
            mode=mode,
            model=selected_model,
            retrieved_docs=retrieved_docs,
            tool_decision=tool_decision,
            paper_evidence_context=paper_evidence_context,
            evidence_sufficiency_context=evidence_sufficiency_context,
            paper_evidence_coverage=paper_evidence_coverage,
            paper_section_streaming=paper_section_streaming,
        )
        state = await graph.ainvoke(initial_state)
        return _graph_result(state)

    async def stream_graph_events(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_message_id: str,
        task: str,
        mode: str,
        model: str | None,
        temperature: float,
        resolved_task: str | None = None,
        conversation_context: str = "",
        answer_intent: str = "direct_answer",
        answer_depth: str = "normal",
        answer_style: str = "natural_technical",
        retrieved_docs: list[dict] | None = None,
        tool_decision: dict | None = None,
        paper_evidence_context: str = "",
        evidence_sufficiency_context: str = "",
        paper_evidence_coverage: list[dict] | None = None,
        paper_section_streaming: bool = False,
    ) -> AsyncIterator[AgentGraphStreamEvent]:
        selected_model = model or self.default_model
        graph = build_agent_graph(self.client, temperature=temperature)
        initial_state = _initial_state(
            run_id=run_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            task=task,
            resolved_task=resolved_task,
            conversation_context=conversation_context,
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_style=answer_style,
            mode=mode,
            model=selected_model,
            retrieved_docs=retrieved_docs,
            tool_decision=tool_decision,
            paper_evidence_context=paper_evidence_context,
            evidence_sufficiency_context=evidence_sufficiency_context,
            paper_evidence_coverage=paper_evidence_coverage,
            paper_section_streaming=paper_section_streaming,
        )
        yield AgentGraphStreamEvent("graph.started", {"node": "LangGraph"})

        async for event in graph.astream_events(initial_state, version="v2"):
            event_type = event.get("event")
            name = event.get("name")
            data = event.get("data") or {}

            if event_type == "on_chain_start" and name in {"router", "planner", "final_prompt"}:
                yield AgentGraphStreamEvent(f"{name}.started", {"node": name})
                continue

            if event_type == "on_chain_end" and name in {"router", "planner", "final_prompt"}:
                output = data.get("output") or {}
                yield AgentGraphStreamEvent(f"{name}.completed", _node_payload(name, output))
                continue

            if event_type == "on_chain_end" and name == "LangGraph":
                state = data.get("output") or {}
                result = _graph_result(state)
                yield AgentGraphStreamEvent(
                    "graph.completed",
                    {
                        "node": "LangGraph",
                        "route": result.route,
                        "mode": result.mode,
                        "plan": result.plan,
                        "selected_tools": result.selected_tools,
                    },
                    result=result,
                )

    async def stream_final_answer(
        self,
        *,
        prompt: str,
        model: str | None,
        temperature: float,
        answer_intent: str = "direct_answer",
        answer_depth: str = "normal",
    ) -> AsyncIterator[AnswerStreamChunk]:
        selected_model = model or self.default_model
        saw_done = False
        async for chunk in self.client.stream_chat(
            model=selected_model,
            temperature=temperature,
            num_predict=generation_max_tokens(answer_intent, answer_depth),
            messages=[
                {"role": "system", "content": _pick_system_prompt(selected_model)},
                {"role": "user", "content": prompt},
            ],
        ):
            if chunk.done:
                if chunk.content:
                    yield AnswerStreamChunk(content=chunk.content)
                yield AnswerStreamChunk(
                    content="",
                    done=True,
                    finish_reason=chunk.finish_reason,
                    metadata=chunk.metadata,
                )
                saw_done = True
            else:
                yield AnswerStreamChunk(content=chunk.content)
        if not saw_done:
            # Some OpenAI-compatible gateways close the stream without a final
            # finish_reason/[DONE] frame. Always let the SSE buffer flush.
            yield AnswerStreamChunk(content="", done=True, finish_reason="stream_closed")


def _initial_state(
    *,
    run_id: str,
    conversation_id: str,
    user_message_id: str,
    task: str,
    resolved_task: str | None,
    conversation_context: str,
    answer_intent: str,
    answer_depth: str,
    answer_style: str,
    mode: str,
    model: str,
    retrieved_docs: list[dict] | None,
    tool_decision: dict | None,
    paper_evidence_context: str,
    evidence_sufficiency_context: str,
    paper_evidence_coverage: list[dict] | None,
    paper_section_streaming: bool,
) -> AgentState:
    return {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "user_task": task,
        "resolved_task": resolved_task or task,
        "conversation_context": conversation_context,
        "answer_intent": answer_intent,
        "answer_depth": answer_depth,
        "answer_style": answer_style,
        "mode": mode,
        "model": model,
        "route": "",
        "plan": [],
        "selected_tools": [],
        "tool_decision": tool_decision or {},
        "retrieved_docs": retrieved_docs or [],
        "paper_evidence_context": paper_evidence_context,
        "evidence_sufficiency_context": evidence_sufficiency_context,
        "paper_evidence_coverage": paper_evidence_coverage or [],
        "paper_section_streaming": paper_section_streaming,
        "tool_calls": [],
        "tool_results": [],
        "needs_fallback": bool((tool_decision or {}).get("needs_fallback")),
        "final_prompt": "",
        "error": None,
    }


def _graph_result(state) -> AgentGraphResult:
    return AgentGraphResult(
        route=state["route"],
        mode=state["mode"],
        plan=state["plan"],
        final_prompt=state["final_prompt"],
        selected_tools=state["selected_tools"],
        retrieved_docs=state["retrieved_docs"],
        tool_decision=state.get("tool_decision") or None,
    )


def _node_payload(name: str, output: dict) -> dict:
    payload = {"node": name}
    if name == "router":
        payload.update(
            {
                "route": output.get("route"),
                "mode": output.get("mode"),
                "selected_tools": output.get("selected_tools") or [],
                "local_context_required": output.get("local_context_required"),
                "tool_decision": output.get("tool_decision") or {},
                "needs_fallback": output.get("needs_fallback", False),
            }
        )
    elif name == "planner":
        payload["plan"] = output.get("plan") or []
    elif name == "final_prompt":
        payload["final_prompt_chars"] = len(str(output.get("final_prompt") or ""))
    return payload
