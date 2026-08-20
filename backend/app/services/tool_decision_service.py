"""Hybrid LLM intent router: classify chat vs RAG vs research without topical wordlists."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from app.llm.ollama_client import OllamaError
from app.services.long_term_memory import sentence_safe_clip

from app.services.conversation_state import looks_like_document_resume
from app.services.document_language import looks_like_document_comparison


DEFAULT_SAFE_TOOLS = {"search_local_docs", "retrieve_visual_assets"}
WEB_SEARCH_TOOL = "web_search"

_VALID_ROUTES = {"chat", "file_qa", "research"}
_ANSWER_INTENTS = {
    "direct_answer",
    "elaborate",
    "compare",
    "infer_structure",
    "simplify",
    "example",
}
_ANSWER_DEPTHS = {"brief", "normal", "detailed"}
_CONFIDENCES = {"high", "medium", "low"}

_UNCERTAIN_ANSWER_MARKERS = {
    "tôi không biết",
    "mình không biết",
    "không chắc",
    "không có đủ thông tin",
    "i don't know",
    "i do not know",
    "not sure",
    "insufficient information",
}

_EXPLICIT_DOCUMENT_REFERENCE_RE = re.compile(
    r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)\s+"
    r"(?:của\s+)?[A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ0-9_.-]{1,}|"
    r"(?<!\w)[A-Za-z][A-Za-z0-9_.-]*\.pdf(?!\w)",
    re.IGNORECASE,
)
_STRUCTURED_DOCUMENT_FACET_RE = re.compile(
    r"(?<!\w)(?:bảng|table|kết\s*quả|results?|benchmark|datasets?|bộ\s+dữ\s+liệu|"
    r"acc(?:uracy)?|f1|ccc|uar|war|wa|ua|figure|fig\.?|hình|ảnh|diagram|sơ\s+đồ|"
    r"architecture|kiến\s+trúc|pipeline|ablation)(?!\w)",
    re.IGNORECASE,
)
_DOCUMENT_ANALYSIS_RE = re.compile(
    r"(?<!\w)(?:giải\s*thích|phân\s*tích|đánh\s*giá|nhận\s*xét|ý\s*nghĩa|"
    r"explain|analy[sz]e|evaluate|interpret)(?!\w)",
    re.IGNORECASE,
)
_DOCUMENT_VISUAL_RE = re.compile(
    r"(?<!\w)(?:figure|fig\.?|hình|ảnh|diagram|sơ\s+đồ|chart|biểu\s+đồ|plot)(?!\w)",
    re.IGNORECASE,
)
_LOCAL_RETRIEVAL_ANCHOR_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:local|indexed|uploaded|attached|my|our|these)\s+(?:papers?|documents?|files?)|"
    r"(?:papers?|documents?|files?)\s+(?:in|from)\s+(?:my|our|the)\s+"
    r"(?:library|collection|workspace|corpus)|"
    r"(?:which|what)\s+(?:papers?|documents?)\b|"
    r"(?:tìm|tra|search)\s+(?:trong|ở)\s+(?:file|tài\s*liệu|thư\s*viện|kho|corpus)|"
    r"(?:file|tài\s*liệu|bài(?:\s+báo)?)\s+(?:đã\s+)?(?:tải|upload|đính\s*kèm)|"
    r"(?:trong|ở|từ)\s+(?:thư\s*viện|kho|workspace|corpus)(?:\s+(?:của\s+)?(?:tôi|mình))?|"
    r"(?:các|những)\s+(?:bài|paper|tài\s*liệu)\s+nào\b"
    r")(?!\w)",
    re.IGNORECASE,
)
_PURE_SOCIAL_ACT_RE = re.compile(
    r"^\s*(?:"
    r"thank\s+you(?:\s+(?:very\s+much|so\s+much))?|thanks(?:\s+(?:a\s+lot|so\s+much))?|"
    r"much\s+appreciated|ok(?:ay|k+)?|got\s+it|sounds\s+good|understood|"
    r"c[ảa]m\s+[ơo]n(?:\s+(?:cậu|bạn|nhé|nha|nhiều))?|"
    r"ừ+|uh+m*|hiểu\s+rồi|được\s+rồi"
    r")(?:\s*[!.…]+)?\s*$",
    re.IGNORECASE,
)

_ROUTER_SYSTEM_PROMPT = """You are the turn-policy router for Aya, a local personal AI assistant.

Decide whether this turn needs local document retrieval (RAG over indexed papers/PDFs/figures) or is general/casual chat.

Capabilities:
- search_local_docs: search indexed local papers and excerpts
- retrieve_visual_assets: fetch figures/diagrams from those papers (only with local search)
- General chat: greetings, banter, general knowledge, coding help unrelated to the local corpus

Return ONLY valid JSON (no markdown) with this shape:
{
  "route": "chat" | "file_qa" | "research",
  "selected_tools": ["search_local_docs", "retrieve_visual_assets"],
  "answer_intent": "direct_answer" | "elaborate" | "compare" | "infer_structure" | "simplify" | "example",
  "answer_depth": "brief" | "normal" | "detailed",
  "confidence": "high" | "medium" | "low",
  "needs_fallback": false,
  "reason": "short_snake_case"
}

Rules:
- Casual, social, jokes, weather, general world knowledge, or unrelated coding → route=chat, selected_tools=[]
- Questions about papers/models/datasets/figures in the local library, or follow-ups that clearly continue a document discussion → route=file_qa and include search_local_docs
- Do not assume that phrases such as "class materials", "provided context", "the text below", or a generic request for sources refer to Aya's indexed local library. With no resolved local document, active document thread, or explicit local-library request, use route=chat.
- If active_working_focus names a paper/topic and the user asks short follow-ups (benchmark, bảng, Acc/F1/CCC, architecture, figure) without naming a different paper → stay file_qa on that focus; do not treat metrics as a new topic
- Add retrieve_visual_assets when the user asks about figures, diagrams, architecture drawings, charts, or images
- Broad compare/synthesize across multiple papers → route=research, tools include search_local_docs, prefer max detail when useful
- needs_fallback=true only when you chose chat but the question might actually need the local corpus and you are uncertain
- Prefer Vietnamese-friendly answer_intent/depth matching how the user asked
- Never invent tools outside the allowed list provided in the user message
"""


class SupportsChat(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int = 512,
    ) -> Any: ...


@dataclass(frozen=True)
class ToolDecision:
    route: str
    selected_tools: list[str]
    reason: str
    confidence: str
    max_tool_rounds: int = 1
    needs_fallback: bool = False
    answer_intent: str = "direct_answer"
    answer_depth: str = "normal"

    @property
    def use_local_retrieval(self) -> bool:
        return "search_local_docs" in self.selected_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "selected_tools": self.selected_tools,
            "reason": self.reason,
            "confidence": self.confidence,
            "max_tool_rounds": self.max_tool_rounds,
            "needs_fallback": self.needs_fallback,
            "use_local_retrieval": self.use_local_retrieval,
            "answer_intent": self.answer_intent,
            "answer_depth": self.answer_depth,
        }


class IntentRouterService:
    """LLM-backed turn policy with soft allowlist / mode gates."""

    def __init__(
        self,
        *,
        client: SupportsChat | None = None,
        default_model: str = "cx/gpt-5.6-sol",
    ) -> None:
        self.client = client
        self.default_model = default_model

    async def decide(
        self,
        *,
        task: str,
        mode: str,
        previous_messages: list[Any],
        has_recent_retrieval: bool,
        allowed_tools: list[str] | None,
        recent_document_ids: list[str] | None = None,
        working_topic: str | None = None,
        working_filenames: list[str] | None = None,
        conversation_summary: str | None = None,
        recent_turn_notes: str | None = None,
        model: str | None = None,
        resolved_document_ids: list[str] | None = None,
    ) -> ToolDecision:
        allowed = _effective_allowed_tools(allowed_tools)
        normalized_mode = (mode or "").lower()

        if normalized_mode == "file_qa":
            return _finalize_decision(
                route="file_qa",
                tools=["search_local_docs", "retrieve_visual_assets"],
                reason="file_qa_mode",
                confidence="high",
                allowed_tools=allowed,
                answer_intent="direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        if normalized_mode == "chat":
            return _finalize_decision(
                route="chat",
                tools=[],
                reason="explicit_chat_mode",
                confidence="high",
                allowed_tools=allowed,
                answer_intent="direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        deterministic_chat_reason = _deterministic_chat_route_reason(
            task,
            has_active_focus=bool(
                working_topic or working_filenames or recent_document_ids
            ),
        )
        if deterministic_chat_reason:
            return _finalize_decision(
                route="chat",
                tools=[],
                reason=deterministic_chat_reason,
                confidence="high",
                allowed_tools=allowed,
                answer_intent="direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        # Catalog identity resolution is stronger and cheaper than asking an
        # LLM whether a uniquely named local document is a local-document ask.
        # A finite, already-resolved set stays file_qa; broad unscoped research
        # still goes through the router/planner below.
        if resolved_document_ids:
            tools = ["search_local_docs"]
            if _DOCUMENT_VISUAL_RE.search(task):
                tools.append("retrieve_visual_assets")
            # Coverage cardinality is not rhetorical intent. ``abstract of A
            # and B`` must retrieve both documents without being rewritten as
            # a comparative analysis.
            comparison = looks_like_document_comparison(task)
            return _finalize_decision(
                route="file_qa",
                tools=tools,
                reason="catalog_document_scope",
                confidence="high",
                allowed_tools=allowed,
                answer_intent="compare" if comparison else "direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        # A direct navigation command is a policy invariant, not a topical guess:
        # keep the existing document scope after a casual detour even if the LLM
        # router would interpret "quay lại" as general chat.
        if (
            looks_like_document_resume(task)
            and (working_topic or working_filenames or recent_document_ids)
        ):
            tools = ["search_local_docs"]
            if re.search(
                r"\b(hình|ảnh|figure|fig\.?|diagram|sơ đồ|chart|biểu đồ|plot)\b",
                task.lower(),
            ):
                tools.append("retrieve_visual_assets")
            return _finalize_decision(
                route="file_qa",
                tools=tools,
                reason="resume_document_thread",
                confidence="high",
                allowed_tools=allowed,
                answer_intent="direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        deterministic_document_reason = _deterministic_document_route_reason(
            task,
            has_active_focus=bool(
                working_topic or working_filenames or recent_document_ids
            ),
        )
        if deterministic_document_reason:
            tools = ["search_local_docs"]
            if _DOCUMENT_VISUAL_RE.search(task):
                tools.append("retrieve_visual_assets")
            return _finalize_decision(
                route="file_qa",
                tools=tools,
                reason=deterministic_document_reason,
                confidence="high",
                allowed_tools=allowed,
                answer_intent=(
                    "elaborate" if _DOCUMENT_ANALYSIS_RE.search(task) else "direct_answer"
                ),
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )

        if self.client is None:
            return _parse_failure_decision(allowed)

        try:
            payload = await self._call_router(
                task=task,
                mode=normalized_mode,
                previous_messages=previous_messages,
                has_recent_retrieval=has_recent_retrieval,
                allowed_tools=sorted(allowed),
                recent_document_ids=recent_document_ids or [],
                working_topic=working_topic,
                working_filenames=working_filenames or [],
                conversation_summary=conversation_summary,
                recent_turn_notes=recent_turn_notes,
                model=model or self.default_model,
            )
        except OllamaError:
            # A quota/timeout/provider failure is an operational failure, not a
            # routing opinion. Let the SSE layer stop and surface it instead of
            # continuing under a deterministic guess that hides the outage.
            raise
        except Exception:
            return _parse_failure_decision(allowed)

        if not payload:
            return _parse_failure_decision(allowed)

        decision = decision_from_payload(payload, allowed_tools=allowed)
        has_local_anchor = bool(
            has_recent_retrieval
            or working_topic
            or working_filenames
            or recent_document_ids
            or _LOCAL_RETRIEVAL_ANCHOR_RE.search(task)
        )
        if decision.use_local_retrieval and not has_local_anchor:
            # The model router may mistake arbitrary external/class/user-provided
            # material for Aya's private indexed corpus. Tool activation needs a
            # local grounding signal; topical similarity alone is not authority
            # to search unrelated local papers.
            return _finalize_decision(
                route="chat",
                tools=[],
                reason="unscoped_local_retrieval_rejected",
                confidence="high",
                allowed_tools=allowed,
                answer_intent=decision.answer_intent,
                answer_depth=decision.answer_depth,
                needs_fallback=False,
                max_tool_rounds=1,
            )
        return decision


    async def _call_router(
        self,
        *,
        task: str,
        mode: str,
        previous_messages: list[Any],
        has_recent_retrieval: bool,
        allowed_tools: list[str],
        recent_document_ids: list[str],
        working_topic: str | None,
        working_filenames: list[str],
        conversation_summary: str | None,
        recent_turn_notes: str | None,
        model: str,
    ) -> dict[str, Any]:
        assert self.client is not None
        history = _format_recent_messages(previous_messages, max_messages=6, max_chars=1800)
        doc_line = ", ".join(recent_document_ids[:4]) if recent_document_ids else "(none)"
        focus_files = ", ".join(working_filenames[:4]) if working_filenames else "(none)"
        focus_topic = working_topic or "(none)"
        summary_block = sentence_safe_clip(conversation_summary or "", 1_200) or "(none)"
        notes_block = sentence_safe_clip(recent_turn_notes or "", 1_500) or "(none)"
        user_content = (
            f"mode: {mode or 'research'}\n"
            f"allowed_tools: {json.dumps(allowed_tools)}\n"
            f"has_recent_retrieval: {str(has_recent_retrieval).lower()}\n"
            f"recent_document_ids: {doc_line}\n"
            f"active_working_focus: topic={focus_topic}; files={focus_files}\n"
            f"conversation_summary: {summary_block}\n"
            f"recent_turn_notes: {notes_block}\n\n"
            f"Recent conversation:\n{history}\n\n"
            f"Latest user message:\n{task}\n"
        )
        completion = await self.client.chat(
            model=model,
            temperature=0.0,
            num_predict=256,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = getattr(completion, "message", None) or str(completion)
        return _parse_json_object(raw)


def _deterministic_chat_route_reason(
    task: str,
    *,
    has_active_focus: bool,
) -> str | None:
    """Fast-path clear casual social/conversational queries when no document scope exists."""
    # A complete social acknowledgement is a chat turn even while L1 keeps a
    # document focus in the background. Requiring the provider router here adds
    # latency and can incorrectly rerun retrieval on "thanks"/"OK".
    if _PURE_SOCIAL_ACT_RE.fullmatch(task or ""):
        return "deterministic_social_acknowledgement"
    if has_active_focus:
        return None
    if _EXPLICIT_DOCUMENT_REFERENCE_RE.search(task):
        return None
    if _STRUCTURED_DOCUMENT_FACET_RE.search(task):
        return None
    if _DOCUMENT_ANALYSIS_RE.search(task):
        return None
    if _DOCUMENT_VISUAL_RE.search(task):
        return None
    if looks_like_document_comparison(task):
        return None

    normalized = " ".join((task or "").lower().split())
    if not normalized:
        return None

    # Clear conversational acts (greetings, banter, personal questions, simple chat)
    casual_patterns = (
        r"^(?:chào|chao|hi|hello|hey|yo|alo|xin\s+chào|good\s+morning|good\s+evening)\b",
        r"^(?:bạn|cậu|em|bot)\s+(?:khoẻ|khoe|là\s+ai|tên\s+là|làm\s+gì|muốn|thích|ăn|uống|nghĩ\s+sao)\b",
        r"^(?:bạn|cậu|em)\s+muốn\s+ăn\s+gì",
        r"^(?:bạn|cậu|em)\s+(?:có|biết)\s+(?:ăn|thích|chơi)\b",
        r"^(?:thời\s+tiết|hôm\s+nay\s+thế\s+nào|kể\s+chuyện|nói\s+phét)\b",
        r"^(?:tell\s+me\s+a\s+joke|how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name)\b",
    )
    for pattern in casual_patterns:
        if re.search(pattern, normalized):
            return "deterministic_casual_chat"

    return None



def _deterministic_document_route_reason(
    task: str,
    *,
    has_active_focus: bool,
) -> str | None:
    """Bypass the provider router only for structurally unambiguous local QA.

    This is a routing invariant, not a topical paper-name allowlist. Cross-paper
    comparisons still use the full router because they may need research depth
    and decomposition policy.
    """

    if not _STRUCTURED_DOCUMENT_FACET_RE.search(task):
        return None
    if looks_like_document_comparison(task):
        return None
    if _EXPLICIT_DOCUMENT_REFERENCE_RE.search(task):
        return "explicit_structured_document_request"
    if has_active_focus:
        return "focused_structured_document_followup"
    return None


async def decide_tools(
    *,
    task: str,
    mode: str,
    previous_messages: list[Any],
    has_recent_retrieval: bool,
    allowed_tools: list[str] | None,
    client: SupportsChat | None = None,
    model: str | None = None,
    recent_document_ids: list[str] | None = None,
    working_topic: str | None = None,
    working_filenames: list[str] | None = None,
    router: IntentRouterService | None = None,
) -> ToolDecision:
    service = router or IntentRouterService(
        client=client,
        default_model=model or "cx/gpt-5.6-sol",
    )
    return await service.decide(
        task=task,
        mode=mode,
        previous_messages=previous_messages,
        has_recent_retrieval=has_recent_retrieval,
        allowed_tools=allowed_tools,
        recent_document_ids=recent_document_ids,
        working_topic=working_topic,
        working_filenames=working_filenames,
        model=model,
    )


def decision_from_payload(
    payload: dict[str, Any],
    *,
    allowed_tools: set[str] | list[str] | None,
) -> ToolDecision:
    """Map router JSON → ToolDecision (used by LLM path and tests)."""
    allowed = _effective_allowed_tools(
        list(allowed_tools) if isinstance(allowed_tools, set) else allowed_tools
    )
    route = _string_choice(payload.get("route"), _VALID_ROUTES, "chat")
    tools = _normalize_tools(payload.get("selected_tools"))
    if route in {"file_qa", "research"} and "search_local_docs" not in tools:
        tools = ["search_local_docs", *tools]
    if route == "chat":
        tools = []
    reason = _snake_reason(payload.get("reason"), default="llm_router")
    confidence = _string_choice(payload.get("confidence"), _CONFIDENCES, "medium")
    answer_intent = _string_choice(payload.get("answer_intent"), _ANSWER_INTENTS, "direct_answer")
    answer_depth = _string_choice(payload.get("answer_depth"), _ANSWER_DEPTHS, "normal")
    needs_fallback = bool(payload.get("needs_fallback", False))
    max_tool_rounds = 2 if route == "research" else 1
    return _finalize_decision(
        route=route,
        tools=tools,
        reason=reason,
        confidence=confidence,
        allowed_tools=allowed,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        needs_fallback=needs_fallback if route == "chat" else False,
        max_tool_rounds=max_tool_rounds,
    )


def _finalize_decision(
    *,
    route: str,
    tools: list[str],
    reason: str,
    confidence: str,
    allowed_tools: set[str],
    answer_intent: str,
    answer_depth: str,
    needs_fallback: bool,
    max_tool_rounds: int,
) -> ToolDecision:
    selected = [tool for tool in _dedupe(tools) if tool in allowed_tools]
    if "search_local_docs" in tools and "search_local_docs" not in selected:
        selected = []
    if "retrieve_visual_assets" in selected and "search_local_docs" not in selected:
        selected = [tool for tool in selected if tool != "retrieve_visual_assets"]

    if not tools:
        final_route = "chat"
        final_reason = reason
        final_confidence = confidence
    elif not selected:
        final_route = "chat"
        final_reason = f"{reason}_tool_not_allowed"
        final_confidence = "low"
    else:
        final_route = route if route in _VALID_ROUTES and route != "chat" else "file_qa"
        final_reason = reason
        final_confidence = confidence

    return ToolDecision(
        route=final_route,
        selected_tools=selected,
        reason=final_reason,
        confidence=final_confidence,
        max_tool_rounds=max_tool_rounds if selected else 1,
        needs_fallback=needs_fallback if final_route == "chat" else False,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
    )


def _allowed_decision(
    *,
    route: str,
    tools: list[str],
    reason: str,
    confidence: str,
    allowed_tools: set[str],
    max_tool_rounds: int = 1,
    answer_intent: str = "direct_answer",
    answer_depth: str = "normal",
    needs_fallback: bool = False,
) -> ToolDecision:
    return _finalize_decision(
        route=route,
        tools=tools,
        reason=reason,
        confidence=confidence,
        allowed_tools=allowed_tools,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        needs_fallback=needs_fallback,
        max_tool_rounds=max_tool_rounds,
    )


def _parse_failure_decision(allowed_tools: set[str]) -> ToolDecision:
    _ = allowed_tools
    return ToolDecision(
        route="chat",
        selected_tools=[],
        reason="router_parse_or_llm_failure",
        confidence="low",
        max_tool_rounds=1,
        needs_fallback=True,
        answer_intent="direct_answer",
        answer_depth="normal",
    )


def _effective_allowed_tools(allowed_tools: list[str] | None) -> set[str]:
    if not allowed_tools:
        return set(DEFAULT_SAFE_TOOLS)
    return {tool for tool in allowed_tools if tool in {*DEFAULT_SAFE_TOOLS, WEB_SEARCH_TOOL}}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_tools(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    known = {*DEFAULT_SAFE_TOOLS, WEB_SEARCH_TOOL}
    tools: list[str] = []
    for item in value:
        name = str(item).strip()
        if name in known and name not in tools:
            tools.append(name)
    return tools


def answer_needs_local_fallback(answer: str, task: str) -> bool:
    _ = task
    lowered_answer = answer.lower()
    return any(marker in lowered_answer for marker in _UNCERTAIN_ANSWER_MARKERS)


def _format_recent_messages(previous_messages: list[Any], *, max_messages: int = 6, max_chars: int = 1600) -> str:
    if not previous_messages:
        return "No recent conversation context."
    lines_reversed: list[str] = []
    used = 0
    for message in reversed(previous_messages[-max_messages:]):
        if isinstance(message, dict):
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "").strip()
        else:
            role = str(getattr(message, "role", "user"))
            content = str(getattr(message, "content", "")).strip()
        if not content:
            continue
        line = f"{role}: {content}"
        if used + len(line) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            line = _clip_context_line(line, remaining)
        if not line:
            break
        lines_reversed.append(line)
        used += len(line)
    lines_reversed.reverse()
    return "\n".join(lines_reversed) if lines_reversed else "No recent conversation context."


def _clip_context_line(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit < 24:
        return text[:limit]
    head = max(12, limit // 2)
    tail = max(8, limit - head - 1)
    return f"{text[:head].rstrip()}…{text[-tail:].lstrip()}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_choice(value: Any, choices: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in choices else default


def _snake_reason(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text[:64] if text else default
