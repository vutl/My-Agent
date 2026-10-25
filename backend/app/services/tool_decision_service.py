"""Hybrid LLM intent router: classify chat vs RAG vs research without topical wordlists."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Protocol

from app.llm.ollama_client import OllamaError
from app.services.long_term_memory import sentence_safe_clip

from app.services.conversation_state import looks_like_document_resume
from app.services.document_language import looks_like_document_comparison


DEFAULT_SAFE_TOOLS = {"search_local_docs", "retrieve_visual_assets"}
WEB_SEARCH_TOOL = "web_search"
# The agent route hands the turn to the tool loop (web, files, code,
# reminders…), which chooses tools itself; clients opt in with this name.
ASSISTANT_TOOLS = "assistant_tools"

_VALID_ROUTES = {"chat", "file_qa", "research", "agent"}
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
  "route": "chat" | "file_qa" | "research" | "agent",
  "selected_tools": ["search_local_docs", "retrieve_visual_assets"],
  "answer_intent": "direct_answer" | "elaborate" | "compare" | "infer_structure" | "simplify" | "example",
  "answer_depth": "brief" | "normal" | "detailed",
  "confidence": "high" | "medium" | "low",
  "needs_fallback": false,
  "library_document": "L3" | null,
  "library_documents": ["L3", "L7"],
  "referent_documents": ["L3"],
  "scope": "active_document" | "named_documents" | "library" | "none",
  "presentation": "answer" | "show_table" | "show_figure",
  "query_translation": "<the latest message in the other language>" | null,
  "sub_questions": ["<one short standalone question per fact asked>"],
  "unlisted_document": "<name as the user wrote it>" | null,
  "ambiguous_referents": ["L3", "L5"],
  "reason": "short_snake_case"
}

Rules:
- Casual, social, jokes, general world knowledge, or unrelated coding → route=chat, selected_tools=[]
- route=agent, selected_tools=["assistant_tools"] (only when "assistant_tools" is allowed): the user asks Aya to DO something or needs information that is outside the local library and changes over time — search the web, open a URL, today's news/weather/prices, read/write/organize files or folders on the computer, run code or calculate over data, set a reminder or a scheduled task, send a message. Questions about the user's indexed documents stay file_qa; stable general knowledge stays chat.
- Casual or indirect requests to DO something with a registered capability still use route=agent, selected_tools=["assistant_tools"] when allowed, even if phrased as small talk and without saying "use a tool": play/control music ("mở nhạc đi", "bật bài…"), set reminders ("nhắc tớ…"), use Apple Reminders/Calendar/Notes, open apps or links ("mở Safari"), search the web ("tìm trên mạng giúp tớ"), or create files ("tạo file…"). Intent to act takes precedence over casual tone; these are examples, not required keywords.
- Questions ABOUT Aya's abilities ("cậu làm được gì?", "what can you do?") stay route=chat, selected_tools=[]; Aya answers from her capability list. Distinguish asking about an ability from asking Aya to perform it.
- Questions about papers/models/datasets/figures in the local library, or follow-ups that clearly continue a document discussion → route=file_qa and include search_local_docs
- local_library lists the user's indexed documents as "L<n>: title". If the latest message refers to one of them — by exact name, a misspelling, an abbreviation, a translation (e.g. Vietnamese words for an English title) or a description of its topic — set library_document to that label and use route=file_qa. Use null when no listed document is meant or when several fit equally.
- A factual question whose subject is the main topic of a listed document (a question about dengue when a dengue article is listed) → route=file_qa with that document, even when the user calls it unrelated, a detour or general knowledge: the user's own documents are the preferred source. Knowledge that does not depend on any source — unit conversions, arithmetic, standard definitions — stays route=chat even when a listed document shares the topic.
- library_documents: every listed document the message refers to, by name or by description ("the model using WavLM as a global feature" and "the one with a CNN pitch encoder" in a comparison → both labels); [] when none.
- referent_documents: the documents the latest message refers to WITHOUT naming them — pronouns and references such as nó/it, bài đấy/bài này/that paper, bài kia/the other one, cái trước/the previous one, cả hai/both, cái đầu tiên/the first paper, cái thứ hai/the second. Resolve them with conversation_documents (documents discussed in this conversation, oldest first, active one marked) and the recent conversation: "this/that/it" usually means the active document; "the other one" means the discussed document that is not active, or, when only one has been discussed, the listed library document most closely related to it; "both/them" means the documents compared most recently; after a detour, "back to it" means the document discussed before the detour. [] when the message names its documents itself, starts a new topic, or refers to no document.
- scope: "library" when the answer spans the whole library (which/how many papers do X, X of each paper, papers that do NOT use Y), even if a name in it matches one document; "named_documents" for papers the user names or describes; "active_document" for follow-ups on the paper in focus; "none" for chat.
- query_translation: for document questions, the latest message translated into English if it is in Vietnamese, or into Vietnamese if it is in English, keeping names, numbers, units and technical terms as written (used to search documents written in the other language); null for chat.
- sub_questions: for a document question that asks several things or compares documents, one short standalone question per fact asked, each naming its document or entity (e.g. "What 2026 GDP growth does the World Bank May 2026 update forecast for Viet Nam?"); at most 4; [] for chat or a single simple question.
- ambiguous_referents: only when a vague reference ("bài kia", "nó", "the other one") could equally mean two or more documents of conversation_documents and nothing in the conversation settles it (e.g. "the other one" after three papers were discussed), list those documents and leave referent_documents empty; [] otherwise. Prefer resolving: "the other one" after exactly two papers is not ambiguous.
- unlisted_document: when the message asks what a specific document says (a paper, report, law, decree, circular, article…) that is NOT in local_library, its name as the user wrote it; null otherwise (also null for general questions that name no document). A different version, edition or year of a listed document counts as not listed ("the WeNet 2.0 paper" when only the original WeNet paper is listed; "the 2019 report" when only 2023 is).
- presentation: "show_table"/"show_figure" only when the user asks to see/display/get a table or figure itself; any question about numbers or contents (how much, which is best, is it true that…) is "answer", even when the answer comes from a table.
- Do not assume that phrases such as "class materials", "provided context", "the text below", or a generic request for sources refer to Aya's indexed local library. With no resolved local document, active document thread, or explicit local-library request, use route=chat.
- If active_working_focus names a paper/topic and the user asks short follow-ups (benchmark, bảng, Acc/F1/CCC, architecture, figure) without naming a different paper → stay file_qa on that focus; do not treat metrics as a new topic
- Add retrieve_visual_assets when the user asks about figures, diagrams, architecture drawings, charts, or images
- Broad compare/synthesize across multiple papers → route=research, tools include search_local_docs, prefer max detail when useful
- needs_fallback=true only when you chose chat but the question might actually need the local corpus and you are uncertain
- Prefer Vietnamese-friendly answer_intent/depth matching how the user asked
- Never invent tools outside the allowed list provided in the user message
"""


_SCOPES = frozenset({"active_document", "named_documents", "library", "none"})
_PRESENTATIONS = frozenset({"answer", "show_table", "show_figure"})


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
    # Library document the router recognised from a name, typo, translation
    # or description ("bài về ngôn ngữ thanh điệu"); None when not applicable.
    library_document_id: str | None = None
    # Router's reading of the turn; None when the decision came from a
    # heuristic fast path (callers then use their own fallbacks).
    scope: str | None = None
    presentation: str | None = None
    library_document_ids: tuple[str, ...] = ()
    query_translation: str | None = None
    # Documents a vague reference ("nó", "bài kia", "the other one") points to.
    referent_document_ids: tuple[str, ...] = ()
    # One standalone question per fact the turn asks (retrieval inside documents).
    sub_questions: tuple[str, ...] = ()
    # A specific document the user asks about that is not in the library.
    unlisted_document: str | None = None
    # Documents a vague reference could equally mean (the user should pick).
    ambiguous_referent_ids: tuple[str, ...] = ()

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
            "library_document_id": self.library_document_id,
            "scope": self.scope,
            "presentation": self.presentation,
            "library_document_ids": list(self.library_document_ids),
            "query_translation": self.query_translation,
            "referent_document_ids": list(self.referent_document_ids),
            "sub_questions": list(self.sub_questions),
            "unlisted_document": self.unlisted_document,
            "ambiguous_referent_ids": list(self.ambiguous_referent_ids),
        }


class IntentRouterService:
    """LLM-backed turn policy with soft allowlist / mode gates."""

    @staticmethod
    def _fallback_decision(
        task: str, inferred_document_ids: list[str] | None, allowed: set[str], document_turn: bool,
    ) -> ToolDecision:
        """Decision when the model router fails: the deterministic scope guess,
        else (a vague reference or a named-document request) local retrieval,
        else the generic parse-failure policy."""
        decision = _inferred_scope_decision(task, inferred_document_ids, allowed)
        if decision is None and document_turn:
            decision = _finalize_decision(
                route="file_qa",
                tools=["search_local_docs"],
                reason="document_turn_fallback",
                confidence="medium",
                allowed_tools=allowed,
                answer_intent="compare" if looks_like_document_comparison(task) else "direct_answer",
                answer_depth="normal",
                needs_fallback=False,
                max_tool_rounds=1,
            )
        return decision or _parse_failure_decision(allowed)

    def __init__(
        self,
        *,
        client: SupportsChat | None = None,
        default_model: str = "cx/gpt-6.1-sol",
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
        library_titles: list[tuple[str, str]] | None = None,
        conversation_documents: list[tuple[str, bool]] | None = None,
        inferred_document_ids: list[str] | None = None,
    ) -> ToolDecision:
        """``resolved_document_ids``: documents the turn names (skips the model).
        ``inferred_document_ids``: the deterministic guess for a turn that names
        none ("nó", "bài kia"); the model decides, and this guess stands only
        when the model cannot answer."""
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

        # "Còn bài kia?", "nó", "the first paper", "back to it": with documents
        # already in the conversation, which one is meant is the model's call;
        # the deterministic shortcuts below would keep the active document.
        vague_reference = bool(
            working_topic or working_filenames or recent_document_ids
        ) and bool(_VAGUE_DOCUMENT_REFERENCE_RE.search(task))

        # A direct navigation command is a policy invariant, not a topical guess:
        # keep the existing document scope after a casual detour even if the LLM
        # router would interpret "quay lại" as general chat.
        if (
            looks_like_document_resume(task)
            and (working_topic or working_filenames or recent_document_ids)
            # With several documents in the conversation, "back to it" needs
            # the model to say which one.
            and not inferred_document_ids
            and not vague_reference
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
        # A document named in the turn that the catalog could not resolve
        # ("bài MAE" for "Masked Autoencoders…") is the router's to identify:
        # it reads every title. Only a follow-up on the focused document skips it.
        if (
            deterministic_document_reason == "focused_structured_document_followup"
            and not inferred_document_ids and not vague_reference
        ):
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
                library_titles=library_titles or [],
                conversation_documents=conversation_documents or [],
            )
        except OllamaError:
            # A quota/timeout/provider failure is an operational failure, not a
            # routing opinion. Let the SSE layer stop and surface it instead of
            # continuing under a deterministic guess that hides the outage.
            raise
        except Exception:
            return self._fallback_decision(
                task, inferred_document_ids, allowed, vague_reference or bool(deterministic_document_reason)
            )

        if not payload:
            return self._fallback_decision(
                task, inferred_document_ids, allowed, vague_reference or bool(deterministic_document_reason)
            )

        decision = decision_from_payload(payload, allowed_tools=allowed)
        scope = payload.get("scope") if payload.get("scope") in _SCOPES else None
        presentation = payload.get("presentation") if payload.get("presentation") in _PRESENTATIONS else None
        labels = payload.get("library_documents") if isinstance(payload.get("library_documents"), list) else []
        library_ids = tuple(dict.fromkeys(
            document_id
            for document_id in (_library_label_to_id(label, library_titles or []) for label in labels)
            if document_id
        ))
        translation = payload.get("query_translation")
        translation = " ".join(str(translation).split())[:400] if isinstance(translation, str) and translation.strip() else None
        referent_labels = payload.get("referent_documents") if isinstance(payload.get("referent_documents"), list) else []
        referent_ids = tuple(dict.fromkeys(
            document_id
            for document_id in (_library_label_to_id(label, library_titles or []) for label in referent_labels)
            if document_id
        ))
        raw_sub_questions = payload.get("sub_questions") if isinstance(payload.get("sub_questions"), list) else []
        sub_questions = tuple(
            " ".join(str(item).split())[:300] for item in raw_sub_questions[:4] if str(item or "").strip()
        )
        unlisted = payload.get("unlisted_document")
        unlisted = " ".join(str(unlisted).split())[:160] if isinstance(unlisted, str) and unlisted.strip() else None
        decision = replace(
            decision, scope=scope, presentation=presentation, library_document_ids=library_ids,
            query_translation=translation, referent_document_ids=referent_ids, sub_questions=sub_questions,
            unlisted_document=unlisted,
            ambiguous_referent_ids=tuple(dict.fromkeys(
                document_id
                for document_id in (
                    _library_label_to_id(label, library_titles or [])
                    for label in (payload.get("ambiguous_referents") if isinstance(payload.get("ambiguous_referents"), list) else [])
                )
                if document_id
            )) if not referent_ids else (),
        )
        library_document_id = _library_label_to_id(payload.get("library_document"), library_titles or [])
        if library_document_id:
            decision = replace(decision, library_document_id=library_document_id)
        has_local_anchor = bool(
            library_document_id
            or decision.library_document_ids
            or decision.referent_document_ids
            or has_recent_retrieval
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
        library_titles: list[tuple[str, str]] | None = None,
        conversation_documents: list[tuple[str, bool]] | None = None,
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
            f"recent_turn_notes: {notes_block}\n"
            f"local_library:\n{_library_block(library_titles or [])}\n\n"
            f"conversation_documents:\n{_conversation_documents_block(conversation_documents or [], library_titles or [])}\n\n"
            f"Recent conversation:\n{history}\n\n"
            f"Latest user message:\n{task}\n"
        )
        completion = await self.client.chat(
            model=model,
            temperature=0.0,
            num_predict=600,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = getattr(completion, "message", None) or str(completion)
        return _parse_json_object(raw)


_LIBRARY_PROMPT_LIMIT = 60


def _library_block(library_titles: list[tuple[str, str]]) -> str:
    if not library_titles:
        return "(empty)"
    lines = [
        f"L{index}: {sentence_safe_clip(title, 140)}"
        for index, (_document_id, title) in enumerate(library_titles[:_LIBRARY_PROMPT_LIMIT], start=1)
    ]
    if len(library_titles) > _LIBRARY_PROMPT_LIMIT:
        lines.append(f"... and {len(library_titles) - _LIBRARY_PROMPT_LIMIT} more (not listed)")
    return "\n".join(lines)


def _conversation_documents_block(
    documents: list[tuple[str, bool]], library_titles: list[tuple[str, str]],
) -> str:
    """Documents discussed in this conversation, oldest first, by library label."""
    labels = {
        document_id: f"L{index}"
        for index, (document_id, _title) in enumerate(library_titles[:_LIBRARY_PROMPT_LIMIT], start=1)
    }
    titles = dict(library_titles)
    lines = []
    for document_id, active in documents:
        label = labels.get(document_id)
        if not label:
            continue
        title = sentence_safe_clip(titles.get(document_id, ""), 100)
        lines.append(f"- {label}: {title}" + (" (active)" if active else ""))
    return "\n".join(lines) if lines else "(none)"


_VAGUE_DOCUMENT_REFERENCE_RE = re.compile(
    r"(?<!\w)(?:"
    r"nó|bài\s+(?:này|đó|đấy|ấy|kia|trước|lúc\s+nãy|vừa\s+rồi|đầu(?:\s+tiên)?|thứ\s+(?:nhất|hai|ba)|còn\s+lại)"
    r"|cái\s+(?:kia|trước|đầu(?:\s+tiên)?|thứ\s+(?:nhất|hai|ba)|còn\s+lại)|cả\s+hai|hai\s+bài"
    r"|it|its|both|the\s+(?:other|former|latter|previous)(?:\s+(?:one|paper|document|report))?"
    r"|the\s+(?:first|second|third)\s+(?:one|paper|document|report)"
    r"|(?:that|this)\s+(?:paper|document|report|one)|back\s+to\s+it"
    r")(?!\w)",
    flags=re.IGNORECASE,
)


def _inferred_scope_decision(
    task: str, inferred_document_ids: list[str] | None, allowed: set[str],
) -> ToolDecision | None:
    """The deterministic follow-up decision used when the model router fails."""
    if not inferred_document_ids:
        return None
    tools = ["search_local_docs"]
    if _DOCUMENT_VISUAL_RE.search(task):
        tools.append("retrieve_visual_assets")
    return _finalize_decision(
        route="file_qa",
        tools=tools,
        reason="inferred_document_scope",
        confidence="medium",
        allowed_tools=allowed,
        answer_intent="compare" if looks_like_document_comparison(task) else "direct_answer",
        answer_depth="normal",
        needs_fallback=False,
        max_tool_rounds=1,
    )


def _library_label_to_id(label: Any, library_titles: list[tuple[str, str]]) -> str | None:
    match = re.fullmatch(r"\s*L?(\d+)\s*", str(label or ""))
    if not match:
        return None
    index = int(match.group(1))
    if 1 <= index <= min(len(library_titles), _LIBRARY_PROMPT_LIMIT):
        return library_titles[index - 1][0]
    return None


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
    # Greetings must fill the whole message: any further request belongs to
    # the router, regardless of punctuation or whether tools are allowed.
    casual_patterns = (
        r"^(?:chào|chao|hi|hello|hey|yo|alo|xin\s+chào|good\s+morning|good\s+evening)"
        r"(?:[\s,]+(?:aya|cậu|bạn|ban|em|bot|there))?[\s,.!?…]*$",
        r"^(?:bạn|cậu|em|bot)\s+(?:khoẻ|khoe|là\s+ai|tên\s+là|làm\s+gì|muốn|thích|ăn|uống|nghĩ\s+sao)\b",
        r"^(?:bạn|cậu|em)\s+muốn\s+ăn\s+gì",
        r"^(?:bạn|cậu|em)\s+(?:có|biết)\s+(?:ăn|thích|chơi)\b",
        r"^(?:hôm\s+nay\s+thế\s+nào|kể\s+chuyện|nói\s+phét)\b",
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
        default_model=model or "cx/gpt-6.1-sol",
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
    if route == "agent":
        tools = [ASSISTANT_TOOLS]
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
    return {tool for tool in allowed_tools if tool in {*DEFAULT_SAFE_TOOLS, WEB_SEARCH_TOOL, ASSISTANT_TOOLS}}


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
    known = {*DEFAULT_SAFE_TOOLS, WEB_SEARCH_TOOL, ASSISTANT_TOOLS}
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
