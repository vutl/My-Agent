"""Conservative, deterministic gate for turns that can skip intent routing.

Catalog/store lookups are supplied by the caller so the gate is pure and can
also replay saved evaluation turns without an LLM or a live database.
"""

import re
import unicodedata
from typing import Any

from app.services.conversation_state import looks_like_document_resume
from app.services.document_language import looks_like_document_comparison
from app.services.tool_decision_service import (
    _DOCUMENT_VISUAL_RE,
    _EXPLICIT_DOCUMENT_REFERENCE_RE,
    _LOCAL_RETRIEVAL_ANCHOR_RE,
    _VAGUE_DOCUMENT_REFERENCE_RE,
)
from .scope import _conversation_documents, _is_library_wide_question


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
                   if not unicodedata.combining(c))


_STOPWORDS = frozenset(_fold(
    "paper papers article articles document documents file files pdf docx txt md "
    "bài báo tài liệu các những của trong với một và hay cho này đó là về "
    "the a an and or of in on for to with from by my our this that these those"
).split())


def _tokens(text: str) -> list[str]:
    # Compose combining marks before splitting words, preserving case so short
    # title acronyms remain distinguishable from ordinary two-letter words.
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text), re.UNICODE)


def _title_tokens(text: str) -> set[str]:
    return {_fold(token) for token in _tokens(text)
            if (len(token) >= 3 or len(token) >= 2 and token.isupper())
            and _fold(token) not in _STOPWORDS}


def library_context_signal(
    *, request: Any, loaded_working_state: Any, scope_resolution: Any,
    recent_document_ids: list[str], scope_failure: bool = False,
) -> str | None:
    """Check request/state/scope before the caller performs library lookups."""
    if str(request.mode).strip().lower() != "research":
        return "request_mode"
    if request.collection_id is not None:
        return "collection"
    if loaded_working_state.active_document_ids:
        return "active_documents"
    if _conversation_documents(loaded_working_state):
        return "conversation_documents"
    if recent_document_ids:
        return "recent_retrieval"
    if loaded_working_state.active_topic or loaded_working_state.active_filenames:
        return "active_topic_or_filenames"
    if scope_resolution.document_ids:
        return "scope_documents"
    if scope_resolution.mentions or scope_resolution.ambiguous_mentions:
        return "scope_mentions"
    if scope_failure or scope_resolution.collection_removed_ids:
        return "scope_failure"
    return None


def library_signal(
    *, request: Any, loaded_working_state: Any, scope_resolution: Any,
    recent_document_ids: list[str], library_titles: list[tuple[str, str]],
    named_document_ids: list[str], asks_library_results: bool,
    supports_tools: bool, scope_failure: bool = False,
) -> str | None:
    """Return the first reason to retain the existing router → scope → RAG path."""
    context_signal = library_context_signal(
        request=request, loaded_working_state=loaded_working_state,
        scope_resolution=scope_resolution, recent_document_ids=recent_document_ids,
        scope_failure=scope_failure,
    )
    if context_signal is not None:
        return context_signal
    if named_document_ids:
        return "named_document_ids"
    if asks_library_results:
        return "library_results"
    task = request.task
    if _is_library_wide_question(task):
        return "library_wide"
    for name, pattern in (
        ("local_retrieval_anchor", _LOCAL_RETRIEVAL_ANCHOR_RE),
        ("vague_document_reference", _VAGUE_DOCUMENT_REFERENCE_RE),
        ("document_visual", _DOCUMENT_VISUAL_RE),
        ("explicit_document_reference", _EXPLICIT_DOCUMENT_REFERENCE_RE),
    ):
        if pattern.search(task):
            return name
    if looks_like_document_resume(task):
        return "document_resume"
    if looks_like_document_comparison(task):
        return "document_comparison"
    # Extract acronyms from the title before folding; the user's spelling can
    # be lowercase ("rl" still refers to a title's "RL").
    task_tokens = {_fold(token) for token in _tokens(task)}
    if any(task_tokens & _title_tokens(title) for _, title in library_titles):
        return "library_title_token"
    if not supports_tools:
        return "client_without_tools"
    if "assistant_tools" not in (request.allowed_tools or []):
        return "assistant_tools_disabled"
    return None
