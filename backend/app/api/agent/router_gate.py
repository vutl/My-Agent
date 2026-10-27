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
)
from .scope import _conversation_documents, _is_library_wide_question


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
                   if not unicodedata.combining(c))


_STOPWORDS = frozenset(_fold(
    "paper papers article articles document documents file files pdf docx txt md "
    "bài báo tài liệu các những của trong với một và hay cho này đó là về "
    "the a an and or of in on for to with from by my our this that these those "
    # Function and question words, now that two-letter words count towards a pair.
    "is are was be do does did can how what why when where which who it its as at me we you your "
    "gì là sao nào thế có không được ở đi đã sẽ thì mà tôi tớ cậu bạn mình nó"
).split())


def _tokens(text: str) -> list[str]:
    # Compose combining marks before splitting words, preserving case so short
    # title acronyms remain distinguishable from ordinary two-letter words.
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text), re.UNICODE)


def _strong_title_tokens(text: str) -> set[str]:
    """Name-like tokens of a title: acronyms, inner capitals, digits ("MAE", "WeNet", "wav2vec2")."""
    return {_fold(token) for token in _tokens(text)
            if len(token) >= 2 and (token.isupper() or any(c.isupper() for c in token[1:])
                                    or any(c.isdigit() for c in token))
            and _fold(token) not in _STOPWORDS}


def _title_words(text: str) -> set[str]:
    # Two-letter words count: many Vietnamese words are ("âm", "xe").
    return {_fold(token) for token in _tokens(text) if len(token) >= 2} - _STOPWORDS


_FILENAME_RE = re.compile(r"\(([^()]*\.[A-Za-z0-9]{1,5})\)\s*$")


def _filename_words(label: str) -> set[str]:
    """Words of the "(file.ext)" suffix of a library label: a file name is an identifier."""
    match = _FILENAME_RE.search(label)
    return {word for word in _title_words(match.group(1)) if len(word) >= 3} if match else set()


def _shares_title(task_tokens: set[str], title: str) -> bool:
    # One ordinary title word ("Calculator", "overfitting") is too weak to call a
    # turn a document question; a name, a file-name word or two title words are not.
    return (bool(task_tokens & (_strong_title_tokens(title) | _filename_words(title)))
            or len(task_tokens & _title_words(title)) >= 2)


def library_context_signal(
    *, request: Any, loaded_working_state: Any, scope_resolution: Any,
    recent_document_ids: list[str], scope_failure: bool = False,
) -> str | None:
    """Check request/state/scope before the caller performs library lookups."""
    # The API default ("research") and the desktop's "auto" leave routing to
    # Aya; any other mode is an explicit choice the router path honours.
    if str(request.mode).strip().lower() not in {"research", "auto"}:
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
    supports_tools: bool, scope_failure: bool = False, named_terms_in_library: bool = False,
) -> str | None:
    """Return the first reason to retain the existing router → scope → RAG path.

    A vague reference ("nó", "it", "bài kia") is not a signal here: it can only
    point at a document already in the conversation, which the context check
    above has caught."""
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
    if named_terms_in_library:
        return "library_named_term"
    task = request.task
    if _is_library_wide_question(task):
        return "library_wide"
    for name, pattern in (
        ("local_retrieval_anchor", _LOCAL_RETRIEVAL_ANCHOR_RE),
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
    if any(_shares_title(task_tokens, title) for _, title in library_titles):
        return "library_title_token"
    if not supports_tools:
        return "client_without_tools"
    if "assistant_tools" not in (request.allowed_tools or []):
        return "assistant_tools_disabled"
    return None
