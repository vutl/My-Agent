from collections.abc import AsyncIterator, Callable
from dataclasses import replace
import re
from typing import Annotated, Any
from app.rag.document_coverage import DocumentCoveragePolicy, detect_document_coverage
from app.rag.figure_caption import (
    best_figure_caption,
    figure_relevance_score,
    requested_figure_number,
)
from app.services.document_scope_service import DocumentScopeResolution, resolve_document_scope
from app.services.query_rewrite_service import (
    QueryRewriteResult,
    QueryRewriteService,
    _classify_answer_depth,
    _classify_answer_intent,
    _explicit_document_target_entities,
    _is_named_entity,
    _looks_like_topic_switch,
    _query_named_entities,
    enrich_retrieval_query,
    format_recent_conversation,
    has_result_table_intent,
    has_visual_intent,
    requires_visual_pixels,
    wants_single_figure,
)
from app.services.rag_service import (
    RagService,
    caption_identifies_figure,
    requested_visual_page_numbers,
    visual_page_reservation_count,
)
from .common import FULL_DOCUMENT_MAX_DOCUMENTS
from .common import logger
from .tables import _requested_table_numbers


def _resolve_answer_document_ids(
    resolver: Callable[[str], list[str]] | None,
    answer: str,
) -> list[str] | None:
    if resolver is None:
        return None
    try:
        return list(dict.fromkeys(str(item) for item in resolver(answer) if item))
    except Exception as exc:  # pragma: no cover - defensive, validation fails closed
        logger.warning(
            "Answer document identity resolution failed: %s",
            type(exc).__name__,
        )
        return []


def _should_reuse_last_retrieval(rewrite: Any, *, original_query: str) -> bool:
    # Page/document-wide aggregation needs a fresh bounded coverage frontier;
    # reusing a prior compact fact retrieval would silently hide other pages.
    if has_visual_intent(original_query) or detect_document_coverage(original_query).enabled:
        return False
    return bool(
        rewrite.is_followup
        and rewrite.use_last_sources
        and rewrite.answer_intent in {"elaborate", "simplify", "example", "infer_structure"}
    )


def _context_budget(answer_intent: str | None, answer_depth: str | None) -> dict[str, int]:
    """Prompt evidence budget per answer shape.

    Sized like a document chat product rather than a snippet router: the
    public-36 artifact-recall gate fell from 0.403 to 0.653 when the source
    count rose from 3-5 to 10, with unchanged retrieval latency and no drop on
    the local 60-case audit (see 00-STATUS, 2026-10-13). Brief answers keep a
    smaller frontier so they stay quick.
    """
    detailed = answer_depth == "detailed" or answer_intent in {"elaborate", "compare", "infer_structure"}
    if detailed:
        return {
            "max_sources": 10,
            "max_chars": 16_000,
            "max_chars_per_source": 1_600,
            "max_chunks_per_document": 4,
            "neighbor_window": 1,
            "max_neighbor_chars": 700,
        }
    if answer_depth == "brief":
        return {
            "max_sources": 6,
            "max_chars": 8_000,
            "max_chars_per_source": 1_200,
            "max_chunks_per_document": 3,
            "neighbor_window": 1,
            "max_neighbor_chars": 480,
        }
    return {
        "max_sources": 10,
        "max_chars": 16_000,
        "max_chars_per_source": 1_600,
        "max_chunks_per_document": 4,
        "neighbor_window": 1,
        "max_neighbor_chars": 700,
    }


def _document_ids_from_retrieval(retrieval: dict[str, Any] | None) -> list[str]:
    if not retrieval:
        return []
    seen: set[str] = set()
    document_ids: list[str] = []
    for document in retrieval.get("documents") or []:
        document_id = document.get("document_id")
        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


def _filenames_from_documents(documents: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    filenames: list[str] = []
    for document in documents:
        name = str(document.get("filename") or "").strip()
        if name and name not in seen:
            seen.add(name)
            filenames.append(name)
    return filenames


def _scope_documents_to_focus(
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not focus_document_ids:
        return documents
    allowed = {str(doc_id) for doc_id in focus_document_ids}
    # Never fall back to unscoped corpus — empty is safer than cross-paper leak.
    return [
        document
        for document in documents
        if str(document.get("document_id") or "") in allowed
    ]


_PAPER_ARTIFACT_RE = re.compile(
    r"\b(?:tables?|figures?|fig\.?|charts?|plots?|diagrams?|equations?|formulas?|"
    r"bảng|hình|biểu đồ|sơ đồ|đồ thị|công thức|phương trình)\b",
    re.IGNORECASE,
)
_LIBRARY_WIDE_RE = re.compile(
    r"\b(?:papers|documents|which paper|what paper|any paper|other paper|all paper|"
    r"các (?:paper|bài|tài liệu)|những (?:paper|bài|tài liệu)|paper nào|bài nào|tài liệu nào|"
    r"bài khác|paper khác)\b",
    re.IGNORECASE,
)


_LIBRARY_SCOPE_RE = re.compile(
    r"\b(?:how many (?:papers|documents)|each (?:paper|document)|every (?:paper|document)|"
    r"all (?:the )?(?:papers|documents)|in (?:my|the) (?:library|collection)|across (?:the )?papers|"
    r"bao nhiêu (?:paper|bài|tài liệu)|mỗi (?:paper|bài|tài liệu)|từng (?:paper|bài|tài liệu)|"
    r"tất cả (?:các )?(?:paper|bài|tài liệu)|trong (?:thư viện|kho)|bao nhieu (?:paper|bai)|"
    r"moi (?:paper|bai)|tung (?:paper|bai)|tat ca (?:cac )?(?:paper|bai)|paper nao|bai nao|tai lieu nao)\b",
    re.IGNORECASE,
)


_WHICH_PAPERS_RE = re.compile(
    r"\b(?:which|any|other)(?: of (?:my|the|these|those))? "
    r"(?:papers?|documents?|works?|files?|reports?|articles?|sources?)\b|"
    r"\bwhat (?:of my )?(?:papers|documents|works|files|reports|articles|sources)\b|"
    r"\b(?:paper|bài|tài liệu|bai|tai lieu|file|văn bản|van ban|báo cáo|bao cao) (?:nào|nao)\b|"
    r"\b(?:các|những|cac|nhung) (?:paper|bài báo|bai bao)\b",
    re.IGNORECASE,
)
# "these two papers", "hai bài đó": a referent, not the library.
_REFERENT_PAPERS_RE = re.compile(
    r"\b(?:these|those|both|the two|two) (?:papers|documents)\b|"
    r"\b(?:hai|2|cả hai|ca hai) (?:paper|bài|tài liệu|bai)\b|\b(?:paper|bài|bai) (?:này|đó|kia|nay|do)\b",
    re.IGNORECASE,
)


def _is_library_wide_question(query: str) -> bool:
    """"Which papers use Whisper?", "GPU của từng paper": the answer spans the
    whole library, not the paper in focus."""
    text = str(query or "")
    if _REFERENT_PAPERS_RE.search(text):
        return False
    return bool(_LIBRARY_SCOPE_RE.search(text) or _WHICH_PAPERS_RE.search(text))


def _asks_for_active_paper_artifact(query: str) -> bool:
    """"Show the comparison table on IEMOCAP and LSSED": a table, figure or
    formula asked for without a paper name belongs to the paper in focus,
    even when dataset or metric names make it look like a new topic."""
    text = str(query or "")
    return bool(_PAPER_ARTIFACT_RE.search(text)) and not _LIBRARY_WIDE_RE.search(text)


def _task_requires_exact_artifact(query: str) -> bool:
    """Exact artifact/quote/page asks must bypass compact semantic cards."""

    lowered = " ".join(str(query or "").lower().split())
    return bool(
        _requested_table_numbers(query)
        or requested_figure_number(query) is not None
        or re.search(r"\b(?:page|trang)\s*\d+\b", lowered)
        or any(marker in lowered for marker in ("exact quote", "verbatim", "trích nguyên", "trich nguyen"))
    )


def _safe_library_titles(
    rag: RagService, collection_id: str | None, *, raise_on_error: bool = False,
) -> list[tuple[str, str]]:
    try:
        return rag.library_titles(collection_id)
    except Exception:  # noqa: BLE001 - routing must not fail on catalog metadata
        if raise_on_error:
            raise  # A bypass gate must distinguish an empty catalog from missing evidence.
        return []


def _query_continues_active_paper(document_id: str, working_state: Any) -> bool:
    return document_id in set(working_state.active_document_ids or [])


def _query_names_new_paper(
    query: str,
    working_state: Any,
    *,
    rag: RagService | None = None,
    collection_id: str | None = None,
) -> bool:
    """Whether the turn names a paper other than the active one.

    Datasets, metrics and method names (IEMOCAP, WavLM, MPCL) are named
    entities too, but they are not papers: with a catalog available, an entity
    switches papers only when it resolves to exactly one document that is not
    already active.
    """
    from app.services.query_rewrite_service import _query_named_entities

    named = _query_named_entities(query)
    if not named:
        return False
    active_topic = (working_state.active_topic or "").lower()
    active_files = " ".join(working_state.active_filenames or []).lower()
    active_ids = set(working_state.active_document_ids or [])
    for entity in named:
        key = entity.lower()
        if not key or key in active_topic or key in active_files:
            continue
        if rag is None:
            return True
        resolved = _decisive_document_ids(rag, entity, collection_id=collection_id)
        if resolved and resolved[0] not in active_ids:
            return True
    return False


def _decisive_document_ids(
    rag: RagService, entity: str, *, collection_id: str | None,
) -> list[str]:
    """Documents an entity identifies unambiguously (exactly one), else none."""
    try:
        resolved = rag.resolve_document_ids_for_entities(
            entities=[entity], collection_id=collection_id, query=entity,
        )
    except Exception:  # noqa: BLE001 - catalog lookups must never break a turn
        return []
    return list(resolved) if len(resolved) == 1 else []


def _canonicalize_explicit_document_target(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
) -> tuple[Any, list[str]]:
    """Make a current-turn catalog target authoritative over stale chat focus.

    The ordinary rewriter intentionally understands conversational language,
    but it cannot know every filename or alias in the user's local catalog.
    Resolve explicit ``paper/file/Table N X`` references against that catalog,
    then keep query, topic, entities and focus consistent.  A model name used
    only as a baseline inside the active paper is not treated as a document
    switch because this path requires explicit document language.
    """

    query = str(rewrite.original_query or "").strip()
    explicit_targets = _explicit_document_target_entities(query)
    has_document_marker = bool(
        re.search(
            r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
            query,
            flags=re.IGNORECASE,
        )
    )
    has_explicit_filename = bool(re.search(r"\.(?:pdf|md|txt)(?!\w)", query, re.I))
    if explicit_targets:
        lookup_entities = list(explicit_targets)
    elif has_document_marker or has_explicit_filename:
        lookup_entities = _query_named_entities(query)
    else:
        return rewrite, []

    document_ids = rag.resolve_explicit_document_ids_for_query(
        query=query,
        entities=lookup_entities,
        collection_id=collection_id,
        compare=rewrite.answer_intent == "compare",
    )
    if not document_ids:
        return rewrite, []

    if len(lookup_entities) == len(document_ids):
        canonical_entities = lookup_entities
    else:
        canonical_entities = []
        for document_id in document_ids:
            document = rag.get_document(document_id) or {}
            filename = str(document.get("filename") or document_id)
            canonical_entities.append(re.sub(r"\.[A-Za-z0-9]+$", "", filename))

    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "explicit_catalog_document_target",
            "document_ids": document_ids,
        }
    )
    return (
        replace(
            rewrite,
            standalone_query=query,
            is_followup=False,
            current_topic=canonical_entities[0] if canonical_entities else rewrite.current_topic,
            required_entities=canonical_entities,
            use_last_sources=False,
            rewrite_used=False,
            diagnostics=diagnostics,
        ),
        document_ids,
    )


def _apply_document_scope_to_rewrite(rewrite: Any, *, scope: Any) -> Any:
    """Keep rewrite facets while making the pre-router scope authoritative."""

    document_ids = list(scope.document_ids)
    labels = list(scope.labels)
    if not document_ids:
        return rewrite
    query = str(rewrite.original_query or "").strip()
    is_referent = scope.source == "plural_referent"
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": scope.source,
            "document_ids": document_ids,
            "catalog_labels": labels,
            "must_cover_all": bool(scope.must_cover_all),
        }
    )
    topic = " / ".join(labels) if len(labels) >= 2 else labels[0] if labels else None
    standalone_query = (
        f"{' '.join(labels)} {query}".strip() if is_referent else query
    )
    return replace(
        rewrite,
        standalone_query=standalone_query,
        is_followup=is_referent,
        current_topic=topic or rewrite.current_topic,
        required_entities=labels or list(rewrite.required_entities or []),
        use_last_sources=is_referent,
        answer_intent=rewrite.answer_intent,
        rewrite_used=False,
        diagnostics=diagnostics,
    )


# "bài kia", "bài giảng còn lại", "the other one" — not ordinals ("the second paper").
_OTHER_REFERENCE_RE = re.compile(
    r"(?<!\w)(?:bài(?:\s+(?:giảng|báo|viết))?|cái|paper|tài\s+liệu|file|báo\s+cáo|văn\s+bản)\s+(?:kia|còn\s+lại)(?!\w)"
    r"|(?<!\w)the\s+other\s+(?:one|paper|document|report|file|article|lecture)(?!\w)",
    re.IGNORECASE,
)


def _ambiguous_reference_question(labels: list[str], task: str) -> str:
    from app.agents.graph import _answer_language

    names = [
        (label.split(":", 1)[0] if ":" in label and len(label.split(":", 1)[0].split()) <= 6 else label).strip()
        for label in dict.fromkeys(labels) if label
    ]
    if _answer_language(task) == "Vietnamese":
        listed = ", ".join(names[:-1]) + f" hay {names[-1]}" if len(names) > 1 else names[0]
        return f"Cậu đang hỏi bài nào nè: {listed}? Mình hỏi lại cho chắc, kẻo trả lời nhầm bài~"
    listed = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
    return f"Which one do you mean: {listed}? Just checking so I answer about the right document~"


def _switched_document_note(
    previous_documents: dict[str, str], focus_document_ids: list[str], retrieved_docs: list[dict],
) -> str:
    """When this turn's documents are not the previous turn's, say so: the
    conversation history otherwise pulls the answer back to the old one."""
    if not previous_documents or not focus_document_ids or set(focus_document_ids) & set(previous_documents):
        return ""
    current: list[str] = []
    for document in retrieved_docs or []:
        if str(document.get("document_id") or "") in set(focus_document_ids):
            label = str(document.get("title") or document.get("document_title") or document.get("filename") or "")
            if label and label not in current:
                current.append(label)
    previous_labels = [label for label in previous_documents.values() if label]
    if not current or not previous_labels:
        return ""
    return (
        f"The document meant in this message is {'; '.join(current)}, not "
        f"{'; '.join(previous_labels)} from the previous turn: answer about it and name it as such."
    )


def _unlisted_document_note(name: str | None) -> str:
    """Chat-route instruction for a question about a document the library
    does not hold: say so, and keep remembered specifics from passing as the
    document's content."""
    if not name:
        return ""
    return (
        f"The user asks about \"{name}\", which is not among their indexed documents. "
        "Say so in your first sentence. If the library excerpts (if any) report the same "
        "subject, give what they say with their citations and name the document they come "
        "from. Give no figures, amounts, rates, dates or article numbers from memory; "
        "general background without such specifics is fine, labelled as general knowledge. "
        "Suggest adding the document for an exact answer."
    )


# Catalog matches certain enough to skip the router: the file name itself, an
# identifier or an identifier-like token of it (GW151226).
_EXACT_ALIAS_SOURCES = frozenset({"filename", "filename_identifier", "filename_token"})


def _catalog_scope_is_loose(scope: DocumentScopeResolution) -> bool:
    """Whether a catalog scope rests on any name match weaker than a file name
    or identifier (a title, a phrase, a description, a context guess)."""
    mentions = [mention for mention in scope.mentions if mention.document_id in scope.document_ids]
    if not mentions:
        # A name shared by 2–3 documents ("USDA" for two WASDE reports while
        # the user means a 1905 bulletin) is loose too.
        return scope.source in {"current_turn_mentions", "explicit_current_turn", "small_ambiguity_set"}
    return any(mention.alias_source not in _EXACT_ALIAS_SOURCES for mention in mentions)


# Scopes the deterministic resolver guessed from context rather than read from
# a name in the turn; the model router decides these ("nó", "bài kia", "cả hai").
INFERRED_SCOPE_SOURCES = frozenset({
    "active_focus", "active_anaphor", "other_referent", "plural_referent", "bounded_plural_referent",
})


def _conversation_documents(state: Any) -> list[tuple[str, bool]]:
    """Documents discussed in the conversation, oldest first; the active ones last."""
    active = [str(item) for item in getattr(state, "active_document_ids", None) or [] if item]
    ordered: list[str] = []
    for thread in getattr(state, "recent_document_threads", None) or ():
        for document_id in thread.document_ids:
            if document_id in ordered:
                ordered.remove(document_id)
            ordered.append(document_id)
    for document_id in [*(getattr(state, "referent_document_ids", None) or []), *active]:
        if document_id in ordered:
            ordered.remove(document_id)
        ordered.append(document_id)
    return [(document_id, document_id in active) for document_id in ordered[-8:]]


def _labels_for_document_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    labels = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        labels.append(str(document.get("title") or document.get("filename") or document_id))
    return labels


def _working_state_for_scope(rag: RagService, working_state: Any, scope: Any, task: str) -> Any:
    """Make an authoritative turn scope the conversation's active documents."""
    scoped_filenames = _filenames_for_document_ids(rag, list(scope.document_ids))
    scoped_topic = (
        " / ".join(scope.labels)
        if len(scope.labels) >= 2
        else scope.labels[0]
        if scope.labels
        else working_state.active_topic
    )
    return replace(
        working_state,
        active_document_ids=list(scope.document_ids),
        active_topic=scoped_topic,
        active_filenames=scoped_filenames,
        last_answer_intent=_classify_answer_intent(task),
        referent_document_ids=(
            list(scope.document_ids) if scope.must_cover_all else working_state.referent_document_ids
        ),
        referent_filenames=(
            scoped_filenames if scope.must_cover_all else working_state.referent_filenames
        ),
        referent_topic=(
            scoped_topic if scope.must_cover_all else working_state.referent_topic
        ),
    )


def _with_sibling_documents(rag: RagService, document_ids: list[str], collection_id: str | None) -> list[str]:
    """``document_ids`` plus each one's closest sibling (same series or
    companion document), at most FULL_DOCUMENT_MAX_DOCUMENTS in total."""
    expanded = list(dict.fromkeys(document_ids))
    finder = getattr(rag, "closest_sibling_document", None)
    if not callable(finder):
        return expanded
    for document_id in list(expanded):
        if len(expanded) >= FULL_DOCUMENT_MAX_DOCUMENTS:
            break
        sibling = finder(document_id, collection_id)
        if sibling and sibling not in expanded:
            expanded.append(sibling)
    return expanded


def _document_scope_failure_message(rag: RagService, scope: Any) -> str | None:
    if not getattr(scope, "authoritative", False) or getattr(scope, "document_ids", ()):
        return None
    if scope.source == "collection_excluded":
        filenames = _filenames_for_document_ids(
            rag,
            list(scope.collection_removed_ids),
        )
        listed = ", ".join(filenames) or "tài liệu được nhắc tới"
        return (
            f"Mình nhận ra đúng nguồn ({listed}), nhưng nguồn đó nằm ngoài collection "
            "đang được giới hạn. Mình không tự mở rộng sang toàn corpus; hãy đổi collection "
            "hoặc bỏ giới hạn rồi hỏi lại."
        )
    if scope.source == "ambiguous_current_turn":
        candidate_ids: list[str] = []
        surfaces: list[str] = []
        for mention in scope.ambiguous_mentions:
            candidate_ids.extend(mention.candidate_ids)
            if mention.surface not in surfaces:
                surfaces.append(mention.surface)
        filenames = _filenames_for_document_ids(
            rag,
            list(dict.fromkeys(candidate_ids)),
        )
        target = ", ".join(surfaces) or "tên đó"
        choices = "; ".join(filenames) or "nhiều file trong catalog"
        return (
            f"Tên “{target}” khớp nhiều tài liệu: {choices}. "
            "Hãy ghi filename hoặc tiêu đề đầy đủ để mình không chọn nhầm."
        )
    return None


def _canonicalize_natural_document_mentions(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
    existing_focus: list[str],
    answer_intent_hint: str | None = None,
) -> tuple[Any, list[str]]:
    """Turn unambiguous catalog mentions into a coherent retrieval scope.

    This covers case/separator variants such as ``msf ser``, ``MSF-SER`` and
    ``msfser`` without teaching the router individual paper names.  A single
    model mention does not dislodge a different sticky paper unless the turn is
    a clear switch/comparison; two or more unique catalog identities are
    authoritative for the current turn.
    """

    query = str(rewrite.original_query or "").strip()
    resolver = getattr(rag, "resolve_document_mentions_for_query", None)
    if not callable(resolver):
        return rewrite, []
    document_ids = resolver(
        query=query,
        collection_id=collection_id,
        limit=8,
    )
    if not document_ids:
        return rewrite, []

    comparison_request = (
        rewrite.answer_intent == "compare"
        or answer_intent_hint == "compare"
        or _looks_like_multi_document_comparison(query)
    )
    active = set(existing_focus)
    if (
        len(document_ids) == 1
        and active
        and document_ids[0] not in active
        and not comparison_request
        and not _looks_like_topic_switch(query)
        and (rewrite.diagnostics or {}).get("reason") != "topic_switch"
    ):
        return rewrite, []

    labels = _document_labels_for_ids(rag, document_ids)
    if not labels:
        return rewrite, []
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "catalog_document_mentions",
            "document_ids": document_ids,
            "catalog_labels": labels,
            "multi_document": len(document_ids) >= 2,
        }
    )
    intent = "compare" if comparison_request and len(document_ids) >= 2 else rewrite.answer_intent
    topic = labels[0] if len(labels) == 1 else " vs ".join(labels)
    return (
        replace(
            rewrite,
            standalone_query=query,
            is_followup=False,
            current_topic=topic,
            required_entities=labels,
            use_last_sources=False,
            answer_intent=intent,
            rewrite_used=False,
            diagnostics=diagnostics,
        ),
        document_ids,
    )


def _canonicalize_plural_document_referent(
    rag: RagService,
    *,
    rewrite: Any,
    document_ids: list[str],
) -> Any:
    """Bind plural anaphora to the last grounded multi-document referent set."""

    unique_ids = list(dict.fromkeys(str(item) for item in document_ids if item))[:8]
    if len(unique_ids) < 2:
        return rewrite
    labels = _document_labels_for_ids(rag, unique_ids)
    if len(labels) < 2:
        return rewrite
    query = str(rewrite.original_query or "").strip()
    diagnostics = dict(rewrite.diagnostics or {})
    diagnostics.update(
        {
            "reason": "plural_document_referent",
            "document_ids": unique_ids,
            "catalog_labels": labels,
        }
    )
    return replace(
        rewrite,
        standalone_query=f"{' '.join(labels)} {query}".strip(),
        is_followup=True,
        current_topic=" vs ".join(labels),
        required_entities=labels,
        use_last_sources=True,
        answer_intent="compare",
        rewrite_used=False,
        diagnostics=diagnostics,
    )


def _recover_recent_document_referents(
    rag: RagService,
    *,
    previous_messages: list[Any],
    collection_id: str | None,
) -> list[str]:
    """Backfill referents from the nearest prior user comparison only."""

    resolver = getattr(rag, "resolve_document_mentions_for_query", None)
    if not callable(resolver):
        return []
    for message in reversed(previous_messages[-40:]):
        role, content = _message_role_and_content(message)
        if role != "user" or not _looks_like_multi_document_comparison(content):
            continue
        document_ids = resolver(
            query=content,
            collection_id=collection_id,
            limit=8,
        )
        if len(document_ids) >= 2:
            return list(document_ids)
    return []


def _looks_like_multi_document_comparison(query: str) -> bool:
    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        re.search(
            r"\b(?:so\s+sánh|đối\s+chiếu|khác\s+nhau|giống\s+nhau|versus|compare|"
            r"comparison|differences?|similarities?|against)\b",
            normalized,
        )
        or re.search(r"(?<![a-z0-9])vs\.?\s", normalized)
    )


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower(), str(message.get("content") or "")
    return (
        str(getattr(message, "role", "") or "").lower(),
        str(getattr(message, "content", "") or ""),
    )


def _document_labels_for_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    labels: list[str] = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        filename = str(document.get("filename") or "").strip()
        label = re.sub(r"\.[A-Za-z0-9]+$", "", filename).strip() or str(document_id)
        if label not in labels:
            labels.append(label)
    return labels


def _filenames_for_document_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    filenames: list[str] = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        filename = str(document.get("filename") or "").strip()
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _comparison_topic_from_filenames(filenames: list[str]) -> str | None:
    labels = [re.sub(r"\.[A-Za-z0-9]+$", "", item) for item in filenames if item]
    return " vs ".join(labels[:8]) if len(labels) >= 2 else None


def _resolve_query_document_focus(
    rag: RagService,
    *,
    rewrite: Any,
    collection_id: str | None,
    existing_focus: list[str],
) -> list[str]:
    explicit_targets = _explicit_document_target_entities(rewrite.original_query)
    has_document_marker = bool(
        re.search(
            r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
            rewrite.original_query,
            flags=re.IGNORECASE,
        )
    )
    has_explicit_filename = bool(
        re.search(r"\.(?:pdf|md|txt)(?!\w)", rewrite.original_query, re.I)
    )
    catalog_entities = (
        explicit_targets
        if explicit_targets
        else _query_named_entities(rewrite.original_query)
        if has_document_marker or has_explicit_filename
        else []
    )
    catalog_resolved = rag.resolve_explicit_document_ids_for_query(
        query=rewrite.original_query,
        entities=catalog_entities,
        collection_id=collection_id,
        compare=rewrite.answer_intent == "compare",
    )
    if catalog_resolved:
        return catalog_resolved[:2] if rewrite.answer_intent == "compare" else catalog_resolved[:1]

    if explicit_targets:
        entities = list(explicit_targets)
        # Do not let other names in a correction sentence pull scope back to the
        # old paper.  Resolve only the current-turn canonical target(s).
        resolution_query = " ".join(explicit_targets)
    else:
        entities = []
        if rewrite.current_topic:
            entities.append(str(rewrite.current_topic))
        for entity in rewrite.required_entities or []:
            if entity not in entities:
                entities.append(entity)
        resolution_query = "\n".join(
            part
            for part in [rewrite.original_query, rewrite.standalone_query, rewrite.current_topic]
            if part
        )

    if explicit_targets:
        entity_resolved = rag.resolve_document_ids_for_entities(
            entities=entities,
            collection_id=collection_id,
            query=resolution_query,
        )
    else:
        # Without an explicit "paper X" target, use only entities that
        # identify one document. A dataset shared by four papers (IEMOCAP)
        # must not pick whichever paper ranks first.
        entity_resolved = []
        for entity in entities:
            for document_id in _decisive_document_ids(rag, entity, collection_id=collection_id):
                if document_id not in entity_resolved:
                    entity_resolved.append(document_id)
    if entity_resolved:
        if rewrite.answer_intent == "compare":
            return entity_resolved[:2]
        primary = entity_resolved[:1]
        if existing_focus and rewrite.use_last_sources:
            overlap = [doc_id for doc_id in existing_focus if doc_id in entity_resolved]
            if overlap:
                return overlap[:1]
        return primary

    return existing_focus[:1] if existing_focus else []


def _text_source_document_ids(documents: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for document in documents:
        if document.get("figure_id"):
            continue
        document_id = document.get("document_id")
        if document_id:
            counts[str(document_id)] = counts.get(str(document_id), 0) + 1
    if not counts:
        return []
    return [sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]]

