"""Catalog-backed document-scope resolution for conversational RAG turns.

This module owns precedence between current-turn identities, plural anaphora and
sticky L1 state.  It deliberately does not decide answer wording or retrieval
facets; those stages may enrich a query but must not silently change this scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from app.services.conversation_state import (
    requests_plural_document_referents,
    resolve_plural_document_referent_ids,
)
from app.services.document_language import looks_like_document_comparison
from app.services.query_rewrite_service import _explicit_document_target_entities
from app.services.rag_service import (
    CatalogDocumentMention,
    CatalogMentionResolution,
    RagService,
)


@dataclass(frozen=True)
class DocumentScopeResolution:
    document_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    source: str = "none"
    must_cover_all: bool = False
    authoritative: bool = False
    mentions: tuple[CatalogDocumentMention, ...] = ()
    ambiguous_mentions: tuple[CatalogDocumentMention, ...] = ()
    collection_removed_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_ids": list(self.document_ids),
            "labels": list(self.labels),
            "source": self.source,
            "must_cover_all": self.must_cover_all,
            "authoritative": self.authoritative,
            "mentions": [_mention_dict(item) for item in self.mentions],
            "ambiguous_mentions": [
                _mention_dict(item) for item in self.ambiguous_mentions
            ],
            "collection_removed_ids": list(self.collection_removed_ids),
        }


def resolve_document_scope(
    rag: RagService,
    *,
    query: str,
    collection_id: str | None,
    working_state: Any,
    previous_messages: list[Any],
) -> DocumentScopeResolution:
    """Resolve one canonical scope before routing or query rewriting."""

    global_mentions = rag.resolve_catalog_mentions(query=query, collection_id=None)
    scoped_mentions = (
        rag.resolve_catalog_mentions(query=query, collection_id=collection_id)
        if collection_id
        else global_mentions
    )
    global_ids = list(global_mentions.document_ids)
    scoped_ids = list(scoped_mentions.document_ids)
    removed_ids = tuple(item for item in global_ids if item not in set(scoped_ids))
    explicit_targets = _explicit_document_target_entities(query)
    has_document_marker = bool(
        re.search(
            r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)(?!\w)",
            query,
            flags=re.IGNORECASE,
        )
    )
    has_explicit_filename = bool(re.search(r"\.(?:pdf|md|txt)(?!\w)", query, re.I))
    comparison_language = looks_like_document_comparison(query)
    global_joint_scope = _mentions_form_joint_scope(
        query,
        global_mentions.mentions,
    )
    joint_current_scope = _mentions_form_joint_scope(
        query,
        scoped_mentions.mentions,
    )
    active_ids = tuple(
        dict.fromkeys(
            str(item)
            for item in (getattr(working_state, "active_document_ids", None) or [])
            if item
        )
    )

    bounded_correction_ids = _bounded_context_correction_ids(
        rag,
        query=query,
        working_state=working_state,
        global_mentions=global_mentions,
    )
    if bounded_correction_ids:
        if collection_id:
            allowed_ids = set(rag.collection_document_ids(collection_id))
            excluded = tuple(
                document_id
                for document_id in bounded_correction_ids
                if document_id not in allowed_ids
            )
            if excluded:
                return DocumentScopeResolution(
                    source="collection_excluded",
                    authoritative=True,
                    collection_removed_ids=excluded,
                )
        return DocumentScopeResolution(
            document_ids=tuple(bounded_correction_ids),
            labels=tuple(_labels_for_ids(rag, bounded_correction_ids)),
            source="context_correction",
            must_cover_all=len(bounded_correction_ids) >= 2,
            authoritative=True,
            mentions=global_mentions.mentions,
        )

    correction_target = _correction_target_resolution(rag, query=query)
    if correction_target is not None:
        if correction_target.ambiguous_mentions:
            return DocumentScopeResolution(
                source="ambiguous_current_turn",
                authoritative=True,
                mentions=correction_target.mentions,
                ambiguous_mentions=correction_target.ambiguous_mentions,
            )
        correction_ids = list(correction_target.document_ids)
        if correction_ids:
            if collection_id:
                allowed_ids = set(rag.collection_document_ids(collection_id))
                excluded = tuple(
                    document_id
                    for document_id in correction_ids
                    if document_id not in allowed_ids
                )
                if excluded:
                    return DocumentScopeResolution(
                        source="collection_excluded",
                        authoritative=True,
                        mentions=correction_target.mentions,
                        collection_removed_ids=excluded,
                    )
            labels = _labels_for_ids(rag, correction_ids)
            return DocumentScopeResolution(
                document_ids=tuple(correction_ids),
                labels=tuple(labels),
                source="correction_current_turn",
                must_cover_all=len(correction_ids) >= 2,
                authoritative=True,
                mentions=correction_target.mentions,
            )

    plural_referent_ids = resolve_plural_document_referent_ids(
        working_state,
        query,
    )
    if len(plural_referent_ids) < 2 and requests_plural_document_referents(query):
        plural_referent_ids = _recover_recent_comparison_ids(
            rag,
            previous_messages=previous_messages,
            collection_id=collection_id,
        )
    if (
        len(plural_referent_ids) >= 2
        and not global_ids
        and global_mentions.ambiguous_mentions
        and _ambiguities_are_bounded_by_referents(
            global_mentions.ambiguous_mentions,
            plural_referent_ids,
        )
    ):
        if collection_id:
            allowed_ids = set(rag.collection_document_ids(collection_id))
            excluded = tuple(
                document_id
                for document_id in plural_referent_ids
                if document_id not in allowed_ids
            )
            if excluded:
                return DocumentScopeResolution(
                    source="collection_excluded",
                    authoritative=True,
                    collection_removed_ids=excluded,
                )
        return DocumentScopeResolution(
            document_ids=tuple(plural_referent_ids),
            labels=tuple(_labels_for_ids(rag, plural_referent_ids)),
            source="bounded_plural_referent",
            must_cover_all=True,
            authoritative=True,
            mentions=global_mentions.mentions,
        )

    contextual_joint_ids, contextual_mentions = _contextual_joint_document_ids(
        rag,
        query=query,
        working_state=working_state,
        catalog_mentions=scoped_mentions,
    )
    if len(contextual_joint_ids) >= 2:
        if collection_id:
            allowed_ids = set(rag.collection_document_ids(collection_id))
            excluded = tuple(
                document_id
                for document_id in contextual_joint_ids
                if document_id not in allowed_ids
            )
            if excluded:
                return DocumentScopeResolution(
                    source="collection_excluded",
                    authoritative=True,
                    mentions=contextual_mentions,
                    collection_removed_ids=excluded,
                )
        return DocumentScopeResolution(
            document_ids=tuple(contextual_joint_ids),
            labels=tuple(_labels_for_ids(rag, contextual_joint_ids)),
            source="contextual_joint_scope",
            must_cover_all=True,
            authoritative=True,
            mentions=contextual_mentions,
        )

    # A joint expression is one atomic obligation. Ambiguous or collection-
    # excluded operands invalidate the whole set instead of silently shrinking
    # it to the remaining uniquely resolved paper.
    if global_joint_scope and global_mentions.ambiguous_mentions:
        return DocumentScopeResolution(
            source="ambiguous_current_turn",
            authoritative=True,
            mentions=global_mentions.mentions,
            ambiguous_mentions=global_mentions.ambiguous_mentions,
            collection_removed_ids=removed_ids,
        )
    if global_joint_scope and removed_ids:
        return DocumentScopeResolution(
            source="collection_excluded",
            authoritative=True,
            mentions=global_mentions.mentions,
            ambiguous_mentions=global_mentions.ambiguous_mentions,
            collection_removed_ids=removed_ids,
        )
    if global_joint_scope and len(scoped_ids) >= 2:
        labels = _labels_for_ids(rag, scoped_ids)
        return DocumentScopeResolution(
            document_ids=tuple(scoped_ids),
            labels=tuple(labels),
            source="current_turn_mentions",
            must_cover_all=True,
            authoritative=True,
            mentions=scoped_mentions.mentions,
            ambiguous_mentions=scoped_mentions.ambiguous_mentions,
            collection_removed_ids=removed_ids,
        )

    # A model/paper name inside a deictic artifact reference is normally a row
    # or baseline in the already active source. Only explicit document syntax
    # is allowed to switch L1 in that grammar.
    if (
        active_ids
        and _references_active_artifact(query)
        and not has_document_marker
        and not has_explicit_filename
    ):
        return DocumentScopeResolution(
            document_ids=active_ids,
            labels=tuple(_labels_for_ids(rag, list(active_ids))),
            source="active_artifact_referent",
            must_cover_all=len(active_ids) >= 2,
            authoritative=True,
            mentions=scoped_mentions.mentions,
        )

    # Explicit source syntax also handles corrections by selecting the last
    # target unless the turn is an actual comparison.
    if explicit_targets or has_document_marker or has_explicit_filename:
        # Document markers (``paper``, ``bài``, ``2 cái``...) are optional
        # surface grammar, not identity rules.  Once the catalog has found two
        # distinct mentions connected as a set, preserve that entire ordered
        # set instead of letting the single-target correction parser collapse
        # it to its first match.
        if len(scoped_ids) >= 2 and joint_current_scope:
            labels = _labels_for_ids(rag, scoped_ids)
            return DocumentScopeResolution(
                document_ids=tuple(scoped_ids),
                labels=tuple(labels),
                source="current_turn_mentions",
                must_cover_all=True,
                authoritative=True,
                mentions=scoped_mentions.mentions,
                ambiguous_mentions=scoped_mentions.ambiguous_mentions,
                collection_removed_ids=removed_ids,
            )
        lookup_entities = explicit_targets or []
        explicit_ids = rag.resolve_explicit_document_ids_for_query(
            query=query,
            entities=lookup_entities,
            collection_id=collection_id,
            compare=comparison_language,
        )
        if explicit_ids:
            labels = _labels_for_ids(rag, explicit_ids)
            return DocumentScopeResolution(
                document_ids=tuple(explicit_ids),
                labels=tuple(labels),
                source="explicit_current_turn",
                must_cover_all=len(explicit_ids) >= 2,
                authoritative=True,
                mentions=scoped_mentions.mentions,
                ambiguous_mentions=scoped_mentions.ambiguous_mentions,
                collection_removed_ids=removed_ids,
            )
        if removed_ids:
            return DocumentScopeResolution(
                source="collection_excluded",
                authoritative=True,
                mentions=global_mentions.mentions,
                ambiguous_mentions=global_mentions.ambiguous_mentions,
                collection_removed_ids=removed_ids,
            )
        if global_mentions.ambiguous_mentions:
            return DocumentScopeResolution(
                source="ambiguous_current_turn",
                authoritative=True,
                mentions=global_mentions.mentions,
                ambiguous_mentions=global_mentions.ambiguous_mentions,
            )

    # Natural identity mentions use catalog provenance and span suppression.
    # Multiple names are a required set only when connected as a joint ask;
    # corrections without a comparison keep the last identity.
    if scoped_ids:
        joint_request = len(scoped_ids) >= 2 and joint_current_scope
        if joint_request:
            labels = _labels_for_ids(rag, scoped_ids)
            return DocumentScopeResolution(
                document_ids=tuple(scoped_ids),
                labels=tuple(labels),
                source="current_turn_mentions",
                must_cover_all=True,
                authoritative=True,
                mentions=scoped_mentions.mentions,
                ambiguous_mentions=scoped_mentions.ambiguous_mentions,
                collection_removed_ids=removed_ids,
            )

        chosen_ids = [scoped_ids[-1]] if len(scoped_ids) > 1 else scoped_ids
        labels = _labels_for_ids(rag, chosen_ids)
        return DocumentScopeResolution(
            document_ids=tuple(chosen_ids),
            labels=tuple(labels),
            source="current_turn_mentions",
            must_cover_all=False,
            authoritative=True,
            mentions=scoped_mentions.mentions,
            ambiguous_mentions=scoped_mentions.ambiguous_mentions,
            collection_removed_ids=removed_ids,
        )

    if removed_ids:
        return DocumentScopeResolution(
            source="collection_excluded",
            authoritative=True,
            mentions=global_mentions.mentions,
            ambiguous_mentions=global_mentions.ambiguous_mentions,
            collection_removed_ids=removed_ids,
        )
    if global_mentions.ambiguous_mentions:
        return DocumentScopeResolution(
            source="ambiguous_current_turn",
            authoritative=True,
            mentions=global_mentions.mentions,
            ambiguous_mentions=global_mentions.ambiguous_mentions,
        )

    # Durable referents win over active single-paper state for plural anaphora.
    if requests_plural_document_referents(query):
        referent_ids = plural_referent_ids
        if len(referent_ids) < 2:
            referent_ids = _recover_recent_comparison_ids(
                rag,
                previous_messages=previous_messages,
                collection_id=collection_id,
            )
        if len(referent_ids) >= 2:
            if collection_id:
                allowed_ids = set(rag.collection_document_ids(collection_id))
                excluded_referents = [
                    document_id
                    for document_id in referent_ids
                    if document_id not in allowed_ids
                ]
                if excluded_referents:
                    return DocumentScopeResolution(
                        source="collection_excluded",
                        authoritative=True,
                        collection_removed_ids=tuple(excluded_referents),
                    )
            labels = _labels_for_ids(rag, referent_ids)
            return DocumentScopeResolution(
                document_ids=tuple(referent_ids),
                labels=tuple(labels),
                source="plural_referent",
                must_cover_all=True,
                authoritative=True,
            )

    if active_ids:
        active_must_cover_all = (
            len(active_ids) >= 2
            and getattr(working_state, "last_answer_intent", None) == "compare"
        )
        if collection_id:
            allowed_ids = set(rag.collection_document_ids(collection_id))
            excluded_active = tuple(
                document_id for document_id in active_ids if document_id not in allowed_ids
            )
            if excluded_active and active_must_cover_all:
                return DocumentScopeResolution(
                    source="collection_excluded",
                    authoritative=True,
                    collection_removed_ids=excluded_active,
                )
            active_ids = tuple(
                document_id for document_id in active_ids if document_id in allowed_ids
            )
            if not active_ids:
                return DocumentScopeResolution()
        return DocumentScopeResolution(
            document_ids=active_ids,
            labels=tuple(_labels_for_ids(rag, list(active_ids))),
            source="active_focus",
            must_cover_all=active_must_cover_all,
            authoritative=False,
        )
    return DocumentScopeResolution()


_SINGULAR_DOCUMENT_ANAPHOR_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:paper|bài(?:\s+báo)?|document|tài\s+liệu|plan|kế\s+hoạch)\s+"
    r"(?:trước|trước\s+đó|vừa\s+nói|previous|prior|before|đó|này)"
    r"|(?:previous|prior|that|this)\s+(?:paper|document|plan)"
    r"|nó|it"
    r")(?!\w)",
    flags=re.IGNORECASE,
)
_CONTEXT_DESCRIPTOR_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "article",
        "bai",
        "document",
        "file",
        "for",
        "from",
        "in",
        "method",
        "model",
        "of",
        "on",
        "paper",
        "pdf",
        "plan",
        "project",
        "study",
        "the",
        "to",
        "with",
    }
)


def _contextual_joint_document_ids(
    rag: RagService,
    *,
    query: str,
    working_state: Any,
    catalog_mentions: CatalogMentionResolution,
) -> tuple[list[str], tuple[CatalogDocumentMention, ...]]:
    """Compose current identities with bounded conversational referents.

    Global aliases stay conservative. Only documents already present in L1 or
    its recent/referent threads may use shorter descriptors such as ``storage``
    in ``the storage plan``. This prevents generic words from becoming global
    document identities while allowing natural mixed expressions like
    ``it with Paper B``.
    """

    unique_catalog_ids = list(catalog_mentions.document_ids)
    if len(unique_catalog_ids) >= 2:
        return [], ()
    context_ids = _context_candidate_ids(working_state)
    if not context_ids:
        return [], ()
    contextual = list(
        _contextual_document_mentions(
            rag,
            query=query,
            candidate_document_ids=context_ids,
        )
    )
    active_ids = list(
        dict.fromkeys(
            str(item)
            for item in (getattr(working_state, "active_document_ids", None) or [])
            if item
        )
    )
    if len(unique_catalog_ids) == 1 and len(active_ids) == 1:
        match = _SINGULAR_DOCUMENT_ANAPHOR_RE.search(query)
        if match is not None and active_ids[0] != unique_catalog_ids[0]:
            start_word, end_word = _character_span_to_word_span(query, match.span())
            contextual.append(
                CatalogDocumentMention(
                    surface=match.group(0),
                    start_word=start_word,
                    end_word=end_word,
                    document_id=active_ids[0],
                    candidate_ids=(active_ids[0],),
                    alias_source="active_anaphor",
                    strength=4,
                )
            )

    combined = [
        *(
            mention
            for mention in catalog_mentions.mentions
            if mention.document_id is not None
        ),
        *contextual,
    ]
    combined.sort(key=lambda item: (item.start_word, -item.strength, -item.end_word))
    deduped: list[CatalogDocumentMention] = []
    seen_ids: set[str] = set()
    for mention in combined:
        if not mention.document_id or mention.document_id in seen_ids:
            continue
        seen_ids.add(mention.document_id)
        deduped.append(mention)
    if len(deduped) < 2 or not _mentions_form_joint_scope(query, tuple(deduped)):
        return [], tuple(deduped)

    # An unrelated unresolved alias still invalidates an atomic comparison.
    resolved_ids = {mention.document_id for mention in deduped if mention.document_id}
    for ambiguity in catalog_mentions.ambiguous_mentions:
        if not set(ambiguity.candidate_ids).issubset(resolved_ids):
            return [], tuple(deduped)
    return [str(mention.document_id) for mention in deduped], tuple(deduped)


def _context_candidate_ids(working_state: Any) -> list[str]:
    values: list[str] = []

    def extend(items: Any) -> None:
        for item in items or []:
            text = str(item or "").strip()
            if text and text not in values:
                values.append(text)

    extend(getattr(working_state, "referent_document_ids", None))
    extend(getattr(working_state, "active_document_ids", None))
    for thread in reversed(
        tuple(getattr(working_state, "recent_document_threads", None) or ())
    ):
        extend(getattr(thread, "document_ids", None))
    return values[:12]


def _contextual_document_mentions(
    rag: RagService,
    *,
    query: str,
    candidate_document_ids: list[str],
) -> tuple[CatalogDocumentMention, ...]:
    query_words = _normalized_context_words(query)
    if not query_words:
        return ()
    aliases: dict[tuple[str, ...], set[str]] = {}
    for document_id in candidate_document_ids:
        document = rag.get_document(document_id) or {}
        filename = re.sub(
            r"\.[A-Za-z0-9]+$",
            "",
            str(document.get("filename") or ""),
        )
        words = _normalized_context_words(filename)
        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                alias = tuple(words[start:end])
                if len(alias) == 1 and (
                    len(alias[0]) < 3 or alias[0] in _CONTEXT_DESCRIPTOR_STOPWORDS
                ):
                    continue
                if len(alias) >= 2 and len("".join(alias)) < 6:
                    continue
                aliases.setdefault(alias, set()).add(document_id)

    matches: list[CatalogDocumentMention] = []
    for alias, owners in aliases.items():
        if len(owners) != 1:
            continue
        position = _sequence_position(query_words, alias)
        if position < 0:
            continue
        document_id = next(iter(owners))
        matches.append(
            CatalogDocumentMention(
                surface=" ".join(query_words[position : position + len(alias)]),
                start_word=position,
                end_word=position + len(alias),
                document_id=document_id,
                candidate_ids=(document_id,),
                alias_source="context_filename",
                strength=4,
            )
        )

    # Prefer the longest descriptor for each document/span.
    matches.sort(
        key=lambda item: (
            item.start_word,
            -(item.end_word - item.start_word),
            str(item.document_id),
        )
    )
    selected: list[CatalogDocumentMention] = []
    seen_ids: set[str] = set()
    for mention in matches:
        if not mention.document_id or mention.document_id in seen_ids:
            continue
        seen_ids.add(mention.document_id)
        selected.append(mention)
    return tuple(selected)


def _normalized_context_words(value: str) -> list[str]:
    """Return accent-insensitive words without changing their sequence.

    Contextual descriptors are intentionally limited to already-grounded L1
    documents, but Vietnamese users may omit accents in a follow-up.  Using
    the same folded representation for the query and filenames keeps that
    conversational convenience local instead of promoting generic global
    aliases.
    """

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    folded = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", folded.casefold())


def _sequence_position(words: list[str], sequence: tuple[str, ...]) -> int:
    width = len(sequence)
    for index in range(len(words) - width + 1):
        if tuple(words[index : index + width]) == sequence:
            return index
    return -1


def _character_span_to_word_span(
    query: str,
    span: tuple[int, int],
) -> tuple[int, int]:
    word_spans = _query_word_spans(query)
    overlapping = [
        index
        for index, (start, end) in enumerate(word_spans)
        if start < span[1] and end > span[0]
    ]
    if not overlapping:
        return 0, 1
    return overlapping[0], overlapping[-1] + 1


def _ambiguities_are_bounded_by_referents(
    ambiguities: tuple[CatalogDocumentMention, ...],
    referent_ids: list[str],
) -> bool:
    referent_set = set(referent_ids)
    return bool(referent_set) and all(
        set(mention.candidate_ids) == referent_set for mention in ambiguities
    )


def _bounded_context_correction_ids(
    rag: RagService,
    *,
    query: str,
    working_state: Any,
    global_mentions: CatalogMentionResolution,
) -> list[str]:
    """Apply a rejection inside a previously grounded ambiguous candidate set."""

    if not _CORRECTION_LANGUAGE_RE.search(query):
        return []
    candidates = _context_candidate_ids(working_state)
    if len(candidates) < 2 or not global_mentions.ambiguous_mentions:
        return []
    candidate_set = set(candidates)
    bounded_ambiguities = [
        mention
        for mention in global_mentions.ambiguous_mentions
        if set(mention.candidate_ids).issubset(candidate_set)
    ]
    if not bounded_ambiguities:
        return []

    rejected: set[str] = set()
    for clause in _CORRECTION_CLAUSE_SPLIT_RE.split(str(query or "")):
        mentions = _contextual_document_mentions(
            rag,
            query=clause,
            candidate_document_ids=candidates,
        )
        if not mentions:
            continue
        negated = bool(
            _REJECTED_CLAUSE_RE.search(clause)
            or _NEGATION_DIVIDER_RE.search(clause)
        )
        if negated:
            rejected.update(
                mention.document_id
                for mention in mentions
                if mention.document_id
            )
    if not rejected:
        return []
    ambiguity_candidates = set().union(
        *(set(mention.candidate_ids) for mention in bounded_ambiguities)
    )
    remaining = [
        document_id
        for document_id in candidates
        if document_id in ambiguity_candidates and document_id not in rejected
    ]
    return remaining if len(remaining) == 1 else []


def _has_joint_document_connector(query: str) -> bool:
    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        re.search(r"\b(?:và|với|cùng|and|with|both)\b", normalized)
        or re.search(r"\bcả\s+(?:hai|2)\b", normalized)
    )


_CORRECTION_LANGUAGE_RE = re.compile(
    r"(?<!\w)(?:"
    r"ý\s+(?:tôi\s+)?là|y\s+(?:toi\s+)?la|i\s+mean|"
    r"không\s+phải|khong\s+phai|chứ\s+không\s+phải|chu\s+khong\s+phai|"
    r"thực\s+ra|thuc\s+ra|actually|instead|rather|over|replace|thay|đổi|doi|"
    r"sorry|correction|"
    r"đính\s+chính|dinh\s+chinh|sửa\s+lại|sua\s+lai|đừng|dung|"
    r"không\s+(?:lấy|đúng)|khong\s+(?:lay|dung)|"
    r"chỉ\s+là\s+ví\s+dụ|chi\s+la\s+vi\s+du|"
    r"mới\s+đúng|moi\s+dung|do\s+not|don't|is\s+wrong|"
    r"just\s+an?\s+example|(?<!\w)no(?!\w)|(?<!\w)not(?!\w)|"
    r"(?:^|\s)cơ(?:\s|$)"
    r")(?!\w)",
    flags=re.IGNORECASE,
)
_ADDITIVE_PAIR_RE = re.compile(
    r"(?<!\w)(?:not\s+only\b.+?\bbut\s+also|"
    r"không\s+chỉ\b.+?\bmà\s+còn|khong\s+chi\b.+?\bma\s+con)(?!\w)",
    flags=re.IGNORECASE | re.DOTALL,
)
_JOINT_BRIDGE_RE = re.compile(
    r"(?<!\w)(?:và|va|với|voi|cùng|cung|and|with|both|vs\.?|versus)(?!\w)|"
    r"[+&,;]",
    flags=re.IGNORECASE,
)
_PAIR_QUANTIFIER_RE = re.compile(
    r"(?<!\w)(?:2|hai|two|both|cả\s+hai|ca\s+hai)"
    r"(?:\s+(?:cái|cai|models?|methods?|documents?|files?|papers?|bài|bai))?"
    r"(?!\w)",
    flags=re.IGNORECASE,
)


def _mentions_form_joint_scope(
    query: str,
    mentions: tuple[CatalogDocumentMention, ...],
) -> bool:
    """Return whether distinct catalog mentions form one requested set.

    This is span/grammar based and deliberately independent of corpus names or
    the presence of words such as ``paper``.  A correction cue wins over loose
    list punctuation so ``A rồi, ý tôi là B`` remains a target replacement.
    """

    resolved = [
        mention
        for mention in mentions
        if mention.document_id or mention.ambiguous
    ]
    mention_keys = {
        ("document", mention.document_id)
        if mention.document_id
        else ("ambiguous", mention.candidate_ids)
        for mention in resolved
    }
    if len(mention_keys) < 2:
        return False
    if looks_like_document_comparison(query):
        return True
    if _ADDITIVE_PAIR_RE.search(query):
        return True
    if _CORRECTION_LANGUAGE_RE.search(query):
        return False

    word_spans = _query_word_spans(query)
    for left, right in zip(resolved, resolved[1:]):
        left_key = left.document_id or left.candidate_ids
        right_key = right.document_id or right.candidate_ids
        if left_key == right_key:
            continue
        left_end = (
            word_spans[left.end_word - 1][1]
            if 0 < left.end_word <= len(word_spans)
            else 0
        )
        right_start = (
            word_spans[right.start_word][0]
            if 0 <= right.start_word < len(word_spans)
            else len(query)
        )
        if _JOINT_BRIDGE_RE.search(query[left_end:right_start]):
            return True

    first_word = min(mention.start_word for mention in resolved)
    prefix_end = word_spans[first_word][0] if first_word < len(word_spans) else len(query)
    prefix = query[:prefix_end]
    return bool(_PAIR_QUANTIFIER_RE.search(prefix))


_CORRECTION_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[;,?!]|(?<!\w)(?:mà|ma|but)(?!\w))\s*",
    flags=re.IGNORECASE,
)
_CORRECTION_RESET_RE = re.compile(
    r"(?<!\w)(?:ý\s+(?:tôi\s+)?là|y\s+(?:toi\s+)?la|i\s+mean|"
    r"thực\s+ra|thuc\s+ra|actually|sorry|correction|"
    r"đính\s+chính|dinh\s+chinh|sửa\s+lại(?:\s+là)?|"
    r"sua\s+lai(?:\s+la)?|no)(?!\w)",
    flags=re.IGNORECASE,
)
_REPLACEMENT_DIVIDER_RE = re.compile(
    r"(?<!\w)(?:replace|thay|đổi|doi)\b.*?"
    r"(?<!\w)(?:with|bằng|bang|sang)(?!\w)",
    flags=re.IGNORECASE,
)
_PREFERENCE_DIVIDER_RE = re.compile(
    r"(?<!\w)(?:instead\s+of|rather\s+than|chứ\s+không\s+phải|"
    r"chu\s+khong\s+phai|thay\s+vì|thay\s+vi|over)(?!\w)",
    flags=re.IGNORECASE,
)
_NEGATION_DIVIDER_RE = re.compile(
    r"(?<!\w)(?:not|không\s+phải|khong\s+phai)(?!\w)",
    flags=re.IGNORECASE,
)
_REJECTED_CLAUSE_RE = re.compile(
    r"(?<!\w)(?:đừng|dung|do\s+not|don't|không\s+lấy|khong\s+lay)(?!\w)|"
    r"(?<!\w)(?:is\s+wrong|is\s+just\s+an?\s+example|wrong|"
    r"không\s+đúng|khong\s+dung|cũng\s+không|cung\s+khong|"
    r"chỉ\s+là\s+ví\s+dụ|chi\s+la\s+vi\s+du)(?!\w)",
    flags=re.IGNORECASE,
)


def _correction_target_resolution(
    rag: RagService,
    *,
    query: str,
) -> Any | None:
    """Resolve accepted identities by their role in a correction discourse.

    The grammar assigns catalog spans to selected/rejected sides of general
    replacement, preference, negation and discourse-reset operators. It never
    contains a paper name, table number, or corpus-specific mapping.
    """

    text = str(query or "")
    if _ADDITIVE_PAIR_RE.search(text) or not _CORRECTION_LANGUAGE_RE.search(text):
        return None

    selected: list[CatalogDocumentMention] = []
    for clause in _CORRECTION_CLAUSE_SPLIT_RE.split(text):
        clause = clause.strip()
        if not clause:
            continue
        reset = bool(_CORRECTION_RESET_RE.search(clause))
        if reset:
            selected.clear()
        resolution = rag.resolve_catalog_mentions(query=clause, collection_id=None)
        mentions = list(resolution.mentions)
        if not mentions:
            continue

        word_spans = _query_word_spans(clause)

        def starts(mention: CatalogDocumentMention) -> int:
            if 0 <= mention.start_word < len(word_spans):
                return word_spans[mention.start_word][0]
            return len(clause)

        def ends(mention: CatalogDocumentMention) -> int:
            if 0 < mention.end_word <= len(word_spans):
                return word_spans[mention.end_word - 1][1]
            return 0

        replacement = _REPLACEMENT_DIVIDER_RE.search(clause)
        if replacement is not None:
            accepted = [mention for mention in mentions if starts(mention) >= replacement.end()]
        else:
            preference = _PREFERENCE_DIVIDER_RE.search(clause)
            if preference is not None:
                accepted = [mention for mention in mentions if ends(mention) <= preference.start()]
            elif _REJECTED_CLAUSE_RE.search(clause):
                accepted = []
            else:
                negation = _NEGATION_DIVIDER_RE.search(clause)
                if negation is not None:
                    accepted = [mention for mention in mentions if ends(mention) <= negation.start()]
                else:
                    accepted = mentions

        for mention in accepted:
            key = mention.document_id or mention.candidate_ids
            selected = [
                existing
                for existing in selected
                if (existing.document_id or existing.candidate_ids) != key
            ]
            selected.append(mention)

    if not selected:
        return None
    return CatalogMentionResolution(mentions=tuple(selected))


_ACTIVE_ARTIFACT_REFERENCE_RE = re.compile(
    r"(?:"
    r"(?<!\w)(?:bảng|table|figure|fig\.?|hình|hinh)"
    r"(?:\s*(?:số\s*)?#?\s*\d+)?\s*"
    r"(?:này|đó|kia|trên|vừa\s+rồi|lúc\s+nãy|this|that|above|previous|last)"
    r"|(?<!\w)(?:this|that|the\s+previous|the\s+last)\s+"
    r"(?:table|figure|fig\.?)"
    r")",
    flags=re.IGNORECASE,
)


def _references_active_artifact(query: str) -> bool:
    return bool(_ACTIVE_ARTIFACT_REFERENCE_RE.search(str(query or "")))


def _query_word_spans(query: str) -> list[tuple[int, int]]:
    # Catalog mention offsets count normalized alphanumeric words.  Unicode
    # decomposition preserves their order/count, while this mapping retains
    # original character offsets so punctuation between mentions is visible.
    spans = [
        match.span()
        for match in re.finditer(r"[0-9A-Za-zÀ-ỹ]+", str(query or ""))
    ]
    if spans:
        return spans
    folded = unicodedata.normalize("NFKD", str(query or ""))
    return [match.span() for match in re.finditer(r"[a-z0-9]+", folded.casefold())]


def _recover_recent_comparison_ids(
    rag: RagService,
    *,
    previous_messages: list[Any],
    collection_id: str | None,
) -> list[str]:
    for message in reversed(previous_messages[-40:]):
        role, content = _message_role_and_content(message)
        if role != "user" or not looks_like_document_comparison(content):
            continue
        ids = rag.resolve_document_mentions_for_query(
            query=content,
            collection_id=collection_id,
            limit=8,
        )
        if len(ids) >= 2:
            return ids
    return []


def _labels_for_ids(rag: RagService, document_ids: list[str]) -> list[str]:
    labels: list[str] = []
    for document_id in document_ids:
        document = rag.get_document(document_id) or {}
        filename = str(document.get("filename") or "").strip()
        label = re.sub(r"\.[A-Za-z0-9]+$", "", filename).strip() or str(document_id)
        if label not in labels:
            labels.append(label)
    return labels


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role") or "").lower(), str(message.get("content") or "")
    return (
        str(getattr(message, "role", "") or "").lower(),
        str(getattr(message, "content", "") or ""),
    )


def _mention_dict(mention: CatalogDocumentMention) -> dict[str, Any]:
    return {
        "surface": mention.surface,
        "start_word": mention.start_word,
        "end_word": mention.end_word,
        "document_id": mention.document_id,
        "candidate_ids": list(mention.candidate_ids),
        "alias_source": mention.alias_source,
        "strength": mention.strength,
        "ambiguous": mention.ambiguous,
    }
