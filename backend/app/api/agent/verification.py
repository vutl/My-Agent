from collections.abc import AsyncIterator, Callable
import asyncio
import json
from pathlib import Path
import re
import time
import traceback
from typing import Annotated, Any
from app.core.events import sse_event
from app.services.agent_run_store import AgentRunStore
from app.services.agent_service import AgentService, AnswerStreamChunk
from app.services.answer_verifier import (
    UnsourcedNumber,
    answer_unit_context,
    apply_corrections,
    keep_sourced_removals,
    miscited_units,
    build_verify_prompt,
    find_unsourced_numbers,
    MAX_VERIFY_UNITS,
    flagged_answer_units,
    numeric_answer_units,
    parse_corrections,
)
from app.services.evidence_validator import (
    AnswerClaimValidationResult,
    validate_answer_claims,
    validate_retrieval_evidence,
)
from app.services.document_language import looks_like_document_comparison
from app.services.debug_trace_service import DebugTraceRecorder
from .common import _elapsed_ms
from .common import _ollama_metrics
from .scope import _resolve_answer_document_ids
from .scope import _scope_documents_to_focus


async def _buffer_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    min_chars: int = 64,
    max_wait_ms: int = 55,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    buffer: list[str] = []
    buffered_chars = 0
    last_flush = time.perf_counter()

    async for chunk in chunks:
        if chunk.done:
            if buffer:
                yield "delta", {"delta": "".join(buffer)}
            yield "finished", {"chunk": chunk}
            return

        if not chunk.content:
            continue

        buffer.append(chunk.content)
        buffered_chars += len(chunk.content)
        elapsed_ms = (time.perf_counter() - last_flush) * 1000
        if buffered_chars >= min_chars or elapsed_ms >= max_wait_ms:
            yield "delta", {"delta": "".join(buffer)}
            buffer.clear()
            buffered_chars = 0
            last_flush = time.perf_counter()


def _validated_grounded_block(
    block: str,
    *,
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> tuple[str, list[Any], bool]:
    """Validate one complete Markdown paragraph before it becomes visible."""

    validation = validate_answer_claims(
        answer=block,
        documents=documents,
        focus_document_ids=focus_document_ids,
    )
    if validation.valid:
        return block, [validation], False
    sanitized = _remove_unsupported_claim_lines(block, validation.to_dict())
    if not sanitized.strip():
        return "", [validation], True
    sanitized_validation = validate_answer_claims(
        answer=sanitized,
        documents=documents,
        focus_document_ids=focus_document_ids,
    )
    if not sanitized_validation.valid:
        return "", [validation, sanitized_validation], True
    if block.endswith("\n\n"):
        sanitized = f"{sanitized.rstrip()}\n\n"
    return sanitized, [validation, sanitized_validation], True


async def _stream_validated_paper_sections(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    documents: list[dict[str, Any]],
    ordered_document_ids: list[str],
    document_identity_resolver: Callable[[str], list[str]] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Release each canonical paper section only after local validation.

    The delimiter is an internal transport contract.  It is never forwarded to
    the desktop. Malformed/missing sections fail closed to a deterministic,
    paper-named insufficiency block instead of leaking an unchecked draft.
    """

    ordered_ids = list(dict.fromkeys(ordered_document_ids))
    pending = ""
    next_index = 0
    validations: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    final_chunk: AnswerStreamChunk | None = None
    synthesis_emitted = False

    async def release_ready() -> AsyncIterator[tuple[str, dict[str, Any]]]:
        nonlocal pending, next_index, synthesis_emitted
        while next_index < len(ordered_ids):
            document_id = ordered_ids[next_index]
            opener = re.compile(
                rf"<paper\s+document_id\s*=\s*['\"]{re.escape(document_id)}['\"]\s*>",
                flags=re.IGNORECASE,
            )
            match = opener.search(pending)
            if match is None:
                return
            close_at = pending.lower().find("</paper>", match.end())
            if close_at < 0:
                return
            raw_section = pending[match.end() : close_at].strip()
            pending = pending[close_at + len("</paper>") :]
            scoped_documents = _scope_documents_to_focus(documents, [document_id])
            safe_section, diagnostics = _validated_paper_section(
                raw_section,
                document_id=document_id,
                documents=scoped_documents,
            )
            validations.append(diagnostics)
            partitions.append(
                {"document_id": document_id, "text": safe_section}
            )
            yield "paper_validated", diagnostics
            yield "delta", {"delta": f"{safe_section.rstrip()}\n\n"}
            next_index += 1

        if next_index == len(ordered_ids) and not synthesis_emitted:
            opener = re.search(r"<synthesis\s*>", pending, flags=re.IGNORECASE)
            if opener is None:
                return
            close_at = pending.lower().find("</synthesis>", opener.end())
            if close_at < 0:
                return
            raw_synthesis = pending[opener.end() : close_at].strip()
            pending = pending[close_at + len("</synthesis>") :]
            if raw_synthesis:
                safe, diagnostics = _validated_synthesis_section(
                    raw_synthesis,
                    documents=documents,
                    ordered_document_ids=ordered_ids,
                )
                if safe:
                    partitions.append({"document_id": None, "text": safe})
                    yield "delta", {"delta": f"{safe.rstrip()}\n"}
                validations.append(diagnostics)
            synthesis_emitted = True

    async for chunk in chunks:
        if chunk.content:
            pending += chunk.content
        async for event in release_ready():
            yield event
        if chunk.done:
            final_chunk = chunk
            break

    # A provider can occasionally obey the requested Markdown organization
    # while dropping the internal XML transport delimiters. Recover only when
    # nothing has been released yet and catalog resolution proves that one
    # structural heading maps uniquely to every requested document in order.
    # This is a presentation-contract repair, not an evidence shortcut: every
    # recovered section still passes the same document-local claim validator.
    if next_index == 0 and not partitions and pending.strip():
        recovered = _recover_untagged_paper_sections(
            pending,
            ordered_document_ids=ordered_ids,
            document_identity_resolver=document_identity_resolver,
        )
        if recovered is not None:
            recovered_sections, recovered_synthesis = recovered
            pending = ""
            for document_id, raw_section in recovered_sections:
                scoped_documents = _scope_documents_to_focus(documents, [document_id])
                safe_section, diagnostics = _validated_paper_section(
                    raw_section,
                    document_id=document_id,
                    documents=scoped_documents,
                )
                diagnostics["contract_recovered"] = True
                validations.append(diagnostics)
                partitions.append({"document_id": document_id, "text": safe_section})
                yield "paper_validated", diagnostics
                yield "delta", {"delta": f"{safe_section.rstrip()}\n\n"}
                next_index += 1
            if recovered_synthesis:
                safe_synthesis, diagnostics = _validated_synthesis_section(
                    recovered_synthesis,
                    documents=documents,
                    ordered_document_ids=ordered_ids,
                )
                diagnostics["contract_recovered"] = True
                validations.append(diagnostics)
                if safe_synthesis:
                    partitions.append({"document_id": None, "text": safe_synthesis})
                    yield "delta", {"delta": f"{safe_synthesis.rstrip()}\n"}
                synthesis_emitted = True

    # Every missing or malformed section becomes an explicit paper-specific
    # insufficiency statement. This preserves must-cover-all atomically without
    # discarding earlier sections that already passed validation.
    while next_index < len(ordered_ids):
        document_id = ordered_ids[next_index]
        fallback = _paper_section_insufficiency(document_id, documents)
        diagnostics = {
            "document_id": document_id,
            "section": "paper",
            "valid": False,
            "reason": "missing_or_malformed_paper_section",
            "fallback_used": True,
        }
        validations.append(diagnostics)
        partitions.append({"document_id": document_id, "text": fallback})
        yield "paper_validated", diagnostics
        yield "delta", {"delta": f"{fallback}\n\n"}
        next_index += 1

    yield "finished", {
        "chunk": final_chunk
        or AnswerStreamChunk(content="", done=True, finish_reason="stream_closed"),
        "validations": validations,
        "partitions": partitions,
    }


_SYNTHESIS_HEADING_RE = re.compile(
    r"(?<!\w)(?:synthesis|overall|conclusion|takeaways?|assessment|"
    r"tổng\s+hợp|tong\s+hop|kết\s+luận|ket\s+luan|tóm\s+lại|tom\s+lai|"
    r"nhận\s+xét|nhan\s+xet)(?!\w)",
    flags=re.IGNORECASE,
)


def _recover_untagged_paper_sections(
    text: str,
    *,
    ordered_document_ids: list[str],
    document_identity_resolver: Callable[[str], list[str]] | None,
) -> tuple[list[tuple[str, str]], str] | None:
    """Recover a fully ordered multi-paper draft from structural headings.

    Recovery is fail-closed: all requested documents must have exactly one
    unambiguous catalog-resolved heading, in the requested order. Bare prose
    and partial/ambiguous drafts are never partitioned heuristically.
    """

    if document_identity_resolver is None or not text.strip():
        return None
    ordered_ids = list(dict.fromkeys(ordered_document_ids))
    if len(ordered_ids) < 2:
        return None
    allowed = set(ordered_ids)
    headings: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        title = ""
        if match := re.match(r"^#{1,6}\s+(.+?)\s*#*$", stripped):
            title = match.group(1).strip()
        elif match := re.match(
            r"^(?:\d+[.)]\s*)?\*\*(.+?)\*\*\s*:?[ ]*$",
            stripped,
        ):
            title = match.group(1).strip()
        elif len(stripped) <= 100 and stripped.endswith(":"):
            title = stripped[:-1].strip()
        if title:
            try:
                resolved = list(dict.fromkeys(document_identity_resolver(title) or []))
            except Exception:
                return None
            headings.append(
                {
                    "start": offset,
                    "body_start": offset + len(line),
                    "title": title,
                    "document_ids": [item for item in resolved if item in allowed],
                }
            )
        offset += len(line)

    selected: list[dict[str, Any]] = []
    candidate_index = 0
    for expected_id in ordered_ids:
        found: dict[str, Any] | None = None
        while candidate_index < len(headings):
            candidate = headings[candidate_index]
            candidate_index += 1
            resolved_ids = candidate["document_ids"]
            if len(resolved_ids) != 1:
                continue
            if resolved_ids[0] != expected_id:
                return None
            found = candidate
            break
        if found is None:
            return None
        selected.append(found)

    synthesis_heading: dict[str, Any] | None = None
    for candidate in headings[candidate_index:]:
        if candidate["start"] <= selected[-1]["start"]:
            continue
        title = str(candidate["title"])
        if looks_like_document_comparison(title) or _SYNTHESIS_HEADING_RE.search(title):
            synthesis_heading = candidate
            break

    recovered_sections: list[tuple[str, str]] = []
    for index, (document_id, heading) in enumerate(zip(ordered_ids, selected)):
        if index + 1 < len(selected):
            end = selected[index + 1]["start"]
        elif synthesis_heading is not None:
            end = synthesis_heading["start"]
        else:
            end = len(text)
        body = text[heading["body_start"] : end].strip()
        if not body:
            return None
        recovered_sections.append((document_id, body))

    synthesis = ""
    if synthesis_heading is not None:
        synthesis = text[synthesis_heading["body_start"] :].strip()
    return recovered_sections, synthesis


def _validated_synthesis_section(
    section: str,
    *,
    documents: list[dict[str, Any]],
    ordered_document_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    validation = validate_answer_claims(
        answer=section,
        documents=documents,
        focus_document_ids=ordered_document_ids,
        answer_document_ids=ordered_document_ids,
    )
    safe = section.strip()
    fallback_used = False
    safe_validation = validation
    if not validation.valid:
        safe = _remove_unsupported_claim_lines(section, validation.to_dict()).strip()
        fallback_used = safe != section.strip()
        if safe:
            safe_validation = validate_answer_claims(
                answer=safe,
                documents=documents,
                focus_document_ids=ordered_document_ids,
                answer_document_ids=ordered_document_ids,
            )
            if not safe_validation.valid:
                safe = ""
    return safe, {
        "document_id": None,
        "section": "synthesis",
        "valid": safe_validation.valid if safe else False,
        "reason": safe_validation.reason if safe else validation.reason,
        "fallback_used": fallback_used,
        **_validation_debug_summary(validation),
    }


def _validate_partitioned_paper_answer(
    *,
    partitions: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    ordered_document_ids: list[str],
) -> AnswerClaimValidationResult:
    """Validate streamed sections using their known document ownership.

    Re-parsing the concatenated Markdown can leak the last paper heading into a
    cross-paper synthesis table. The transport parser already knows the exact
    partition ownership, so retain that structure through final validation
    instead of guessing it again from presentation text.
    """

    ordered_ids = list(dict.fromkeys(ordered_document_ids))
    checked = []
    supported = []
    unsupported = []
    foreign: set[str] = set()
    unparsed: list[dict[str, Any]] = []
    covered: set[str] = set()
    for partition in partitions:
        document_id = str(partition.get("document_id") or "").strip()
        text = str(partition.get("text") or "").strip()
        if not text:
            continue
        if document_id:
            scoped_documents = _scope_documents_to_focus(documents, [document_id])
            validation = validate_answer_claims(
                answer=text,
                documents=scoped_documents,
                focus_document_ids=[document_id],
                answer_document_ids=[document_id],
            )
            if validation.valid:
                covered.add(document_id)
        else:
            validation = validate_answer_claims(
                answer=text,
                documents=documents,
                focus_document_ids=ordered_ids,
                answer_document_ids=ordered_ids,
            )
        checked.extend(validation.checked_claims)
        supported.extend(validation.supported_claims)
        unsupported.extend(validation.unsupported_claims)
        foreign.update(validation.foreign_document_ids)
        unparsed.extend(validation.unparsed_signals)

    missing = [document_id for document_id in ordered_ids if document_id not in covered]
    if unsupported:
        reason = "unsupported_metric_values"
    elif unparsed:
        reason = "unparsed_metric_values"
    elif missing:
        reason = "missing_answer_documents"
    elif checked:
        reason = "metric_values_supported"
    else:
        reason = "no_metric_value_claims"
    valid = not unsupported and not unparsed and not missing
    return AnswerClaimValidationResult(
        valid=valid,
        retry_required=not valid,
        reason=reason,
        checked_claims=checked,
        supported_claims=supported,
        unsupported_claims=unsupported,
        foreign_document_ids=sorted(foreign),
        unparsed_signals=unparsed,
        covered_document_ids=[document_id for document_id in ordered_ids if document_id in covered],
        missing_document_ids=missing,
    )


def _validated_paper_section(
    section: str,
    *,
    document_id: str,
    documents: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not section.strip() or not documents:
        return _paper_section_insufficiency(document_id, documents), {
            "document_id": document_id,
            "section": "paper",
            "valid": False,
            "reason": "empty_or_missing_evidence",
            "fallback_used": True,
        }
    validation = validate_answer_claims(
        answer=section,
        documents=documents,
        focus_document_ids=[document_id],
        answer_document_ids=[document_id],
    )
    safe = section.strip()
    fallback_used = False
    if not validation.valid:
        safe = _remove_unsupported_claim_lines(safe, validation.to_dict()).strip()
        sanitized = validate_answer_claims(
            answer=safe,
            documents=documents,
            focus_document_ids=[document_id],
            answer_document_ids=[document_id],
        ) if safe else validation
        if not safe or not sanitized.valid:
            safe = _paper_section_insufficiency(document_id, documents)
        fallback_used = True
    heading = _paper_section_heading(document_id, documents)
    if heading.lower() not in safe.lower():
        safe = f"**{heading}**\n\n{safe}"
    return safe, {
        "document_id": document_id,
        "section": "paper",
        "valid": validation.valid,
        "reason": validation.reason,
        "fallback_used": fallback_used,
        **_validation_debug_summary(validation),
    }


def _validation_debug_summary(
    validation: AnswerClaimValidationResult,
) -> dict[str, Any]:
    """Expose bounded claim diagnostics without persisting a raw model draft."""

    return {
        "checked_claim_count": len(validation.checked_claims),
        "unsupported_claim_count": len(validation.unsupported_claims),
        "unsupported_claims": [
            claim.to_dict() for claim in validation.unsupported_claims[:12]
        ],
        "unparsed_signal_count": len(validation.unparsed_signals),
        "unparsed_signals": validation.unparsed_signals[:12],
    }


def _paper_section_heading(document_id: str, documents: list[dict[str, Any]]) -> str:
    for document in documents:
        if str(document.get("document_id") or "") == document_id:
            return str(document.get("filename") or document.get("title") or document_id)
    return document_id


def _paper_section_insufficiency(document_id: str, documents: list[dict[str, Any]]) -> str:
    heading = _paper_section_heading(document_id, documents)
    return (
        f"**{heading}**\n\n"
        "Mình chưa có đủ canonical evidence hợp lệ cho phần này, nên không suy đoán thêm."
    )


def _is_number_free_block(block: str) -> bool:
    """A block with no stand-alone number and no table row cannot carry a
    numeric claim. Digits inside identifiers (miF1, wav2vec2) are names."""
    text = str(block or "")
    return (
        bool(text.strip())
        and not re.search(r"(?<![A-Za-z_\-])\d", text)
        and text.count("|") < 2
    )


_MISSING_INFORMATION_RE = re.compile(
    r"(?:không|chưa)\s+(?:thấy|nêu|ghi|chứa|đề\s*cập|báo\s*cáo|công\s*bố|cung\s*cấp|nói\s*rõ|mô\s*tả|"
    r"xác\s*nhận\s*được|đọc\s*được|lấy\s*được|"
    r"trích\s*xuất\s*được|có\s+(?:thông\s*tin|số\s*liệu|phương\s*trình|công\s*thức|chi\s*tiết))"
    r"|(?:khong|chua)\s+(?:thay|neu|ghi|chua|de\s*cap)"
    r"|\b(?:does\s+not|doesn't|do\s+not|don't|did\s+not)\s+(?:include|contain|state|report|mention|specify|show|provide|list|disclose|give)"
    r"|\bnot\s+(?:stated|reported|mentioned|specified|included|provided|available|shown|found|visible)\b"
    r"|\b(?:cut\s+off|truncated|incomplete\s+(?:table|excerpt))\b"
    r"|\b(?:do|does|did)\s+not\s+(?:expose|show|cover|give)\b|\bmissing\s+from\b"
    r"|\bcan(?:not|'t|’t)\s+(?:verify|confirm|determine|find)\b"
    r"|chưa\s+(?:xuất\s*hiện|có\s+số\s*liệu|thể\s+xác\s*(?:nhận|định))"
    r"|bị\s+cắt|không\s+hiển\s+thị\s+(?:đủ|đầy\s+đủ)|chưa\s+đầy\s+đủ|thiếu\s+(?:các\s+)?(?:dòng|hàng)",
    re.IGNORECASE,
)


_MORE_MISSING_INFORMATION_RE = re.compile(
    r"\b(?:only|just)\s+(?:the\s+)?(?:table\s+)?(?:header|heading|caption|title)s?\b"
    r"|\bI(?:'d| would)\s+need\b|\bwould\s+need\s+the\b"
    r"|\b(?:can(?:not|'t)|unable\s+to)\s+(?:reliably\s+)?(?:give|quote|tell|read|provide|state|compare)\b"
    r"|\bnot\s+(?:fully\s+)?(?:visible|readable|legible|covered|included|present)\b"
    r"|\b(?:is|are)\s+(?:not\s+)?(?:missing|absent)\b"
    r"|không\s+(?:đọc|thấy)\s+được|chỉ\s+có\s+(?:tiêu\s+đề|phần\s+đầu)|cần\s+(?:dòng|hàng|bảng)\b",
    re.IGNORECASE,
)


def _answer_reports_missing_information(answer: str) -> bool:
    """The answer says the evidence it saw lacks what was asked."""
    # Models write typographic apostrophes ("don’t show"); match them too.
    text = str(answer or "").replace("\u2019", "'").replace("\u2018", "'")
    return bool(_MISSING_INFORMATION_RE.search(text) or _MORE_MISSING_INFORMATION_RE.search(text))


NEED_CONTEXT_MARKER = "⟦need-more-context⟧"


def _split_marker_tail(text: str) -> tuple[str, str]:
    """(text safe to show, tail that might start the marker)."""
    text = text.replace(NEED_CONTEXT_MARKER, "")
    for size in range(min(len(NEED_CONTEXT_MARKER) - 1, len(text)), 0, -1):
        if NEED_CONTEXT_MARKER.startswith(text[-size:]):
            return text[:-size], text[-size:]
    return text, ""


def _verification_sources(
    documents: list[dict[str, Any]], conversation_context: str,
) -> list[dict[str, Any]]:
    """What the answer saw: this turn's excerpts plus the conversation so far.

    A follow-up ("what did the lexical-only model get?") is often answered
    from a table shown in an earlier turn; verifying only this turn's
    excerpts deletes those correct values.
    """
    context = str(conversation_context or "").strip()
    if not context:
        return documents
    return [*documents, {"filename": "earlier turns of this conversation", "content": context[-12_000:]}]


async def _stream_then_verify(
    stream: AsyncIterator[AnswerStreamChunk],
    *,
    service: AgentService,
    run_store: AgentRunStore,
    request: Any,
    run_id: str,
    conversation_id: str,
    task: str,
    validation_documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
    require_all_focus_documents: bool,
    document_identity_resolver: Callable[[str], list[str]] | None,
    assistant_chunks: list[str],
    generation_started_at: float,
    verify_scope: str = "numeric",
    conversation_context: str = "",
    answer_signals: dict[str, Any] | None = None,
    cited_documents: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Stream like a chat product, then repair only suspicious numeric lines.

    The draft is shown as it is generated. Afterwards the metric validator and
    the number-sourcing check only *flag* lines; one model call decides for
    each flagged line whether to keep, rewrite (from the sources) or remove it.
    A changed answer replaces the shown draft through ``message.reset``. No
    full regeneration, no "not enough evidence" replacement.
    """
    parts: list[str] = []
    pending = ""
    final_chunk: AnswerStreamChunk | None = None
    first_delta_sent = False
    async for chunk in stream:
        if chunk.content:
            if not first_delta_sent:
                first_delta_sent = True
                yield sse_event(
                    "timing",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "stage": "first_validated_token",
                        "ms": _elapsed_ms(generation_started_at),
                        "streaming_mode": "post_verify",
                    },
                )
            # Hold back a marker-sized tail so the model's private
            # "need more context" marker never reaches the screen.
            pending += chunk.content
            emit, pending = _split_marker_tail(pending)
            if emit:
                parts.append(emit)
                assistant_chunks.append(emit)
                yield sse_event("message.delta", {"delta": emit})
        if chunk.done:
            final_chunk = chunk
            break
    if NEED_CONTEXT_MARKER in pending or NEED_CONTEXT_MARKER in "".join(parts) + pending:
        if answer_signals is not None:
            answer_signals["needs_more_context"] = True
    tail = pending.replace(NEED_CONTEXT_MARKER, "").rstrip()
    if tail:
        parts.append(tail)
        assistant_chunks.append(tail)
        yield sse_event("message.delta", {"delta": tail})
    answer = "".join(parts)

    legacy = validate_answer_claims(
        answer=answer,
        documents=validation_documents,
        focus_document_ids=focus_document_ids,
        require_all_focus_documents=require_all_focus_documents,
        answer_document_ids=_resolve_answer_document_ids(document_identity_resolver, answer),
        question=task,
    )
    sourcing = find_unsourced_numbers(answer, validation_documents)
    flagged_values = [claim.value for claim in legacy.unsupported_claims] + [
        item.value for item in sourcing.flagged
    ]
    # Lines whose numbers are absent from the very sources they cite go first.
    miscited = miscited_units(answer, cited_documents or [])
    unit_notes = dict(miscited)
    units = [unit for unit, _note in miscited]
    units += [unit for unit in flagged_answer_units(answer, flagged_values) if unit not in units]
    if verify_scope == "numeric":
        # Heuristics miss misattributed numbers that do occur in the sources
        # (other period, row or condition); the model checks every numeric line.
        units += [unit for unit in numeric_answer_units(answer) if unit not in units]
    units = units[:MAX_VERIFY_UNITS]
    corrections: list[dict[str, Any]] = []
    verify_error: str | None = None
    changed = False
    if units:
        try:
            completion = await asyncio.wait_for(
                service.client.chat(
                    model=request.model or service.default_model,
                    temperature=0.0,
                    num_predict=1_600,
                    messages=[{
                        "role": "user",
                        "content": build_verify_prompt(
                            [
                                UnsourcedNumber(
                                    value="", sentence=unit,
                                    context="; ".join(part for part in (
                                        answer_unit_context(answer, unit), unit_notes.get(unit, ""),
                                    ) if part),
                                )
                                for unit in units
                            ],
                            _verification_sources(validation_documents, conversation_context),
                            question=task,
                        ),
                    }],
                ),
                timeout=_POST_VERIFY_TIMEOUT_SECONDS,
            )
            corrections = keep_sourced_removals(
                parse_corrections(getattr(completion, "message", "") or ""),
                _verification_sources(validation_documents, conversation_context),
            )
            corrected = apply_corrections(answer, corrections)
            if corrected.strip() and corrected != answer:
                answer = corrected
                changed = True
        except Exception as exc:  # noqa: BLE001 - verification never blocks an answer
            frames = traceback.extract_tb(exc.__traceback__)
            where = f" @ {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
            verify_error = f"{type(exc).__name__}: {exc}{where}"[:300]
    if changed:
        assistant_chunks.clear()
        yield sse_event(
            "message.reset",
            {"run_id": run_id, "conversation_id": conversation_id, "reason": "post_verification"},
        )
        for offset in range(0, len(answer), 220):
            piece = answer[offset : offset + 220]
            assistant_chunks.append(piece)
            yield sse_event("message.delta", {"delta": piece})

    run_store.record_tool_call(
        run_id=run_id,
        tool_name="answer_verification",
        input_payload={"flagged_units": units[:12]},
        output_payload={
            "mode": "post",
            "scope": verify_scope,
            "legacy_valid": legacy.valid,
            "legacy_reason": legacy.reason,
            "unsourced": [item.value for item in sourcing.flagged][:12],
            "miscited": [note for _unit, note in miscited][:12],
            "corrections": corrections[:12],
            "changed": changed,
            "error": verify_error,
        },
    )
    yield sse_event(
        "answer.evidence.validated",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            **legacy.to_dict(),
            "attempts": 1,
            "fallback_used": False,
            "streaming_mode": "post_verify",
            "post_verification": {"flagged": len(units), "changed": changed, "error": verify_error},
        },
    )
    final_chunk = final_chunk or AnswerStreamChunk(content="", done=True, finish_reason="stream_closed")
    metrics = _ollama_metrics(final_chunk.metadata)
    yield sse_event(
        "message.finished",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "finish_reason": final_chunk.finish_reason,
            "eval_count": final_chunk.metadata.get("eval_count") if final_chunk.metadata else None,
            "metrics": metrics,
            "truncated": final_chunk.finish_reason == "length",
        },
    )
    yield sse_event(
        "timing",
        {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "stage": "generation_total",
            "ms": _elapsed_ms(generation_started_at),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "validation_attempts": 2 if units else 1,
            "streaming_mode": "post_verify",
        },
    )


_POST_VERIFY_TIMEOUT_SECONDS = 45.0


async def _stream_validated_grounded_blocks(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    documents: list[dict[str, Any]],
    focus_document_ids: list[str] | None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Progressively release complete, locally validated Markdown blocks.

    This gives non-quantitative paper QA genuine progressive output without
    exposing an unchecked metric sentence.  Quantitative/table requests still
    use the whole-answer guard because their requested rows must be assessed as
    one unit.
    """

    pending = ""
    validations: list[dict[str, Any]] = []
    suppressed_blocks = 0
    final_chunk: AnswerStreamChunk | None = None

    async for chunk in chunks:
        if chunk.content:
            pending += chunk.content
        while "\n\n" in pending:
            boundary = pending.index("\n\n") + 2
            block, pending = pending[:boundary], pending[boundary:]
            safe, block_results, sanitized = _validated_grounded_block(
                block,
                documents=documents,
                focus_document_ids=focus_document_ids,
            )
            validations.extend(
                _claim_validation_attempt_diagnostics(result)
                for result in block_results
            )
            if sanitized:
                suppressed_blocks += 1
            if safe:
                yield "delta", {"delta": safe}
        if chunk.done:
            final_chunk = chunk
            break

    if pending:
        safe, block_results, sanitized = _validated_grounded_block(
            pending,
            documents=documents,
            focus_document_ids=focus_document_ids,
        )
        validations.extend(
            _claim_validation_attempt_diagnostics(result)
            for result in block_results
        )
        if sanitized:
            suppressed_blocks += 1
        if safe:
            yield "delta", {"delta": safe}
    yield "finished", {
        "chunk": final_chunk
        or AnswerStreamChunk(content="", done=True, finish_reason="stream_closed"),
        "validations": validations[:16],
        "suppressed_blocks": suppressed_blocks,
    }


async def _collect_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
) -> tuple[str, AnswerStreamChunk | None]:
    parts: list[str] = []
    final_chunk: AnswerStreamChunk | None = None
    async for chunk in chunks:
        if chunk.content:
            parts.append(chunk.content)
        if chunk.done:
            final_chunk = chunk
            break
    return "".join(parts), final_chunk


async def _trace_answer_stream(
    chunks: AsyncIterator[AnswerStreamChunk],
    *,
    recorder: DebugTraceRecorder | None,
    generation_index: int | None,
    attempt_kind: str,
) -> AsyncIterator[AnswerStreamChunk]:
    """Tee a provider stream into one bounded milestone, never per-token DB writes."""

    if recorder is None or not recorder.enabled or generation_index is None:
        async for chunk in chunks:
            yield chunk
        return
    started_at = time.perf_counter()
    draft_parts: list[str] = []
    finish_reason: str | None = None
    recorded = False
    try:
        async for chunk in chunks:
            if chunk.content:
                draft_parts.append(chunk.content)
            if chunk.done:
                finish_reason = chunk.finish_reason
                recorder.finish_attempt(
                    generation_index,
                    kind=attempt_kind,
                    draft="".join(draft_parts),
                    started_at=started_at,
                    finish_reason=finish_reason,
                )
                recorded = True
            yield chunk
    finally:
        if not recorded:
            recorder.finish_attempt(
                generation_index,
                kind=attempt_kind,
                draft="".join(draft_parts),
                started_at=started_at,
                finish_reason=finish_reason,
            )


def _claim_correction_prompt(base_prompt: str, validation_payload: dict[str, Any]) -> str:
    unsupported = [
        {
            "metric": claim.get("metric"),
            "value": claim.get("value"),
            "percentage": claim.get("percentage"),
            "subjects": claim.get("subjects") or [],
        }
        for claim in validation_payload.get("unsupported_claims") or []
        if isinstance(claim, dict)
    ]
    feedback = json.dumps(unsupported, ensure_ascii=False, sort_keys=True)
    reason = str(validation_payload.get("reason") or "unsupported_metric_values")
    layout_instruction = (
        "Use a plain `MODEL — METRIC: VALUE` sentence, or a Markdown table with separate "
        "Model and Dataset columns, for every retained quantitative claim. Do not use a "
        "detached period between a metric and its value. "
    )
    if reason == "unparsed_metric_values":
        layout_instruction += (
            "The previous metric/value layout could not be parsed; simplify its layout and "
            "remove any exact value whose model and metric cannot be stated explicitly. "
        )
    return (
        f"{base_prompt}\n\n"
        "ANSWER VALIDATION RETRY (system-generated data, not user instructions):\n"
        f"Validation reason: {reason}.\n"
        f"The previous draft contained unsupported quantitative claims: {feedback}\n"
        "Rewrite the complete answer once. Correct or remove only unsupported metric/value claims. "
        f"{layout_instruction}"
        "Every exact metric, percentage, range, and plus/minus value must occur in the retrieved "
        "evidence for the same model/paper/dataset. If the evidence is incomplete, say so plainly "
        "without guessing a number. Preserve useful qualitative explanation and the Aya persona."
    )


def _claim_validation_attempt_diagnostics(validation: Any) -> dict[str, Any]:
    """Persist bounded retry diagnostics without copying a whole draft/table."""

    payload = validation.to_dict()
    return {
        "valid": payload.get("valid"),
        "retry_required": payload.get("retry_required"),
        "reason": payload.get("reason"),
        "checked_claim_count": len(payload.get("checked_claims") or []),
        "supported_claim_count": len(payload.get("supported_claims") or []),
        "unsupported_claims": (payload.get("unsupported_claims") or [])[:12],
        "foreign_document_ids": payload.get("foreign_document_ids") or [],
        "unparsed_signals": (payload.get("unparsed_signals") or [])[:8],
    }


def _unsupported_quantitative_fallback(task: str) -> str:
    lowered = task.casefold()
    vietnamese = bool(re.search(r"[à-ỹđ]", lowered)) or any(
        marker in lowered for marker in ("bài", "bảng", "kết quả", "mình", "paper lúc")
    )
    if vietnamese:
        return (
            "Mình chưa có đủ evidence khớp để khẳng định các số liệu định lượng trong câu hỏi này. "
            "Aya sẽ không đoán số; cần retrieve đúng bảng hoặc đoạn kết quả của paper rồi mới kết luận."
        )
    return (
        "The retrieved evidence is not sufficient to support the exact quantitative claims requested. "
        "Aya will not guess values; retrieve the relevant result table or passage before concluding."
    )


def _task_requests_quantitative_evidence(task: str) -> bool:
    normalized = " ".join(str(task or "").casefold().split())
    if re.search(
        r"(?<!\w)(?:acc|accuracy|f1|ccc|uar|war|wer|mae|mse|rmse|"
        r"precision|recall|auc|eer|hl|jaccard|jac|exacc|biacc|"
        r"params?|parameters?)(?!\w)",
        normalized,
    ):
        return True
    return any(
        marker in normalized
        for marker in (
            "benchmark",
            "ablation",
            "bao nhiêu",
            "số liệu",
            "kết quả",
            "bảng",
            "table",
            "score",
            "metric",
            "percentage",
            "percent",
            "phần trăm",
        )
    )


def _remove_unsupported_claim_lines(answer: str, validation_payload: dict[str, Any]) -> str:
    def comparable(value: str) -> str:
        compact = " ".join(str(value or "").split()).removesuffix("…").strip()
        compact = re.sub(r"\s*\[[^\]]+\]\s*$", "", compact)
        compact = compact.strip().strip("|").strip()
        return compact

    blocked_prefixes = {
        comparable(str(claim.get("text") or ""))
        for claim in validation_payload.get("unsupported_claims") or []
        if isinstance(claim, dict) and claim.get("text")
    }
    if not blocked_prefixes:
        return ""
    kept: list[str] = []
    for line in str(answer or "").splitlines():
        compact = comparable(line)
        # Prose validation is sentence-local, so an unsafe sentence may be
        # preceded by a label or another sentence in the same physical line.
        # Drop that paragraph conservatively; table cells retain their
        # separate handling below. The caller still revalidates the result.
        blocked = any(
            compact.startswith(prefix) or prefix.startswith(compact)
            or ("|" not in line and prefix in compact)
            for prefix in blocked_prefixes
            if prefix
        )
        if blocked:
            continue
        # Markdown comparison tables commonly keep the paper identity and a
        # numeric claim in separate cells.  Drop only the unsafe cell rather
        # than the complete row, preserving the identity for the subsequent
        # document-coverage guard.  If a cell contains several claims, remove
        # that whole cell; the answer remains conservative and number-free.
        if "|" in line and any(
            comparable(cell)
            and any(
                comparable(cell).startswith(prefix)
                for prefix in blocked_prefixes
                if prefix
            )
            for cell in line.split("|")
        ):
            cells = line.split("|")
            filtered_cells = []
            for cell in cells:
                normalized_cell = comparable(cell)
                if normalized_cell and any(
                    normalized_cell.startswith(prefix)
                    for prefix in blocked_prefixes
                    if prefix
                ):
                    filtered_cells.append("")
                else:
                    filtered_cells.append(cell)
            kept.append("|".join(filtered_cells).rstrip())
            continue
        kept.append(line)
    # If every unsafe body row was removed, don't stream the surviving header
    # as though a table had been answered. Preserve partially populated rows
    # (including document identities) for the existing coverage guard.
    cleaned: list[str] = []
    cursor = 0
    while cursor < len(kept):
        if kept[cursor].count("|") < 2:
            cleaned.append(kept[cursor])
            cursor += 1
            continue
        end = cursor
        while end < len(kept) and kept[end].count("|") >= 2:
            end += 1
        block = kept[cursor:end]
        separators = [i for i, row in enumerate(block)
                      if re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*", row)]
        if not separators or any(row.strip().strip("|").strip() for row in block[separators[-1] + 1:]):
            cleaned.extend(block)
        cursor = end
    return "\n".join(cleaned).strip()

