"""Offline/resumable GPT-5.6 builder for canonical paper evidence cards."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.core.config import APPROVED_9ROUTER_MODELS
from app.llm.ollama_client import OllamaError
from app.rag.paper_facets import CORE_PAPER_FACETS
from app.services.paper_evidence_service import (
    EVIDENCE_CARD_PROVIDER,
    EvidenceFacetDraft,
    EvidenceItemDraft,
    EvidenceRefDraft,
    PaperEvidenceDraft,
    PaperEvidenceService,
)


class PaperEvidenceBuilder:
    def __init__(
        self,
        *,
        service: PaperEvidenceService,
        client: Any,
        model: str,
        max_concurrency: int = 2,
    ) -> None:
        if model not in APPROVED_9ROUTER_MODELS:
            raise ValueError(
                f"Paper evidence model must be an approved 9router model, got {model!r}"
            )
        self.service = service
        self.client = client
        self.model = model
        self.max_concurrency = max(1, min(int(max_concurrency), 4))

    async def build_document(self, document_id: str, *, force: bool = False) -> dict[str, Any]:
        existing = self.service.card_for_document(document_id)
        if (
            not force
            and existing
            and not existing.get("stale")
            # A partial card is a successful terminal build: it means one or
            # more core facets had no canonical support, not that the provider
            # failed. Runtime retrieval deliberately fills those facets from
            # raw evidence. Rebuilding every partial card on each resumable run
            # wastes provider calls and cannot create evidence that is absent
            # from the indexed document; --force remains the explicit retry.
            and existing.get("status") in {"complete", "partial"}
        ):
            return {"document_id": document_id, "status": "skipped", "card": existing}

        self.service.mark_job(document_id, "building", model=self.model)
        try:
            candidates = self.service.candidate_sources(document_id, per_facet=3)
            if not candidates:
                raise ValueError("No canonical candidate sources available")
            user_prompt = _build_prompt(document_id, candidates)
            completion_text = await _stream_json_completion(
                self.client,
                model=self.model,
                num_predict=3000,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            generation_attempts = 1
            try:
                payload = _parse_json_object(completion_text)
            except ValueError:
                # One bounded same-model retry is reserved for serialization
                # failures only. Re-extracting with a smaller output contract
                # is safer than attempting to repair/trust a truncated object.
                # Provider, quota and timeout errors still fail immediately,
                # and canonical source validation remains unchanged.
                completion_text = await _stream_json_completion(
                    self.client,
                    model=self.model,
                    num_predict=2200,
                    messages=[
                        {"role": "system", "content": _JSON_REPAIR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                generation_attempts = 2
                payload = _parse_json_object(completion_text)
            draft = _draft_from_payload(
                payload,
                document_id=document_id,
                model=self.model,
                candidates=candidates,
                generation_attempts=generation_attempts,
            )
            card = self.service.publish(draft)
            return {"document_id": document_id, "status": "complete", "card": card}
        except Exception as exc:
            self.service.mark_job(document_id, "failed", model=self.model, error=str(exc))
            raise

    async def build_all(
        self,
        *,
        document_ids: list[str] | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        ids = list(document_ids or self.service.list_document_ids(limit=limit))
        if limit is not None:
            ids = ids[: max(0, int(limit))]
        completed: list[dict[str, Any]] = []
        # Batch instead of spawning the full corpus at once. If quota/provider
        # fails, no later batch is launched and the durable job rows show the
        # exact resume point. No model/provider fallback is attempted.
        for offset in range(0, len(ids), self.max_concurrency):
            batch = ids[offset : offset + self.max_concurrency]
            results = await asyncio.gather(
                *(self.build_document(document_id, force=force) for document_id in batch),
                return_exceptions=True,
            )
            first_error: Exception | None = None
            for document_id, result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    completed.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error": " ".join(str(result).split())[:500],
                        }
                    )
                    if first_error is None:
                        first_error = result
                else:
                    completed.append(result)
            if first_error is not None:
                if isinstance(first_error, OllamaError):
                    raise first_error
                raise RuntimeError(
                    f"Evidence-card backfill stopped after failed batch: {first_error}"
                ) from first_error
        return {
            "requested": len(ids),
            "completed": sum(item.get("status") == "complete" for item in completed),
            "skipped": sum(item.get("status") == "skipped" for item in completed),
            "results": completed,
        }


_SYSTEM_PROMPT = """You extract compact evidence cards from ONE research paper.
Return one JSON object only. Never use outside knowledge. Every item must cite one or
more supplied source_id values and copy a short exact quote from those sources.
Do not infer missing metrics, datasets, contributions, or architecture. Numeric claims
must preserve the exact model/owner, dataset, metric and value visible in evidence.
Use these facet keys exactly: task, architecture, dataset_setup, benchmark_results,
contributions. If a facet is unsupported, return an empty items list.
The output schema is:
{"facets":{"task":{"confidence":0.0,"items":[{"claim":"...","refs":[{"source_id":"...","quote":"exact quote"}]}]},"architecture":...,"dataset_setup":...,"benchmark_results":...,"contributions":...}}
Keep at most 4 items per facet. The application derives every synopsis only from
items that pass canonical source and claim validation."""


_JSON_REPAIR_SYSTEM_PROMPT = _SYSTEM_PROMPT + """

The previous serialization attempt was invalid. Return one compact JSON object only:
no Markdown, no preamble, no trailing explanation. Use at most 2 items per facet and
keep each exact quote under 40 words. Omit unsupported facets by using an empty items
list. Do not change model/provider and do not use outside knowledge."""


async def _stream_json_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    num_predict: int,
) -> str:
    """Collect native SSE so 9router never performs fragile stream conversion."""

    parts: list[str] = []
    async for chunk in client.stream_chat(
        model=model,
        messages=messages,
        temperature=0.0,
        num_predict=num_predict,
        response_format={"type": "json_object"},
    ):
        content = str(getattr(chunk, "content", "") or "")
        if content:
            parts.append(content)
    rendered = "".join(parts).strip()
    if not rendered:
        raise OllamaError("Evidence-card model returned an empty streamed completion")
    return rendered


def _build_prompt(document_id: str, candidates: list[dict[str, Any]]) -> str:
    rendered = []
    for candidate in candidates:
        rendered.append(
            json.dumps(
                {
                    "source_id": candidate["source_id"],
                    "source_kind": candidate["source_kind"],
                    "page": candidate.get("page"),
                    "label": candidate.get("label"),
                    "content": str(candidate.get("content") or "")[:1800],
                },
                ensure_ascii=False,
            )
        )
    return (
        f"document_id={document_id}\n"
        "Extract the five requested facets from these canonical candidates.\n\n"
        + "\n".join(rendered)
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        return payload
    if payload is not None:
        raise ValueError("Evidence-card model returned a non-object JSON value")

    # Providers occasionally wrap an otherwise valid object in a short prose
    # preamble or Markdown fence despite a JSON-only instruction. Extract only
    # unambiguous object candidates that contain the required top-level key;
    # never use a greedy first-"{"/last-"}" slice that can merge an example
    # schema and an answer into invalid or misleading evidence.
    candidates: list[dict[str, Any]] = []
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in fenced_blocks:
        try:
            decoded = json.loads(block.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(decoded, dict) and isinstance(decoded.get("facets"), dict):
            candidates.append(decoded)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            decoded, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and isinstance(decoded.get("facets"), dict):
            candidates.append(decoded)

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        fingerprint = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        unique[fingerprint] = candidate
    if len(unique) == 1:
        return next(iter(unique.values()))
    if len(unique) > 1:
        raise ValueError("Evidence-card model returned ambiguous multiple JSON objects")
    raise ValueError("Evidence-card model returned invalid JSON")


def _draft_from_payload(
    payload: dict[str, Any],
    *,
    document_id: str,
    model: str,
    candidates: list[dict[str, Any]],
    generation_attempts: int = 1,
) -> PaperEvidenceDraft:
    raw_facets = payload.get("facets")
    if not isinstance(raw_facets, dict):
        raise ValueError("Evidence-card JSON is missing facets object")
    candidate_by_id = {str(item["source_id"]): item for item in candidates}
    facets: list[EvidenceFacetDraft] = []
    for facet_name in CORE_PAPER_FACETS:
        raw = raw_facets.get(facet_name) or {}
        if not isinstance(raw, dict):
            continue
        items: list[EvidenceItemDraft] = []
        raw_items = raw.get("items") or []
        if not isinstance(raw_items, list):
            raw_items = []
        for raw_item in raw_items[:4]:
            if not isinstance(raw_item, dict):
                continue
            refs: list[EvidenceRefDraft] = []
            for raw_ref in (raw_item.get("refs") or [])[:4]:
                if not isinstance(raw_ref, dict):
                    continue
                source_id = str(raw_ref.get("source_id") or "").strip()
                candidate = candidate_by_id.get(source_id)
                quote = str(raw_ref.get("quote") or "").strip()
                if candidate is None or not quote:
                    continue
                refs.append(
                    EvidenceRefDraft(
                        source_kind=str(candidate["source_kind"]),
                        source_id=source_id,
                        quote=quote,
                        page=candidate.get("page"),
                        section_title=str(candidate.get("label") or "") or None,
                        source_content_hash=str(candidate.get("source_content_hash") or "") or None,
                    )
                )
            claim = str(raw_item.get("claim") or "").strip()
            if claim and refs:
                items.append(EvidenceItemDraft(claim_text=claim, evidence_refs=refs))
        facets.append(
            EvidenceFacetDraft(
                facet=facet_name,
                synopsis=str(raw.get("synopsis") or "").strip(),
                items=items,
                status="complete" if items else "unavailable",
                confidence=_bounded_float(raw.get("confidence")),
            )
        )
    return PaperEvidenceDraft(
        document_id=document_id,
        facets=facets,
        generator_model=model,
        generator_provider=EVIDENCE_CARD_PROVIDER,
        metadata={
            "candidate_source_count": len(candidates),
            "generation_attempts": max(1, int(generation_attempts)),
        },
    )


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
