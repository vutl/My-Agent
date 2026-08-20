from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from app.db.sqlite import connect
from app.rag.figure_quality import extract_figure_label, merge_visual_quality_metadata
from app.rag.vision import (
    FigureDocumentContext,
    OllamaVisionSummarizer,
    OpenAICompatibleVisionSummarizer,
    VisionSummaryError,
    is_low_signal_figure_asset,
)


@dataclass(frozen=True)
class FigureEnrichService:
    db_path: Path
    ollama_host: str
    vision_model: str
    vision_provider: str = "openai_compatible"
    openai_api_base: str = "http://localhost:20128/v1"
    openai_api_key: str = "any"
    request_timeout_seconds: float = 120.0
    artifact_root: Path | None = None

    def enrich_document(
        self,
        document_id: str | None = None,
        *,
        force: bool = False,
        limit: int | None = None,
    ) -> dict:
        figures = self._list_figures(document_id=document_id, limit=limit)
        if self.vision_provider == "openai_compatible":
            summarizer = OpenAICompatibleVisionSummarizer(
                base_url=self.openai_api_base,
                api_key=self.openai_api_key or "any",
                model=self.vision_model,
                timeout_seconds=max(self.request_timeout_seconds, 300.0),
            )
        elif self.vision_provider == "ollama":
            summarizer = OllamaVisionSummarizer(
                host=self.ollama_host,
                model=self.vision_model,
                timeout_seconds=self.request_timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported VISION_PROVIDER: {self.vision_provider}")

        enriched = 0
        skipped = 0
        failed = 0
        results: list[dict] = []
        incomplete_document_ids: set[str] = set()
        processed_document_ids: set[str] = set()
        rate_limited = False
        remaining = 0

        for figure_offset, figure in enumerate(figures):
            figure_id = str(figure["id"])
            processed_document_ids.add(str(figure.get("document_id") or ""))
            caption = figure.get("caption")
            extraction_method = figure.get("extraction_method")
            image_path_raw = figure.get("image_path")
            existing_summary = (figure.get("visual_summary") or "").strip()
            metadata = _json_object(figure.get("metadata_json"))

            if is_low_signal_figure_asset(caption=caption, extraction_method=extraction_method):
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "low_signal",
                    }
                )
                continue

            if metadata.get("quality_status") == "rejected":
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "quality_rejected",
                    }
                )
                continue

            if _is_unanchored_review(metadata=metadata, caption=caption) and not force:
                # Ambiguous, uncaptioned crops are excluded from retrieval by
                # the quality gate already. Bulk enrichment should spend local
                # VLM time on grounded Figure N assets; `force` remains the
                # explicit audit path for reviewing these leftovers.
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "unanchored_needs_review",
                    }
                )
                continue

            has_structured_context = all(
                key in metadata for key in ("asset_kind", "is_content", "is_complete")
            )
            expected_provider = (
                "9router" if self.vision_provider == "openai_compatible" else "ollama"
            )
            has_current_provenance = (
                metadata.get("vision_provider") == expected_provider
                and metadata.get("vision_model") == self.vision_model
            )
            if (
                existing_summary
                and has_structured_context
                and has_current_provenance
                and not force
            ):
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": True,
                        "skipped": True,
                        "reason": "already_enriched",
                    }
                )
                continue

            if not image_path_raw:
                incomplete_document_ids.add(str(figure.get("document_id") or ""))
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "missing_image",
                    }
                )
                continue

            image_path = Path(str(image_path_raw)).expanduser()
            if not image_path.is_file():
                incomplete_document_ids.add(str(figure.get("document_id") or ""))
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "image_not_found",
                    }
                )
                continue

            try:
                document_context = self._document_context(figure)
                context = summarizer.summarize_image_context(
                    image_path,
                    caption=caption,
                    page_text=document_context.nearby_text,
                    extraction_method=extraction_method,
                    document_context=document_context,
                    page_image_path=self._page_image_path(figure),
                )
            except VisionSummaryError as exc:
                incomplete_document_ids.add(str(figure.get("document_id") or ""))
                failed += 1
                error_text = str(exc)
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "error": error_text,
                    }
                )
                if _is_rate_limit_error(error_text):
                    # Persist successes from this batch and stop immediately;
                    # hammering the remaining figures cannot clear a gateway
                    # quota and only produces dozens of redundant failures.
                    rate_limited = True
                    remaining = len(figures) - figure_offset - 1
                    break
                continue

            if context is None or not context.raw_text.strip():
                incomplete_document_ids.add(str(figure.get("document_id") or ""))
                skipped += 1
                results.append(
                    {
                        "figure_id": figure_id,
                        "ok": False,
                        "skipped": True,
                        "reason": "empty_context",
                    }
                )
                continue

            self._update_figure_context(
                figure_id=figure_id,
                visual_summary=context.raw_text,
                metadata_patch=context.to_metadata(),
                existing_metadata_json=figure.get("metadata_json"),
            )
            enriched += 1
            results.append(
                {
                    "figure_id": figure_id,
                    "document_id": figure.get("document_id"),
                    "figure_index": figure.get("figure_index"),
                    "ok": True,
                    "figure_type": context.figure_type,
                    "asset_kind": context.asset_kind,
                    "is_content": context.is_content,
                    "is_complete": context.is_complete,
                    "quality_status": context.to_metadata().get("quality_status"),
                    "vision_provider": context.vision_provider,
                    "vision_model": context.vision_model,
                    "summary_chars": len(context.raw_text),
                }
            )

        completed_document_ids = {
            processed_document_id
            for processed_document_id in processed_document_ids - incomplete_document_ids - {""}
            if self._document_vision_is_complete(processed_document_id)
        }
        for completed_document_id in completed_document_ids:
            self._mark_document_vision_fingerprint(completed_document_id)

        return {
            "document_id": document_id,
            "total": len(figures),
            "enriched": enriched,
            "skipped": skipped,
            "failed": failed,
            "rate_limited": rate_limited,
            "remaining": remaining,
            "results": results,
        }

    def _list_figures(
        self,
        *,
        document_id: str | None,
        limit: int | None,
    ) -> list[dict]:
        query = """
            SELECT f.id, f.document_id, f.figure_index, f.page_number, f.caption,
                   f.image_path, f.visual_summary, f.extraction_method, f.metadata_json,
                   d.filename, d.source_path, d.title AS document_title,
                   dc.title_guess, dc.short_summary
            FROM document_figures f
            JOIN documents d ON d.id = f.document_id
            LEFT JOIN document_cards dc ON dc.document_id = f.document_id
        """
        params: list[object] = []
        if document_id:
            query += " WHERE f.document_id = ?"
            params.append(document_id)
        query += " ORDER BY d.indexed_at ASC, f.figure_index ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with connect(self.db_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def _document_context(self, figure: dict) -> FigureDocumentContext:
        page_text = self._page_text_excerpt(figure)
        return FigureDocumentContext(
            filename=figure.get("filename"),
            title=figure.get("title_guess") or figure.get("document_title"),
            summary=_clean_document_summary(figure.get("short_summary")),
            section_title=self._section_title(figure),
            page_number=figure.get("page_number"),
            nearby_text=page_text,
            reference_sentences=tuple(self._reference_sentences(figure, page_text)),
            nearby_tables=tuple(self._nearby_tables(figure)),
        )

    def _page_text_excerpt(self, figure: dict) -> str | None:
        metadata = json.loads(figure.get("metadata_json") or "{}")
        excerpt = metadata.get("page_text_excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            return excerpt
        page_number = figure.get("page_number")
        document_id = figure.get("document_id")
        if page_number is None or not document_id:
            return None
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT content
                FROM chunks
                WHERE document_id = ? AND page_number = ?
                ORDER BY chunk_index ASC
                LIMIT 3
                """,
                (document_id, page_number),
            ).fetchall()
        if not rows:
            return None
        return "\n\n".join(str(row["content"]) for row in rows if row["content"])[:3000]

    def _section_title(self, figure: dict) -> str | None:
        page_number = figure.get("page_number")
        document_id = figure.get("document_id")
        if page_number is None or not document_id:
            return None
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM chunks
                WHERE document_id = ? AND page_number = ?
                ORDER BY chunk_index ASC
                LIMIT 6
                """,
                (document_id, page_number),
            ).fetchall()
        for row in rows:
            metadata = _json_object(row["metadata_json"])
            section = metadata.get("section_title")
            if isinstance(section, str) and section.strip():
                return section.strip()
        return None

    def _reference_sentences(self, figure: dict, page_text: str | None) -> list[str]:
        text = " ".join((page_text or "").split())
        if not text:
            return []
        caption = str(figure.get("caption") or "")
        match = re.match(r"\s*(?:fig(?:ure)?\.?|hình)\s*(\d+)", caption, flags=re.IGNORECASE)
        if not match:
            return []
        number = re.escape(match.group(1))
        reference = re.compile(rf"\b(?:fig(?:ure)?\.?|hình)\s*{number}\b", flags=re.IGNORECASE)
        protected = re.sub(r"\b(Fig|fig)\.", r"\1<FIG_DOT>", text)
        sentences = [
            sentence.replace("<FIG_DOT>", ".")
            for sentence in re.split(r"(?<=[.!?])\s+", protected)
        ]
        return [sentence[:600] for sentence in sentences if reference.search(sentence)][:4]

    def _nearby_tables(self, figure: dict) -> list[str]:
        page_number = figure.get("page_number")
        document_id = figure.get("document_id")
        if page_number is None or not document_id:
            return []
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT caption, markdown
                FROM document_tables
                WHERE document_id = ? AND page_number = ?
                ORDER BY table_index ASC
                LIMIT 3
                """,
                (document_id, page_number),
            ).fetchall()
        return [
            "\n".join(part for part in (row["caption"], row["markdown"]) if part)[:800]
            for row in rows
            if row["caption"] or row["markdown"]
        ]

    def _page_image_path(self, figure: dict) -> Path | None:
        page_number = figure.get("page_number")
        document_id = figure.get("document_id")
        if page_number is None:
            return None
        candidates: list[Path] = []
        if self.artifact_root is not None and document_id:
            candidates.append(
                self.artifact_root / str(document_id) / "pages" / f"page_{int(page_number):03d}.png"
            )
        image_path_raw = figure.get("image_path")
        if image_path_raw:
            image_path = Path(str(image_path_raw)).expanduser()
            if image_path.parent.name == "figures":
                candidates.append(image_path.parent.parent / "pages" / f"page_{int(page_number):03d}.png")
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def _update_figure_context(
        self,
        *,
        figure_id: str,
        visual_summary: str,
        metadata_patch: dict,
        existing_metadata_json: str | None,
    ) -> None:
        metadata = json.loads(existing_metadata_json or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
        merge_visual_quality_metadata(metadata, metadata_patch)
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE document_figures
                SET visual_summary = ?, metadata_json = ?
                WHERE id = ?
                """,
                (visual_summary, json.dumps(metadata, ensure_ascii=False), figure_id),
            )
            connection.commit()

    def _mark_document_vision_fingerprint(self, document_id: str) -> None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                return
            metadata = _json_object(row["metadata_json"])
            metadata.update(
                {
                    "vision_provider": self.vision_provider,
                    "vision_model": self.vision_model,
                }
            )
            connection.execute(
                "UPDATE documents SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), document_id),
            )

    def _document_vision_is_complete(self, document_id: str) -> bool:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT caption, extraction_method, visual_summary, metadata_json
                FROM document_figures
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchall()
        expected_provider = (
            "9router" if self.vision_provider == "openai_compatible" else "ollama"
        )
        for row in rows:
            caption = row["caption"]
            extraction_method = row["extraction_method"]
            metadata = _json_object(row["metadata_json"])
            if is_low_signal_figure_asset(
                caption=caption,
                extraction_method=extraction_method,
            ):
                continue
            if metadata.get("quality_status") == "rejected":
                continue
            if _is_unanchored_review(metadata=metadata, caption=caption):
                continue
            if not str(row["visual_summary"] or "").strip():
                return False
            if not all(key in metadata for key in ("asset_kind", "is_content", "is_complete")):
                return False
            if metadata.get("vision_provider") != expected_provider:
                return False
            if metadata.get("vision_model") != self.vision_model:
                return False
        return True


def _json_object(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_unanchored_review(*, metadata: dict, caption: str | None) -> bool:
    if metadata.get("quality_status") != "needs_review":
        return False
    return not bool(metadata.get("figure_number") or extract_figure_label(caption))


def _is_rate_limit_error(error: str) -> bool:
    normalized = " ".join((error or "").lower().split())
    return "429" in normalized and (
        "rate" in normalized or "usage limit" in normalized or "quota" in normalized
    )


def _clean_document_summary(summary: str | None) -> str | None:
    text = " ".join((summary or "").split())
    if not text:
        return None
    # Publisher download pages often precede the paper abstract. Keep this
    # context short and strip the most common operational metadata fragments;
    # it remains explicitly untrusted in the VLM prompt.
    text = re.sub(
        r"\b(?:PDF Download|Total Citations|Total Downloads|Citation in BibTeX format)\b[^.]*\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())[:1400] or None
