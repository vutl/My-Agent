"""Revision-scoped ColSmol visual page retrieval.

This module is an optional Gate-E backend.  It keeps page rendering and
multi-vector embeddings outside canonical text/table/figure projections.  The
provider is deliberately linear-scan for the isolated benchmark: promotion to
a larger corpus requires a measured PLAID or equivalent index, not an implied
production claim from a small smoke.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Protocol

from app.rag.visual_retrieval import VisualPageRecord, VisualSearchResult


DEFAULT_COLSMOL_MODEL = "vidore/colSmol-256M"
DEFAULT_COLSMOL_REVISION = "c79b633e17e060cc109b11aa2bf92a1517d3bd6f"
DEFAULT_COLSMOL_BASE_MODEL = "vidore/ColSmolVLM-Instruct-256M-base"
DEFAULT_COLSMOL_BASE_REVISION = "99ca96f1f6b95b3a69e6abef74a2416cb738fed0"


class VisualRetrievalUnavailable(RuntimeError):
    """The optional visual provider cannot be loaded or its index is invalid."""


def resolve_colsmol_device(requested_device: str, *, mps_available: bool) -> str:
    """Resolve a safe device for the dependency-isolated visual worker.

    ``torch.backends.mps.is_available()`` is not a sufficient capability
    probe for ColSmol: on some Apple/torch/transformers combinations the
    process aborts while materialising Idefics3.  Therefore ``auto`` is
    conservative and uses CPU unless MPS is explicitly opted in.  An
    explicit ``mps`` request still fails closed when unavailable, while an
    opt-in MPS crash can be retried by the worker client on CPU.
    """

    if requested_device not in {"auto", "cpu", "mps"}:
        raise VisualRetrievalUnavailable(f"unsupported visual device: {requested_device}")
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "mps":
        if not mps_available:
            raise VisualRetrievalUnavailable("requested MPS device is unavailable")
        return "mps"
    allow_mps = os.getenv("COLSMOL_ALLOW_MPS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return "mps" if mps_available and allow_mps else "cpu"


def resolve_colsmol_score_batch_size(value: str | None = None) -> int:
    """Bound late-interaction scoring before MaxSim materialises its tensor.

    A single visual page has many token vectors.  On a CPU-only M1 runtime a
    64-page MaxSim batch can cross an allocator/cache cliff on long documents,
    even though a 16-page batch has identical scores.  Keep a conservative
    default while retaining an explicit environment override for hosts with a
    measured larger safe envelope.
    """

    try:
        requested = int(value if value is not None else os.getenv("COLSMOL_SCORE_BATCH_SIZE", "16"))
    except (TypeError, ValueError):
        requested = 16
    return max(1, min(requested, 128))


class MultiVectorVisualEncoder(Protocol):
    model_fingerprint: str

    def encode_image(self, image_path: Path) -> Any:
        ...

    def encode_images(self, image_paths: list[Path]) -> list[Any]:
        ...

    def encode_query(self, query: str) -> Any:
        ...

    def score(self, query_embedding: Any, document_embeddings: list[Any]) -> list[float]:
        ...


class ColSmolEncoder:
    """Lazy ColSmol encoder with independently pinned adapter/base snapshots."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        model_name: str = DEFAULT_COLSMOL_MODEL,
        model_revision: str = DEFAULT_COLSMOL_REVISION,
        base_model_name: str = DEFAULT_COLSMOL_BASE_MODEL,
        base_model_revision: str = DEFAULT_COLSMOL_BASE_REVISION,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.runtime_root = runtime_root.expanduser().resolve()
        self.model_name = model_name
        self.model_revision = model_revision
        self.base_model_name = base_model_name
        self.base_model_revision = base_model_revision
        self.requested_device = device
        self.local_files_only = local_files_only
        self.model_fingerprint = _fingerprint(
            {
                "provider": "colsmol",
                "schema": "colsmol-multivector-v1",
                "model": model_name,
                "model_revision": model_revision,
                "base_model": base_model_name,
                "base_model_revision": base_model_revision,
            }
        )
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._dtype: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from colpali_engine.models import ColIdefics3, ColIdefics3Processor
                from huggingface_hub import snapshot_download
            except Exception as exc:  # pragma: no cover - environment dependent
                raise VisualRetrievalUnavailable(
                    "colpali-engine visual extra is not installed"
                ) from exc

            # The visual worker is isolated from Aya's main process, so it is
            # safe to tune CPU parallelism here.  Some PyTorch wheels default
            # to only four threads even on an 8-core Apple Silicon machine,
            # which makes late-interaction scoring of long PDFs look hung.
            # Keep an explicit escape hatch for constrained hosts and never
            # exceed the machine's advertised logical CPU count.
            try:
                requested_threads = int(
                    os.getenv("COLSMOL_TORCH_THREADS", str(os.cpu_count() or 1))
                )
            except ValueError:
                requested_threads = os.cpu_count() or 1
            thread_count = max(1, min(requested_threads, os.cpu_count() or 1))
            try:
                torch.set_num_threads(thread_count)
            except RuntimeError:
                # A host may have initialized a parallel region before the
                # lazy model load.  In that case retain PyTorch's safe default
                # rather than failing the optional visual provider.
                thread_count = torch.get_num_threads()

            device = resolve_colsmol_device(
                self.requested_device,
                mps_available=bool(torch.backends.mps.is_available()),
            )
            dtype = torch.float16 if device == "mps" else torch.float32

            model_root = self.runtime_root / "models" / self.model_fingerprint
            model_root.mkdir(parents=True, exist_ok=True)
            try:
                adapter_snapshot = Path(
                    snapshot_download(
                        self.model_name,
                        revision=self.model_revision,
                        local_files_only=self.local_files_only,
                    )
                )
                base_snapshot = Path(
                    snapshot_download(
                        self.base_model_name,
                        revision=self.base_model_revision,
                        local_files_only=self.local_files_only,
                    )
                )
                resolved_adapter = model_root / "resolved-adapter"
                resolved_adapter.mkdir(parents=True, exist_ok=True)
                for item in adapter_snapshot.iterdir():
                    target = resolved_adapter / item.name
                    if target.exists() or target.is_symlink():
                        continue
                    if item.is_dir():
                        shutil.copytree(item, target, symlinks=True)
                    else:
                        target.symlink_to(item)
                adapter_config_path = adapter_snapshot / "adapter_config.json"
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                adapter_config["base_model_name_or_path"] = str(base_snapshot)
                _atomic_json_write(resolved_adapter / "adapter_config.json", adapter_config)

                model = ColIdefics3.from_pretrained(
                    str(resolved_adapter),
                    torch_dtype=dtype,
                    device_map=device,
                    local_files_only=True,
                ).eval()
                processor = ColIdefics3Processor.from_pretrained(
                    str(adapter_snapshot),
                    local_files_only=True,
                )
            except Exception as exc:
                raise VisualRetrievalUnavailable(f"ColSmol load failed: {exc}") from exc

            self._torch = torch
            self._torch_threads = thread_count
            self._device = device
            self._dtype = dtype
            self._model = model
            self._processor = processor

    def encode_image(self, image_path: Path) -> Any:
        self._ensure_loaded()
        from PIL import Image

        with self._inference_lock, Image.open(image_path) as opened:
            image = opened.convert("RGB")
            inputs = self._processor.process_images([image]).to(self._device)
            with self._torch.inference_mode():
                embedding = self._model(**inputs)[0]
            return embedding.detach().to("cpu", dtype=self._torch.float32).contiguous()

    def encode_images(self, image_paths: list[Path]) -> list[Any]:
        """Encode a bounded page batch in one model call.

        The worker already bounds requests to a small number of pages.  Using
        that same batch here avoids one Transformers/PIL/model dispatch per
        page while preserving the exact per-page embeddings and identities.
        """

        self._ensure_loaded()
        if not image_paths:
            return []
        from PIL import Image

        with self._inference_lock:
            images = []
            try:
                for image_path in image_paths:
                    with Image.open(image_path) as opened:
                        images.append(opened.convert("RGB"))
                inputs = self._processor.process_images(images).to(self._device)
                with self._torch.inference_mode():
                    raw_embeddings = self._model(**inputs)
                # Col* models return ``(batch, patches, dim)`` directly.  A
                # few older wrappers return a one-item tuple, so normalize
                # that compatibility shape without treating the first batch
                # item as the whole result.
                embeddings = (
                    raw_embeddings[0]
                    if isinstance(raw_embeddings, (tuple, list))
                    else raw_embeddings
                )
                if len(embeddings) != len(images):
                    raise VisualRetrievalUnavailable(
                        "ColSmol returned a batch size different from requested pages"
                    )
                return [
                    embedding.detach()
                    .to("cpu", dtype=self._torch.float32)
                    .contiguous()
                    for embedding in embeddings
                ]
            finally:
                for image in images:
                    image.close()

    def encode_query(self, query: str) -> Any:
        self._ensure_loaded()
        if not query.strip():
            raise ValueError("visual query must not be empty")
        with self._inference_lock:
            inputs = self._processor.process_queries([query]).to(self._device)
            with self._torch.inference_mode():
                embedding = self._model(**inputs)[0]
            return embedding.detach().to("cpu", dtype=self._torch.float32).contiguous()

    def score(self, query_embedding: Any, document_embeddings: list[Any]) -> list[float]:
        self._ensure_loaded()
        if not document_embeddings:
            return []
        # Keep one query/device dispatch for the whole candidate set; the
        # processor tiles passages internally. The bounded tile size prevents
        # one large visual document from materialising an impractically large
        # CPU MaxSim tensor while preserving exact per-page scores.
        score_batch_size = resolve_colsmol_score_batch_size()
        with self._inference_lock, self._torch.inference_mode():
            query = query_embedding.to(self._device, dtype=self._dtype)
            documents = [item.to(self._device, dtype=self._dtype) for item in document_embeddings]
            scores = self._processor.score_multi_vector(
                [query],
                documents,
                batch_size=min(score_batch_size, len(documents)),
                device=self._device,
            )[0]
        return [float(value) for value in scores.detach().to("cpu", dtype=self._torch.float32)]


class ColSmolVisualRetrievalProvider:
    """Persistent exact late-interaction index for isolated visual Gate E."""

    name = "colsmol"

    def __init__(
        self,
        *,
        index_root: Path,
        encoder: MultiVectorVisualEncoder | None = None,
        model_name: str = DEFAULT_COLSMOL_MODEL,
        model_revision: str = DEFAULT_COLSMOL_REVISION,
        base_model_name: str = DEFAULT_COLSMOL_BASE_MODEL,
        base_model_revision: str = DEFAULT_COLSMOL_BASE_REVISION,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.index_root = index_root.expanduser().resolve()
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or ColSmolEncoder(
            runtime_root=self.index_root,
            model_name=model_name,
            model_revision=model_revision,
            base_model_name=base_model_name,
            base_model_revision=base_model_revision,
            device=device,
            local_files_only=local_files_only,
        )
        self.model_fingerprint = self.encoder.model_fingerprint
        self._manifest_path = self.index_root / "manifest.json"
        self._embedding_root = self.index_root / "embeddings"
        self._embedding_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    async def index_pages(
        self,
        pages: list[VisualPageRecord],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(self._index_pages_sync, pages)
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        **result,
                        "processed_pages": len(pages),
                        "remaining_pages": 0,
                    }
                )
            except Exception:
                pass
        return result

    def health(self) -> dict[str, Any]:
        """Return cheap worker telemetry without forcing model inference.

        The isolated visual worker is intentionally a separate process.  A
        benchmark that reports only the parent process RSS can therefore
        under-report the expensive model's memory footprint.  Keep this
        method side-effect free: it validates the manifest and reports the
        number/size of committed entries, while ``model_loaded`` tells the
        caller whether the lazy encoder has actually been materialized.
        """

        with self._lock:
            manifest = self._load_manifest()
            entries = manifest.get("entries", {})
            entry_count = len(entries) if isinstance(entries, dict) else 0
            index_bytes = 0
            for path in self.index_root.rglob("*"):
                if path.is_file():
                    try:
                        index_bytes += path.stat().st_size
                    except OSError:
                        continue
        return {
            "ok": True,
            "provider": self.name,
            "model_fingerprint": self.model_fingerprint,
            "index_entries": entry_count,
            "index_size_bytes": index_bytes,
            "model_loaded": bool(getattr(self.encoder, "_model", None) is not None),
            "torch_threads": int(getattr(self.encoder, "_torch_threads", 0) or 0),
        }

    def _index_pages_sync(self, pages: list[VisualPageRecord]) -> dict[str, Any]:
        from safetensors.torch import save_file

        started = time.perf_counter()
        indexed = reused = failed = 0
        errors: list[dict[str, str]] = []
        with self._lock:
            manifest = self._load_manifest()
            entries = manifest.setdefault("entries", {})
            seen_page_ids: dict[str, tuple[str, str, int]] = {}
            pending: list[
                tuple[
                    VisualPageRecord,
                    Path,
                    str,
                    str,
                    Path,
                    dict[str, Any] | None,
                ]
            ] = []
            for page in pages:
                image_path = Path(page.image_path).expanduser().resolve()
                try:
                    identity = (page.document_id, page.revision_id, page.page_number)
                    prior_identity = seen_page_ids.get(page.page_id)
                    if prior_identity is not None and prior_identity != identity:
                        raise ValueError(f"duplicate page_id with conflicting identity: {page.page_id}")
                    seen_page_ids[page.page_id] = identity
                    if not all((page.page_id, page.document_id, page.revision_id)):
                        raise ValueError("page_id, document_id and revision_id are required")
                    if page.page_number < 1:
                        raise ValueError("page_number must be positive")
                    if not image_path.is_file():
                        raise FileNotFoundError(str(image_path))
                    image_sha = _sha256_file(image_path)
                    embedding_name = hashlib.sha256(
                        f"{self.model_fingerprint}:{page.page_id}:{image_sha}".encode("utf-8")
                    ).hexdigest() + ".safetensors"
                    embedding_path = self._embedding_root / embedding_name
                    current = entries.get(page.page_id)
                    if (
                        isinstance(current, dict)
                        and current.get("image_sha256") == image_sha
                        and current.get("model_fingerprint") == self.model_fingerprint
                        and current.get("embedding_file") == embedding_name
                        and embedding_path.is_file()
                    ):
                        reused += 1
                        continue
                    pending.append(
                        (page, image_path, image_sha, embedding_name, embedding_path, current)
                    )
                except Exception as exc:  # retain other pages and durable progress
                    failed += 1
                    errors.append({"page_id": page.page_id, "error": f"{type(exc).__name__}: {exc}"})

            if pending:
                batch_encoder = getattr(self.encoder, "encode_images", None)
                try:
                    if callable(batch_encoder):
                        embeddings = batch_encoder([item[1] for item in pending])
                    else:
                        embeddings = [self.encoder.encode_image(item[1]) for item in pending]
                    if len(embeddings) != len(pending):
                        raise VisualRetrievalUnavailable(
                            "visual encoder returned an unexpected page batch length"
                        )
                except Exception as exc:
                    failed += len(pending)
                    errors.extend(
                        {
                            "page_id": item[0].page_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        for item in pending
                    )
                    embeddings = []

                for item, embedding in zip(pending, embeddings, strict=False):
                    page, image_path, image_sha, embedding_name, embedding_path, current = item
                    try:
                        temporary = embedding_path.with_suffix(".safetensors.tmp")
                        try:
                            save_file(
                                {"embedding": embedding.to(dtype=_torch_float16()).contiguous()},
                                str(temporary),
                            )
                            temporary.replace(embedding_path)
                        finally:
                            temporary.unlink(missing_ok=True)
                        old_name = (
                            current.get("embedding_file")
                            if isinstance(current, dict)
                            else None
                        )
                        entries[page.page_id] = {
                            **asdict(page),
                            "image_path": str(image_path),
                            "image_sha256": image_sha,
                            "embedding_file": embedding_name,
                            "model_fingerprint": self.model_fingerprint,
                        }
                        self._write_manifest(manifest)
                        if old_name and old_name != embedding_name:
                            self._safe_unlink_embedding(str(old_name))
                        indexed += 1
                    except Exception as exc:
                        failed += 1
                        errors.append(
                            {
                                "page_id": page.page_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            self._write_manifest(manifest)
        return {
            "ok": failed == 0,
            "status": "complete" if failed == 0 else "partial",
            "provider": self.name,
            "model_fingerprint": self.model_fingerprint,
            "requested_pages": len(pages),
            "indexed_pages": indexed,
            "reused_pages": reused,
            "failed_pages": failed,
            "errors": errors,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    async def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[VisualSearchResult]:
        return await asyncio.to_thread(
            self._search_sync,
            query,
            document_ids=document_ids,
            revision_ids=revision_ids,
            top_k=top_k,
        )

    async def get_pages(
        self,
        *,
        page_numbers: list[int],
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
    ) -> list[VisualSearchResult]:
        """Resolve explicit page/slide hints from the committed manifest.

        An exact page reference is a lexical constraint, not a semantic
        similarity query. Resolve it without another model call and keep the
        result revision-scoped; visual ranking still handles queries without a
        page hint.
        """

        return await asyncio.to_thread(
            self._get_pages_sync,
            page_numbers=page_numbers,
            document_ids=document_ids,
            revision_ids=revision_ids,
        )

    def _get_pages_sync(
        self,
        *,
        page_numbers: list[int],
        document_ids: list[str] | None,
        revision_ids: list[str] | None,
    ) -> list[VisualSearchResult]:
        requested = list(dict.fromkeys(int(value) for value in page_numbers if int(value) > 0))
        if not requested:
            return []
        allowed_documents = set(document_ids or [])
        allowed_revisions = set(revision_ids or [])
        wanted = set(requested)
        with self._lock:
            entries = list(self._load_manifest().get("entries", {}).values())
        candidates = {
            (str(entry.get("document_id") or ""), int(entry.get("page_number") or 0)): entry
            for entry in entries
            if (not allowed_documents or entry.get("document_id") in allowed_documents)
            and (not allowed_revisions or entry.get("revision_id") in allowed_revisions)
            and entry.get("model_fingerprint") == self.model_fingerprint
            and int(entry.get("page_number") or 0) in wanted
            and Path(str(entry.get("image_path") or "")).is_file()
        }
        results: list[VisualSearchResult] = []
        for document_id in dict.fromkeys(str(value) for value in (document_ids or ["-"])):
            for page_number in requested:
                entry = candidates.get((document_id, page_number))
                if entry is None:
                    continue
                results.append(
                    VisualSearchResult(
                        page_id=str(entry["page_id"]),
                        document_id=str(entry["document_id"]),
                        revision_id=str(entry["revision_id"]),
                        page_number=page_number,
                        score=1.0,
                        image_path=str(entry["image_path"]),
                        metadata={
                            **dict(entry.get("metadata") or {}),
                            "provider": self.name,
                            "model_fingerprint": self.model_fingerprint,
                            "explicit_page_hint": True,
                        },
                    )
                )
        if document_ids is None:
            for page_number in requested:
                for (document_id, candidate_page), entry in sorted(candidates.items()):
                    if candidate_page != page_number:
                        continue
                    if any(item.page_id == str(entry["page_id"]) for item in results):
                        continue
                    results.append(
                        VisualSearchResult(
                            page_id=str(entry["page_id"]),
                            document_id=document_id,
                            revision_id=str(entry["revision_id"]),
                            page_number=page_number,
                            score=1.0,
                            image_path=str(entry["image_path"]),
                            metadata={
                                **dict(entry.get("metadata") or {}),
                                "provider": self.name,
                                "model_fingerprint": self.model_fingerprint,
                                "explicit_page_hint": True,
                            },
                        )
                    )
        return results

    def _search_sync(
        self,
        query: str,
        *,
        document_ids: list[str] | None,
        revision_ids: list[str] | None,
        top_k: int,
    ) -> list[VisualSearchResult]:
        from safetensors.torch import load_file

        if top_k < 1 or not query.strip():
            return []
        allowed_documents = set(document_ids or [])
        allowed_revisions = set(revision_ids or [])
        with self._lock:
            entries = list(self._load_manifest().get("entries", {}).values())
        candidates = [
            entry
            for entry in entries
            if (not allowed_documents or entry.get("document_id") in allowed_documents)
            and (not allowed_revisions or entry.get("revision_id") in allowed_revisions)
            and entry.get("model_fingerprint") == self.model_fingerprint
            and Path(str(entry.get("image_path") or "")).is_file()
        ]
        candidates.sort(key=lambda item: str(item.get("page_id") or ""))
        if not candidates:
            return []
        query_embedding = self.encoder.encode_query(query)
        # Load all valid page embeddings once and let the encoder/processor
        # perform bounded internal tiling.  The old outer loop invoked
        # ``score`` once per 16-page slice, repeatedly padding and dispatching
        # the same query; that made a long PDF appear hung on CPU while also
        # producing the same exact scores.
        embeddings = []
        valid_entries = []
        for entry in candidates:
            embedding_path = self._safe_embedding_path(
                str(entry.get("embedding_file") or "")
            )
            if not embedding_path.is_file():
                continue
            embeddings.append(load_file(str(embedding_path), device="cpu")["embedding"].float())
            valid_entries.append(entry)
        scored = [
            (score, entry)
            for score, entry in zip(
                self.encoder.score(query_embedding, embeddings),
                valid_entries,
                strict=True,
            )
        ]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("page_id") or "")))
        return [
            VisualSearchResult(
                page_id=str(entry["page_id"]),
                document_id=str(entry["document_id"]),
                revision_id=str(entry["revision_id"]),
                page_number=int(entry["page_number"]),
                score=float(score),
                image_path=str(entry["image_path"]),
                metadata={
                    **dict(entry.get("metadata") or {}),
                    "provider": self.name,
                    "model_fingerprint": self.model_fingerprint,
                },
            )
            for score, entry in scored[:top_k]
        ]

    async def delete_document(self, document_id: str, revision_id: str | None = None) -> None:
        await asyncio.to_thread(self._delete_document_sync, document_id, revision_id)

    def _delete_document_sync(self, document_id: str, revision_id: str | None) -> None:
        with self._lock:
            manifest = self._load_manifest()
            entries = manifest.setdefault("entries", {})
            removed = [
                page_id
                for page_id, entry in entries.items()
                if entry.get("document_id") == document_id
                and (revision_id is None or entry.get("revision_id") == revision_id)
            ]
            for page_id in removed:
                entry = entries.pop(page_id)
                self._safe_unlink_embedding(str(entry.get("embedding_file") or ""))
            self._write_manifest(manifest)

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.is_file():
            return {
                "schema_version": "colsmol-page-index-v1",
                "provider": self.name,
                "model_fingerprint": self.model_fingerprint,
                "entries": {},
            }
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise VisualRetrievalUnavailable(f"visual manifest is unreadable: {exc}") from exc
        if value.get("model_fingerprint") != self.model_fingerprint:
            raise VisualRetrievalUnavailable("visual index model fingerprint mismatch")
        if not isinstance(value.get("entries"), dict):
            raise VisualRetrievalUnavailable("visual manifest entries are invalid")
        return value

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest.update(
            {
                "schema_version": "colsmol-page-index-v1",
                "provider": self.name,
                "model_fingerprint": self.model_fingerprint,
            }
        )
        _atomic_json_write(self._manifest_path, manifest)

    def _safe_unlink_embedding(self, filename: str) -> None:
        if not filename:
            return
        candidate = self._safe_embedding_path(filename)
        candidate.unlink(missing_ok=True)

    def _safe_embedding_path(self, filename: str) -> Path:
        candidate = (self._embedding_root / filename).resolve()
        if candidate.parent != self._embedding_root.resolve():
            raise VisualRetrievalUnavailable("refusing unsafe visual embedding path")
        return candidate


def _torch_float16() -> Any:
    import torch

    return torch.float16


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
