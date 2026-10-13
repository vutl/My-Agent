"""Async client for Aya's dependency-isolated visual retrieval worker."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import uuid
from typing import Any, Callable

from app.rag.colsmol_retrieval import (
    DEFAULT_COLSMOL_BASE_MODEL,
    DEFAULT_COLSMOL_BASE_REVISION,
    DEFAULT_COLSMOL_MODEL,
    DEFAULT_COLSMOL_REVISION,
    VisualRetrievalUnavailable,
    _fingerprint,
)
from app.rag.visual_retrieval import VisualPageRecord, VisualSearchResult


def resolve_colsmol_image_batch_pixels(value: str | int | None = None) -> int:
    """Return the CPU-safe aggregate image budget for one forward pass."""

    try:
        requested = int(
            value
            if value is not None
            else os.getenv("COLSMOL_IMAGE_BATCH_MAX_PIXELS", "4000000")
        )
    except (TypeError, ValueError):
        requested = 4_000_000
    return max(262_144, min(requested, 32_000_000))


def plan_colsmol_page_batches(
    pages: list[VisualPageRecord],
    *,
    max_pages: int,
    max_pixels: int,
) -> list[list[VisualPageRecord]]:
    """Batch rendered pages by both count and source pixel footprint.

    A PDF's media box is untrusted.  Eight ordinary A4 pages and eight large
    slides have radically different pre-processing/forward memory costs, so a
    count-only worker batch is insufficient on a CPU-only host.  A single
    oversized page remains indexable by itself instead of poisoning the rest
    of the document's resumable batch.
    """

    if max_pages < 1 or max_pixels < 1:
        raise ValueError("max_pages and max_pixels must be positive")
    planned: list[list[VisualPageRecord]] = []
    current: list[VisualPageRecord] = []
    current_pixels = 0
    for page in pages:
        pixels = 0
        try:
            from PIL import Image

            with Image.open(Path(page.image_path)) as image:
                width, height = image.size
                pixels = max(1, int(width) * int(height))
        except Exception:
            # Preserve the provider's detailed per-page file error instead of
            # failing planning before the worker sees the page.
            pixels = 0
        would_exceed = current and (
            len(current) >= max_pages
            or (pixels > 0 and current_pixels + pixels > max_pixels)
        )
        if would_exceed:
            planned.append(current)
            current = []
            current_pixels = 0
        current.append(page)
        current_pixels += pixels
        if len(current) >= max_pages or (pixels > 0 and pixels >= max_pixels):
            planned.append(current)
            current = []
            current_pixels = 0
    if current:
        planned.append(current)
    return planned


class ColSmolWorkerClient:
    """Keep one ColSmol model alive without importing Transformers 5 in Aya."""

    name = "colsmol_worker"

    def __init__(
        self,
        *,
        python_executable: Path,
        index_root: Path,
        model_name: str = DEFAULT_COLSMOL_MODEL,
        model_revision: str = DEFAULT_COLSMOL_REVISION,
        base_model_name: str = DEFAULT_COLSMOL_BASE_MODEL,
        base_model_revision: str = DEFAULT_COLSMOL_BASE_REVISION,
        device: str = "auto",
        local_files_only: bool = False,
        backend_root: Path | None = None,
        worker_module: str = "app.rag.visual_worker",
        response_timeout_seconds: float = 3_600.0,
        max_pages_per_request: int = 8,
        max_image_pixels_per_request: int | None = None,
    ) -> None:
        # Do not resolve the venv's ``bin/python`` symlink: resolving it to the
        # base interpreter silently drops that venv's site-packages.
        self.python_executable = Path(
            os.path.abspath(str(python_executable.expanduser()))
        )
        self.index_root = index_root.expanduser().resolve()
        self.model_name = model_name
        self.model_revision = model_revision
        self.base_model_name = base_model_name
        self.base_model_revision = base_model_revision
        self.device = device
        # ``auto`` may select MPS on Apple Silicon, but some ColSmol/torch
        # combinations abort the child process (SIGABRT) before they can
        # return a structured error.  Keep the model and index identical and
        # downgrade only that device failure to CPU so a benchmark can resume
        # safely.  Explicit ``mps`` remains fail-closed: callers asked for it.
        self._effective_device = device
        self._device_fallback_reason: str | None = None
        self.local_files_only = local_files_only
        self.backend_root = (
            backend_root.expanduser().resolve()
            if backend_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.worker_module = worker_module
        self.response_timeout_seconds = response_timeout_seconds
        if max_pages_per_request < 1:
            raise ValueError("max_pages_per_request must be positive")
        # Keep the NDJSON request bounded.  A long PDF can contain hundreds of
        # rendered pages; sending one giant JSON line makes pipe back-pressure
        # look like a hung worker and prevents durable progress checkpoints.
        self.max_pages_per_request = max_pages_per_request
        self.max_image_pixels_per_request = resolve_colsmol_image_batch_pixels(
            max_image_pixels_per_request
        )
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
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        # Cancellation of an async request cannot cancel the synchronous
        # ``select`` call running in ``to_thread``.  Mark the client closed
        # before terminating the child so that the still-unwinding request
        # cannot respawn a worker while the evaluator is shutting down.
        self._closing = False

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._closing:
            raise VisualRetrievalUnavailable("visual worker client is closed")
        process = self._process
        if process is not None and process.poll() is None:
            return process
        if not self.python_executable.is_file():
            raise VisualRetrievalUnavailable(
                f"visual worker Python is missing: {self.python_executable}; "
                "create backend/.venv-visual and install requirements-visual-worker.txt"
            )
        environment = dict(os.environ)
        current_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.backend_root), current_path) if part
        )
        command = [
            str(self.python_executable),
            "-m",
            self.worker_module,
            "--index-root",
            str(self.index_root),
            "--model",
            self.model_name,
            "--model-revision",
            self.model_revision,
            "--base-model",
            self.base_model_name,
            "--base-revision",
            self.base_model_revision,
            "--device",
            self._effective_device,
        ]
        if self.local_files_only:
            command.append("--local-files-only")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            cwd=str(self.backend_root.parent),
            env=environment,
        )
        return self._process

    def _request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            for attempt in range(2):
                process = self._ensure_process()
                if process.stdin is None or process.stdout is None:
                    raise VisualRetrievalUnavailable("visual worker pipes are unavailable")
                request_id = uuid.uuid4().hex
                process.stdin.write(
                    json.dumps({"id": request_id, "op": operation, **payload}, ensure_ascii=False)
                    + "\n"
                )
                process.stdin.flush()
                ready, _, _ = select.select(
                    [process.stdout], [], [], self.response_timeout_seconds
                )
                if not ready:
                    process.terminate()
                    self._process = None
                    raise VisualRetrievalUnavailable(
                        f"visual worker timed out after {self.response_timeout_seconds:.1f}s"
                    )
                response_line = process.stdout.readline()
                if not response_line:
                    code = process.poll()
                    self._process = None
                    if (
                        attempt == 0
                        and operation != "shutdown"
                        and self.device == "auto"
                        and self._effective_device == "auto"
                        # SIGABRT (macOS often reports -6/134) and SIGSEGV
                        # (-11) are both observed when an optional MPS
                        # backend cannot materialize ColSmol.  Retry once on
                        # CPU; explicit non-auto devices remain fail-closed.
                        and code in {-6, 134, -11, 139}
                    ):
                        self._effective_device = "cpu"
                        self._device_fallback_reason = (
                            f"auto device worker aborted (exit={code}); retried on cpu"
                        )
                        continue
                    raise VisualRetrievalUnavailable(
                        f"visual worker exited before response (exit={code})"
                    )
                try:
                    response = json.loads(response_line)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise VisualRetrievalUnavailable(
                        f"visual worker returned invalid NDJSON: {response_line[:200]!r}"
                    ) from exc
                if response.get("id") != request_id:
                    raise VisualRetrievalUnavailable("visual worker response id mismatch")
                if not response.get("ok"):
                    raise VisualRetrievalUnavailable(
                        f"{response.get('error_type')}: {response.get('error')}"
                    )
                return response.get("result")
            raise VisualRetrievalUnavailable("visual worker retry budget exhausted")

    async def health(self) -> dict[str, Any]:
        result = await asyncio.to_thread(self._request, "health")
        if isinstance(result, dict):
            # The worker resolves conservative ``auto`` itself.  Mirror that
            # resolved value in parent telemetry so reports do not claim the
            # process ran on an unspecified device.
            worker_device = str(result.get("effective_device") or "").strip()
            if worker_device in {"cpu", "mps"}:
                self._effective_device = worker_device
            result = {
                **result,
                "client_requested_device": self.device,
                "client_effective_device": self._effective_device,
                "device_fallback_reason": self._device_fallback_reason,
            }
        return result

    async def index_pages(
        self,
        pages: list[VisualPageRecord],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not pages:
            return await asyncio.to_thread(self._request, "index", pages=[])

        # Index in small, ordered batches.  The worker commits each batch's
        # manifest atomically, so an interruption can resume from the last
        # completed batch without replaying an unbounded request.
        aggregate: dict[str, Any] = {
            "ok": True,
            "status": "complete",
            "provider": self.name,
            "model_fingerprint": self.model_fingerprint,
            "requested_pages": len(pages),
            "indexed_pages": 0,
            "reused_pages": 0,
            "failed_pages": 0,
            "errors": [],
            "batches": 0,
            "elapsed_ms": 0.0,
            "max_image_pixels_per_request": self.max_image_pixels_per_request,
        }
        started = asyncio.get_running_loop().time()
        for batch in plan_colsmol_page_batches(
            pages,
            max_pages=self.max_pages_per_request,
            max_pixels=self.max_image_pixels_per_request,
        ):
            result = await asyncio.to_thread(
                self._request, "index", pages=[asdict(page) for page in batch]
            )
            if not isinstance(result, dict):
                raise VisualRetrievalUnavailable("visual worker index response is invalid")
            aggregate["batches"] += 1
            aggregate["indexed_pages"] += int(result.get("indexed_pages") or 0)
            aggregate["reused_pages"] += int(result.get("reused_pages") or 0)
            aggregate["failed_pages"] += int(result.get("failed_pages") or 0)
            aggregate["errors"].extend(result.get("errors") or [])
            if not result.get("ok", False):
                aggregate["ok"] = False
                aggregate["status"] = "partial"
            if progress_callback is not None:
                # Callbacks are intentionally best-effort telemetry. A broken
                # status sink must never invalidate a successfully committed
                # visual batch or turn a local enrichment job into a retry.
                try:
                    progress_callback(
                        {
                            **aggregate,
                            "processed_pages": min(
                                len(pages),
                                int(aggregate["indexed_pages"])
                                + int(aggregate["reused_pages"])
                                + int(aggregate["failed_pages"]),
                            ),
                            "remaining_pages": max(
                                0,
                                len(pages)
                                - int(aggregate["indexed_pages"])
                                - int(aggregate["reused_pages"])
                                - int(aggregate["failed_pages"]),
                            ),
                        }
                    )
                except Exception:
                    pass
        aggregate["elapsed_ms"] = round(
            (asyncio.get_running_loop().time() - started) * 1000, 3
        )
        return aggregate

    async def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[VisualSearchResult]:
        values = await asyncio.to_thread(
            self._request,
            "search",
            query=query,
            document_ids=document_ids,
            revision_ids=revision_ids,
            top_k=top_k,
        )
        return [VisualSearchResult(**value) for value in values or []]

    async def get_pages(
        self,
        *,
        page_numbers: list[int],
        document_ids: list[str] | None = None,
        revision_ids: list[str] | None = None,
    ) -> list[VisualSearchResult]:
        values = await asyncio.to_thread(
            self._request,
            "get_pages",
            page_numbers=[int(value) for value in page_numbers],
            document_ids=document_ids,
            revision_ids=revision_ids,
        )
        return [VisualSearchResult(**value) for value in values or []]

    async def delete_document(self, document_id: str, revision_id: str | None = None) -> None:
        await asyncio.to_thread(
            self._request,
            "delete",
            document_id=document_id,
            revision_id=revision_id,
        )

    def close(self) -> None:
        # Do not acquire ``_lock`` here.  It is deliberately held around the
        # blocking pipe request, so waiting for it would defeat evaluator
        # timeouts when a worker is stuck encoding a large page batch.  Kill
        # the child directly; any in-flight request thread observes EOF and
        # exits without being able to spawn a replacement (``_closing``).
        self._closing = True
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def abort(self) -> None:
        """Terminate the current worker but keep the client reusable.

        Evaluators and optional background enrichment may time-box one visual
        request without making the provider permanently unavailable.  This is
        deliberately separate from :meth:`close`: an aborted request can be
        retried with a fresh worker, while ``close`` remains terminal during
        shutdown.
        """

        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self) -> "ColSmolWorkerClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
