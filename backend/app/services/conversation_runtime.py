"""In-process foreground turn coordination for conversation runtimes.

The gate deliberately owns no persistence and does not start background work.
It gives API/runtime code one place to serialize turns for a conversation and
gives lower-priority workers a process-wide foreground busy/idle signal.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import inspect
import logging
from typing import Any


RuntimeHook = Callable[[], Awaitable[None] | None]

_LOGGER = logging.getLogger(__name__)


@dataclass
class _ConversationEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    references: int = 0


class ConversationRuntimeGate:
    """Serialize foreground turns per conversation and advertise global idle.

    ``foreground_active_count`` includes both lock holders and foreground turns
    waiting for the same conversation. Treating waiters as active prevents an
    idle pulse between two already-queued turns, which would otherwise let a
    lower-priority L2 fold start in the middle of a foreground burst.

    The object is intended for one asyncio event loop, matching one backend
    process. Cross-process coordination belongs in a durable store, not here.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ConversationEntry] = {}
        self._foreground_active_count = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._busy_hooks: dict[int, RuntimeHook] = {}
        self._idle_hooks: dict[int, RuntimeHook] = {}
        self._next_hook_id = 1
        self._hook_tasks: set[asyncio.Task[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def foreground_active_count(self) -> int:
        """Number of foreground turns currently running or waiting for a lock."""
        return self._foreground_active_count

    @property
    def is_foreground_idle(self) -> bool:
        return self._foreground_active_count == 0

    @property
    def tracked_conversation_count(self) -> int:
        """Number of live per-conversation lock entries (diagnostics/tests)."""
        return len(self._entries)

    async def wait_until_idle(self) -> None:
        """Wait until no foreground turn is running or queued."""
        self._bind_loop()
        await self._idle.wait()

    def add_busy_callback(self, callback: RuntimeHook) -> Callable[[], None]:
        """Run ``callback`` on the foreground transition from idle to busy.

        The returned function unregisters the hook. Async callbacks are
        scheduled without blocking the foreground turn; callback failures are
        logged and isolated from gate bookkeeping.
        """
        return self._add_hook(self._busy_hooks, callback)

    def add_idle_callback(self, callback: RuntimeHook) -> Callable[[], None]:
        """Run ``callback`` on the foreground transition from busy to idle."""
        return self._add_hook(self._idle_hooks, callback)

    @asynccontextmanager
    async def turn(self, conversation_id: str) -> AsyncIterator[None]:
        """Acquire the foreground execution slot for ``conversation_id``.

        Different conversation IDs use different locks. Registration, active
        accounting, lock release, and reference cleanup have no await points,
        so cancellation cannot strand a lock entry during those transitions.
        """
        self._bind_loop()
        selected_id = _conversation_id(conversation_id)
        entry = self._register_turn(selected_id)
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            self._unregister_turn(selected_id, entry)

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("ConversationRuntimeGate cannot be shared across event loops")
        return loop

    def _register_turn(self, conversation_id: str) -> _ConversationEntry:
        entry = self._entries.get(conversation_id)
        if entry is None:
            entry = _ConversationEntry()
            self._entries[conversation_id] = entry
        entry.references += 1

        was_idle = self._foreground_active_count == 0
        self._foreground_active_count += 1
        if was_idle:
            self._idle.clear()
            self._notify(self._busy_hooks)
        return entry

    def _unregister_turn(
        self,
        conversation_id: str,
        entry: _ConversationEntry,
    ) -> None:
        entry.references -= 1
        if entry.references < 0:
            raise RuntimeError("conversation runtime reference count became negative")
        if entry.references == 0:
            current = self._entries.get(conversation_id)
            if current is entry:
                if entry.lock.locked():
                    raise RuntimeError("cannot remove a locked conversation runtime entry")
                del self._entries[conversation_id]

        self._foreground_active_count -= 1
        if self._foreground_active_count < 0:
            raise RuntimeError("foreground active count became negative")
        if self._foreground_active_count == 0:
            self._idle.set()
            self._notify(self._idle_hooks)

    def _add_hook(
        self,
        hooks: dict[int, RuntimeHook],
        callback: RuntimeHook,
    ) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("runtime callback must be callable")
        hook_id = self._next_hook_id
        self._next_hook_id += 1
        hooks[hook_id] = callback

        def remove() -> None:
            hooks.pop(hook_id, None)

        return remove

    def _notify(self, hooks: dict[int, RuntimeHook]) -> None:
        for callback in tuple(hooks.values()):
            try:
                result = callback()
            except Exception:
                _LOGGER.exception("conversation runtime callback failed")
                continue
            if not inspect.isawaitable(result):
                continue
            try:
                task = self._bind_loop().create_task(_await_hook(result))
            except Exception:
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                _LOGGER.exception("could not schedule conversation runtime callback")
                continue
            self._hook_tasks.add(task)
            task.add_done_callback(self._on_hook_done)

    def _on_hook_done(self, task: asyncio.Task[Any]) -> None:
        self._hook_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("asynchronous conversation runtime callback failed")


def _conversation_id(value: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError("conversation_id must not be empty")
    return selected


async def _await_hook(awaitable: Awaitable[None]) -> None:
    await awaitable
