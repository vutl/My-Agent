"""Deterministic concurrency tests for the foreground conversation gate."""

from __future__ import annotations

import asyncio

import pytest

from app.services.conversation_runtime import ConversationRuntimeGate


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=1.0)


def test_same_conversation_turns_are_serialized_without_idle_gap() -> None:
    async def _run() -> None:
        gate = ConversationRuntimeGate()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_attempting = asyncio.Event()
        second_entered = asyncio.Event()
        release_second = asyncio.Event()
        transitions: list[str] = []
        order: list[str] = []
        gate.add_busy_callback(lambda: transitions.append("busy"))
        gate.add_idle_callback(lambda: transitions.append("idle"))

        async def first() -> None:
            async with gate.turn("same"):
                order.append("first-enter")
                first_entered.set()
                await _wait(release_first)
                order.append("first-exit")

        async def second() -> None:
            await _wait(first_entered)
            second_attempting.set()
            async with gate.turn("same"):
                order.append("second-enter")
                second_entered.set()
                await _wait(release_second)
                order.append("second-exit")

        first_task = asyncio.create_task(first())
        await _wait(first_entered)
        second_task = asyncio.create_task(second())
        await _wait(second_attempting)

        assert second_entered.is_set() is False
        assert gate.foreground_active_count == 2
        assert transitions == ["busy"]

        release_first.set()
        await _wait(second_entered)
        assert order[:3] == ["first-enter", "first-exit", "second-enter"]
        assert gate.is_foreground_idle is False
        assert transitions == ["busy"]

        release_second.set()
        await asyncio.gather(first_task, second_task)
        assert order == [
            "first-enter",
            "first-exit",
            "second-enter",
            "second-exit",
        ]
        assert transitions == ["busy", "idle"]
        assert gate.foreground_active_count == 0
        assert gate.tracked_conversation_count == 0

    asyncio.run(_run())


def test_different_conversations_can_run_in_parallel() -> None:
    async def _run() -> None:
        gate = ConversationRuntimeGate()
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release = asyncio.Event()

        async def turn(conversation_id: str, entered: asyncio.Event) -> None:
            async with gate.turn(conversation_id):
                entered.set()
                await _wait(release)

        first_task = asyncio.create_task(turn("one", first_entered))
        await _wait(first_entered)
        second_task = asyncio.create_task(turn("two", second_entered))
        await _wait(second_entered)

        assert gate.foreground_active_count == 2
        assert gate.tracked_conversation_count == 2
        assert gate.is_foreground_idle is False

        release.set()
        await asyncio.gather(first_task, second_task)
        assert gate.foreground_active_count == 0
        assert gate.tracked_conversation_count == 0

    asyncio.run(_run())


def test_cancellation_cleans_waiters_holders_and_lock_entries() -> None:
    async def _run() -> None:
        gate = ConversationRuntimeGate()
        holder_entered = asyncio.Event()
        waiter_attempting = asyncio.Event()
        waiter_entered = asyncio.Event()
        never_release = asyncio.Event()

        async def holder() -> None:
            async with gate.turn("same"):
                holder_entered.set()
                await _wait(never_release)

        async def waiter() -> None:
            waiter_attempting.set()
            async with gate.turn("same"):
                waiter_entered.set()

        holder_task = asyncio.create_task(holder())
        await _wait(holder_entered)
        waiter_task = asyncio.create_task(waiter())
        await _wait(waiter_attempting)
        assert gate.foreground_active_count == 2

        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task
        assert waiter_entered.is_set() is False
        assert gate.foreground_active_count == 1
        assert gate.tracked_conversation_count == 1

        holder_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder_task
        assert gate.foreground_active_count == 0
        assert gate.tracked_conversation_count == 0
        assert gate.is_foreground_idle is True

        # A cancellation must not leave a locked orphan behind.
        async with gate.turn("same"):
            assert gate.foreground_active_count == 1
        assert gate.foreground_active_count == 0
        assert gate.tracked_conversation_count == 0

    asyncio.run(_run())


def test_idle_waiter_and_transition_callbacks_follow_foreground_lifetime() -> None:
    async def _run() -> None:
        gate = ConversationRuntimeGate()
        await asyncio.wait_for(gate.wait_until_idle(), timeout=1.0)

        entered = asyncio.Event()
        release = asyncio.Event()
        idle_wait_started = asyncio.Event()
        idle_wait_returned = asyncio.Event()
        sync_transitions: list[str] = []
        async_idle_called = asyncio.Event()
        remove_busy = gate.add_busy_callback(lambda: sync_transitions.append("busy"))
        gate.add_idle_callback(lambda: sync_transitions.append("idle"))

        async def async_idle_hook() -> None:
            async_idle_called.set()

        remove_async_idle = gate.add_idle_callback(async_idle_hook)

        async def foreground() -> None:
            async with gate.turn("conversation"):
                entered.set()
                await _wait(release)

        async def observe_idle() -> None:
            idle_wait_started.set()
            await gate.wait_until_idle()
            idle_wait_returned.set()

        foreground_task = asyncio.create_task(foreground())
        await _wait(entered)
        idle_task = asyncio.create_task(observe_idle())
        await _wait(idle_wait_started)

        assert gate.is_foreground_idle is False
        assert idle_wait_returned.is_set() is False
        assert sync_transitions == ["busy"]

        release.set()
        await asyncio.gather(foreground_task, idle_task)
        await _wait(async_idle_called)
        assert gate.is_foreground_idle is True
        assert sync_transitions == ["busy", "idle"]

        remove_busy()
        remove_async_idle()
        async with gate.turn("another"):
            pass
        assert sync_transitions == ["busy", "idle", "idle"]

    asyncio.run(_run())
