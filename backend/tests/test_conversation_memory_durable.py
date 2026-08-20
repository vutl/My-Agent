"""Durability, coalescing, recovery, and failure tests for L2 memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re

import pytest

import app.services.conversation_memory as conversation_memory_module
from app.db.sqlite import init_db
from app.db.sqlite import connect
from app.services.agent_run_store import AgentRunStore
from app.services.chat_history import ChatHistory
from app.services.conversation_memory import (
    ConversationMemoryStore,
    FOLD_INPUT_MAX_CHARS,
    PENDING_PROMPT_MAX_CHARS,
    fold_conversation_summary_result,
    shutdown_memory_fold_coordinators,
    start_memory_fold_coordinator,
)


class _CountingLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def chat(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            content = kwargs["messages"][-1]["content"]
            turns = [int(item) for item in re.findall(r"turn (\d+)", content)]

            class _Completion:
                message = json.dumps(
                    {
                        "summary": f"summary through {max(turns, default=0)}",
                        "memory_ops": [],
                    }
                )

            return _Completion()
        finally:
            self.active -= 1


class _FailingLLM:
    def __init__(self) -> None:
        self.models: list[str] = []

    async def chat(self, **kwargs):  # noqa: ANN003
        self.models.append(kwargs["model"])
        raise RuntimeError("9router quota unavailable")


class _CancellableLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = 0

    async def chat(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        class _Completion:
            message = '{"summary":"resumed safely", "memory_ops":[]}'

        return _Completion()


class _MemoryOpsLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs):  # noqa: ANN003
        del kwargs
        self.calls += 1

        class _Completion:
            message = json.dumps(
                {
                    "summary": "User explicitly prefers concise answers.",
                    "memory_ops": [
                        {
                            "action": "upsert",
                            "scope": "user",
                            "kind": "semantic",
                            "key": "answer_style",
                            "content": "User prefers concise answers.",
                            "confidence": 0.98,
                        },
                        {
                            "action": "upsert",
                            "scope": "user",
                            "kind": "semantic",
                            "key": "api_key",
                            "content": "secret token abc",
                            "confidence": 1.0,
                        },
                    ],
                }
            )

        return _Completion()


class _SensitiveMemoryOpsLLM:
    async def chat(self, **kwargs):  # noqa: ANN003
        del kwargs

        class _Completion:
            message = json.dumps(
                {
                    "summary": "User asked to forget an old credential.",
                    "memory_ops": [
                        {
                            "action": "forget",
                            "scope": "user",
                            "kind": "semantic",
                            "key": "api_key",
                            "content": "sk-oldcredential123456789",
                            "confidence": 1.0,
                        },
                        {
                            "action": "upsert",
                            "scope": "user",
                            "kind": "semantic",
                            "key": "innocent_note",
                            "content": "Token was sk-livecredential123456789",
                            "confidence": 1.0,
                        },
                    ],
                }
            )

        return _Completion()


async def _wait_until(predicate, *, timeout: float = 1.5) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def _conversation(tmp_path: Path, first_message: str = "hello") -> tuple[ChatHistory, ConversationMemoryStore, str]:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    conversation_id = history.ensure_conversation(None, first_message)
    return history, ConversationMemoryStore(db_path), conversation_id


def test_all_pending_turns_are_durable_beyond_recent_beat_ring(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    for index in range(1, 8):
        store.record_completed_turn(
            conversation_id,
            user_text=f"user sentence {index}. second sentence stays whole.",
            assistant_text=f"assistant answer {index}.",
        )

    memory = store.get_memory(conversation_id)
    assert [turn.turn_seq for turn in memory.pending_turns] == list(range(1, 8))
    assert [beat.revision for beat in memory.recent_beats] == [5, 6, 7]
    block = memory.prompt_block()
    assert "turn 1 user" in block
    assert "turn 7 assistant" in block
    # Pending full turns replace their duplicate clipped beats in the prompt.
    assert "Recent turn notes" not in block
    job = store.get_job(conversation_id)
    assert job is not None
    assert (job.dirty_through_seq, job.summary_through_seq) == (7, 0)


def test_coordinator_coalesces_and_globally_serializes_conversations(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    first = history.ensure_conversation(None, "first")
    second = history.ensure_conversation(None, "second")
    for conversation_id in (first, second):
        for index in range(1, 5):
            store.record_completed_turn(
                conversation_id,
                user_text=f"question {index}",
                assistant_text=f"answer {index}",
            )
    client = _CountingLLM()

    async def _run() -> None:
        coordinator = start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            debounce_seconds=0.01,
        )
        coordinator.enqueue(first)
        coordinator.enqueue(second)
        await _wait_until(
            lambda: store.get_memory(first).summary_revision == 4
            and store.get_memory(second).summary_revision == 4
        )
        await shutdown_memory_fold_coordinators(timeout_seconds=0.2)

    asyncio.run(_run())
    assert client.max_active == 1
    assert len(client.calls) == 2
    assert store.get_memory(first).summary == "summary through 4"
    assert store.get_memory(second).summary == "summary through 4"


def test_fold_failure_stays_pending_without_heuristic_or_model_switch(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="Remember the exact decision.",
        assistant_text="Acknowledged.",
    )
    client = _FailingLLM()

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="quota unavailable"):
            await fold_conversation_summary_result(
                store=store,
                conversation_id=conversation_id,
                client=client,
                model="cx/gpt-5.5",
                revision=1,
            )

    asyncio.run(_run())
    memory = store.get_memory(conversation_id)
    job = store.get_job(conversation_id)
    assert memory.summary is None
    assert memory.summary_revision == 0
    assert [turn.turn_seq for turn in memory.pending_turns] == [1]
    assert job is not None
    assert job.status == "pending"
    assert job.attempt_count == 1
    assert "quota unavailable" in (job.last_error or "")
    assert client.models == ["cx/gpt-5.5"]


def test_fold_batches_advance_only_through_included_prefix(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    for index in range(1, 4):
        store.record_completed_turn(
            conversation_id,
            user_text=f"question {index}.",
            assistant_text=(f"answer {index}. " + ("x" * (FOLD_INPUT_MAX_CHARS // 2))),
        )
    client = _CountingLLM()

    async def _run() -> None:
        first = await fold_conversation_summary_result(
            store=store,
            conversation_id=conversation_id,
            client=client,
            model="cx/gpt-5.5",
            revision=3,
        )
        assert first.target_revision == 1
        assert first.memory.summary_revision == 1
        assert [turn.turn_seq for turn in first.memory.pending_turns] == [2, 3]
        second = await fold_conversation_summary_result(
            store=store,
            conversation_id=conversation_id,
            client=client,
            model="cx/gpt-5.5",
            revision=3,
        )
        assert second.target_revision == 2
        assert second.memory.summary_revision == 2

    asyncio.run(_run())


def test_pending_prompt_is_bounded_with_explicit_omission_marker(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    for index in range(1, 6):
        store.record_completed_turn(
            conversation_id,
            user_text=f"question {index}.",
            assistant_text=f"answer {index}. " + ("detail " * 1800),
        )
    block = store.get_memory(conversation_id).prompt_block()
    assert "older pending turn(s) omitted" in block
    assert "remain durable" in block
    assert len(block) <= PENDING_PROMPT_MAX_CHARS + 500
    assert len(store.list_turns(conversation_id)) == 5


def test_completed_run_recovery_is_idempotent_and_resolves_message_ids(tmp_path: Path) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="recover me",
        model="cx/gpt-5.5",
    )
    assistant = history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="durable answer",
        model="cx/gpt-5.5",
    )
    runs = AgentRunStore(store.db_path)
    run = runs.create_run(
        conversation_id=conversation_id,
        user_message_id=user.id,
        mode="chat",
    )
    runs.complete_run(run.id, "durable answer")

    # Agent-owned messages are never claimed by the conservative direct-chat
    # recovery scan, even if that scan happens to run first.
    assert store.recover_completed_message_pairs() == []
    assert store.recover_completed_runs() == [conversation_id]
    assert store.recover_completed_runs() == []
    turns = store.list_turns(conversation_id)
    assert len(turns) == 1
    assert turns[0].user_message_id == user.id
    assert turns[0].assistant_message_id == assistant.id
    assert turns[0].user_text == "recover me"
    assert turns[0].assistant_text == "durable answer"
    # This conversation was created by the durable-memory runtime.  A missing
    # queue row is therefore a real finalization/enqueue crash gap, not legacy
    # backlog, and startup must retry it without waiting for another user turn.
    assert store.get_job(conversation_id).status == "pending"
    assert conversation_id in store.list_pending_conversation_ids()


def test_direct_chat_crash_gap_is_recovered_and_folded_on_startup(
    tmp_path: Path,
) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="direct question",
        model="cx/gpt-5.5",
    )
    assistant = history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="direct answer",
        model="cx/gpt-5.5",
    )
    client = _CountingLLM()

    async def _run() -> None:
        start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            debounce_seconds=0.0,
        )
        await _wait_until(
            lambda: store.get_memory(conversation_id).summary_revision == 1
        )
        await shutdown_memory_fold_coordinators(timeout_seconds=0.2)

    asyncio.run(_run())
    turns = store.list_turns(conversation_id)
    assert len(turns) == 1
    assert turns[0].user_message_id == user.id
    assert turns[0].assistant_message_id == assistant.id
    assert store.get_job(conversation_id).status == "idle"
    assert client.calls and len(client.calls) == 1
    assert store.recover_completed_message_pairs() == []


def test_direct_chat_recovery_does_not_pair_across_intervening_roles(
    tmp_path: Path,
) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    orphan_user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="orphan user",
        model="cx/gpt-5.5",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="system",
        content="intervening system message",
        model="cx/gpt-5.5",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="must not pair across system",
        model="cx/gpt-5.5",
    )
    first_repeated_user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="first repeated user",
        model="cx/gpt-5.5",
    )
    adjacent_user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="adjacent user",
        model="cx/gpt-5.5",
    )
    adjacent_assistant = history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="adjacent answer",
        model="cx/gpt-5.5",
    )

    assert store.recover_completed_message_pairs() == [conversation_id]
    turns = store.list_turns(conversation_id)
    assert len(turns) == 1
    assert turns[0].user_message_id == adjacent_user.id
    assert turns[0].assistant_message_id == adjacent_assistant.id
    assert turns[0].user_message_id not in {
        orphan_user.id,
        first_repeated_user.id,
    }
    assert store.recover_completed_message_pairs() == []


def test_unmarked_legacy_direct_pairs_remain_dormant(tmp_path: Path) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    with connect(store.db_path) as connection:
        connection.execute(
            "UPDATE conversations SET metadata_json = NULL WHERE id = ?",
            (conversation_id,),
        )
    history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="legacy direct question",
        model="cx/gpt-5.5",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="legacy direct answer",
        model="cx/gpt-5.5",
    )

    assert store.recover_completed_message_pairs() == [conversation_id]
    assert store.get_job(conversation_id).status == "dormant"
    assert conversation_id not in store.list_pending_conversation_ids()
    assert store.recover_completed_message_pairs() == []


def test_unmarked_legacy_recovery_stays_dormant_until_a_new_turn(tmp_path: Path) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    with connect(store.db_path) as connection:
        connection.execute(
            "UPDATE conversations SET metadata_json = NULL WHERE id = ?",
            (conversation_id,),
        )
    user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="legacy question",
        model="cx/gpt-5.5",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="legacy answer",
        model="cx/gpt-5.5",
    )
    runs = AgentRunStore(store.db_path)
    run = runs.create_run(
        conversation_id=conversation_id,
        user_message_id=user.id,
        mode="chat",
    )
    runs.complete_run(run.id, "legacy answer")

    assert store.recover_completed_runs() == [conversation_id]
    assert store.get_job(conversation_id).status == "dormant"
    assert conversation_id not in store.list_pending_conversation_ids()

    # A real new turn wakes the lazy legacy backlog; untouched historical
    # conversations still cannot cause a bulk startup summarization storm.
    store.record_completed_turn(
        conversation_id,
        user_text="new turn",
        assistant_text="new answer",
    )
    assert store.get_job(conversation_id).status == "pending"
    assert conversation_id in store.list_pending_conversation_ids()


def test_touching_legacy_thread_marks_a_new_crash_gap_as_pending(tmp_path: Path) -> None:
    history, store, conversation_id = _conversation(tmp_path)
    with connect(store.db_path) as connection:
        connection.execute(
            "UPDATE conversations SET metadata_json = NULL WHERE id = ?",
            (conversation_id,),
        )

    # Opening this existing thread for a new foreground turn upgrades only its
    # generation marker. It does not create or enqueue legacy fold work.
    assert history.ensure_conversation(conversation_id, "resume") == conversation_id
    assert store.get_job(conversation_id) is None

    user = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="new post-migration question",
        model="cx/gpt-5.5",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="new post-migration answer",
        model="cx/gpt-5.5",
    )
    runs = AgentRunStore(store.db_path)
    run = runs.create_run(
        conversation_id=conversation_id,
        user_message_id=user.id,
        mode="chat",
    )
    runs.complete_run(run.id, "new post-migration answer")

    assert store.recover_completed_runs() == [conversation_id]
    assert store.get_job(conversation_id).status == "pending"
    assert conversation_id in store.list_pending_conversation_ids()


def test_foreground_cancellation_preserves_provider_retry_backoff(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="remember this",
        assistant_text="acknowledged",
    )
    client = _FailingLLM()

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="quota unavailable"):
            await fold_conversation_summary_result(
                store=store,
                conversation_id=conversation_id,
                client=client,
                model="cx/gpt-5.5",
                revision=1,
            )
        retry_at = store.get_job(conversation_id).next_attempt_at
        assert retry_at is not None

        coordinator = start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            debounce_seconds=0.0,
        )
        coordinator.enqueue(conversation_id)
        await asyncio.sleep(0.02)
        coordinator.global_foreground_started()
        await _wait_until(lambda: conversation_id not in coordinator._tasks)

        job = store.get_job(conversation_id)
        assert job is not None
        assert job.status == "pending"
        assert job.next_attempt_at == retry_at
        coordinator.global_foreground_finished()
        await shutdown_memory_fold_coordinators(timeout_seconds=0.0)

    asyncio.run(_run())
    # Foreground activity did not manufacture an early quota retry.
    assert client.models == ["cx/gpt-5.5"]


def test_global_foreground_cancels_other_conversation_fold_then_resumes(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    foreground_conversation = history.ensure_conversation(None, "foreground")
    background_conversation = history.ensure_conversation(None, "background")
    store.record_completed_turn(
        background_conversation,
        user_text="background question",
        assistant_text="background answer",
    )
    client = _CancellableLLM()

    async def _run() -> None:
        coordinator = start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            debounce_seconds=0.0,
        )
        coordinator.enqueue(background_conversation)
        await asyncio.wait_for(client.started.wait(), timeout=1.0)
        coordinator.foreground_started(foreground_conversation)
        await asyncio.wait_for(client.cancelled.wait(), timeout=1.0)
        assert store.get_memory(background_conversation).summary_revision == 0
        assert store.get_job(background_conversation).status == "pending"
        coordinator.foreground_finished(foreground_conversation)
        await _wait_until(
            lambda: store.get_memory(background_conversation).summary_revision == 1
        )
        await shutdown_memory_fold_coordinators(timeout_seconds=0.2)

    asyncio.run(_run())
    assert client.calls == 2
    assert store.get_memory(background_conversation).summary == "resumed safely"


def test_structured_memory_ops_are_exposed_only_after_contract_validation(tmp_path: Path) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="Tôi thích câu trả lời ngắn gọn.",
        assistant_text="Đã rõ.",
    )

    async def _run():
        return await fold_conversation_summary_result(
            store=store,
            conversation_id=conversation_id,
            client=_MemoryOpsLLM(),
            model="cx/gpt-5.5",
            revision=1,
        )

    result = asyncio.run(_run())
    assert result.output.memory_ops == (
        {
            "action": "upsert",
            "scope": "user",
            "kind": "semantic",
            "key": "answer_style",
            "content": "User prefers concise answers.",
            "confidence": 0.98,
        },
    )


def test_memory_ops_allow_secret_forget_but_never_persist_echoed_secret(
    tmp_path: Path,
) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="Quên API key cũ đi.",
        assistant_text="Đã rõ.",
    )

    async def _run():
        return await fold_conversation_summary_result(
            store=store,
            conversation_id=conversation_id,
            client=_SensitiveMemoryOpsLLM(),
            model="cx/gpt-5.5",
            revision=1,
        )

    result = asyncio.run(_run())
    assert result.output.memory_ops == (
        {
            "action": "forget",
            "scope": "user",
            "kind": "semantic",
            "key": "api_key",
            "content": "",
            "confidence": 1.0,
        },
    )
    pending = store.get_next_pending_l3_operations(conversation_id)
    assert pending is not None
    assert pending.operations == result.output.memory_ops


def test_l3_outbox_replays_after_startup_without_another_model_call(
    tmp_path: Path,
) -> None:
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="Tôi thích câu trả lời ngắn gọn.",
        assistant_text="Đã rõ.",
    )
    client = _MemoryOpsLLM()

    async def _run() -> None:
        await fold_conversation_summary_result(
            store=store,
            conversation_id=conversation_id,
            client=client,
            model="cx/gpt-5.5",
            revision=1,
        )
        pending = store.get_next_pending_l3_operations(conversation_id)
        assert pending is not None
        assert pending.source_turn_seq == 1
        assert pending.operations[0]["key"] == "answer_style"
        assert store.get_job(conversation_id).status == "idle"

        applied: list[tuple[str, int, list[dict]]] = []

        async def apply_ops(
            selected_conversation_id: str,
            source_turn_seq: int,
            operations: list[dict],
        ) -> None:
            applied.append(
                (selected_conversation_id, source_turn_seq, operations)
            )

        start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            on_memory_ops=apply_ops,
            debounce_seconds=0.0,
        )
        await _wait_until(
            lambda: store.get_next_pending_l3_operations(conversation_id) is None
        )
        await shutdown_memory_fold_coordinators(timeout_seconds=0.2)

        assert len(applied) == 1
        assert applied[0][0:2] == (conversation_id, 1)
        assert applied[0][2][0]["key"] == "answer_style"

    asyncio.run(_run())
    assert client.calls == 1
    with connect(store.db_path) as connection:
        row = connection.execute(
            """
            SELECT status, attempt_count, delivered_at
            FROM conversation_memory_l3_outbox
            WHERE conversation_id = ? AND source_turn_seq = 1
            """,
            (conversation_id,),
        ).fetchone()
    assert row["status"] == "delivered"
    assert row["attempt_count"] == 0
    assert row["delivered_at"] is not None


def test_l3_callback_failure_stays_pending_and_restart_replays_same_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        conversation_memory_module,
        "MEMORY_L3_RETRY_BASE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        conversation_memory_module,
        "MEMORY_L3_RETRY_MAX_SECONDS",
        0.05,
    )
    _, store, conversation_id = _conversation(tmp_path)
    store.record_completed_turn(
        conversation_id,
        user_text="Tôi thích câu trả lời ngắn gọn.",
        assistant_text="Đã rõ.",
    )
    client = _MemoryOpsLLM()
    failed_calls: list[int] = []

    async def fail_ops(
        selected_conversation_id: str,
        source_turn_seq: int,
        operations: list[dict],
    ) -> None:
        del selected_conversation_id, operations
        failed_calls.append(source_turn_seq)
        raise RuntimeError("temporary L3 store failure")

    async def _run_first_worker() -> None:
        coordinator = start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            on_memory_ops=fail_ops,
            debounce_seconds=0.0,
        )
        coordinator.enqueue(conversation_id)
        await _wait_until(
            lambda: (
                (pending := store.get_next_pending_l3_operations(conversation_id))
                is not None
                and pending.attempt_count == 1
            )
        )
        pending = store.get_next_pending_l3_operations(conversation_id)
        assert pending is not None
        assert pending.source_turn_seq == 1
        assert "temporary L3 store failure" in (pending.last_error or "")
        assert store.get_memory(conversation_id).summary_revision == 1
        assert store.get_job(conversation_id).status == "idle"
        await shutdown_memory_fold_coordinators(timeout_seconds=0.0)

    asyncio.run(_run_first_worker())
    assert failed_calls == [1]
    assert client.calls == 1

    replayed: list[int] = []

    async def succeed_ops(
        selected_conversation_id: str,
        source_turn_seq: int,
        operations: list[dict],
    ) -> None:
        del selected_conversation_id, operations
        replayed.append(source_turn_seq)

    async def _run_restarted_worker() -> None:
        start_memory_fold_coordinator(
            store=store,
            client=client,
            model="cx/gpt-5.5",
            on_memory_ops=succeed_ops,
            debounce_seconds=0.0,
        )
        await _wait_until(
            lambda: store.get_next_pending_l3_operations(conversation_id) is None
        )
        await shutdown_memory_fold_coordinators(timeout_seconds=0.2)

    asyncio.run(_run_restarted_worker())
    assert replayed == [1]
    # Replaying the post-commit outbox never asks GPT to summarize again.
    assert client.calls == 1
