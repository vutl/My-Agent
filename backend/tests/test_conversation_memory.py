"""L2 conversation memory tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.db.sqlite import init_db
from app.services.chat_history import ChatHistory
from app.services.conversation_memory import (
    ConversationMemoryStore,
    fold_conversation_summary,
    schedule_memory_fold,
)
from app.services.conversation_state import ConversationStateStore, ConversationWorkingState


class _FakeLLM:
    async def chat(self, **kwargs):
        class _C:
            message = (
                "User goal / thread: discussing ASPIRE paper, interrupted by pizza joke.\n"
                "Papers/files: ASPIRE.pdf\n"
                "Side topics: user asked about pineapple pizza preference.\n"
                "Open threads: resume ASPIRE benchmarks later.\n"
            )

        return _C()


def test_recent_beats_keep_last_three(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    cid = history.ensure_conversation(None, "hello")

    for i in range(4):
        store.append_turn_beat(
            cid,
            user_text=f"user turn {i}",
            assistant_text=f"assistant turn {i}",
        )

    memory = store.get_memory(cid)
    assert len(memory.recent_beats) == 3
    assert memory.recent_beats[0].user == "user turn 1"
    assert memory.recent_beats[-1].user == "user turn 3"
    block = memory.prompt_block()
    assert "Recent turn notes" in block
    assert "user turn 0" not in block


def test_fold_summary_preserves_side_topic(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    cid = history.ensure_conversation(None, "ASPIRE")

    memory = asyncio.run(
        fold_conversation_summary(
            store=store,
            conversation_id=cid,
            client=_FakeLLM(),
            model="test",
            user_text="thích pizza pineapple không?",
            assistant_text="Có nha, tùy người.",
            working_topic="aspire",
            working_filenames=["ASPIRE.pdf"],
        )
    )
    assert memory.summary is not None
    assert "pizza" in memory.summary.lower()
    assert "aspire" in memory.summary.lower()


def test_schedule_memory_fold_writes_beat_immediately(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    cid = history.ensure_conversation(None, "ASPIRE")

    async def _run() -> None:
        schedule_memory_fold(
            store=store,
            conversation_id=cid,
            client=_FakeLLM(),
            model="test",
            user_text="random: thích trà đá không?",
            assistant_text="Thích.",
            working_topic="aspire",
            working_filenames=["ASPIRE.pdf"],
        )
        # beat is sync; summary fold is async
        memory = store.get_memory(cid)
        assert len(memory.recent_beats) == 1
        assert "trà đá" in memory.recent_beats[0].user
        await asyncio.sleep(0.05)
        memory = store.get_memory(cid)
        assert memory.summary and "aspire" in memory.summary.lower()

    asyncio.run(_run())


def test_l1_focus_survives_casual_while_l2_records_digression(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    state = ConversationStateStore(db_path)
    memory = ConversationMemoryStore(db_path)
    cid = history.ensure_conversation(None, "ASPIRE")

    state.set_working_state(
        cid,
        ConversationWorkingState(
            active_document_ids=["doc-aspire"],
            active_topic="aspire",
            active_filenames=["ASPIRE.pdf"],
        ),
    )
    memory.append_turn_beat(
        cid,
        user_text="architecture ASPIRE",
        assistant_text="ASPIRE uses AV guidance",
    )
    memory.append_turn_beat(
        cid,
        user_text="btw hôm nay ăn gì?",
        assistant_text="Phở cũng được.",
    )

    working = state.get_working_state(cid)
    mem = memory.get_memory(cid)
    assert working.active_topic == "aspire"
    assert working.active_document_ids == ["doc-aspire"]
    assert any("ăn gì" in beat.user for beat in mem.recent_beats)
    packed = "\n\n".join(
        part
        for part in (working.prompt_block(), mem.prompt_block())
        if part
    )
    assert "ASPIRE.pdf" in packed
    assert "ăn gì" in packed


def test_out_of_order_l2_fold_cannot_overwrite_newer_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationMemoryStore(db_path)
    cid = history.ensure_conversation(None, "ASPIRE")
    older = store.append_turn_beat(
        cid,
        user_text="older A",
        assistant_text="answer A",
    )
    newer = store.append_turn_beat(
        cid,
        user_text="newer B",
        assistant_text="answer B",
    )

    class _DelayedLLM:
        def __init__(self, delay: float, summary: str) -> None:
            self.delay = delay
            self.summary = summary

        async def chat(self, **kwargs):  # noqa: ANN003
            await asyncio.sleep(self.delay)

            class _C:
                pass

            completion = _C()
            completion.message = self.summary
            return completion

    async def _run() -> None:
        await asyncio.gather(
            fold_conversation_summary(
                store=store,
                conversation_id=cid,
                client=_DelayedLLM(0.05, "OLDER_A"),
                model="test",
                user_text="older A",
                assistant_text="answer A",
                revision=older.revision,
            ),
            fold_conversation_summary(
                store=store,
                conversation_id=cid,
                client=_DelayedLLM(0.0, "NEWER_B_INCLUDES_A"),
                model="test",
                user_text="newer B",
                assistant_text="answer B",
                revision=newer.revision,
            ),
        )

    asyncio.run(_run())
    memory = store.get_memory(cid)
    assert memory.summary == "NEWER_B_INCLUDES_A"
    assert memory.summary_revision == newer.revision
