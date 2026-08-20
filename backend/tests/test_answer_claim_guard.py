import asyncio
from types import SimpleNamespace

from app.api.agent import (
    AgentRunRequest,
    _stream_graph_and_answer,
    _stream_validated_grounded_blocks,
    _stream_validated_paper_sections,
)
from app.services.agent_service import (
    AgentGraphResult,
    AgentGraphStreamEvent,
    AnswerStreamChunk,
)


class _Service:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.graph_calls = 0

    async def stream_graph_events(self, **_kwargs):
        self.graph_calls += 1
        result = AgentGraphResult(
            route="file_qa",
            mode="research",
            plan=["Answer from evidence"],
            final_prompt="Use only the retrieved evidence.",
            selected_tools=["search_local_docs"],
            retrieved_docs=[],
        )
        yield AgentGraphStreamEvent(
            "graph.completed",
            {"route": "file_qa", "mode": "research"},
            result=result,
        )

    async def stream_final_answer(self, *, prompt, **_kwargs):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        yield AnswerStreamChunk(content=answer)
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")


class _RunStore:
    def __init__(self) -> None:
        self.calls = []

    def update_plan(self, *_args):
        return None

    def record_tool_call(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="tool")


def _run_guard(
    answers: list[str],
    *,
    task: str = "Benchmark F1 của ASPIRE là bao nhiêu?",
) -> tuple[str, _Service, _RunStore, list[str]]:
    service = _Service(answers)
    store = _RunStore()
    assistant_chunks: list[str] = []

    async def consume() -> str:
        events = []
        async for event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(task=task),
            run_id="run",
            conversation_id="conversation",
            user_message_id="message",
            task=task,
            resolved_task="ASPIRE benchmark F1",
            conversation_context="",
            answer_intent="direct_answer",
            answer_depth="normal",
            retrieved_docs=[
                {
                    "document_id": "aspire",
                    "filename": "ASPIRE.pdf",
                    "content": "ASPIRE reports F1 80.4%.",
                }
            ],
            tool_decision={"route": "file_qa"},
            assistant_chunks=assistant_chunks,
            focus_document_ids=["aspire"],
            validate_quantitative_claims=True,
        ):
            events.append(event)
        return "".join(events)

    return asyncio.run(consume()), service, store, assistant_chunks


def test_claim_guard_retries_before_exposing_unsupported_number() -> None:
    events, service, store, chunks = _run_guard(
        ["ASPIRE đạt F1 91.7%.", "Evidence hiện chỉ hỗ trợ F1 80.4%."]
    )

    assert len(service.prompts) == 2
    assert "Validation reason: unsupported_metric_values" in service.prompts[1]
    assert "MODEL — METRIC: VALUE" in service.prompts[1]
    assert "91.7" not in "".join(chunks)
    assert "80.4" in "".join(chunks)
    assert "91.7" not in events
    assert store.calls[-1]["tool_name"] == "validate_answer_claims"
    assert store.calls[-1]["output_payload"]["fallback_used"] is False
    attempts = store.calls[-1]["output_payload"]["attempt_validations"]
    assert [item["valid"] for item in attempts] == [False, True]
    assert attempts[0]["unsupported_claims"][0]["value"] == "91.7"


def test_claim_guard_uses_number_free_fallback_after_failed_retry() -> None:
    events, _service, store, chunks = _run_guard(
        ["ASPIRE đạt F1 91.7%.", "ASPIRE đạt F1 92.1%."]
    )

    answer = "".join(chunks)
    assert "91.7" not in answer and "92.1" not in answer
    assert "chưa có đủ evidence" in answer
    assert "91.7" not in events and "92.1" not in events
    assert store.calls[-1]["output_payload"]["fallback_used"] is True
    attempts = store.calls[-1]["output_payload"]["attempt_validations"]
    assert [item["valid"] for item in attempts] == [False, False]
    assert attempts[1]["unsupported_claims"][0]["value"] == "92.1"


def test_qualitative_question_drops_only_unsupported_numeric_line() -> None:
    events, service, store, chunks = _run_guard(
        [
            "ASPIRE kết hợp semantic và prosody.\nASPIRE đạt F1 91.7%.",
            (
                "- ASPIRE kết hợp semantic và prosody để hướng dẫn nhận diện cảm xúc.\n"
                "- ASPIRE đạt F1 92.1%."
            ),
        ],
        task="Tóm tắt contribution chính của ASPIRE.",
    )

    answer = "".join(chunks)
    assert "semantic và prosody" in answer
    assert "91.7" not in answer and "92.1" not in answer
    assert "chưa có đủ evidence" not in answer
    assert "92.1" not in events
    assert store.calls[-1]["output_payload"]["fallback_used"] is False
    attempts = store.calls[-1]["output_payload"]["attempt_validations"]
    assert [item["valid"] for item in attempts] == [False, True]
    assert len(service.prompts) == 1


def test_quantitative_question_keeps_supported_row_and_drops_unsupported_row() -> None:
    events, _service, store, chunks = _run_guard(
        [
            "ASPIRE đạt F1 91.7%.",
            "- ASPIRE đạt F1 80.4%.\n- Other đạt F1 92.1%.",
        ]
    )

    answer = "".join(chunks)
    assert "80.4" in answer
    assert "91.7" not in answer and "92.1" not in answer
    assert "chưa có đủ evidence" not in answer
    assert "92.1" not in events
    assert store.calls[-1]["output_payload"]["fallback_used"] is False
    attempts = store.calls[-1]["output_payload"]["attempt_validations"]
    assert [item["valid"] for item in attempts] == [False, False, True]


def test_quantitative_sanitizer_avoids_retry_when_requested_metric_survives() -> None:
    events, service, store, chunks = _run_guard(
        ["- ASPIRE đạt F1 80.4%.\n- Other đạt F1 92.1%."]
    )

    answer = "".join(chunks)
    assert "80.4" in answer
    assert "92.1" not in answer and "92.1" not in events
    assert len(service.prompts) == 1
    assert store.calls[-1]["output_payload"]["attempts"] == 1
    attempts = store.calls[-1]["output_payload"]["attempt_validations"]
    assert [item["valid"] for item in attempts] == [False, True]


def test_nonquantitative_grounded_stream_releases_first_validated_block_early() -> None:
    state = {"second_requested": False}

    async def chunks():
        yield AnswerStreamChunk(
            content="ASPIRE kết hợp semantic và prosody.\n\n"
        )
        state["second_requested"] = True
        yield AnswerStreamChunk(content="Thiết kế này hỗ trợ nhận diện cảm xúc.")
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")

    async def consume_first() -> tuple[str, list[tuple[str, dict]]]:
        stream = _stream_validated_grounded_blocks(
            chunks(),
            documents=[
                {
                    "document_id": "aspire",
                    "filename": "ASPIRE.pdf",
                    "content": "ASPIRE combines semantic and prosodic information.",
                }
            ],
            focus_document_ids=["aspire"],
        )
        event_type, payload = await anext(stream)
        assert state["second_requested"] is False
        remaining = [item async for item in stream]
        return payload["delta"], [(event_type, payload), *remaining]

    first_delta, events = asyncio.run(consume_first())

    assert "semantic và prosody" in first_delta
    assert events[-1][0] == "finished"
    assert state["second_requested"] is True


def test_multi_paper_stream_releases_each_validated_section_before_generation_end() -> None:
    state = {"second_requested": False}

    async def chunks():
        yield AnswerStreamChunk(
            content='<paper document_id="a">ASPIRE uses semantic and prosodic cues.</paper>'
        )
        state["second_requested"] = True
        yield AnswerStreamChunk(
            content='<paper document_id="b">ViSEC uses pitch information.</paper>'
            '<synthesis>The papers use different speech cues.</synthesis>'
        )
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")

    documents = [
        {
            "document_id": "a",
            "filename": "ASPIRE.pdf",
            "content": "ASPIRE uses semantic and prosodic cues.",
        },
        {
            "document_id": "b",
            "filename": "ViSEC.pdf",
            "content": "ViSEC uses pitch information.",
        },
    ]

    async def consume_first():
        stream = _stream_validated_paper_sections(
            chunks(), documents=documents, ordered_document_ids=["a", "b"]
        )
        first = await anext(stream)
        second = await anext(stream)
        assert state["second_requested"] is False
        remaining = [item async for item in stream]
        return first, second, remaining

    first, second, remaining = asyncio.run(consume_first())
    assert first[0] == "paper_validated"
    assert first[1]["document_id"] == "a"
    assert second[0] == "delta"
    assert "ASPIRE.pdf" in second[1]["delta"]
    assert all("<paper" not in str(event) for event in [first, second, *remaining])
    assert remaining[-1][0] == "finished"


def test_multi_paper_stream_fails_closed_for_missing_second_section() -> None:
    async def chunks():
        yield AnswerStreamChunk(
            content='<paper document_id="a">ASPIRE uses semantic cues.</paper>'
        )
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")

    documents = [
        {"document_id": "a", "filename": "ASPIRE.pdf", "content": "ASPIRE uses semantic cues."},
        {"document_id": "b", "filename": "ViSEC.pdf", "content": "ViSEC uses pitch."},
    ]

    async def consume():
        return [
            item
            async for item in _stream_validated_paper_sections(
                chunks(), documents=documents, ordered_document_ids=["a", "b"]
            )
        ]

    events = asyncio.run(consume())
    visible = "".join(
        str(payload.get("delta") or "") for event, payload in events if event == "delta"
    )
    assert "ASPIRE.pdf" in visible
    assert "ViSEC.pdf" in visible
    assert "chưa có đủ canonical evidence" in visible
    assert any(
        event == "paper_validated"
        and payload.get("document_id") == "b"
        and payload.get("fallback_used")
        for event, payload in events
    )


def test_exact_table_fast_path_skips_graph_and_answer_model() -> None:
    service = _Service([])
    store = _RunStore()
    assistant_chunks: list[str] = []

    async def consume() -> str:
        events: list[str] = []
        async for event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(task="Đưa mình bảng 2 của paper"),
            run_id="run-table",
            conversation_id="conversation-table",
            user_message_id="message-table",
            task="Đưa mình bảng 2 của paper",
            resolved_task="Đưa mình bảng 2 của paper",
            conversation_context="",
            answer_intent="direct_answer",
            answer_depth="normal",
            retrieved_docs=[
                {
                    "document_id": "visec",
                    "table_id": "table-2",
                    "table_index": 1,
                    "filename": "ViSEC.pdf",
                    "page_number": 4,
                    "caption": "Table 2. Experimental results",
                    "content": (
                        "caption: Table 2. Experimental results\ncontent:\n"
                        "| Model | UA (%) | WA (%) |\n"
                        "|---|---:|---:|\n"
                        "| Pitch-fusion | 72.72 | 71.90 |"
                    ),
                }
            ],
            tool_decision={"route": "file_qa"},
            assistant_chunks=assistant_chunks,
            focus_document_ids=["visec"],
            validate_quantitative_claims=True,
        ):
            events.append(event)
        return "".join(events)

    events = asyncio.run(consume())

    assert service.graph_calls == 0
    assert service.prompts == []
    assert "72.72" in "".join(assistant_chunks)
    assert "direct_canonical_table" in events
    assert store.calls[-1]["output_payload"]["attempts"] == 0


def test_multi_document_table_request_never_collapses_to_one_direct_table() -> None:
    service = _Service(["So sánh dựa trên evidence của cả hai paper."])
    store = _RunStore()
    assistant_chunks: list[str] = []

    async def consume() -> str:
        events: list[str] = []
        async for event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(
                task="So sánh bảng kết quả của MSF-SER và wav2small"
            ),
            run_id="run-multi-table",
            conversation_id="conversation-multi-table",
            user_message_id="message-multi-table",
            task="So sánh bảng kết quả của MSF-SER và wav2small",
            resolved_task="So sánh bảng kết quả của MSF-SER và wav2small",
            conversation_context="",
            answer_intent="compare",
            answer_depth="normal",
            retrieved_docs=[
                {
                    "document_id": "msf-ser",
                    "table_id": "msf-table",
                    "filename": "MSF-SER.pdf",
                    "caption": "Table 3. Main results",
                    "content": "| Model | F1 |\n|---|---:|\n| MSF-SER | 0.70 |",
                },
                {
                    "document_id": "wav2small",
                    "table_id": "wav-table",
                    "filename": "wav2small.pdf",
                    "caption": "Table 1. Main results",
                    "content": "| Model | F1 |\n|---|---:|\n| wav2small | 0.69 |",
                },
            ],
            tool_decision={"route": "file_qa"},
            assistant_chunks=assistant_chunks,
            focus_document_ids=["msf-ser", "wav2small"],
            validate_quantitative_claims=False,
        ):
            events.append(event)
        return "".join(events)

    events = asyncio.run(consume())

    assert service.graph_calls == 1
    assert len(service.prompts) == 1
    assert "cả hai paper" in "".join(assistant_chunks)
    assert "direct_canonical_table" not in events


def test_multi_document_answer_retries_when_one_required_paper_is_omitted() -> None:
    service = _Service(
        [
            "AlphaModel đạt F1 0.80.",
            "AlphaModel đạt F1 0.80, còn BetaModel đạt F1 0.70.",
        ]
    )
    store = _RunStore()
    assistant_chunks: list[str] = []

    async def consume() -> str:
        events: list[str] = []
        async for event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(task="So sánh kết quả AlphaModel và BetaModel"),
            run_id="run-answer-coverage",
            conversation_id="conversation-answer-coverage",
            user_message_id="message-answer-coverage",
            task="So sánh kết quả AlphaModel và BetaModel",
            resolved_task="So sánh kết quả AlphaModel và BetaModel",
            conversation_context="",
            answer_intent="compare",
            answer_depth="normal",
            retrieved_docs=[
                {
                    "document_id": "alpha-id",
                    "filename": "AlphaModel.pdf",
                    "content": "AlphaModel reports F1 0.80.",
                },
                {
                    "document_id": "beta-id",
                    "filename": "BetaModel.pdf",
                    "content": "BetaModel reports F1 0.70.",
                },
            ],
            tool_decision={"route": "file_qa"},
            assistant_chunks=assistant_chunks,
            focus_document_ids=["alpha-id", "beta-id"],
            validate_quantitative_claims=True,
            require_all_focus_documents=True,
        ):
            events.append(event)
        return "".join(events)

    events = asyncio.run(consume())

    assert len(service.prompts) == 2
    assert "missing_answer_documents" in service.prompts[1]
    assert "AlphaModel đạt F1 0.80." not in "".join(assistant_chunks)
    assert "BetaModel đạt F1 0.70" in "".join(assistant_chunks)
    assert (
        store.calls[-1]["output_payload"]["attempt_validations"][0]["reason"]
        == "missing_answer_documents"
    )
