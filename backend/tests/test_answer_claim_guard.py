import asyncio
from types import SimpleNamespace

from app.api.agent import (
    AgentRunRequest,
    _remove_unsupported_claim_lines,
    _sanitize_answer_before_retry,
    _stream_graph_and_answer,
    _stream_validated_grounded_blocks,
    _stream_validated_paper_sections,
    _validate_partitioned_paper_answer,
)
from app.services.agent_service import (
    AgentGraphResult,
    AgentGraphStreamEvent,
    AnswerStreamChunk,
)
from app.services.evidence_validator import validate_answer_claims


def test_numeric_sanitization_removes_orphan_table_scaffolding():
    answer = "Datasets are listed below.\n\n| Dataset | Train | Test |\n|---|---:|---:|\n| Corpus-X | 999 | 555 |\n\nThe annotations are categorical."
    result = _remove_unsupported_claim_lines(answer, {
        "unsupported_claims": [{"text": "| Corpus-X | 999 | 555 |"}],
    })
    assert "| Dataset" not in result and "|---" not in result
    assert "Datasets are listed below." in result
    assert "The annotations are categorical." in result


def test_numeric_sanitization_preserves_remaining_safe_table_rows():
    answer = "| Item | Count |\n|---|---:|\n| Amber | 12 |\n| Violet | 999 |"
    result = _remove_unsupported_claim_lines(answer, {
        "unsupported_claims": [{"text": "| Violet | 999 |"}],
    })
    assert "| Item" in result and "| Amber | 12 |" in result
    assert "Violet" not in result


def test_sanitization_preserves_missing_document_coverage_without_numbers() -> None:
    documents = [
        {
            "document_id": "msf-id",
            "filename": "MSF-SER.pdf",
            "content": "MSF-SER CCC avg 0.638.",
        },
        {
            "document_id": "mamba-id",
            "filename": "FROM_SINGLE_TO_MULTI_LABEL_SER__DATASET_AND_MAMBA_BASED_FUSION_MODEL (7).pdf",
            "content": "Mamba-based fusion model multi-label results.",
        },
    ]
    draft = (
        "| Paper / Model | Result |\n|---|---|\n"
        "| MSF-SER | CCC avg: 0.638 |\n"
        "| Mamba-Based Fusion Model | Acc 0.99 (unsupported) |"
    )
    validation = validate_answer_claims(
        answer=draft,
        documents=documents,
        focus_document_ids=["msf-id", "mamba-id"],
        require_all_focus_documents=True,
    )
    sanitized, safe_validation, accepted = _sanitize_answer_before_retry(
        answer=draft,
        validation=validation,
        documents=documents,
        focus_document_ids=["msf-id", "mamba-id"],
        task="So sánh kết quả hai bài đó.",
        require_all_focus_documents=True,
    )

    assert accepted is True
    assert safe_validation is not None and safe_validation.valid is True
    assert "Insufficient evidence" in sanitized
    assert "0.99" not in sanitized


def test_sanitization_keeps_supported_partial_comparison_when_no_line_is_unsupported() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaModel reports F1 0.80.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "The retrieved excerpt discusses setup only.",
        },
    ]
    draft = "**AlphaStudy.pdf**\n\nAlphaModel reports F1 0.80."
    validation = validate_answer_claims(
        answer=draft,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
    )

    sanitized, safe_validation, accepted = _sanitize_answer_before_retry(
        answer=draft,
        validation=validation,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        task="So sánh kết quả hai bài.",
        require_all_focus_documents=True,
    )

    assert accepted is True
    assert safe_validation is not None and safe_validation.valid is True
    assert "AlphaModel reports F1 0.80" in sanitized
    assert "BetaStudy" in sanitized
    assert "Insufficient evidence" in sanitized


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


def test_visual_guard_abstains_when_only_text_or_ocr_is_retrieved() -> None:
    events, service, store, chunks = _run_guard(
        ["Biểu đồ cho thấy đường màu xanh tăng mạnh ở cuối tập dữ liệu."],
        task="Hình này cho thấy gì?",
    )

    answer = "".join(chunks)
    assert len(service.prompts) == 1
    assert "không thể xác định nội dung trực quan" in answer
    assert "đường màu xanh" not in answer
    assert "không thể xác định nội dung trực quan" in events
    assert store.calls[-1]["output_payload"]["fallback_used"] is True
    assert store.calls[-1]["output_payload"]["visual_guard_triggered"] is True


def test_visual_guard_also_handles_empty_retrieval_without_leaking_chat_draft() -> None:
    service = _Service(["The chart shows a rising blue line."])
    store = _RunStore()
    chunks: list[str] = []

    async def consume() -> str:
        events: list[str] = []
        async for event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(task="what does the chart show?"),
            run_id="run-empty-visual",
            conversation_id="conversation",
            user_message_id="message",
            task="what does the chart show?",
            resolved_task="what does the chart show?",
            conversation_context="",
            answer_intent="direct_answer",
            answer_depth="normal",
            retrieved_docs=[],
            tool_decision={"route": "chat"},
            assistant_chunks=chunks,
            validate_quantitative_claims=True,
        ):
            events.append(event)
        return "".join(events)

    events = asyncio.run(consume())
    answer = "".join(chunks)
    assert "canonical figure/page/image" in answer
    assert "pixels were supplied" in answer
    assert "cannot determine" in answer
    assert "blue line" not in answer
    assert "canonical figure/page/image" in events
    assert "pixels were supplied" in events
    assert store.calls[-1]["tool_name"] == "validate_visual_evidence"


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
    assert [
        partition["document_id"]
        for partition in remaining[-1][1]["partitions"]
    ] == ["a", "b", None]


def test_explicit_synthesis_rows_and_partitioned_validation_keep_paper_ownership() -> None:
    documents = [
        {
            "document_id": "alpha-id",
            "filename": "AlphaStudy.pdf",
            "content": "AlphaStudy reports F1 0.80.",
        },
        {
            "document_id": "beta-id",
            "filename": "BetaStudy.pdf",
            "content": "BetaStudy reports F1 0.70.",
        },
    ]
    synthesis = """
| Paper/model | Result |
|---|---|
| AlphaStudy | F1 0.80 |
| BetaStudy | F1 0.70 |
"""
    concatenated = (
        "**AlphaStudy.pdf**\n\nArchitecture summary.\n\n"
        "**BetaStudy.pdf**\n\nArchitecture summary.\n\n"
        + synthesis
    )

    reparsed = validate_answer_claims(
        answer=concatenated,
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    cross_assigned = validate_answer_claims(
        answer=concatenated.replace("AlphaStudy | F1 0.80", "AlphaStudy | F1 0.70"),
        documents=documents,
        focus_document_ids=["alpha-id", "beta-id"],
        require_all_focus_documents=True,
        answer_document_ids=["alpha-id", "beta-id"],
    )
    partitioned = _validate_partitioned_paper_answer(
        partitions=[
            {"document_id": "alpha-id", "text": "Architecture summary."},
            {"document_id": "beta-id", "text": "Architecture summary."},
            {"document_id": None, "text": synthesis},
        ],
        documents=documents,
        ordered_document_ids=["alpha-id", "beta-id"],
    )

    # Explicit row identities are now sufficient even when the presentation
    # follows the last paper heading; transport partitions remain the stronger
    # source of ownership and must agree with the same result.
    assert reparsed.valid is True
    assert partitioned.valid is True
    assert cross_assigned.valid is False
    assert {claim.value for claim in partitioned.supported_claims} == {"0.8", "0.7"}


def test_answer_guard_validates_against_full_canonical_evidence_not_prompt_projection() -> None:
    service = _Service(["AlphaStudy reports F1 0.80."])
    store = _RunStore()
    assistant_chunks: list[str] = []

    async def consume() -> None:
        async for _event in _stream_graph_and_answer(
            service=service,
            run_store=store,
            request=AgentRunRequest(task="Give the benchmark result."),
            run_id="run-validation-evidence",
            conversation_id="conversation-validation-evidence",
            user_message_id="message-validation-evidence",
            task="Give the benchmark result.",
            resolved_task="AlphaStudy benchmark result",
            conversation_context="",
            answer_intent="direct_answer",
            answer_depth="normal",
            retrieved_docs=[
                {
                    "document_id": "alpha-id",
                    "filename": "AlphaStudy.pdf",
                    "content": "Projected architecture-only context.",
                }
            ],
            validation_docs=[
                {
                    "document_id": "alpha-id",
                    "filename": "AlphaStudy.pdf",
                    "content": "AlphaStudy reports F1 0.80.",
                }
            ],
            tool_decision={"route": "file_qa"},
            assistant_chunks=assistant_chunks,
            focus_document_ids=["alpha-id"],
            validate_quantitative_claims=True,
        ):
            pass

    asyncio.run(consume())

    assert service.prompts == ["Use only the retrieved evidence."]
    assert "F1 0.80" in "".join(assistant_chunks)
    assert store.calls[-1]["output_payload"]["valid"] is True


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


def test_multi_paper_stream_recovers_ordered_catalog_headings_without_xml() -> None:
    async def chunks():
        yield AnswerStreamChunk(
            content=(
                "### ASPIRE\nASPIRE uses semantic cues.\n\n"
                "### ViSEC / Pitch-fusion\nViSEC uses pitch information.\n\n"
                "### Direct comparison\nThe papers use different speech cues."
            )
        )
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")

    documents = [
        {"document_id": "a", "filename": "ASPIRE.pdf", "content": "ASPIRE uses semantic cues."},
        {"document_id": "b", "filename": "ViSEC.pdf", "content": "ViSEC uses pitch information."},
    ]

    def resolve(value: str) -> list[str]:
        result = []
        if "aspire" in value.casefold():
            result.append("a")
        if "visec" in value.casefold() or "pitch-fusion" in value.casefold():
            result.append("b")
        return result

    async def consume():
        return [
            item
            async for item in _stream_validated_paper_sections(
                chunks(),
                documents=documents,
                ordered_document_ids=["a", "b"],
                document_identity_resolver=resolve,
            )
        ]

    events = asyncio.run(consume())
    finished = events[-1][1]
    visible = "".join(
        str(payload.get("delta") or "") for event, payload in events if event == "delta"
    )
    assert "ASPIRE uses semantic cues" in visible
    assert "ViSEC uses pitch information" in visible
    assert "different speech cues" in visible
    assert "chưa có đủ canonical evidence" not in visible
    assert [part["document_id"] for part in finished["partitions"]] == ["a", "b", None]
    assert all(item.get("contract_recovered") for item in finished["validations"])


def test_multi_paper_heading_recovery_fails_closed_for_wrong_document_order() -> None:
    async def chunks():
        yield AnswerStreamChunk(
            content=(
                "### ViSEC\nViSEC uses pitch information.\n\n"
                "### ASPIRE\nASPIRE uses semantic cues."
            )
        )
        yield AnswerStreamChunk(content="", done=True, finish_reason="stop")

    documents = [
        {"document_id": "a", "filename": "ASPIRE.pdf", "content": "ASPIRE uses semantic cues."},
        {"document_id": "b", "filename": "ViSEC.pdf", "content": "ViSEC uses pitch information."},
    ]

    def resolve(value: str) -> list[str]:
        if "aspire" in value.casefold():
            return ["a"]
        if "visec" in value.casefold():
            return ["b"]
        return []

    async def consume():
        return [
            item
            async for item in _stream_validated_paper_sections(
                chunks(),
                documents=documents,
                ordered_document_ids=["a", "b"],
                document_identity_resolver=resolve,
            )
        ]

    events = asyncio.run(consume())
    visible = "".join(
        str(payload.get("delta") or "") for event, payload in events if event == "delta"
    )
    assert visible.count("chưa có đủ canonical evidence") == 2
    assert all(
        payload.get("reason") == "missing_or_malformed_paper_section"
        for event, payload in events
        if event == "paper_validated"
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


def test_multi_document_answer_preserves_scoped_partial_evidence_when_one_paper_is_omitted() -> None:
    service = _Service(
        ["AlphaModel đạt F1 0.80."]
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

    assert len(service.prompts) == 1
    answer = "".join(assistant_chunks)
    assert "AlphaModel đạt F1 0.80." in answer
    assert "BetaModel" in answer
    assert "Insufficient evidence" in answer
    assert (
        store.calls[-1]["output_payload"]["attempt_validations"][0]["reason"]
        == "missing_answer_documents"
    )


def test_number_free_opening_streams_before_numeric_validation() -> None:
    events, service, _store, chunks = _run_guard(
        ["ASPIRE kết hợp semantic và prosody.\n\nASPIRE đạt F1 80.4%."]
    )

    assert len(service.prompts) == 1
    assert "".join(chunks) == "ASPIRE kết hợp semantic và prosody.\n\nASPIRE đạt F1 80.4%."
    # The opening is sent before validation is reported; the number after.
    assert events.index("semantic và prosody") < events.index("answer.evidence.validated")
    assert events.index("answer.evidence.validated") < events.index("80.4")
    assert "message.reset" not in events


def test_streamed_opening_is_reset_when_the_validated_answer_changes() -> None:
    events, _service, _store, chunks = _run_guard(
        [
            "ASPIRE dùng semantic.\n\nASPIRE đạt F1 91.7%.",
            "Theo evidence, ASPIRE đạt F1 80.4%.",
        ]
    )

    assert "message.reset" in events
    assert events.index("ASPIRE dùng semantic") < events.index("message.reset")
    assert "".join(chunks) == "Theo evidence, ASPIRE đạt F1 80.4%."
    assert "91.7" not in events


class _VerifyClient:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def chat(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(message=self.reply)


def _run_post(answer: str, reply: str | Exception = '{"corrections": []}'):
    service = _Service([answer])
    service.client = _VerifyClient(reply)
    service.default_model = "test"
    store = _RunStore()
    chunks: list[str] = []

    async def consume() -> list[str]:
        events = []
        async for event in _stream_graph_and_answer(
            service=service, run_store=store, request=AgentRunRequest(task="F1 của ASPIRE?"),
            run_id="run", conversation_id="conversation", user_message_id="message",
            task="F1 của ASPIRE?", resolved_task="ASPIRE F1", conversation_context="",
            answer_intent="direct_answer", answer_depth="normal",
            retrieved_docs=[{"document_id": "aspire", "filename": "ASPIRE.pdf",
                             "content": "ASPIRE reports F1 80.4% on IEMOCAP."}],
            tool_decision={"route": "file_qa"}, assistant_chunks=chunks,
            focus_document_ids=["aspire"], validate_quantitative_claims=True,
            answer_verify_mode="post",
        ):
            events.append(event)
        return events

    events = asyncio.run(consume())
    return "".join(events), service, store, "".join(chunks)


def _with_verify_scope(monkeypatch, scope: str) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("ANSWER_VERIFY_SCOPE", scope)
    get_settings.cache_clear()


def test_post_mode_flagged_scope_skips_the_verify_call_for_supported_answers(monkeypatch) -> None:
    from app.core.config import get_settings

    _with_verify_scope(monkeypatch, "flagged")
    try:
        events, service, store, answer = _run_post("ASPIRE đạt F1 80.4% trên IEMOCAP.")
    finally:
        get_settings.cache_clear()
    assert answer == "ASPIRE đạt F1 80.4% trên IEMOCAP."
    assert service.client.prompts == [] and "message.reset" not in events
    assert len(service.prompts) == 1  # one generation, no retry


def test_post_mode_numeric_scope_checks_sourced_numbers_too(monkeypatch) -> None:
    # A number can occur in the sources yet belong to another row or period;
    # heuristics do not flag it, so the default scope asks the model.
    from app.core.config import get_settings

    _with_verify_scope(monkeypatch, "numeric")
    try:
        events, service, store, answer = _run_post(
            "ASPIRE kết hợp semantic và prosody.\n\nASPIRE đạt F1 80.4% trên IEMOCAP. Xem Table 3."
        )
    finally:
        get_settings.cache_clear()
    assert len(service.client.prompts) == 1 and "message.reset" not in events
    prompt = service.client.prompts[0]
    assert "80.4%" in prompt and "semantic và prosody." not in prompt.split("Sources")[0]
    assert "- Xem Table 3." not in prompt  # reference numbers are not claims
    record = next(call for call in store.calls if call["tool_name"] == "answer_verification")
    assert record["output_payload"]["scope"] == "numeric"


def test_post_mode_rewrites_only_the_flagged_line() -> None:
    reply = ('{"corrections": [{"sentence": "ASPIRE đạt F1 91.7%.", "action": "rewrite",'
             ' "rewritten": "ASPIRE đạt F1 80.4%."}]}')
    events, service, store, answer = _run_post("ASPIRE kết hợp semantic và prosody.\n\nASPIRE đạt F1 91.7%.", reply)
    assert "message.reset" in events
    assert answer == "ASPIRE kết hợp semantic và prosody.\n\nASPIRE đạt F1 80.4%."
    assert "91.7" in service.client.prompts[0] and "semantic và prosody" not in service.client.prompts[0]
    record = next(call for call in store.calls if call["tool_name"] == "answer_verification")
    assert record["output_payload"]["changed"] is True
    assert "chưa có đủ evidence" not in answer


def test_post_mode_respects_a_keep_verdict() -> None:
    reply = '{"corrections": [{"sentence": "ASPIRE đạt F1 91.7%.", "action": "keep", "rewritten": null}]}'
    events, _service, _store, answer = _run_post("ASPIRE đạt F1 91.7%.", reply)
    assert answer == "ASPIRE đạt F1 91.7%." and "message.reset" not in events


def test_post_mode_never_blocks_when_verification_fails() -> None:
    events, _service, store, answer = _run_post("ASPIRE đạt F1 91.7%.", RuntimeError("provider down"))
    assert answer == "ASPIRE đạt F1 91.7%." and "message.reset" not in events
    record = next(call for call in store.calls if call["tool_name"] == "answer_verification")
    assert "provider down" in record["output_payload"]["error"]


def test_post_verify_also_sees_the_earlier_conversation():
    from app.api.agent import _verification_sources

    sources = _verification_sources(
        [{"filename": "CMDM.pdf", "content": "Figure 2"}],
        "Assistant: | L | Ours | 65.4% | 64.8% |",
    )
    assert sources[-1]["filename"] == "earlier turns of this conversation"
    assert "65.4%" in sources[-1]["content"]
    assert _verification_sources([{"content": "x"}], "") == [{"content": "x"}]


def test_visual_words_without_a_depiction_question_are_answered_from_text() -> None:
    from app.api.agent import _visual_answer_guard_required, _visual_sufficiency_fallback

    text_only = [{"document_id": "d", "content": "Niño-3.4 index was -0.6°C.", "chunk_type": "text"}]
    assert not _visual_answer_guard_required(
        "NOAA's slide update and its diagnostic discussion give different Niño-3.4 values. What are they?", text_only, None,
    )
    assert not _visual_answer_guard_required(
        "Which province had the highest GRDP per capita, and what was the national figure?", text_only, None,
    )
    assert _visual_answer_guard_required("What does the chart show?", text_only, None)
    assert _visual_sufficiency_fallback("What does the El Niño chart show?").startswith("No ")
