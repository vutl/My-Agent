from app.services.evidence_validator import EvidenceValidationResult
from app.services.retrieval_agent_service import (
    MAX_ADAPTIVE_SUBQUERIES,
    MAX_RETRIEVAL_HOPS,
    assess_facet_coverage,
    assess_retrieval_gaps,
    extract_graph_bridge_anchors,
    extract_query_facets,
    plan_retrieval_decomposition,
    plan_second_retrieval_pass,
    smart_retrieval_enabled,
)


def test_smart_retrieval_disabled_for_fast_mode() -> None:
    assert (
        smart_retrieval_enabled(
            "fast",
            answer_intent="compare",
            query="so sánh KST với Mamba",
            focus_document_ids=["a", "b"],
        )
        is False
    )


def test_smart_retrieval_auto_enables_for_compare() -> None:
    assert (
        smart_retrieval_enabled(
            "auto",
            answer_intent="compare",
            query="so sánh KST với Mamba",
            focus_document_ids=["a", "b"],
        )
        is True
    )


def test_smart_retrieval_auto_enables_for_structural_and_multifacet_tasks() -> None:
    assert smart_retrieval_enabled(
        "auto",
        answer_intent="infer_structure",
        query="Explain the model",
        focus_document_ids=["doc-a"],
    )
    assert smart_retrieval_enabled(
        "auto",
        answer_intent="elaborate",
        query="Explain the architecture and benchmark results",
        focus_document_ids=["doc-a"],
    )
    assert not smart_retrieval_enabled(
        "auto",
        answer_intent="direct_answer",
        query="What is the architecture?",
        focus_document_ids=["doc-a"],
    )


def test_compare_decomposition_creates_one_canonical_scope_per_document() -> None:
    branches = plan_retrieval_decomposition(
        query="compare architecture and benchmark",
        answer_intent="compare",
        focus_document_ids=["doc-a", "doc-b", "doc-a"],
        enabled=True,
    )

    assert [branch.focus_document_ids for branch in branches] == [["doc-a"], ["doc-b"]]
    assert all(branch.reason == "compare_per_document" for branch in branches)


def test_decomposition_skips_single_document_and_non_compare_queries() -> None:
    assert (
        plan_retrieval_decomposition(
            query="explain architecture",
            answer_intent="elaborate",
            focus_document_ids=["doc-a", "doc-b"],
            enabled=True,
        )
        == []
    )
    # More than the bounded branch budget falls back to the caller's normal
    # scoped retrieval instead of silently omitting a canonical document.
    assert (
        plan_retrieval_decomposition(
            query="compare all",
            answer_intent="compare",
            focus_document_ids=["doc-a", "doc-b", "doc-c", "doc-d"],
            enabled=True,
        )
        == []
    )
    assert (
        plan_retrieval_decomposition(
            query="compare",
            answer_intent="compare",
            focus_document_ids=["doc-a"],
            enabled=True,
        )
        == []
    )


def test_must_cover_decomposition_is_independent_from_comparison_style() -> None:
    branches = plan_retrieval_decomposition(
        query="Đưa abstract của ASPIRE và KST",
        answer_intent="direct_answer",
        focus_document_ids=["doc-aspire", "doc-kst"],
        enabled=True,
        must_cover_all=True,
    )

    assert [branch.focus_document_ids for branch in branches] == [
        ["doc-aspire"],
        ["doc-kst"],
    ]


def test_assess_retrieval_gaps_detects_missing_compare_document() -> None:
    gap = assess_retrieval_gaps(
        documents=[
            {"document_id": "doc-kst", "content": "KST architecture"},
            {"document_id": "doc-kst", "figure_id": "fig-1", "content": "fig"},
        ],
        answer_intent="compare",
        focus_document_ids=["doc-kst", "doc-mamba"],
        query="so sánh KST với Mamba fig 2",
    )
    assert gap.needs_second_pass is True
    assert gap.reason == "missing_compare_document_text"
    assert gap.missing_document_ids == ["doc-mamba"]


def test_assess_retrieval_gaps_honors_scope_obligation_even_for_direct_wording() -> None:
    gap = assess_retrieval_gaps(
        documents=[{"document_id": "doc-a", "content": "Table A"}],
        answer_intent="direct_answer",
        focus_document_ids=["doc-a", "doc-b"],
        query="đưa bảng kết quả của cả hai",
        must_cover_all=True,
    )
    assert gap.needs_second_pass is True
    assert gap.reason == "missing_compare_document_text"
    assert gap.missing_document_ids == ["doc-b"]


def test_plan_second_retrieval_pass_refines_query_for_compare_gap() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="required_entities_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="so sánh KST với Mamba",
        original_task="so sánh KST với Mamba fig 2",
        topic="KST",
        entities=["KST", "Mamba"],
        answer_intent="compare",
        focus_document_ids=["doc-kst", "doc-mamba"],
        validation=validation,
        smart_allowed=True,
        documents=[{"document_id": "doc-kst", "content": "KST only"}],
    )
    assert plan is not None
    assert "missing_compare_document_text" in plan.reasons
    assert "architecture" in plan.query.lower()
    assert "figure" in plan.query.lower()


def test_plan_second_retrieval_pass_skips_when_fast_and_validation_ok() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="required_entities_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="xin chào",
        original_task="xin chào",
        topic=None,
        entities=[],
        answer_intent="direct_answer",
        focus_document_ids=[],
        validation=validation,
        smart_allowed=False,
        documents=[{"document_id": "doc-a", "content": "hello"}],
    )
    assert plan is None


def test_fast_mode_never_retries_even_when_validation_requests_it() -> None:
    validation = EvidenceValidationResult(
        valid=False,
        retry_required=True,
        reason="no_documents",
        required_entities=["ASPIRE"],
        matched_entities=[],
        missing_entities=["ASPIRE"],
    )
    assert (
        plan_second_retrieval_pass(
            retrieval_query="ASPIRE architecture",
            original_task="ASPIRE architecture",
            topic="ASPIRE",
            entities=["ASPIRE"],
            answer_intent="elaborate",
            focus_document_ids=["doc-aspire"],
            validation=validation,
            smart_allowed=False,
            documents=[],
        )
        is None
    )


def test_fast_mode_repairs_atomic_multidocument_text_coverage() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="focused_documents_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="đưa kết quả của ASPIRE và KST",
        original_task="đưa kết quả của ASPIRE và KST",
        topic="ASPIRE / KST",
        entities=["ASPIRE", "KST"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-aspire", "doc-kst"],
        validation=validation,
        smart_allowed=False,
        documents=[
            {"document_id": "doc-aspire", "content": "ASPIRE result table"},
            {
                "document_id": "doc-kst",
                "figure_id": "kst-fig-1",
                "content": "KST architecture figure",
            },
        ],
        must_cover_all=True,
    )

    assert plan is not None
    assert plan.reasons == ["missing_compare_document_text"]
    assert [branch.focus_document_ids for branch in plan.branches] == [["doc-kst"]]
    assert plan.missing_facets == []
    assert plan.bridge_anchors == []


def test_atomic_scope_retry_still_respects_the_global_hop_budget() -> None:
    validation = EvidenceValidationResult(
        valid=False,
        retry_required=True,
        reason="missing_focus_documents",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
        missing_document_ids=["doc-b"],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="A và B",
        original_task="A và B",
        topic="A / B",
        entities=["A", "B"],
        answer_intent="direct_answer",
        focus_document_ids=["doc-a", "doc-b"],
        validation=validation,
        smart_allowed=False,
        documents=[{"document_id": "doc-a", "content": "A evidence"}],
        must_cover_all=True,
        retry_budget_available=False,
    )

    assert plan is None


def test_second_hop_never_widens_compare_document_scope() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="focused_documents_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="compare KST and Mamba architecture",
        original_task="compare KST and Mamba architecture",
        topic="KST",
        entities=["KST", "Mamba"],
        answer_intent="compare",
        focus_document_ids=["doc-kst", "doc-mamba"],
        validation=validation,
        smart_allowed=True,
        documents=[{"document_id": "doc-kst", "content": "KST evidence"}],
    )

    assert plan is not None
    assert [branch.focus_document_ids for branch in plan.branches] == [["doc-mamba"]]
    assert all(
        set(branch.focus_document_ids) <= {"doc-kst", "doc-mamba"}
        for branch in plan.branches
    )
    assert plan.hop_count == MAX_RETRIEVAL_HOPS


def test_second_hop_is_bounded_and_splits_explicit_missing_facets() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="focused_documents_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    task = (
        "Explain architecture, training objective, benchmark results, "
        "dataset setup, ablation, and limitations"
    )
    plan = plan_second_retrieval_pass(
        retrieval_query=task,
        original_task=task,
        topic="ASPIRE",
        entities=["ASPIRE"],
        answer_intent="elaborate",
        focus_document_ids=["doc-aspire"],
        validation=validation,
        smart_allowed=True,
        documents=[
            {
                "document_id": "doc-aspire",
                "content": "first-hop evidence",
                "metadata": {"covered_facets": []},
            }
        ],
    )

    assert extract_query_facets(task)[:3] == [
        "architecture",
        "training_method",
        "benchmark_results",
    ]
    assert plan is not None
    assert len(plan.queries) == MAX_ADAPTIVE_SUBQUERIES
    assert len(plan.branches) == MAX_ADAPTIVE_SUBQUERIES
    assert len(set(plan.queries)) == MAX_ADAPTIVE_SUBQUERIES
    assert [branch.facets for branch in plan.branches] == [
        ["architecture"],
        ["training_method"],
        ["benchmark_results"],
    ]
    assert all(branch.focus_document_ids == ["doc-aspire"] for branch in plan.branches)
    assert all(branch.hop == MAX_RETRIEVAL_HOPS for branch in plan.branches)


def test_facet_coverage_uses_real_structured_retrieval_fields_only() -> None:
    coverage = assess_facet_coverage(
        query=(
            "Explain architecture, training method, benchmark results, "
            "and show a figure"
        ),
        documents=[
            {
                "document_id": "doc-aspire",
                "heading_path": ["Methods", "Proposed model architecture"],
                "section_title": "Training objective",
                "table_id": "benchmark-table",
                "artifact_type": "table",
            },
            {
                "document_id": "doc-aspire",
                "figure_id": "architecture-figure",
                "figure_type": "architecture",
                "artifact_type": "figure",
            },
        ],
        focus_document_ids=["doc-aspire"],
    )

    assert coverage.coverage_observed is True
    assert coverage.missing_facets == []
    assert set(coverage.covered_facets) >= {
        "architecture",
        "training_method",
        "benchmark_results",
        "visual_evidence",
    }

    prose_only = assess_facet_coverage(
        query="architecture and benchmark results",
        documents=[
            {
                "document_id": "doc-aspire",
                "content": (
                    "This arbitrary passage says architecture and benchmark, "
                    "but carries no structured coverage metadata."
                ),
            }
        ],
        focus_document_ids=["doc-aspire"],
    )
    assert prose_only.coverage_observed is False
    assert prose_only.missing_facets == []


def test_sufficient_explicit_coverage_does_not_add_a_hop() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="focused_documents_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="explain the architecture",
        original_task="explain the architecture",
        topic="ASPIRE",
        entities=["ASPIRE"],
        answer_intent="elaborate",
        focus_document_ids=["doc-aspire"],
        validation=validation,
        smart_allowed=True,
        documents=[
            {
                "document_id": "doc-aspire",
                "content": "complete evidence",
                "metadata": {"covered_facets": ["architecture"]},
            }
        ],
        graph_bridge_metadata=[
            {
                "document_id": "doc-aspire",
                "anchor": "Cross-modal fusion",
                "covered": True,
            }
        ],
    )
    assert plan is None


def test_graph_bridge_query_uses_only_explicit_scoped_metadata() -> None:
    validation = EvidenceValidationResult(
        valid=False,
        retry_required=True,
        reason="missing_required_entities",
        required_entities=["ASPIRE"],
        matched_entities=[],
        missing_entities=["ASPIRE"],
    )
    documents = [
        {
            "document_id": "doc-aspire",
            "content": "HALLUCINATED_SECRET occurs only in arbitrary source prose.",
            "entity_name": "Cross-modal Fusion",
            "coverage_status": "unresolved",
            "retrieval_channels": ["lightrag_entity"],
        }
    ]
    graph_metadata = [
        {
            "document_id": "doc-other",
            "anchor": "FOREIGN_GRAPH_ANCHOR",
            "coverage_status": "unresolved",
        },
        {
            "document_id": "doc-aspire",
            "anchors": ["Figure 2", "Acoustic-visual alignment"],
            "coverage_status": "unresolved",
        },
    ]
    anchors = extract_graph_bridge_anchors(
        documents=documents,
        graph_bridge_metadata=graph_metadata,
        focus_document_ids=["doc-aspire"],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="ASPIRE fusion",
        original_task="Explain ASPIRE fusion",
        topic="ASPIRE",
        entities=["ASPIRE"],
        answer_intent="elaborate",
        focus_document_ids=["doc-aspire"],
        validation=validation,
        smart_allowed=True,
        documents=documents,
        graph_bridge_metadata=graph_metadata,
    )

    assert anchors == ["Cross-modal Fusion", "Acoustic-visual alignment"]
    assert plan is not None
    assert "unresolved_graph_bridge" in plan.reasons
    assert plan.bridge_anchors == anchors
    assert "cross-modal fusion" in plan.query.lower()
    assert "acoustic-visual alignment" in plan.query.lower()
    assert "hallucinated_secret" not in plan.query.lower()
    assert "foreign_graph_anchor" not in plan.query.lower()
    assert "figure 2" not in plan.query.lower()


def test_unscoped_smart_compare_retries_only_on_demonstrable_coverage_gap() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="required_entities_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    plan = plan_second_retrieval_pass(
        retrieval_query="compare architecture across papers",
        original_task="compare architecture across papers",
        topic=None,
        entities=[],
        answer_intent="compare",
        focus_document_ids=[],
        validation=validation,
        smart_allowed=True,
        documents=[{"document_id": "doc-one", "content": "one paper only"}],
    )
    assert plan is not None
    assert "insufficient_compare_document_coverage" in plan.reasons
    assert all(branch.focus_document_ids == [] for branch in plan.branches)

    assert (
        plan_second_retrieval_pass(
            retrieval_query="What is a transformer?",
            original_task="What is a transformer?",
            topic=None,
            entities=[],
            answer_intent="direct_answer",
            focus_document_ids=[],
            validation=EvidenceValidationResult(
                valid=False,
                retry_required=True,
                reason="no_documents",
                required_entities=["transformer"],
                matched_entities=[],
                missing_entities=["transformer"],
            ),
            smart_allowed=True,
            documents=[],
        )
        is None
    )


def test_second_hop_stops_at_hop_limit_and_when_queries_make_no_progress() -> None:
    validation = EvidenceValidationResult(
        valid=True,
        retry_required=False,
        reason="focused_documents_present",
        required_entities=[],
        matched_entities=[],
        missing_entities=[],
    )
    kwargs = {
        "retrieval_query": "ASPIRE architecture and benchmark",
        "original_task": "ASPIRE architecture and benchmark",
        "topic": "ASPIRE",
        "entities": ["ASPIRE"],
        "answer_intent": "elaborate",
        "focus_document_ids": ["doc-aspire"],
        "validation": validation,
        "smart_allowed": True,
        "documents": [
            {
                "document_id": "doc-aspire",
                "metadata": {"covered_facets": []},
            }
        ],
    }
    first = plan_second_retrieval_pass(**kwargs)
    assert first is not None
    assert plan_second_retrieval_pass(
        **kwargs,
        previous_queries=first.queries,
    ) is None
    assert plan_second_retrieval_pass(
        **kwargs,
        completed_hops=MAX_RETRIEVAL_HOPS,
    ) is None
