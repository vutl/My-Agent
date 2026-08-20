from __future__ import annotations

from app.rag.figure_quality import (
    classify_visual_asset,
    extract_figure_label,
    figure_is_indexable,
    group_visual_candidates,
)


def test_extract_figure_label_does_not_use_iteration_index() -> None:
    assert extract_figure_label("Fig. 7. Loss curves") is not None
    assert extract_figure_label("Fig. 7. Loss curves").number == 7
    assert extract_figure_label("Hình 12: Kiến trúc").label == "Hình 12"
    assert extract_figure_label("Figure extracted from page 3") is None


def test_repeated_header_asset_is_rejected_even_when_large() -> None:
    decision = classify_visual_asset(
        caption="Conference branding",
        extraction_method="docling_picture",
        bbox={"x0": 20, "y0": 760, "x1": 220, "y1": 820},
        metadata={"width": 600, "height": 180, "page_width": 600, "page_height": 840},
        repeated_asset_count=6,
    )
    assert decision.status == "rejected"
    assert decision.asset_kind == "branding"


def test_small_fallback_caption_does_not_bypass_quality_gate() -> None:
    decision = classify_visual_asset(
        caption="Figure 2: Model architecture",
        extraction_method="docling_picture",
        bbox={"x0": 10, "y0": 300, "x1": 80, "y1": 330},
        metadata={
            "width": 140,
            "height": 60,
            "page_width": 600,
            "page_height": 840,
            "caption_source": "fallback_sequence",
        },
    )
    assert decision.status == "needs_review"
    assert decision.accepted is False


def test_legitimate_wide_figure_with_direct_caption_is_accepted() -> None:
    decision = classify_visual_asset(
        caption="Figure 1: Attention weights",
        extraction_method="docling_picture",
        bbox={"x0": 45, "y0": 300, "x1": 545, "y1": 365},
        metadata={
            "width": 487,
            "height": 57,
            "page_width": 600,
            "page_height": 840,
            "caption_source": "docling_direct",
        },
    )
    assert decision.status == "accepted"
    assert decision.accepted is True


def test_cmdm_panels_group_into_two_logical_figures_stably() -> None:
    candidates = [
        {"page_number": 9, "bbox": {"x0": 63.3, "y0": 697.3, "x1": 159.4, "y1": 603.0}, "image_hash": "a"},
        {"page_number": 9, "bbox": {"x0": 60.4, "y0": 587.5, "x1": 159.0, "y1": 493.1}, "image_hash": "b"},
        {"page_number": 9, "bbox": {"x0": 181.6, "y0": 696.1, "x1": 279.5, "y1": 602.9}, "image_hash": "c"},
        {"page_number": 9, "bbox": {"x0": 181.0, "y0": 587.4, "x1": 279.1, "y1": 494.3}, "image_hash": "d"},
        {"page_number": 9, "bbox": {"x0": 319.1, "y0": 692.3, "x1": 546.8, "y1": 575.4}, "image_hash": "e"},
    ]

    grouped = group_visual_candidates(candidates, document_key="cmdm")
    shuffled = group_visual_candidates(list(reversed(candidates)), document_key="cmdm")

    assert sorted(len(group.member_indices) for group in grouped) == [1, 4]
    assert {group.logical_group_id for group in grouped} == {
        group.logical_group_id for group in shuffled
    }


def test_distinct_figure_labels_are_never_merged() -> None:
    groups = group_visual_candidates(
        [
            {
                "page_number": 2,
                "bbox": {"x0": 20, "y0": 100, "x1": 200, "y1": 260},
                "caption": "Figure 2: First result",
            },
            {
                "page_number": 2,
                "bbox": {"x0": 205, "y0": 100, "x1": 390, "y1": 260},
                "caption": "Figure 5: Second result",
            },
        ],
        document_key="paper",
    )
    assert len(groups) == 2


def test_index_gate_rejects_logo_and_incomplete_panel() -> None:
    assert not figure_is_indexable(
        caption="Publisher logo",
        extraction_method="docling_picture",
        metadata={"quality_status": "rejected", "asset_kind": "logo"},
    )
    assert not figure_is_indexable(
        caption="Figure 6 panel",
        extraction_method="docling_picture",
        metadata={"quality_status": "needs_review", "asset_kind": "panel"},
    )
    assert figure_is_indexable(
        caption="Figure 6: Confusion matrices",
        extraction_method="logical_composite",
        metadata={"quality_status": "accepted", "asset_kind": "figure", "is_complete": True},
    )
