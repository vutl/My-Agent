from app.rag.figure_caption import (
    best_figure_caption,
    caption_looks_truncated,
    dedupe_repeated_figure_prefix,
    extract_figure_caption_from_content,
    extract_figure_title_sentence,
    figure_relevance_score,
    normalize_caption_text,
)

MAMBA_FIG2_CHUNK = (
    "Figure 2:Architecture of the dual-branch Mamba-based fu-\n\n"
    "sion model. Raw audio is transformed into MFCC and log-\n\n"
    "mel features, encoded by separate Mamba stacks, fused by a\n\n"
    "lightweight gated head, and mapped to into multi-label sigmoid\n\n"
    "outputs."
)

DUPLICATED_CAPTION = (
    "Figure 2:Architecture of the dual-branch Mamba-based fu- "
    "Figure 2:Architecture of the dual-branch Mamba-based fu- "
    "sion model. Raw audio is transformed into MFCC and log- mel features."
)


def test_normalize_caption_text_joins_hyphen_breaks() -> None:
    text = "dual-branch Mamba-based fu-\n\nsion model"
    assert normalize_caption_text(text) == "dual-branch Mamba-based fusion model"


def test_dedupe_repeated_figure_prefix() -> None:
    cleaned = dedupe_repeated_figure_prefix(DUPLICATED_CAPTION)
    assert cleaned.count("Figure 2:") == 1
    assert "fusion model" in cleaned


def test_extract_figure_title_sentence_is_short() -> None:
    title = extract_figure_title_sentence(MAMBA_FIG2_CHUNK)
    assert title.startswith("Figure 2:")
    assert "fusion model" in title
    assert "MFCC branch" not in title
    assert len(title) < 220


def test_extract_figure_caption_from_content() -> None:
    caption = extract_figure_caption_from_content(MAMBA_FIG2_CHUNK, figure_number=2)
    assert caption is not None
    assert "fusion model" in caption
    assert not caption.endswith("fu-")


def test_best_figure_caption_prefers_clean_title() -> None:
    caption = best_figure_caption(
        caption="Figure 2:Architecture of the dual-branch Mamba-based fu-",
        content=MAMBA_FIG2_CHUNK,
        figure_number=2,
    )
    assert "fusion model" in caption
    assert not caption.endswith("fu-")
    assert caption.count("Figure 2:") == 1


def test_extraction_order_is_never_treated_as_paper_figure_number() -> None:
    content = "Figure 7: Correct caption.\n\nFigure 2: Different caption."

    caption = extract_figure_caption_from_content(content, figure_index=0)

    # Without an explicit figure_number the first real caption wins; the legacy
    # figure_index keyword cannot select or manufacture Figure 1.
    assert caption == "Figure 7: Correct caption."
    assert best_figure_caption(figure_index=0) == "Figure"


def test_explicit_paper_caption_wins_over_structured_vlm_payload() -> None:
    caption = best_figure_caption(
        caption="Figure 1: Proposed architecture.",
        visual_summary=(
            "asset_kind: figure\n"
            "is_content: true\n"
            "is_complete: true\n"
            "title: Architecture interpreted with paper context\n"
            "observed_visual: blocks and arrows"
        ),
    )

    assert caption == "Figure 1: Proposed architecture."
    assert "asset_kind" not in caption


def test_caption_looks_truncated() -> None:
    assert caption_looks_truncated("Figure 2:Architecture of the dual-branch Mamba-based fu-")
    assert not caption_looks_truncated(
        "Figure 2: Architecture of the dual-branch Mamba-based fusion model."
    )


def test_architecture_query_prefers_architecture_over_higher_scoring_plot() -> None:
    architecture = {
        "score": 0.18,
        "caption": "Figure 1: Architecture of the proposed model",
        "content": "figure_type: architecture\nobserved_visual: encoders and fusion blocks",
    }
    plot = {
        "score": 0.62,
        "caption": "Figure 5: Predicted arousal-valence spaces",
        "content": "figure_type: plot\nobserved_visual: two scatter plots",
    }

    query = "Mục tiêu và kiến trúc tổng thể của mô hình là gì?"
    assert figure_relevance_score(architecture, query=query) > figure_relevance_score(
        plot,
        query=query,
    )
