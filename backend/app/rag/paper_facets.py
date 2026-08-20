"""Shared paper-facet vocabulary for indexing, retrieval, prompts and evals.

Keep this module dependency-light.  Several routing services import it before
the agent graph is built, so duplicating marker lists elsewhere creates subtle
intent/coverage drift.
"""

from __future__ import annotations

from collections.abc import Iterable
import re


CORE_PAPER_FACETS: tuple[str, ...] = (
    "task",
    "architecture",
    "dataset_setup",
    "benchmark_results",
    "contributions",
)

AUXILIARY_PAPER_FACETS: tuple[str, ...] = (
    "training_method",
    "ablation",
    "limitations",
    "visual_evidence",
)

ALL_PAPER_FACETS = CORE_PAPER_FACETS + AUXILIARY_PAPER_FACETS

FACET_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "task",
        (
            "research question",
            "problem statement",
            "what does",
            "what is the task",
            "what problem",
            "tasks",
            "task",
            "làm gì",
            "lam gi",
            "bài toán",
            "bai toan",
            "mục tiêu",
            "muc tieu",
        ),
    ),
    (
        "architecture",
        (
            "architecture",
            "kiến trúc",
            "kien truc",
            "pipeline",
            "framework",
            "cấu trúc",
            "cau truc",
            "model design",
        ),
    ),
    (
        "training_method",
        (
            "training objective",
            "training method",
            "loss function",
            "methodology",
            "phương pháp",
            "phuong phap",
            "huấn luyện",
            "huan luyen",
            "distillation",
            "optimizer",
        ),
    ),
    (
        "benchmark_results",
        (
            "benchmark",
            "results",
            "result",
            "kết quả",
            "ket qua",
            "accuracy",
            "f1",
            "ccc",
            "uar",
            "war",
            "wer",
            "performance",
        ),
    ),
    (
        "dataset_setup",
        (
            "dataset",
            "datasets",
            "dữ liệu",
            "du lieu",
            "evaluation setup",
            "experimental setup",
            "data split",
            "protocol",
        ),
    ),
    (
        "contributions",
        (
            "contribution",
            "contributions",
            "novelty",
            "đóng góp",
            "dong gop",
            "điểm mới",
            "diem moi",
            "propose",
            "proposed",
        ),
    ),
    (
        "ablation",
        (
            "ablation",
            "loại bỏ thành phần",
            "loai bo thanh phan",
            "without component",
        ),
    ),
    (
        "limitations",
        (
            "limitations",
            "limitation",
            "hạn chế",
            "han che",
            "failure cases",
        ),
    ),
    (
        "visual_evidence",
        (
            "figure",
            "fig.",
            "diagram",
            "sơ đồ",
            "so do",
            "biểu đồ",
            "bieu do",
        ),
    ),
)

FACET_QUERY_TERMS: dict[str, str] = {
    "task": "research problem task input output objective",
    "architecture": "architecture pipeline components feature flow fusion",
    "training_method": "training objective method loss optimizer",
    "benchmark_results": "benchmark results metrics comparison performance",
    "dataset_setup": "dataset split labels modalities evaluation experimental setup",
    "contributions": "main contributions novelty proposed method",
    "ablation": "ablation component contribution without module",
    "limitations": "limitations failure cases threats validity",
    "visual_evidence": "figure diagram caption architecture plot",
}

_GENERAL_COMPARE_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "so sánh",
    "so sanh",
    "đối chiếu",
    "doi chieu",
    "contrast",
    "differences",
    "similarities",
    "khác nhau",
    "khac nhau",
)

_GENERIC_PAPER_OVERVIEW_MARKERS = (
    "paper làm gì",
    "bài làm gì",
    "bài này",
    "paper này",
    "overview",
    "summarize the paper",
    "tóm tắt bài",
    "tom tat bai",
)


def normalize_facet(value: str) -> str | None:
    normalized = " ".join(str(value or "").lower().replace("_", " ").split())
    if not normalized:
        return None
    for facet, markers in FACET_MARKERS:
        if normalized == facet.replace("_", " ") or any(
            _facet_marker_position(normalized, marker) >= 0 for marker in markers
        ):
            return facet
    candidate = normalized.replace(" ", "_")
    return candidate if candidate in ALL_PAPER_FACETS else None


def normalize_facets(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        facet = normalize_facet(value)
        if facet and facet not in result:
            result.append(facet)
    return result


def extract_query_facets(query: str) -> list[str]:
    """Return explicitly mentioned facets in surface order."""

    lowered = " ".join(str(query or "").lower().split())
    located: list[tuple[int, str]] = []
    for facet, markers in FACET_MARKERS:
        positions = [
            position
            for marker in markers
            if (position := _facet_marker_position(lowered, marker)) >= 0
        ]
        if positions:
            located.append((min(positions), facet))
    return [facet for _, facet in sorted(located)]


def _facet_marker_position(text: str, marker: str) -> int:
    parts = [part for part in str(marker).split() if part]
    if not parts:
        return -1
    pattern = r"\s+".join(re.escape(part) for part in parts)
    match = re.search(rf"(?<!\w){pattern}(?!\w)", text)
    return match.start() if match is not None else -1


def requested_paper_facets(
    query: str,
    *,
    answer_intent: str | None = None,
    focused_document_count: int = 1,
) -> list[str]:
    """Infer the smallest useful canonical facet set for a grounded turn.

    Exact table/figure selection is intentionally handled by the artifact path;
    this helper describes semantic paper coverage, not artifact identity.
    """

    explicit = extract_query_facets(query)
    if explicit:
        # A paper's goal and its contribution are complementary views of the
        # same high-level question. Natural requests such as "what does it do?"
        # and "explain its contribution" routinely require both, even when the
        # surface text names only one. Keep the requested facet first and add
        # only this bounded semantic companion pair.
        if any(facet in {"task", "contributions"} for facet in explicit):
            explicit.extend(
                facet
                for facet in ("task", "contributions")
                if facet not in explicit
            )
        return explicit

    lowered = f" {' '.join(str(query or '').lower().split())} "
    comparison = answer_intent == "compare" or any(
        marker in lowered for marker in _GENERAL_COMPARE_MARKERS
    )
    if comparison or focused_document_count >= 2:
        return list(CORE_PAPER_FACETS)
    if any(marker in lowered for marker in _GENERIC_PAPER_OVERVIEW_MARKERS):
        return ["task", "contributions"]
    return ["task", "contributions"]


def facet_query_terms(facets: Iterable[str]) -> str:
    normalized = normalize_facets(facets)
    return " ".join(FACET_QUERY_TERMS[facet] for facet in normalized)
