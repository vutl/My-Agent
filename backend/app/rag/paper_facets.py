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
            "architectures",
            "kiến trúc",
            "kien truc",
            "pipeline",
            "framework",
            "cấu trúc",
            "cau truc",
            "model design",
            # Bare "model"/"mô hình"/"design" appear in almost every research
            # question ("which dataset was the model evaluated on?") and must
            # not add an architecture obligation. Use component/structure
            # phrasing instead.
            "model structure",
            "model components",
            "components of the model",
            "how does the model work",
            "how the model works",
            "module",
            "modules",
            "encoder",
            "decoder",
            "fusion",
            "thành phần",
            "thanh phan",
            "cấu tạo",
            "cau tao",
            "mô hình gồm",
            "mo hinh gom",
            "mô hình hoạt động",
            "mo hinh hoat dong",
            "thiết kế",
            "thiet ke",
            "designed",
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
            # Formula requests: losses, objectives and equations live in the
            # method/optimization part of a paper.
            "formula",
            "formulas",
            "equation",
            "equations",
            "công thức",
            "cong thuc",
            "phương trình",
            "phuong trinh",
            "loss",
            "objective",
            "minimize",
            "minimizes",
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
            "visual",
            "visual grounding",
            "image",
            "images",
            "image-text",
            "image text",
            "pixel",
            "grounding",
        ),
    ),
)

FACET_QUERY_TERMS: dict[str, str] = {
    "task": "abstract introduction research problem task input output objective purpose",
    "architecture": "method model architecture pipeline components feature flow fusion design",
    "training_method": "training objective method loss optimizer equation defined as minimize optimization",
    # Keep the result facet semantically broad enough to reach prose that
    # names the actual evaluation measures (for example Pearson correlation,
    # precision/recall or macro averages) even when a table caption only says
    # ``tasks`` or ``baseline performance``.  These are generic metric
    # synonyms, not paper- or benchmark-specific answer terms.
    "benchmark_results": (
        "benchmark results metrics comparison performance evaluation "
        "accuracy precision recall f1 ccc pearson correlation macro average"
    ),
    "dataset_setup": "dataset split labels modalities evaluation experimental setup",
    "contributions": "main contributions novelty proposed method",
    "ablation": "ablation component contribution without module",
    "limitations": "limitations failure cases threats validity",
    "visual_evidence": (
        "figure diagram caption architecture plot visual image images "
        "grounding image-text pixel"
    ),
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
    "what is",
    "what does",
    "about",
    "purpose",
    "mục đích",
    "muc dich",
    "nói về",
    "noi ve",
    "giải quyết",
    "giai quyet",
    "làm gì",
    "lam gi",
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


# Generic nouns that point at the model/method only when nothing more specific
# is asked: "How many layers does the model have?" is architecture, but
# "Which dataset was the model evaluated on?" is only a dataset question.
_WEAK_ARCHITECTURE_MARKERS = (
    "model", "models", "mô hình", "mo hinh", "approach", "design", "structure",
    "layer", "layers",
)
# Filenames/titles ("…Mapping Network for SER.pdf") are identity, not intent.
_FILENAME_SPAN_RE = re.compile(r"[^?!\n]{0,240}?\.(?:pdf|docx?|md|markdown|txt|pptx?)(?!\w)", re.I)


def extract_query_facets(query: str, *, include_weak: bool = True) -> list[str]:
    """Return explicitly mentioned facets in surface order.

    With ``include_weak``, a question with no explicit facet that still names
    the model/approach falls back to ``architecture``.
    """

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
    if len(located) > 1 and any(facet == "task" for _, facet in located):
        # ``performance on supervised tasks`` refers to an evaluation axis,
        # not a request for the paper's research goal. Bare task nouns are
        # ambiguous; preserve explicit task questions/lists, but do not turn
        # a prepositional context into an extra task/contribution obligation.
        task_markers = dict(FACET_MARKERS)["task"]
        has_explicit_task_role = any(
            _facet_marker_position(lowered, marker) >= 0
            for marker in task_markers if marker not in {"task", "tasks"}
        )
        task_mentions = list(re.finditer(r"(?<!\w)tasks?(?!\w)", lowered))
        only_contextual_mentions = bool(task_mentions) and all(
            re.search(r"\b(?:on|in|for|with|across|over|during)"
                      r"(?:\s+[\w-]+){0,3}\s+$", lowered[:match.start()])
            and not re.search(r"\b(?:what|which|whose)\s+$", lowered[:match.start()])
            for match in task_mentions
        )
        if only_contextual_mentions and not has_explicit_task_role:
            located = [(position, facet) for position, facet in located if facet != "task"]
    without_filenames = _FILENAME_SPAN_RE.sub(" ", lowered)
    if not located and include_weak and any(
        _facet_marker_position(without_filenames, marker) >= 0
        for marker in _WEAK_ARCHITECTURE_MARKERS
    ):
        return ["architecture"]
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


def retrieval_query_expansion(
    query: str,
    *,
    answer_intent: str | None = None,
    focused_document_count: int = 0,
) -> tuple[str, list[str]]:
    """Add bounded structural anchors for underspecified paper questions.

    This is a deterministic retrieval aid, not an answer or qrel lookup. It
    only appends the shared facet vocabulary when the query explicitly asks
    for a facet, is a paper comparison, or is a generic paper-overview
    question. Exact table/figure/number tokens already present in the query
    remain untouched in the original lexical branch.
    """

    text = " ".join(str(query or "").split())
    if not text:
        return "", []
    lowered = f" {text.lower()} "
    explicit = extract_query_facets(text)
    comparison = answer_intent == "compare" or any(
        marker in lowered for marker in _GENERAL_COMPARE_MARKERS
    )
    generic_overview = any(
        marker in lowered for marker in _GENERIC_PAPER_OVERVIEW_MARKERS
    )
    if not (explicit or comparison or generic_overview or focused_document_count >= 2):
        return text, []
    facets = requested_paper_facets(
        text,
        answer_intent=answer_intent,
        focused_document_count=focused_document_count,
    )
    terms = facet_query_terms(facets)
    if not terms:
        return text, facets
    return f"{text} {terms}", facets
