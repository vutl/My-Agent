import re
from typing import Annotated, Any
from app.rag.paper_facets import extract_query_facets, facet_query_terms, requested_paper_facets
from app.services.query_rewrite_service import (
    QueryRewriteResult,
    QueryRewriteService,
    _classify_answer_depth,
    _classify_answer_intent,
    _explicit_document_target_entities,
    _is_named_entity,
    _looks_like_topic_switch,
    _query_named_entities,
    enrich_retrieval_query,
    format_recent_conversation,
    has_result_table_intent,
    has_visual_intent,
    requires_visual_pixels,
    wants_single_figure,
)


_EXACT_TABLE_REQUEST_RE = re.compile(
    r"(?<!\w)(?:bảng|table)\s*(?:số\s*)?#?\s*(?P<number>\d+)(?!\w)",
    re.IGNORECASE,
)
_TABLE_NUMBER_SEQUENCE_RE = re.compile(
    r"(?<!\w)(?:bảng|tables?)\s*(?:số\s*)?#?\s*"
    r"(?P<sequence>\d+(?:\s*(?:,|;|\+|&|và|va|với|voi|and|with)\s*"
    r"(?:(?:bảng|tables?)\s*(?:số\s*)?#?\s*)?\d+)*)",
    re.IGNORECASE,
)
_SHOW_TABLE_RE = re.compile(
    r"(?<!\w)(?:show|display|give\s+me|print|list|reproduce|copy|"
    r"(?:cho\s+(?:mình|tôi|em|tớ|minh|toi)\s+)?xem|cho(?:\s+(?:mình|tôi|em|tớ|minh|toi))?|"
    r"muốn\s+xem|đưa|dua|hiển\s*thị|hien\s*thi|in\s+ra|xuất|lấy|chép|trích)"
    r"(?!\w)[^?]{0,80}(?<!\w)(?:table|tables|bảng|bang)(?!\w)|"
    r"(?<!\w)(?:table|bảng|bang)\s*#?\d+\s*(?:đâu|please|nhé|nha|giúp)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_TABLE_QUESTION_RE = re.compile(
    r"(?<!\w)(?:gì|nào|bao\s*nhiêu|sao|thế\s*nào|what|which|how|who|whom|why|when|does|do|is|are|can)(?!\w)[^.!]*\?\s*$",
    re.IGNORECASE,
)
_TABLE_ANALYSIS_RE = re.compile(
    r"(?<!\w)(?:giải\s*thích|phân\s*tích|đánh\s*giá|nhận\s*xét|ý\s*nghĩa|"
    r"explain|analy[sz]e|evaluate|interpret|"
    # A question about the table's content is answered, not rendered.
    r"tốt\s*nhất|cao\s*nhất|thấp\s*nhất|kém\s*nhất|đúng\s*không|phải\s*không|hơn\s*bao\s*nhiêu|"
    r"thắng|vượt|best|worst|highest|lowest|beats?|outperforms?|better|right\?|correct\?|"
    r"how\s+much|which\s+(?:model|method|variant|row|k)|(?:model|method|variant)\s+nào|k\s+nào)(?!\w)",
    re.IGNORECASE,
)
_GENERIC_RESULT_TABLE_REQUEST_RE = re.compile(
    r"(?:bảng|table).{0,64}(?:kết\s*quả|result|benchmark|performance)|"
    r"(?:kết\s*quả|result|benchmark|performance).{0,64}(?:bảng|table)",
    re.IGNORECASE,
)
_MAIN_RESULT_TABLE_CAPTION_RE = re.compile(
    r"(?:main\s+results?|experimental\s+results?|comparison|comparative|"
    r"performance|benchmark)",
    re.IGNORECASE,
)
_NON_MAIN_RESULT_TABLE_CAPTION_RE = re.compile(
    r"(?:ablation|distribution|statistics|dataset\s+(?:summary|distribution)|"
    r"hyperparameters?|training\s+settings?)",
    re.IGNORECASE,
)


def _requested_metric_names(task: str) -> set[str]:
    normalized = " ".join(str(task or "").casefold().split())
    aliases = {
        "accuracy": r"(?<!\w)(?:acc|accuracy)(?!\w)",
        "f1": r"(?<!\w)f1(?!\w)",
        "ccc": r"(?<!\w)ccc(?!\w)",
        "uar": r"(?<!\w)uar(?!\w)",
        "wa": r"(?<!\w)(?:wa|war)(?!\w)",
        "wer": r"(?<!\w)wer(?!\w)",
        "mae": r"(?<!\w)mae(?!\w)",
        "mse": r"(?<!\w)mse(?!\w)",
        "rmse": r"(?<!\w)rmse(?!\w)",
        "precision": r"(?<!\w)precision(?!\w)",
        "recall": r"(?<!\w)recall(?!\w)",
        "auc": r"(?<!\w)auc(?!\w)",
        "eer": r"(?<!\w)eer(?!\w)",
        "hamming_loss": r"(?<!\w)(?:hamming\s+loss|hl)(?!\w)",
        "jaccard": r"(?<!\w)(?:jaccard(?:\s+index)?|jac)(?!\w)",
        "exact_accuracy": r"(?<!\w)(?:exact\s+(?:match\s+)?accuracy|exacc)(?!\w)",
        "binary_accuracy": r"(?<!\w)(?:binary\s+accuracy|per-label\s+accuracy|biacc)(?!\w)",
        "parameters": r"(?<!\w)(?:params?|parameters?)(?!\w)",
    }
    return {
        metric
        for metric, pattern in aliases.items()
        if re.search(pattern, normalized)
    }


def _is_canonical_result_table_request(query: str) -> bool:
    """Return whether a result-shaped query can use canonical table rendering.

    A user does not have to say ``table`` to ask for a benchmark table.  Metric
    follow-ups such as ``benchmark Acc, F1 and CCC`` are common after a paper
    has been focused.  They should receive the same deterministic artifact
    path as an explicit table request, otherwise a stochastic model can pick
    an ablation table or trigger a quantitative fallback.  We still require a
    result/table intent plus at least one named metric, and decline analytical
    prompts where a prose interpretation is expected.
    """

    normalized = " ".join(str(query or "").casefold().split())
    if not normalized or _TABLE_ANALYSIS_RE.search(normalized) or "ablation" in extract_query_facets(normalized):
        return False
    if _GENERIC_RESULT_TABLE_REQUEST_RE.search(normalized):
        return True
    return bool(has_result_table_intent(normalized) and _requested_metric_names(normalized))


_MAIN_RESULT_TABLES_PER_DOCUMENT = 2


def _wants_main_result_evidence(query: str) -> bool:
    """Gather primary result evidence even when analysis requires generation.

    Rendering eligibility must not gate retrieval. A benchmark/metric or
    comparative-results request can need a main table outside vector top-k.
    Vague structured facts and explicit ablations retain their own frontier.
    """
    facets = set(extract_query_facets(query))
    if "ablation" in facets or "benchmark_results" not in facets:
        return False
    return bool(
        _GENERIC_RESULT_TABLE_REQUEST_RE.search(query)
        or _requested_metric_names(query)
        or re.search(r"(?<!\w)(?:benchmark|performance|compare|comparison|so\s+sánh)(?!\w)", query, re.I)
    )


def _wants_result_tables(query: str) -> bool:
    return has_result_table_intent(query)


def _wants_table_inventory(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    return bool(
        re.search(
            r"(?:mấy|bao\s+nhiêu|liệt\s+kê|danh\s+sách|how\s+many|list|count)"
            r"(?:\s+\S+){0,4}\s+(?:bảng|tables?)\b|"
            r"(?:bảng|tables?)(?:\s+\S+){0,4}\s+"
            r"(?:mấy|bao\s+nhiêu|liệt\s+kê|how\s+many|list|count)\b",
            normalized,
        )
        or re.search(
            r"\b(?:(?:những|các|toàn\s+bộ|tất\s+cả)\s+bảng|bảng\s+nào)\b|"
            r"\b(?:có|gồm|chứa)\s+(?:(?:những|các|toàn\s+bộ|tất\s+cả)\s+)?"
            r"bảng\s+(?:gì|nào)\b|"
            r"\b(?:what|which)\s+tables?\b|"
            r"\b(?:show|display|give)\s+(?:me\s+)?(?:all|the)\s+tables?\b|"
            r"\ball\s+tables?\b",
            normalized,
        )
    )


def _caption_table_number(caption: str) -> int | None:
    match = _EXACT_TABLE_REQUEST_RE.search(str(caption or ""))
    return int(match.group("number")) if match is not None else None


_DESCRIPTION_STOPWORDS = frozenset(
    "show give me the a an of on in for with and or to table bảng cho mình xem đưa "
    "please can you what is are this that its it về của các những trên với và "
    "bài paper nào nhất tốt từng mỗi bao nhiêu which best".split()
)


def _description_words(text: str) -> set[str]:
    words = re.findall(r"[0-9a-zà-ỹđ]+(?:-[0-9a-zà-ỹđ]+)*", str(text or "").casefold())
    return {word for word in words if len(word) >= 3 and word not in _DESCRIPTION_STOPWORDS
            and not word.isdigit()}


def _described_table_numbers(
    query: str, tables: list[dict[str, Any]], requested: tuple[int, ...],
) -> tuple[int, ...]:
    """Trust the user's description over a misremembered table number.

    "Table 1, the experimental results on tonal languages" when Table 1 is the
    data distribution and Table 2 is the results: the described table wins
    when the numbered one shares no description word and another table of the
    same document shares at least two, or when another table matches every
    description word the numbered one matches plus a distinguishing one
    ("Table 5, fully-supervised results on IEMOCAP" when Table 5 is the
    semi-supervised IEMOCAP table and Table 4 the fully-supervised one).
    """
    if not requested or not tables:
        return requested
    description = _description_words(_TABLE_NUMBER_SEQUENCE_RE.sub(" ", query))
    if len(description) < 2:
        return requested
    by_number: dict[int, list[dict[str, Any]]] = {}
    for table in tables:
        number = _canonical_table_number(table)
        if number is not None:
            by_number.setdefault(number, []).append(table)

    def matched(tables_with_number: list[dict[str, Any]]) -> set[str]:
        # Caption plus header row: "IEMOCAP" in a table's columns describes it
        # as much as a caption word does.
        words: set[str] = set()
        for table in tables_with_number:
            header = str(table.get("markdown") or table.get("content") or "")[:600]
            words |= description & _description_words(f"{table.get('caption') or ''} {header}")
        return words

    corrected: list[int] = []
    for number in requested:
        own = matched(by_number.get(number, []))
        best_number, best_words = number, own
        for other_number, others in by_number.items():
            words = matched(others)
            if len(words) > len(best_words):
                best_number, best_words = other_number, words
        if best_number != number and (
            (not own and len(best_words) >= 2) or (own and own < best_words)
        ):
            number = best_number
        if number not in corrected:
            corrected.append(number)
    return tuple(corrected)


def _requested_table_numbers(query: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for match in _TABLE_NUMBER_SEQUENCE_RE.finditer(str(query or "")):
        for raw_number in re.findall(r"\d+", match.group("sequence")):
            number = int(raw_number)
            if number not in numbers:
                numbers.append(number)
    return tuple(numbers)


def _first_markdown_table_block(text: str) -> str:
    """Return the first contiguous Markdown table (header..last row)."""
    block: list[str] = []
    for line in str(text or "").splitlines():
        if line.strip().startswith("|") or (block and line.count("|") >= 2):
            block.append(line.rstrip())
        elif block:
            break
    return "\n".join(block) if block else str(text or "")


def _canonical_table_number(table: dict[str, Any]) -> int | None:
    """Prefer the table number printed in the caption over positional index."""

    caption_number = _caption_table_number(str(table.get("caption") or ""))
    if caption_number is not None:
        return caption_number
    table_index = table.get("table_index")
    if isinstance(table_index, int) or str(table_index or "").isdigit():
        return int(table_index) + 1
    return None

