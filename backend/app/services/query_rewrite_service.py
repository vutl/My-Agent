from dataclasses import dataclass
import json
import re
from typing import Any

from app.llm.ollama_client import OllamaClient
from app.rag.paper_facets import extract_query_facets, facet_query_terms
from app.services.conversation_state import looks_like_document_resume
from app.services.document_language import looks_like_document_comparison


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    standalone_query: str
    is_followup: bool
    current_topic: str | None
    required_entities: list[str]
    use_last_sources: bool
    answer_intent: str
    answer_depth: str
    rewrite_used: bool
    diagnostics: dict[str, Any]


_EXPLAIN_MARKERS = (
    "giải thích",
    "explain",
    "là gì",
    "what is",
    "what's",
    "mô tả",
    "describe",
    "overview",
    "tổng quan",
)

_VISUAL_MARKERS = (
    "ảnh",
    "figure",
    "figures",
    "fig",
    "fig.",
    "fig ",
    "diagram",
    "diagrams",
    "sơ đồ",
    "chart",
    "charts",
    "biểu đồ",
    "plot",
    "plots",
    # Public slide/document corpora often call page imagery a "picture",
    # "photo", "screenshot" or "slide" without using the paper-centric
    # figure/diagram vocabulary.  Keep these content-bearing visual cues in
    # the shared intent policy so visual sufficiency is not bypassed by a
    # format-specific noun.
    "picture",
    "pictures",
    "photo",
    "photos",
    "screenshot",
    "slide",
    "slides",
    # Research-topic words ("multimodal", "vision", "visual grounding",
    # "image-text") describe what a paper studies, not a request to look at
    # pixels. They must not route ordinary text questions to the visual
    # channel or its fail-closed fallback.
)

_MODEL_PHRASE_RE = re.compile(
    r"(?<!\w)(?:mô\s+hình|mo\s+hinh)(?!\w)"
    # Modality compounds name a research topic ("image-text retrieval",
    # "vision-language model"), not an artifact to look at.
    r"|(?<!\w)(?:image|vision|visual)[- ](?:to[- ])?(?:text|language|caption(?:ing)?)(?!\w)",
    flags=re.IGNORECASE,
)
_IMAGE_WORD_RE = re.compile(r"(?<!\w)(?:hình|hinh)(?!\w)", flags=re.IGNORECASE)

# Arbitrary documents often ask about an observable property without naming a
# figure (for example, a person wearing a coloured shirt on a page, or the
# words printed in a highlighted colour). Keep this vocabulary generic and
# require a scope/companion cue so ordinary text questions about people or
# datasets do not automatically become multimodal requests.
# Strong cues describe appearance and are pixel-dependent wherever they occur
# on a page. Weak cues (people, text, words) are ordinary in text questions
# ("who wrote this document?", "tóm tắt nội dung từ trang 3") and only count
# inside an explicitly visual scope (photo, slide, chart), never a bare
# page/document. "từ" ("from") is deliberately absent.
_VISUAL_STRONG_OBSERVATION_RE = re.compile(
    r"(?<!\w)(?:"
    r"red|yellow|blue|green|orange|purple|black|white|brown|pink|"
    r"colour|color|shirt|t-shirt|wearing|photograph|photographs|"
    r"photo|photos|logo|highlighted|pictured|"
    r"màu|áo|mặc|nhìn\s+thấy"
    r")(?!\w)",
    flags=re.IGNORECASE,
)
_VISUAL_WEAK_OBSERVATION_RE = re.compile(
    r"(?<!\w)(?:people|person|man|woman|word|words|text|visible|"
    r"người|chữ|xuất\s+hiện)(?!\w)",
    flags=re.IGNORECASE,
)
_VISUAL_OBSERVATION_RE = re.compile(
    rf"{_VISUAL_STRONG_OBSERVATION_RE.pattern}|{_VISUAL_WEAK_OBSERVATION_RE.pattern}",
    flags=re.IGNORECASE,
)
_VISUAL_SCOPE_RE = re.compile(
    r"(?<!\w)(?:page|pages|slide|slides|picture|pictures|image|images|"
    r"photo|photograph|figure|diagram|chart|plot|"
    r"trang|hình|ảnh|bức\s+hình|biểu\s+đồ|sơ\s+đồ)(?!\w)",
    flags=re.IGNORECASE,
)
_VISUAL_MEDIA_SCOPE_RE = re.compile(
    r"(?<!\w)(?:slide|slides|picture|pictures|image|images|photo|photograph|"
    r"figure|diagram|chart|plot|hình|ảnh|bức\s+hình|biểu\s+đồ|sơ\s+đồ)(?!\w)",
    flags=re.IGNORECASE,
)
_VISUAL_COLOR_WORD_RE = re.compile(
    r"(?<!\w)(?:red|yellow|blue|green|orange|purple|black|white|brown|pink|"
    r"colour|color|màu)(?!\w)",
    flags=re.IGNORECASE,
)

_VISUAL_PIXEL_MARKERS = (
    "what is shown",
    "what does the image show",
    "what does the picture show",
    "what does the figure show",
    "depict",
    "pictured",
    "look like",
    "appearance",
    "visual content",
    "cho tôi xem",
    "xem hình",
    "đưa hình",
    "hiển thị hình",
    "trông như",
    "thể hiện trong hình",
    "hình cho thấy",
    "ảnh cho thấy",
    "mô tả hình",
    "mô tả ảnh",
    "biểu đồ cho thấy",
    "sơ đồ thể hiện",
)
_VISUAL_SHOW_RE = re.compile(
    r"\bwhat\s+does\b.{0,80}\bshow(?:s|ing)?\b"
    # "show me the accuracy" asks for a value; only showing a visual is pixels.
    r"|\bshow\s+(?:me\s+)?(?:the\s+|a\s+|an\s+)?(?:[\w-]+\s+){0,2}"
    r"(?:figure|image|picture|photo|diagram|chart|plot|slide|screenshot)s?\b",
    flags=re.IGNORECASE,
)

# A visual noun alone is not enough to require pixels: questions such as
# ``what is the caption of Figure 2?`` are answerable from canonical metadata.
# When the question explicitly asks for something *in/from/on* the visual,
# however, a caption/OCR snippet is not a safe substitute for the image.  Keep
# this relation-based detector language- and format-agnostic rather than adding
# benchmark or paper-specific words.
_VISUAL_RELATION_RE = re.compile(
    r"(?:"
    r"\b(?:in|on|from|within|inside|according\s+to|based\s+on)\s+"
    r"(?:the\s+)?(?:picture|photo|image|figure|diagram|chart|plot|slide)s?\b"
    r"|\b(?:picture|photo|image|figure|diagram|chart|plot|slide)s?\s+"
    r"(?:shows?|depicts?|contains?|includes?|illustrates?)\b"
    r"|\b(?:trong|từ|từ|theo|dựa\s+trên)\s+"
    r"(?:hình|ảnh|bức\s+hình|biểu\s+đồ|sơ\s+đồ|figure|chart|slide)\b"
    r"|\b(?:hình|ảnh|biểu\s+đồ|sơ\s+đồ|figure|chart|slide)\s+"
    r"(?:cho\s+thấy|thể\s+hiện|mô\s+tả|chứa|gồm)\b"
    r")",
    flags=re.IGNORECASE,
)

_RESULT_TABLE_MARKERS = (
    "bảng",
    "table",
    "benchmark",
    "kết quả",
    # Natural questions often ask for the evaluation information without
    # saying ``table`` explicitly (for example, ``which metrics were used``
    # or ``how did performance compare``).  These are content-bearing result
    # cues, not paper- or benchmark-specific aliases, so they should activate
    # the same canonical table reservation as an explicit table request.
    "metric",
    "score",
    "performance",
    "result",
    # Quantitative questions are often phrased without naming a table.  These
    # cues cover ratios, rates and reported values while remaining independent
    # of any paper's model/dataset vocabulary.
    "ratio",
    "percentage",
    "percent",
    "proportion",
    "accuracy",
    "compare with",
    "so với",
    "baseline",
    "ablation",
)

_RESULT_METRIC_PATTERN = re.compile(
    r"(?<!\w)(?:acc|f1|ccc|uar|war|wa|ua|wer|mae|mse|rmse)(?!\w)",
    flags=re.IGNORECASE,
)

# Explicit list/count requests. Interrogatives such as "what"/"which" or
# Vietnamese plural articles only count when they introduce a plural domain
# noun ("which languages", "các bộ dữ liệu"); "what is the model
# architecture?" is an explanation, not an enumerated evidence request.
_ENUMERATED_REQUEST_MARKERS = (
    "list", "enumerate", "how many", "name all", "identify all",
    "liệt kê", "kể tên", "bao nhiêu",
)
_ENUMERATED_DOMAIN_MARKERS = (
    "languages", "ngôn ngữ", "classes", "lớp", "categories", "models",
    "mô hình", "datasets", "bộ dữ liệu", "methods", "phương pháp", "sources",
    "nguồn", "metrics", "chỉ số", "values", "items",
)
_ENUMERATED_PLURAL_ASK_RE = re.compile(
    r"(?<!\w)(?:(?:which|what)\s+(?:[\w-]+\s+){0,2}"
    r"(?:languages|classes|categories|models|datasets|methods|sources|metrics|values|items)"
    r"|(?:các|những)\s+(?:ngôn\s+ngữ|lớp|mô\s+hình|bộ\s+dữ\s+liệu|phương\s+pháp|nguồn|chỉ\s+số)"
    r"(?:\s+[\w-]+){0,4}\s+nào)(?!\w)",
    flags=re.IGNORECASE,
)
# Explanations of how something is built or done want prose/diagrams.
_EXPLANATION_FACETS = frozenset({"architecture", "training_method"})
_STRUCTURED_FACETS = frozenset({"benchmark_results", "dataset_setup"})
_ENUMERATED_DOCUMENT_MARKERS = (
    "paper", "papers", "article", "articles", "study", "studies",
    "author", "authors", "publication", "publications",
)


def has_enumerated_evidence_intent(query: str | None) -> bool:
    """Detect generic list/count asks that benefit from structured evidence.

    This is deliberately vocabulary-based rather than tied to a benchmark,
    filename, or paper.  A list request about languages/models/metrics should
    reserve a table when one is available; ordinary factual questions do not.
    """

    lowered = " ".join(str(query or "").casefold().split())
    if not lowered:
        return False
    # Document-discovery questions such as "which paper introduced ..."
    # should stay in the text/card path; a table reservation would be noise.
    if any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered)
        for marker in _ENUMERATED_DOCUMENT_MARKERS
    ):
        return False
    facets = set(extract_query_facets(lowered, include_weak=False))
    if facets & _EXPLANATION_FACETS and not facets & _STRUCTURED_FACETS:
        return False
    if _ENUMERATED_PLURAL_ASK_RE.search(lowered):
        return True
    asks_for_list = any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered)
        for marker in _ENUMERATED_REQUEST_MARKERS
    )
    has_domain = any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered)
        for marker in _ENUMERATED_DOMAIN_MARKERS
    )
    return asks_for_list and has_domain

_VISUAL_NOUN_PATTERN = (
    r"(?:hình|hinh|figure|figures|diagram|diagrams|sơ\s+đồ|so\s+do|"
    r"chart|charts|plot|plots|image|images)"
)
_EXPLICIT_SINGLE_VISUAL_RE = re.compile(
    rf"(?:"
    rf"(?<!\w)(?:đúng\s+|chỉ\s+)?(?:một|1)\s+{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf"|(?<!\w)(?:only\s+|exactly\s+|just\s+)?(?:one|a\s+single)\s+"
    rf"(?:[\w-]+\s+){{0,4}}{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf")",
    flags=re.IGNORECASE,
)
_EXPLICIT_MULTI_VISUAL_RE = re.compile(
    rf"(?:"
    rf"(?<!\w)(?:các|những|nhiều|một\s+vài|vài|several|multiple|a\s+few|some)\s+"
    rf"(?:[\w-]+\s+){{0,3}}{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf"|(?<!\w)(?:top\s+)?(?:[2-9]|[1-9]\d+)\s+"
    rf"(?:[\w-]+\s+){{0,3}}{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf"|(?<!\w)(?:two|three|four|five|six|seven|eight|nine|ten)\s+"
    rf"(?:[\w-]+\s+){{0,3}}{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf")",
    flags=re.IGNORECASE,
)
_BEST_VISUAL_RE = re.compile(
    rf"(?:"
    rf"{_VISUAL_NOUN_PATTERN}(?:\s+\S+){{0,5}}\s+"
    rf"(?:phù\s+hợp|phu\s+hop|thích\s+hợp|thich\s+hop|liên\s+quan|lien\s+quan|"
    rf"tốt|tot|rõ|ro|quan\s+trọng|quan\s+trong)\s+nhất"
    rf"|(?<!\w)(?:best|most\s+relevant|most\s+useful|most\s+representative)\s+"
    rf"(?:[\w-]+\s+){{0,4}}{_VISUAL_NOUN_PATTERN}(?!\w)"
    rf"|{_VISUAL_NOUN_PATTERN}(?:\s+\S+){{0,4}}\s+"
    rf"(?:best|most\s+relevant|most\s+useful|most\s+representative)(?!\w)"
    rf")",
    flags=re.IGNORECASE,
)


def has_visual_intent(query: str) -> bool:
    lowered = " ".join(query.lower().split())
    # Vietnamese "mô hình" means *model*, not a request to show an image.
    # Remove only that phrase; an explicit "sơ đồ/figure/ảnh" elsewhere still
    # turns visual retrieval on.
    without_model_phrase = _MODEL_PHRASE_RE.sub(" ", lowered)
    # Markers include generic modality terms (``visual``, ``vision`` and
    # ``image text``).  Token-aware matching prevents accidental activation
    # for unrelated words such as ``television`` or ``provision`` while still
    # accepting punctuation/hyphenated forms.
    marker_intent = any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", without_model_phrase)
        for marker in _VISUAL_MARKERS
    ) or bool(_IMAGE_WORD_RE.search(without_model_phrase))
    if marker_intent:
        return True
    strong = bool(_VISUAL_STRONG_OBSERVATION_RE.search(without_model_phrase))
    weak = bool(_VISUAL_WEAK_OBSERVATION_RE.search(without_model_phrase))
    if not (strong or weak):
        return False
    # Appearance cues plus any scope ("shirt on page 14") or weak cues plus a
    # media scope ("text on the slide", "people in the photo") are visual.
    if strong and _VISUAL_SCOPE_RE.search(without_model_phrase):
        return True
    if weak and _VISUAL_MEDIA_SCOPE_RE.search(without_model_phrase):
        return True
    # Coloured/highlighted words are layout-dependent even without a noun.
    return bool(
        _VISUAL_COLOR_WORD_RE.search(without_model_phrase)
        and re.search(r"(?<!\w)(?:word|words|text|chữ)(?!\w)", without_model_phrase)
    )


def requires_visual_pixels(query: str) -> bool:
    """Return whether a visual request needs rendered pixels, not just caption metadata.

    Figure IDs/captions can answer artifact-inventory and caption questions.
    Appearance, depiction, and ``what is shown`` questions require an actual
    image/page artifact; treating a caption as visual proof causes silent
    hallucinations when a parser extracted metadata but lost the pixels.
    """

    lowered = " ".join(str(query or "").casefold().split())
    return (
        any(marker in lowered for marker in _VISUAL_PIXEL_MARKERS)
        or bool(_VISUAL_SHOW_RE.search(lowered))
        or bool(_VISUAL_RELATION_RE.search(lowered))
        or has_visual_intent(lowered)
        and bool(_VISUAL_OBSERVATION_RE.search(lowered))
    )


def wants_single_figure(query: str) -> bool:
    """Return whether a visual request explicitly asks for one top attachment.

    This is deliberately phrasing-based rather than paper- or figure-number-
    based.  Explicit plural/count requests win over a loose superlative so
    ``top 3 most relevant figures`` is not collapsed to one.
    """

    lowered = " ".join(query.lower().split())
    if not has_visual_intent(lowered):
        return False
    if _EXPLICIT_SINGLE_VISUAL_RE.search(lowered):
        return True
    if _EXPLICIT_MULTI_VISUAL_RE.search(lowered):
        return False
    return bool(_BEST_VISUAL_RE.search(lowered))


def has_result_table_intent(query: str) -> bool:
    """Detect result/table asks without treating substrings such as ``access`` as Acc."""

    lowered = " ".join(query.lower().split())
    # Keep marker matching token-aware.  The marker set intentionally contains
    # generic words (``result``, ``value`` and ``metric``); raw substring
    # matching would turn words such as ``resulting`` or ``valuation`` into a
    # table request and would make casual/non-RAG routing less predictable.
    if any(
        re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lowered)
        for marker in _RESULT_TABLE_MARKERS
    ):
        return True
    if "benchmark_results" in extract_query_facets(lowered):
        return True
    return bool(_RESULT_METRIC_PATTERN.search(lowered)) or has_enumerated_evidence_intent(lowered)


def enrich_retrieval_query(
    query: str,
    *,
    topic: str | None,
    entities: list[str] | None,
    answer_intent: str,
    focus_document_ids: list[str] | None,
) -> str:
    """Expand scoped questions toward their requested facets, not a default model."""
    lowered = " ".join(query.lower().split())
    topic_name = (topic or "").strip()
    if not topic_name and entities:
        topic_name = str(entities[0]).strip()
    if not topic_name:
        return query

    wants_explanation = any(marker in lowered for marker in _EXPLAIN_MARKERS)
    wants_explanation = wants_explanation or answer_intent in {
        "direct_answer",
        "elaborate",
        "infer_structure",
        "compare",
    }
    if not wants_explanation:
        return query

    scoped = bool(focus_document_ids) or _is_named_entity(topic_name)
    if not scoped:
        return query

    explicit_facets = extract_query_facets(query)

    if has_result_table_intent(query):
        # Result asks should not be diluted with architecture/introduction
        # anchors.  Keeping this query evidence-shaped materially improves the
        # table-only LanceDB channel while remaining paper-agnostic.
        if "ablation" in lowered:
            anchors = f"{topic_name} ablation table results metrics component contribution"
        elif re.search(r"(?<!\w)(?:bảng|table)\s*(?:số\s*)?#?\s*\d+(?!\w)", lowered):
            # The original query already carries the requested number.  Avoid
            # adding "main" or "ablation", either of which can pull retrieval
            # toward a semantically related but differently numbered table.
            anchors = f"{topic_name} table results metrics"
        else:
            anchors = (
                f"{topic_name} main experimental table results metrics "
                "performance comparison baselines benchmark"
            )
        # Result preference is a channel choice, not permission to erase other
        # requested facets (for example dataset + results). Generic `model`
        # mentions must not pull a result query back toward architecture.
        companion_terms = facet_query_terms(
            facet for facet in explicit_facets
            if facet not in {"benchmark_results", "architecture", "visual_evidence"}
        )
        if companion_terms:
            anchors = f"{anchors} {companion_terms}"
    elif has_visual_intent(query):
        anchors = (
            f"{topic_name} figure diagram chart plot table visualization results "
            "benchmark arousal valence CCC architecture overview"
        )
        companion_terms = facet_query_terms(
            facet for facet in explicit_facets
            if facet not in {"visual_evidence", "architecture", "benchmark_results"}
        )
        if companion_terms:
            anchors = f"{anchors} {companion_terms}"
    elif explicit_facets:
        # Shared vocabulary is also used by card coverage and second-hop
        # planning. Do not append training/architecture synonyms to a dataset,
        # limitation or contribution request just because it names a paper.
        anchors = f"{topic_name} {facet_query_terms(explicit_facets)}"
    else:
        anchors = (
            f"{topic_name} introduction abstract architecture model purpose "
            "pipeline components training distillation definition overview"
        )
    if anchors.lower() in lowered:
        return query
    return f"{anchors} {query}".strip()


def _is_named_entity(value: str) -> bool:
    token = value.strip()
    if len(token) < 3:
        return False
    blocked = {
        "aya",
        "pdf",
        "fig",
        "figure",
        "hình",
        "model",
        "paper",
        "architecture",
        "pipeline",
        "diagram",
        "overview",
        "framework",
        "emotion",
        "speech",
        "audio",
        "visual",
        "dataset",
        "recognition",
        "classifier",
        "attention",
        "transformer",
        "fusion",
        "guidance",
        "arousal",
        "valence",
        "dominance",
        # Metrics / result asks — not paper names.
        "benchmark",
        "baseline",
        "accuracy",
        "result",
        "results",
        "table",
        "ablation",
        "comparison",
        "acc",
        "f1",
        "ccc",
        "uar",
        "wer",
        "mae",
        "mse",
        "rmse",
        "sota",
        # Conversation/navigation words and generic evidence vocabulary.
        "quay",
        "return",
        "resume",
        "previous",
        "prior",
        "evidence",
        "source",
        "context",
        "method",
        "approach",
        "system",
        "analysis",
        "access",
        "followup",
        "follow-up",
        "llm",
        "rag",
    }
    if token.lower() in blocked:
        return False
    # Tiny metric tokens with mixed case (Acc, F1, CCC, WA, UA).
    if re.fullmatch(r"(?i)acc|f1|ccc|uar|wa|ua|wer|mae|mse|rmse|sota", token):
        return False
    # Catalog names are strong even when written lowercase (derived from the
    # indexed filenames/keywords, not a fixed list).
    from app.services.rag_service import default_catalog_vocabulary, document_match_tokens

    if document_match_tokens(entities=[], query=token, vocabulary=default_catalog_vocabulary()):
        return True
    if token.lower().endswith(".pdf"):
        return True
    compact = token.replace("_", "").replace("-", "")
    # Acronyms / model IDs (ASPIRE, MSF-SER, GPT4) are strong. A merely
    # title-cased sentence word such as "Quay" or "Evidence" is not.
    if re.fullmatch(r"[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)*", token) and len(compact) >= 2:
        return True
    if re.search(r"[A-Za-z]", token) and re.search(r"\d", token) and len(compact) >= 3:
        return True
    if re.search(r"[a-z][A-Z]", token):
        return True
    if "_" in token and len(compact) >= 4:
        return True
    return False


def _query_named_entities(query: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for entity in _extract_entities(query):
        if not _is_named_entity(entity):
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity)
    # Preserve original casing from the raw query when possible.
    restored: list[str] = []
    for entity in entities:
        match = re.search(rf"\b{re.escape(entity)}\b", query, flags=re.IGNORECASE)
        restored.append(match.group(0) if match else entity)
    return restored


_EXPLICIT_DOCUMENT_TARGET_RE = re.compile(
    r"(?<!\w)(?:bài(?:\s+báo)?|paper|file|document|tài\s+liệu)\s+"
    r"(?:của\s+)?(?P<target>[A-Za-z][A-Za-z0-9_.-]{1,})(?!\w)",
    flags=re.IGNORECASE,
)
_TABLE_ADJACENT_TARGET_RE = re.compile(
    r"(?<!\w)(?:bảng|table)\s*(?:số\s*)?#?\s*\d+\s+"
    r"(?:(?:của|trong)\s+)?(?:bài\s+|paper\s+)?"
    r"(?P<target>[A-Za-z][A-Za-z0-9_.-]{1,})(?!\w)",
    flags=re.IGNORECASE,
)


def _explicit_document_target_entities(query: str) -> list[str]:
    """Return paper names explicitly targeted by the current user turn.

    A name merely appearing in recent history may be a baseline inside another
    paper.  Phrases such as ``bài MSF-SER`` and ``Table 2 ASPIRE`` are stronger:
    they identify the source document the user wants now, independent of sticky
    conversation focus.  For corrections, the last explicit target wins; real
    comparison asks retain every named paper.
    """

    matches: list[tuple[int, str]] = []
    for pattern in (_EXPLICIT_DOCUMENT_TARGET_RE, _TABLE_ADJACENT_TARGET_RE):
        for match in pattern.finditer(query):
            target = match.group("target").strip()
            if _is_named_entity(target):
                matches.append((match.start("target"), target))

    # An explicit PDF filename is always a document target, even without a
    # preceding paper/file noun.
    for match in re.finditer(r"(?<!\w)([A-Za-z][A-Za-z0-9_.-]*\.pdf)(?!\w)", query, re.I):
        matches.append((match.start(1), match.group(1)))

    if not matches:
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for _, target in sorted(matches, key=lambda item: item[0]):
        key = re.sub(r"[^a-z0-9]+", "", target.casefold())
        if key and key not in seen:
            seen.add(key)
            ordered.append(target)

    return ordered if looks_like_document_comparison(query) else ordered[-1:]


@dataclass(frozen=True)
class QueryRewriteService:
    client: OllamaClient
    default_model: str

    async def rewrite(
        self,
        *,
        query: str,
        previous_messages: list,
        model: str | None = None,
        working_topic: str | None = None,
        working_document_hint: str | None = None,
    ) -> QueryRewriteResult:
        heuristic_followup = looks_like_followup(query)
        answer_intent = _classify_answer_intent(query)
        answer_depth = _classify_answer_depth(query)
        explicit_targets = _explicit_document_target_entities(query)
        if explicit_targets:
            return QueryRewriteResult(
                original_query=query,
                standalone_query=query,
                is_followup=False,
                current_topic=explicit_targets[0],
                required_entities=explicit_targets,
                use_last_sources=False,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={"reason": "explicit_document_target"},
            )
        if working_topic and looks_like_document_resume(query):
            heuristic_rewrite = _generic_deepen_rewrite(
                query,
                previous_messages,
                working_topic=working_topic,
            )
            if heuristic_rewrite is None:
                standalone_query = f"{working_topic} {query}".strip()
                required_entities = [working_topic] if _is_named_entity(working_topic) else []
            else:
                _, standalone_query, required_entities = heuristic_rewrite
            return QueryRewriteResult(
                original_query=query,
                standalone_query=standalone_query,
                is_followup=True,
                current_topic=working_topic,
                required_entities=required_entities,
                use_last_sources=True,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={"reason": "resume_working_focus"},
            )
        if _looks_like_topic_switch(query, previous_messages=previous_messages):
            entities = _query_named_entities(query) or _extract_entities(query)
            topic = entities[0] if entities else _explicit_topic(query)
            return QueryRewriteResult(
                original_query=query,
                standalone_query=query,
                is_followup=False,
                current_topic=topic,
                required_entities=entities,
                use_last_sources=False,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={"reason": "topic_switch"},
            )
        if not previous_messages or not heuristic_followup:
            return QueryRewriteResult(
                original_query=query,
                standalone_query=query,
                is_followup=False,
                current_topic=_explicit_topic(query),
                required_entities=_extract_entities(query),
                use_last_sources=False,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={"reason": "direct_query"},
            )

        history = format_recent_conversation(previous_messages, max_messages=6, max_chars=1800)
        visual_rewrite = _visual_followup_rewrite(
            query,
            previous_messages,
            working_topic=working_topic,
        )
        if visual_rewrite is not None:
            topic, standalone_query, required_entities = visual_rewrite
            named_now = bool(_query_named_entities(query))
            return QueryRewriteResult(
                original_query=query,
                standalone_query=standalone_query,
                is_followup=True,
                current_topic=topic,
                required_entities=required_entities,
                # No new paper named → stay on sticky / last retrieved focus.
                use_last_sources=not named_now,
                answer_intent=(
                    "infer_structure"
                    if any(m in query.lower() for m in ("architecture", "kiến trúc", "pipeline", "sơ đồ"))
                    else "elaborate"
                ),
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={
                    "reason": "visual_followup",
                    "working_topic": working_topic,
                    "working_document_hint": working_document_hint,
                },
            )
        heuristic_rewrite = _generic_deepen_rewrite(
            query,
            previous_messages,
            working_topic=working_topic,
        )
        if heuristic_rewrite is not None:
            topic, standalone_query, required_entities = heuristic_rewrite
            return QueryRewriteResult(
                original_query=query,
                standalone_query=standalone_query,
                is_followup=True,
                current_topic=topic,
                required_entities=required_entities,
                use_last_sources=True,
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                rewrite_used=False,
                diagnostics={"reason": "heuristic_deepen_followup"},
            )

        selected_model = model or self.default_model
        completion = await self.client.chat(
            model=selected_model,
            temperature=0.0,
            num_predict=280,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions for local document retrieval. "
                        "Return only valid JSON. Do not answer the user. "
                        "The standalone_query must be specific enough for search without chat history."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Recent conversation:\n"
                        f"{history}\n\n"
                        f"Latest user question:\n{query}\n\n"
                        "Return JSON with this shape:\n"
                        "{\n"
                        '  "standalone_query": "rewritten retrieval query",\n'
                        '  "is_followup": true,\n'
                        '  "current_topic": "main topic or null",\n'
                        '  "required_entities": ["specific acronyms, model names, dataset names"],\n'
                        '  "use_last_sources": true,\n'
                        '  "answer_intent": "direct_answer|elaborate|compare|infer_structure|simplify|example",\n'
                        '  "answer_depth": "brief|normal|detailed"\n'
                        "}\n"
                        "Rules:\n"
                        "- Preserve Vietnamese if the user writes Vietnamese.\n"
                        "- Include exact entities from the previous turn when the question says this/that/it/more.\n"
                        "- If the user changes topic, set use_last_sources=false and include the new entity.\n"
                        "- Do not include markdown fences."
                    ),
                },
            ],
        )
        payload = _parse_json_object(completion.message)
        fallback_context = f"{history}\n{query}"
        standalone_query = str(payload.get("standalone_query") or "").strip() or _fallback_query(
            query,
            previous_messages,
        )
        named_now = _query_named_entities(query)
        payload_topic = _string_or_none(payload.get("current_topic"))
        current_topic = (
            (named_now[0] if named_now else None)
            or (working_topic or "").strip()
            or (payload_topic if payload_topic and _is_named_entity(payload_topic) else None)
            or _explicit_topic(fallback_context)
        )
        required_entities = _normalize_entities(payload.get("required_entities"))
        if not named_now and working_topic:
            required_entities = [working_topic]
        elif not required_entities:
            required_entities = _extract_entities(fallback_context)
        if current_topic and _is_named_entity(current_topic):
            required_entities = [
                current_topic,
                *[item for item in required_entities if item.lower() != current_topic.lower()],
            ]
        return QueryRewriteResult(
            original_query=query,
            standalone_query=standalone_query,
            is_followup=bool(payload.get("is_followup", True)),
            current_topic=current_topic,
            required_entities=required_entities,
            use_last_sources=bool(payload.get("use_last_sources", True)),
            answer_intent=_string_choice(payload.get("answer_intent"), _ANSWER_INTENTS, answer_intent),
            answer_depth=_string_choice(payload.get("answer_depth"), _ANSWER_DEPTHS, answer_depth),
            rewrite_used=True,
            diagnostics={
                "reason": "llm_rewrite",
                "raw_chars": len(completion.message),
            },
        )


def _looks_like_topic_switch(query: str, previous_messages: list | None = None) -> bool:
    if looks_like_document_resume(query):
        return False
    if _explicit_document_target_entities(query):
        return True
    normalized = " ".join(query.lower().split())
    query_entities = _query_named_entities(query)
    known_query_entities = False
    if previous_messages and query_entities:
        history = format_recent_conversation(previous_messages, max_messages=6, max_chars=1600).lower()
        history_topic = (_topic_from_history(history) or "").lower()
        known_query_entities = True
        for entity in query_entities:
            key = entity.lower()
            # New paper/model named in this turn, not the active history topic.
            if key and not _entity_mentioned(key, history) and not _entity_mentioned(
                key, history_topic
            ):
                return True
            if (
                history_topic
                and not _entity_mentioned(key, history_topic)
                and not _entity_mentioned(key, history)
            ):
                return True
    if re.search(r"\b(thế|vậy|còn|sang|chuyển|instead|switch)\b", normalized):
        # Discourse markers such as "thế" often mean "then/so" inside the
        # same paper.  An entity already present in recent turns is not a topic
        # switch merely because that marker appears before it.
        if known_query_entities:
            return False
        return bool(query_entities or _explicit_topic(query))
    return False


def _entity_mentioned(entity: str, text: str) -> bool:
    """Match aliases across harmless spacing/hyphen differences."""

    key = (entity or "").lower().strip()
    haystack = (text or "").lower()
    if not key:
        return False
    if key in haystack:
        return True
    compact_key = re.sub(r"[^a-z0-9]+", "", key)
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    return bool(compact_key and compact_key in compact_haystack)


def looks_like_followup(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    if looks_like_document_resume(query):
        return True
    followup_markers = [
        "nói rõ hơn",
        "giải thích thêm",
        "chi tiết hơn",
        "kĩ hơn",
        "kỹ hơn",
        "rõ hơn",
        "cái đó",
        "cái này",
        "nó ",
        "này ",
        "đó ",
        "so sánh với",
        "so sánh",
        "khác gì",
        "khác nhau",
        "model khác",
        "bảng",
        "kết quả",
        "benchmark",
        "baseline",
        "ablation",
        "đoán",
        "suy ra",
        "phác thảo",
        "cấu trúc",
        "pipeline",
        "architecture",
        "what about",
        "explain more",
        "more detail",
        "compare with",
    ]
    if any(marker in normalized for marker in followup_markers):
        return True
    if re.search(r"\b(acc|f1|ccc|uar|wa|ua)\b", normalized):
        return True

    has_explicit_topic = bool(re.search(r"\b[A-Za-z]*[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b|\.pdf\b", query))
    return len(normalized) <= 48 and not has_explicit_topic


def _generic_deepen_rewrite(
    query: str,
    previous_messages: list,
    *,
    working_topic: str | None = None,
) -> tuple[str, str, list[str]] | None:
    normalized = " ".join(query.lower().split())
    inference_markers = [
        "đoán",
        "suy ra",
        "infer",
        "phác thảo",
        "ước lượng",
        "cấu trúc",
        "pipeline",
        "architecture",
    ]
    deepen_markers = [
        "nói rõ hơn",
        "giải thích thêm",
        "chi tiết hơn",
        "kĩ hơn",
        "kỹ hơn",
        "rõ hơn",
        "more detail",
        "explain more",
    ]
    result_markers = [
        "benchmark",
        "baseline",
        "ablation",
        "bảng",
        "kết quả",
        "so sánh",
        "model khác",
        "accuracy",
        "acc",
        "f1",
        "ccc",
        "uar",
    ]
    is_inference_request = any(marker in normalized for marker in inference_markers)
    is_result_request = any(marker in normalized for marker in result_markers)
    if not (
        is_inference_request
        or is_result_request
        or any(marker in normalized for marker in deepen_markers)
    ):
        return None

    # The metric/table expansion below is intentionally specialized for an
    # authoritative local-document thread. A generic conversation may use
    # words such as "benchmark", "results", or "baseline" with no paper at
    # all; inferring a topic from free-form history in that case previously
    # produced nonsense such as an IBM query for Acc/F1/CCC. The catalog/scope
    # layer supplies working_topic for actual focused-paper turns.
    if is_result_request and not (working_topic or "").strip():
        return None

    query_entities = _query_named_entities(query)
    history_text = format_recent_conversation(previous_messages, max_messages=4, max_chars=1600)
    history_topic = _topic_from_history(history_text)
    # Sticky L1 topic beats scanning older history mentions.
    sticky_topic = (working_topic or "").strip() or None
    # Named paper/model in this turn wins over history deepen (e.g. WhiSER → ASPIRE).
    if query_entities:
        primary = query_entities[0]
        baseline = sticky_topic or history_topic
        if baseline and primary.lower() not in baseline.lower():
            # A method/model already discussed under the sticky paper is a
            # same-paper result follow-up, not a request to abandon the paper
            # scope.  Truly new names still fall through to the topic-switch
            # path above.
            if sticky_topic and _entity_mentioned(primary, history_text):
                topic = sticky_topic
            else:
                return None
        else:
            topic = primary
    else:
        topic = sticky_topic or history_topic
    if topic is None:
        return None

    if is_inference_request:
        entities = query_entities or ([topic] if topic else []) or _extract_entities(history_text)
        required = entities[:4] or [topic]
        standalone_query = (
            f"Suy luận cấu trúc/thành phần và pipeline high-level của {topic} "
            "từ bằng chứng đã truy xuất. "
            "Tách rõ: sự thật có trong source, suy luận hợp lý ở mức high-level, và chi tiết không thể xác nhận."
        )
    elif is_result_request:
        required = [topic]
        standalone_query = (
            f"{topic} benchmark comparison table Acc F1 CCC arousal valence "
            "results vs baselines ablation accuracy"
        )
    else:
        standalone_query = (
            f"Giải thích kỹ hơn về {topic}: định nghĩa, mục đích, cấu trúc/thành phần, "
            "pipeline hoạt động, dữ liệu vào/ra và các chi tiết quan trọng trong tài liệu."
        )
        required = query_entities[:4] or [topic]

    return topic, standalone_query, required


_ANSWER_INTENTS = {"direct_answer", "elaborate", "compare", "infer_structure", "simplify", "example"}
_ANSWER_DEPTHS = {"brief", "normal", "detailed"}


def _classify_answer_intent(query: str) -> str:
    normalized = " ".join(query.lower().split())
    if any(marker in normalized for marker in ["đoán", "suy ra", "infer", "phác thảo", "ước lượng"]):
        return "infer_structure"
    if any(marker in normalized for marker in ["cấu trúc", "pipeline", "architecture", "trông như nào"]):
        return "infer_structure"
    if "khác gì" in normalized or looks_like_document_comparison(normalized):
        return "compare"
    if any(marker in normalized for marker in ["ví dụ", "example"]):
        return "example"
    if any(marker in normalized for marker in ["dễ hiểu", "đơn giản", "ngắn gọn"]):
        return "simplify"
    if any(marker in normalized for marker in ["nói rõ hơn", "giải thích thêm", "chi tiết hơn", "kĩ hơn", "kỹ hơn"]):
        return "elaborate"
    return "direct_answer"


def _classify_answer_depth(query: str) -> str:
    normalized = " ".join(query.lower().split())
    if any(marker in normalized for marker in ["kĩ", "kỹ", "chi tiết", "rõ hơn", "đào sâu", "giải thích kỹ"]):
        return "detailed"
    if any(marker in normalized for marker in ["ngắn gọn", "tóm tắt", "brief"]):
        return "brief"
    return "normal"


def _topic_from_history(history_text: str) -> str | None:
    """Pick the most recently mentioned paper/model topic in conversation text.

    Names are recognised through the catalog-derived vocabulary (filenames,
    acronyms, hyphenated content keywords), never a fixed paper list.
    """
    from app.services.rag_service import default_catalog_vocabulary, document_match_tokens

    vocabulary = default_catalog_vocabulary()
    latest: str | None = None
    if vocabulary is not None:
        for line in str(history_text or "").splitlines():
            for entity in _query_named_entities(line):
                tokens = document_match_tokens(entities=[], query=entity, vocabulary=vocabulary)
                if any(
                    token in vocabulary.acronyms
                    or token in vocabulary.aliases
                    or vocabulary.is_distinctive(token)
                    or len(tokens) > 1  # a compound of catalog name words
                    for token in tokens
                ):
                    latest = entity
    if latest is not None:
        return latest

    entities = _extract_entities(history_text)
    return entities[0] if entities else None


def _visual_followup_rewrite(
    query: str,
    previous_messages: list,
    *,
    working_topic: str | None = None,
) -> tuple[str, str, list[str]] | None:
    if not has_visual_intent(query):
        return None
    query_entities = _query_named_entities(query)
    history = format_recent_conversation(previous_messages, max_messages=6, max_chars=1800)
    sticky = (working_topic or "").strip() or None
    # Prefer paper named in this turn; else sticky L1; else most recent history topic.
    topic = query_entities[0] if query_entities else sticky or _topic_from_history(history)
    if not topic:
        return None
    standalone_query = (
        f"{topic} figures diagrams charts plots tables visualization results "
        "benchmark arousal valence CCC architecture overview"
    )
    return topic, standalone_query, [topic]


def format_recent_conversation(
    messages: list,
    *,
    max_messages: int = 12,
    max_chars: int = 4500,
) -> str:
    if not messages:
        return "No recent conversation context."

    # Fill the budget newest-first, then restore chronological order. This keeps
    # the most recent exchange even when an older message is unusually long.
    lines_reversed: list[str] = []
    used = 0
    for message in reversed(messages[-max_messages:]):
        if isinstance(message, dict):
            role = str(message.get("role") or "message")
            content = " ".join(str(message.get("content") or "").split())
        else:
            role = getattr(message, "role", "message")
            content = " ".join(str(getattr(message, "content", "")).split())
        if not content:
            continue
        line = f"{role}: {content}"
        if used + len(line) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            line = _context_clip(line, remaining)
        if not line:
            break
        lines_reversed.append(line)
        used += len(line)
        if used >= max_chars:
            break

    lines_reversed.reverse()
    return "\n".join(lines_reversed) if lines_reversed else "No recent conversation context."


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _fallback_query(query: str, previous_messages: list) -> str:
    history = format_recent_conversation(previous_messages, max_messages=4, max_chars=1200)
    return (
        "Use the recent conversation to resolve this follow-up question.\n\n"
        f"{history}\n\n"
        f"Current follow-up question:\n{query}"
    )


def _normalize_entities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    entities: list[str] = []
    seen: set[str] = set()
    for item in value:
        entity = str(item).strip()
        if (
            not entity
            or entity.lower() in {"null", "none", "n/a"}
            or not _is_named_entity(entity)
        ):
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity)
    return entities[:8]


def _extract_entities(text: str) -> list[str]:
    from app.services.rag_service import default_catalog_vocabulary, document_match_tokens

    # Preserve every explicit acronym/model in textual order.  Previously, one
    # known alias (e.g. ASPIRE) caused an early return that hid a second acronym
    # (e.g. MSF-SER) in the same correction or comparison query.
    entities = re.findall(r"\b[A-Za-z][A-Za-z0-9_.-]{1,}\b", text)
    keep: list[str] = []
    seen: set[str] = set()
    stop = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "how",
        "why",
        "explain",
        "more",
        "about",
        "source",
        "use",
        "recent",
        "conversation",
        "current",
        "question",
        "follow",
        "latest",
        "user",
        "assistant",
        "quay",
        "return",
        "resume",
        "previous",
        "paper",
        "document",
        "evidence",
        "benchmark",
    }
    for entity in entities:
        if entity.lower() in stop:
            continue
        if not _is_named_entity(entity):
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            keep.append(entity)

    # Add canonical compound aliases that span multiple raw words (for example
    # "pitch fusion") after the literal entities above.
    compact_seen = {re.sub(r"[^a-z0-9]+", "", key) for key in seen}
    for token in document_match_tokens(
        entities=[], query=text, vocabulary=default_catalog_vocabulary(),
    ):
        key = token.lower()
        compact = re.sub(r"[^a-z0-9]+", "", key)
        # "MSF-SER" and its compact alias "msfser" are one entity.
        if key not in seen and compact not in compact_seen and _is_named_entity(token):
            seen.add(key)
            compact_seen.add(compact)
            keep.append(token)
    return keep[:8]


def _context_clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit < 48:
        return ""

    # A character slice can turn a claim/table row into a misleading fragment.
    # Keep complete sentence-like units only; the full message remains in L0
    # and can be retrieved separately when it does not fit this recent window.
    prefix, separator, body = text.partition(": ")
    role_prefix = f"{prefix}: " if separator else ""
    body = body if separator else text
    budget = limit - len(role_prefix)
    if budget < 32:
        return ""
    units = [
        item.strip()
        for item in re.findall(r".+?(?:[.!?…]+(?=\s|$)|$)", body)
        if item.strip()
    ]
    fitting = [item for item in units if len(item) <= budget]
    if not fitting:
        marker = "[long message retained in full history]"
        return f"{role_prefix}{marker}" if len(role_prefix) + len(marker) <= limit else ""

    selected: list[str] = []
    used = 0
    # Prefer the newest/end sentences, which normally contain conclusions and
    # follow-up anchors, then use spare room for the beginning.
    for unit in reversed(fitting):
        extra = len(unit) + (1 if selected else 0)
        if used + extra > budget:
            continue
        selected.insert(0, unit)
        used += extra
    clipped = " ".join(selected)
    if clipped != body.strip():
        marker = " …"
        if len(clipped) + len(marker) <= budget:
            clipped += marker
    return f"{role_prefix}{clipped}"


def _explicit_topic(text: str) -> str | None:
    entities = _extract_entities(text)
    return entities[0] if entities else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def _string_choice(value: Any, choices: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in choices else default
