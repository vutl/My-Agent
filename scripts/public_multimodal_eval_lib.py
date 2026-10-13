"""Safe, dataset-faithful contracts for public long-document/visual QA evals.

Gold answers are carried only for post-generation scoring.  Prompt builders in
this module accept evidence/context explicitly and therefore cannot receive a
``BenchmarkCase`` object by accident.
"""

from __future__ import annotations

import ast
import base64
from collections import defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import hashlib
from io import StringIO
import importlib.util
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
from functools import lru_cache
from typing import Any, Iterable
from zipfile import ZipFile

from external_rag_eval_contract import rouge_l_f1, token_f1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "data" / "retrieval_eval" / "public"
MMLONG_ROOT = PUBLIC_ROOT / "raw" / "github" / "mmlongbench_doc"
MMLONG_V2_ROOT = PUBLIC_ROOT / "raw" / "github" / "mmlongbench_doc_v2"
SPIQA_ROOT = PUBLIC_ROOT / "raw" / "huggingface" / "spiqa"
MMLONG_SAMPLES = MMLONG_ROOT / "data" / "samples.json"
MMLONG_DOCUMENTS = MMLONG_ROOT / "data" / "documents"
SPIQA_TEST_C = SPIQA_ROOT / "test-C" / "SPIQA_testC.json"
# Preserve the upstream image geometry and glyph resolution.  The optional
# 224px archive is a different experimental condition, not a substitute for
# the document's native visual evidence.  Never silently fall back to it.
SPIQA_TEST_C_IMAGES = SPIQA_ROOT / "test-C" / "SPIQA_testC_Images.zip"
SPIQA_IMAGE_INPUT_POLICY = "original-resolution-no-resize-v1"
OFFICIAL_MMLONG_SCORER = MMLONG_ROOT / "eval" / "eval_score.py"


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    case_id: str
    document_id: str
    title: str
    question: str
    answers: tuple[str, ...]
    answer_format: str
    source_path: Path | None = None
    evidence_pages: tuple[int, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    evidence_text: tuple[str, ...] = ()
    artifacts: tuple[dict[str, str], ...] = ()
    referred_artifacts: tuple[str, ...] = ()
    full_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stratum(self) -> str:
        if self.suite == "mmlongbench-doc":
            page_class = "multi_page" if len(self.evidence_pages) > 1 else "single_page"
            sources = "+".join(sorted(self.evidence_sources)) or "unknown"
            if any(answer.lower() == "not answerable" for answer in self.answers):
                page_class = "unanswerable"
            return f"{page_class}:{sources}"
        figure = bool(self.metadata.get("figure_in_evidence"))
        table = bool(self.metadata.get("table_in_evidence"))
        if any(answer.lower() == "not answerable" for answer in self.answers):
            return "unanswerable"
        return "figure+table" if figure and table else "figure" if figure else "table" if table else "text"


@dataclass(frozen=True)
class BoundedContext:
    text: str
    original_chars: int
    included_chars: int
    truncated: bool
    label: str
    original_tokens: int | None = None
    included_tokens: int | None = None
    budget_tokens: int | None = None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        return [value]
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def load_mmlong_cases(*, version: str = "v1") -> list[BenchmarkCase]:
    """Load a pinned MMLong question set without changing its semantics.

    v1 remains the compatibility default for historical reports.  New
    evaluations should request v2 explicitly: it corrects published
    annotations and removes unresolvable duplicate/wrong-document rows.  The
    V2 repository intentionally does not redistribute PDFs, so the loader
    resolves its unchanged filenames against the already downloaded upstream
    document directory and fails closed if a source is missing.
    """

    normalized_version = str(version or "v1").strip().casefold()
    if normalized_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported MMLongBench-Doc version: {version!r}")
    samples_path = MMLONG_SAMPLES if normalized_version == "v1" else MMLONG_V2_ROOT / "data" / "samples.json"
    document_root = MMLONG_DOCUMENTS
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"MMLongBench-Doc {normalized_version} samples are missing: {samples_path}"
        )
    cases: list[BenchmarkCase] = []
    for index, row in enumerate(_json(samples_path)):
        doc_id = str(row.get("doc_id") or "").strip()
        source_path = (document_root / doc_id).resolve()
        if not doc_id or not source_path.is_file():
            raise ValueError(
                f"MMLongBench-Doc {normalized_version} document is missing at row {index}: {doc_id}"
            )
        cases.append(
            BenchmarkCase(
                suite="mmlongbench-doc",
                case_id=f"mmlongbench-doc-{normalized_version}:{index:04d}",
                document_id=doc_id,
                title=doc_id,
                question=str(row.get("question") or "").strip(),
                answers=(str(row.get("answer") or "").strip(),),
                answer_format=str(row.get("answer_format") or "Str"),
                source_path=source_path,
                evidence_pages=tuple(int(value) for value in _literal_list(row.get("evidence_pages"))),
                evidence_sources=tuple(str(value) for value in _literal_list(row.get("evidence_sources"))),
                metadata={"doc_type": row.get("doc_type")},
            )
        )
    return cases


def _answer_text(answer: Any) -> str:
    if not isinstance(answer, dict):
        return str(answer or "").strip()
    if answer.get("unanswerable"):
        return "Not answerable"
    if answer.get("yes_no") is True:
        return "Yes"
    if answer.get("yes_no") is False:
        return "No"
    free_form = str(answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    spans = answer.get("extractive_spans") or []
    return "; ".join(str(value).strip() for value in spans if str(value).strip())


def _spiqa_answer_format(answer: Any) -> str:
    if isinstance(answer, dict) and answer.get("yes_no") is not None:
        return "YesNo"
    return "free_form"


def _spiqa_markdown(row: dict[str, Any]) -> str:
    blocks = [f"# {row.get('paper_title') or row.get('arxiv_id')}"]
    abstract = str(row.get("abstract") or "").strip()
    if abstract:
        blocks.extend(("## Abstract", abstract))
    for section in row.get("full_text") or []:
        title = str(section.get("section_name") or "Section").strip()
        paragraphs = "\n\n".join(
            str(value).strip() for value in section.get("paragraphs") or [] if str(value).strip()
        )
        if paragraphs:
            blocks.extend((f"## {title}", paragraphs))
    artifacts = row.get("figures_and_tables") or []
    if artifacts:
        blocks.append("## Figures and tables")
        blocks.extend(
            f"- {item.get('file')}: {item.get('caption')}"
            for item in artifacts
            if item.get("file") or item.get("caption")
        )
    return "\n\n".join(blocks)


def load_spiqa_test_c_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for record_key, row in _json(SPIQA_TEST_C).items():
        questions = row.get("question") or []
        answers = row.get("answer") or []
        referred = row.get("referred_figures_tables") or []
        question_keys = row.get("question_key") or []
        figures = row.get("Is_figure_in_evidence") or []
        tables = row.get("Is_table_in_evidence") or []
        artifacts = tuple(
            {
                "file": str(item.get("file") or "").strip(),
                "caption": str(item.get("caption") or "").strip(),
            }
            for item in row.get("figures_and_tables") or []
        )
        markdown = _spiqa_markdown(row)
        for index, question in enumerate(questions):
            if index >= len(answers):
                raise ValueError(f"SPIQA Test C answer missing for {record_key}:{index}")
            answer = answers[index]
            gold = _answer_text(answer)
            if not gold:
                raise ValueError(f"SPIQA Test C answer is empty for {record_key}:{index}")
            evidence = answer.get("highlighted_evidence") or answer.get("evidence") or []
            cases.append(
                BenchmarkCase(
                    suite="spiqa-test-c",
                    case_id=f"spiqa-c:{record_key}:{index:04d}",
                    document_id=str(row.get("arxiv_id") or record_key),
                    title=str(row.get("paper_title") or record_key),
                    question=str(question).strip(),
                    answers=(gold,),
                    answer_format=_spiqa_answer_format(answer),
                    evidence_text=tuple(str(value).strip() for value in evidence if str(value).strip()),
                    artifacts=artifacts,
                    referred_artifacts=tuple(
                        str(value).strip()
                        for value in (referred[index] if index < len(referred) else [])
                        if str(value).strip()
                    ),
                    full_text=markdown,
                    metadata={
                        "record_key": record_key,
                        "question_key": question_keys[index] if index < len(question_keys) else None,
                        "figure_in_evidence": bool(figures[index]) if index < len(figures) else False,
                        "table_in_evidence": bool(tables[index]) if index < len(tables) else False,
                    },
                )
            )
    return cases


def stable_balanced_sample(cases: Iterable[BenchmarkCase], *, limit: int) -> list[BenchmarkCase]:
    """Round-robin deterministic strata so a tiny smoke is not one document/type."""

    if limit < 1:
        raise ValueError("limit must be positive")
    grouped: dict[str, deque[BenchmarkCase]] = defaultdict(deque)
    for case in cases:
        grouped[case.stratum].append(case)
    for key, values in grouped.items():
        grouped[key] = deque(
            sorted(
                values,
                key=lambda case: hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(),
            )
        )
    selected: list[BenchmarkCase] = []
    keys = sorted(grouped)
    while len(selected) < limit and any(grouped.values()):
        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].popleft())
                if len(selected) == limit:
                    break
    return selected


def resolve_case_ids(
    cases: Iterable[BenchmarkCase], case_ids: Iterable[str]
) -> list[BenchmarkCase]:
    """Resolve an explicit, ordered case manifest without answer peeking.

    Retrieval and answer-quality runners must be able to share exactly the
    same case frontier.  Re-running two independent deterministic samplers
    can otherwise produce two disjoint "20-case" reports that cannot be
    paired.  This resolver is deliberately strict: duplicate or unknown IDs
    fail closed instead of silently dropping rows or substituting another
    document.
    """

    requested = [str(case_id).strip() for case_id in case_ids if str(case_id).strip()]
    if not requested:
        raise ValueError("case manifest is empty")
    if len(set(requested)) != len(requested):
        raise ValueError("case manifest contains duplicate case IDs")
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"case manifest contains unknown case IDs: {missing[:5]}")
    return [by_id[case_id] for case_id in requested]


def stable_spiqa_case_sample(
    cases: Iterable[BenchmarkCase],
    *,
    limit: int,
    cases_per_document: int = 1,
) -> list[BenchmarkCase]:
    """Choose deterministic, stratum-balanced SPIQA cases with optional reuse.

    SPIQA contains many questions for the same paper.  Sampling one question
    per paper is useful for a corpus census, but it repeatedly pays the page
    rendering/visual-index cost.  This sampler can take a bounded number of
    questions from each selected document while rotating through figure/table
    strata.  Selection uses only case/document IDs and strata; annotations and
    answer text never influence which cases are selected.
    """

    if limit < 1 or cases_per_document < 1:
        raise ValueError("limit and cases_per_document must be positive")
    grouped: dict[str, dict[str, list[BenchmarkCase]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_document: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[case.stratum][case.document_id].append(case)
        by_document[case.document_id].append(case)

    # Choose a deterministic document frontier first.  A paper containing
    # several strata can satisfy more of the sample without paying another
    # page-image index cost.  Hash ordering is blind to answers/evidence, so
    # this is only a resource policy and cannot tune the score.
    ordered_documents = sorted(
        by_document,
        key=lambda document_id: hashlib.sha256(
            f"spiqa:document:{document_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected_documents: list[str] = []
    for stratum in sorted(grouped):
        candidates = [
            document_id
            for document_id in ordered_documents
            if document_id in grouped[stratum]
        ]
        if candidates and candidates[0] not in selected_documents:
            selected_documents.append(candidates[0])
    # Add more documents until the nominal capacity reaches the request.  A
    # paper may contain fewer questions than the bound, so the fill loop below
    # can still extend this frontier deterministically when needed.
    capacity = len(selected_documents) * cases_per_document
    for document_id in ordered_documents:
        if capacity >= limit:
            break
        if document_id not in selected_documents:
            selected_documents.append(document_id)
            capacity += cases_per_document

    ordered_cases: dict[str, dict[str, deque[BenchmarkCase]]] = {
        stratum: {
            document_id: deque(
                sorted(
                    document_cases,
                    key=lambda case: hashlib.sha256(
                        case.case_id.encode("utf-8")
                    ).hexdigest(),
                )
            )
            for document_id, document_cases in by_document_for_stratum.items()
        }
        for stratum, by_document_for_stratum in grouped.items()
    }
    selected: list[BenchmarkCase] = []
    per_document_counts: dict[str, int] = defaultdict(int)
    # Round-robin strata and documents.  This retains modality diversity while
    # filling several questions from the already-selected document frontier.
    while len(selected) < limit:
        progressed = False
        for stratum in sorted(ordered_cases):
            for document_id in selected_documents:
                if per_document_counts[document_id] >= cases_per_document:
                    continue
                queue = ordered_cases[stratum].get(document_id)
                if not queue:
                    continue
                selected.append(queue.popleft())
                per_document_counts[document_id] += 1
                progressed = True
                if len(selected) >= limit:
                    return selected
        if not progressed:
            # The initial frontier can be sparse (for example, a paper with a
            # single figure question).  Do not silently return fewer rows than
            # requested: extend with the next deterministic document and keep
            # the same per-document cap.
            remaining = next(
                (
                    document_id
                    for document_id in ordered_documents
                    if document_id not in selected_documents
                ),
                None,
            )
            if remaining is None:
                break
            selected_documents.append(remaining)
    return selected


@lru_cache(maxsize=2_048)
def _pdf_text_profile(source_path: str) -> str:
    try:
        from pypdf import PdfReader

        pages = PdfReader(source_path).pages
        if not pages:
            return "unreadable"
        probe = sorted({0, len(pages) // 2, len(pages) - 1})
        counts = []
        for index in probe:
            try:
                counts.append(len(re.sub(r"\s+", "", str(pages[index].extract_text() or ""))))
            except Exception:
                counts.append(0)
    except Exception:
        return "unreadable"
    if not counts or not any(counts):
        return "image_only"
    usable = sum(count >= 40 for count in counts) / len(counts)
    return "native_text" if usable >= 0.75 else "mixed_text_image"


def mmlong_document_profile(case: BenchmarkCase) -> str:
    """Classify a PDF by searchable-text coverage for stratified diagnostics."""

    if case.source_path is None:
        return "extracted_adapter"
    return _pdf_text_profile(str(case.source_path))


def stable_mmlong_profile_sample(
    cases: Iterable[BenchmarkCase],
    *,
    per_profile: int,
) -> list[BenchmarkCase]:
    """Choose one reproducible, document-balanced sample per PDF profile."""

    if per_profile < 1:
        raise ValueError("per_profile must be positive")
    grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[mmlong_document_profile(case)].append(case)
    selected: list[BenchmarkCase] = []
    for profile in sorted(grouped):
        ordered = sorted(
            grouped[profile],
            key=lambda case: hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(),
        )
        seen_documents: set[str] = set()
        profile_count = 0
        for case in ordered:
            if case.document_id in seen_documents:
                continue
            selected.append(case)
            seen_documents.add(case.document_id)
            profile_count += 1
            if profile_count >= per_profile:
                break
    return selected


def stable_mmlong_case_sample(
    cases: Iterable[BenchmarkCase],
    *,
    per_profile: int,
    cases_per_document: int = 1,
) -> list[BenchmarkCase]:
    """Select deterministic profile-balanced cases while reusing PDFs.

    The broad profile sampler intentionally chooses distinct documents.  That
    is useful for a corpus census but makes a 20-case visual run repeatedly
    pay the cold page-indexing cost.  This variant chooses a bounded number of
    documents per profile and several questions from each selected document.
    Selection depends only on profile/document/case IDs, never gold answers or
    evidence, so it is a resource-bounded diagnostic rather than a tuned score.
    Incomplete visual frontiers remain non-promotable and callers must record
    the resulting case IDs.
    """

    if per_profile < 1 or cases_per_document < 1:
        raise ValueError("per_profile and cases_per_document must be positive")
    grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[mmlong_document_profile(case)].append(case)

    selected: list[BenchmarkCase] = []
    for profile in sorted(grouped):
        by_document: dict[str, list[BenchmarkCase]] = defaultdict(list)
        for case in grouped[profile]:
            by_document[case.document_id].append(case)
        ordered_documents = sorted(
            by_document,
            key=lambda document_id: hashlib.sha256(
                f"{profile}:{document_id}".encode("utf-8")
            ).hexdigest(),
        )
        for document_id in ordered_documents[:per_profile]:
            ordered_cases = sorted(
                by_document[document_id],
                key=lambda case: hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(),
            )
            selected.extend(ordered_cases[:cases_per_document])
    return selected


def stable_mmlong_answer_sample(
    cases: Iterable[BenchmarkCase],
    *,
    limit: int,
    cases_per_document: int = 2,
) -> list[BenchmarkCase]:
    """Select answer-quality cases across PDF profiles without gold peeking.

    The visual runner already has a profile-aware sampler, but answer-quality
    runs need the same protection from accidentally selecting only easy native
    PDFs.  Selection depends on source readability, document IDs and case IDs
    only; answers and evidence annotations are never inspected.  A bounded
    document reuse cap amortizes expensive parsing/visual preparation while
    retaining a profile frontier.
    """

    if limit < 1 or cases_per_document < 1:
        raise ValueError("limit and cases_per_document must be positive")
    grouped: dict[str, dict[str, list[BenchmarkCase]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in cases:
        grouped[mmlong_document_profile(case)][case.document_id].append(case)

    ordered_by_profile: dict[str, list[str]] = {}
    for profile, by_document in grouped.items():
        ordered_by_profile[profile] = sorted(
            by_document,
            key=lambda document_id: hashlib.sha256(
                f"answer:{profile}:{document_id}".encode("utf-8")
            ).hexdigest(),
        )

    queues: dict[tuple[str, str], deque[BenchmarkCase]] = {}
    for profile, by_document in grouped.items():
        for document_id, document_cases in by_document.items():
            queues[(profile, document_id)] = deque(
                sorted(
                    document_cases,
                    key=lambda case: hashlib.sha256(
                        f"answer:{case.case_id}".encode("utf-8")
                    ).hexdigest(),
                )
            )

    selected: list[BenchmarkCase] = []
    per_document: dict[str, int] = defaultdict(int)
    profiles = sorted(ordered_by_profile)
    document_cursor: dict[str, int] = defaultdict(int)
    while len(selected) < limit:
        progressed = False
        for profile in profiles:
            documents = ordered_by_profile[profile]
            if not documents:
                continue
            # Rotate documents within each profile so one paper cannot consume
            # every slot when several candidates are available.
            for offset in range(len(documents)):
                index = (document_cursor[profile] + offset) % len(documents)
                document_id = documents[index]
                if per_document[document_id] >= cases_per_document:
                    continue
                queue = queues.get((profile, document_id))
                if not queue:
                    continue
                selected.append(queue.popleft())
                per_document[document_id] += 1
                document_cursor[profile] = (index + 1) % len(documents)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def bounded_context(text: str, *, max_chars: int, label: str) -> BoundedContext:
    normalized = str(text or "").strip()
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(normalized) <= max_chars:
        return BoundedContext(
            normalized,
            len(normalized),
            len(normalized),
            False,
            label,
            original_tokens=_lexical_token_count(normalized),
            included_tokens=_lexical_token_count(normalized),
        )
    marker = "\n\n[... CONTEXT TRUNCATED: HEAD + TAIL ONLY ...]\n\n"
    available = max(1, max_chars - len(marker))
    head = available * 2 // 3
    rendered = normalized[:head] + marker + normalized[-(available - head) :]
    return BoundedContext(
        rendered,
        len(normalized),
        len(rendered),
        True,
        label,
        original_tokens=_lexical_token_count(normalized),
        included_tokens=_lexical_token_count(rendered),
    )


def bounded_context_tokens(
    text: str,
    *,
    max_tokens: int,
    label: str,
) -> BoundedContext:
    """Bound a long-document baseline by deterministic lexical tokens."""

    normalized = str(text or "").strip()
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    original_tokens = _lexical_tokens(normalized)
    if len(original_tokens) <= max_tokens:
        return BoundedContext(
            normalized,
            len(normalized),
            len(normalized),
            False,
            label,
            original_tokens=len(original_tokens),
            included_tokens=len(original_tokens),
            budget_tokens=max_tokens,
        )
    marker = "\n\n[... CONTEXT WINDOWED: HEAD + TAIL TOKENS ONLY ...]\n\n"
    marker_tokens = len(_lexical_tokens(marker))
    available = max(1, max_tokens - marker_tokens)
    head_count = available * 2 // 3
    tail_count = available - head_count
    rendered = (
        " ".join(original_tokens[:head_count])
        + marker
        + " ".join(original_tokens[-tail_count:])
    )
    return BoundedContext(
        rendered,
        len(normalized),
        len(rendered),
        True,
        label,
        original_tokens=len(original_tokens),
        included_tokens=len(_lexical_tokens(rendered)),
        budget_tokens=max_tokens,
    )


def _lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", str(text or ""), flags=re.UNICODE)


def _lexical_token_count(text: str) -> int:
    return len(_lexical_tokens(text))


def spiqa_gold_context(case: BenchmarkCase) -> str:
    captions = {
        item["file"]: item["caption"]
        for item in case.artifacts
        if item.get("file") in set(case.referred_artifacts)
    }
    blocks = [*case.evidence_text]
    blocks.extend(f"{name}: {captions.get(name, '')}" for name in case.referred_artifacts)
    return "\n\n".join(block for block in blocks if block.strip())


def read_spiqa_images(case: BenchmarkCase, names: Iterable[str]) -> list[tuple[str, bytes]]:
    requested = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
    if not requested:
        return []
    for name in requested:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"Unsafe SPIQA image name: {name}")
    if not SPIQA_TEST_C_IMAGES.is_file():
        raise FileNotFoundError(
            "SPIQA original-resolution image archive missing: "
            f"{SPIQA_TEST_C_IMAGES}. Download the pinned official archive; "
            "224px thumbnails are not an equivalent input and will not be used."
        )
    suffixes = {
        f"/{case.document_id}/{name}": name
        for name in requested
    }
    found: dict[str, bytes] = {}
    with ZipFile(SPIQA_TEST_C_IMAGES) as archive:
        for member in archive.namelist():
            for suffix, name in suffixes.items():
                if member.endswith(suffix):
                    if name in found:
                        raise ValueError(f"Ambiguous SPIQA image member: {name}")
                    found[name] = archive.read(member)
    missing = [name for name in requested if name not in found]
    if missing:
        raise ValueError(f"SPIQA images missing: {missing}")
    return [(name, found[name]) for name in requested]


def image_data_url(name: str, payload: bytes) -> str:
    mime = mimetypes.guess_type(name)[0] or "image/png"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def supplied_image_provenance(name: str, payload: bytes) -> dict[str, Any]:
    """Audit the exact pixels supplied to a model, not just file names.

    Only headers are inspected: no resizing, decoded image mutation or gold
    evidence enters this record. Bytes and dimensions distinguish a thumbnail
    experiment from original-resolution evidence even when names are identical.
    """
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
    return {"name": name, "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "width": width, "height": height}


def build_eval_prompt(
    *,
    question: str,
    context: str,
    context_label: str,
    answer_format: str = "free_form",
    visual_context: bool = False,
) -> str:
    """Build a scoring prompt without accepting or interpolating a gold answer."""

    format_instruction = {
        "Int": "Return an integer only inside the answer field.",
        "Float": "Return the requested number only, preserving a percent sign only when useful.",
        "Str": "Return one short deterministic string inside the answer field.",
        "None": (
            "Return a concise but complete answer inside the answer field. Preserve every "
            "directly supported item, metric, numeric value or range, comparison, and "
            "qualification needed to answer each clause; do not summarize away requested details."
        ),
        "List": (
            "Return a JSON array with one short atomic item for each distinct entity, value, "
            "or requested field, in question order. Keep entities joined by 'and' as separate "
            "items when the question asks which things; never merge multiple outputs into one "
            "item; do not group items by page/slide unless the question explicitly asks for grouping. "
            "For example, 'higher or lower, and by how much?' needs two items. A numeric item "
            "must be the bare number without units or explanatory words."
        ),
        "YesNo": "Return exactly Yes or No inside the answer field.",
    }.get(
        answer_format,
        "Return a concise but complete answer. Preserve every directly supported item, "
        "metric, numeric value or range, comparison, and qualification needed to answer "
        "each clause of the question; do not summarize away requested details.",
    )
    visual_instruction = ""
    if visual_context:
        visual_instruction = (
            "\nVisual-evidence rule: inspect the attached image(s) and keep claims tied to "
            "the requested page/artifact. Do not transfer visual or numeric values from "
            "another page. If the supplied visual evidence does not directly support the "
            "question, return exactly: Not answerable.\n"
        )
    return f"""Answer the question using only the evidence below.
If the evidence cannot answer it, use exactly: Not answerable
Return one JSON object only: {{"answer": <short answer string or JSON list>}}.
Do not add explanation, citations, Markdown, or extra keys.
Expected answer format: {answer_format}. {format_instruction}
{visual_instruction}

Context mode: {context_label}

Question:
{question.strip()}

Evidence:
{context.strip() or '[No evidence retrieved]'}"""


def parse_answer_payload(text: str) -> str:
    raw = str(text or "").strip()
    candidates = [raw]
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.I | re.S)
    if fence:
        candidates.insert(0, fence.group(1))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and set(data) == {"answer"}:
            answer = data["answer"]
            if isinstance(answer, list):
                return json.dumps(answer, ensure_ascii=False)
            return str(answer).strip()
    prefix = re.sub(r"^\s*(?:final answer|answer)\s*:\s*", "", raw, flags=re.I)
    return prefix.strip().strip("`")


_MMLONG_MODULE: Any = None


def _official_mmlong_module() -> Any:
    global _MMLONG_MODULE
    if _MMLONG_MODULE is None:
        spec = importlib.util.spec_from_file_location("mmlongbench_official_eval_score", OFFICIAL_MMLONG_SCORER)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import official MMLongBench scorer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MMLONG_MODULE = module
    return _MMLONG_MODULE


def official_mmlong_score(*, gold: str, prediction: str, answer_format: str) -> float:
    safe_gold: Any = gold
    safe_prediction: Any = prediction
    if answer_format not in {"Int", "Float", "Str", "None"}:
        for name, value in (("gold", gold), ("prediction", prediction)):
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    parsed = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    parsed = value
                if name == "gold":
                    safe_gold = parsed
                else:
                    safe_prediction = parsed
    # Upstream prints list diagnostics; keep the benchmark report machine-readable.
    with redirect_stdout(StringIO()):
        score = _official_mmlong_module().eval_score(
            safe_gold, safe_prediction, answer_format
        )
    return float(score)


def spiqa_diagnostic_scores(*, answers: Iterable[str], prediction: str) -> dict[str, float]:
    references = [str(answer) for answer in answers if str(answer).strip()]
    if not references:
        return {"token_f1": 0.0, "rouge_l_f1": 0.0}
    return {
        "token_f1": max(token_f1(prediction, answer) for answer in references),
        "rouge_l_f1": max(rouge_l_f1(prediction, answer) for answer in references),
    }


def ensure_public_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = (PUBLIC_ROOT / "results").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Evaluation output must stay under {allowed}")
    return resolved
