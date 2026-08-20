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
from typing import Any, Iterable
from zipfile import ZipFile

from external_rag_eval_contract import rouge_l_f1, token_f1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "data" / "retrieval_eval" / "public"
MMLONG_ROOT = PUBLIC_ROOT / "raw" / "github" / "mmlongbench_doc"
SPIQA_ROOT = PUBLIC_ROOT / "raw" / "huggingface" / "spiqa"
MMLONG_SAMPLES = MMLONG_ROOT / "data" / "samples.json"
MMLONG_DOCUMENTS = MMLONG_ROOT / "data" / "documents"
SPIQA_TEST_C = SPIQA_ROOT / "test-C" / "SPIQA_testC.json"
SPIQA_TEST_C_IMAGES = SPIQA_ROOT / "test-C" / "SPIQA_testC_Images_224px.zip"
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


def load_mmlong_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for index, row in enumerate(_json(MMLONG_SAMPLES)):
        doc_id = str(row.get("doc_id") or "").strip()
        source_path = (MMLONG_DOCUMENTS / doc_id).resolve()
        if not doc_id or not source_path.is_file():
            raise ValueError(f"MMLongBench document is missing at row {index}: {doc_id}")
        cases.append(
            BenchmarkCase(
                suite="mmlongbench-doc",
                case_id=f"mmlongbench-doc:{index:04d}",
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


def bounded_context(text: str, *, max_chars: int, label: str) -> BoundedContext:
    normalized = str(text or "").strip()
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(normalized) <= max_chars:
        return BoundedContext(normalized, len(normalized), len(normalized), False, label)
    marker = "\n\n[... CONTEXT TRUNCATED: HEAD + TAIL ONLY ...]\n\n"
    available = max(1, max_chars - len(marker))
    head = available * 2 // 3
    rendered = normalized[:head] + marker + normalized[-(available - head) :]
    return BoundedContext(rendered, len(normalized), len(rendered), True, label)


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


def build_eval_prompt(
    *,
    question: str,
    context: str,
    context_label: str,
    answer_format: str = "free_form",
) -> str:
    """Build a scoring prompt without accepting or interpolating a gold answer."""

    format_instruction = {
        "Int": "Return an integer only inside the answer field.",
        "Float": "Return the requested number only, preserving a percent sign only when useful.",
        "Str": "Return one short deterministic string inside the answer field.",
        "None": "Return one short deterministic string inside the answer field.",
        "List": (
            "Return a JSON array with one short atomic item for each requested field or "
            "interrogative clause, in question order. Never merge two requested outputs into "
            "one item; for example, 'higher or lower, and by how much?' needs two items. "
            "A numeric item must be the bare number without units or explanatory words."
        ),
        "YesNo": "Return exactly Yes or No inside the answer field.",
    }.get(answer_format, "Return the shortest answer that completely answers the question.")
    return f"""Answer the question using only the evidence below.
If the evidence cannot answer it, use exactly: Not answerable
Return one JSON object only: {{"answer": <short answer string or JSON list>}}.
Do not add explanation, citations, Markdown, or extra keys.
Expected answer format: {answer_format}. {format_instruction}

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
