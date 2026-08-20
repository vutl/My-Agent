"""Dataset-neutral contracts for evaluating Aya on an external text corpus.

External passages are adapted only at the final evidence boundary.  They are
never registered as catalog papers and this module never opens or writes Aya's
production SQLite, LanceDB, LightRAG, artifact, or memory stores.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ExternalPassage:
    passage_id: str
    collection: str
    title: str
    text: str


def stable_stratified_sample(
    rows: Iterable[dict[str, Any]],
    *,
    strata: Sequence[str],
    id_field: str,
    per_stratum: int,
) -> list[dict[str, Any]]:
    """Select reproducible, order-independent examples from every stratum."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_id = str(row.get(id_field) or "").strip()
        if not row_id:
            raise ValueError(f"Every evaluation row needs {id_field}")
        key = tuple(str(row.get(field) or "UNKNOWN") for field in strata)
        grouped[key].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                str(row[id_field]).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:per_stratum])
    return selected


def adapt_passages_to_aya_documents(
    passages: Sequence[ExternalPassage],
    *,
    channels_by_id: dict[str, list[str]] | None = None,
    max_text_chars: int = 8_000,
) -> list[dict[str, Any]]:
    """Map canonical external passages to Aya's generic evidence contract.

    Dataset reference answers and qrels are intentionally not accepted by this
    function, making accidental evaluation leakage impossible at this boundary.
    """

    channels_by_id = channels_by_id or {}
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, passage in enumerate(passages, start=1):
        passage_id = str(passage.passage_id).strip()
        if not passage_id or passage_id in seen:
            continue
        seen.add(passage_id)
        collection = str(passage.collection).strip() or "external"
        title = str(passage.title).strip() or passage_id
        documents.append(
            {
                "document_id": f"external:{collection}:{passage_id}",
                "source_id": f"SOURCE {rank}",
                "filename": title,
                "source_path": f"external://{collection}/{passage_id}",
                "content": str(passage.text).strip()[:max_text_chars],
                "chunk_type": "text",
                "retrieval_channels": list(channels_by_id.get(passage_id) or []),
                "external_passage_id": passage_id,
                "external_collection": collection,
            }
        )
    return documents


def format_conversation_context(messages: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or message.get("speaker") or "").lower()
        content = str(message.get("content") or message.get("text") or "").strip()
        if not content:
            continue
        label = "Assistant" if role in {"assistant", "agent"} else "User"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_ABSTENTION_RE = re.compile(
    r"\b(?:i (?:do not|don't) (?:know|have)|not enough (?:information|evidence)|"
    r"cannot (?:answer|determine)|can't (?:answer|determine)|unable to (?:answer|determine)|"
    r"no (?:relevant )?(?:information|evidence)|insufficient (?:information|evidence))\b",
    re.IGNORECASE,
)
_EVIDENCE_LIMITATION_RE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?(?:available|provided|retrieved)\s+(?:evidence|information|excerpts?|material)\s+"
    r"(?:does\s+not|doesn't|do\s+not|don't)\s+(?:directly\s+)?"
    r"(?:answer|confirm|specify|identify|establish|contain|show)|"
    r"(?:is|are)\s+not\s+(?:identified|specified|established)\s+in\s+"
    r"(?:the\s+)?(?:available|provided|retrieved)\s+(?:evidence|information|excerpts?|material)"
    r")\b",
    re.IGNORECASE,
)


def normalized_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def token_recall(prediction: str, target: str) -> float:
    from collections import Counter

    target_tokens = normalized_tokens(target)
    if not target_tokens:
        return 0.0
    overlap = Counter(normalized_tokens(prediction)) & Counter(target_tokens)
    return sum(overlap.values()) / len(target_tokens)


def token_f1(prediction: str, target: str) -> float:
    from collections import Counter

    prediction_tokens = normalized_tokens(prediction)
    target_tokens = normalized_tokens(target)
    if not prediction_tokens or not target_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, target: str) -> float:
    left = normalized_tokens(prediction)
    right = normalized_tokens(target)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if not lcs:
        return 0.0
    precision = lcs / len(left)
    recall = lcs / len(right)
    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    text = str(answer or "")
    return bool(_ABSTENTION_RE.search(text) or _EVIDENCE_LIMITATION_RE.search(text))
