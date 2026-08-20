from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "retrieval_eval"


def _load_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_conversational_eval_is_frozen_grouped_and_corpus_wide() -> None:
    dev = _load_jsonl("conversational-dev-v1.jsonl")
    heldout = _load_jsonl("conversational-heldout-v1.jsonl")
    manifest = json.loads(
        (EVAL_DIR / "conversational-heldout-v1.manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(dev) == 20
    assert len(heldout) == 60
    assert Counter(item["conversation_group"] for item in heldout) == Counter(
        {f"h{index:02d}": 4 for index in range(1, 16)}
    )
    assert Counter(item["language"] for item in heldout) == {"vi": 36, "en": 24}
    assert sum(bool(item["must_cover_all"]) for item in heldout) >= 15
    assert sum(bool(item["expected_abstention"]) for item in heldout) >= 3

    expected_documents = {
        item["document_id"] for item in manifest["corpus"]["documents"]
    }
    mentions = Counter(
        document_id
        for item in heldout
        for document_id in item["expected_document_ids"]
    )
    assert expected_documents == set(mentions)
    assert min(mentions.values()) >= 2


def test_heldout_manifest_checksums_match_frozen_files() -> None:
    manifest = json.loads(
        (EVAL_DIR / "conversational-heldout-v1.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, expected in manifest["files"].items():
        payload = (EVAL_DIR / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
        assert len(payload.decode("utf-8").splitlines()) == expected["turns"]
