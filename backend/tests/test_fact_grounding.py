"""Ingest-time facts give prose evidence the row/column structure tables have."""
import asyncio
from types import SimpleNamespace

import pytest

from app.db.sqlite import connect, init_db
from app.services.fact_extraction_service import (
    FactStore,
    build_document_facts,
    extract_facts,
    validate_fact,
)
from app.services.numeric_grounding import ground_answer_numbers

_PROSE = (
    "Gói Tiêu chuẩn có giá 1.250.000 đồng/tháng, chưa gồm VAT; gói Doanh nghiệp "
    "giá 3.900.000 đồng/tháng, đã gồm VAT. Phí vận chuyển của gói Tiêu chuẩn là "
    "45.000 đồng, gói Doanh nghiệp được miễn phí vận chuyển."
)
_FACTS = [
    {"entity": "Gói Tiêu chuẩn", "attribute": "giá", "conditions": ["chưa gồm VAT"],
     "value": "1.250.000", "unit": "đồng/tháng", "quote": "Gói Tiêu chuẩn có giá 1.250.000 đồng/tháng, chưa gồm VAT"},
    {"entity": "gói Doanh nghiệp", "attribute": "giá", "conditions": ["đã gồm VAT"],
     "value": "3.900.000", "unit": "đồng/tháng", "quote": "gói Doanh nghiệp giá 3.900.000 đồng/tháng, đã gồm VAT"},
    {"entity": "gói Tiêu chuẩn", "attribute": "Phí vận chuyển", "conditions": [],
     "value": "45.000", "unit": "đồng", "quote": "Phí vận chuyển của gói Tiêu chuẩn là 45.000 đồng"},
]


def _docs(with_facts=True):
    document = {"document_id": "p", "filename": "bang_gia.md", "content": _PROSE}
    if with_facts:
        document["facts"] = _FACTS
    return [document]


@pytest.mark.parametrize("answer, expected", [
    ("Gói Tiêu chuẩn giá 1.250.000 đồng/tháng, chưa gồm VAT.", True),
    ("Gói Doanh nghiệp giá 1.250.000 đồng/tháng.", False),         # other entity
    ("Phí vận chuyển của gói Doanh nghiệp là 45.000 đồng.", False),  # other entity
    ("Gói Tiêu chuẩn có giá 45.000 đồng.", False),                    # other attribute
])
def test_prose_values_are_bound_to_their_entity_and_attribute(answer, expected):
    assert ground_answer_numbers(answer, _docs()).valid is expected


def test_without_facts_prose_numbers_are_only_checked_for_existence():
    # Documents the limitation the facts remove: a borrowed prose value passes.
    assert ground_answer_numbers("Gói Doanh nghiệp giá 1.250.000 đồng/tháng.", _docs(False)).valid


def test_validate_fact_requires_verbatim_quote_containing_the_value():
    chunk = "Company revenue in Q2 was 438.2 million, up from 410.5 million."
    good = {"entity": "company", "attribute": "revenue", "conditions": ["Q2"],
            "value": "438.2", "quote": "Company revenue in Q2 was 438.2 million"}
    assert validate_fact(good, chunk) is not None
    assert validate_fact({**good, "value": "440.0"}, chunk) is None          # not in quote
    assert validate_fact({**good, "quote": "Revenue in Q2 was 440 million"}, chunk) is None  # not verbatim
    assert validate_fact({**good, "entity": ""}, chunk) is None


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def stream_chat(self, **kwargs):
        self.calls += 1
        import json
        yield SimpleNamespace(content=json.dumps(self.payload))


def test_extraction_drops_invented_numbers_and_unknown_chunks():
    chunks = [{"chunk_id": "c1", "content": _PROSE}]
    client = _FakeClient({"facts": [
        {**_FACTS[0], "chunk": 0},
        {**_FACTS[1], "chunk": 0, "value": "4.000.000"},   # invented value
        {**_FACTS[2], "chunk": 7},                         # no such chunk
    ]})
    result = asyncio.run(extract_facts(chunks, client=client, model="cx/gpt-5.6-sol"))
    assert [fact.value_text for fact in result.facts] == ["1.250.000"]
    assert result.rejected == 2
    with pytest.raises(ValueError):
        asyncio.run(extract_facts(chunks, client=client, model="llama3"))


def test_facts_are_stored_per_active_revision_and_resumable(tmp_path):
    from app.services.indexing_service import IndexingService

    db = tmp_path / "app.db"
    init_db(db)
    source = tmp_path / "bang_gia.md"
    source.write_text(f"# Bảng giá\n\n{_PROSE}\n", encoding="utf-8")
    document = IndexingService(db).index_file(source_path=str(source))
    with connect(db) as connection:
        chunk_id = connection.execute(
            "SELECT id FROM chunks WHERE document_id = ? AND content LIKE '%1.250.000%'", (document.id,)
        ).fetchone()[0]
    client = _FakeClient({"facts": [{**fact, "chunk": 0} for fact in _FACTS]})
    first = asyncio.run(build_document_facts(db, document.id, client=client, model="cx/gpt-5.6-sol"))
    assert first["status"] == "complete" and first["facts"] == 3
    again = asyncio.run(build_document_facts(db, document.id, client=client, model="cx/gpt-5.6-sol"))
    assert again["status"] == "skipped" and client.calls == 1
    stored = FactStore(db).facts_for_chunks([chunk_id])
    assert {fact["value"] for fact in stored[chunk_id]} == {"1.250.000", "3.900.000", "45.000"}
