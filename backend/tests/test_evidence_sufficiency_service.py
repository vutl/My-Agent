from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.services.evidence_sufficiency_service import EvidenceSufficiencyService


class _Client:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def chat(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return SimpleNamespace(message=self.response)


def test_sufficiency_classifier_is_relation_based_not_dataset_named() -> None:
    client = _Client(
        '{"verdict":"insufficient","confidence":"high",'
        '"reason":"The passage describes Product A but never relates it to Product B.",'
        '"supported_facets":["Product A is immutable"],'
        '"missing_facets":["relationship between Product A and Product B"]}'
    )
    assessment = asyncio.run(
        EvidenceSufficiencyService(client=client, default_model="model").assess(
            question="Are Product A and Product B the same?",
            documents=[{"filename": "A", "content": "Product A is immutable."}],
        )
    )
    assert assessment.verdict == "insufficient"
    assert assessment.can_answer is False
    prompt = client.calls[0]["messages"][0]["content"]
    assert "A passage describing A does not establish a comparison" in prompt
    assert "MTRAG" not in prompt
    assert "ASPIRE" not in prompt


def test_sufficiency_classifier_accepts_direct_support_and_strips_fence() -> None:
    client = _Client(
        "```json\n"
        '{"verdict":"sufficient","confidence":"high","reason":"Direct support.",'
        '"supported_facets":["location"],"missing_facets":[]}\n```'
    )
    assessment = asyncio.run(
        EvidenceSufficiencyService(client=client, default_model="model").assess(
            question="Where is it?",
            documents=[{"content": "It is in London."}],
        )
    )
    assert assessment.can_answer is True
    assert assessment.supported_facets == ["location"]


def test_ambiguous_evidence_allows_only_a_conditional_or_clarifying_response() -> None:
    client = _Client(
        '{"verdict":"ambiguous","confidence":"high",'
        '"reason":"Two variants have different origins.",'
        '"supported_facets":["variant A","variant B"],'
        '"missing_facets":["intended variant"]}'
    )
    assessment = asyncio.run(
        EvidenceSufficiencyService(client=client, default_model="model").assess(
            question="Where was it invented?",
            documents=[{"content": "A was invented in X; B was invented in Y."}],
        )
    )
    assert assessment.verdict == "ambiguous"
    assert assessment.can_answer is True


def test_sufficiency_classifier_fails_closed_on_invalid_schema_or_no_documents() -> None:
    client = _Client("not json")
    service = EvidenceSufficiencyService(client=client, default_model="model")
    invalid = asyncio.run(
        service.assess(question="What happened?", documents=[{"content": "Related."}])
    )
    empty = asyncio.run(service.assess(question="What happened?", documents=[]))
    assert invalid.verdict == "insufficient"
    assert invalid.confidence == "low"
    assert empty.verdict == "insufficient"
    assert len(client.calls) == 1
