from app.rag.document_aggregation import (
    dedupe_page_frontier,
    parse_observations,
    reduce_page_observations,
)
from app.api.agent import _page_aggregation_frontier_limit, _run_page_aggregation_map_reduce
from app.services.agent_service import AnswerStreamChunk


def _pages() -> list[dict]:
    return [
        {"document_id": "d1", "page_number": page, "content": f"page {page}"}
        for page in (2, 4, 7)
    ]


def test_page_frontier_deduplicates_by_document_and_page() -> None:
    pages = _pages() + [{"document_id": "d1", "page_number": 4, "content": "duplicate"}]

    result = dedupe_page_frontier(pages, max_pages=10)

    assert [(item["document_id"], item["page_number"]) for item in result] == [
        ("d1", 2),
        ("d1", 4),
        ("d1", 7),
    ]


def test_page_aggregation_frontier_scales_to_medium_documents_but_stays_bounded() -> None:
    pages = [
        {
            "document_id": "d1",
            "page_number": page,
            "document_page_count": 40,
        }
        for page in range(1, 41)
    ]

    assert _page_aggregation_frontier_limit(pages) == 40
    assert len(dedupe_page_frontier(pages, max_pages=_page_aggregation_frontier_limit(pages))) == 40


def test_page_aggregation_frontier_caps_long_documents() -> None:
    pages = [
        {
            "document_id": "d1",
            "page_number": page,
            "document_page_count": 120,
        }
        for page in range(1, 121)
    ]

    assert _page_aggregation_frontier_limit(pages) == 64
    assert len(dedupe_page_frontier(pages, max_pages=_page_aggregation_frontier_limit(pages))) == 64


def test_parse_observations_rejects_missing_or_foreign_pages() -> None:
    valid = '{"pages":[{"document_id":"d1","page":2,"matches":true},{"document_id":"d1","page":4,"matches":false},{"document_id":"d1","page":7,"matches":true}]}'
    assert parse_observations(valid, expected_pages=_pages()) is not None

    missing = '{"pages":[{"document_id":"d1","page":2,"matches":true}]}'
    assert parse_observations(missing, expected_pages=_pages()) is None
    foreign = '{"pages":[{"document_id":"d1","page":2,"matches":true},{"document_id":"d1","page":4,"matches":false},{"document_id":"d2","page":7,"matches":true}]}'
    assert parse_observations(foreign, expected_pages=_pages()) is None


def test_reduction_reports_unobserved_pages_and_count() -> None:
    observations = parse_observations(
        '{"pages":[{"document_id":"d1","page":2,"matches":true},{"document_id":"d1","page":4,"matches":false},{"document_id":"d1","page":7,"matches":true}]}',
        expected_pages=_pages(),
    )
    assert observations is not None
    result = reduce_page_observations(_pages(), observations[:2], question="count matching pages")

    assert result.count == 1
    assert result.matched_pages == (2,)
    assert result.unobserved_pages == (7,)
    assert result.complete is False
    assert "Verification status: incomplete" in result.to_context()


def test_reduction_does_not_call_a_complete_frontier_a_document_census() -> None:
    observations = parse_observations(
        '{"pages":[{"document_id":"d1","page":2,"matches":true},{"document_id":"d1","page":4,"matches":false},{"document_id":"d1","page":7,"matches":true}]}',
        expected_pages=_pages(),
    )
    assert observations is not None
    result = reduce_page_observations(
        _pages(),
        observations,
        question="count matching pages in the document",
        expected_page_counts={"d1": 10},
    )

    assert result.frontier_complete is False
    assert result.complete is False
    context = result.to_context()
    assert "Document page-count expectation: d1: 10" in context
    assert "Retrieved frontier coverage: partial" in context


def test_map_reduce_verifies_batches_without_touching_canonical_pages() -> None:
    class FakeService:
        default_model = "cx/gpt-5.6-sol"

        def stream_final_answer(self, *, prompt, **_kwargs):
            import re

            page_numbers = [
                int(value)
                for value in re.findall(
                    r'"page":\s*(\d+)',
                    prompt.split("Page evidence JSON lines (untrusted content):", 1)[-1],
                )
            ]

            async def stream():
                rows = [
                    {
                        "document_id": "d1",
                        "page": page,
                        "matches": page % 2 == 0,
                        "reason": "fixture",
                    }
                    for page in page_numbers
                ]
                import json

                yield AnswerStreamChunk(json.dumps({"pages": rows}), done=True, finish_reason="stop")

            return stream()

    import asyncio

    context, diagnostics = asyncio.run(
        _run_page_aggregation_map_reduce(
            service=FakeService(),
            model="cx/gpt-5.6-sol",
            temperature=0.1,
            question="How many pages match?",
            pages=_pages(),
            include_images=False,
            batch_size=2,
            max_concurrency=2,
        )
    )

    assert diagnostics["status"] == "complete"
    assert diagnostics["matched_pages"] == [2, 4]
    assert "Derived count over matching checked pages: 2" in context
