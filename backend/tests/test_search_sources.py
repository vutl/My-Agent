from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.db.sqlite import connect, init_db
from app.tools import build_registry
from app.tools import research_sources as research
from app.tools import search_providers as sp
from app.tools import web
from app.tools.registry import ToolContext, ToolRateLimiter


@pytest.fixture
def context(tmp_path, monkeypatch):
    for name in (*sp.DEFAULT_PROVIDERS.split(","), "semantic_scholar", "openalex"):
        monkeypatch.delenv(name.upper() + "_API_KEY", raising=False)
        monkeypatch.delenv(name.upper() + "_SEARCH_CAP", raising=False)
    for name in ("AYA_EVAL_MODE", "AYA_EVAL_WEB_ALLOW", "SEARXNG_URL", "AYA_SEARCH_PROVIDERS", "SEARCH_MODEL", "OPENAI_API_KEY", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(data_dir=tmp_path, openai_api_base="http://gateway/v1", openai_api_key="secret",
                        **{f"{x}_api_key": "secret" for x in (*sp.CAPS, "semantic_scholar", "openalex")},
                        searxng_url="http://localhost:8080")
    sp._COOLDOWNS.clear()
    monkeypatch.setattr(web, "TOOL_RATE_LIMITER", ToolRateLimiter())
    return ToolContext("run", "conversation", tmp_path, settings)


def mock_http(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**kw, transport=httpx.MockTransport(handler)))


def test_codex_responses_only(context, monkeypatch):
    payload = json.loads((Path(__file__).parent / "fixtures/search/codex_response.json").read_text())
    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert "chat/completions" not in str(request.url)
        assert body["tools"] == [{"type": "web_search"}]
        assert body["model"] == context.settings.router_model
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json=payload)

    mock_http(monkeypatch, respond)
    hits = asyncio.run(sp.Provider("codex", context.settings).search("q", max_results=8))
    assert len(requests) == 1 and len(hits) == 3
    assert hits[0].snippet == "Official evidence supports this claim."
    assert hits[1].snippet == "Another finding."
    assert hits[-1].url == "" and hits[-1].snippet == payload["output"][1]["content"][0]["text"]


@pytest.mark.parametrize("name,method,host,header,payload", [
    ("tavily", "POST", "api.tavily.com", "authorization", {"results": [{"title": "T", "url": "https://x.org", "content": "body"}]}),
    ("serper", "POST", "google.serper.dev", "x-api-key", {"organic": [{"title": "T", "link": "https://x.org", "snippet": "body"}]}),
    ("exa", "POST", "api.exa.ai", "x-api-key", {"results": [{"title": "T", "url": "https://x.org", "highlights": ["body"]}]}),
    ("brave", "GET", "api.search.brave.com", "x-subscription-token", {"web": {"results": [{"title": "T", "url": "https://x.org", "description": "body"}]}}),
    ("jina", "GET", "s.jina.ai", "authorization", {"data": [{"title": "T", "url": "https://x.org", "description": "body"}]}),
    ("searxng", "GET", "localhost", None, {"results": [{"title": "T", "url": "https://x.org", "content": "body"}]}),
])
def test_provider_shapes(context, monkeypatch, name, method, host, header, payload):
    def respond(request):
        assert request.method == method and request.url.host == host
        assert "secret" not in str(request.url)
        if header:
            assert "secret" in request.headers[header]
        body = json.loads(request.content) if request.content else dict(request.url.params)
        if name == "tavily":
            assert body["topic"] == "news" and body["query"] == "q"
        if name == "serper":
            assert body["gl"] == "vn" and body["hl"] == "vi"
        if name == "exa":
            assert body["type"] == "auto" and body["contents"]["highlights"] is True
        if name == "jina":
            assert request.headers["accept"] == "application/json" and body["q"] == "q"
        if name == "searxng":
            assert request.url.path == "/search" and body["format"] == "json" and body["language"] == "vi"
        return httpx.Response(200, json=payload)

    mock_http(monkeypatch, respond)
    hits = asyncio.run(sp.Provider(name, context.settings).search("q", max_results=4, lang="vi", recency_days=3))
    assert hits == [sp.SearchHit("T", "https://x.org", "body", name)]


def test_duckduckgo(context, monkeypatch):
    import ddgs

    class FakeDDGS:
        def __init__(self, **kw):
            assert kw["timeout"] == 15
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def text(self, query, **kw):
            assert query == "q" and kw["max_results"] == 2
            return [{"title": "T", "href": "https://x.org", "body": "body"}]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    assert asyncio.run(sp.Provider("duckduckgo", context.settings).search("q", max_results=2))[0].provider == "duckduckgo"


def test_caps_persist_and_serper_is_total(context, monkeypatch):
    init_db(context.settings.sqlite_db_path)
    for name in ("tavily", "serper"):
        provider = sp.Provider(name, context.settings)
        provider.cap = 2
        assert provider._reserve() and provider._reserve() and not provider._reserve()
        assert provider.used() == 2 and not provider.available()
        another = sp.Provider(name, context.settings)
        another.cap = 2
        assert not another.available()
        if name == "serper":
            assert another.period == "total"
    with connect(context.settings.sqlite_db_path) as db:
        db.execute("UPDATE search_usage SET period='1900-01' WHERE provider='tavily'")
    assert sp.Provider("tavily", context.settings).used() == 0
    assert sp.Provider("serper", context.settings).used() == 2


def test_cap_reservation_atomic(context):
    provider = sp.Provider("tavily", context.settings)
    provider.cap = 2
    with ThreadPoolExecutor(max_workers=5) as executor:
        assert sum(executor.map(lambda _: provider._reserve(), range(10))) == 2
    assert provider.used() == 2


@pytest.mark.parametrize("status", [401, 402, 429])
def test_cooldown(context, monkeypatch, status):
    mock_http(monkeypatch, lambda request: httpx.Response(status))
    provider = sp.Provider("tavily", context.settings)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.search("q", max_results=2))
    assert not sp.Provider("tavily", context.settings).available()
    assert provider.status()["cooldown_until"] > datetime.now(timezone.utc).timestamp()
    monkeypatch.setattr(sp.time, "time", lambda: 10**12)
    assert provider.available()


class FakeProvider:
    def __init__(self, name, result, calls, available=True, barrier=None):
        self.name, self.result, self.calls = name, result, calls
        self.enabled, self.barrier = available, barrier
    def available(self):
        return self.enabled
    async def search(self, query, **kw):
        self.calls.append(self.name)
        if self.barrier:
            self.barrier[self.name].set()
            other = next(x for x in self.barrier if x != self.name)
            await asyncio.wait_for(self.barrier[other].wait(), 1)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def hit(name, url="https://example.org/path", snippet="body"):
    return sp.SearchHit("T", url, snippet, name)


def test_fanout_concurrency_merge_summary(context, monkeypatch):
    calls = []
    async def run():
        barrier = {n: asyncio.Event() for n in ("codex", "tavily")}
        providers = {n: FakeProvider(n, [hit(n, u)], calls, barrier=barrier) for n, u in [
            ("codex", "https://WWW.Example.org/path/?utm_source=test#part"), ("tavily", "https://example.org/path")]}
        providers["codex"].result += [hit("codex", "", "whole answer")]
        providers["exa"] = FakeProvider("exa", [hit("exa", "https://other.org")], calls)
        providers["duckduckgo"] = FakeProvider("duckduckgo", [], calls)
        monkeypatch.setattr(web, "build_providers", lambda _: providers)
        return await web.web_search({"query": "q", "max_results": 2}, context)
    result = asyncio.run(run())
    assert calls == ["codex", "tavily", "exa"]
    assert len(result.sources) == 2 and result.sources[0].url == "https://example.org/path"
    assert result.data["results"][0]["providers"] == ["codex", "tavily"]
    assert "codex, tavily, exa" in result.summary
    assert result.data["codex_summary"] == "whole answer"
    assert result.content.index("chưa kiểm chứng") > result.content.index("https://other.org")
    assert result.untrusted


@pytest.mark.parametrize("answer,expected", [(False, ["tavily", "exa", "duckduckgo"]), (True, ["tavily", "exa"])])
def test_fallback_ddg_only_when_no_answer(context, monkeypatch, answer, expected):
    calls = []
    providers = {"codex": FakeProvider("codex", [], calls, available=False),
                 "tavily": FakeProvider("tavily", RuntimeError("secret"), calls),
                 "exa": FakeProvider("exa", [hit("exa")] if answer else RuntimeError(), calls),
                 "duckduckgo": FakeProvider("duckduckgo", [hit("duckduckgo")], calls)}
    monkeypatch.setattr(web, "build_providers", lambda _: providers)
    result = asyncio.run(web.web_search({"query": "q", "max_results": 8}, context))
    assert calls == expected and result.ok
    assert "secret" not in result.content


def test_explicit_provider_order(context, monkeypatch):
    calls = []
    providers = {n: FakeProvider(n, [hit(n)], calls) for n in ("tavily", "serper", "duckduckgo")}
    monkeypatch.setattr(web, "build_providers", lambda _: providers)
    asyncio.run(web.web_search({"query": "q", "providers": ["serper"]}, context))
    assert calls == ["serper"]


@pytest.mark.parametrize("url,expected", [
    ("https://WWW.Example.org/path/?UTM_Source=x&b=2&fbclid=z&a=1#x", "https://example.org/path?a=1&b=2"),
    ("https://example.org/", "https://example.org"), ("file:///private/x", ""),
    ("https://name:secret@example.org", ""), ("https://example.org:bad/", "")])
def test_canonical_url(url, expected):
    assert sp.canonical_url(url) == expected


def make_pdf(padding_bytes=0):
    import pymupdf
    with pymupdf.open() as pdf:
        pdf.set_metadata({"title": "Generated document"})
        for i in range(3):
            pdf.new_page().insert_text((50, 50), f"Content on physical page {i+1}.")
            if i == 0 and padding_bytes:
                xref = pdf.get_new_xref()
                pdf.update_object(xref, "<<>>")
                pdf.update_stream(xref, b"x" * padding_bytes, compress=False)
        return pdf.tobytes()


@pytest.mark.parametrize("mime", ["application/pdf", "application/octet-stream"])
def test_pdf_paging(context, monkeypatch, mime):
    monkeypatch.setattr(web, "_check_public_url", lambda _: None)
    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    monkeypatch.setattr(web, "record_network", lambda *a: None)
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=make_pdf(), headers={"content-type": mime}))
    result = asyncio.run(web.web_fetch({"url": "https://example.org/report", "start_page": 2, "max_pages": 1}, context))
    assert result.ok and result.untrusted
    assert "--- page 2 ---" in result.content and "physical page 2" in result.content
    assert "page 1" not in result.content and "page 3" not in result.content
    assert result.sources[0].title == "Generated document"


def test_pdf_default_pages_and_first_line_title():
    import pymupdf
    with pymupdf.open() as pdf:
        for i in range(10):
            pdf.new_page().insert_text((50, 50), f"Page title {i+1}")
        text, title = web._extract_pdf(pdf.tobytes(), 1, 8)
    assert "--- page 8 ---" in text and "--- page 9 ---" not in text
    assert title == "Page title 1"


@pytest.mark.parametrize("mime", ["application/pdf", "application/octet-stream"])
def test_pdf_reads_requested_page_beyond_html_byte_limit(context, monkeypatch, mime):
    body = make_pdf(padding_bytes=4_300_000)
    assert len(body) > web._MAX_PAGE_BYTES
    monkeypatch.setattr(web, "_check_public_url", lambda _: None)
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=body, headers={"content-type": mime}))
    result = asyncio.run(web.web_fetch({"url": "https://example.org/report", "start_page": 2, "max_pages": 1}, context))
    assert result.ok and "--- page 2 ---" in result.content
    assert "Content on physical page 2." in result.content
    assert "page 1" not in result.content and "page 3" not in result.content


@pytest.mark.parametrize("mime", ["application/pdf", "application/octet-stream"])
def test_pdf_download_limit_rejects_and_closes_stream(context, monkeypatch, mime):
    received, closed = [], []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in [b"%PDF" + b"x" * 1020] + [b"x" * 1024] * 10:
                received.append(chunk)
                yield chunk

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(web, "_check_public_url", lambda _: None)
    monkeypatch.setattr(web, "_MAX_PDF_BYTES", 2048)
    mock_http(monkeypatch, lambda request: httpx.Response(200, stream=Body(), headers={"content-type": mime}))
    result = asyncio.run(web.web_fetch({"url": "https://example.org/report"}, context))
    assert not result.ok and "PDF download exceeds" in result.content and "2048 bytes" in result.content
    assert len(received) == 3 and closed == [True]
    assert not result.sources


def test_short_pdf_page_is_preserved_with_jina_enabled(context, monkeypatch):
    calls = []
    monkeypatch.setattr(web, "_check_public_url", lambda _: None)

    def respond(request):
        calls.append(request)
        if request.url.host == "r.jina.ai":
            return httpx.Response(200, json={"data": {"content": "Content on physical page 1. " * 100}})
        return httpx.Response(200, content=make_pdf(), headers={"content-type": "application/pdf"})

    mock_http(monkeypatch, respond)
    result = asyncio.run(web.web_fetch({"url": "https://example.org/report", "start_page": 2, "max_pages": 1}, context))
    assert result.ok and len(calls) == 1
    assert result.content == "--- page 2 ---\nContent on physical page 2."
    assert result.sources[0].title == "Generated document"


@pytest.mark.parametrize("status", [200, 403, 429])
def test_jina_fallback(context, monkeypatch, status):
    calls, guards, audit = [], [], []
    monkeypatch.setattr(web, "_check_public_url", lambda url: guards.append(url))
    monkeypatch.setattr(web, "record_network", lambda *args: audit.append(args))
    def respond(request):
        calls.append(request)
        if request.url.host == "r.jina.ai":
            assert guards == ["https://example.org"]
            assert request.headers["Authorization"] == "Bearer secret"
            return httpx.Response(200, json={"data": {"title": "Reader title", "content": "Readable " * 100}}, headers={"content-type": "application/json"})
        return httpx.Response(status, text="short")
    mock_http(monkeypatch, respond)
    result = asyncio.run(web.web_fetch({"url": "https://example.org"}, context))
    assert len(calls) == 2 and "Readable" in result.content
    assert result.sources[0].url == "https://example.org"
    assert audit[-1] == ("https://r.jina.ai/https://example.org", "web_fetch_jina")


def test_jina_eval_block_and_keyless_rate_limit(context, monkeypatch):
    calls = []
    mock_http(monkeypatch, lambda req: (calls.append(req) or httpx.Response(200, text="short")))
    monkeypatch.setattr(web, "_check_public_url", lambda _: None)
    monkeypatch.setattr(web, "record_network", lambda *a: None)
    monkeypatch.setenv("AYA_EVAL_MODE", "1")
    result = asyncio.run(web.web_fetch({"url": "https://example.org"}, context))
    assert len(calls) == 1 and result.content == "short"
    monkeypatch.setenv("AYA_EVAL_WEB_ALLOW", "r.jina.ai:443")
    context.settings.jina_api_key = ""
    async def run():
        for _ in range(16):
            await web._jina_read("https://example.org", context)
    asyncio.run(run())
    assert len(calls) == 16
    assert all("authorization" not in r.headers for r in calls[1:])


@pytest.mark.parametrize("redirect", [False, True])
def test_ssrf_before_jina(context, monkeypatch, redirect):
    calls = []
    def guard(url):
        if "127.0.0.1" in url:
            raise web.UnsafeUrlError("private address")
    monkeypatch.setattr(web, "_check_public_url", guard)
    mock_http(monkeypatch, lambda r: (calls.append(r) or httpx.Response(302, headers={"location": "http://127.0.0.1"})))
    result = asyncio.run(web.web_fetch({"url": "https://example.org" if redirect else "http://127.0.0.1"}, context))
    assert not result.ok and len(calls) == int(redirect)
    assert all(r.url.host != "r.jina.ai" for r in calls)


def test_paper_alias_merging():
    papers = [research.Paper("First title", doi="https://doi.org/10.1/ABC", providers=["crossref"]),
              research.Paper("Different title", arxiv_id="2401.12345v2", providers=["arxiv"]),
              research.Paper("A third title", doi="10.1/abc", arxiv_id="2401.12345", pdf_url="https://arxiv.org/pdf/2401.12345", providers=["semantic_scholar"]),
              research.Paper("First: title!", authors=[str(i) for i in range(8)], abstract="x" * 1000, providers=["openalex"])]
    result = research.merge_papers(papers)
    assert len(result) == 1
    assert set(result[0].providers) == set(research.PAPER_SOURCES)
    assert len(result[0].authors) == 5 and len(result[0].abstract) == 600
    assert result[0].pdf_url.endswith("2401.12345")


@pytest.mark.parametrize("identifiers", [
    ({"doi": "10.1/first"}, {"doi": "10.1/second"}),
    ({"arxiv_id": "2401.12345"}, {"arxiv_id": "2401.54321"}),
])
def test_same_title_does_not_merge_conflicting_paper_identifiers(identifiers):
    first, second = identifiers
    result = research.merge_papers([
        research.Paper("Editorial", authors=["First author"], providers=["crossref"], **first),
        research.Paper("Editorial", authors=["Second author"], citation_count=42, providers=["openalex"], **second),
    ])
    assert len(result) == 2
    assert result[0].authors == ["First author"] and result[0].citation_count is None
    assert result[0].providers == ["crossref"]
    assert result[1].authors == ["Second author"] and result[1].citation_count == 42


def test_paper_identifier_match_takes_priority_over_title():
    result = research.merge_papers([
        research.Paper("Editorial", doi="10.1/first", providers=["crossref"]),
        research.Paper("Other title", doi="10.1/second", providers=["arxiv"]),
        research.Paper("Editorial", doi="10.1/second", citation_count=42, providers=["openalex"]),
    ])
    assert len(result) == 2 and result[0].citation_count is None
    assert result[1].doi == "10.1/second" and result[1].citation_count == 42
    assert result[1].providers == ["arxiv", "openalex"]


def test_title_only_paper_does_not_bridge_conflicting_identifiers():
    result = research.merge_papers([
        research.Paper("Editorial", doi="10.1/first"),
        research.Paper("Editorial", doi="10.1/second"),
        research.Paper("Editorial", citation_count=42),
    ])
    assert len(result) == 3
    assert all(paper.citation_count is None for paper in result[:2])


def test_paper_http_and_partial_failure(context, monkeypatch):
    seen = []
    def respond(request):
        seen.append(request)
        assert "secret" not in str(request.url)
        if request.url.host == "export.arxiv.org":
            assert "submittedDate" in request.url.params["search_query"]
            return httpx.Response(200, text='''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"><entry><id>https://arxiv.org/abs/2401.12345v1</id><title>Study</title><published>2024-01-02</published><author><name>A</name></author><summary>Abstract</summary><arxiv:doi>10.1/test</arxiv:doi><link title="pdf" href="https://arxiv.org/pdf/2401.12345"/></entry></feed>''')
        if request.url.host == "api.semanticscholar.org":
            assert request.headers["x-api-key"] == "secret" and request.url.params["year"] == "2020-"
            return httpx.Response(200, json={"data": [{"title": "Study", "authors": [{"name": "A"}], "year": 2024, "citationCount": 5, "externalIds": {"DOI": "10.1/test"}, "url": "https://example.org/paper"}]})
        if request.url.host == "api.openalex.org":
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(200, json={"results": [{"display_name": "Study", "publication_year": 2024, "doi": "https://doi.org/10.1/test", "abstract_inverted_index": {"abstract": [1], "The": [0]}, "primary_location": {"landing_page_url": "https://example.org/paper"}}]})
        return httpx.Response(503)
    mock_http(monkeypatch, respond)
    result = asyncio.run(research.search_papers({"query": "science", "year_from": 2020}, context))
    assert len(seen) == 4 and result.ok and result.untrusted
    assert len(result.data["papers"]) == 1 and result.data["papers"][0]["citation_count"] == 5
    assert "https://arxiv.org/pdf/2401.12345" in result.content
    assert result.sources[0].url == "https://arxiv.org/abs/2401.12345v1"


def test_crossref_and_openalex_parsing(context, monkeypatch):
    def respond(request):
        if request.url.host == "api.crossref.org":
            return httpx.Response(200, json={"message": {"items": [{"title": ["Crossref study"], "author": [{"given": "A", "family": "B"}], "issued": {"date-parts": [[2023]]}, "container-title": ["Journal"], "abstract": "<jats:p>Evidence</jats:p>", "DOI": "10.1/xyz", "URL": "https://doi.org/10.1/xyz", "link": [{"content-type": "application/pdf", "URL": "https://example.org/pdf"}]}]}})
        return httpx.Response(200, json={"results": [{"display_name": "Other study", "publication_year": 2025, "abstract_inverted_index": {"second": [1], "First": [0]}, "authorships": [{"author": {"display_name": "C"}}], "best_oa_location": {"pdf_url": "https://example.org/oa"}, "primary_location": {"source": {"display_name": "Venue"}, "landing_page_url": "https://example.org/study"}, "cited_by_count": 2}]})
    mock_http(monkeypatch, respond)
    result = asyncio.run(research.search_papers({"query": "q", "sources": ["crossref", "openalex"]}, context))
    crossref, oa = result.data["papers"]
    assert crossref["authors"] == ["A B"] and crossref["year"] == 2023 and crossref["abstract"] == "Evidence"
    assert oa["abstract"] == "First second" and oa["venue"] == "Venue" and oa["pdf_url"].endswith("/oa")


def test_wikipedia(context, monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        if "/search/page" in request.url.path:
            assert request.url.host == "en.wikipedia.org"
            return httpx.Response(200, json={"pages": [{"key": "Test_page", "title": "Test page"}]})
        return httpx.Response(200, json={"title": "Test page", "extract": "Encyclopedia extract", "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Test_page"}}})
    mock_http(monkeypatch, respond)
    result = asyncio.run(research.wikipedia({"query": "test", "lang": "en"}, context))
    assert len(calls) == 2 and result.untrusted and result.ok
    assert result.sources[0].url.endswith("/Test_page") and "Encyclopedia extract" in result.content


def test_status_endpoint_no_secrets(context):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.tools import router
    from app.core.config import get_settings
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: context.settings
    response = TestClient(app).get("/tools/search-providers")
    assert response.status_code == 200
    assert "secret" not in response.text and "gateway" not in response.text
    assert len(response.json()) == 8
    assert set(response.json()[0]) == {"name", "configured", "available", "used_this_period", "cap", "cooldown_until"}


def test_registry():
    registry = build_registry()
    for name in ("search_papers", "wikipedia"):
        tool = registry.get(name)
        assert tool.group == "research" and tool.rate_limit == (20, 60.0)
        assert not tool.side_effects and tool.permission == "auto"


def test_zero_cap_and_uncapped_usage(context):
    context.settings.tavily_search_cap = 0
    provider = sp.Provider("tavily", context.settings)
    assert not provider.available() and not provider._reserve()
    provider = sp.Provider("codex", context.settings)
    assert provider.cap is None and provider._reserve() and provider.used() == 1
    assert sp.Provider("codex", context.settings).status()["used_this_period"] == 1


def test_search_configuration_env(monkeypatch):
    from app.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("SEARCH_MODEL", "cx/gpt-6.1-sol")
    monkeypatch.setenv("TAVILY_SEARCH_CAP", "0")
    monkeypatch.setenv("AYA_SEARCH_PROVIDERS", "serper,duckduckgo")
    monkeypatch.setenv("SERPER_API_KEY", "configured-secret")
    try:
        settings = get_settings()
        assert settings.search_model == "cx/gpt-6.1-sol"
        assert settings.tavily_search_cap == 0
        assert settings.aya_search_providers == "serper,duckduckgo"
        assert settings.serper_api_key == "configured-secret"
    finally:
        get_settings.cache_clear()


def test_search_model_policy(context):
    from app.core.config import validate_runtime_model_policy
    context.settings.search_model = "ag/non-codex-model"
    with pytest.raises(RuntimeError, match="SEARCH_MODEL"):
        validate_runtime_model_policy(context.settings)


def test_search_configured_order_and_invalid_urls(context, monkeypatch):
    calls = []
    context.settings.aya_search_providers = "exa,tavily,duckduckgo"
    providers = {"exa": FakeProvider("exa", [hit("exa", "file:///unsafe")], calls),
                 "tavily": FakeProvider("tavily", [], calls),
                 "duckduckgo": FakeProvider("duckduckgo", [hit("duckduckgo")], calls)}
    monkeypatch.setattr(web, "build_providers", lambda _: providers)
    result = asyncio.run(web.web_search({"query": "q"}, context))
    assert calls == ["exa", "tavily", "duckduckgo"]
    assert result.data["providers"] == ["duckduckgo"]


def test_merge_rank_by_agreement_then_position():
    result = web._merge_hits([(0, hit("tavily", "https://a.org")),
                             (1, hit("tavily", "https://b.org")),
                             (2, hit("codex", "https://b.org")),
                             (0, hit("codex", "https://c.org"))])
    assert [x["url"] for x in result] == ["https://b.org", "https://a.org", "https://c.org"]


def test_pdf_character_cap():
    import pymupdf
    with pymupdf.open() as pdf:
        for i in range(12):
            page = pdf.new_page()
            page.insert_textbox((20, 20, 500, 800), ("Readable evidence " * 15 + "\n") * 20, fontsize=8)
        text, _ = web._extract_pdf(pdf.tobytes(), 1, 12)
    assert len(text) == web._MAX_PAGE_CHARS


def test_provider_network_audit(context, monkeypatch):
    audited = []
    monkeypatch.setattr(sp, "record_network", lambda *args: audited.append(args))
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"results": []}))
    asyncio.run(sp.Provider("tavily", context.settings).search("q", max_results=1))
    assert audited == [("https://api.tavily.com/search", "web_search")]


def test_all_search_failures_do_not_expose_errors(context, monkeypatch):
    calls = []
    providers = {"tavily": FakeProvider("tavily", RuntimeError("secret"), calls)}
    monkeypatch.setattr(web, "build_providers", lambda _: providers)
    result = asyncio.run(web.web_search({"query": "q"}, context))
    assert not result.ok and result.untrusted and "secret" not in result.content


def test_codex_citation_marker_uses_preceding_sentence():
    text = "Background. Evidence for the finding. [1]"
    start = text.index("[1]")
    hits = sp.parse_codex({"output": [{"type": "message", "content": [{"text": text,
        "annotations": [{"type": "url_citation", "url": "https://example.org", "start_index": start, "end_index": start + 3}]}]}]})
    assert hits[0].snippet == "Evidence for the finding."
