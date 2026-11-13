"""Read-only academic metadata and encyclopedia lookups."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET

import httpx

from app.core.eval_mode import record_network
from app.tools.registry import Tool, ToolContext, ToolResult, ToolSource
from app.tools.search_providers import setting

PAPER_SOURCES = ("arxiv", "semantic_scholar", "openalex", "crossref")
_HEADERS = {"User-Agent": "AyaAssistant/1.0 (research metadata lookup)"}


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citation_count: int | None = None
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pdf_url: str = ""
    url: str = ""
    providers: list[str] = field(default_factory=list)


def _doi(value: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).casefold()


def _arxiv(value: str) -> str:
    return re.sub(r"v\d+$", "", re.sub(r"^https?://arxiv.org/(?:abs|pdf)/", "", value).removesuffix(".pdf"))


def _title(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def merge_papers(papers: list[Paper]) -> list[Paper]:
    # Identifier aliases are retained across merges (including bridge records
    # carrying both DOI and arXiv ID), not just a single preferred key.
    groups: list[tuple[Paper, set[str]]] = []
    for paper in papers:
        paper.doi, paper.arxiv_id = _doi(paper.doi), _arxiv(paper.arxiv_id)
        paper.authors, paper.abstract = paper.authors[:5], paper.abstract[:600]
        paper.url = paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else "")
        aliases = {key for key in ("doi:" + paper.doi if paper.doi else "",
                                  "arxiv:" + paper.arxiv_id if paper.arxiv_id else "",
                                  "title:" + _title(paper.title) if _title(paper.title) else "") if key}
        identifiers = {key for key in aliases if not key.startswith("title:")}
        matching = [(p, keys) for p, keys in groups if keys & identifiers]
        if not matching:
            # Titles are only a fallback: common titles must not join different
            # DOI/arXiv identities, even indirectly via a record with no IDs.
            matching = [(p, keys) for p, keys in groups if keys & aliases and not any(
                {key for key in keys if key.startswith(prefix)}
                and {key for key in aliases if key.startswith(prefix)}
                and not {key for key in keys & aliases if key.startswith(prefix)}
                for prefix in ("doi:", "arxiv:")
            )]
            if len(matching) > 1:
                matching = []  # Ambiguous title-only metadata belongs to neither identity.
        if not matching:
            groups.append((paper, aliases))
            continue
        target, keys = matching[0]
        for other, other_keys in matching[1:] + [(paper, aliases)]:
            keys.update(other_keys)
            for name, value in asdict(other).items():
                if name == "providers":
                    target.providers = list(dict.fromkeys(target.providers + value))
                elif not getattr(target, name) and value:
                    setattr(target, name, value)
        groups = [(p, k) for p, k in groups if p is target or all(p is not other for other, _ in matching)]
    return [p for p, _ in groups]


async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    record_network(url, "research_search")
    response = await client.get(url, **kwargs)
    response.raise_for_status()
    return response


def _year(value: Any) -> int | None:
    try:
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return None


async def _papers(source: str, query: str, count: int, year: int | None, settings: Any) -> list[Paper]:
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        if source == "arxiv":
            response = await _get(client, "https://export.arxiv.org/api/query", params={
                "search_query": "all:" + query + (f" AND submittedDate:[{year}01010000 TO 999912312359]" if year else ""),
                "max_results": count})
            ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
            result = []
            for entry in ET.fromstring(response.text).findall("a:entry", ns):
                url = entry.findtext("a:id", "", ns)
                result.append(Paper(title=" ".join(entry.findtext("a:title", "", ns).split()),
                    authors=[x.findtext("a:name", "", ns) for x in entry.findall("a:author", ns)][:5],
                    year=_year(entry.findtext("a:published", "", ns)),
                    venue=entry.findtext("x:journal_ref", "", ns),
                    abstract=" ".join(entry.findtext("a:summary", "", ns).split())[:600],
                    doi=entry.findtext("x:doi", "", ns), arxiv_id=_arxiv(url), url=url,
                    pdf_url=next((x.get("href", "") for x in entry.findall("a:link", ns) if x.get("title") == "pdf" or x.get("type") == "application/pdf"), ""),
                    providers=[source]))
            return result
        headers = {}
        if source == "semantic_scholar":
            key = setting(settings, "semantic_scholar_api_key")
            if key:
                headers["x-api-key"] = key
            params = {"query": query, "limit": count,
                      "fields": "title,authors,year,venue,citationCount,abstract,externalIds,openAccessPdf,url"}
            if year:
                params["year"] = f"{year}-"
            data = (await _get(client, "https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers)).json()
            return [Paper(title=x.get("title") or "", authors=[a.get("name", "") for a in x.get("authors", [])][:5],
                year=x.get("year"), venue=x.get("venue") or "", citation_count=x.get("citationCount"),
                abstract=(x.get("abstract") or "")[:600], doi=(x.get("externalIds") or {}).get("DOI") or "",
                arxiv_id=(x.get("externalIds") or {}).get("ArXiv") or "", pdf_url=(x.get("openAccessPdf") or {}).get("url") or "",
                url=x.get("url") or "", providers=[source]) for x in data.get("data", [])]
        if source == "openalex":
            params = {"search": query, "per-page": count}
            if year:
                params["filter"] = f"from_publication_date:{year}-01-01"
            if mailto := setting(settings, "openalex_mailto"):
                params["mailto"] = mailto
            if key := setting(settings, "openalex_api_key"):
                headers["Authorization"] = f"Bearer {key}"
            data = (await _get(client, "https://api.openalex.org/works", params=params, headers=headers)).json()
            result = []
            for x in data.get("results", []):
                abstract = x.get("abstract_inverted_index") or {}
                words = sorted((position, word) for word, positions in abstract.items() for position in positions)
                location = x.get("primary_location") or {}
                oa = x.get("best_oa_location") or {}
                result.append(Paper(title=x.get("display_name") or x.get("title") or "",
                    authors=[(a.get("author") or {}).get("display_name", "") for a in x.get("authorships", [])][:5],
                    year=x.get("publication_year"), venue=(location.get("source") or {}).get("display_name") or "",
                    citation_count=x.get("cited_by_count"), abstract=" ".join(w for _, w in words)[:600],
                    doi=x.get("doi") or "", pdf_url=oa.get("pdf_url") or location.get("pdf_url") or "",
                    url=location.get("landing_page_url") or x.get("doi") or x.get("id") or "", providers=[source]))
            return result
        params = {"query": query, "rows": count}
        if year:
            params["filter"] = f"from-pub-date:{year}-01-01"
        data = (await _get(client, "https://api.crossref.org/works", params=params)).json()
        result = []
        for x in (data.get("message") or {}).get("items", []):
            dates = (x.get("published") or x.get("issued") or {}).get("date-parts") or [[]]
            result.append(Paper(title=next(iter(x.get("title") or []), ""),
                authors=[a.get("name") or " ".join(filter(None, [a.get("given"), a.get("family")])) for a in x.get("author", [])][:5],
                year=dates[0][0] if dates[0] else None, venue=next(iter(x.get("container-title") or []), ""),
                citation_count=x.get("is-referenced-by-count"), abstract=re.sub("<[^>]+>", "", x.get("abstract") or "")[:600],
                doi=x.get("DOI") or "", url=x.get("URL") or "",
                pdf_url=next((link.get("URL", "") for link in x.get("link", []) if link.get("content-type") == "application/pdf"), ""),
                providers=[source]))
        return result


async def search_papers(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(arguments.get("query") or "").split())[:300]
    if not query:
        return ToolResult("Empty query.", "Thiếu từ khóa tìm bài báo", ok=False)
    count = max(1, min(int(arguments.get("max_results") or 8), 20))
    selected = list(dict.fromkeys(x for x in arguments.get("sources", PAPER_SOURCES) if x in PAPER_SOURCES))
    results = await asyncio.gather(*(_papers(x, query, count, arguments.get("year_from"), context.settings)
                                     for x in selected), return_exceptions=True)
    papers = merge_papers([p for result in results if isinstance(result, list) for p in result])
    if year := arguments.get("year_from"):
        papers = [p for p in papers if p.year is not None and p.year >= year]
    papers = papers[:count]
    content = []
    for i, paper in enumerate(papers, 1):
        content.append(f"{i}. {paper.title}\nAuthors: {', '.join(paper.authors)}; year: {paper.year}; venue: {paper.venue}; citations: {paper.citation_count}\n"
                       f"{paper.abstract}\nDOI: {paper.doi}; arXiv: {paper.arxiv_id}\nLanding: {paper.url}\nPDF: {paper.pdf_url}")
    return ToolResult("\n\n".join(content) or "No papers found.", f"Tìm bài báo — {len(papers)} kết quả",
        sources=[ToolSource(title=p.title, url=p.url, snippet=p.abstract) for p in papers if p.url],
        untrusted=True, ok=any(isinstance(x, list) for x in results),
        data={"query": query, "papers": [asdict(p) for p in papers],
              "providers": [name for name, result in zip(selected, results) if isinstance(result, list)]})


async def wikipedia(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(arguments.get("query") or "").split())[:300]
    lang = "vi" if arguments.get("lang", "vi") == "vi" else "en"
    if not query:
        return ToolResult("Empty query.", "Thiếu từ khóa Wikipedia", ok=False)
    base = f"https://{lang}.wikipedia.org"
    try:
        async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
            response = await _get(client, base + "/w/rest.php/v1/search/page", params={"q": query, "limit": 3})
            pages = response.json().get("pages", [])[:3]
            results = await asyncio.gather(*(_get(client, base + "/api/rest_v1/page/summary/" + quote(p.get("key") or p["title"], safe=""))
                                             for p in pages), return_exceptions=True)
        sources, content = [], []
        for result in results:
            if isinstance(result, BaseException):
                continue
            item = result.json()
            title, extract = str(item.get("title") or ""), str(item.get("extract") or "")
            url = ((item.get("content_urls") or {}).get("desktop") or {}).get("page") or base + "/wiki/" + quote(title)
            sources.append(ToolSource(title=title, url=url, snippet=extract[:600]))
            content.append(f"{title} ({url})\n{extract[:4000]}")
        return ToolResult("\n\n".join(content) or "No results.", f"Wikipedia — {len(sources)} kết quả", sources=sources, untrusted=True)
    except Exception:
        return ToolResult("Wikipedia lookup failed.", "Không tra được Wikipedia", ok=False, untrusted=True)


SEARCH_PAPERS = Tool(name="search_papers", label="Tìm bài báo khoa học", group="research", rate_limit=(20, 60.0),
    description="Search arXiv, Semantic Scholar, OpenAlex and Crossref for papers. Returns metadata and open PDF URLs to read with web_fetch.",
    parameters={"type": "object", "properties": {"query": {"type": "string"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
        "year_from": {"type": "integer", "minimum": 1},
        "sources": {"type": "array", "items": {"type": "string", "enum": list(PAPER_SOURCES)}}}, "required": ["query"]}, handler=search_papers)
WIKIPEDIA = Tool(name="wikipedia", label="Tra Wikipedia", group="research", rate_limit=(20, 60.0),
    description="Search Vietnamese or English Wikipedia and read the top three summaries for background context.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}, "lang": {"type": "string", "enum": ["vi", "en"], "default": "vi"}},
                "required": ["query"]}, handler=wikipedia)
