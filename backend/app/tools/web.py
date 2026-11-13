"""Multi-provider search and public HTML/PDF reading.

Fetched pages are untrusted. ``web_fetch`` only reaches public http(s) hosts:
loopback, private, link-local and reserved addresses are refused before the
request and after every redirect, so a page cannot steer Aya into the local
backend, the router gateway or the LAN.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.core.eval_mode import eval_mode_enabled, record_network, web_allowed
from app.tools.registry import TOOL_RATE_LIMITER, Tool, ToolContext, ToolResult, ToolSource
from app.tools.search_providers import DEFAULT_PROVIDERS, SearchHit, SearchProvider, build_providers, canonical_url, setting

_MAX_PAGE_BYTES = 3_000_000
_MAX_PDF_BYTES = 30_000_000
_MAX_PAGE_CHARS = 14_000
_MAX_DOCUMENT_CHARS = 3_000_000
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AyaAssistant/1.0"


class UnsafeUrlError(ValueError):
    pass


class PdfTooLargeError(ValueError):
    pass


def _check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("only http(s) URLs with a host are allowed")
    if web_allowed(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        return  # evaluation mode: one explicitly allowed fixture server
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"cannot resolve {parsed.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private or address.is_loopback or address.is_link_local
            or address.is_reserved or address.is_multicast or address.is_unspecified
        ):
            raise UnsafeUrlError(f"{parsed.hostname} resolves to a non-public address")


def _merge_hits(hits: list[tuple[int, SearchHit]]) -> list[dict[str, Any]]:
    merged = {}
    for position, hit in hits:
        key = canonical_url(hit.url)
        if not key:
            continue
        if key not in merged:
            # Canonicalization is only for merging; navigation needs the original URL.
            merged[key] = {"title": hit.title or hit.url, "url": hit.url, "snippet": hit.snippet,
                           "published": hit.published, "providers": [], "position": position}
        item = merged[key]
        item["position"] = min(item["position"], position)
        if hit.provider not in item["providers"]:
            item["providers"].append(hit.provider)
    return sorted(merged.values(), key=lambda x: (-len(x["providers"]), x["position"]))


async def web_search(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(arguments.get("query") or "").split())[:300]
    if not query:
        return ToolResult(content="Empty query.", summary="Thiếu từ khóa tìm kiếm", ok=False)
    count = max(1, min(int(arguments.get("max_results") or 5), 8))
    results, used, summary = await search_hits(query, max_results=count, lang=arguments.get("lang"),
        recency_days=arguments.get("recency_days"), providers=arguments.get("providers"), context=context)
    summaries = [summary] if summary else []
    content = "\n".join(f"{i}. {item['title']} ({item['url']}): {item['snippet']}" for i, item in enumerate(results, 1)) or "No results."
    data = {"providers": used, "query": query, "results": results}
    if summaries:
        data["codex_summary"] = "\n\n".join(summaries)
        content += "\n\nTóm tắt từ công cụ tìm kiếm (chưa kiểm chứng):\n" + data["codex_summary"]
    return ToolResult(content=content, summary=f"Tìm “{query}” — {len(results)} kết quả ({', '.join(used)})",
                      sources=[ToolSource(title=x["title"], url=x["url"], snippet=x["snippet"]) for x in results],
                      untrusted=True, ok=bool(results), data=data)


async def search_hits(query: str, *, max_results: int = 5, lang: str | None = None,
                      recency_days: int | None = None, providers: list[str] | None = None,
                      context: ToolContext) -> tuple[list[dict[str, Any]], list[str], str]:
    count = max(1, min(max_results, 8))
    available_providers = build_providers(context.settings)
    order = providers
    if order is None:
        order = str(setting(context.settings, "aya_search_providers", DEFAULT_PROVIDERS)).split(",")
    order = list(dict.fromkeys(str(name).strip().lower() for name in order))
    available = [available_providers[name] for name in order if name in available_providers and available_providers[name].available()]
    primary = [p for p in available if p.name != "duckduckgo"]
    fallback = next((p for p in available if p.name == "duckduckgo"), None)
    hits: list[tuple[int, SearchHit]] = []
    used: list[str] = []
    summaries: list[str] = []

    async def run(provider: SearchProvider) -> list[SearchHit] | None:
        try:
            return await provider.search(query, max_results=count, lang=lang,
                                         recency_days=recency_days)
        except Exception:  # Vendor details may contain credentials; never expose them.
            return None

    def collect(provider: SearchProvider, result: list[SearchHit] | None) -> None:
        result = [hit for hit in result or [] if canonical_url(hit.url) or (hit.provider == "codex" and not hit.url and hit.snippet.strip())]
        if not result:
            return
        used.append(provider.name)
        for position, hit in enumerate(result):
            if hit.provider == "codex" and not hit.url:
                summaries.append(hit.snippet)
            else:
                hits.append((position, hit))

    first = primary[:2]
    for provider, result in zip(first, await asyncio.gather(*(run(p) for p in first))):
        collect(provider, result)
    for provider in primary[2:]:
        if len(_merge_hits(hits)) >= count:
            break
        collect(provider, await run(provider))
    if not hits and fallback:
        collect(fallback, await run(fallback))
    results = _merge_hits(hits)[:count]
    return results, used, "\n\n".join(summaries)


def _extract_text(html: str, url: str) -> tuple[str, str]:
    import trafilatura  # type: ignore

    text = trafilatura.extract(html, url=url, include_tables=True, include_links=False, favor_recall=True) or ""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = str(getattr(metadata, "title", "") or "") if metadata else ""
    return text, title


def _extract_pdf(body: bytes, start_page: int, max_pages: int, *, max_chars: int = _MAX_PAGE_CHARS) -> tuple[str, str]:
    import pymupdf

    with pymupdf.open(stream=body, filetype="pdf") as document:
        chunks = []
        remaining = max_chars
        for index in range(start_page - 1, min(len(document), start_page - 1 + max_pages)):
            chunk = f"--- page {index + 1} ---\n" + document[index].get_text()
            chunks.append(chunk[:remaining])
            remaining -= len(chunks[-1])
            if remaining <= 0:
                break
        text = "\n".join(chunks)[:max_chars]
        title = str((document.metadata or {}).get("title") or "")
        if not title and len(document):
            title = _largest_text_line(document[0])
        return text, title


async def _jina_read(url: str, context: ToolContext) -> tuple[str, str] | None:
    # The target has already been checked at every original redirect hop.
    if eval_mode_enabled() and not web_allowed("r.jina.ai", 443):
        return None
    if not TOOL_RATE_LIMITER.allow(JINA_READER):
        return None
    headers = {"Accept": "application/json"}
    key = setting(context.settings, "jina_api_key")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    target = "https://r.jina.ai/" + url
    record_network(target, "web_fetch_jina")
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        response = await client.get(target, headers=headers)
        response.raise_for_status()
        if "json" in response.headers.get("content-type", ""):
            data = response.json().get("data") or {}
            return str(data.get("content") or ""), str(data.get("title") or "")
        return response.text, ""


def _largest_text_line(page: Any) -> str:
    """The first page's title is usually its largest text: join the lines set
    in the largest font size (a title can wrap over two lines)."""
    lines: list[tuple[float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            if abs(line.get("dir", (1, 0))[0] - 1) > 0.01:
                continue  # Rotated margin stamps (arXiv ids) are not titles.
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            size = max((span.get("size", 0) for span in line.get("spans", [])), default=0)
            if len(text) >= 4 and any(c.isalpha() for c in text):
                lines.append((round(size, 1), text))
    if not lines:
        return ""
    biggest = max(size for size, _ in lines)
    return " ".join(text for size, text in lines if size == biggest)[:200]


async def _decodo_read(url: str, context: ToolContext) -> tuple[str, str] | None:
    """Last-resort reader (Decodo web data API) for pages that block both a
    direct fetch and the Jina reader, or need JavaScript."""
    key = setting(context.settings, "decodo_auth_token")
    if not key or (eval_mode_enabled() and not web_allowed("data.decodo.com", 443)):
        return None
    if not TOOL_RATE_LIMITER.allow(DECODO_READER):
        return None
    record_network("https://data.decodo.com/v1/scrape", "web_fetch_decodo")
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        response = await client.post("https://data.decodo.com/v1/scrape", json={"url": url, "markdown": True},
                                     headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        response.raise_for_status()
        results = response.json().get("results") or [{}]
        content = results[0].get("content")
        return (content if isinstance(content, str) else ""), ""


async def _read_fetch_body(response: httpx.Response) -> tuple[bytes, bool]:
    is_pdf = "application/pdf" in response.headers.get("content-type", "").lower()
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        is_pdf = is_pdf or body.startswith(b"%PDF")
        if is_pdf:
            if len(body) > _MAX_PDF_BYTES:
                raise PdfTooLargeError
        elif len(body) >= _MAX_PAGE_BYTES:
            return bytes(body[:_MAX_PAGE_BYTES]), False
    return bytes(body), is_pdf


class _PageLinks(HTMLParser):
    def __init__(self, base):
        super().__init__()
        self.base, self.links = base, []

    def handle_starttag(self, tag, attrs):
        if tag != "a" or len(self.links) >= 100:
            return
        href = dict(attrs).get("href")
        target = urljoin(self.base, href or "")
        if href and urlparse(target).scheme in {"https", "http"} and target not in self.links:
            self.links.append(target)


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    final_url: str
    title: str
    text: str
    kind: str
    published: str | None = None
    links: tuple[str, ...] = ()


async def fetch_document(url: str, *, context: ToolContext, start_page: int = 1,
                         max_pages: int = 8) -> FetchedDocument:
    current = url
    text, title = "", ""
    is_pdf = False
    links = ()
    async with httpx.AsyncClient(timeout=25, follow_redirects=False, headers={"User-Agent": _USER_AGENT}) as client:
        for _hop in range(6):
            await asyncio.to_thread(_check_public_url, current)
            record_network(current, "web_fetch")
            async with client.stream("GET", current) as response:
                if response.is_redirect and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                if response.status_code not in {403, 429}:
                    response.raise_for_status()
                    body, is_pdf = await _read_fetch_body(response)
                    content_type = response.headers.get("content-type", "").lower()
                    if is_pdf:
                        text, title = await asyncio.to_thread(_extract_pdf, body,
                            max(1, int(start_page or 1)),
                            max(1, min(50, int(max_pages or 8))), max_chars=_MAX_DOCUMENT_CHARS)
                    else:
                        raw = body.decode(response.encoding or "utf-8", errors="replace")
                        if "html" in content_type or raw.lstrip().startswith("<"):
                            text, title = await asyncio.to_thread(_extract_text, raw, current)
                            parser = _PageLinks(current)
                            parser.feed(raw)
                            links = tuple(parser.links)
                        else:
                            text = raw
            break
        else:
            raise ValueError("too many redirects")
    # The reader returns a whole document, so it cannot replace a PDF page range.
    refused = response.status_code in {403, 429}
    start_page = max(1, int(start_page or 1))
    if refused and start_page > 1:
        raise ValueError(f"The site refused the download (HTTP {response.status_code}). The reader fallback only returns "
                         f"the whole document from its start, so page {start_page} cannot be served; retry with start_page=1.")
    if (not is_pdf and len(text.strip()) < 200) or refused:
        for reader in (_jina_read, _decodo_read):
            try:
                fallback = await reader(current, context)
            except Exception:
                continue  # Keep the original extraction on an unavailable reader.
            if fallback and len(fallback[0].strip()) > len(text.strip()):
                text, title = fallback[0], fallback[1] or title
                if refused:
                    text = "(Reader fallback: the whole document from its start, without page markers.)\n\n" + text
            if len(text.strip()) >= 200:
                break
        if not text and response.is_error:
            response.raise_for_status()
    return FetchedDocument(url, current, title or urlparse(current).hostname or current,
                           text.strip()[:_MAX_DOCUMENT_CHARS], "pdf" if is_pdf else "web", links=links)


async def web_fetch(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(arguments.get("url") or "").strip()
    try:
        document = await fetch_document(url, context=context,
            start_page=max(1, int(arguments.get("start_page") or 1)),
            max_pages=max(1, min(50, int(arguments.get("max_pages") or 8))))
    except UnsafeUrlError as exc:
        return ToolResult(content=f"Refused: {exc}.", summary=f"Không mở {url}: địa chỉ không an toàn", ok=False)
    except PdfTooLargeError:
        return ToolResult(content=f"PDF download exceeds the {_MAX_PDF_BYTES} bytes limit.",
                          summary=f"Không mở {url}: PDF quá lớn", ok=False)
    except ValueError as exc:
        if "cannot be served; retry with start_page=1" in str(exc):
            return ToolResult(content=str(exc), summary=f"Không đọc được trang {max(1, int(arguments.get('start_page') or 1))} của {url}", ok=False)
        return ToolResult(content="Fetch failed (ValueError).", summary=f"Không mở được {url}", ok=False)
    except Exception as exc:
        return ToolResult(content=f"Fetch failed ({type(exc).__name__}).", summary=f"Không mở được {url}", ok=False)
    text = document.text[:_MAX_PAGE_CHARS] or "(no readable text)"
    return ToolResult(content=text, summary=f"Đọc “{document.title[:70]}”",
                      sources=[ToolSource(title=document.title, url=document.final_url, snippet=text[:300])], untrusted=True)


WEB_SEARCH = Tool(
    rate_limit=(10, 60.0),
    group="web",
    name="web_search",
    description=(
        "Search the web for current or external information (news, prices, weather, docs, facts "
        "not in the user's library). Returns titles, URLs and snippets; open promising results "
        "with web_fetch before relying on details."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query; add the language or place when it matters."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
            "recency_days": {"type": "integer", "minimum": 1},
            "lang": {"type": "string"},
            "providers": {"type": "array", "items": {"type": "string", "enum": DEFAULT_PROVIDERS.split(",")}},
        },
        "required": ["query"],
    },
    handler=web_search,
    label="Tìm trên web",
)

WEB_FETCH = Tool(
    rate_limit=(20, 60.0),
    group="web",
    name="web_fetch",
    description="Read a public web page or PDF. PDF pages have page markers; use start_page and max_pages for paging.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."},
                       "start_page": {"type": "integer", "minimum": 1, "default": 1},
                       "max_pages": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8}},
        "required": ["url"],
    },
    handler=web_fetch,
    label="Đọc trang web",
)

# Internal reader budgets are separate from web_fetch and shared across turns.
DECODO_READER = Tool(name="decodo_reader", description="Reader fallback", parameters={},
                     handler=lambda *_: None, rate_limit=(10, 60.0))  # type: ignore[arg-type]
# Internal reader budget is separate from web_fetch and shared across turns.
JINA_READER = Tool(name="jina_reader", description="Reader fallback", parameters={},
                   handler=web_fetch, rate_limit=(15, 60.0))
