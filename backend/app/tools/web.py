"""Multi-provider search and public HTML/PDF reading.

Fetched pages are untrusted. ``web_fetch`` only reaches public http(s) hosts:
loopback, private, link-local and reserved addresses are refused before the
request and after every redirect, so a page cannot steer Aya into the local
backend, the router gateway or the LAN.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.core.eval_mode import eval_mode_enabled, record_network, web_allowed
from app.tools.registry import TOOL_RATE_LIMITER, Tool, ToolContext, ToolResult, ToolSource
from app.tools.search_providers import DEFAULT_PROVIDERS, SearchHit, SearchProvider, build_providers, canonical_url, setting

_MAX_PAGE_BYTES = 3_000_000
_MAX_PAGE_CHARS = 14_000
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AyaAssistant/1.0"


class UnsafeUrlError(ValueError):
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
            merged[key] = {"title": hit.title or key, "url": key, "snippet": hit.snippet,
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
    providers = build_providers(context.settings)
    order = arguments.get("providers")
    if order is None:
        order = str(setting(context.settings, "aya_search_providers", DEFAULT_PROVIDERS)).split(",")
    order = list(dict.fromkeys(str(name).strip().lower() for name in order))
    available = [providers[name] for name in order if name in providers and providers[name].available()]
    primary = [p for p in available if p.name != "duckduckgo"]
    fallback = next((p for p in available if p.name == "duckduckgo"), None)
    hits: list[tuple[int, SearchHit]] = []
    used: list[str] = []
    summaries: list[str] = []
    succeeded = False

    async def run(provider: SearchProvider) -> list[SearchHit] | None:
        try:
            return await provider.search(query, max_results=count, lang=arguments.get("lang"),
                                         recency_days=arguments.get("recency_days"))
        except Exception:  # Vendor details may contain credentials; never expose them.
            return None

    def collect(provider: SearchProvider, result: list[SearchHit] | None) -> None:
        nonlocal succeeded
        result = [hit for hit in result or [] if canonical_url(hit.url) or (hit.provider == "codex" and not hit.url and hit.snippet.strip())]
        if not result:
            return
        succeeded = True
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
    if not succeeded and fallback:
        collect(fallback, await run(fallback))
    results = _merge_hits(hits)[:count]
    content = "\n".join(f"{i}. {item['title']} ({item['url']}): {item['snippet']}" for i, item in enumerate(results, 1)) or "No results."
    data = {"providers": used, "query": query, "results": results}
    if summaries:
        data["codex_summary"] = "\n\n".join(summaries)
        content += "\n\nTóm tắt từ công cụ tìm kiếm (chưa kiểm chứng):\n" + data["codex_summary"]
    return ToolResult(content=content, summary=f"Tìm “{query}” — {len(results)} kết quả ({', '.join(used)})",
                      sources=[ToolSource(title=x["title"], url=x["url"], snippet=x["snippet"]) for x in results],
                      untrusted=True, ok=succeeded, data=data)


def _extract_text(html: str, url: str) -> tuple[str, str]:
    import trafilatura  # type: ignore

    text = trafilatura.extract(html, url=url, include_tables=True, include_links=False, favor_recall=True) or ""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = str(getattr(metadata, "title", "") or "") if metadata else ""
    return text, title


def _extract_pdf(body: bytes, start_page: int, max_pages: int) -> tuple[str, str]:
    import pymupdf

    with pymupdf.open(stream=body, filetype="pdf") as document:
        chunks = []
        remaining = _MAX_PAGE_CHARS
        for index in range(start_page - 1, min(len(document), start_page - 1 + max_pages)):
            chunk = f"--- page {index + 1} ---\n" + document[index].get_text()
            chunks.append(chunk[:remaining])
            remaining -= len(chunks[-1])
            if remaining <= 0:
                break
        text = "\n".join(chunks)[:_MAX_PAGE_CHARS]
        title = str((document.metadata or {}).get("title") or "")
        if not title and start_page <= len(document):
            lines = document[start_page - 1].get_text().strip().splitlines()
            title = lines[0][:200] if lines else ""
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


async def web_fetch(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(arguments.get("url") or "").strip()
    try:
        current = url
        text, title = "", ""
        async with httpx.AsyncClient(timeout=25, follow_redirects=False, headers={"User-Agent": _USER_AGENT}) as client:
            for _hop in range(6):
                await asyncio.to_thread(_check_public_url, current)
                record_network(current, "web_fetch")
                response = await client.get(current)
                if response.is_redirect and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                break
            else:
                raise ValueError("too many redirects")
            if response.status_code not in {403, 429}:
                response.raise_for_status()
                body = response.content[:_MAX_PAGE_BYTES]
                content_type = response.headers.get("content-type", "").lower()
                if "application/pdf" in content_type or body.startswith(b"%PDF"):
                    text, title = await asyncio.to_thread(_extract_pdf, body,
                        max(1, int(arguments.get("start_page") or 1)),
                        max(1, min(50, int(arguments.get("max_pages") or 8))))
                else:
                    raw = body.decode(response.encoding or "utf-8", errors="replace")
                    if "html" in content_type or raw.lstrip().startswith("<"):
                        text, title = await asyncio.to_thread(_extract_text, raw, current)
                    else:
                        text = raw
        if len(text.strip()) < 200 or response.status_code in {403, 429}:
            try:
                fallback = await _jina_read(current, context)
                if fallback and len(fallback[0].strip()) > len(text.strip()):
                    text, title = fallback[0], fallback[1] or title
            except Exception:
                pass  # Keep the original extraction on an unavailable reader.
            if not text and response.is_error:
                response.raise_for_status()
    except UnsafeUrlError as exc:
        return ToolResult(content=f"Refused: {exc}.", summary=f"Không mở {url}: địa chỉ không an toàn", ok=False)
    except Exception as exc:
        return ToolResult(content=f"Fetch failed ({type(exc).__name__}).", summary=f"Không mở được {url}", ok=False)
    text = text.strip()[:_MAX_PAGE_CHARS] or "(no readable text)"
    title = title or urlparse(current).hostname or current
    return ToolResult(content=text, summary=f"Đọc “{title[:70]}”",
                      sources=[ToolSource(title=title, url=current, snippet=text[:300])], untrusted=True)


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

# Internal reader budget is separate from web_fetch and shared across turns.
JINA_READER = Tool(name="jina_reader", description="Reader fallback", parameters={},
                   handler=web_fetch, rate_limit=(15, 60.0))
