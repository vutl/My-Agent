"""Web tools: search (Tavily when configured, else DuckDuckGo) and page fetch.

Fetched pages are untrusted. ``web_fetch`` only reaches public http(s) hosts:
loopback, private, link-local and reserved addresses are refused before the
request and after every redirect, so a page cannot steer Aya into the local
backend, the router gateway or the LAN.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.tools.registry import Tool, ToolContext, ToolResult, ToolSource

_MAX_PAGE_BYTES = 3_000_000
_MAX_PAGE_CHARS = 14_000
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AyaAssistant/1.0"


class UnsafeUrlError(ValueError):
    pass


def _check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("only http(s) URLs with a host are allowed")
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


async def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    return [
        {"title": str(item.get("title") or ""), "url": str(item.get("url") or ""),
         "snippet": str(item.get("content") or "")[:600]}
        for item in results
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS  # type: ignore

    with DDGS() as client:
        results = list(client.text(query, max_results=max_results))
    return [
        {"title": str(item.get("title") or ""), "url": str(item.get("href") or item.get("url") or ""),
         "snippet": str(item.get("body") or "")[:600]}
        for item in results
    ]


async def web_search(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(arguments.get("query") or "").split())[:300]
    if not query:
        return ToolResult(content="Empty query.", summary="Thiếu từ khóa tìm kiếm", ok=False)
    max_results = max(1, min(int(arguments.get("max_results") or 5), 8))
    api_key = os.getenv("TAVILY_API_KEY") or str(getattr(context.settings, "tavily_api_key", "") or "")
    provider = "tavily" if api_key else "duckduckgo"
    try:
        results = (
            await _search_tavily(query, max_results, api_key) if api_key
            else await asyncio.to_thread(_search_duckduckgo, query, max_results)
        )
    except Exception as exc:  # noqa: BLE001 - the model reads the failure and adapts
        return ToolResult(content=f"Search failed ({type(exc).__name__}).", summary=f"Tìm “{query}” thất bại", ok=False)
    results = [item for item in results if item["url"].startswith(("http://", "https://"))]
    sources = [ToolSource(title=item["title"] or item["url"], url=item["url"], snippet=item["snippet"]) for item in results]
    content = "\n".join(f"- {item['title']} ({item['url']}): {item['snippet']}" for item in results) or "No results."
    return ToolResult(
        content=content, summary=f"Tìm “{query}” — {len(results)} kết quả ({provider})",
        sources=sources, untrusted=True, data={"provider": provider, "query": query},
    )


def _extract_text(html: str, url: str) -> tuple[str, str]:
    import trafilatura  # type: ignore

    text = trafilatura.extract(html, url=url, include_tables=True, include_links=False, favor_recall=True) or ""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = str(getattr(metadata, "title", "") or "") if metadata else ""
    return text, title


async def web_fetch(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(arguments.get("url") or "").strip()
    try:
        current = url
        async with httpx.AsyncClient(timeout=25, follow_redirects=False, headers={"User-Agent": _USER_AGENT}) as client:
            for _hop in range(6):
                await asyncio.to_thread(_check_public_url, current)
                response = await client.get(current)
                if response.is_redirect and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                break
            response.raise_for_status()
            body = response.content[:_MAX_PAGE_BYTES]
    except UnsafeUrlError as exc:
        return ToolResult(content=f"Refused: {exc}.", summary=f"Không mở {url}: địa chỉ không an toàn", ok=False)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(content=f"Fetch failed ({type(exc).__name__}).", summary=f"Không mở được {url}", ok=False)
    content_type = response.headers.get("content-type", "")
    raw = body.decode(response.encoding or "utf-8", errors="replace")
    if "html" in content_type or raw.lstrip().startswith("<"):
        text, title = await asyncio.to_thread(_extract_text, raw, current)
    else:
        text, title = raw, ""
    text = text.strip()[:_MAX_PAGE_CHARS] or "(no readable text)"
    title = title or urlparse(current).hostname or current
    return ToolResult(
        content=text, summary=f"Đọc “{title[:70]}”",
        sources=[ToolSource(title=title, url=current, snippet=text[:300])], untrusted=True,
    )


WEB_SEARCH = Tool(
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
        },
        "required": ["query"],
    },
    handler=web_search,
    label="Tìm trên web",
)

WEB_FETCH = Tool(
    group="web",
    name="web_fetch",
    description="Open a public web page (http/https) and return its main text.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
        "required": ["url"],
    },
    handler=web_fetch,
    label="Đọc trang web",
)
