"""Search adapters, durable soft quotas and shared cooldowns.

Secrets are sent in headers only. Provider failures are handled by the caller;
never include a vendor response body or exception message in tool output.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import threading
import time
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from app.core.eval_mode import record_network
from app.db.sqlite import connect

DEFAULT_PROVIDERS = "codex,tavily,serper,decodo,exa,brave,jina,searxng,duckduckgo"
CAPS = {"tavily": 950, "serper": 2400, "decodo": 9500, "exa": 900, "brave": 900, "jina": 900}
USAGE_SCHEMA = """CREATE TABLE IF NOT EXISTS search_usage (
    provider TEXT NOT NULL, period TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, period))"""
_COOLDOWNS: dict[tuple[str, str], float] = {}
_COOLDOWN_LOCK = threading.Lock()


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    provider: str
    published: str | None = None


class SearchProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def search(self, query: str, *, max_results: int, lang: str | None,
                     recency_days: int | None, model_call=None) -> list[SearchHit]: ...


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return ""
        host = parts.hostname.lower().removeprefix("www.")
        if ":" in host:
            host = f"[{host}]"
        if parts.port and parts.port != (443 if parts.scheme.lower() == "https" else 80):
            host += f":{parts.port}"
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
        return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), urlencode(sorted(query)), ""))
    except ValueError:
        return ""


def _without_tracking(url: str) -> str:
    """Drop utm_* parameters (the hosted search appends utm_source=openai);
    the rest of the URL is kept exactly."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "utm_" not in parts.query:
        return url
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def setting(settings: Any, name: str, default: Any = "") -> Any:
    value = os.getenv(name.upper())
    if value is None or value == "":
        value = getattr(settings, name, default)
    return default if value is None or value == "" else value


class Provider:
    def __init__(self, name: str, settings: Any):
        self.name = name
        self.settings = settings
        self.key = str(setting(settings, "decodo_auth_token" if name == "decodo" else f"{name}_api_key"))
        self.db_path = Path(settings.sqlite_db_path)
        self.cap = int(setting(settings, f"{name}_search_cap", CAPS[name])) if name in CAPS else None
        self.base = str(setting(settings, "openai_api_base")) if name == "codex" else str(setting(settings, "searxng_url"))
        if name == "codex":
            self.key = str(setting(settings, "openai_api_key"))
        self.cooldown_key = (str(self.db_path), name)

    @property
    def period(self) -> str:
        return "total" if self.name == "serper" else datetime.now(timezone.utc).strftime("%Y-%m")

    @property
    def configured(self) -> bool:
        if self.name == "duckduckgo":
            return True
        if self.name == "searxng":
            return bool(self.base)
        if self.name == "codex":
            return bool(self.base and self.key)
        return bool(self.key)

    def used(self) -> int:
        if not self.db_path.exists():
            return 0
        with connect(self.db_path) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='search_usage'").fetchone()
            row = db.execute("SELECT count FROM search_usage WHERE provider=? AND period=?",
                             (self.name, self.period)).fetchone() if exists else None
        return int(row[0]) if row else 0

    def available(self) -> bool:
        return (self.configured and _COOLDOWNS.get(self.cooldown_key, 0) <= time.time()
                and (self.cap is None or self.used() < self.cap))

    def _reserve(self) -> bool:
        if not self.available():
            return False
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        period = self.period
        with connect(self.db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(USAGE_SCHEMA)
            db.execute("INSERT OR IGNORE INTO search_usage(provider,period,count) VALUES(?,?,0)",
                       (self.name, period))
            return bool(db.execute("UPDATE search_usage SET count=count+1 WHERE provider=? AND period=? AND (? IS NULL OR count<?)",
                                   (self.name, period, self.cap, self.cap)).rowcount)

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "configured": self.configured, "available": self.available(),
                "used_this_period": self.used(), "cap": self.cap,
                "cooldown_until": _COOLDOWNS.get(self.cooldown_key) if _COOLDOWNS.get(self.cooldown_key, 0) > time.time() else None}

    async def search(self, query: str, *, max_results: int, lang: str | None = None,
                     recency_days: int | None = None, model_call=None) -> list[SearchHit]:
        if not await asyncio.to_thread(self._reserve):
            return []
        try:
            if self.name == "duckduckgo":
                record_network("https://duckduckgo.com/", "web_search")
                return await asyncio.to_thread(self._ddg, query, max_results, lang, recency_days)
            async with httpx.AsyncClient(timeout=45 if self.name in {"codex", "decodo"} else 15) as client:
                if self.name == "codex" and model_call is not None:
                    # Research owns scheduling/accounting/retries before provider
                    # cooldowns or the search fan-out can swallow quota failures.
                    return await model_call(lambda: self._http_search(client, query, max_results, lang,
                        recency_days, model=self.settings.research_worker_model))
                return await self._http_search(client, query, max_results, lang, recency_days)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 402, 429}:
                with _COOLDOWN_LOCK:
                    _COOLDOWNS[self.cooldown_key] = time.time() + 600
            raise

    def _ddg(self, query: str, count: int, lang: str | None, days: int | None) -> list[SearchHit]:
        from ddgs import DDGS

        with DDGS(timeout=15) as client:
            rows = client.text(query, max_results=count, region=lang or "wt-wt",
                               timelimit=("d" if days == 1 else "w" if days and days <= 7 else "m" if days and days <= 31 else "y") if days else None)
            return [self._hit(x.get("title"), x.get("href") or x.get("url"), x.get("body")) for x in rows]

    def _hit(self, title: Any, url: Any, snippet: Any, published: Any = None) -> SearchHit:
        return SearchHit(str(title or ""), str(url or ""), str(snippet or "")[:600], self.name,
                         str(published) if published else None)

    async def _http_search(self, client: httpx.AsyncClient, query: str, count: int,
                           lang: str | None, days: int | None, *, model: str | None = None) -> list[SearchHit]:
        headers: dict[str, str] = {}
        params: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        method = "POST"
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else None
        if self.name == "codex":
            url = f"{self.base.rstrip('/')}/responses"
            headers = {"Authorization": f"Bearer {self.key}"}
            prompt = query + (f"\nSearch language: {lang}." if lang else "") + (f"\nOnly sources published since {since}." if since else "")
            payload = {"model": model or setting(self.settings, "search_model", self.settings.router_model),
                       "input": prompt, "tools": [{"type": "web_search"}]}
        elif self.name == "tavily":
            url = "https://api.tavily.com/search"
            headers = {"Authorization": f"Bearer {self.key}"}
            payload = {"query": query, "max_results": count, "search_depth": "basic",
                       "topic": "news" if days and days <= 7 else "general"}
            if days:
                payload["time_range"] = "day" if days == 1 else "week" if days <= 7 else "month" if days <= 31 else "year"
        elif self.name == "serper":
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": self.key}
            payload = {"q": query, "num": count}
            if lang:
                pieces = lang.lower().replace("_", "-").split("-")
                payload.update(hl=pieces[0], gl=pieces[-1] if len(pieces) > 1 else {"vi": "vn", "en": "us"}.get(pieces[0], pieces[0]))
            if days:
                payload["tbs"] = f"qdr:d{days}"
        elif self.name == "exa":
            url = "https://api.exa.ai/search"
            headers = {"x-api-key": self.key}
            payload = {"query": query, "numResults": count, "type": "auto", "contents": {"highlights": True}}
            if since:
                payload["startPublishedDate"] = since
        elif self.name == "brave":
            method, url = "GET", "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": self.key}
            params = {"q": query, "count": count}
            if lang:
                params["search_lang"] = lang.split("-")[0]
            if days:
                params["freshness"] = "pd" if days == 1 else "pw" if days <= 7 else "pm" if days <= 31 else "py"
        elif self.name == "decodo":
            url = "https://data.decodo.com/v1/scrape"
            headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}
            payload = {"target": "google_search", "query": query, "parse": True}
            if lang:
                payload["google_results_language"] = lang.split("-")[0]
            if days:
                payload["google_tbs"] = "qdr:d" if days == 1 else "qdr:w" if days <= 7 else "qdr:m" if days <= 31 else "qdr:y"
        elif self.name == "jina":
            method, url = "GET", "https://s.jina.ai/"
            # Titles, URLs and snippets only: pages are read with web_fetch.
            headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json",
                       "X-Respond-With": "no-content"}
            params = {"q": query, "num": count}
        else:
            method, url = "GET", f"{self.base.rstrip('/')}/search"
            params = {"format": "json", "q": query, "language": lang or "auto"}
            if days:
                params["time_range"] = "day" if days == 1 else "week" if days <= 7 else "month" if days <= 31 else "year"
        record_network(url, "web_search")
        response = await client.request(method, url, headers=headers, params=params,
                                        **({"json": payload} if method == "POST" else {}))
        response.raise_for_status()
        data = response.json()
        if self.name == "codex":
            return parse_codex(data)
        if self.name == "decodo":
            content = ((data.get("results") or [{}])[0] or {}).get("content") or {}
            organic = (((content.get("results") or {}).get("results") or {}).get("organic") or []) if isinstance(content, dict) else []
            return [self._hit(row.get("title"), row.get("url"), row.get("desc")) for row in organic[:count]]
        rows = (data.get("organic", []) if self.name == "serper" else
                (data.get("web") or {}).get("results", []) if self.name == "brave" else
                data.get("data", []) if self.name == "jina" else data.get("results", []))
        hits = []
        for row in rows[:count]:
            snippet = row.get("content") or row.get("snippet") or row.get("description") or " ".join(row.get("highlights") or [])
            hits.append(self._hit(row.get("title"), row.get("url") or row.get("link"), snippet, row.get("publishedDate") or row.get("published_date") or row.get("date")))
        return hits


def parse_codex(data: dict[str, Any]) -> list[SearchHit]:
    hits: dict[str, SearchHit] = {}
    summaries = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            text = str(content.get("text") or "")
            if text:
                summaries.append(text)
            for citation in content.get("annotations") or []:
                if citation.get("type") != "url_citation":
                    continue
                url = _without_tracking(str(citation.get("url") or ""))
                key = canonical_url(url)
                if not key or key in hits:
                    continue
                start = max(0, min(int(citation.get("start_index") or 0), len(text)))
                end = max(start, min(int(citation.get("end_index") or start), len(text)))
                # Some gateways annotate the citation marker rather than the
                # claim. Include the surrounding sentence in that case.
                anchor = start
                while anchor < len(text) and text[anchor].isspace():
                    anchor += 1
                marker = text[start:end].lstrip().startswith(("[", ""))
                if marker:
                    anchor = max(0, len(text[:start].rstrip(" \n.!?")) - 1)
                boundary = text.rfind(". ", 0, anchor)
                left = max(boundary + 2 if boundary >= 0 else 0, text.rfind("\n", 0, anchor) + 1)
                right = text.find(". ", anchor if marker else max(anchor, end - 1))
                span = text[left:right + 1 if right >= 0 else len(text)]
                hits[key] = SearchHit(str(citation.get("title") or url), url, span[:600], "codex")
    result = list(hits.values())
    if summaries:
        result.append(SearchHit("Search summary", "", "\n\n".join(summaries), "codex"))
    return result


def build_providers(settings: Any) -> dict[str, Provider]:
    return {name: Provider(name, settings) for name in DEFAULT_PROVIDERS.split(",")}
