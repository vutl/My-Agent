"""The user's indexed library as a tool for the agent loop (the RAG route
stays the path for plain document questions)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.tools.registry import Tool, ToolContext, ToolResult, ToolSource

LibrarySearcher = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


def library_tool(searcher: LibrarySearcher) -> Tool:
    async def search_library(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = " ".join(str(arguments.get("query") or "").split())[:300]
        if not query:
            return ToolResult(content="Empty query.", summary="Thiếu từ khóa", ok=False)
        passages = await searcher(query, max(1, min(int(arguments.get("top_k") or 6), 10)))
        sources = [
            ToolSource(
                title=f"{item.get('filename') or 'document'}"
                + (f", p.{item['page_number']}" if item.get("page_number") else ""),
                path=str(item.get("filename") or ""),
                snippet=" ".join(str(item.get("text") or item.get("content") or "").split())[:900],
            )
            for item in passages
        ]
        return ToolResult(
            content="No matching passages." if not sources else "",
            summary=f"Tìm trong thư viện “{query}” — {len(sources)} đoạn",
            sources=sources,
        )

    return Tool(
        name="search_library",
        description="Search the user's indexed documents (papers, reports, notes) and return matching passages.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"],
        },
        handler=search_library,
        label="Tìm trong thư viện",
    )
