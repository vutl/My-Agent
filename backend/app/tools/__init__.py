"""Aya's first-party tools."""

from __future__ import annotations

from app.tools.code import CODE_TOOLS
from app.tools.files import FILE_TOOLS
from app.tools.library import LibrarySearcher, library_tool
from app.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult, ToolSource
from app.tools.web import WEB_FETCH, WEB_SEARCH


def build_registry(*, library_searcher: LibrarySearcher | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH)
    registry.register(WEB_FETCH)
    for tool in (*FILE_TOOLS, *CODE_TOOLS):
        registry.register(tool)
    if library_searcher is not None:
        registry.register(library_tool(library_searcher))
    return registry


__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "ToolSource", "build_registry"]
