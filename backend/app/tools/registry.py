"""First-party tools Aya can call in the agent loop.

A tool declares its JSON schema (sent to the model as a function), a
permission level and an async handler. ``auto`` tools run directly; ``ask``
tools wait for the user's approval in the UI. Tool output that comes from the
outside world (web pages, files, messages) is marked untrusted: the loop
wraps it so the model reads it as data, never as instructions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Permission = Literal["auto", "ask"]


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    conversation_id: str
    workspace: Path
    settings: Any = None


@dataclass
class ToolSource:
    """A citable source a tool produced (a web page, a file)."""

    title: str
    url: str = ""
    path: str = ""
    snippet: str = ""


@dataclass
class ToolResult:
    content: str  # what the model reads
    summary: str  # one line for the UI card
    sources: list[ToolSource] = field(default_factory=list)
    untrusted: bool = False
    ok: bool = True
    data: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    permission: Permission = "auto"
    label: str = ""  # short Vietnamese label for the UI card
    # Changes something outside Aya (writes, deletes, sends, runs code). After
    # untrusted content entered the turn such calls always need approval.
    side_effects: bool = False
    # Decides per call whether an ``auto`` tool still needs approval
    # (e.g. a file write outside the workspace).
    needs_approval: Callable[[dict[str, Any], ToolContext], bool] | None = None
    group: str = "other"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def requires_approval(self, arguments: dict[str, Any], context: ToolContext) -> bool:
        if self.permission == "ask":
            return True
        return bool(self.needs_approval and self.needs_approval(arguments, context))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names if names is not None else self.names()
        return [self._tools[name].schema() for name in selected if name in self._tools]
