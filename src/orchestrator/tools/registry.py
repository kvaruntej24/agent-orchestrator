import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp import types as mcp_types
from pydantic import BaseModel, Field, ValidationError


class ToolResult(BaseModel):
    """Standardized response envelope returned by every tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)


@dataclass(frozen=True)
class RegisteredTool:
    """Metadata and callable handler for a single registered tool."""

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    requires_approval: bool = False  # Hook for Phase 3 HITL interceptors


class ToolRegistry:
    """Central registry for validating, executing, and exporting agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        handler: Callable[[BaseModel], Any],
        requires_approval: bool = False,
    ) -> None:
        """Register a new tool; fails fast if the name is already taken."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            handler=handler,
            requires_approval=requires_approval,
        )

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """Return all registered tool names in sorted order."""
        return sorted(self._tools.keys())

    def export_mcp_tools(self) -> list[mcp_types.Tool]:
        """Export every registered tool as an official MCP Tool descriptor."""
        return [
            mcp_types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.args_schema.model_json_schema(),
            )
            for tool in self._tools.values()
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments, invoke the tool, and return a structured ToolResult.

        Never raises unhandled tool exceptions so the orchestration graph can
        inspect errors and decide whether to retry or escalate.
        """
        start = time.perf_counter()
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool '{tool_name}'. Available: {self.list_names()}",
                error_type="ToolNotFoundError",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        try:
            validated_args = tool.args_schema.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Invalid arguments for '{tool_name}': {exc.errors()}",
                error_type="ValidationError",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        try:
            output = tool.handler(validated_args)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        except PermissionError as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
                error_type="SandboxViolationError",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )