import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp import types as mcp_types
from pydantic import BaseModel, Field, ValidationError
from opentelemetry.trace import Status, StatusCode

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

    def __init__(self, tracer: Any | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._tracer = tracer

    def set_tracer(self, tracer: Any | None) -> None:
        """Attach or detach an ExecutionTracer for automatic tool span capture."""
        self._tracer = tracer

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

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate arguments, invoke the tool safely, and record OpenTelemetry spans."""
        raw_args = arguments or {}

        if self._tracer is None:
            return self._execute_core(name, raw_args)

        with self._tracer.start_tool_span(name, list(raw_args.keys())) as span:
            result = self._execute_core(name, raw_args)
            span.set_attribute("tool.success", result.success)
            span.set_attribute("tool.duration_ms", result.duration_ms)
            if result.success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_attribute("tool.error", result.error or "Unknown tool error")
                span.set_status(
                    Status(StatusCode.ERROR, result.error or "Tool execution failed")
                )
            return result

    def _execute_core(self, name: str, raw_args: dict[str, Any]) -> ToolResult:
        """Core validation, execution, timing, and error-envelope logic."""
        start = time.perf_counter()

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"ToolNotFoundError: '{name}' is not registered.",
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
            )

        try:
            validated_args = tool.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"ArgumentValidationError: {exc.errors()}",
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
            )

        try:
            output = tool.handler(validated_args)
            return ToolResult(
                tool_name=name,
                success=True,
                output=output,
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
            )