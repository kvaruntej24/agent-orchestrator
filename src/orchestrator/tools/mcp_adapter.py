import json
from typing import Any

from mcp import types as mcp_types
from mcp.server import Server

from src.orchestrator.tools.registry import ToolRegistry


def create_mcp_server(
    registry: ToolRegistry,
    server_name: str = "agent-orchestrator-tools",
) -> Server:
    """Expose a ToolRegistry as an official Model Context Protocol (MCP) v2 server."""

    async def handle_list_tools(
        ctx: Any,
        params: Any,
    ) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=registry.export_mcp_tools())

    async def handle_call_tool(
        ctx: Any,
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        result = registry.execute(params.name, params.arguments or {})
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(result.model_dump()),
                )
            ],
            is_error=not result.success,
        )

    return Server(
        server_name,
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )