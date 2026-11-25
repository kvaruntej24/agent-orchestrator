import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.orchestrator.tools.registry import ToolRegistry


# ---------- Input Schemas ----------

class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    max_results: int = Field(default=3, ge=1, le=5)


class FileReadArgs(BaseModel):
    relative_path: str = Field(..., min_length=1, description="Path inside workspace sandbox")


class FileWriteArgs(BaseModel):
    relative_path: str = Field(..., min_length=1, description="Path inside workspace sandbox")
    content: str = Field(..., description="UTF-8 text content to write")


class DBQueryArgs(BaseModel):
    sql: str = Field(..., min_length=6, description="Read-only SELECT SQL query")


# ---------- Sandboxed Implementations ----------

_MOCK_KNOWLEDGE_BASE = [
    {
        "title": "Q3 Cloud Cost Audit Summary",
        "snippet": "Idle GPU nodes in staging accounted for 34% of monthly compute spend.",
        "source": "internal-wiki://cloud-finops-q3",
    },
    {
        "title": "Azure Reserved Instances Guide",
        "snippet": "Committing 1-year reserved capacity cuts baseline compute spend by up to 40%.",
        "source": "docs://azure-cost-optimization",
    },
    {
        "title": "Enterprise Agent Governance Policy",
        "snippet": "All file writes and database mutations require human approval before execution.",
        "source": "policy://ai-governance-v2",
    },
]


def _resolve_sandbox_path(workspace_dir: Path, relative_path: str) -> Path:
    """Resolve a path strictly inside workspace_dir; block path traversal attacks."""
    root = workspace_dir.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(
            f"Access denied: '{relative_path}' escapes sandbox directory '{root}'."
        )
    return target


def _create_mock_db() -> sqlite3.Connection:
    """Build an in-memory SQLite database with sample enterprise records."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE cloud_spend (
            id INTEGER PRIMARY KEY,
            department TEXT NOT NULL,
            service TEXT NOT NULL,
            quarter TEXT NOT NULL,
            cost_usd REAL NOT NULL
        );
        INSERT INTO cloud_spend (department, service, quarter, cost_usd) VALUES
            ('Engineering', 'GPU Compute', 'Q3', 18400.00),
            ('Engineering', 'Managed Postgres', 'Q3', 4200.00),
            ('Analytics', 'Data Warehouse', 'Q3', 9650.00),
            ('Security', 'SIEM Logging', 'Q3', 3100.00);
        """
    )
    return conn


def register_builtin_tools(
    registry: ToolRegistry,
    workspace_dir: Path | str = "workspace",
) -> ToolRegistry:
    """Register all four sandboxed Phase 1 enterprise tools on the given registry."""
    sandbox_root = Path(workspace_dir)
    sandbox_root.mkdir(parents=True, exist_ok=True)
    db_conn = _create_mock_db()

    def web_search_handler(args: BaseModel) -> list[dict[str, str]]:
        assert isinstance(args, WebSearchArgs)
        terms = [t.lower() for t in args.query.split() if len(t) > 2]
        matches = [
            doc
            for doc in _MOCK_KNOWLEDGE_BASE
            if any(
                term in doc["title"].lower() or term in doc["snippet"].lower()
                for term in terms
            )
        ]
        results = matches if matches else _MOCK_KNOWLEDGE_BASE
        return results[: args.max_results]

    def file_read_handler(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, FileReadArgs)
        target = _resolve_sandbox_path(sandbox_root, args.relative_path)
        if not target.exists():
            raise FileNotFoundError(f"File '{args.relative_path}' not found in workspace.")
        return {"path": args.relative_path, "content": target.read_text(encoding="utf-8")}

    def file_write_handler(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, FileWriteArgs)
        target = _resolve_sandbox_path(sandbox_root, args.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        return {"path": args.relative_path, "bytes_written": len(args.content.encode("utf-8"))}

    def db_query_handler(args: BaseModel) -> list[dict[str, Any]]:
        assert isinstance(args, DBQueryArgs)
        cleaned = args.sql.strip()
        if not re.match(r"(?i)^select\b", cleaned):
            raise PermissionError("Only read-only SELECT queries are permitted.")
        if ";" in cleaned.rstrip(";"):
            raise PermissionError("Multi-statement SQL queries are not permitted.")
        cursor = db_conn.execute(cleaned)
        return [dict(row) for row in cursor.fetchall()]

    registry.register(
        name="web_search",
        description="Search internal docs and web knowledge base for relevant facts.",
        args_schema=WebSearchArgs,
        handler=web_search_handler,
        requires_approval=False,
    )
    registry.register(
        name="file_read",
        description="Read a UTF-8 text file from inside the sandboxed workspace directory.",
        args_schema=FileReadArgs,
        handler=file_read_handler,
        requires_approval=False,
    )
    registry.register(
        name="file_write",
        description="Write a UTF-8 text file inside the sandboxed workspace directory.",
        args_schema=FileWriteArgs,
        handler=file_write_handler,
        requires_approval=True,
    )
    registry.register(
        name="db_query",
        description="Run a read-only SELECT SQL query against the enterprise cloud_spend database.",
        args_schema=DBQueryArgs,
        handler=db_query_handler,
        requires_approval=False,
    )
    return registry