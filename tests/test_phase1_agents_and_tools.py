import pytest
from pydantic import ValidationError

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import ExecutionPlan, Subtask, TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.config import get_settings
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.mcp_adapter import create_mcp_server
from src.orchestrator.tools.registry import ToolRegistry


def test_settings_load_with_defaults() -> None:
    settings = get_settings()
    assert settings.openai_api_key
    assert settings.anthropic_api_key
    assert settings.orchestrator_storage_mode == "in_memory"
    assert settings.orchestrator_live_llm is False


def test_dag_topological_order_and_cycle_rejection() -> None:
    plan = ExecutionPlan(
        goal_summary="Order dependent steps",
        subtasks=[
            Subtask(
                step_id="s2",
                title="Analyze data",
                specialist="analyst",
                instruction="Summarize spend",
                depends_on=["s1"],
            ),
            Subtask(
                step_id="s1",
                title="Gather benchmarks",
                specialist="researcher",
                instruction="Search cloud benchmarks",
                depends_on=[],
            ),
        ],
    )
    assert [t.step_id for t in plan.ordered_subtasks()] == ["s1", "s2"]

    with pytest.raises(ValidationError):
        ExecutionPlan(
            goal_summary="Invalid cycle",
            subtasks=[
                Subtask(
                    step_id="a",
                    title="Step A",
                    specialist="researcher",
                    instruction="Do A",
                    depends_on=["b"],
                ),
                Subtask(
                    step_id="b",
                    title="Step B",
                    specialist="analyst",
                    instruction="Do B",
                    depends_on=["a"],
                ),
            ],
        )


def test_tool_registry_sandboxing_and_mcp_export() -> None:
    registry = register_builtin_tools(ToolRegistry())
    mcp_tools = registry.export_mcp_tools()
    assert len(mcp_tools) == 4
    assert create_mcp_server(registry) is not None

    traversal = registry.execute("file_read", {"relative_path": "../.env"})
    assert traversal.success is False
    assert "PermissionError" in (traversal.error or "")
    assert "escapes sandbox" in (traversal.error or "").lower()

    destructive_sql = registry.execute("db_query", {"sql": "DROP TABLE cloud_spend"})
    assert destructive_sql.success is False
    assert destructive_sql.error is not None
    assert any(
        kw in destructive_sql.error.lower()
        for kw in ("select", "read-only", "forbidden", "denied", "sandbox")
    )

    missing = registry.execute("nonexistent_tool", {})
    assert missing.success is False
    assert "ToolNotFoundError" in (missing.error or "")


def test_graph_executes_all_steps_and_reviewer_rejection_loop() -> None:
    graph = build_graph(planner=TaskPlanner(mock=True))
    result = graph.invoke(create_initial_state("t-p1-ok", "Audit Q3 cloud spend"))
    assert result["review_status"] == "approved"
    assert len(result["subtask_results"]) == 3

    calls = {"count": 0}

    def reject_once_policy(output: str, state: dict) -> tuple[bool, str]:
        calls["count"] += 1
        if calls["count"] == 1:
            return False, "Add concrete GPU savings breakdown."
        return True, "Revision accepted."

    rev_graph = build_graph(
        planner=TaskPlanner(mock=True),
        review_policy=reject_once_policy,
    )
    rev_result = rev_graph.invoke(
        create_initial_state("t-p1-rev", "Audit Q3 cloud spend")
    )
    assert rev_result["review_status"] == "approved"
    assert rev_result["review_cycles"] == 1
    assert "Add concrete GPU savings breakdown." in rev_result["final_output"]