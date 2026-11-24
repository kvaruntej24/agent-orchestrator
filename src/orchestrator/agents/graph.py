from typing import Any

from langgraph.graph import END, START, StateGraph

from src.orchestrator.agents.nodes import (
    reviewer_node,
    specialist_node,
    supervisor_node,
)
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState


def build_graph(planner: TaskPlanner | None = None) -> Any:
    """Assemble and compile the agent orchestration graph.

    Accepts an optional TaskPlanner for dependency injection in tests and offline runs.
    Conditional routing and multi-step loops are wired in Step 1.5.
    """
    active_planner = planner or TaskPlanner()

    def _supervisor(state: AgentState) -> dict[str, Any]:
        return supervisor_node(state, planner=active_planner)

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", _supervisor)
    builder.add_node("specialist", specialist_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "specialist")
    builder.add_edge("specialist", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()