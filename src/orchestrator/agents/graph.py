from typing import Any

from langgraph.graph import END, START, StateGraph

from src.orchestrator.agents.nodes import (
    reviewer_node,
    specialist_node,
    supervisor_node,
)
from src.orchestrator.agents.state import AgentState


def build_graph() -> Any:
    """Assemble and compile the baseline agent hierarchy graph.

    Uses linear edges for now. Conditional routing, retry loops,
    and reviewer rejection cycles are added in Step 1.5.
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("specialist", specialist_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "specialist")
    builder.add_edge("specialist", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()