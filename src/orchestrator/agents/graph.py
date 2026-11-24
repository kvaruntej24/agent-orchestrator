from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.orchestrator.agents.nodes import (
    ReviewPolicy,
    reviewer_node,
    specialist_node,
    supervisor_node,
)
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState
from src.orchestrator.memory.working_memory import WorkingMemoryStore
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry


def route_after_supervisor(state: AgentState) -> Literal["specialist", "reviewer"]:
    """Send remaining subtasks to specialist; otherwise proceed to reviewer."""
    if state["current_step_index"] < len(state["plan"]):
        return "specialist"
    return "reviewer"


def route_after_specialist(
    state: AgentState,
) -> Literal["specialist", "supervisor", "reviewer"]:
    """Handle tool retries, next-step dispatch, or escalation to reviewer."""
    if state["last_error"]:
        if state["retry_count"] <= state["max_retries"]:
            return "specialist"
        return "reviewer"

    if state["current_step_index"] < len(state["plan"]):
        return "supervisor"
    return "reviewer"


def route_after_reviewer(state: AgentState) -> Literal["supervisor", "__end__"]:
    """Loop rejected work back to supervisor; finish on approved or failed."""
    if state["review_status"] == "rejected":
        return "supervisor"
    return END


def build_graph(
    planner: TaskPlanner | None = None,
    registry: ToolRegistry | None = None,
    review_policy: ReviewPolicy | None = None,
    working_memory: WorkingMemoryStore | None = None,
) -> Any:
    """Assemble and compile the LangGraph state machine with optional Redis memory."""
    active_planner = planner or TaskPlanner()
    active_registry = registry or register_builtin_tools(ToolRegistry())

    def _supervisor(state: AgentState) -> dict[str, Any]:
        return supervisor_node(state, planner=active_planner, working_memory=working_memory)

    def _specialist(state: AgentState) -> dict[str, Any]:
        return specialist_node(state, registry=active_registry, working_memory=working_memory)

    def _reviewer(state: AgentState) -> dict[str, Any]:
        return reviewer_node(state, review_policy=review_policy, working_memory=working_memory)

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", _supervisor)
    builder.add_node("specialist", _specialist)
    builder.add_node("reviewer", _reviewer)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"specialist": "specialist", "reviewer": "reviewer"},
    )
    builder.add_conditional_edges(
        "specialist",
        route_after_specialist,
        {
            "specialist": "specialist",
            "supervisor": "supervisor",
            "reviewer": "reviewer",
        },
    )
    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"supervisor": "supervisor", END: END},
    )

    return builder.compile()