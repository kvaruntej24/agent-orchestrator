from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.orchestrator.agents.nodes import (
    ReviewPolicy,
    human_approval_node,
    reviewer_node,
    specialist_node,
    supervisor_node,
)
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState
from src.orchestrator.hitl.approval_queue import ApprovalQueue
from src.orchestrator.hitl.triggers import EscalationPolicy
from src.orchestrator.memory.long_term_memory import LongTermMemoryStore
from src.orchestrator.memory.retrieval import MemoryRetriever
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
) -> Literal["human_approval", "specialist", "supervisor", "reviewer"]:
    """Route HITL pauses, tool retries, next-step dispatch, or final review."""
    if state["approval_status"] == "pending":
        return "human_approval"

    if state["last_error"]:
        if state["retry_count"] <= state["max_retries"]:
            return "specialist"
        return "reviewer"

    if state["current_step_index"] < len(state["plan"]):
        return "supervisor"
    return "reviewer"


def route_after_human_approval(
    state: AgentState,
) -> Literal["specialist", "reviewer"]:
    """Resume execution on human approval, or fail cleanly via reviewer on rejection."""
    if state["last_error"]:
        return "reviewer"
    return "specialist"


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
    long_term_memory: LongTermMemoryStore | None = None,
    memory_retriever: MemoryRetriever | None = None,
    escalation_policy: EscalationPolicy | None = None,
    approval_queue: ApprovalQueue | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Assemble the LangGraph state machine with memory and HITL pause/resume checkpointing."""
    active_planner = planner or TaskPlanner()
    active_registry = registry or register_builtin_tools(ToolRegistry())

    # Auto-enable checkpointing whenever HITL escalation is active
    active_checkpointer = checkpointer
    if escalation_policy is not None and active_checkpointer is None:
        active_checkpointer = MemorySaver()

    def _supervisor(state: AgentState) -> dict[str, Any]:
        return supervisor_node(
            state,
            planner=active_planner,
            working_memory=working_memory,
            memory_retriever=memory_retriever,
        )

    def _specialist(state: AgentState) -> dict[str, Any]:
        return specialist_node(
            state,
            registry=active_registry,
            working_memory=working_memory,
            escalation_policy=escalation_policy,
            approval_queue=approval_queue,
        )

    def _human_approval(state: AgentState) -> dict[str, Any]:
        return human_approval_node(state, working_memory=working_memory)

    def _reviewer(state: AgentState) -> dict[str, Any]:
        return reviewer_node(
            state,
            review_policy=review_policy,
            working_memory=working_memory,
            long_term_memory=long_term_memory,
        )

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", _supervisor)
    builder.add_node("specialist", _specialist)
    builder.add_node("human_approval", _human_approval)
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
            "human_approval": "human_approval",
            "specialist": "specialist",
            "supervisor": "supervisor",
            "reviewer": "reviewer",
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {"specialist": "specialist", "reviewer": "reviewer"},
    )
    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"supervisor": "supervisor", END: END},
    )

    if active_checkpointer is not None:
        return builder.compile(
            checkpointer=active_checkpointer,
            interrupt_before=["human_approval"],
        )
    return builder.compile()