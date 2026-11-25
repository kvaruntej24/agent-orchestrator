from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Status, StatusCode
from src.orchestrator.observability.tracing import ExecutionTracer
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
from src.orchestrator.observability.cost_tracker import (
    CostTracker,
    estimate_tokens,
)

def route_after_supervisor(state: AgentState) -> Literal["specialist", "reviewer"]:
    """Exit immediately on budget breach; otherwise dispatch remaining subtasks."""
    if state["budget_exceeded"]:
        return "reviewer"
    if state["current_step_index"] < len(state["plan"]):
        return "specialist"
    return "reviewer"


def route_after_specialist(
    state: AgentState,
) -> Literal["human_approval", "specialist", "supervisor", "reviewer"]:
    """Route budget halts, HITL pauses, tool retries, next-step dispatch, or final review."""
    if state["budget_exceeded"]:
        return "reviewer"

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
) -> Literal["supervisor", "specialist", "reviewer"]:
    """Route post-HITL resolution according to the applied approval tier."""
    if state["last_error"]:
        return "reviewer"

    decision_payload = state["human_decision"] or {}
    tier = str(decision_payload.get("tier", ""))
    decision = str(decision_payload.get("decision") or decision_payload.get("status") or "")

    if tier == "human_takeover" or decision == "taken_over":
        return "reviewer"
    if state["current_step_index"] >= len(state["plan"]):
        return "reviewer"
    if tier == "modify_plan" or decision == "modified":
        return "supervisor"
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
    tracer: ExecutionTracer | None = None,
    cost_tracker: CostTracker | None = None,
) -> Any:
    """Assemble the LangGraph state machine with memory, HITL, tracing, and cost governance."""
    active_planner = planner or TaskPlanner()
    active_registry = registry or register_builtin_tools(ToolRegistry())
    if tracer is not None:
        active_registry.set_tracer(tracer)

    # Auto-enable checkpointing whenever HITL escalation is active
    active_checkpointer = checkpointer
    if escalation_policy is not None and active_checkpointer is None:
        active_checkpointer = MemorySaver()

    def _execute_node_with_budget(
        node_name: str,
        state: AgentState,
        fn: Any,
    ) -> dict[str, Any]:
        task_id = state["task_id"]

        # Pre-execution hard circuit breaker (never blocks reviewer from finalizing failure)
        if cost_tracker is not None and node_name in ("supervisor", "specialist"):
            within_budget, breach_reason = cost_tracker.check_budget(task_id)
            if not within_budget:
                summary = cost_tracker.get_task_summary(task_id)
                return {
                    "budget_exceeded": True,
                    "last_error": breach_reason,
                    "total_tokens": summary.total_tokens,
                    "total_cost_usd": summary.total_cost_usd,
                    "trace_log": [f"{node_name}: halted by {breach_reason}"],
                }

        updates = dict(fn())

        if cost_tracker is not None and node_name in (
            "supervisor",
            "specialist",
            "reviewer",
        ):
            prompt_tokens = estimate_tokens(
                f"{state['user_request']} {state['retrieved_memory_context']} "
                f"{state['review_feedback']}"
            )
            completion_tokens = estimate_tokens(updates)
            cost_tracker.record_usage(
                task_id=task_id,
                node_name=node_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            summary = cost_tracker.get_task_summary(task_id)
            updates["total_tokens"] = summary.total_tokens
            updates["total_cost_usd"] = summary.total_cost_usd

            if summary.circuit_breaker_tripped and node_name != "reviewer":
                updates["budget_exceeded"] = True
                updates["last_error"] = summary.breach_reason
                existing_traces = list(updates.get("trace_log", []))
                existing_traces.append(
                    f"{node_name}: tripped {summary.breach_reason}"
                )
                updates["trace_log"] = existing_traces

        return updates

    def _run_traced_node(
        node_name: str,
        state: AgentState,
        fn: Any,
    ) -> dict[str, Any]:
        if tracer is None:
            return _execute_node_with_budget(node_name, state, fn)
        with tracer.start_node_span(node_name, dict(state)) as span:
            updates = _execute_node_with_budget(node_name, state, fn)
            if updates.get("last_error"):
                span.set_attribute("node.error", str(updates["last_error"]))
                span.set_status(Status(StatusCode.ERROR, str(updates["last_error"])))
            else:
                span.set_status(Status(StatusCode.OK))
            return updates

    def _supervisor(state: AgentState) -> dict[str, Any]:
        return _run_traced_node(
            "supervisor",
            state,
            lambda: supervisor_node(
                state,
                planner=active_planner,
                working_memory=working_memory,
                memory_retriever=memory_retriever,
            ),
        )

    def _specialist(state: AgentState) -> dict[str, Any]:
        return _run_traced_node(
            "specialist",
            state,
            lambda: specialist_node(
                state,
                registry=active_registry,
                working_memory=working_memory,
                escalation_policy=escalation_policy,
                approval_queue=approval_queue,
            ),
        )

    def _human_approval(state: AgentState) -> dict[str, Any]:
        return _run_traced_node(
            "human_approval",
            state,
            lambda: human_approval_node(state, working_memory=working_memory),
        )

    def _reviewer(state: AgentState) -> dict[str, Any]:
        return _run_traced_node(
            "reviewer",
            state,
            lambda: reviewer_node(
                state,
                review_policy=review_policy,
                working_memory=working_memory,
                long_term_memory=long_term_memory,
            ),
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
        {
            "supervisor": "supervisor",
            "specialist": "specialist",
            "reviewer": "reviewer",
        },
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