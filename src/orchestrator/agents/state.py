import operator
from typing import Annotated, Any, Literal, TypedDict


class AgentState(TypedDict):
    """Shared state passed between all nodes in the orchestration graph."""

    task_id: str
    user_request: str
    plan: list[dict[str, Any]]
    current_step_index: int
    assigned_specialist: str
    subtask_results: Annotated[list[dict[str, Any]], operator.add]
    review_status: Literal["pending", "approved", "rejected"]
    review_feedback: str
    retry_count: int
    confidence_score: float
    final_output: str
    trace_log: Annotated[list[str], operator.add]


def create_initial_state(task_id: str, user_request: str) -> AgentState:
    """Create a valid starting state for a new orchestration run."""
    return {
        "task_id": task_id,
        "user_request": user_request,
        "plan": [],
        "current_step_index": 0,
        "assigned_specialist": "",
        "subtask_results": [],
        "review_status": "pending",
        "review_feedback": "",
        "retry_count": 0,
        "confidence_score": 1.0,
        "final_output": "",
        "trace_log": [],
    }