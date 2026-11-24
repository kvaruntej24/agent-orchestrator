from typing import Any

from src.orchestrator.agents.state import AgentState


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Decompose the user request and pick the next specialist.

    Stubbed in Step 1.2; real LLM task decomposition is wired in Step 1.3.
    """
    step_preview = {
        "step_id": "step_1",
        "specialist": "researcher",
        "instruction": f"Analyze request: {state['user_request']}",
    }
    return {
        "plan": [step_preview],
        "assigned_specialist": "researcher",
        "trace_log": ["supervisor: created placeholder plan"],
    }


def specialist_node(state: AgentState) -> dict[str, Any]:
    """Execute the assigned subtask using tools.

    Stubbed in Step 1.2; tool execution is wired in Step 1.4.
    """
    specialist = state["assigned_specialist"] or "generalist"
    result = {
        "step_index": state["current_step_index"],
        "specialist": specialist,
        "output": f"[{specialist}] Completed placeholder work for: {state['user_request']}",
        "status": "success",
    }
    return {
        "subtask_results": [result],
        "current_step_index": state["current_step_index"] + 1,
        "trace_log": [f"specialist({specialist}): finished subtask"],
    }


def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Validate specialist output quality before completing the task.

    Stubbed in Step 1.2; rejection cycles and HITL triggers come later.
    """
    latest_output = (
        state["subtask_results"][-1]["output"]
        if state["subtask_results"]
        else "No specialist output produced."
    )
    return {
        "review_status": "approved",
        "review_feedback": "Stub review passed.",
        "final_output": latest_output,
        "trace_log": ["reviewer: approved output"],
    }