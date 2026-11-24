from typing import Any

from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState


def supervisor_node(
    state: AgentState,
    planner: TaskPlanner | None = None,
) -> dict[str, Any]:
    """Decompose the user request into a validated DAG plan and select the next specialist."""
    active_planner = planner or TaskPlanner()
    existing_plan = state["plan"]

    if not existing_plan:
        execution_plan = active_planner.create_plan(state["user_request"])
        ordered = execution_plan.ordered_subtasks()
        serialized_plan = [task.model_dump() for task in ordered]
        trace_entry = (
            f"supervisor: generated {len(serialized_plan)}-step plan "
            f"({execution_plan.goal_summary})"
        )
    else:
        serialized_plan = existing_plan
        trace_entry = "supervisor: routing next step from existing plan"

    step_idx = state["current_step_index"]
    next_specialist = (
        serialized_plan[step_idx]["specialist"]
        if step_idx < len(serialized_plan)
        else ""
    )

    return {
        "plan": serialized_plan,
        "assigned_specialist": next_specialist,
        "trace_log": [trace_entry],
    }


def specialist_node(state: AgentState) -> dict[str, Any]:
    """Execute the current subtask from the plan.

    Tool execution is wired in Step 1.4.
    """
    step_idx = state["current_step_index"]
    current_task = state["plan"][step_idx]
    specialist = current_task["specialist"]

    result = {
        "step_id": current_task["step_id"],
        "step_index": step_idx,
        "specialist": specialist,
        "title": current_task["title"],
        "output": f"[{specialist}] Completed '{current_task['title']}': {current_task['instruction']}",
        "status": "success",
    }
    return {
        "subtask_results": [result],
        "current_step_index": step_idx + 1,
        "trace_log": [f"specialist({specialist}): finished {current_task['step_id']}"],
    }


def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Validate specialist output quality before completing the task."""
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