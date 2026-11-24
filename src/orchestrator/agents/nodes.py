from collections.abc import Callable
from typing import Any

from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState
from src.orchestrator.memory.working_memory import WorkingMemoryStore
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry

# Optional custom review hook: (latest_output, state) -> (is_approved, feedback)
ReviewPolicy = Callable[[str, AgentState], tuple[bool, str]]


def _persist_state(
    working_memory: WorkingMemoryStore | None,
    task_id: str,
    snapshot: dict[str, Any],
) -> None:
    """Best-effort write to working memory when configured."""
    if working_memory is not None:
        working_memory.save_state(task_id, snapshot)


def supervisor_node(
    state: AgentState,
    planner: TaskPlanner | None = None,
    working_memory: WorkingMemoryStore | None = None,
) -> dict[str, Any]:
    """Decompose the user request once into a DAG plan, then route the next step."""
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
        if working_memory is not None:
            working_memory.set_context(state["task_id"], "user_request", state["user_request"])
            working_memory.set_context(state["task_id"], "goal_summary", execution_plan.goal_summary)
            working_memory.set_context(state["task_id"], "plan", serialized_plan)
    else:
        serialized_plan = existing_plan
        step_idx = state["current_step_index"]
        step_label = (
            serialized_plan[step_idx]["step_id"]
            if step_idx < len(serialized_plan)
            else "done"
        )
        trace_entry = f"supervisor: dispatching {step_label}"

    step_idx = state["current_step_index"]
    next_specialist = (
        serialized_plan[step_idx]["specialist"]
        if step_idx < len(serialized_plan)
        else ""
    )

    _persist_state(
        working_memory,
        state["task_id"],
        {
            "phase": "supervisor",
            "current_step_index": step_idx,
            "assigned_specialist": next_specialist,
            "review_status": state["review_status"],
        },
    )

    return {
        "plan": serialized_plan,
        "assigned_specialist": next_specialist,
        "trace_log": [trace_entry],
    }


def _select_tool_call(
    specialist: str,
    current_task: dict[str, Any],
    state: AgentState,
) -> tuple[str, dict[str, Any]]:
    """Map a specialist role and subtask to a concrete ToolRegistry call."""
    if specialist == "researcher":
        return "web_search", {"query": current_task["instruction"], "max_results": 3}

    if specialist in ("analyst", "db_operator"):
        return (
            "db_query",
            {
                "sql": (
                    "SELECT department, service, cost_usd "
                    "FROM cloud_spend ORDER BY cost_usd DESC"
                )
            },
        )

    prior_summaries = [
        f"- [{r['specialist']}] {r['output']}"
        for r in state["subtask_results"]
        if r.get("status") == "success"
    ]
    draft_lines = [
        f"Report for: {state['user_request']}",
        *prior_summaries,
    ]
    if state["review_feedback"]:
        draft_lines.append(f"Revision addressing feedback: {state['review_feedback']}")

    draft_text = "\n".join(draft_lines)
    return (
        "file_write",
        {
            "relative_path": f"{state['task_id']}_report.txt",
            "content": draft_text,
        },
    )


def specialist_node(
    state: AgentState,
    registry: ToolRegistry | None = None,
    working_memory: WorkingMemoryStore | None = None,
) -> dict[str, Any]:
    """Execute the current subtask using the ToolRegistry with structured error tracking."""
    active_registry = registry or register_builtin_tools(ToolRegistry())
    step_idx = state["current_step_index"]

    if step_idx >= len(state["plan"]):
        return {
            "last_error": "",
            "trace_log": ["specialist: no remaining subtasks to execute"],
        }

    current_task = state["plan"][step_idx]
    specialist = current_task["specialist"]
    step_id = current_task["step_id"]

    tool_name, tool_args = _select_tool_call(specialist, current_task, state)
    tool_result = active_registry.execute(tool_name, tool_args)

    if not tool_result.success:
        next_retry = state["retry_count"] + 1
        error_msg = tool_result.error or "Unknown tool error"
        failed_record = {
            "step_id": step_id,
            "step_index": step_idx,
            "specialist": specialist,
            "tool": tool_name,
            "output": None,
            "error": error_msg,
            "status": "error",
            "attempt": next_retry,
        }
        if working_memory is not None:
            working_memory.append_subtask_output(state["task_id"], failed_record)
            working_memory.set_context(state["task_id"], "last_error", error_msg)
        _persist_state(
            working_memory,
            state["task_id"],
            {
                "phase": "specialist_error",
                "current_step_index": step_idx,
                "retry_count": next_retry,
                "last_error": error_msg,
            },
        )
        return {
            "subtask_results": [failed_record],
            "retry_count": next_retry,
            "last_error": error_msg,
            "trace_log": [
                f"specialist({specialist}): {step_id} tool '{tool_name}' failed "
                f"(attempt {next_retry}): {error_msg}"
            ],
        }

    output_payload = (
        tool_args["content"] if tool_name == "file_write" else tool_result.output
    )
    success_record = {
        "step_id": step_id,
        "step_index": step_idx,
        "specialist": specialist,
        "tool": tool_name,
        "title": current_task["title"],
        "output": output_payload,
        "duration_ms": tool_result.duration_ms,
        "status": "success",
    }
    if working_memory is not None:
        working_memory.append_subtask_output(state["task_id"], success_record)
        working_memory.set_context(state["task_id"], f"output_{step_id}", output_payload)
    _persist_state(
        working_memory,
        state["task_id"],
        {
            "phase": "specialist_done",
            "current_step_index": step_idx + 1,
            "retry_count": 0,
            "last_error": "",
        },
    )
    return {
        "subtask_results": [success_record],
        "current_step_index": step_idx + 1,
        "retry_count": 0,
        "last_error": "",
        "trace_log": [
            f"specialist({specialist}): finished {step_id} via '{tool_name}' "
            f"({tool_result.duration_ms}ms)"
        ],
    }


def _default_review_policy(latest_output: str, state: AgentState) -> tuple[bool, str]:
    """Baseline deterministic quality gate for offline validation."""
    if not latest_output or len(latest_output.strip()) < 20:
        return False, "Output is too short; provide a complete summary."
    return True, "All subtasks succeeded and output met quality checks."


def reviewer_node(
    state: AgentState,
    review_policy: ReviewPolicy | None = None,
    working_memory: WorkingMemoryStore | None = None,
) -> dict[str, Any]:
    """Validate final output, fail cleanly on exhausted retries, or trigger revision."""
    if state["last_error"]:
        feedback = f"Halted after retries exhausted: {state['last_error']}"
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "reviewer_failed", "review_status": "failed", "feedback": feedback},
        )
        return {
            "review_status": "failed",
            "review_feedback": feedback,
            "confidence_score": 0.0,
            "final_output": "",
            "trace_log": [f"reviewer: failed task due to unrecovered error ({state['last_error']})"],
        }

    successful_results = [
        r for r in state["subtask_results"] if r.get("status") == "success"
    ]
    latest_output = str(successful_results[-1]["output"]) if successful_results else ""

    policy = review_policy or _default_review_policy
    approved, feedback = policy(latest_output, state)

    if approved:
        if working_memory is not None:
            working_memory.set_context(state["task_id"], "review_feedback", feedback)
            working_memory.set_context(state["task_id"], "final_output", latest_output)
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "completed", "review_status": "approved", "confidence_score": 0.92},
        )
        return {
            "review_status": "approved",
            "review_feedback": feedback,
            "confidence_score": 0.92,
            "final_output": latest_output,
            "trace_log": ["reviewer: approved output"],
        }

    next_cycles = state["review_cycles"] + 1
    if next_cycles > state["max_review_cycles"]:
        fail_feedback = f"Max review cycles exceeded. Last feedback: {feedback}"
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "reviewer_failed", "review_status": "failed", "feedback": fail_feedback},
        )
        return {
            "review_status": "failed",
            "review_cycles": next_cycles,
            "review_feedback": fail_feedback,
            "confidence_score": 0.3,
            "final_output": latest_output,
            "trace_log": [f"reviewer: failed after exceeding max review cycles ({next_cycles})"],
        }

    rewind_index = max(0, len(state["plan"]) - 1)
    if working_memory is not None:
        working_memory.set_context(state["task_id"], "review_feedback", feedback)
    _persist_state(
        working_memory,
        state["task_id"],
        {
            "phase": "reviewer_rejected",
            "review_status": "rejected",
            "review_cycles": next_cycles,
            "feedback": feedback,
        },
    )
    return {
        "review_status": "rejected",
        "review_cycles": next_cycles,
        "review_feedback": feedback,
        "current_step_index": rewind_index,
        "confidence_score": 0.55,
        "trace_log": [f"reviewer: rejected output (cycle {next_cycles}) - {feedback}"],
    }