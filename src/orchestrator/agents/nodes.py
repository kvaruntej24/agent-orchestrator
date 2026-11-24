from collections.abc import Callable
from typing import Any

from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import AgentState
from src.orchestrator.hitl.approval_queue import ApprovalQueue, ReviewRequest
from src.orchestrator.hitl.triggers import EscalationPolicy
from src.orchestrator.memory.long_term_memory import LongTermMemoryStore
from src.orchestrator.memory.maintenance import compute_importance_score
from src.orchestrator.memory.retrieval import MemoryRetriever
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
    memory_retriever: MemoryRetriever | None = None,
) -> dict[str, Any]:
    """Retrieve relevant past memory, decompose the request once, and route the next step."""
    active_planner = planner or TaskPlanner()
    existing_plan = state["plan"]
    memory_block = state["retrieved_memory_context"]
    trace_entries: list[str] = []

    if not existing_plan:
        if memory_retriever is not None:
            retrieved = memory_retriever.retrieve_for_planning(
                user_request=state["user_request"],
                user_id=state["user_id"],
            )
            memory_block = retrieved.formatted_prompt_block
            trace_entries.append(
                "supervisor: retrieved memory "
                f"(strategies={len(retrieved.past_strategies)}, "
                f"tools={len(retrieved.tool_precedents)}, "
                f"prefs={len(retrieved.user_preferences)})"
            )
            if working_memory is not None and memory_block:
                working_memory.set_context(
                    state["task_id"],
                    "retrieved_memory_context",
                    memory_block,
                )

        execution_plan = active_planner.create_plan(
            state["user_request"],
            memory_context=memory_block,
        )
        ordered = execution_plan.ordered_subtasks()
        serialized_plan = [task.model_dump() for task in ordered]
        trace_entries.append(
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
        trace_entries.append(f"supervisor: dispatching {step_label}")

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
        "retrieved_memory_context": memory_block,
        "trace_log": trace_entries,
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
    escalation_policy: EscalationPolicy | None = None,
    approval_queue: ApprovalQueue | None = None,
) -> dict[str, Any]:
    """Execute the current subtask, pausing for HITL approval when guardrails fire."""
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

    # Pre-execution HITL gate (skips if this step_id was already human-approved)
    if escalation_policy is not None and step_id not in state["approved_steps"]:
        decision = escalation_policy.evaluate(
            state=state,
            pending_tool_name=tool_name,
            pending_tool_args=tool_args,
            registry=active_registry,
        )
        if decision.should_escalate:
            review_req = ReviewRequest.from_escalation(
                task_id=state["task_id"],
                thread_id=state["task_id"],
                step_id=step_id,
                decision=decision,
            )
            if approval_queue is not None:
                approval_queue.enqueue(review_req)
            _persist_state(
                working_memory,
                state["task_id"],
                {
                    "phase": "awaiting_human_approval",
                    "step_id": step_id,
                    "review_id": review_req.review_id,
                    "reason": decision.reason,
                },
            )
            return {
                "approval_status": "pending",
                "pending_approval": review_req.model_dump(),
                "trace_log": [
                    f"specialist({specialist}): paused {step_id} for HITL approval "
                    f"(review_id={review_req.review_id}, reason={decision.reason})"
                ],
            }

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


def human_approval_node(
    state: AgentState,
    working_memory: WorkingMemoryStore | None = None,
) -> dict[str, Any]:
    """Process a human approve/reject decision after LangGraph resumes from checkpoint."""
    pending = state["pending_approval"] or {}
    decision_payload = state["human_decision"] or {}
    step_id = str(pending.get("step_id", "unknown_step"))
    reviewer = str(decision_payload.get("reviewer", "human_reviewer"))
    comment = str(decision_payload.get("reviewer_comment", ""))
    decision_status = str(decision_payload.get("status", state["approval_status"]))

    if decision_status == "approved":
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "human_approved", "step_id": step_id, "reviewer": reviewer},
        )
        return {
            "approval_status": "none",
            "pending_approval": None,
            "approved_steps": [step_id],
            "trace_log": [
                f"human_approval: '{step_id}' approved by {reviewer}"
                + (f" ({comment})" if comment else "")
            ],
        }

    rejection_msg = (
        f"Human reviewer '{reviewer}' rejected '{step_id}'"
        + (f": {comment}" if comment else "")
    )
    _persist_state(
        working_memory,
        state["task_id"],
        {"phase": "human_rejected", "step_id": step_id, "reviewer": reviewer},
    )
    return {
        "approval_status": "none",
        "pending_approval": None,
        "last_error": rejection_msg,
        "trace_log": [f"human_approval: {rejection_msg}"],
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
    long_term_memory: LongTermMemoryStore | None = None,
) -> dict[str, Any]:
    """Validate final output, trigger revision if needed, and archive terminal runs."""
    if state["last_error"]:
        feedback = f"Halted: {state['last_error']}"
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "reviewer_failed", "review_status": "failed", "feedback": feedback},
        )
        updates: dict[str, Any] = {
            "review_status": "failed",
            "review_feedback": feedback,
            "confidence_score": 0.0,
            "final_output": "",
            "trace_log": [f"reviewer: failed task ({state['last_error']})"],
        }
        if long_term_memory is not None:
            merged_state: AgentState = {**state, **updates}  # type: ignore[typeddict-item]
            score = compute_importance_score(merged_state)
            long_term_memory.record_completed_task(merged_state, importance_score=score)
        return updates

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
        updates = {
            "review_status": "approved",
            "review_feedback": feedback,
            "confidence_score": 0.92,
            "final_output": latest_output,
            "trace_log": ["reviewer: approved output"],
        }
        if long_term_memory is not None:
            merged_state = {**state, **updates}  # type: ignore[typeddict-item]
            score = compute_importance_score(merged_state)
            long_term_memory.record_completed_task(merged_state, importance_score=score)
            updates["trace_log"].append("reviewer: archived task into long-term memory")
        return updates

    next_cycles = state["review_cycles"] + 1
    if next_cycles > state["max_review_cycles"]:
        fail_feedback = f"Max review cycles exceeded. Last feedback: {feedback}"
        _persist_state(
            working_memory,
            state["task_id"],
            {"phase": "reviewer_failed", "review_status": "failed", "feedback": fail_feedback},
        )
        updates = {
            "review_status": "failed",
            "review_cycles": next_cycles,
            "review_feedback": fail_feedback,
            "confidence_score": 0.3,
            "final_output": latest_output,
            "trace_log": [f"reviewer: failed after exceeding max review cycles ({next_cycles})"],
        }
        if long_term_memory is not None:
            merged_state = {**state, **updates}  # type: ignore[typeddict-item]
            score = compute_importance_score(merged_state)
            long_term_memory.record_completed_task(merged_state, importance_score=score)
        return updates

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