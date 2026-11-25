from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.orchestrator.agents.planner import ExecutionPlan, Subtask
from src.orchestrator.agents.state import AgentState
from src.orchestrator.hitl.approval_queue import ApprovalQueue
from src.orchestrator.hitl.triggers import ApprovalTier

TierDecisionStatus = Literal[
    "acknowledged",
    "approved",
    "rejected",
    "modified",
    "taken_over",
]


class HumanResolutionPayload(BaseModel):
    """Typed payload submitted by a human reviewer across all four HITL tiers."""

    tier: ApprovalTier
    decision: TierDecisionStatus
    reviewer: str = Field(..., min_length=1)
    comment: str = ""
    modified_tool_args: dict[str, Any] | None = None
    modified_plan: list[dict[str, Any]] | None = None
    takeover_output: str | None = None

    @model_validator(mode="after")
    def validate_tier_requirements(self) -> "HumanResolutionPayload":
        """Enforce strict payload contracts for each approval tier."""
        if self.tier == "notify":
            if self.decision != "acknowledged":
                raise ValueError("Notify tier resolutions must use decision='acknowledged'.")
            return self

        if self.tier == "approve_action":
            if self.decision not in ("approved", "rejected"):
                raise ValueError(
                    "approve_action tier requires decision='approved' or 'rejected'."
                )
            return self

        if self.tier == "modify_plan":
            if self.decision == "rejected":
                return self
            if self.decision != "modified":
                raise ValueError(
                    "modify_plan tier requires decision='modified' or 'rejected'."
                )
            if not self.modified_plan:
                raise ValueError(
                    "modify_plan tier with decision='modified' requires a non-empty modified_plan."
                )
            # Validate human-edited plan through ExecutionPlan so no cycles or bad roles enter
            validated = ExecutionPlan(
                goal_summary="Human-modified execution plan",
                subtasks=[Subtask.model_validate(step) for step in self.modified_plan],
            )
            self.modified_plan = [t.model_dump() for t in validated.ordered_subtasks()]
            return self

        if self.tier == "human_takeover":
            if self.decision != "taken_over":
                raise ValueError("human_takeover tier requires decision='taken_over'.")
            if not self.takeover_output or not self.takeover_output.strip():
                raise ValueError(
                    "human_takeover tier requires a non-empty takeover_output."
                )
            return self

        return self


def apply_human_resolution(
    state: AgentState,
    decision_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compute typed AgentState updates for any of the four HITL approval tiers.

    Supports both full HumanResolutionPayload dicts (Step 3.3+) and legacy
    Step 3.2 ReviewRequest dicts for backward compatibility.
    """
    pending = state["pending_approval"] or {}
    step_id = str(pending.get("step_id", "unknown_step"))
    reviewer = str(decision_payload.get("reviewer", "human_reviewer"))
    comment = str(
        decision_payload.get("comment")
        or decision_payload.get("reviewer_comment")
        or ""
    )
    tier = str(
        decision_payload.get("tier")
        or pending.get("recommended_tier")
        or "approve_action"
    )
    decision = str(
        decision_payload.get("decision")
        or decision_payload.get("status")
        or state["approval_status"]
    )
    suffix = f" ({comment})" if comment else ""

    # Tier 1: Notify (non-blocking audit acknowledgement)
    if tier == "notify" or decision == "acknowledged":
        return {
            "approval_status": "none",
            "pending_approval": None,
            "approved_steps": [step_id],
            "trace_log": [f"human_approval[notify]: acknowledged '{step_id}' by {reviewer}{suffix}"],
        }

    # Tier 4: Human Takeover (inject human-authored deliverable & bypass remaining steps)
    if tier == "human_takeover" or decision == "taken_over":
        takeover_output = str(decision_payload.get("takeover_output", "")).strip()
        if not takeover_output:
            return {
                "approval_status": "none",
                "pending_approval": None,
                "last_error": f"Human takeover by '{reviewer}' missing takeover_output.",
                "trace_log": [f"human_approval[human_takeover]: missing output from {reviewer}"],
            }
        takeover_record = {
            "step_id": step_id,
            "step_index": state["current_step_index"],
            "specialist": "human_takeover",
            "tool": "human_override",
            "title": f"Human takeover by {reviewer}",
            "output": takeover_output,
            "duration_ms": 0.0,
            "status": "success",
        }
        return {
            "approval_status": "none",
            "pending_approval": None,
            "subtask_results": [takeover_record],
            "current_step_index": len(state["plan"]),
            "retry_count": 0,
            "last_error": "",
            "confidence_score": 1.0,
            "trace_log": [
                f"human_approval[human_takeover]: '{reviewer}' completed task manually{suffix}"
            ],
        }

    # Tier 3: Modify Plan (replace remaining steps with human-validated DAG & re-dispatch)
    if (tier == "modify_plan" or decision == "modified") and decision != "rejected":
        raw_plan = decision_payload.get("modified_plan") or []
        validated = ExecutionPlan(
            goal_summary="Human-modified execution plan",
            subtasks=[Subtask.model_validate(step) for step in raw_plan],
        )
        completed_prefix = state["plan"][: state["current_step_index"]]
        updated_plan = completed_prefix + [t.model_dump() for t in validated.ordered_subtasks()]
        return {
            "approval_status": "none",
            "pending_approval": None,
            "plan": updated_plan,
            "retry_count": 0,
            "last_error": "",
            "confidence_score": 0.95,
            "trace_log": [
                f"human_approval[modify_plan]: '{reviewer}' replaced remaining plan with "
                f"{len(raw_plan)} step(s){suffix}"
            ],
        }

    # Tier 2: Approve Action (approve with optional arg override, or reject cleanly)
    if decision == "approved":
        overrides = dict(state["tool_arg_overrides"])
        modified_args = decision_payload.get("modified_tool_args")
        if isinstance(modified_args, dict) and modified_args:
            overrides[step_id] = modified_args

        override_note = " with modified tool args" if modified_args else ""
        return {
            "approval_status": "none",
            "pending_approval": None,
            "approved_steps": [step_id],
            "tool_arg_overrides": overrides,
            "retry_count": 0,
            "last_error": "",
            "trace_log": [
                f"human_approval[approve_action]: '{step_id}' approved by {reviewer}"
                f"{override_note}{suffix}"
            ],
        }

    rejection_msg = (
        f"Human reviewer '{reviewer}' rejected '{step_id}'"
        + (f": {comment}" if comment else "")
    )
    return {
        "approval_status": "none",
        "pending_approval": None,
        "last_error": rejection_msg,
        "trace_log": [f"human_approval[{tier}]: {rejection_msg}"],
    }


def resolve_and_resume_with_tier(
    graph: Any,
    thread_id: str,
    approval_queue: ApprovalQueue,
    review_id: str,
    resolution: HumanResolutionPayload,
) -> dict[str, Any]:
    """Resolve a queued HITL item using any approval tier and resume the paused graph."""
    queue_decision = (
        "approved" if resolution.decision == "acknowledged" else resolution.decision
    )
    approval_queue.resolve(
        review_id=review_id,
        decision=queue_decision,  # type: ignore[arg-type]
        reviewer=resolution.reviewer,
        comment=resolution.comment,
    )
    config = {"configurable": {"thread_id": thread_id}}
    # Preserve approval_status='pending' so route_after_specialist targets human_approval
    graph.update_state(
        config,
        {"human_decision": resolution.model_dump()},
    )
    return dict(graph.invoke(None, config=config))