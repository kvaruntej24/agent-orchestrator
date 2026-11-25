import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when launched via `streamlit run src/orchestrator/ui/hitl_dashboard.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.orchestrator.hitl.approval_queue import ReviewRequest
from src.orchestrator.hitl.tiers import HumanResolutionPayload
from src.orchestrator.ui.review_helpers import (
    build_execution_timeline,
    compute_unified_diff,
    create_ui_runtime,
    launch_demo_task,
    submit_ui_resolution,
)


def _get_runtime() -> dict[str, Any]:
    """Initialize and cache the orchestration runtime inside Streamlit session_state."""
    if "hitl_runtime" not in st.session_state:
        runtime = create_ui_runtime()
        # Seed one paused task immediately so the dashboard opens with a live review ready
        launch_demo_task(
            runtime,
            user_request="Audit Q3 GPU cloud spend and generate FinOps savings report",
        )
        st.session_state["hitl_runtime"] = runtime
    return dict(st.session_state["hitl_runtime"])


def _render_execution_path(thread_info: dict[str, Any]) -> None:
    """Display the agent's plan, completed steps, checkpoint status, and trace log."""
    state = thread_info["state"]
    next_nodes = thread_info["next_nodes"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Task ID", thread_info["task_id"])
    col2.metric("Review Status", state.get("review_status", "pending"))
    col3.metric("Confidence", f"{float(state.get('confidence_score', 0.0)):.2f}")
    col4.metric("Paused Before", ", ".join(next_nodes) if next_nodes else "Completed")

    timeline = build_execution_timeline(state)
    st.subheader("Agent Execution Path")
    if timeline:
        st.dataframe(timeline, use_container_width=True, hide_index=True)
    else:
        st.info("No subtasks executed yet.")

    with st.expander("Full Execution Trace Log", expanded=False):
        for idx, line in enumerate(state.get("trace_log", []), start=1):
            st.text(f"{idx:02d}. {line}")


def _render_pending_review(runtime: dict[str, Any], req: ReviewRequest) -> None:
    """Display the proposed payload, live diff editor, and granular override buttons."""
    st.divider()
    st.subheader(f"Pending HITL Review: {req.review_id}")

    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.markdown(f"**Step:** `{req.step_id}`")
    meta_col2.markdown(f"**Reason:** `{req.reason}`")
    meta_col3.markdown(f"**Severity:** `{req.severity.upper()}`")
    meta_col4.markdown(f"**Recommended Tier:** `{req.recommended_tier}`")
    st.warning(f"**Trigger Details:** {req.details}")

    reviewer_col, comment_col = st.columns([1, 2])
    reviewer = reviewer_col.text_input(
        "Reviewer Name",
        value="ops_lead",
        key=f"reviewer_{req.review_id}",
    )
    comment = comment_col.text_input(
        "Audit Comment",
        value="Reviewed payload and verified policy compliance.",
        key=f"comment_{req.review_id}",
    )

    tab_approve, tab_modify, tab_takeover = st.tabs(
        [
            "Tier 2: Approve / Reject Action (Diff Editor)",
            "Tier 3: Modify Remaining Plan (DAG Editor)",
            "Tier 4: Human Takeover",
        ]
    )

    # ---------- Tab 1: Approve Action with Live Payload Diff Editor ----------
    with tab_approve:
        original_args = req.proposed_action.get("arguments", {})
        tool_name = req.proposed_action.get("tool_name", "unknown")
        st.markdown(f"**Proposed Tool:** `{tool_name}`")

        left_col, right_col = st.columns(2)
        with left_col:
            edited_args_text = st.text_area(
                "Edit Proposed Tool Arguments (JSON)",
                value=json.dumps(original_args, indent=2, sort_keys=True),
                height=220,
                key=f"args_editor_{req.review_id}",
            )
        with right_col:
            st.markdown("**Live Unified Diff**")
            diff_text = compute_unified_diff(
                original_args,
                edited_args_text,
                label=f"{tool_name}_args.json",
            )
            st.code(diff_text, language="diff")

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button(
            "Approve & Resume Execution",
            type="primary",
            use_container_width=True,
            key=f"btn_approve_{req.review_id}",
        ):
            try:
                parsed_args = json.loads(edited_args_text)
                if not isinstance(parsed_args, dict):
                    raise ValueError("Edited arguments must be a JSON object.")
                overrides = parsed_args if parsed_args != original_args else None
                resolution = HumanResolutionPayload(
                    tier="approve_action",
                    decision="approved",
                    reviewer=reviewer.strip() or "ops_lead",
                    comment=comment.strip(),
                    modified_tool_args=overrides,
                )
                submit_ui_resolution(runtime, req, resolution)
                st.success(f"Approved {req.review_id} and resumed execution.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not apply approval: {exc}")

        if btn_col2.button(
            "Reject & Halt Task",
            use_container_width=True,
            key=f"btn_reject_{req.review_id}",
        ):
            try:
                resolution = HumanResolutionPayload(
                    tier="approve_action",
                    decision="rejected",
                    reviewer=reviewer.strip() or "ops_lead",
                    comment=comment.strip() or "Rejected by reviewer.",
                )
                submit_ui_resolution(runtime, req, resolution)
                st.warning(f"Rejected {req.review_id}; task halted safely.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not apply rejection: {exc}")

    # ---------- Tab 2: Modify Remaining Plan with Live Diff ----------
    with tab_modify:
        thread_state = runtime["threads"][req.thread_id]["state"]
        step_idx = int(thread_state.get("current_step_index", 0))
        remaining_plan = thread_state.get("plan", [])[step_idx:]

        plan_left, plan_right = st.columns(2)
        with plan_left:
            edited_plan_text = st.text_area(
                "Edit Remaining Plan Steps (JSON Array — Validated via Kahn's DAG Check)",
                value=json.dumps(remaining_plan, indent=2, sort_keys=True),
                height=220,
                key=f"plan_editor_{req.review_id}",
            )
        with plan_right:
            st.markdown("**Live Plan Diff**")
            plan_diff = compute_unified_diff(
                remaining_plan,
                edited_plan_text,
                label="remaining_plan.json",
            )
            st.code(plan_diff, language="diff")

        if st.button(
            "Apply Modified Plan & Re-Dispatch",
            type="primary",
            use_container_width=True,
            key=f"btn_modify_{req.review_id}",
        ):
            try:
                parsed_plan = json.loads(edited_plan_text)
                if not isinstance(parsed_plan, list):
                    raise ValueError("Modified plan must be a JSON list of subtasks.")
                resolution = HumanResolutionPayload(
                    tier="modify_plan",
                    decision="modified",
                    reviewer=reviewer.strip() or "ops_lead",
                    comment=comment.strip(),
                    modified_plan=parsed_plan,
                )
                submit_ui_resolution(runtime, req, resolution)
                st.success(f"Replaced remaining plan for {req.task_id} and resumed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Plan validation failed: {exc}")

    # ---------- Tab 3: Human Takeover ----------
    with tab_takeover:
        takeover_text = st.text_area(
            "Enter Verified Human Deliverable (Bypasses Remaining Automated Steps)",
            value="Verified Q3 Cloud Audit: Cut idle staging GPUs ($18,400) via 1-yr Reserved Instances.",
            height=140,
            key=f"takeover_{req.review_id}",
        )
        if st.button(
            "Complete via Human Takeover",
            type="primary",
            use_container_width=True,
            key=f"btn_takeover_{req.review_id}",
        ):
            try:
                resolution = HumanResolutionPayload(
                    tier="human_takeover",
                    decision="taken_over",
                    reviewer=reviewer.strip() or "ops_lead",
                    comment=comment.strip(),
                    takeover_output=takeover_text.strip(),
                )
                submit_ui_resolution(runtime, req, resolution)
                st.success(f"Human takeover completed for {req.task_id}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Human takeover failed: {exc}")


def main() -> None:
    """Render the Agent Orchestration HITL Review Dashboard."""
    st.set_page_config(
        page_title="Agent Orchestrator — HITL Review Dashboard",
        layout="wide",
    )
    st.title("Agent Orchestrator — Human-in-the-Loop Review Console")
    st.caption(
        "Inspect paused agent threads, diff proposed payloads, and resolve escalations across all four approval tiers."
    )

    runtime = _get_runtime()
    queue = runtime["approval_queue"]
    pending_list = queue.list_pending()
    all_history = queue.list_all()

    # ---------- Sidebar: Launch Tasks & Select Thread ----------
    with st.sidebar:
        st.header("Task Control")
        new_prompt = st.text_input(
            "New Task Request",
            value="Audit Q4 database spend and draft cost-reduction memo",
        )
        if st.button("Launch New Task", type="primary", use_container_width=True):
            launch_demo_task(runtime, user_request=new_prompt)
            st.rerun()

        st.divider()
        st.metric("Pending Approvals", len(pending_list))
        st.metric(
            "Non-Blocking Notifications",
            sum(1 for r in all_history if r.status == "notified"),
        )
        st.metric(
            "Resolved Reviews",
            sum(
                1
                for r in all_history
                if r.status in ("approved", "rejected", "modified", "taken_over")
            ),
        )

        thread_ids = list(runtime["threads"].keys())
        selected_thread = st.selectbox(
            "Inspect Execution Thread",
            options=thread_ids,
            index=len(thread_ids) - 1,
        )

    # ---------- Main Panel ----------
    thread_info = runtime["threads"][selected_thread]
    _render_execution_path(thread_info)

    matching_pending = [r for r in pending_list if r.thread_id == selected_thread]
    if matching_pending:
        _render_pending_review(runtime, matching_pending[0])
    else:
        final_out = thread_info["state"].get("final_output", "")
        feedback = thread_info["state"].get("review_feedback", "")
        st.divider()
        st.subheader("Final Execution Outcome")
        if thread_info["state"].get("review_status") == "approved":
            st.success(f"Task Completed — {feedback}")
        else:
            st.error(f"Task Halted — {feedback}")
        if final_out:
            st.code(final_out, language="text")

    # ---------- Audit History Table ----------
    st.divider()
    st.subheader("HITL Queue & Tier-1 Notification Audit Log")
    if all_history:
        audit_rows = [
            {
                "review_id": r.review_id,
                "task_id": r.task_id,
                "step_id": r.step_id,
                "tier": r.recommended_tier,
                "status": r.status,
                "reviewer": r.reviewer or "-",
                "comment": r.reviewer_comment or "-",
            }
            for r in all_history
        ]
        st.dataframe(audit_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No HITL events recorded yet.")


if __name__ == "__main__":
    main()