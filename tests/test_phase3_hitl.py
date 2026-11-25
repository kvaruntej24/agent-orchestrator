import json

import pytest

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.hitl.approval_queue import ApprovalQueue, ReviewRequest
from src.orchestrator.hitl.tiers import (
    HumanResolutionPayload,
    resolve_and_resume_with_tier,
)
from src.orchestrator.hitl.triggers import EscalationPolicy
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry
from src.orchestrator.ui.review_helpers import (
    build_execution_timeline,
    compute_unified_diff,
    create_ui_runtime,
    launch_demo_task,
    submit_ui_resolution,
)


def test_escalation_policy_priority_order() -> None:
    policy = EscalationPolicy(confidence_threshold=0.75, consecutive_failure_cap=2)
    registry = register_builtin_tools(ToolRegistry())
    state = create_initial_state("t-esc", "Test triggers")
    state["confidence_score"] = 0.40

    # Sensitive action takes priority over low confidence
    decision = policy.evaluate(
        state=state,
        pending_tool_name="file_write",
        pending_tool_args={"relative_path": "out.txt", "content": "hi"},
        registry=registry,
    )
    assert decision.should_escalate is True
    assert decision.reason == "sensitive_action"
    assert decision.recommended_tier == "approve_action"

    # Low confidence triggers modify_plan when tool is safe
    low_conf = policy.evaluate(
        state=state,
        pending_tool_name="web_search",
        pending_tool_args={"query": "benchmarks"},
        registry=registry,
    )
    assert low_conf.should_escalate is True
    assert low_conf.reason == "low_confidence"
    assert low_conf.recommended_tier == "modify_plan"


def test_approval_queue_double_resolution_guard() -> None:
    queue = ApprovalQueue.in_memory()
    req = queue.enqueue(
        ReviewRequest(
            task_id="t-dup",
            thread_id="t-dup",
            step_id="step_3",
            reason="sensitive_action",
            severity="high",
            recommended_tier="approve_action",
            details="Write guarded",
        )
    )
    queue.resolve(req.review_id, "approved", "alice")
    with pytest.raises(ValueError, match="already resolved"):
        queue.resolve(req.review_id, "rejected", "bob")


def test_all_four_hitl_tiers_pause_and_resume() -> None:
    # Tier 1 (Notify on db_query) + Tier 2 (Approve Action with arg override on file_write)
    q1 = ApprovalQueue.in_memory()
    g1 = build_graph(
        planner=TaskPlanner(mock=True),
        escalation_policy=EscalationPolicy(notify_tools={"db_query"}),
        approval_queue=q1,
    )
    cfg1 = {"configurable": {"thread_id": "t-tier-12"}}
    g1.invoke(create_initial_state("t-tier-12", "Audit Q3 cloud spend"), config=cfg1)
    assert g1.get_state(cfg1).next == ("human_approval",)
    assert any(r.status == "notified" for r in q1.list_all())

    req1 = q1.list_pending()[0]
    res1 = resolve_and_resume_with_tier(
        g1,
        "t-tier-12",
        q1,
        req1.review_id,
        HumanResolutionPayload(
            tier="approve_action",
            decision="approved",
            reviewer="ops_lead",
            modified_tool_args={"relative_path": "pytest_approved.txt"},
        ),
    )
    assert res1["review_status"] == "approved"
    assert q1.list_pending() == []

    # Tier 3 (Modify Plan — replaces file_write with read-only analyst query)
    q3 = ApprovalQueue.in_memory()
    g3 = build_graph(
        planner=TaskPlanner(mock=True),
        escalation_policy=EscalationPolicy(),
        approval_queue=q3,
    )
    cfg3 = {"configurable": {"thread_id": "t-tier-3"}}
    g3.invoke(create_initial_state("t-tier-3", "Audit Q3 cloud spend"), config=cfg3)
    req3 = q3.list_pending()[0]
    res3 = resolve_and_resume_with_tier(
        g3,
        "t-tier-3",
        q3,
        req3.review_id,
        HumanResolutionPayload(
            tier="modify_plan",
            decision="modified",
            reviewer="architect",
            modified_plan=[
                {
                    "step_id": "step_3_safe",
                    "title": "Verify spend read-only",
                    "specialist": "analyst",
                    "instruction": "Read-only verification",
                    "depends_on": [],
                }
            ],
        ),
    )
    assert res3["review_status"] == "approved"
    assert res3["subtask_results"][-1]["step_id"] == "step_3_safe"

    # Tier 4 (Human Takeover)
    q4 = ApprovalQueue.in_memory()
    g4 = build_graph(
        planner=TaskPlanner(mock=True),
        escalation_policy=EscalationPolicy(),
        approval_queue=q4,
    )
    cfg4 = {"configurable": {"thread_id": "t-tier-4"}}
    g4.invoke(create_initial_state("t-tier-4", "Audit Q3 cloud spend"), config=cfg4)
    req4 = q4.list_pending()[0]
    res4 = resolve_and_resume_with_tier(
        g4,
        "t-tier-4",
        q4,
        req4.review_id,
        HumanResolutionPayload(
            tier="human_takeover",
            decision="taken_over",
            reviewer="oncall_eng",
            takeover_output="Manual verified FinOps report saving $18,400 in Q3.",
        ),
    )
    assert res4["review_status"] == "approved"
    assert "$18,400" in res4["final_output"]


def test_ui_helpers_diff_and_override_submission() -> None:
    diff = compute_unified_diff(
        {"relative_path": "raw.txt"},
        json.dumps({"relative_path": "redacted.txt"}, indent=2, sort_keys=True),
    )
    assert '-  "relative_path": "raw.txt"' in diff
    assert '+  "relative_path": "redacted.txt"' in diff

    rt = create_ui_runtime()
    thread = launch_demo_task(rt, "Audit Q3 cloud spend")
    assert thread["next_nodes"] == ("human_approval",)
    assert len(build_execution_timeline(thread["state"])) == 2

    req = rt["approval_queue"].list_pending()[0]
    final = submit_ui_resolution(
        rt,
        req,
        HumanResolutionPayload(
            tier="approve_action",
            decision="approved",
            reviewer="ui_tester",
        ),
    )
    assert final["review_status"] == "approved"
    assert rt["threads"][req.thread_id]["next_nodes"] == ()