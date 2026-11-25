from typing import Any

from fastapi.testclient import TestClient

from src.orchestrator.api.app import OrchestratorService, create_app


def run_enterprise_finops_demo() -> dict[str, Any]:
    """Execute an end-to-end FinOps scenario exercising Phases 1-4 via FastAPI."""
    service = OrchestratorService.in_memory(max_cost_usd=0.50, max_tokens=50000)
    app = create_app(service)
    client = TestClient(app)

    # 1. Health check
    health = client.get("/health").json()

    # 2. Run #1: Submit FinOps audit with HITL enabled (pauses before file_write)
    run1_paused = client.post(
        "/tasks",
        json={
            "user_request": "Audit Q3 GPU cloud spend and generate FinOps savings memo",
            "user_id": "finops_lead",
            "user_preferences": {
                "format": "executive bullets",
                "focus": "GPU cost savings",
            },
            "enable_hitl": True,
        },
    ).json()

    review_id = run1_paused["pending_review_id"]
    pending_before = client.get("/hitl/pending").json()

    # 3. Human approves Tier 2 action with a redacted output filename override
    run1_completed = client.post(
        f"/hitl/{review_id}/resolve",
        json={
            "tier": "approve_action",
            "decision": "approved",
            "reviewer": "vp_finops",
            "comment": "Approved with redacted executive filename",
            "modified_tool_args": {"relative_path": "q3_finops_redacted.txt"},
        },
    ).json()

    # 4. Run #2: Similar Q4 request proves closed-loop memory recall of Run #1 + preferences
    run2_completed = client.post(
        "/tasks",
        json={
            "user_request": "Optimize Q4 GPU cloud spend and reserved instance utilization",
            "user_id": "finops_lead",
            "enable_hitl": False,
        },
    ).json()

    # 5. Inspect OpenTelemetry trace, CostTracker metrics, and deterministic + counterfactual replay
    trace_info = client.get(f"/observability/traces/{run1_completed['task_id']}").json()
    global_costs = client.get("/observability/costs").json()

    rec_id = run1_completed["recording_id"]
    replay_exact = client.post(f"/replay/{rec_id}", json={}).json()
    replay_whatif = client.post(
        f"/replay/{rec_id}",
        json={
            "call_overrides": {
                "0": {
                    "tool_name": "web_search",
                    "success": False,
                    "output": None,
                    "error": "Simulated503: search gateway timeout",
                    "duration_ms": 1.0,
                }
            }
        },
    ).json()

    # 6. Exercise Phase 2.4 GDPR/privacy purge endpoint on a disposable user run
    temp_run = client.post(
        "/tasks",
        json={
            "user_request": "Draft temporary Q3 compliance check",
            "user_id": "temp_auditor",
            "user_preferences": {"tone": "formal"},
            "enable_hitl": False,
        },
    ).json()
    purge_report = client.delete(
        "/memory/users/temp_auditor",
        params=[("task_ids", temp_run["task_id"])],
    ).json()

    return {
        "health_ok": health["status"] == "ok",
        "run1_paused_before": run1_paused["paused_before"],
        "run1_pending_count_before_approval": len(pending_before),
        "run1_final_status": run1_completed["review_status"],
        "run1_recording_id": rec_id,
        "run2_final_status": run2_completed["review_status"],
        "run2_memory_injected": run2_completed["memory_injected"],
        "run1_trace_roots": trace_info["root_span_count"],
        "global_tasks_tracked": global_costs["tasks_tracked"],
        "global_total_tokens": global_costs["total_tokens"],
        "replay_exact_match": replay_exact["deterministic_match"],
        "replay_whatif_diverged": not replay_whatif["deterministic_match"],
        "privacy_purge_cleared_all_stores": (
            purge_report["redis_keys_deleted"] == 3
            and purge_report["relational_tasks_deleted"] == 1
            and purge_report["preferences_deleted"] == 1
            and purge_report["chroma_vectors_deleted"] == 4
        ),
    }


def main() -> None:
    summary = run_enterprise_finops_demo()
    print("=== Enterprise Agent Orchestration Platform — End-to-End Demo ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()