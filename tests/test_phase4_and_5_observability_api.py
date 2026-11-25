from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.api.app import build_service_from_env, create_app
from src.orchestrator.demo.enterprise_scenario import run_enterprise_finops_demo
from src.orchestrator.observability.cost_tracker import (
    BudgetExceededError,
    CostTracker,
)
from src.orchestrator.observability.replay import (
    ExecutionRecording,
    ExecutionReplayEngine,
)
from src.orchestrator.observability.tracing import ExecutionTracer, run_traced_graph
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry, ToolResult


def test_opentelemetry_three_level_trace_and_cost_attributes() -> None:
    tracer = ExecutionTracer()
    cost_tracker = CostTracker(max_cost_usd=0.50, max_tokens=50000)
    graph = build_graph(
        planner=TaskPlanner(mock=True),
        tracer=tracer,
        cost_tracker=cost_tracker,
    )

    out = run_traced_graph(
        graph,
        create_initial_state("t-otel-1", "Audit Q3 cloud spend"),
        tracer,
    )
    assert out["review_status"] == "approved"

    spans = tracer.get_finished_spans("t-otel-1")
    assert len({s["trace_id"] for s in spans}) == 1
    tree = tracer.get_trace_tree("t-otel-1")
    assert len(tree) == 1
    nested_tools = [c["name"] for n in tree[0]["children"] for c in n["children"]]
    assert nested_tools == ["tool.web_search", "tool.db_query", "tool.file_write"]

    cost_spans = [s for s in spans if "llm.cost.usd" in s["attributes"]]
    assert len(cost_spans) >= 5


def test_budget_circuit_breaker_halts_graph_and_raises_hard_error() -> None:
    tight_tracker = CostTracker(max_cost_usd=0.50, max_tokens=30)
    graph = build_graph(planner=TaskPlanner(mock=True), cost_tracker=tight_tracker)

    out = graph.invoke(create_initial_state("t-budget-halt", "Audit Q3 cloud spend"))
    assert out["review_status"] == "failed"
    assert out["budget_exceeded"] is True
    assert len(out["subtask_results"]) == 0
    assert tight_tracker.get_task_summary("t-budget-halt").circuit_breaker_tripped is True

    usd_tracker = CostTracker(max_cost_usd=0.001, max_tokens=100000)
    usd_tracker.record_usage(
        "t-usd",
        "supervisor",
        prompt_tokens=100,
        completion_tokens=50,
        model="gpt-4o",
        raise_on_breach=True,
    )
    with pytest.raises(BudgetExceededError):
        usd_tracker.record_usage(
            "t-usd",
            "specialist",
            prompt_tokens=500,
            completion_tokens=200,
            model="gpt-4o",
            raise_on_breach=True,
        )


def test_deterministic_and_counterfactual_replay_roundtrip(tmp_path: Path) -> None:
    engine = ExecutionReplayEngine()
    _, rec = engine.capture_run(
        create_initial_state("t-rep", "Audit Q3 cloud spend", max_retries=0),
        planner=TaskPlanner(mock=True),
        registry=register_builtin_tools(ToolRegistry()),
        cost_tracker=CostTracker(),
    )

    saved_path = rec.save_json(tmp_path)
    loaded = ExecutionRecording.load_json(saved_path)
    assert loaded.recording_id == rec.recording_id

    exact = engine.replay(loaded)
    assert exact.deterministic_match is True
    assert exact.divergences == []

    what_if = engine.replay(
        loaded,
        call_overrides={
            0: ToolResult(
                tool_name="web_search",
                success=False,
                error="Simulated503: upstream timeout",
                duration_ms=1.0,
            )
        },
    )
    assert what_if.deterministic_match is False
    assert what_if.replayed_status == "failed"


def test_fastapi_bootstrap_and_end_to_end_enterprise_scenario() -> None:
    client = TestClient(create_app(build_service_from_env()))
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/tasks/does-not-exist").status_code == 404

    demo = run_enterprise_finops_demo()
    assert demo["health_ok"] is True
    assert demo["run1_paused_before"] == ["human_approval"]
    assert demo["run1_final_status"] == "approved"
    assert demo["run2_final_status"] == "approved"
    assert demo["run2_memory_injected"] is True
    assert demo["replay_exact_match"] is True
    assert demo["replay_whatif_diverged"] is True
    assert demo["privacy_purge_cleared_all_stores"] is True