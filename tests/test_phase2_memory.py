from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.api.memory_routes import create_memory_router
from src.orchestrator.memory.long_term_memory import LongTermMemoryStore
from src.orchestrator.memory.maintenance import (
    MemoryMaintenanceService,
    compute_importance_score,
)
from src.orchestrator.memory.retrieval import MemoryRetriever
from src.orchestrator.memory.working_memory import WorkingMemoryStore


def test_working_memory_crud_and_purge() -> None:
    wm = WorkingMemoryStore.in_memory()
    assert wm.health_check() is True

    wm.save_state("t-wm", {"phase": "specialist"})
    wm.append_subtask_output("t-wm", {"step_id": "step_1", "status": "success"})
    wm.set_context("t-wm", "goal", "Audit spend")

    snapshot = wm.get_task_snapshot("t-wm")
    assert snapshot["state"]["phase"] == "specialist"
    assert len(snapshot["subtask_outputs"]) == 1
    assert snapshot["context"]["goal"] == "Audit spend"

    deleted = wm.clear_task("t-wm")
    assert deleted == 3


def test_closed_loop_retrieval_quality_gate_and_budget_cap() -> None:
    ltm = LongTermMemoryStore.in_memory()
    retriever = MemoryRetriever(ltm, max_prompt_chars=240)
    retriever.set_user_preference("finops_user", "format", "executive bullets")

    planner = TaskPlanner(mock=True)
    graph = build_graph(
        planner=planner,
        long_term_memory=ltm,
        memory_retriever=retriever,
    )

    # Run 1 succeeds and archives into LTM
    r1 = graph.invoke(
        create_initial_state(
            "t-mem-1",
            "Audit Q3 GPU cloud spend and cut idle costs",
            user_id="finops_user",
        )
    )
    assert r1["review_status"] == "approved"

    # Run 2 recalls Run 1 strategy + user preferences within strict char budget
    r2 = graph.invoke(
        create_initial_state(
            "t-mem-2",
            "Optimize Q4 GPU cloud spend and idle clusters",
            user_id="finops_user",
        )
    )
    assert r2["review_status"] == "approved"
    assert r2["retrieved_memory_context"]
    assert len(r2["retrieved_memory_context"]) <= 240
    assert "executive bullets" in r2["retrieved_memory_context"]


def test_importance_scoring_and_three_store_privacy_purge() -> None:
    wm = WorkingMemoryStore.in_memory()
    ltm = LongTermMemoryStore.in_memory()
    retriever = MemoryRetriever(ltm)
    retriever.set_user_preference("alice", "tone", "concise")

    graph = build_graph(
        planner=TaskPlanner(mock=True),
        working_memory=wm,
        long_term_memory=ltm,
        memory_retriever=retriever,
    )
    final_ok = graph.invoke(
        create_initial_state("t-alice-1", "Audit Q3 cloud spend", user_id="alice")
    )
    failed_state = create_initial_state("t-fail", "Broken task", user_id="alice")
    failed_state["review_status"] = "failed"
    failed_state["confidence_score"] = 0.0

    assert compute_importance_score(final_ok) > compute_importance_score(failed_state)

    maintenance = MemoryMaintenanceService(long_term_memory=ltm, working_memory=wm)
    app = FastAPI()
    app.include_router(create_memory_router(maintenance))
    client = TestClient(app)

    purge_resp = client.delete(
        "/memory/users/alice",
        params=[("task_ids", "t-alice-1")],
    )
    assert purge_resp.status_code == 200
    report = purge_resp.json()
    assert report["redis_keys_deleted"] == 3
    assert report["relational_tasks_deleted"] == 1
    assert report["preferences_deleted"] == 1
    assert report["chroma_vectors_deleted"] == 4