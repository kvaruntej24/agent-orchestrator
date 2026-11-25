import difflib
import json
import uuid
from typing import Any

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.hitl.approval_queue import ApprovalQueue, ReviewRequest
from src.orchestrator.hitl.tiers import (
    HumanResolutionPayload,
    resolve_and_resume_with_tier,
)
from src.orchestrator.hitl.triggers import EscalationPolicy
from src.orchestrator.memory.long_term_memory import LongTermMemoryStore
from src.orchestrator.memory.retrieval import MemoryRetriever
from src.orchestrator.memory.working_memory import WorkingMemoryStore
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry


def compute_unified_diff(
    original_payload: Any,
    edited_text: str,
    label: str = "proposed_payload.json",
) -> str:
    """Compute a unified diff between an original JSON payload and edited JSON text."""
    original_text = (
        original_payload
        if isinstance(original_payload, str)
        else json.dumps(original_payload, indent=2, sort_keys=True)
    )
    diff_lines = list(
        difflib.unified_diff(
            original_text.splitlines(),
            edited_text.splitlines(),
            fromfile=f"original/{label}",
            tofile=f"edited/{label}",
            lineterm="",
        )
    )
    if not diff_lines:
        return "No changes detected."
    return "\n".join(diff_lines)


def build_execution_timeline(state_values: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw AgentState subtask_results into clean rows for UI display."""
    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(state_values.get("subtask_results", []), start=1):
        rows.append(
            {
                "order": idx,
                "step_id": result.get("step_id", f"step_{idx}"),
                "specialist": result.get("specialist", "unknown"),
                "tool": result.get("tool", "unknown"),
                "status": result.get("status", "unknown"),
                "duration_ms": result.get("duration_ms", 0.0),
                "preview": str(result.get("output") or result.get("error") or "")[:120],
            }
        )
    return rows


def create_ui_runtime() -> dict[str, Any]:
    """Initialize an isolated in-memory orchestration stack for interactive dashboard review."""
    working_memory = WorkingMemoryStore.in_memory()
    long_term_memory = LongTermMemoryStore.in_memory()
    retriever = MemoryRetriever(long_term_memory)
    approval_queue = ApprovalQueue.in_memory()
    registry = register_builtin_tools(ToolRegistry())
    policy = EscalationPolicy(notify_tools={"db_query"})
    graph = build_graph(
        planner=TaskPlanner(mock=True),
        registry=registry,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        memory_retriever=retriever,
        escalation_policy=policy,
        approval_queue=approval_queue,
    )
    return {
        "graph": graph,
        "approval_queue": approval_queue,
        "working_memory": working_memory,
        "long_term_memory": long_term_memory,
        "threads": {},
    }


def launch_demo_task(
    runtime: dict[str, Any],
    user_request: str,
    user_id: str = "finops_lead",
) -> dict[str, Any]:
    """Start a task in the runtime; pauses at human_approval before sensitive tools."""
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": task_id}}
    state = create_initial_state(
        task_id=task_id,
        user_request=user_request,
        user_id=user_id,
    )
    result = dict(runtime["graph"].invoke(state, config=config))
    snapshot = runtime["graph"].get_state(config)
    runtime["threads"][task_id] = {
        "task_id": task_id,
        "user_request": user_request,
        "next_nodes": tuple(snapshot.next),
        "state": result,
    }
    return runtime["threads"][task_id]


def submit_ui_resolution(
    runtime: dict[str, Any],
    review_request: ReviewRequest,
    resolution: HumanResolutionPayload,
) -> dict[str, Any]:
    """Apply a human override from the UI, resume the paused graph, and refresh thread state."""
    final_state = resolve_and_resume_with_tier(
        graph=runtime["graph"],
        thread_id=review_request.thread_id,
        approval_queue=runtime["approval_queue"],
        review_id=review_request.review_id,
        resolution=resolution,
    )
    config = {"configurable": {"thread_id": review_request.thread_id}}
    snapshot = runtime["graph"].get_state(config)
    runtime["threads"][review_request.thread_id] = {
        "task_id": review_request.task_id,
        "user_request": final_state.get("user_request", ""),
        "next_nodes": tuple(snapshot.next),
        "state": final_state,
    }
    return final_state