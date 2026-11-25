import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.nodes import ReviewPolicy
from src.orchestrator.agents.planner import ExecutionPlan, Subtask, TaskPlanner
from src.orchestrator.agents.state import AgentState, create_initial_state
from src.orchestrator.observability.cost_tracker import CostTracker
from src.orchestrator.observability.tracing import ExecutionTracer, run_traced_graph
from src.orchestrator.tools.registry import RegisteredTool, ToolRegistry, ToolResult


class RecordedToolCall(BaseModel):
    """Immutable snapshot of a single tool invocation captured during execution."""

    call_index: int = Field(..., ge=0)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult


class ExecutionRecording(BaseModel):
    """Portable flight-recorder artifact capturing everything needed for deterministic replay."""

    recording_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:10]}")
    task_id: str
    user_id: str = "default"
    user_request: str
    max_retries: int = 2
    max_review_cycles: int = 2
    plan: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[RecordedToolCall] = Field(default_factory=list)
    final_review_status: str = "pending"
    final_output: str = ""
    review_feedback: str = ""
    confidence_score: float = 0.0
    subtask_count: int = 0
    trace_log: list[str] = Field(default_factory=list)
    otel_trace_tree: list[dict[str, Any]] = Field(default_factory=list)
    cost_summary: dict[str, Any] | None = None
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save_json(self, directory: str | Path) -> Path:
        """Persist this recording as formatted JSON on disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{self.recording_id}.json"
        file_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return file_path

    @classmethod
    def load_json(cls, file_path: str | Path) -> "ExecutionRecording":
        """Load a previously persisted ExecutionRecording from JSON."""
        raw = Path(file_path).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)


class ReplayDiffReport(BaseModel):
    """Comparison between an original recorded run and its deterministic replay."""

    recording_id: str
    task_id: str
    deterministic_match: bool
    original_status: str
    replayed_status: str
    original_subtask_count: int
    replayed_subtask_count: int
    tool_calls_replayed: int
    divergences: list[str] = Field(default_factory=list)
    replay_duration_ms: float = 0.0
    replayed_final_output: str = ""
    replayed_review_feedback: str = ""
    replayed_trace_log: list[str] = Field(default_factory=list)


class RecordingToolRegistry:
    """Transparent proxy around ToolRegistry that records every tool call and result."""

    def __init__(self, inner: ToolRegistry) -> None:
        self._inner = inner
        self.recorded_calls: list[RecordedToolCall] = []

    def get(self, name: str) -> RegisteredTool | None:
        return self._inner.get(name)

    def set_tracer(self, tracer: Any | None) -> None:
        self._inner.set_tracer(tracer)

    def export_mcp_tools(self) -> list[Any]:
        return self._inner.export_mcp_tools()

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        args_copy = dict(arguments or {})
        result = self._inner.execute(name, args_copy)
        self.recorded_calls.append(
            RecordedToolCall(
                call_index=len(self.recorded_calls),
                tool_name=name,
                arguments=args_copy,
                result=result.model_copy(deep=True),
            )
        )
        return result


class ReplayTaskPlanner(TaskPlanner):
    """Deterministic planner that replays a captured ExecutionPlan with zero LLM calls."""

    def __init__(self, recorded_plan: list[dict[str, Any]]) -> None:
        super().__init__(mock=True)
        self._recorded_plan = [dict(step) for step in recorded_plan]

    def create_plan(
        self,
        user_request: str,
        memory_context: str = "",
    ) -> ExecutionPlan:
        self.last_memory_context = memory_context.strip()
        subtasks = [Subtask.model_validate(step) for step in self._recorded_plan]
        return ExecutionPlan(
            goal_summary=f"Replayed plan for: {user_request}",
            subtasks=subtasks,
        )


class ReplayToolRegistry:
    """Replays captured ToolResults deterministically with zero live side effects.

    Supports counterfactual debugging via `call_overrides` (keyed by 0-based call_index)
    so engineers can test what happens if a failing tool succeeds—or vice versa.
    """

    def __init__(
        self,
        recorded_calls: list[RecordedToolCall],
        call_overrides: dict[int, ToolResult] | None = None,
        metadata_source: ToolRegistry | None = None,
    ) -> None:
        self._recorded_calls = list(recorded_calls)
        self._call_overrides = dict(call_overrides or {})
        self._metadata_source = metadata_source
        self._tracer: Any | None = None
        self.cursor: int = 0
        self.executed_count: int = 0

    def get(self, name: str) -> RegisteredTool | None:
        if self._metadata_source is not None:
            return self._metadata_source.get(name)
        return None

    def set_tracer(self, tracer: Any | None) -> None:
        self._tracer = tracer

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        idx = self.cursor
        self.cursor += 1
        self.executed_count += 1

        if idx in self._call_overrides:
            result = self._call_overrides[idx].model_copy(deep=True)
        elif idx < len(self._recorded_calls):
            recorded = self._recorded_calls[idx]
            if recorded.tool_name != name:
                result = ToolResult(
                    tool_name=name,
                    success=False,
                    error=(
                        f"ReplayMismatchError: expected tool '{recorded.tool_name}' "
                        f"at call #{idx}, got '{name}'."
                    ),
                    duration_ms=0.0,
                )
            else:
                result = recorded.result.model_copy(deep=True)
        else:
            # Allow counterfactual extra steps (e.g., an added reviewer revision) to succeed cleanly
            result = ToolResult(
                tool_name=name,
                success=True,
                output=f"[replayed fallback output for '{name}' call #{idx}]",
                duration_ms=0.0,
            )

        if self._tracer is not None:
            with self._tracer.start_tool_span(name, list((arguments or {}).keys())) as span:
                span.set_attribute("tool.replayed", True)
                span.set_attribute("tool.call_index", idx)
                span.set_attribute("tool.success", result.success)
        return result


class ExecutionReplayEngine:
    """Captures live/mock agent executions and replays them deterministically offline."""

    def __init__(self) -> None:
        self._recordings: dict[str, ExecutionRecording] = {}

    def capture_run(
        self,
        initial_state: AgentState,
        planner: TaskPlanner,
        registry: ToolRegistry,
        review_policy: ReviewPolicy | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> tuple[dict[str, Any], ExecutionRecording]:
        """Run the graph with transparent tool recording, OTel tracing, and cost capture."""
        tracer = ExecutionTracer(service_name="agent-orchestrator-capture")
        recording_registry = RecordingToolRegistry(registry)
        graph = build_graph(
            planner=planner,
            registry=recording_registry,  # type: ignore[arg-type]
            review_policy=review_policy,
            tracer=tracer,
            cost_tracker=cost_tracker,
        )

        final_state = run_traced_graph(graph, dict(initial_state), tracer)
        cost_payload = (
            cost_tracker.get_task_summary(initial_state["task_id"]).model_dump()
            if cost_tracker is not None
            else None
        )

        recording = ExecutionRecording(
            task_id=initial_state["task_id"],
            user_id=initial_state["user_id"],
            user_request=initial_state["user_request"],
            max_retries=initial_state["max_retries"],
            max_review_cycles=initial_state["max_review_cycles"],
            plan=list(final_state.get("plan", [])),
            tool_calls=list(recording_registry.recorded_calls),
            final_review_status=str(final_state.get("review_status", "pending")),
            final_output=str(final_state.get("final_output", "")),
            review_feedback=str(final_state.get("review_feedback", "")),
            confidence_score=float(final_state.get("confidence_score", 0.0)),
            subtask_count=len(final_state.get("subtask_results", [])),
            trace_log=list(final_state.get("trace_log", [])),
            otel_trace_tree=tracer.get_trace_tree(initial_state["task_id"]),
            cost_summary=cost_payload,
        )
        self._recordings[recording.recording_id] = recording
        return final_state, recording

    def get_recording(self, recording_id: str) -> ExecutionRecording | None:
        """Retrieve an in-memory recording by ID."""
        return self._recordings.get(recording_id)

    def replay(
        self,
        recording: ExecutionRecording,
        call_overrides: dict[int, ToolResult] | None = None,
        review_policy: ReviewPolicy | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> ReplayDiffReport:
        """Re-run a recorded execution deterministically and diff against the original."""
        start = time.perf_counter()

        replay_planner = ReplayTaskPlanner(recording.plan)
        replay_registry = ReplayToolRegistry(
            recorded_calls=recording.tool_calls,
            call_overrides=call_overrides,
        )
        replay_tracer = ExecutionTracer(service_name="agent-orchestrator-replay")
        graph = build_graph(
            planner=replay_planner,
            registry=replay_registry,  # type: ignore[arg-type]
            review_policy=review_policy,
            tracer=replay_tracer,
            cost_tracker=cost_tracker,
        )

        seed_state = create_initial_state(
            task_id=recording.task_id,
            user_request=recording.user_request,
            user_id=recording.user_id,
            max_retries=recording.max_retries,
            max_review_cycles=recording.max_review_cycles,
        )
        replayed = run_traced_graph(graph, dict(seed_state), replay_tracer)
        duration_ms = round((time.perf_counter() - start) * 1000, 3)

        replayed_status = str(replayed.get("review_status", "pending"))
        replayed_output = str(replayed.get("final_output", ""))
        replayed_subtask_count = len(replayed.get("subtask_results", []))

        divergences: list[str] = []
        if replayed_status != recording.final_review_status:
            divergences.append(
                f"review_status changed: '{recording.final_review_status}' -> '{replayed_status}'"
            )
        if replayed_subtask_count != recording.subtask_count:
            divergences.append(
                f"subtask_count changed: {recording.subtask_count} -> {replayed_subtask_count}"
            )
        if replayed_output != recording.final_output:
            divergences.append("final_output differs from original recording")

        return ReplayDiffReport(
            recording_id=recording.recording_id,
            task_id=recording.task_id,
            deterministic_match=len(divergences) == 0,
            original_status=recording.final_review_status,
            replayed_status=replayed_status,
            original_subtask_count=recording.subtask_count,
            replayed_subtask_count=replayed_subtask_count,
            tool_calls_replayed=replay_registry.executed_count,
            divergences=divergences,
            replay_duration_ms=duration_ms,
            replayed_final_output=replayed_output,
            replayed_review_feedback=str(replayed.get("review_feedback", "")),
            replayed_trace_log=list(replayed.get("trace_log", [])),
        )