import uuid
import os
import uuid
from typing import Any
from typing import Any
from src.orchestrator.config import get_settings
from src.orchestrator.memory.long_term_memory import (
    LongTermMemoryStore,
    OpenAIEmbedder,
)
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.orchestrator.agents.graph import build_graph
from src.orchestrator.agents.planner import TaskPlanner
from src.orchestrator.agents.state import create_initial_state
from src.orchestrator.api.memory_routes import create_memory_router
from src.orchestrator.hitl.approval_queue import ApprovalQueue, ReviewRequest
from src.orchestrator.hitl.tiers import (
    HumanResolutionPayload,
    resolve_and_resume_with_tier,
)
from src.orchestrator.hitl.triggers import EscalationPolicy
from src.orchestrator.memory.long_term_memory import LongTermMemoryStore
from src.orchestrator.memory.maintenance import MemoryMaintenanceService
from src.orchestrator.memory.retrieval import MemoryRetriever
from src.orchestrator.memory.working_memory import WorkingMemoryStore
from src.orchestrator.observability.cost_tracker import CostTracker, TaskCostSummary
from src.orchestrator.observability.replay import (
    ExecutionRecording,
    ExecutionReplayEngine,
    RecordingToolRegistry,
    ReplayDiffReport,
)
from src.orchestrator.observability.tracing import ExecutionTracer, run_traced_graph
from src.orchestrator.tools.builtin import register_builtin_tools
from src.orchestrator.tools.registry import ToolRegistry, ToolResult


# ---------- API Request / Response Contracts ----------

class TaskSubmissionRequest(BaseModel):
    """Payload to start a new multi-agent orchestration run."""

    user_request: str = Field(..., min_length=3)
    user_id: str = Field(default="default", min_length=1)
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    enable_hitl: bool = True
    max_retries: int = Field(default=2, ge=0, le=5)
    max_review_cycles: int = Field(default=2, ge=0, le=5)


class TaskExecutionResponse(BaseModel):
    """Unified response returned after launching or resuming a task."""

    task_id: str
    user_id: str
    status: str
    paused_before: list[str] = Field(default_factory=list)
    pending_review_id: str | None = None
    review_status: str
    review_feedback: str = ""
    confidence_score: float = 0.0
    subtask_count: int = 0
    memory_injected: bool = False
    recording_id: str | None = None
    final_output: str = ""
    cost_summary: TaskCostSummary
    trace_log: list[str] = Field(default_factory=list)


class ReplayRequest(BaseModel):
    """Optional counterfactual tool overrides for offline deterministic replay."""

    call_overrides: dict[int, ToolResult] | None = None


# ---------- Unified Orchestrator Service ----------

class OrchestratorService:
    """Wires Phases 1-4 into one cohesive, dependency-injected runtime."""

    def __init__(
        self,
        working_memory: WorkingMemoryStore,
        long_term_memory: LongTermMemoryStore,
        approval_queue: ApprovalQueue,
        planner: TaskPlanner | None = None,
        registry: ToolRegistry | None = None,
        escalation_policy: EscalationPolicy | None = None,
        tracer: ExecutionTracer | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.memory_retriever = MemoryRetriever(long_term_memory)
        self.maintenance = MemoryMaintenanceService(
            long_term_memory=long_term_memory,
            working_memory=working_memory,
        )
        self.approval_queue = approval_queue
        self.planner = planner or TaskPlanner(mock=True)
        self.base_registry = registry or register_builtin_tools(ToolRegistry())
        self.escalation_policy = escalation_policy or EscalationPolicy(
            notify_tools={"db_query"}
        )
        self.tracer = tracer or ExecutionTracer(service_name="agent-orchestrator-api")
        self.cost_tracker = cost_tracker or CostTracker()
        self.replay_engine = ExecutionReplayEngine()
        self.checkpointer = MemorySaver()

        # Per-task runtime registries for clean HITL resume and isolated flight recordings
        self._task_graphs: dict[str, Any] = {}
        self._task_recorders: dict[str, RecordingToolRegistry] = {}
        self._task_states: dict[str, dict[str, Any]] = {}
        self._task_configs: dict[str, dict[str, Any]] = {}
        self._recordings_by_id: dict[str, ExecutionRecording] = {}
        self._recording_id_by_task: dict[str, str] = {}

    @classmethod
    def in_memory(
        cls,
        max_cost_usd: float = 0.50,
        max_tokens: int = 50000,
    ) -> "OrchestratorService":
        """Build a zero-infrastructure runtime ideal for local dev, demos, and Pytest."""
        return cls(
            working_memory=WorkingMemoryStore.in_memory(),
            long_term_memory=LongTermMemoryStore.in_memory(),
            approval_queue=ApprovalQueue.in_memory(),
            cost_tracker=CostTracker(
                max_cost_usd=max_cost_usd,
                max_tokens=max_tokens,
            ),
        )

    
    @classmethod
    def from_urls(
        cls,
        redis_url: str,
        postgres_dsn: str,
        chroma_persist_dir: str = "chroma_data",
        working_memory_ttl_seconds: int = 3600,
        max_cost_usd: float = 0.50,
        max_tokens: int = 50000,
        planner: TaskPlanner | None = None,
        embedder: Any | None = None,
    ) -> "OrchestratorService":
        """Build a live production runtime connected to Redis, PostgreSQL, and ChromaDB."""
        return cls(
            working_memory=WorkingMemoryStore.from_url(
                redis_url=redis_url,
                ttl_seconds=working_memory_ttl_seconds,
            ),
            long_term_memory=LongTermMemoryStore.from_dsn(
                postgres_dsn=postgres_dsn,
                chroma_persist_dir=chroma_persist_dir,
                embedder=embedder,
            ),
            approval_queue=ApprovalQueue.from_urls(
                postgres_dsn=postgres_dsn,
                redis_url=redis_url,
            ),
            planner=planner,
            cost_tracker=CostTracker(
                max_cost_usd=max_cost_usd,
                max_tokens=max_tokens,
            ),
        )

    def _finalize_recording(self, task_id: str, state: dict[str, Any]) -> str:
        """Build and store a portable ExecutionRecording once a task reaches a terminal state."""
        recorder = self._task_recorders[task_id]
        cost_summary = self.cost_tracker.get_task_summary(task_id).model_dump()
        recording = ExecutionRecording(
            task_id=task_id,
            user_id=str(state.get("user_id", "default")),
            user_request=str(state.get("user_request", "")),
            max_retries=int(state.get("max_retries", 2)),
            max_review_cycles=int(state.get("max_review_cycles", 2)),
            plan=list(state.get("plan", [])),
            tool_calls=list(recorder.recorded_calls),
            final_review_status=str(state.get("review_status", "pending")),
            final_output=str(state.get("final_output", "")),
            review_feedback=str(state.get("review_feedback", "")),
            confidence_score=float(state.get("confidence_score", 0.0)),
            subtask_count=len(state.get("subtask_results", [])),
            trace_log=list(state.get("trace_log", [])),
            otel_trace_tree=self.tracer.get_trace_tree(task_id),
            cost_summary=cost_summary,
        )
        self._recordings_by_id[recording.recording_id] = recording
        self._recording_id_by_task[task_id] = recording.recording_id
        return recording.recording_id

    def _build_task_response(
        self,
        task_id: str,
        state: dict[str, Any],
        paused_before: list[str],
    ) -> TaskExecutionResponse:
        pending_review_id: str | None = None
        if paused_before:
            pending_items = [
                r for r in self.approval_queue.list_pending() if r.task_id == task_id
            ]
            if pending_items:
                pending_review_id = pending_items[0].review_id
            status = "awaiting_approval"
            recording_id = None
        else:
            status = str(state.get("review_status", "completed"))
            recording_id = self._recording_id_by_task.get(task_id)

        return TaskExecutionResponse(
            task_id=task_id,
            user_id=str(state.get("user_id", "default")),
            status=status,
            paused_before=paused_before,
            pending_review_id=pending_review_id,
            review_status=str(state.get("review_status", "pending")),
            review_feedback=str(state.get("review_feedback", "")),
            confidence_score=float(state.get("confidence_score", 0.0)),
            subtask_count=len(state.get("subtask_results", [])),
            memory_injected=bool(state.get("retrieved_memory_context")),
            recording_id=recording_id,
            final_output=str(state.get("final_output", "")),
            cost_summary=self.cost_tracker.get_task_summary(task_id),
            trace_log=list(state.get("trace_log", [])),
        )

    def submit_task(self, payload: TaskSubmissionRequest) -> TaskExecutionResponse:
        """Start a new traced, budget-governed orchestration run."""
        for pref_key, pref_val in payload.user_preferences.items():
            self.memory_retriever.set_user_preference(
                payload.user_id, pref_key, pref_val
            )

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        recorder = RecordingToolRegistry(self.base_registry)
        active_policy = self.escalation_policy if payload.enable_hitl else None

        graph = build_graph(
            planner=self.planner,
            registry=recorder,  # type: ignore[arg-type]
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory,
            memory_retriever=self.memory_retriever,
            escalation_policy=active_policy,
            approval_queue=self.approval_queue if payload.enable_hitl else None,
            checkpointer=self.checkpointer if payload.enable_hitl else None,
            tracer=self.tracer,
            cost_tracker=self.cost_tracker,
        )

        config = {"configurable": {"thread_id": task_id}}
        initial_state = create_initial_state(
            task_id=task_id,
            user_request=payload.user_request,
            user_id=payload.user_id,
            max_retries=payload.max_retries,
            max_review_cycles=payload.max_review_cycles,
        )

        self._task_graphs[task_id] = graph
        self._task_recorders[task_id] = recorder
        self._task_configs[task_id] = config

        result = run_traced_graph(
            graph,
            dict(initial_state),
            self.tracer,
            config=config if payload.enable_hitl else None,
        )
        self._task_states[task_id] = result

        paused_before: list[str] = []
        if payload.enable_hitl:
            snapshot = graph.get_state(config)
            paused_before = list(snapshot.next)

        if not paused_before:
            self._finalize_recording(task_id, result)

        return self._build_task_response(task_id, result, paused_before)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Inspect live state, working memory snapshot, and recording metadata for a task."""
        if task_id not in self._task_states:
            raise KeyError(f"Task '{task_id}' not found.")

        graph = self._task_graphs[task_id]
        config = self._task_configs[task_id]
        paused_before: list[str] = []
        try:
            snapshot = graph.get_state(config)
            paused_before = list(snapshot.next)
        except Exception:
            paused_before = []

        return {
            "task": self._build_task_response(
                task_id, self._task_states[task_id], paused_before
            ).model_dump(),
            "working_memory": self.working_memory.get_task_snapshot(task_id),
            "long_term_record": self.long_term_memory.get_task_record(task_id),
        }

    def resolve_hitl(
        self,
        review_id: str,
        resolution: HumanResolutionPayload,
    ) -> TaskExecutionResponse:
        """Resolve a paused HITL ticket across any tier and resume execution inside OTel."""
        req = self.approval_queue.get(review_id)
        if req is None:
            raise KeyError(f"Review request '{review_id}' not found.")
        if req.task_id not in self._task_graphs:
            raise KeyError(f"No active checkpoint graph found for task '{req.task_id}'.")

        task_id = req.task_id
        graph = self._task_graphs[task_id]
        config = self._task_configs[task_id]
        user_id = str(self._task_states[task_id].get("user_id", "default"))

        with self.tracer.start_run_span(
            task_id=task_id,
            user_id=user_id,
            user_request=f"[HITL resume:{resolution.tier}] {review_id}",
        ):
            resumed = resolve_and_resume_with_tier(
                graph=graph,
                thread_id=req.thread_id,
                approval_queue=self.approval_queue,
                review_id=review_id,
                resolution=resolution,
            )

        self._task_states[task_id] = resumed
        snapshot = graph.get_state(config)
        paused_before = list(snapshot.next)

        if not paused_before:
            self._finalize_recording(task_id, resumed)

        return self._build_task_response(task_id, resumed, paused_before)

    def replay_recording(
        self,
        recording_id: str,
        call_overrides: dict[int, ToolResult] | None = None,
    ) -> ReplayDiffReport:
        """Deterministically replay a captured task offline with optional counterfactuals."""
        recording = self._recordings_by_id.get(recording_id)
        if recording is None:
            raise KeyError(f"Recording '{recording_id}' not found.")
        return self.replay_engine.replay(
            recording=recording,
            call_overrides=call_overrides,
        )

def build_service_from_env() -> OrchestratorService:
    """Build an OrchestratorService from environment variables (in_memory or live Docker stack)."""
    storage_mode = os.getenv("ORCHESTRATOR_STORAGE_MODE", "in_memory").strip().lower()
    live_llm = os.getenv("ORCHESTRATOR_LIVE_LLM", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if storage_mode == "live":
        settings = get_settings()
        planner = TaskPlanner(mock=not live_llm)
        embedder = (
            OpenAIEmbedder(api_key=settings.openai_api_key) if live_llm else None
        )
        return OrchestratorService.from_urls(
            redis_url=settings.redis_url,
            postgres_dsn=settings.postgres_dsn,
            chroma_persist_dir=settings.chroma_persist_dir,
            working_memory_ttl_seconds=settings.working_memory_ttl_seconds,
            max_cost_usd=settings.max_task_cost_usd,
            max_tokens=settings.max_task_tokens,
            planner=planner,
            embedder=embedder,
        )

    if live_llm:
        settings = get_settings()
        return OrchestratorService(
            working_memory=WorkingMemoryStore.in_memory(),
            long_term_memory=LongTermMemoryStore.in_memory(
                embedder=OpenAIEmbedder(api_key=settings.openai_api_key)
            ),
            approval_queue=ApprovalQueue.in_memory(),
            planner=TaskPlanner(mock=False),
            cost_tracker=CostTracker(
                max_cost_usd=settings.max_task_cost_usd,
                max_tokens=settings.max_task_tokens,
            ),
        )

    return OrchestratorService.in_memory()
# ---------- FastAPI Application Factory ----------

def create_app(service: OrchestratorService | None = None) -> FastAPI:
    """Create the unified FastAPI application exposing all orchestration subsystems."""
    active_service = service or OrchestratorService.in_memory()

    app = FastAPI(
        title="Enterprise Agent Orchestration Platform",
        version="1.0.0",
        description=(
            "Multi-agent LangGraph orchestrator with MCP tools, Redis/PostgreSQL/ChromaDB "
            "memory, 4-tier HITL approvals, OpenTelemetry tracing, cost circuit breakers, "
            "and deterministic replay."
        ),
    )
    @app.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "service": "Enterprise Agent Orchestration Platform",
            "docs": "/docs",
            "health": "/health",
        }
    # Mount Phase 2.4 memory consolidation & GDPR privacy purge endpoints
    app.include_router(create_memory_router(active_service.maintenance))

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "working_memory_healthy": active_service.working_memory.health_check(),
            "registered_tools": [
                t.name for t in active_service.base_registry.export_mcp_tools()
            ],
            "pending_hitl_approvals": len(active_service.approval_queue.list_pending()),
        }

    @app.post("/tasks", response_model=TaskExecutionResponse, tags=["orchestration"])
    def submit_task(payload: TaskSubmissionRequest) -> TaskExecutionResponse:
        return active_service.submit_task(payload)

    @app.get("/tasks/{task_id}", tags=["orchestration"])
    def inspect_task(task_id: str) -> dict[str, Any]:
        try:
            return active_service.get_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/hitl/pending", response_model=list[ReviewRequest], tags=["hitl"])
    def list_pending_reviews() -> list[ReviewRequest]:
        return active_service.approval_queue.list_pending()

    @app.get("/hitl/history", response_model=list[ReviewRequest], tags=["hitl"])
    def list_review_history(limit: int = 50) -> list[ReviewRequest]:
        return active_service.approval_queue.list_all(limit=limit)

    @app.post(
        "/hitl/{review_id}/resolve",
        response_model=TaskExecutionResponse,
        tags=["hitl"],
    )
    def resolve_review(
        review_id: str,
        resolution: HumanResolutionPayload,
    ) -> TaskExecutionResponse:
        try:
            return active_service.resolve_hitl(review_id, resolution)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/observability/traces/{task_id}", tags=["observability"])
    def get_task_trace(task_id: str) -> dict[str, Any]:
        tree = active_service.tracer.get_trace_tree(task_id)
        if not tree:
            raise HTTPException(
                status_code=404,
                detail=f"No OpenTelemetry trace found for task '{task_id}'.",
            )
        return {
            "task_id": task_id,
            "root_span_count": len(tree),
            "trace_tree": tree,
        }

    @app.get(
        "/observability/costs/{task_id}",
        response_model=TaskCostSummary,
        tags=["observability"],
    )
    def get_task_cost(task_id: str) -> TaskCostSummary:
        return active_service.cost_tracker.get_task_summary(task_id)

    @app.get("/observability/costs", tags=["observability"])
    def get_global_costs() -> dict[str, Any]:
        return active_service.cost_tracker.get_global_report()

    @app.post(
        "/replay/{recording_id}",
        response_model=ReplayDiffReport,
        tags=["observability"],
    )
    def replay_task(
        recording_id: str,
        payload: ReplayRequest | None = None,
    ) -> ReplayDiffReport:
        try:
            overrides = payload.call_overrides if payload is not None else None
            return active_service.replay_recording(
                recording_id=recording_id,
                call_overrides=overrides,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


# Default ASGI app entrypoint for `uvicorn src.orchestrator.api.app:app --reload`
app = create_app(build_service_from_env())