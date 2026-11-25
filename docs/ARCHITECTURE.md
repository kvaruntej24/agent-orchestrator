# System Architecture & Design Notes

This document summarizes the key engineering decisions, failure-mode guarantees, and runtime trade-offs inside the **Enterprise Agent Orchestration Platform**.

## 1. Explicit State Machine Over Free-Form Loops
All control flow runs through a compiled LangGraph `StateGraph` backed by a strongly typed `AgentState` (`src/orchestrator/agents/state.py`):
- **Append-only reducers (`Annotated[list, operator.add]`):** `subtask_results`, `trace_log`, and `approved_steps` accumulate safely across transitions without any node overwriting prior audit entries.
- **DAG validation before execution:** `ExecutionPlan` runs Kahn's topological sort inside a Pydantic `@model_validator`, rejecting duplicate step IDs, missing dependencies, invalid specialist roles, and dependency cycles before any tool executes.
- **Guaranteed termination:** Both `retry_count` (`max_retries`) and `review_cycles` (`max_review_cycles`) are bounded in typed state. Exhausting either limit routes deterministically to `reviewer_node` with `review_status="failed"`.

## 2. Defense-in-Depth Tool Sandboxing & MCP v2
Tools live in a single `ToolRegistry` (`src/orchestrator/tools/registry.py`) and are exposed externally via an official MCP v2 server (`src/orchestrator/tools/mcp_adapter.py`):
- **No prompt-only security:** File tools enforce `Path.resolve().is_relative_to(workspace_root)` to stop traversal (`../.env`); `db_query` enforces single-statement read-only `SELECT` queries; mutating tools carry `requires_approval=True`.
- **Threadpool safety:** Synchronous FastAPI routes run in an AnyIO worker threadpool; `db_query`'s read-only in-memory SQLite connection sets `check_same_thread=False` so concurrent API requests do not trigger cross-thread errors.

## 3. Dual-Tier Memory & Anti-Poisoning Guards
- **Working memory (Redis 7):** Uses three separate keys per task—String (`SETEX`) for phase state, List (`RPUSH`) for ordered subtask outputs, and Hash (`HSET`) for key-value context—with pipelined TTL refreshes to avoid read-modify-write races and key leaks.
- **Long-term memory (PostgreSQL 16 + ChromaDB):** PostgreSQL stores ACID task records, cascading tool-decision audit rows, and user preferences; ChromaDB stores semantic vectors via a pluggable `TextEmbedder` protocol (`LocalHashEmbedder` offline, `OpenAIEmbedder` live).
- **Quality gating & privacy:** `MemoryRetriever` injects only `approved` past task summaries and `success` tool precedents above a similarity floor within a hard `max_prompt_chars` budget. `MemoryMaintenanceService` computes zero-token deterministic importance scores, consolidates noisy entries, and executes atomic 3-store purges across Redis, PostgreSQL, and ChromaDB.

## 4. Checkpointed 4-Tier HITL Governance
`EscalationPolicy` evaluates guardrails in strict priority order: (1) sensitive write/delete interception, (2) trailing consecutive failure streak, (3) confidence floor.
- **Non-blocking pause/resume:** LangGraph compiles with `interrupt_before=["human_approval"]` and freezes thread state without holding open compute while `ApprovalQueue` persists tickets to PostgreSQL and pushes fast-poll IDs to Redis.
- **Resume routing invariant:** Calling `graph.update_state()` re-evaluates the preceding node's conditional router (`route_after_specialist`). Keeping `approval_status="pending"` while injecting `human_decision` guarantees the resumed thread enters `human_approval_node` cleanly instead of re-pausing in `specialist_node`.
- **Four graduated tiers:** `notify` (non-blocking audit), `approve_action` (with live JSON diff argument override), `modify_plan` (re-validated through Kahn's DAG check before re-dispatch), and `human_takeover` (manual verified output routed straight to final review).

## 5. Observability, Budget Circuit Breakers & Deterministic Replay
- **OpenTelemetry 3-level tracing:** Each `ExecutionTracer` owns an isolated `TracerProvider` so tests never collide while Python `contextvars` automatically nest `orchestrator.run` -> `node.*` -> `tool.*` under one `trace_id`.
- **Hard budget circuit breakers:** `CostTracker` checks `max_cost_usd` and `max_tokens` before and after every node, annotates the active OTel span (`llm.cost.usd`, `budget.tripped`), and routes breaches immediately past remaining specialists directly into `reviewer_node` for clean failure persistence.
- **Flight-recorder replay:** `RecordingToolRegistry` captures tool calls into portable JSON `ExecutionRecording` artifacts. `ExecutionReplayEngine` replays runs offline via `ReplayTaskPlanner` and `ReplayToolRegistry` with zero side effects and supports 0-based `call_overrides` for counterfactual debugging.

## 6. Environment-Driven Deployment & CI
`build_service_from_env()` switches cleanly between offline development and live Docker services using two flags:
- `ORCHESTRATOR_STORAGE_MODE=in_memory|live`
- `ORCHESTRATOR_LIVE_LLM=false|true`

A single non-root Python 3.11 `Dockerfile` serves both the FastAPI API (`8000`) and Streamlit HITL UI (`8501`), gated behind PostgreSQL and Redis health checks in `docker-compose.yml` and verified on every push by `.github/workflows/ci.yml`.