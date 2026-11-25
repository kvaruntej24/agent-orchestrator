import math
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field

# USD per 1,000,000 tokens: (input_rate_usd, output_rate_usd)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}
DEFAULT_MODEL_PRICING: tuple[float, float] = (0.50, 1.50)


class BudgetExceededError(RuntimeError):
    """Raised when a task exceeds its configured USD cost or token budget."""


class UsageRecord(BaseModel):
    """Single node or LLM call token and cost accounting entry."""

    task_id: str
    node_name: str
    model: str
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TaskCostSummary(BaseModel):
    """Aggregated token, USD cost, and circuit-breaker status for one task."""

    task_id: str
    call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    max_cost_usd: float
    max_tokens: int
    budget_remaining_usd: float
    budget_utilization_pct: float = 0.0
    warning_triggered: bool = False
    circuit_breaker_tripped: bool = False
    breach_reason: str = ""
    per_node_cost_usd: dict[str, float] = Field(default_factory=dict)
    per_node_tokens: dict[str, int] = Field(default_factory=dict)


def estimate_tokens(payload: Any) -> int:
    """Deterministic ~4 chars/token estimator for offline/mock telemetry."""
    text = str(payload or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


class CostTracker:
    """Tracks per-task and per-node token usage/cost with hard budget circuit breakers."""

    def __init__(
        self,
        max_cost_usd: float = 0.50,
        max_tokens: int = 50000,
        warn_ratio: float = 0.80,
        default_model: str = "gpt-4o-mini",
        pricing_table: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if not 0.0 < warn_ratio <= 1.0:
            raise ValueError("warn_ratio must be in (0.0, 1.0].")

        self.max_cost_usd = float(max_cost_usd)
        self.max_tokens = int(max_tokens)
        self.warn_ratio = float(warn_ratio)
        self.default_model = default_model
        self._pricing = dict(MODEL_PRICING_PER_1M)
        if pricing_table:
            self._pricing.update(pricing_table)
        self._records_by_task: dict[str, list[UsageRecord]] = {}

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Compute exact USD cost for a model call from per-1M input/output rates."""
        in_rate, out_rate = self._pricing.get(model, DEFAULT_MODEL_PRICING)
        cost = (prompt_tokens / 1_000_000) * in_rate + (
            completion_tokens / 1_000_000
        ) * out_rate
        return round(cost, 8)

    def check_budget(self, task_id: str) -> tuple[bool, str]:
        """Return (is_within_budget, breach_reason) for a task."""
        records = self._records_by_task.get(task_id, [])
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = round(sum(r.cost_usd for r in records), 8)

        if total_cost > self.max_cost_usd:
            return (
                False,
                f"BudgetCircuitBreaker: task '{task_id}' spent ${total_cost:.6f} "
                f"(cap ${self.max_cost_usd:.6f}).",
            )
        if total_tokens > self.max_tokens:
            return (
                False,
                f"BudgetCircuitBreaker: task '{task_id}' used {total_tokens} tokens "
                f"(cap {self.max_tokens}).",
            )
        return True, ""

    def record_usage(
        self,
        task_id: str,
        node_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        raise_on_breach: bool = False,
    ) -> UsageRecord:
        """Record token/cost usage, annotate active OpenTelemetry span, and check caps."""
        active_model = model or self.default_model
        p_toks = max(0, int(prompt_tokens))
        c_toks = max(0, int(completion_tokens))
        tot_toks = p_toks + c_toks
        cost_usd = self.calculate_cost(active_model, p_toks, c_toks)

        record = UsageRecord(
            task_id=task_id,
            node_name=node_name,
            model=active_model,
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            total_tokens=tot_toks,
            cost_usd=cost_usd,
        )
        self._records_by_task.setdefault(task_id, []).append(record)

        # Enrich active OpenTelemetry span automatically if tracing is active
        current_span = otel_trace.get_current_span()
        if current_span is not None and current_span.is_recording():
            current_span.set_attribute("llm.model", active_model)
            current_span.set_attribute("llm.tokens.prompt", p_toks)
            current_span.set_attribute("llm.tokens.completion", c_toks)
            current_span.set_attribute("llm.tokens.total", tot_toks)
            current_span.set_attribute("llm.cost.usd", cost_usd)

        within_budget, reason = self.check_budget(task_id)
        if not within_budget:
            if current_span is not None and current_span.is_recording():
                current_span.set_attribute("budget.tripped", True)
                current_span.set_attribute("budget.breach_reason", reason)
            if raise_on_breach:
                raise BudgetExceededError(reason)

        return record

    def get_task_summary(self, task_id: str) -> TaskCostSummary:
        """Return aggregated token counts, USD spend, per-node breakdown, and breaker state."""
        records = self._records_by_task.get(task_id, [])
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = round(sum(r.cost_usd for r in records), 8)

        per_node_cost: dict[str, float] = {}
        per_node_tokens: dict[str, int] = {}
        for r in records:
            per_node_cost[r.node_name] = round(
                per_node_cost.get(r.node_name, 0.0) + r.cost_usd, 8
            )
            per_node_tokens[r.node_name] = (
                per_node_tokens.get(r.node_name, 0) + r.total_tokens
            )

        within_budget, breach_reason = self.check_budget(task_id)
        cost_ratio = total_cost / self.max_cost_usd
        token_ratio = total_tokens / self.max_tokens
        max_ratio = max(cost_ratio, token_ratio)

        return TaskCostSummary(
            task_id=task_id,
            call_count=len(records),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            max_cost_usd=self.max_cost_usd,
            max_tokens=self.max_tokens,
            budget_remaining_usd=round(max(0.0, self.max_cost_usd - total_cost), 8),
            budget_utilization_pct=round(min(100.0, max_ratio * 100.0), 2),
            warning_triggered=max_ratio >= self.warn_ratio,
            circuit_breaker_tripped=not within_budget,
            breach_reason=breach_reason,
            per_node_cost_usd=per_node_cost,
            per_node_tokens=per_node_tokens,
        )

    def get_records(self, task_id: str) -> list[UsageRecord]:
        """Return chronological list of all recorded usage entries for a task."""
        return list(self._records_by_task.get(task_id, []))

    def get_global_report(self) -> dict[str, Any]:
        """Summarize cumulative spend and breaker trips across all tracked tasks."""
        summaries = [self.get_task_summary(tid) for tid in self._records_by_task]
        return {
            "tasks_tracked": len(summaries),
            "total_tokens": sum(s.total_tokens for s in summaries),
            "total_cost_usd": round(sum(s.total_cost_usd for s in summaries), 8),
            "circuit_breakers_tripped": sum(
                1 for s in summaries if s.circuit_breaker_tripped
            ),
        }