import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.orchestrator.agents.state import AgentState
from src.orchestrator.tools.registry import ToolRegistry

EscalationReason = Literal[
    "sensitive_action",
    "consecutive_failures",
    "low_confidence",
]
EscalationSeverity = Literal["low", "medium", "high"]
ApprovalTier = Literal["notify", "approve_action", "modify_plan", "human_takeover"]

_DESTRUCTIVE_SQL_PATTERN = re.compile(
    r"\b(delete|drop|truncate|update|insert|alter|grant|revoke)\b",
    re.IGNORECASE,
)
_SENSITIVE_NAME_TOKENS = ("write", "delete", "remove", "drop", "update", "mutate")


class EscalationDecision(BaseModel):
    """Structured decision returned by HITL trigger evaluation."""

    should_escalate: bool = False
    reason: EscalationReason | None = None
    severity: EscalationSeverity = "low"
    recommended_tier: ApprovalTier = "notify"
    details: str = ""
    proposed_action: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class EscalationPolicy:
    """Evaluates deterministic HITL guardrails before tool execution or completion.

    Priority order:
      1. Sensitive write/delete action interceptor (pre-execution safety gate)
      2. Consecutive specialist failure cap (stops runaway failure loops)
      3. Confidence threshold fallback (catches uncertain outputs before delivery)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        consecutive_failure_cap: int = 2,
        sensitive_tools: set[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if consecutive_failure_cap < 1:
            raise ValueError("consecutive_failure_cap must be at least 1.")

        self.confidence_threshold = confidence_threshold
        self.consecutive_failure_cap = consecutive_failure_cap
        self.sensitive_tools = sensitive_tools or {
            "file_write",
            "file_delete",
            "db_write",
            "db_delete",
        }

    def check_sensitive_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        registry: ToolRegistry | None = None,
    ) -> EscalationDecision:
        """Intercept write, delete, or approval-flagged tool calls before execution."""
        reasons: list[str] = []

        if registry is not None:
            registered = registry.get(tool_name)
            if registered is not None and registered.requires_approval:
                reasons.append(f"tool '{tool_name}' is flagged requires_approval=True")

        lowered_name = tool_name.lower()
        if lowered_name in self.sensitive_tools or any(
            token in lowered_name for token in _SENSITIVE_NAME_TOKENS
        ):
            reasons.append(f"tool '{tool_name}' matches sensitive write/delete policy")

        sql_arg = arguments.get("sql")
        if isinstance(sql_arg, str) and _DESTRUCTIVE_SQL_PATTERN.search(sql_arg):
            reasons.append("SQL payload contains a write/destructive keyword")

        if not reasons:
            return EscalationDecision(
                should_escalate=False,
                details=f"Tool '{tool_name}' is safe for autonomous execution.",
            )

        return EscalationDecision(
            should_escalate=True,
            reason="sensitive_action",
            severity="high",
            recommended_tier="approve_action",
            details="; ".join(reasons),
            proposed_action={"tool_name": tool_name, "arguments": arguments},
        )

    def check_consecutive_failures(self, state: AgentState) -> EscalationDecision:
        """Escalate when consecutive trailing specialist failures hit the configured cap."""
        trailing_errors = 0
        for result in reversed(state["subtask_results"]):
            if result.get("status") == "error":
                trailing_errors += 1
            else:
                break

        effective_streak = max(trailing_errors, int(state["retry_count"]))
        if effective_streak >= self.consecutive_failure_cap:
            return EscalationDecision(
                should_escalate=True,
                reason="consecutive_failures",
                severity="high",
                recommended_tier="human_takeover",
                details=(
                    f"Specialist hit {effective_streak} consecutive failures "
                    f"(cap={self.consecutive_failure_cap}). "
                    f"Last error: {state['last_error'] or 'unspecified'}"
                ),
                metrics={
                    "consecutive_failures": effective_streak,
                    "failure_cap": self.consecutive_failure_cap,
                },
            )

        return EscalationDecision(
            should_escalate=False,
            details=(
                f"Consecutive failures ({effective_streak}) below cap "
                f"({self.consecutive_failure_cap})."
            ),
            metrics={
                "consecutive_failures": effective_streak,
                "failure_cap": self.consecutive_failure_cap,
            },
        )

    def check_confidence_threshold(self, state: AgentState) -> EscalationDecision:
        """Escalate when agent/reviewer confidence drops below the safety threshold."""
        score = float(state["confidence_score"])
        if score < self.confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                reason="low_confidence",
                severity="medium",
                recommended_tier="modify_plan",
                details=(
                    f"Confidence score {score:.2f} is below threshold "
                    f"{self.confidence_threshold:.2f}."
                ),
                metrics={
                    "confidence_score": score,
                    "confidence_threshold": self.confidence_threshold,
                },
            )

        return EscalationDecision(
            should_escalate=False,
            details=(
                f"Confidence score {score:.2f} meets threshold "
                f"{self.confidence_threshold:.2f}."
            ),
            metrics={
                "confidence_score": score,
                "confidence_threshold": self.confidence_threshold,
            },
        )

    def evaluate(
        self,
        state: AgentState,
        pending_tool_name: str | None = None,
        pending_tool_args: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> EscalationDecision:
        """Run all HITL triggers in priority order and return the first escalation."""
        if pending_tool_name is not None:
            action_decision = self.check_sensitive_action(
                tool_name=pending_tool_name,
                arguments=pending_tool_args or {},
                registry=registry,
            )
            if action_decision.should_escalate:
                return action_decision

        failure_decision = self.check_consecutive_failures(state)
        if failure_decision.should_escalate:
            return failure_decision

        confidence_decision = self.check_confidence_threshold(state)
        if confidence_decision.should_escalate:
            return confidence_decision

        return EscalationDecision(
            should_escalate=False,
            details="All HITL guardrails passed.",
        )