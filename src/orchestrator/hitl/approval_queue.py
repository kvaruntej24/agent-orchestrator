import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import fakeredis
import redis
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from src.orchestrator.hitl.triggers import (
    ApprovalTier,
    EscalationDecision,
    EscalationReason,
    EscalationSeverity,
)

ReviewDecisionStatus = Literal["pending", "approved", "rejected"]


class HitlBase(DeclarativeBase):
    """Declarative base for durable HITL approval request records."""


class ApprovalRequestRecord(HitlBase):
    """Relational audit table storing every HITL escalation and human decision."""

    __tablename__ = "approval_requests"

    review_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    step_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    recommended_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_action_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="pending")
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    reviewer_comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )


class ReviewRequest(BaseModel):
    """Structured HITL review payload packaged into Redis and PostgreSQL."""

    review_id: str = Field(default_factory=lambda: f"rev-{uuid.uuid4().hex[:10]}")
    task_id: str
    thread_id: str
    step_id: str
    reason: EscalationReason
    severity: EscalationSeverity
    recommended_tier: ApprovalTier
    details: str
    proposed_action: dict[str, Any] = Field(default_factory=dict)
    status: ReviewDecisionStatus = "pending"
    reviewer: str = ""
    reviewer_comment: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = None

    @classmethod
    def from_escalation(
        cls,
        task_id: str,
        thread_id: str,
        step_id: str,
        decision: EscalationDecision,
    ) -> "ReviewRequest":
        """Build a ReviewRequest directly from an EscalationDecision."""
        if not decision.should_escalate or decision.reason is None:
            raise ValueError("Cannot create ReviewRequest from a non-escalating decision.")
        return cls(
            task_id=task_id,
            thread_id=thread_id,
            step_id=step_id,
            reason=decision.reason,
            severity=decision.severity,
            recommended_tier=decision.recommended_tier,
            details=decision.details,
            proposed_action=decision.proposed_action or {},
        )


class ApprovalQueue:
    """Dual-backed async approval queue using Redis (fast queue) and PostgreSQL (audit log)."""

    def __init__(
        self,
        engine: Engine,
        redis_client: redis.Redis,
        key_prefix: str = "orchestrator:hitl",
    ) -> None:
        self._engine = engine
        self._redis = redis_client
        self.key_prefix = key_prefix
        HitlBase.metadata.create_all(self._engine)

    @classmethod
    def from_urls(cls, postgres_dsn: str, redis_url: str) -> "ApprovalQueue":
        """Connect to live PostgreSQL and Redis servers."""
        engine = create_engine(postgres_dsn, pool_pre_ping=True)
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        return cls(engine=engine, redis_client=client)

    @classmethod
    def in_memory(cls) -> "ApprovalQueue":
        """Create an isolated SQLite + FakeRedis approval queue for offline dev/tests."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        client = fakeredis.FakeRedis(decode_responses=True)
        return cls(engine=engine, redis_client=client)

    def _pending_list_key(self) -> str:
        return f"{self.key_prefix}:pending_ids"

    def _request_key(self, review_id: str) -> str:
        return f"{self.key_prefix}:request:{review_id}"

    def enqueue(self, request: ReviewRequest) -> ReviewRequest:
        """Persist a new pending review payload into PostgreSQL and Redis."""
        with Session(self._engine) as session:
            session.add(
                ApprovalRequestRecord(
                    review_id=request.review_id,
                    task_id=request.task_id,
                    thread_id=request.thread_id,
                    step_id=request.step_id,
                    reason=request.reason,
                    severity=request.severity,
                    recommended_tier=request.recommended_tier,
                    details=request.details,
                    proposed_action_json=json.dumps(request.proposed_action, default=str),
                    status="pending",
                )
            )
            session.commit()

        payload = request.model_dump_json()
        with self._redis.pipeline() as pipe:
            pipe.set(self._request_key(request.review_id), payload)
            pipe.rpush(self._pending_list_key(), request.review_id)
            pipe.execute()

        return request

    def get(self, review_id: str) -> ReviewRequest | None:
        """Fetch a review payload (fast path from Redis, fallback to PostgreSQL)."""
        cached = self._redis.get(self._request_key(review_id))
        if cached is not None:
            return ReviewRequest.model_validate_json(cached)

        with Session(self._engine) as session:
            row = session.get(ApprovalRequestRecord, review_id)
            if row is None:
                return None
            return self._row_to_model(row)

    def list_pending(self) -> list[ReviewRequest]:
        """Return all pending review payloads in FIFO queue order."""
        pending_ids = self._redis.lrange(self._pending_list_key(), 0, -1)
        results: list[ReviewRequest] = []
        for review_id in pending_ids:
            item = self.get(review_id)
            if item is not None and item.status == "pending":
                results.append(item)
        if results:
            return results

        # Fallback to relational DB if Redis cache was flushed
        with Session(self._engine) as session:
            stmt = (
                select(ApprovalRequestRecord)
                .where(ApprovalRequestRecord.status == "pending")
                .order_by(ApprovalRequestRecord.created_at.asc())
            )
            return [self._row_to_model(row) for row in session.scalars(stmt).all()]

    def resolve(
        self,
        review_id: str,
        decision: Literal["approved", "rejected"],
        reviewer: str,
        comment: str = "",
    ) -> ReviewRequest:
        """Record a human approve/reject decision in PostgreSQL and update Redis."""
        if not reviewer.strip():
            raise ValueError("reviewer must be a non-empty string.")

        now = datetime.now(timezone.utc)
        with Session(self._engine) as session:
            row = session.get(ApprovalRequestRecord, review_id)
            if row is None:
                raise KeyError(f"Review request '{review_id}' not found.")
            if row.status != "pending":
                raise ValueError(
                    f"Review request '{review_id}' is already resolved ({row.status})."
                )
            row.status = decision
            row.reviewer = reviewer
            row.reviewer_comment = comment
            row.resolved_at = now
            session.commit()
            resolved = self._row_to_model(row)

        with self._redis.pipeline() as pipe:
            pipe.lrem(self._pending_list_key(), 0, review_id)
            pipe.set(self._request_key(review_id), resolved.model_dump_json())
            pipe.execute()

        return resolved

    @staticmethod
    def _row_to_model(row: ApprovalRequestRecord) -> ReviewRequest:
        return ReviewRequest(
            review_id=row.review_id,
            task_id=row.task_id,
            thread_id=row.thread_id,
            step_id=row.step_id,
            reason=row.reason,  # type: ignore[arg-type]
            severity=row.severity,  # type: ignore[arg-type]
            recommended_tier=row.recommended_tier,  # type: ignore[arg-type]
            details=row.details,
            proposed_action=json.loads(row.proposed_action_json),
            status=row.status,  # type: ignore[arg-type]
            reviewer=row.reviewer,
            reviewer_comment=row.reviewer_comment,
            created_at=row.created_at.isoformat(),
            resolved_at=row.resolved_at.isoformat() if row.resolved_at else None,
        )


def resume_after_human_decision(
    graph: Any,
    thread_id: str,
    approval_queue: ApprovalQueue,
    review_id: str,
    decision: Literal["approved", "rejected"],
    reviewer: str,
    comment: str = "",
) -> dict[str, Any]:
    """Resolve a pending HITL item, update the LangGraph checkpoint, and resume execution."""
    resolved = approval_queue.resolve(
        review_id=review_id,
        decision=decision,
        reviewer=reviewer,
        comment=comment,
    )
    config = {"configurable": {"thread_id": thread_id}}
    # Leave approval_status='pending' so route_after_specialist keeps targeting human_approval
    graph.update_state(
        config,
        {"human_decision": resolved.model_dump()},
    )
    return dict(graph.invoke(None, config=config))