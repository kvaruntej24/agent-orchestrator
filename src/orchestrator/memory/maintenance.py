from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.orchestrator.agents.state import AgentState
from src.orchestrator.memory.long_term_memory import (
    LongTermMemoryStore,
    TaskMemoryRecord,
)
from src.orchestrator.memory.retrieval import PreferenceBase, UserPreferenceRecord
from src.orchestrator.memory.working_memory import WorkingMemoryStore


def compute_importance_score(state: AgentState) -> float:
    """Compute a deterministic [0.05, 1.0] importance score for a completed task.

    Weights four signals:
      - Outcome quality (approved tasks score highest)
      - Reviewer confidence score
      - Execution complexity (distinct tools invoked)
      - Self-healing / revision precedent (recovered retries or review cycles)
    """
    outcome_weights = {
        "approved": 0.45,
        "rejected": 0.20,
        "failed": 0.15,
        "pending": 0.10,
    }
    base = outcome_weights.get(state["review_status"], 0.10)
    confidence_term = 0.25 * max(0.0, min(1.0, float(state["confidence_score"])))

    unique_tools = {
        str(r.get("tool"))
        for r in state["subtask_results"]
        if r.get("tool")
    }
    complexity_term = min(0.20, len(unique_tools) * 0.06)

    had_recovery = state["review_status"] == "approved" and (
        state["review_cycles"] > 0
        or any(r.get("status") == "error" for r in state["subtask_results"])
    )
    recovery_bonus = 0.10 if had_recovery else 0.0

    raw_score = base + confidence_term + complexity_term + recovery_bonus
    return round(max(0.05, min(1.0, raw_score)), 4)


class ConsolidationReport(BaseModel):
    """Result summary returned after consolidating low-importance memories."""

    consolidated: bool
    consolidated_task_id: str | None = None
    tasks_merged: int = 0
    vectors_pruned: int = 0
    summary_preview: str = ""


class PurgeReport(BaseModel):
    """Audit summary of records deleted across Redis, PostgreSQL, and ChromaDB."""

    scope: str
    target_id: str
    redis_keys_deleted: int = 0
    relational_tasks_deleted: int = 0
    preferences_deleted: int = 0
    chroma_vectors_deleted: int = 0
    purged_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MemoryMaintenanceService:
    """Handles importance scoring, memory consolidation, and cross-store privacy purges."""

    def __init__(
        self,
        long_term_memory: LongTermMemoryStore,
        working_memory: WorkingMemoryStore | None = None,
    ) -> None:
        self._ltm = long_term_memory
        self._wm = working_memory
        # Ensure user_preferences table exists even if MemoryRetriever was not instantiated
        PreferenceBase.metadata.create_all(self._ltm._engine)

    def update_task_importance(self, task_id: str, new_score: float) -> bool:
        """Update importance_score across both PostgreSQL/SQLite and ChromaDB."""
        clamped = round(max(0.0, min(1.0, float(new_score))), 4)
        with Session(self._ltm._engine) as session:
            record = session.get(TaskMemoryRecord, task_id)
            if record is None:
                return False
            record.importance_score = clamped
            session.commit()

        vec_id = f"task:{task_id}"
        existing_vec = self._ltm._collection.get(ids=[vec_id])
        if existing_vec["ids"] and existing_vec["metadatas"]:
            updated_meta = dict(existing_vec["metadatas"][0])
            updated_meta["importance_score"] = clamped
            self._ltm._collection.update(ids=[vec_id], metadatas=[updated_meta])
        return True

    def consolidate_low_importance_memories(
        self,
        importance_threshold: float = 0.45,
        min_batch_size: int = 2,
    ) -> ConsolidationReport:
        """Roll up low-importance tasks into one compact summary and prune noisy vectors."""
        with Session(self._ltm._engine) as session:
            stmt = (
                select(TaskMemoryRecord)
                .where(
                    TaskMemoryRecord.importance_score < importance_threshold,
                    TaskMemoryRecord.outcome_status != "consolidated",
                )
                .order_by(TaskMemoryRecord.created_at.asc())
            )
            candidates = list(session.scalars(stmt).all())

            if len(candidates) < min_batch_size:
                return ConsolidationReport(consolidated=False)

            merged_ids = [row.task_id for row in candidates]
            snippets = [
                f"[{row.task_id} | {row.outcome_status}] {row.user_request}"
                for row in candidates
            ]
            consolidated_id = (
                f"consolidated:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            )
            rollup_summary = (
                f"Consolidated archive of {len(candidates)} low-importance runs: "
                + "; ".join(snippets)
            )
            avg_confidence = sum(r.confidence_score for r in candidates) / len(candidates)

            # Remove noisy source rows (cascades tool_decisions automatically)
            for row in candidates:
                session.delete(row)
            session.flush()

            # Persist single compact consolidated record
            session.add(
                TaskMemoryRecord(
                    task_id=consolidated_id,
                    user_request=f"Consolidated rollup ({len(candidates)} tasks)",
                    summary=rollup_summary,
                    outcome_status="consolidated",
                    confidence_score=round(avg_confidence, 4),
                    importance_score=round(importance_threshold, 4),
                    tools_used_json="[]",
                )
            )
            session.commit()

        # Prune old granular vectors from ChromaDB and upsert single consolidated vector
        vectors_pruned = 0
        for old_task_id in merged_ids:
            vectors_pruned += self._delete_chroma_vectors_for_task(old_task_id)

        self._ltm._collection.upsert(
            ids=[f"task:{consolidated_id}"],
            documents=[rollup_summary],
            embeddings=[self._ltm._embedder.embed(rollup_summary)],
            metadatas=[
                {
                    "task_id": consolidated_id,
                    "doc_type": "task_summary",
                    "outcome_status": "consolidated",
                    "importance_score": round(importance_threshold, 4),
                }
            ],
        )

        return ConsolidationReport(
            consolidated=True,
            consolidated_task_id=consolidated_id,
            tasks_merged=len(merged_ids),
            vectors_pruned=vectors_pruned,
            summary_preview=rollup_summary[:220],
        )

    def purge_task(self, task_id: str) -> PurgeReport:
        """Completely delete one task across Redis, PostgreSQL/SQLite, and ChromaDB."""
        redis_deleted = self._wm.clear_task(task_id) if self._wm is not None else 0

        with Session(self._ltm._engine) as session:
            record = session.get(TaskMemoryRecord, task_id)
            relational_deleted = 0
            if record is not None:
                session.delete(record)
                session.commit()
                relational_deleted = 1

        chroma_deleted = self._delete_chroma_vectors_for_task(task_id)

        return PurgeReport(
            scope="task",
            target_id=task_id,
            redis_keys_deleted=redis_deleted,
            relational_tasks_deleted=relational_deleted,
            chroma_vectors_deleted=chroma_deleted,
        )

    def purge_user_data(
        self,
        user_id: str,
        task_ids: list[str] | None = None,
    ) -> PurgeReport:
        """Privacy/GDPR purge: erase a user's preferences and any specified tasks."""
        with Session(self._ltm._engine) as session:
            stmt = delete(UserPreferenceRecord).where(
                UserPreferenceRecord.user_id == user_id
            )
            result = session.execute(stmt)
            prefs_deleted = int(result.rowcount or 0)
            session.commit()

        redis_deleted = 0
        relational_deleted = 0
        chroma_deleted = 0
        for tid in task_ids or []:
            task_report = self.purge_task(tid)
            redis_deleted += task_report.redis_keys_deleted
            relational_deleted += task_report.relational_tasks_deleted
            chroma_deleted += task_report.chroma_vectors_deleted

        return PurgeReport(
            scope="user",
            target_id=user_id,
            redis_keys_deleted=redis_deleted,
            relational_tasks_deleted=relational_deleted,
            preferences_deleted=prefs_deleted,
            chroma_vectors_deleted=chroma_deleted,
        )

    def _delete_chroma_vectors_for_task(self, task_id: str) -> int:
        """Safely find and delete all ChromaDB vectors tagged with task_id."""
        matched = self._ltm._collection.get(where={"task_id": task_id})
        matched_ids = matched.get("ids", [])
        if not matched_ids:
            return 0
        self._ltm._collection.delete(ids=matched_ids)
        return len(matched_ids)