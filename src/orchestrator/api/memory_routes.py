from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.orchestrator.memory.maintenance import (
    ConsolidationReport,
    MemoryMaintenanceService,
    PurgeReport,
)


class ConsolidateRequest(BaseModel):
    """Parameters for triggering low-importance memory consolidation."""

    importance_threshold: float = Field(default=0.45, ge=0.05, le=1.0)
    min_batch_size: int = Field(default=2, ge=1, le=100)


def create_memory_router(service: MemoryMaintenanceService) -> APIRouter:
    """Build a FastAPI router exposing memory consolidation and privacy purge endpoints."""
    router = APIRouter(prefix="/memory", tags=["memory-maintenance"])

    @router.post("/consolidate", response_model=ConsolidationReport)
    def consolidate_memories(payload: ConsolidateRequest) -> ConsolidationReport:
        """Consolidate low-importance tasks into a single rollup memory."""
        return service.consolidate_low_importance_memories(
            importance_threshold=payload.importance_threshold,
            min_batch_size=payload.min_batch_size,
        )

    @router.delete("/tasks/{task_id}", response_model=PurgeReport)
    def purge_task_memory(task_id: str) -> PurgeReport:
        """Delete a single task from Redis, PostgreSQL, and ChromaDB."""
        if not task_id.strip():
            raise HTTPException(status_code=400, detail="task_id must not be empty.")
        return service.purge_task(task_id)

    @router.delete("/users/{user_id}", response_model=PurgeReport)
    def purge_user_privacy_data(
        user_id: str,
        task_ids: list[str] | None = Query(default=None),
    ) -> PurgeReport:
        """GDPR/privacy endpoint erasing a user's preferences and associated tasks."""
        if not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id must not be empty.")
        return service.purge_user_data(user_id=user_id, task_ids=task_ids)

    return router