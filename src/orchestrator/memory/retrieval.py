import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, String, Text, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.orchestrator.memory.long_term_memory import LongTermMemoryStore


class PreferenceBase(DeclarativeBase):
    """Declarative base for persistent user preferences."""


class UserPreferenceRecord(PreferenceBase):
    """Key-value user preference row stored in PostgreSQL / SQLite."""

    __tablename__ = "user_preferences"

    pref_id: Mapped[str] = mapped_column(String(192), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    pref_key: Mapped[str] = mapped_column(String(64), nullable=False)
    pref_value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class RetrievedMemoryContext(BaseModel):
    """Structured retrieval payload injected into Supervisor planning prompts."""

    user_id: str
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    past_strategies: list[dict[str, Any]] = Field(default_factory=list)
    tool_precedents: list[dict[str, Any]] = Field(default_factory=list)
    formatted_prompt_block: str = ""


class MemoryRetriever:
    """Retrieves quality-gated past strategies and user preferences for planning."""

    def __init__(
        self,
        long_term_memory: LongTermMemoryStore,
        min_similarity: float = 0.15,
        max_prompt_chars: int = 1200,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0.")
        if max_prompt_chars < 100:
            raise ValueError("max_prompt_chars must be at least 100.")

        self._ltm = long_term_memory
        self._engine: Engine = long_term_memory._engine
        self.min_similarity = min_similarity
        self.max_prompt_chars = max_prompt_chars
        PreferenceBase.metadata.create_all(self._engine)

    def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """Upsert a persistent preference (e.g., tone, format, cost focus) for a user."""
        pref_id = f"{user_id}:{key}"
        with Session(self._engine) as session:
            existing = session.get(UserPreferenceRecord, pref_id)
            serialized = json.dumps(value, default=str)
            now = datetime.now(timezone.utc)
            if existing is not None:
                existing.pref_value_json = serialized
                existing.updated_at = now
            else:
                session.add(
                    UserPreferenceRecord(
                        pref_id=pref_id,
                        user_id=user_id,
                        pref_key=key,
                        pref_value_json=serialized,
                        updated_at=now,
                    )
                )
            session.commit()

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Return all stored preferences for a given user_id."""
        with Session(self._engine) as session:
            stmt = select(UserPreferenceRecord).where(
                UserPreferenceRecord.user_id == user_id
            )
            rows = session.scalars(stmt).all()
            return {row.pref_key: json.loads(row.pref_value_json) for row in rows}

    def retrieve_for_planning(
        self,
        user_request: str,
        user_id: str = "default",
        top_k: int = 3,
    ) -> RetrievedMemoryContext:
        """Gather user preferences, approved past strategies, and tool precedents."""
        preferences = self.get_user_preferences(user_id)

        # Bound top_k by available stored tasks to avoid Chroma over-request warnings
        known_tasks = len(self._ltm.list_recent_tasks(limit=100))
        strategy_k = min(top_k, known_tasks)

        past_strategies: list[dict[str, Any]] = []
        if strategy_k > 0:
            raw_tasks = self._ltm.search_semantic_memory(
                query=user_request,
                top_k=strategy_k,
                doc_type="task_summary",
            )
            # Quality gate: inject ONLY approved past strategies above similarity floor
            past_strategies = [
                hit
                for hit in raw_tasks
                if hit["metadata"].get("outcome_status") == "approved"
                and hit["similarity_score"] >= self.min_similarity
            ]

        tool_precedents: list[dict[str, Any]] = []
        if known_tasks > 0:
            raw_tools = self._ltm.search_semantic_memory(
                query=user_request,
                top_k=min(top_k, known_tasks * 3),
                doc_type="tool_decision",
            )
            tool_precedents = [
                hit
                for hit in raw_tools
                if hit["metadata"].get("status") == "success"
                and hit["similarity_score"] >= self.min_similarity
            ][:top_k]

        prompt_block = self._format_prompt_block(
            preferences=preferences,
            past_strategies=past_strategies,
            tool_precedents=tool_precedents,
        )

        return RetrievedMemoryContext(
            user_id=user_id,
            user_preferences=preferences,
            past_strategies=past_strategies,
            tool_precedents=tool_precedents,
            formatted_prompt_block=prompt_block,
        )

    def _format_prompt_block(
        self,
        preferences: dict[str, Any],
        past_strategies: list[dict[str, Any]],
        tool_precedents: list[dict[str, Any]],
    ) -> str:
        """Render a clean, budget-capped context block for the Supervisor prompt."""
        sections: list[str] = []

        if preferences:
            pref_lines = [f"- {k}: {v}" for k, v in sorted(preferences.items())]
            sections.append("USER PREFERENCES:\n" + "\n".join(pref_lines))

        if past_strategies:
            strat_lines = [
                f"- (sim={hit['similarity_score']:.2f}, task={hit['metadata'].get('task_id')}) "
                f"{hit['document']}"
                for hit in past_strategies
            ]
            sections.append("PAST SUCCESSFUL STRATEGIES:\n" + "\n".join(strat_lines))

        if tool_precedents:
            tool_lines = [
                f"- (tool={hit['metadata'].get('tool_name')}, sim={hit['similarity_score']:.2f}) "
                f"{hit['document']}"
                for hit in tool_precedents
            ]
            sections.append("RELEVANT TOOL PRECEDENTS:\n" + "\n".join(tool_lines))

        if not sections:
            return ""

        full_text = "HISTORICAL MEMORY CONTEXT (use to guide planning):\n\n" + "\n\n".join(sections)
        if len(full_text) <= self.max_prompt_chars:
            return full_text

        suffix = "\n... [memory truncated to fit prompt budget]"
        return full_text[: self.max_prompt_chars - len(suffix)] + suffix