import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Protocol

import chromadb
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.pool import StaticPool

from src.orchestrator.agents.state import AgentState


# ---------- SQLAlchemy Relational Schema (PostgreSQL / SQLite) ----------

class Base(DeclarativeBase):
    """Declarative base for relational long-term memory tables."""


class TaskMemoryRecord(Base):
    """Structured record of a completed orchestration run."""

    __tablename__ = "task_memories"

    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_status: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    tools_used_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    tool_decisions: Mapped[list["ToolDecisionRecord"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
    )


class ToolDecisionRecord(Base):
    """Granular audit log of each tool invocation made during a task."""

    __tablename__ = "tool_decisions"

    decision_id: Mapped[str] = mapped_column(String(192), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("task_memories.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    step_id: Mapped[str] = mapped_column(String(64), nullable=False)
    specialist: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    task: Mapped[TaskMemoryRecord] = relationship(back_populates="tool_decisions")


# ---------- Pluggable Text Embedders ----------

class TextEmbedder(Protocol):
    """Protocol for converting text into a fixed-length float vector."""

    def embed(self, text: str) -> list[float]: ...


class LocalHashEmbedder:
    """Deterministic, zero-download feature-hashing embedder for offline dev and CI.

    Hashes unigrams and bigrams into a normalized dense vector so ChromaDB cosine
    similarity reliably ranks semantically overlapping tasks highest offline.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return [0.0] * self.dim

        features = list(tokens) + [
            f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
        ]
        vec = [0.0] * self.dim
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [round(v / norm, 6) for v in vec]


class OpenAIEmbedder:
    """Production embedder backed by OpenAI embedding models."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._embeddings = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._embeddings.embed_query(text)]


# ---------- Task Summarizer Helper ----------

def summarize_completed_task(state: AgentState) -> tuple[str, list[dict[str, Any]]]:
    """Build a concise searchable task summary and structured tool-decision list."""
    decisions: list[dict[str, Any]] = []
    tools_used: list[str] = []

    for idx, result in enumerate(state["subtask_results"]):
        tool_name = str(result.get("tool", "unknown"))
        specialist = str(result.get("specialist", "unknown"))
        step_id = str(result.get("step_id", f"step_{idx + 1}"))
        status = str(result.get("status", "unknown"))
        duration_ms = float(result.get("duration_ms", 0.0))

        if tool_name not in tools_used:
            tools_used.append(tool_name)

        if status == "success":
            snippet = str(result.get("output", ""))[:180].replace("\n", " ")
            rationale = (
                f"Specialist '{specialist}' ran '{tool_name}' on {step_id} "
                f"for request '{state['user_request']}' -> succeeded: {snippet}"
            )
        else:
            err = str(result.get("error", "unknown error"))
            rationale = (
                f"Specialist '{specialist}' ran '{tool_name}' on {step_id} "
                f"for request '{state['user_request']}' -> failed: {err}"
            )

        decisions.append(
            {
                "decision_id": f"{state['task_id']}:{step_id}:{idx}",
                "step_id": step_id,
                "specialist": specialist,
                "tool_name": tool_name,
                "status": status,
                "rationale": rationale,
                "duration_ms": duration_ms,
            }
        )

    final_excerpt = state["final_output"][:240].replace("\n", " ") if state["final_output"] else "No final output."
    summary = (
        f"Request: {state['user_request']} | "
        f"Outcome: {state['review_status']} (confidence={state['confidence_score']:.2f}) | "
        f"Tools: {', '.join(tools_used) or 'none'} | "
        f"Result: {final_excerpt}"
    )
    return summary, decisions


# ---------- Dual-Store Long-Term Semantic Memory ----------

class LongTermMemoryStore:
    """Coordinates PostgreSQL (structured audit truth) and ChromaDB (semantic vector index)."""

    def __init__(
        self,
        engine: Engine,
        chroma_client: Any,
        embedder: TextEmbedder | None = None,
        collection_name: str = "agent_semantic_memory",
    ) -> None:
        self._engine = engine
        self._embedder: TextEmbedder = embedder or LocalHashEmbedder()
        Base.metadata.create_all(self._engine)
        self._collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_dsn(
        cls,
        postgres_dsn: str,
        chroma_persist_dir: str = "chroma_data",
        embedder: TextEmbedder | None = None,
        collection_name: str = "agent_semantic_memory",
    ) -> "LongTermMemoryStore":
        """Connect to live PostgreSQL and persistent on-disk ChromaDB."""
        engine = create_engine(postgres_dsn, pool_pre_ping=True)
        chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        return cls(
            engine=engine,
            chroma_client=chroma_client,
            embedder=embedder,
            collection_name=collection_name,
        )

    @classmethod
    def in_memory(
        cls,
        embedder: TextEmbedder | None = None,
        collection_name: str = "agent_semantic_memory_test",
    ) -> "LongTermMemoryStore":
        """Create an isolated SQLite + ephemeral ChromaDB store for offline dev and tests."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        chroma_client = chromadb.EphemeralClient()
        # Reset collection if reused within the same Python process
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass
        return cls(
            engine=engine,
            chroma_client=chroma_client,
            embedder=embedder,
            collection_name=collection_name,
        )

    def record_completed_task(
        self,
        state: AgentState,
        importance_score: float = 0.5,
    ) -> dict[str, Any]:
        """Summarize a finished run and persist it across PostgreSQL and ChromaDB."""
        summary, decisions = summarize_completed_task(state)
        tools_used = sorted({d["tool_name"] for d in decisions})
        task_id = state["task_id"]

        # 1. Persist structured source of truth in relational DB
        with Session(self._engine) as session:
            existing = session.get(TaskMemoryRecord, task_id)
            if existing is not None:
                session.delete(existing)
                session.flush()

            record = TaskMemoryRecord(
                task_id=task_id,
                user_request=state["user_request"],
                summary=summary,
                outcome_status=state["review_status"],
                confidence_score=float(state["confidence_score"]),
                importance_score=float(importance_score),
                tools_used_json=json.dumps(tools_used),
            )
            for dec in decisions:
                record.tool_decisions.append(
                    ToolDecisionRecord(
                        decision_id=dec["decision_id"],
                        step_id=dec["step_id"],
                        specialist=dec["specialist"],
                        tool_name=dec["tool_name"],
                        status=dec["status"],
                        rationale=dec["rationale"],
                        duration_ms=dec["duration_ms"],
                    )
                )
            session.add(record)
            session.commit()

        # 2. Index task summary and tool decisions as vectors in ChromaDB
        ids = [f"task:{task_id}"]
        documents = [summary]
        embeddings = [self._embedder.embed(summary)]
        metadatas: list[dict[str, Any]] = [
            {
                "task_id": task_id,
                "doc_type": "task_summary",
                "outcome_status": state["review_status"],
                "importance_score": float(importance_score),
            }
        ]

        for dec in decisions:
            ids.append(f"tool:{dec['decision_id']}")
            documents.append(dec["rationale"])
            embeddings.append(self._embedder.embed(dec["rationale"]))
            metadatas.append(
                {
                    "task_id": task_id,
                    "doc_type": "tool_decision",
                    "specialist": dec["specialist"],
                    "tool_name": dec["tool_name"],
                    "status": dec["status"],
                }
            )

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "task_id": task_id,
            "summary": summary,
            "decisions_indexed": len(decisions),
            "vectors_upserted": len(ids),
        }

    def search_semantic_memory(
        self,
        query: str,
        top_k: int = 3,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve semantically similar past task summaries or tool decisions."""
        total_vectors = self._collection.count()
        if total_vectors == 0 or not query.strip():
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [self._embedder.embed(query)],
            "n_results": min(top_k, total_vectors),
        }
        if doc_type is not None:
            query_kwargs["where"] = {"doc_type": doc_type}

        raw = self._collection.query(**query_kwargs)
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        matches: list[dict[str, Any]] = []
        for vec_id, doc, meta, dist in zip(ids, docs, metas, distances):
            matches.append(
                {
                    "id": vec_id,
                    "document": doc,
                    "metadata": meta,
                    "similarity_score": round(max(0.0, 1.0 - float(dist)), 4),
                }
            )
        return matches

    def get_task_record(self, task_id: str) -> dict[str, Any] | None:
        """Fetch a structured task record and its tool decisions from PostgreSQL/SQLite."""
        with Session(self._engine) as session:
            record = session.get(TaskMemoryRecord, task_id)
            if record is None:
                return None
            return {
                "task_id": record.task_id,
                "user_request": record.user_request,
                "summary": record.summary,
                "outcome_status": record.outcome_status,
                "confidence_score": record.confidence_score,
                "importance_score": record.importance_score,
                "tools_used": json.loads(record.tools_used_json),
                "tool_decisions": [
                    {
                        "decision_id": d.decision_id,
                        "step_id": d.step_id,
                        "specialist": d.specialist,
                        "tool_name": d.tool_name,
                        "status": d.status,
                        "duration_ms": d.duration_ms,
                    }
                    for d in record.tool_decisions
                ],
            }

    def list_recent_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """List the most recent completed task summaries from relational storage."""
        with Session(self._engine) as session:
            stmt = (
                select(TaskMemoryRecord)
                .order_by(TaskMemoryRecord.created_at.desc())
                .limit(limit)
            )
            rows = session.scalars(stmt).all()
            return [
                {
                    "task_id": r.task_id,
                    "user_request": r.user_request,
                    "outcome_status": r.outcome_status,
                    "importance_score": r.importance_score,
                }
                for r in rows
            ]