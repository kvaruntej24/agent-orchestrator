import json
from datetime import datetime, timezone
from typing import Any

import fakeredis
import redis


class WorkingMemoryStore:
    """Redis-backed short-term working memory for active orchestration runs.

    Stores three namespaced Redis structures per task_id, each refreshed with a TTL:
      - {prefix}:{task_id}:state    -> JSON string of latest execution state snapshot
      - {prefix}:{task_id}:subtasks -> Redis List of ordered JSON subtask outputs
      - {prefix}:{task_id}:context  -> Redis Hash of intermediate key-value task context
    """

    def __init__(
        self,
        client: redis.Redis,
        ttl_seconds: int = 3600,
        key_prefix: str = "orchestrator:task",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer.")
        self._client = client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        ttl_seconds: int = 3600,
    ) -> "WorkingMemoryStore":
        """Connect to a live Redis server using a standard redis:// URL."""
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        return cls(client=client, ttl_seconds=ttl_seconds)

    @classmethod
    def in_memory(cls, ttl_seconds: int = 3600) -> "WorkingMemoryStore":
        """Create a Redis-protocol-compatible in-memory store for offline dev/tests."""
        client = fakeredis.FakeRedis(decode_responses=True)
        return cls(client=client, ttl_seconds=ttl_seconds)

    def _state_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:{task_id}:state"

    def _subtasks_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:{task_id}:subtasks"

    def _context_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:{task_id}:context"

    def health_check(self) -> bool:
        """Verify Redis connectivity."""
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def save_state(self, task_id: str, state_snapshot: dict[str, Any]) -> None:
        """Persist the latest ephemeral execution state snapshot with TTL."""
        payload = {
            **state_snapshot,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._client.setex(
            self._state_key(task_id),
            self.ttl_seconds,
            json.dumps(payload, default=str),
        )

    def get_state(self, task_id: str) -> dict[str, Any] | None:
        """Fetch the latest execution state snapshot, or None if missing/expired."""
        raw = self._client.get(self._state_key(task_id))
        if raw is None:
            return None
        return dict(json.loads(raw))

    def append_subtask_output(self, task_id: str, output: dict[str, Any]) -> int:
        """Append a subtask output to the task's Redis List and refresh TTL atomically."""
        key = self._subtasks_key(task_id)
        serialized = json.dumps(output, default=str)
        with self._client.pipeline() as pipe:
            pipe.rpush(key, serialized)
            pipe.expire(key, self.ttl_seconds)
            results = pipe.execute()
        return int(results[0])

    def get_subtask_outputs(self, task_id: str) -> list[dict[str, Any]]:
        """Return all recorded subtask outputs in chronological execution order."""
        raw_items = self._client.lrange(self._subtasks_key(task_id), 0, -1)
        return [dict(json.loads(item)) for item in raw_items]

    def set_context(self, task_id: str, field: str, value: Any) -> None:
        """Store an intermediate context value in the task's Redis Hash with TTL."""
        key = self._context_key(task_id)
        with self._client.pipeline() as pipe:
            pipe.hset(key, field, json.dumps(value, default=str))
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def get_context(self, task_id: str) -> dict[str, Any]:
        """Retrieve all intermediate context key-value pairs for a task."""
        raw_map = self._client.hgetall(self._context_key(task_id))
        return {k: json.loads(v) for k, v in raw_map.items()}

    def get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        """Return a unified view of state, context, subtask outputs, and remaining TTL."""
        return {
            "task_id": task_id,
            "state": self.get_state(task_id),
            "context": self.get_context(task_id),
            "subtask_outputs": self.get_subtask_outputs(task_id),
            "ttl_remaining_seconds": self._client.ttl(self._state_key(task_id)),
        }

    def clear_task(self, task_id: str) -> int:
        """Immediately purge all working memory keys for a task."""
        return int(
            self._client.delete(
                self._state_key(task_id),
                self._subtasks_key(task_id),
                self._context_key(task_id),
            )
        )