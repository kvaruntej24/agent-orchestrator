from collections import deque
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from src.orchestrator.config import get_settings

SpecialistRole = Literal["researcher", "analyst", "writer", "db_operator"]

PLANNER_SYSTEM_PROMPT = """You are a Supervisor AI planning an enterprise workflow.
Decompose the user's request into 1 to 6 clear, ordered subtasks.
Assign each subtask to exactly one specialist role:
- researcher: gathers facts, web/docs research
- db_operator: queries internal databases or records
- analyst: crunches numbers, compares findings, detects anomalies
- writer: drafts polished summaries, reports, or final deliverables

Rules:
- Every step_id must be unique (e.g., "step_1", "step_2").
- Use depends_on to reference earlier step_ids whose outputs are required.
- Never create circular dependencies.
"""


class Subtask(BaseModel):
    """A single atomic unit of work assigned to one specialist."""

    step_id: str = Field(..., min_length=1, description="Unique ID, e.g., 'step_1'")
    title: str = Field(..., min_length=3, description="Short human-readable task title")
    specialist: SpecialistRole = Field(..., description="Specialist role responsible")
    instruction: str = Field(..., min_length=5, description="Actionable instruction")
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step_ids that must finish before this subtask starts",
    )


class ExecutionPlan(BaseModel):
    """Validated, cycle-free execution plan produced by the Supervisor."""

    goal_summary: str = Field(..., min_length=5)
    subtasks: list[Subtask] = Field(..., min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_dag(self) -> "ExecutionPlan":
        """Ensure unique IDs, valid dependency references, and no cycles."""
        ids = [task.step_id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate step_id found in execution plan.")

        id_set = set(ids)
        for task in self.subtasks:
            for dep in task.depends_on:
                if dep == task.step_id:
                    raise ValueError(f"Subtask '{task.step_id}' cannot depend on itself.")
                if dep not in id_set:
                    raise ValueError(
                        f"Subtask '{task.step_id}' depends on unknown step_id '{dep}'."
                    )

        # Kahn's algorithm to detect cycles in the dependency graph
        in_degree: dict[str, int] = {task.step_id: len(task.depends_on) for task in self.subtasks}
        dependents: dict[str, list[str]] = {task.step_id: [] for task in self.subtasks}
        for task in self.subtasks:
            for dep in task.depends_on:
                dependents[dep].append(task.step_id)

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for nxt in dependents[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(self.subtasks):
            raise ValueError("Circular dependency detected in execution plan.")

        return self

    def ordered_subtasks(self) -> list[Subtask]:
        """Return subtasks sorted in valid topological execution order."""
        by_id = {task.step_id: task for task in self.subtasks}
        in_degree: dict[str, int] = {task.step_id: len(task.depends_on) for task in self.subtasks}
        dependents: dict[str, list[str]] = {task.step_id: [] for task in self.subtasks}
        for task in self.subtasks:
            for dep in task.depends_on:
                dependents[dep].append(task.step_id)

        queue = deque([task.step_id for task in self.subtasks if in_degree[task.step_id] == 0])
        ordered: list[Subtask] = []
        while queue:
            current = queue.popleft()
            ordered.append(by_id[current])
            for nxt in dependents[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return ordered


class TaskPlanner:
    """Decomposes a user request into a validated ExecutionPlan."""

    def __init__(self, model_name: str = "gpt-4o-mini", mock: bool = False) -> None:
        self.model_name = model_name
        self.mock = mock
        self._structured_llm: Any | None = None
        self.last_memory_context: str = ""

    def _get_structured_llm(self) -> Any:
        if self._structured_llm is None:
            settings = get_settings()
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
            self._structured_llm = llm.with_structured_output(ExecutionPlan)
        return self._structured_llm

    def create_plan(
        self,
        user_request: str,
        memory_context: str = "",
    ) -> ExecutionPlan:
        """Generate and validate an ExecutionPlan enriched with optional memory context."""
        if not user_request or not user_request.strip():
            raise ValueError("user_request must be a non-empty string.")

        self.last_memory_context = memory_context.strip()

        if self.mock:
            return self._mock_plan(user_request, self.last_memory_context)

        try:
            llm = self._get_structured_llm()
            messages: list[Any] = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)]
            if self.last_memory_context:
                messages.append(SystemMessage(content=self.last_memory_context))
            messages.append(HumanMessage(content=user_request))
            plan: ExecutionPlan = llm.invoke(messages)
            return plan
        except Exception as exc:
            raise RuntimeError(f"Task decomposition failed: {exc}") from exc

    @staticmethod
    def _mock_plan(user_request: str, memory_context: str = "") -> ExecutionPlan:
        """Deterministic multi-step plan for offline verification and unit tests."""
        goal_suffix = " (memory-informed)" if memory_context else ""
        return ExecutionPlan(
            goal_summary=f"Execute and verify request: {user_request}{goal_suffix}",
            subtasks=[
                Subtask(
                    step_id="step_1",
                    title="Gather relevant context",
                    specialist="researcher",
                    instruction=f"Collect background information for: {user_request}",
                    depends_on=[],
                ),
                Subtask(
                    step_id="step_2",
                    title="Analyze findings",
                    specialist="analyst",
                    instruction=f"Analyze gathered data and extract insights for: {user_request}",
                    depends_on=["step_1"],
                ),
                Subtask(
                    step_id="step_3",
                    title="Draft executive summary",
                    specialist="writer",
                    instruction=f"Write a clear final answer based on analysis for: {user_request}",
                    depends_on=["step_2"],
                ),
            ],
        )