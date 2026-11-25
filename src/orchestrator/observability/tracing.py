import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode


def _serialize_span(span: ReadableSpan) -> dict[str, Any]:
    """Convert an OpenTelemetry ReadableSpan into a clean JSON-serializable dict."""
    ctx = span.context
    parent_ctx = span.parent
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns
    duration_ms = round(max(0, end_ns - start_ns) / 1_000_000, 3)

    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else "",
        "span_id": format(ctx.span_id, "016x") if ctx else "",
        "parent_span_id": format(parent_ctx.span_id, "016x") if parent_ctx else None,
        "status": span.status.status_code.name,
        "status_description": span.status.description or "",
        "duration_ms": duration_ms,
        "attributes": dict(span.attributes) if span.attributes else {},
        "events": [
            {
                "name": ev.name,
                "attributes": dict(ev.attributes) if ev.attributes else {},
            }
            for ev in span.events
        ],
    }


class ExecutionTracer:
    """Isolated OpenTelemetry tracer capturing hierarchical agent & tool spans.

    Uses a dedicated TracerProvider (not the global singleton) so multiple tests
    and runtimes remain isolated, with SimpleSpanProcessor for synchronous flushing.
    """

    def __init__(
        self,
        service_name: str = "agent-orchestrator",
        console_export: bool = False,
    ) -> None:
        self.service_name = service_name
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        if console_export:
            self._provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        self._tracer = self._provider.get_tracer(service_name)

    @contextmanager
    def start_run_span(
        self,
        task_id: str,
        user_id: str,
        user_request: str,
    ) -> Iterator[Span]:
        """Root span wrapping an entire orchestration graph invocation."""
        with self._tracer.start_as_current_span("orchestrator.run") as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("user.id", user_id)
            span.set_attribute("user.request", user_request[:200])
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    @contextmanager
    def start_node_span(
        self,
        node_name: str,
        state: dict[str, Any],
    ) -> Iterator[Span]:
        """Child span wrapping one LangGraph node execution."""
        with self._tracer.start_as_current_span(f"node.{node_name}") as span:
            span.set_attribute("task.id", str(state.get("task_id", "")))
            span.set_attribute("node.name", node_name)
            span.set_attribute(
                "plan.step_index",
                int(state.get("current_step_index", 0)),
            )
            if state.get("assigned_specialist"):
                span.set_attribute(
                    "specialist.role",
                    str(state["assigned_specialist"]),
                )
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    @contextmanager
    def start_tool_span(
        self,
        tool_name: str,
        argument_keys: list[str] | None = None,
    ) -> Iterator[Span]:
        """Grandchild span wrapping a single sandboxed ToolRegistry invocation."""
        with self._tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            if argument_keys:
                span.set_attribute("tool.arg_keys", ",".join(sorted(argument_keys)))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def get_finished_spans(
        self,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return finished spans as dicts, optionally filtered by task_id."""
        serialized = [_serialize_span(s) for s in self._exporter.get_finished_spans()]
        if task_id is None:
            return serialized

        # Match spans carrying task.id directly OR sharing a trace_id with that task
        matching_traces = {
            s["trace_id"]
            for s in serialized
            if s["attributes"].get("task.id") == task_id
        }
        return [
            s
            for s in serialized
            if s["attributes"].get("task.id") == task_id
            or s["trace_id"] in matching_traces
        ]

    def get_trace_tree(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """Assemble spans into a nested parent -> children tree for inspection."""
        spans = self.get_finished_spans(task_id=task_id)
        nodes_by_id: dict[str, dict[str, Any]] = {
            s["span_id"]: {**s, "children": []} for s in spans
        }
        roots: list[dict[str, Any]] = []
        for node in nodes_by_id.values():
            parent_id = node["parent_span_id"]
            if parent_id and parent_id in nodes_by_id:
                nodes_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def export_trace_json(self, task_id: str | None = None) -> str:
        """Export the structured trace tree as formatted JSON (ready for replay/audit)."""
        return json.dumps(self.get_trace_tree(task_id=task_id), indent=2)

    def clear(self) -> None:
        """Clear all recorded in-memory spans."""
        self._exporter.clear()


def run_traced_graph(
    graph: Any,
    state: dict[str, Any],
    tracer: ExecutionTracer,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a compiled LangGraph state machine inside an OpenTelemetry root span."""
    task_id = str(state.get("task_id", "unknown"))
    user_id = str(state.get("user_id", "default"))
    user_request = str(state.get("user_request", ""))

    with tracer.start_run_span(task_id, user_id, user_request) as root_span:
        if config is not None:
            result = dict(graph.invoke(state, config=config))
        else:
            result = dict(graph.invoke(state))

        root_span.set_attribute("review.status", str(result.get("review_status", "")))
        root_span.set_attribute(
            "confidence.score",
            float(result.get("confidence_score", 0.0)),
        )
        root_span.set_attribute(
            "subtask.count",
            len(result.get("subtask_results", [])),
        )
        if result.get("review_status") == "failed":
            root_span.set_status(
                Status(StatusCode.ERROR, str(result.get("review_feedback", "Task failed")))
            )
        else:
            root_span.set_status(Status(StatusCode.OK))
        return result