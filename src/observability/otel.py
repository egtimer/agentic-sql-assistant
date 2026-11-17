"""OpenTelemetry tracer that follows the GenAI semantic conventions (ADR-012).

It implements the same :class:`~src.observability.tracing.Tracer` interface as the Langfuse
tracer, so the agents call one API and both backends receive the run (``tracing.MultiTracer``).
Spans:

- ``invoke_agent {agent}`` for an agent run (``agent_query`` is agent v1, ``sql_agent_v2``
  agent v2): ``gen_ai.operation.name = invoke_agent``, ``gen_ai.agent.name``.
- ``chat {model}`` for each LLM call: ``gen_ai.operation.name = chat``,
  ``gen_ai.provider.name``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` (every
  prompt token, cached or not), ``gen_ai.usage.output_tokens``,
  ``gen_ai.usage.cache_read.input_tokens``, ``gen_ai.usage.cache_creation.input_tokens``,
  ``gen_ai.response.finish_reasons``. The conventions have no cost attribute, so the cost the
  run accounts goes in ``app.llm.cost_usd``, and the call's purpose in ``app.llm.tag``.
- ``execute_tool {tool}`` for a span named ``tool:{tool}``: ``gen_ai.operation.name =
  execute_tool``, ``gen_ai.tool.name``.
- Any other span (graph nodes, ``recall``, eval roots) keeps its name, with
  ``app.span.kind = internal``.

Prompts, answers and tool inputs are not recorded: the conventions make message content
opt-in, and the content already goes to Langfuse, which keeps it on the owner's server. A span
whose observation fails gets status ERROR and ``error.type``. Conventions version: the
``gen_ai`` attributes of ``opentelemetry-semantic-conventions`` 0.66b1 (incubating).

Export is OTLP over HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT`` when that variable is set
(``build_from_env``); without it no OTel tracer is built. Tests pass their own exporter.
"""

import contextlib
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

SERVICE_NAME = "agentic-sql-assistant"
ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
AGENT_NAMES = {"agent_query": "agent_v1", "sql_agent_v2": "agent_v2"}
TOOL_PREFIX = "tool:"

OPERATION = "gen_ai.operation.name"
AGENT_NAME = "gen_ai.agent.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
CACHE_CREATION = "gen_ai.usage.cache_creation.input_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"
TOOL_NAME = "gen_ai.tool.name"
ERROR_TYPE = "error.type"
COST = "app.llm.cost_usd"
TAG = "app.llm.tag"
SPAN_KIND = "app.span.kind"
SESSION = "session.id"
USER = "user.id"


def _ns(moment: datetime | None) -> int | None:
    return None if moment is None else int(moment.timestamp() * 1e9)


def span_name_and_attributes(name: str) -> tuple[str, dict[str, Any]]:
    """The OTel span name and GenAI attributes for an observation called ``name``."""
    if name in AGENT_NAMES:
        agent = AGENT_NAMES[name]
        return f"invoke_agent {agent}", {OPERATION: "invoke_agent", AGENT_NAME: agent}
    if name.startswith(TOOL_PREFIX):
        tool = name[len(TOOL_PREFIX) :]
        return f"execute_tool {tool}", {OPERATION: "execute_tool", TOOL_NAME: tool}
    return name, {SPAN_KIND: "internal"}


class _OtelHandle:
    """Handle over one OTel span: ``update`` maps a level or status message to the status."""

    id = None  # scores go to Langfuse trace ids; OTel has none to offer here

    def __init__(self, span: Span) -> None:
        self._span = span

    def update(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        try:
            if level == "ERROR" or (status_message and level != "WARNING"):
                self._span.set_status(Status(StatusCode.ERROR, status_message))
                if status_message:
                    self._span.set_attribute(ERROR_TYPE, status_message.split(":", 1)[0])
        except Exception as e:  # tracing never breaks a request
            logger.warning("otel: span update failed: %s", e)


class OtelTracer:
    """A :class:`~src.observability.tracing.Tracer` that emits GenAI-convention OTel spans."""

    enabled = True

    def __init__(self, provider: TracerProvider) -> None:
        self._provider = provider
        self._tracer = provider.get_tracer("src.observability.otel")

    @contextlib.contextmanager
    def _span(
        self, name: str, *, root: bool, extra: Mapping[str, Any] | None = None
    ) -> Iterator[_OtelHandle]:
        span_name, attributes = span_name_and_attributes(name)
        attributes.update({k: v for k, v in (extra or {}).items() if v is not None})
        parent = otel_context.Context() if root else None
        with self._tracer.start_as_current_span(
            span_name, context=parent, attributes=attributes, record_exception=False
        ) as span:
            handle = _OtelHandle(span)
            try:
                yield handle
            except BaseException as e:
                span.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"))
                span.set_attribute(ERROR_TYPE, type(e).__name__)
                raise

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> contextlib.AbstractContextManager[_OtelHandle]:
        return self._span(name, root=True, extra={SESSION: session_id, USER: user_id})

    def span(
        self, name: str, *, input: Any = None, metadata: Mapping[str, Any] | None = None
    ) -> contextlib.AbstractContextManager[_OtelHandle]:
        return self._span(name, root=False)

    def record_generation(
        self,
        name: str,
        *,
        model: str | None,
        input: Any,
        output: Any,
        usage: Mapping[str, int] | None = None,
        cost_usd: float | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        provider: str | None = None,
    ) -> None:
        try:
            attributes: dict[str, Any] = {OPERATION: "chat", TAG: name}
            if model:
                attributes[REQUEST_MODEL] = model
            if provider:
                attributes[PROVIDER] = provider
            if usage is not None:
                uncached = int(usage.get("input", 0))
                read = int(usage.get("cache_read", 0))
                write = int(usage.get("cache_write", 0))
                attributes[INPUT_TOKENS] = uncached + read + write
                attributes[OUTPUT_TOKENS] = int(usage.get("output", 0))
                attributes[CACHE_READ] = read
                attributes[CACHE_CREATION] = write
            if cost_usd is not None:
                attributes[COST] = float(cost_usd)
            stop = (metadata or {}).get("stop_reason")
            if stop:
                attributes[FINISH_REASONS] = [str(stop)]
            span = self._tracer.start_span(
                f"chat {model}" if model else "chat",
                attributes=attributes,
                start_time=_ns(start),
            )
            if level == "ERROR":
                span.set_status(Status(StatusCode.ERROR, status_message))
                if status_message:
                    span.set_attribute(ERROR_TYPE, status_message.split(":", 1)[0])
            span.end(end_time=_ns(end))
        except Exception as e:
            logger.warning("otel: could not record generation %s: %s", name, e)

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None:
        return None  # scores live in Langfuse

    def flush(self) -> None:
        try:
            self._provider.force_flush()
        except Exception as e:
            logger.warning("otel: flush failed: %s", e)


def provider_with(processor: SpanProcessor) -> TracerProvider:
    """A tracer provider of this service that sends spans to ``processor`` (tests, export)."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(processor)
    return provider


def build_from_env() -> OtelTracer | None:
    """An OTLP/HTTP-exporting tracer when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, else None."""
    if not os.environ.get(ENDPOINT_ENV):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer = OtelTracer(provider_with(BatchSpanProcessor(OTLPSpanExporter())))
        logger.info("OpenTelemetry tracing enabled (%s)", os.environ[ENDPOINT_ENV])
        return tracer
    except Exception as e:
        logger.warning("OpenTelemetry could not be initialised: %s", e)
        return None


__all__ = [
    "AGENT_NAMES",
    "ENDPOINT_ENV",
    "TOOL_PREFIX",
    "OtelTracer",
    "build_from_env",
    "provider_with",
    "span_name_and_attributes",
]
