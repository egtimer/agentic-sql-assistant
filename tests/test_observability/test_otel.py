"""OpenTelemetry spans follow the GenAI conventions, nest like the Langfuse ones, and go to
both backends at once. In-memory exporter only: no collector, no network."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.agent.graph import run_agent
from src.llm import CallUsage, Completion, reset_client, set_client
from src.llm.client import trace_completion
from src.observability import otel, tracing
from src.sql_agent.graph import run_sql_agent
from tests.test_observability.helpers import of_type
from tests.test_observability.test_agent_traces import make_client


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    tracer = otel.OtelTracer(otel.provider_with(SimpleSpanProcessor(exp)))
    with tracing.use_tracer(tracer):
        yield exp


def by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans}


def parent_of(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> str | None:
    if span.parent is None:
        return None
    return next(s.name for s in spans if s.context.span_id == span.parent.span_id)


def make_db(tmp_path: Path) -> Path:
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 10) for i in range(1, 4)])
    conn.commit()
    conn.close()
    return path


def test_agent_v1_genai_spans(exporter: InMemorySpanExporter) -> None:
    client, _ = make_client(
        ['[{"tool": "calculator", "input": "1+1", "reason": "sum"}]', "The answer is 2."]
    )
    token = set_client(client)
    try:
        with tracing.trace("api.agent_query", session_id="s1", user_id="u1"):
            run_agent("what is 1+1?")
    finally:
        reset_client(token)
    spans = exporter.get_finished_spans()
    names = by_name(spans)
    root = names["api.agent_query"]
    assert root.parent is None and root.attributes["session.id"] == "s1"
    agent = names["invoke_agent agent_v1"]
    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["gen_ai.agent.name"] == "agent_v1"
    assert parent_of(agent, spans) == "api.agent_query"
    tool = names["execute_tool calculator"]
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "calculator"
    assert parent_of(tool, spans) == "execute"
    chats = [s for s in spans if s.name == "chat fake-model"]
    assert [parent_of(c, spans) for c in chats] == ["reason", "respond"]
    chat = chats[0].attributes
    assert chat["gen_ai.operation.name"] == "chat"
    assert chat["gen_ai.provider.name"] == "ollama"
    assert chat["gen_ai.request.model"] == "fake-model"
    assert chat["gen_ai.usage.input_tokens"] == 11 and chat["gen_ai.usage.output_tokens"] == 3
    assert chat["app.llm.tag"] == "reason" and "app.llm.cost_usd" in chat
    # no prompt or answer text on any span
    assert not any("1+1" in str(v) for s in spans for v in (s.attributes or {}).values())
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}


def test_agent_v2_tool_span_and_failed_query(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    client, _ = make_client(
        ["```sql\nSELECT nope FROM t\n```", "```sql\nSELECT v FROM t WHERE id = 2\n```", "OK"]
    )
    token = set_client(client)
    try:
        with tracing.trace("text2sql/agent_v2"):
            run_sql_agent(
                "value for id 2?",
                "",
                make_db(tmp_path),
                sample_values=False,
                skip_respond=True,
                use_check=False,
            )
    finally:
        reset_client(token)
    spans = exporter.get_finished_spans()
    tools = [s for s in spans if s.name == "execute_tool run_query"]
    assert tools and all(parent_of(t, spans) == "execute" for t in tools)
    assert tools[0].status.status_code is StatusCode.ERROR
    assert tools[0].attributes["error.type"] == "QueryError"
    assert parent_of(by_name(spans)["invoke_agent agent_v2"], spans) == "text2sql/agent_v2"


def test_cache_tokens_cost_and_errors(exporter: InMemorySpanExporter) -> None:
    from datetime import UTC, datetime

    usage = CallUsage(
        tag="generate",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=100,
        cache_write_tokens=20,
        cost_usd=0.0012,
    )
    usage.stop_reason = "end_turn"
    with tracing.trace("root"):
        trace_completion(
            "claude-haiku-5-5",
            "sys",
            "pre",
            "u",
            Completion(text="x", usage=usage),
            datetime.now(UTC),
            tag="generate",
            provider="anthropic",
        )
        trace_completion(
            "claude-haiku-5-5",
            "sys",
            None,
            "u",
            None,
            datetime.now(UTC),
            tag="check",
            error=RuntimeError("boom"),
            provider="anthropic",
        )
    ok, failed = [s for s in exporter.get_finished_spans() if s.name.startswith("chat")]
    a = ok.attributes
    assert a["gen_ai.usage.input_tokens"] == 130  # every prompt token, cached or not
    assert a["gen_ai.usage.cache_read.input_tokens"] == 100
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 20
    assert a["app.llm.cost_usd"] == pytest.approx(0.0012)
    assert a["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert a["gen_ai.provider.name"] == "anthropic"
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.attributes["error.type"] == "RuntimeError"


def test_no_spans_outside_a_trace(exporter: InMemorySpanExporter) -> None:
    with tracing.span("orphan"):
        pass
    tracing.record_generation("x", model="m", input=None, output=None)
    assert exporter.get_finished_spans() == ()


def test_exception_marks_span(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError), tracing.trace("root"), tracing.span("tool:sql_query"):
        raise ValueError("bad")
    tool = by_name(exporter.get_finished_spans())["execute_tool sql_query"]
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.attributes["error.type"] == "ValueError"


def test_multi_tracer_sends_to_both(events: list[dict[str, Any]]) -> None:
    langfuse = tracing.get_tracer()
    exp = InMemorySpanExporter()
    both = tracing.MultiTracer(
        [langfuse, otel.OtelTracer(otel.provider_with(SimpleSpanProcessor(exp)))]
    )
    client, _ = make_client(["[]", "Hello."])
    token = set_client(client)
    try:
        with tracing.use_tracer(both), tracing.trace("api.agent_query") as root:
            run_agent("hi")
    finally:
        reset_client(token)
    assert root.id is not None  # the Langfuse trace id, for scores
    langfuse_spans = [s["name"] for s in of_type(events, "span-create")]
    otel_spans = [s.name for s in exp.get_finished_spans()]
    assert "agent_query" in langfuse_spans and "invoke_agent agent_v1" in otel_spans
    assert len(of_type(events, "generation-create")) == sum(
        1 for n in otel_spans if n.startswith("chat ")
    )


def test_build_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(otel.ENDPOINT_ENV, raising=False)
    assert otel.build_from_env() is None
    monkeypatch.setenv(otel.ENDPOINT_ENV, "http://127.0.0.1:1")
    tracer = otel.build_from_env()
    assert isinstance(tracer, otel.OtelTracer)


def test_unknown_names_stay_internal() -> None:
    assert otel.span_name_and_attributes("check") == ("check", {"app.span.kind": "internal"})
    assert otel.span_name_and_attributes("sql_agent_v2")[0] == "invoke_agent agent_v2"
