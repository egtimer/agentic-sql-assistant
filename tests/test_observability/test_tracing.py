"""Tracing: no-op default, nesting through the context variable, and the real Langfuse SDK
objects (events are captured before they would be sent, so nothing touches the network)."""

from datetime import UTC, datetime
from typing import Any

import pytest

from src.observability import tracing
from src.observability.langfuse_setup import trace_node
from src.observability.tracing import LangfuseTracer, NoopTracer, jsonable
from tests.test_observability.helpers import of_type


def test_default_without_keys_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "")
    previous = tracing.set_tracer(None)
    try:
        assert isinstance(tracing.get_tracer(), NoopTracer)
        assert not tracing.enabled()
        with tracing.trace("t") as handle:
            assert handle.id is None
            with tracing.span("s") as inner:
                inner.update(output={"x": 1})
            tracing.record_generation("g", model="m", input="i", output="o")
            assert not tracing.active()
        tracing.score(None, "x", 1.0)
        tracing.flush()
    finally:
        tracing.set_tracer(previous)


def test_span_without_trace_does_nothing(events: list[dict[str, Any]]) -> None:
    with tracing.span("orphan") as handle:
        handle.update(output="x")
    tracing.record_generation("g", model="m", input="i", output="o")
    assert events == []


def test_nested_spans_generation_and_score(events: list[dict[str, Any]]) -> None:
    with tracing.trace(
        "root", input={"q": 1}, tags=["a", "b"], session_id="run-1", user_id="u"
    ) as root:
        assert root.id is not None and tracing.active()
        with tracing.span("node", input={"sql": "SELECT 1"}) as node:
            tracing.record_generation(
                "generate",
                model="m",
                input=[{"role": "user", "content": "hi"}],
                output="SELECT 1",
                usage={"input": 10, "output": 2, "cache_read": 5, "cache_write": 0},
                cost_usd=0.001,
                start=datetime.now(UTC),
                metadata={"stop_reason": "end_turn"},
            )
            node.update(output={"rows": 1})
        root.update(output={"done": True})
        tracing.score(root.id, "execution_accuracy", 1.0, comment="ok")
    assert not tracing.active()

    trace_body, trace_update = of_type(events, "trace-create")  # create, then upsert of output
    assert trace_update["id"] == root.id and trace_update["output"] == {"done": True}
    assert trace_body["id"] == root.id
    assert trace_body["sessionId"] == "run-1" and trace_body["tags"] == ["a", "b"]
    assert trace_body["userId"] == "u" and trace_body["name"] == "root"
    (span_body,) = of_type(events, "span-create")
    assert span_body["traceId"] == root.id and span_body["input"] == {"sql": "SELECT 1"}
    assert span_body.get("parentObservationId") is None
    (end_body,) = of_type(events, "span-update")
    assert end_body["id"] == span_body["id"] and end_body["output"] == {"rows": 1}
    assert end_body["endTime"] is not None
    (gen,) = of_type(events, "generation-create")
    assert gen["parentObservationId"] == span_body["id"] and gen["traceId"] == root.id
    assert gen["usageDetails"] == {"input": 10, "output": 2, "cache_read": 5, "cache_write": 0}
    assert gen["costDetails"] == {"total": 0.001}
    assert gen["model"] == "m" and gen["output"] == "SELECT 1"
    (sc,) = of_type(events, "score-create")
    assert sc["traceId"] == root.id and sc["name"] == "execution_accuracy" and sc["value"] == 1.0


def test_span_error_sets_level_and_reraises(events: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="boom"), tracing.trace("root"), tracing.span("node"):
        raise ValueError("boom")
    (end_body,) = of_type(events, "span-update")
    assert end_body["level"] == "ERROR"
    assert end_body["statusMessage"] == "ValueError: boom"
    assert any("boom" in str(b.get("metadata")) for b in of_type(events, "trace-create"))


def test_trace_or_span_nests_when_active(events: list[dict[str, Any]]) -> None:
    with tracing.trace_or_span("alone"), tracing.trace_or_span("inner"):
        pass
    assert [b["name"] for b in of_type(events, "trace-create")][:1] == ["alone"]
    assert [b["name"] for b in of_type(events, "span-create")] == ["inner"]


def test_tracer_failures_never_break_the_caller() -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("langfuse is down")

    with tracing.use_tracer(LangfuseTracer(Broken())):
        with tracing.trace("t") as handle:
            assert handle.id is None
            with tracing.span("s"):
                pass
            tracing.record_generation("g", model="m", input="i", output="o")
        tracing.score("id", "x", 1.0)
        tracing.flush()


def test_trace_node_records_span_for_graph_node(events: list[dict[str, Any]]) -> None:
    from pydantic import BaseModel

    class State(BaseModel):
        query: str
        messages: list[str] = []

    @trace_node("reason")
    def node(state: State) -> dict[str, Any]:
        return {"planned": [state.query]}

    with tracing.trace("root"):
        assert node(State(query="q")) == {"planned": ["q"]}
    (span_body,) = of_type(events, "span-create")
    assert span_body["name"] == "reason" and span_body["input"]["query"] == "q"
    assert "messages" not in span_body["input"]
    assert of_type(events, "span-update")[0]["output"] == {"planned": ["q"]}


def test_trace_node_is_plain_call_without_trace() -> None:
    calls: list[int] = []

    @trace_node("x")
    def node(state: object) -> int:
        calls.append(1)
        return 7

    assert node(object()) == 7 and calls == [1]


def test_jsonable_handles_messages_models_and_depth() -> None:
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    out = jsonable({"m": HumanMessage(content="hi"), "p": M(a=1), "s": {1, 2}, "x": object()})
    assert out["m"] == {"type": "human", "content": "hi"}
    assert out["p"] == {"a": 1}
    assert sorted(out["s"]) == [1, 2]
    assert isinstance(out["x"], str)
    assert jsonable("a" * 30_000, max_str=10) == "a" * 10 + "..."
