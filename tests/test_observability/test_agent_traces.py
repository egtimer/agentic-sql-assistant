"""Both agents and the LLM client produce nested spans and generations in one trace."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import run_agent
from src.llm import LLMConfig, OllamaClient, UsageTracker, reset_client, set_client
from src.observability import tracing
from src.sql_agent.graph import run_sql_agent
from tests.test_observability.helpers import of_type


class FakeChat:
    """Stands in for ChatOllama: answers from a list, with usage metadata."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[Any] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.prompts.append(messages)
        text = self.answers.pop(0)
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
            response_metadata={"done_reason": "stop"},
        )


def make_client(answers: list[str]) -> tuple[OllamaClient, FakeChat]:
    config = LLMConfig(provider="ollama", model="fake-model")
    client = OllamaClient(config, UsageTracker())
    chat = FakeChat(answers)
    client._llm = chat  # type: ignore[assignment]
    return client, chat


def by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observations = of_type(events, "span-create") + of_type(events, "generation-create")
    return {o["id"]: o for o in observations}


def parent_name(events: list[dict[str, Any]], obs: dict[str, Any]) -> str | None:
    parent = obs.get("parentObservationId")
    return None if parent is None else by_id(events)[parent]["name"]


def test_agent_v1_spans_and_generations(events: list[dict[str, Any]]) -> None:
    client, _ = make_client(
        ['[{"tool": "calculator", "input": "1+1", "reason": "sum"}]', "The answer is 2."]
    )
    token = set_client(client)
    try:
        with tracing.trace("test") as root:
            result = run_agent("what is 1+1?")
    finally:
        reset_client(token)
    assert result["response"] == "The answer is 2."

    spans = of_type(events, "span-create")
    assert [s["name"] for s in spans] == ["agent_query", "reason", "execute", "validate", "respond"]
    assert all(s["traceId"] == root.id for s in spans)
    gens = of_type(events, "generation-create")
    assert [g["name"] for g in gens] == ["reason", "respond"]
    assert [parent_name(events, g) for g in gens] == ["reason", "respond"]
    assert [parent_name(events, s) for s in spans] == [None, "agent_query", "agent_query"] + [
        "agent_query"
    ] * 2
    for g in gens:
        assert g["model"] == "fake-model"
        assert g["usage"]["input"] == 11 and g["usage"]["output"] == 3
        assert "totalCost" in g["usage"] and g["output"]
        assert g["input"][-1]["role"] == "user"


def make_db(tmp_path: Path) -> Path:
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 10) for i in range(1, 4)])
    conn.commit()
    conn.close()
    return path


def test_sql_agent_v2_spans_with_check_and_fix(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    client, _ = make_client(
        [
            "```sql\nSELECT nope FROM t\n```",
            "No such column.\n```sql\nSELECT v FROM t WHERE id = 2\n```",
            "OK",
        ]
    )
    token = set_client(client)
    try:
        with tracing.trace("test") as root:
            result = run_sql_agent(
                "value for id 2?", "", make_db(tmp_path), sample_values=False, skip_respond=True
            )
    finally:
        reset_client(token)
    assert result.final_sql == "SELECT v FROM t WHERE id = 2"

    spans = of_type(events, "span-create")
    names = [s["name"] for s in spans]
    assert names == [
        "sql_agent_v2",
        "introspect",
        "generate",
        "execute",
        "check",
        "execute",
        "check",
        "respond",
    ]
    assert all(s["traceId"] == root.id for s in spans)
    assert all(parent_name(events, s) == "sql_agent_v2" for s in spans[1:])
    gens = of_type(events, "generation-create")
    assert [(g["name"], parent_name(events, g)) for g in gens] == [
        ("generate", "generate"),
        ("check", "check"),
        ("check", "check"),
    ]
    ends = {u["id"]: u for u in of_type(events, "span-update")}
    first_exec, second_exec = (s for s in spans if s["name"] == "execute")
    out = ends[first_exec["id"]]["output"]
    assert out["error"] and out["blocked"] is False and out["row_count"] == "n/a"
    out = ends[second_exec["id"]]["output"]
    assert out["error"] is None and out["columns"] == ["v"] and out["row_count"] == "1"
    assert out["truncated"] is False and "elapsed_s" in out
    assert first_exec["input"] == {"sql": "SELECT nope FROM t"}
    checks = [ends[s["id"]]["output"] for s in spans if s["name"] == "check"]
    assert checks[0]["verdict"] == "fix" and checks[0]["fix_sql"].startswith("SELECT v")
    assert checks[1]["verdict"] == "ok"


def test_sql_agent_v2_blocked_query_reports_layer(
    events: list[dict[str, Any]], tmp_path: Path
) -> None:
    client, _ = make_client(["```sql\nDROP TABLE t\n```", "OK"])
    token = set_client(client)
    try:
        with tracing.trace("test"):
            run_sql_agent(
                "drop it",
                "",
                make_db(tmp_path),
                sample_values=False,
                skip_respond=True,
                max_fixes=1,
            )
    finally:
        reset_client(token)
    spans = of_type(events, "span-create")
    ends = {u["id"]: u for u in of_type(events, "span-update")}
    execute = next(s for s in spans if s["name"] == "execute")
    out = ends[execute["id"]]["output"]
    assert out["blocked"] is True and out["layer"] and out["block_reason"]


def test_sql_agent_v2_opens_its_own_trace(events: list[dict[str, Any]], tmp_path: Path) -> None:
    client, _ = make_client(["```sql\nSELECT 1\n```"])
    token = set_client(client)
    try:
        run_sql_agent(
            "q", "", make_db(tmp_path), sample_values=False, use_check=False, skip_respond=True
        )
    finally:
        reset_client(token)
    traces = of_type(events, "trace-create")
    assert traces[0]["name"] == "sql_agent_v2"
    assert of_type(events, "span-create")[0]["name"] == "introspect"


def test_failed_llm_call_is_recorded_as_error_generation(events: list[dict[str, Any]]) -> None:
    config = LLMConfig(provider="ollama", model="fake-model")
    client = OllamaClient(config, UsageTracker())
    client._llm = SimpleNamespace(invoke=lambda m: (_ for _ in ()).throw(RuntimeError("down")))  # type: ignore[assignment]
    with tracing.trace("t"), pytest.raises(RuntimeError):
        client.complete("sys", "user", tag="generate")
    (gen,) = of_type(events, "generation-create")
    assert gen["level"] == "ERROR" and "down" in gen["statusMessage"]


def test_no_trace_means_no_events(events: list[dict[str, Any]]) -> None:
    client, _ = make_client(["hello"])
    assert client.complete("s", "u", tag="x").text == "hello"
    assert events == []
