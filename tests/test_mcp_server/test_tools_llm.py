"""ask, teach and forget through an in-process MCP client with a scripted language model."""

from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from src.mcp_server.catalog import Catalog
from src.mcp_server.server import create_server
from src.mcp_server.service import LLM_FAILED, LLM_NOT_CONFIGURED, Service
from tests.test_mcp_server.helpers import (
    DEFAULT_SCRIPT,
    Rig,
    data_of,
    fence,
    text_of,
)
from tests.test_observability.helpers import of_type

ASK = {"question": "Which city is the biggest?", "database": "toy"}


@pytest.fixture
def rig(databases_dir: Path, tmp_path: Path) -> Rig:
    return Rig(databases_dir, tmp_path / "state")


async def test_all_six_tools_are_listed_with_documented_inputs(rig: Rig):
    async with Client(create_server(rig.service)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "list_databases",
        "describe_database",
        "run_sql",
        "ask",
        "teach",
        "forget",
    }
    ask = tools["ask"].input_schema
    assert set(ask["required"]) == {"question", "database"}
    assert set(ask["properties"]) == {"question", "database", "evidence", "thread_id", "user_id"}
    assert all(p.get("description") for p in ask["properties"].values())
    assert tools["ask"].output_schema is not None and tools["teach"].output_schema is not None
    assert set(tools["forget"].input_schema["required"]) == {"database", "user_id"}
    assert tools["forget"].annotations is not None
    assert tools["forget"].annotations.destructive_hint is True


async def test_ask_returns_answer_sql_rows_and_cost(databases_dir: Path, tmp_path: Path):
    rig = Rig(databases_dir, tmp_path / "state", cost_per_call=0.002)
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", ASK)
    assert not result.is_error
    data = data_of(result)
    assert data["answer"] == "Rome is the biggest city."
    assert data["sql"] == "SELECT name, pop FROM city ORDER BY pop DESC LIMIT 1"
    assert data["columns"] == ["name", "pop"]
    assert data["rows"] == [["Rome", "9"]]
    assert data["row_count"] == "1"
    assert data["blocked"] is False and data["error"] is None
    assert data["llm_calls"] == 3  # generate, check, respond
    assert data["cost_usd"] == pytest.approx(0.006)
    assert data["trace_id"] is None  # tracing is not configured
    text = text_of(result)
    assert "Rome is the biggest city." in text and "SELECT name" in text
    assert rig.service.tracker.total_cost == pytest.approx(0.006)


async def test_ask_cost_is_per_call(databases_dir: Path, tmp_path: Path):
    rig = Rig(databases_dir, tmp_path / "state", cost_per_call=0.002)
    async with Client(create_server(rig.service)) as client:
        first = data_of(await client.call_tool("ask", ASK))
        second = data_of(await client.call_tool("ask", ASK))
    assert first["cost_usd"] == second["cost_usd"] == pytest.approx(0.006)


async def test_ask_passes_the_evidence_to_the_agent(rig: Rig):
    async with Client(create_server(rig.service)) as client:
        await client.call_tool("ask", {**ASK, "evidence": "big means pop above 5"})
    assert "big means pop above 5" in rig.llm.prompts("generate")[0]


async def test_guard_blocks_a_model_written_query(databases_dir: Path, tmp_path: Path):
    script = {
        **DEFAULT_SCRIPT,
        "generate": [fence("DROP TABLE city")],
        "check": ["Cannot fix it.\n\nOK"],
    }
    rig = Rig(databases_dir, tmp_path / "state", script=script)
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", ASK)
        check = await client.call_tool(
            "run_sql", {"sql": "SELECT count(*) FROM city", "database": "toy"}
        )
    data = data_of(result)
    assert not result.is_error
    assert data["blocked"] is True and data["error"].startswith("blocked: ")
    assert data_of(check)["rows"] == [[3]]  # the table is still there


async def test_ask_unknown_database(rig: Rig):
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", {"question": "x", "database": "nope"})
    assert result.is_error and "Valid names" in text_of(result)
    assert rig.clients[0].calls == []


@pytest.mark.parametrize("field", ["thread_id", "user_id"])
@pytest.mark.parametrize("value", ["", "a b", "../x", "x" * 129])
async def test_bad_ids_are_rejected(rig: Rig, field: str, value: str):
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", {**ASK, field: value})
    assert result.is_error
    assert rig.clients[0].calls == []


async def test_thread_memory_across_two_asks(databases_dir: Path, tmp_path: Path):
    state = tmp_path / "state"
    rig = Rig(databases_dir, state)
    async with Client(create_server(rig.service)) as client:
        await client.call_tool("ask", {**ASK, "thread_id": "t1"})
        await client.call_tool(
            "ask", {"question": "And the smallest?", "database": "toy", "thread_id": "t1"}
        )
        await client.call_tool(
            "ask", {"question": "And the smallest?", "database": "toy", "thread_id": "t2"}
        )
        await client.call_tool("ask", {"question": "And the smallest?", "database": "toy"})
    generate = rig.llm.prompts("generate")
    assert "Earlier in this conversation" not in generate[0]
    assert "Earlier in this conversation" in generate[1]
    assert "Which city is the biggest?" in generate[1]
    assert "Earlier in this conversation" not in generate[2]  # another thread
    assert "Earlier in this conversation" not in generate[3]  # no thread
    assert (state / "checkpoints.sqlite").exists()


async def test_thread_memory_survives_a_restart_and_is_per_database(
    databases_dir: Path, tmp_path: Path
):
    state = tmp_path / "state"
    first = Rig(databases_dir, state)
    async with Client(create_server(first.service)) as client:
        await client.call_tool("ask", {**ASK, "thread_id": "t1"})
    first.service.close()

    second = Rig(databases_dir, state)
    async with Client(create_server(second.service)) as client:
        await client.call_tool("ask", {**ASK, "thread_id": "t1"})
        await client.call_tool("ask", {**ASK, "database": "other", "thread_id": "t1"})
    generate = second.llm.prompts("generate")
    assert "Earlier in this conversation" in generate[0]
    assert "Earlier in this conversation" not in generate[1]
    second.service.close()


async def test_teach_then_ask_recalls_the_definition(rig: Rig):
    async with Client(create_server(rig.service)) as client:
        taught = await client.call_tool(
            "teach",
            {"text": "A big city is a city with pop above 5.", "database": "toy", "user_id": "ana"},
        )
        await client.call_tool("ask", {**ASK, "user_id": "ana"})
        await client.call_tool("ask", {**ASK, "user_id": "bob"})
        await client.call_tool("ask", ASK)
    data = data_of(taught)
    assert data["stored"] == [{"term": "big city", "definition": "a city with pop above 5"}]
    assert data["llm_calls"] == 1
    assert "big city: a city with pop above 5" in text_of(taught)
    generate = rig.llm.prompts("generate")
    assert "a city with pop above 5" in generate[0]
    assert "a city with pop above 5" not in generate[1]  # another user
    assert "a city with pop above 5" not in generate[2]  # no user


async def test_teach_drops_terms_the_user_did_not_write(databases_dir: Path, tmp_path: Path):
    script = {**DEFAULT_SCRIPT, "memorize": ['[{"term": "invented", "definition": "x"}]']}
    rig = Rig(databases_dir, tmp_path / "state", script=script)
    async with Client(create_server(rig.service)) as client:
        taught = await client.call_tool(
            "teach", {"text": "Nothing special here.", "database": "toy", "user_id": "ana"}
        )
    assert data_of(taught)["stored"] == []
    assert "Nothing durable" in text_of(taught)


async def test_forget_one_term_then_all(databases_dir: Path, tmp_path: Path):
    script = {
        **DEFAULT_SCRIPT,
        "memorize": [
            '[{"term": "big city", "definition": "pop above 5"},'
            ' {"term": "small city", "definition": "pop below 5"}]'
        ],
    }
    rig = Rig(databases_dir, tmp_path / "state", script=script)
    text = "A big city has pop above 5, a small city has pop below 5."
    async with Client(create_server(rig.service)) as client:
        await client.call_tool("teach", {"text": text, "database": "toy", "user_id": "ana"})
        await client.call_tool("teach", {"text": text, "database": "other", "user_id": "ana"})
        one = await client.call_tool(
            "forget", {"database": "toy", "user_id": "ana", "term": "  BIG city "}
        )
        await client.call_tool("ask", {**ASK, "user_id": "ana"})
        everything = await client.call_tool("forget", {"database": "toy", "user_id": "ana"})
        nothing = await client.call_tool("forget", {"database": "toy", "user_id": "ana"})
        await client.call_tool("ask", {**ASK, "user_id": "ana"})
        other = await client.call_tool("ask", {**ASK, "database": "other", "user_id": "ana"})
    assert data_of(one)["deleted"] == ["big city"]
    assert data_of(everything)["deleted"] == ["small city"]
    assert data_of(nothing)["deleted"] == [] and text_of(nothing) == "Nothing to forget."
    generate = rig.llm.prompts("generate")
    assert "pop above 5" not in generate[0] and "pop below 5" in generate[0]
    assert "pop below 5" not in generate[1]
    assert "pop above 5" in generate[2]  # the other database keeps its memories
    assert not other.is_error


async def test_budget_exhaustion_stops_llm_tools_but_not_run_sql(
    databases_dir: Path, tmp_path: Path
):
    rig = Rig(databases_dir, tmp_path / "state", cost_per_call=0.01, budget_usd=0.025)
    async with Client(create_server(rig.service)) as client:
        first = await client.call_tool("ask", ASK)  # 3 calls, 0.03 spent
        second = await client.call_tool("ask", ASK)
        taught = await client.call_tool(
            "teach", {"text": "A big city is big.", "database": "toy", "user_id": "ana"}
        )
        sql = await client.call_tool(
            "run_sql", {"sql": "SELECT count(*) FROM city", "database": "toy"}
        )
        described = await client.call_tool("describe_database", {"database": "toy"})
        listed = await client.call_tool("list_databases", {})
    assert not first.is_error
    for refused in (second, taught):
        assert refused.is_error and "budget" in text_of(refused)
    assert not sql.is_error and not described.is_error and not listed.is_error
    assert len(rig.llm.calls) == 3  # nothing reached the model after the cap


async def test_budget_reached_during_a_request_is_a_tool_error(databases_dir: Path, tmp_path: Path):
    rig = Rig(databases_dir, tmp_path / "state", cost_per_call=0.01, budget_usd=0.015)
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", ASK)  # generate, check ok; respond refused
    assert result.is_error and "budget of $0.01" in text_of(result)


@pytest.mark.parametrize("secret", ["/home/someone/secret.db", "Traceback (most recent call last)"])
async def test_unexpected_errors_do_not_leak(databases_dir: Path, tmp_path: Path, secret: str):
    rig = Rig(databases_dir, tmp_path / "state", error=RuntimeError(secret))
    async with Client(create_server(rig.service)) as client:
        ask = await client.call_tool("ask", ASK)
        teach = await client.call_tool(
            "teach", {"text": "A big city is big.", "database": "toy", "user_id": "ana"}
        )
    for result in (ask, teach):
        assert result.is_error and text_of(result).endswith(LLM_FAILED)
        assert secret not in text_of(result)


async def test_without_a_language_model_only_the_sql_tools_work(
    databases_dir: Path, tmp_path: Path
):
    service = Service(Catalog.scan(databases_dir), state_dir=tmp_path / "state")
    async with Client(create_server(service)) as client:
        ask = await client.call_tool("ask", ASK)
        sql = await client.call_tool("run_sql", {"sql": "SELECT 1", "database": "toy"})
    assert ask.is_error and text_of(ask).endswith(LLM_NOT_CONFIGURED)
    assert not sql.is_error


async def test_error_texts_have_no_paths_or_tracebacks(rig: Rig, databases_dir: Path):
    async with Client(create_server(rig.service)) as client:
        results: list[Any] = [
            await client.call_tool("ask", {"question": "x", "database": "nope"}),
            await client.call_tool("ask", {"database": "toy"}),
            await client.call_tool("run_sql", {"sql": 5, "database": "toy"}),
            await client.call_tool("no_such_tool", {}),
            await client.call_tool("forget", {"database": "nope", "user_id": "ana"}),
        ]
    for result in results:
        assert result.is_error
        text = text_of(result)
        assert "Traceback" not in text and str(databases_dir) not in text
        assert 'File "' not in text and "/home/" not in text


async def test_one_trace_per_ask(rig: Rig, events: list[dict[str, Any]]):
    async with Client(create_server(rig.service)) as client:
        result = await client.call_tool("ask", {**ASK, "thread_id": "t1", "user_id": "ana"})
        await client.call_tool("ask", ASK)
    traces = [t for t in of_type(events, "trace-create") if "name" in t]
    assert [t["name"] for t in traces] == ["mcp_ask", "mcp_ask"]
    assert data_of(result)["trace_id"] == traces[0]["id"]
    assert traces[0]["sessionId"] == "t1" and traces[0]["userId"] == "ana"
    spans = {s["name"] for s in of_type(events, "span-create")}
    assert "sql_agent_v2" in spans
