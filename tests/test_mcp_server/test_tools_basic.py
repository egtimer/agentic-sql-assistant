"""In-process MCP client talking to the server: listing, schema and guarded SQL."""

from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp_types import CallToolResult, TextContent

from src.mcp_server.catalog import Catalog
from src.mcp_server.server import Service, create_server


def text_of(result: CallToolResult) -> str:
    assert result.content and isinstance(result.content[0], TextContent)
    return result.content[0].text


def server_for(directory: Path) -> Any:
    return create_server(Service(Catalog.scan(directory)))


async def test_tool_list_and_input_schemas(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert {"list_databases", "describe_database", "run_sql"} <= set(tools)
    run_sql = tools["run_sql"]
    assert set(run_sql.input_schema["required"]) == {"sql", "database"}
    for prop in run_sql.input_schema["properties"].values():
        assert prop["description"]
    assert run_sql.output_schema is not None
    assert tools["list_databases"].input_schema.get("properties", {}) == {}
    assert all(t.description for t in tools.values())


async def test_list_databases_gives_names_and_table_counts(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool("list_databases", {})
    assert not result.is_error
    assert result.structured_content == {
        "databases": [{"name": "other", "tables": 2}, {"name": "toy", "tables": 2}]
    }
    assert "toy (2 tables)" in text_of(result)


async def test_describe_database_returns_ddl_and_examples(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool("describe_database", {"database": "toy"})
        resource = await client.read_resource("schema://toy")
    text = text_of(result)
    assert "CREATE TABLE city" in text
    assert "'Oslo'" in text
    assert resource.contents[0].text == text  # type: ignore[union-attr]
    assert str(databases_dir) not in text


async def test_run_sql_happy_path(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool(
            "run_sql", {"sql": "SELECT name, pop FROM city ORDER BY id LIMIT 2", "database": "toy"}
        )
    assert not result.is_error
    data = result.structured_content
    assert data is not None
    assert data["ok"] is True and data["blocked"] is False
    assert data["columns"] == ["name", "pop"]
    assert data["rows"] == [["Oslo", 5], ["Rome", 9]]
    assert data["row_count"] == 2 and data["truncated"] is False
    assert "Oslo | 5" in text_of(result)


async def test_run_sql_truncates_long_values_and_caps_rows(databases_dir: Path):
    sql = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 250) "
        "SELECT i, (SELECT name FROM city WHERE id = 3) AS long_name FROM n"
    )
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool("run_sql", {"sql": sql, "database": "toy"})
    data = result.structured_content
    assert data is not None
    assert data["row_count"] == 100 and data["truncated"] is True
    assert data["rows"][0][1].endswith("...") and len(data["rows"][0][1]) < 300


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE city",
        "DELETE FROM city",
        "ATTACH DATABASE 'other.db' AS other",
        "VACUUM INTO 'copy.db'",
        "SELECT 1; SELECT 2",
        "PRAGMA writable_schema = 1",
        "SELECT load_extension('x')",
    ],
)
async def test_guard_blocks_are_results_not_crashes(databases_dir: Path, sql: str):
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool("run_sql", {"sql": sql, "database": "toy"})
        # the server is still fine afterwards and the table is intact
        again = await client.call_tool(
            "run_sql", {"sql": "SELECT count(*) FROM city", "database": "toy"}
        )
    assert not result.is_error
    data = result.structured_content
    assert data is not None
    assert data["ok"] is False and data["blocked"] is True
    assert data["error"].startswith("blocked: ")
    assert again.structured_content is not None and again.structured_content["rows"] == [[3]]


async def test_sql_error_is_a_result_with_blocked_false(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        result = await client.call_tool(
            "run_sql", {"sql": "SELECT nope FROM city", "database": "toy"}
        )
    data = result.structured_content
    assert data is not None
    assert not result.is_error and data["ok"] is False and data["blocked"] is False
    assert "no such column" in data["error"]


@pytest.mark.parametrize("name", ["nope", "toy.sqlite", "../dbs/toy", ""])
async def test_unknown_database_is_rejected_with_valid_names(databases_dir: Path, name: str):
    async with Client(server_for(databases_dir)) as client:
        for tool, args in (
            ("describe_database", {"database": name}),
            ("run_sql", {"sql": "SELECT 1", "database": name}),
        ):
            result = await client.call_tool(tool, args)
            assert result.is_error
            message = text_of(result)
            assert "'other', 'toy'" in message
            assert str(databases_dir) not in message and "Traceback" not in message


async def test_unknown_resource_is_an_error_without_paths(databases_dir: Path):
    async with Client(server_for(databases_dir)) as client:
        with pytest.raises(Exception) as info:
            await client.read_resource("schema://nope")
    assert str(databases_dir) not in str(info.value) and "Traceback" not in str(info.value)
