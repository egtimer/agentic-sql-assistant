"""Offline end-to-end smoke test of the MCP server over stdio.

Starts the real server (``python -m src.mcp_server``) as a subprocess with a temporary
databases directory that holds one tiny database, connects to it with the SDK's stdio client
and calls ``list_databases``, ``describe_database`` and ``run_sql`` (one query that runs, one
that the SQL guard refuses). No language-model tool is called, so no model, network or key is
needed. Exits 1 on any unexpected result. Usage:

    python -m src.mcp_server.smoke
"""

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from src.mcp_server.demo_data import REPO_ROOT

TOOLS = {
    "list_databases",
    "describe_database",
    "run_sql",
    "ask",
    "teach",
    "forget",
    "doc_search",
    "calculator",
}
TIMEOUT_S = 90


def make_fixture(directory: Path) -> None:
    conn = sqlite3.connect(directory / "smoke.sqlite")
    conn.execute("CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop INTEGER)")
    conn.executemany("INSERT INTO city VALUES (?, ?, ?)", [(1, "Oslo", 5), (2, "Rome", 9)])
    conn.commit()
    conn.close()


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


async def run(databases_dir: Path, state_dir: Path) -> list[str]:
    failures: list[str] = []
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "src.mcp_server",
            "--databases-dir",
            str(databases_dir),
            "--state-dir",
            str(state_dir),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT)},
    )
    with anyio.fail_after(TIMEOUT_S):
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            check(tools == TOOLS, f"tools are {sorted(TOOLS)}", failures)

            listed = await client.call_tool("list_databases", {})
            expected: dict[str, Any] = {"databases": [{"name": "smoke", "tables": 1}]}
            check(
                not listed.is_error and listed.structured_content == expected,
                "list_databases returns the fixture database",
                failures,
            )

            described = await client.call_tool("describe_database", {"database": "smoke"})
            text = getattr(described.content[0], "text", "") if described.content else ""
            check(
                not described.is_error and "CREATE TABLE city" in text and "Oslo" in text,
                "describe_database returns DDL and example values",
                failures,
            )

            ok = await client.call_tool(
                "run_sql",
                {"sql": "SELECT name FROM city ORDER BY pop DESC", "database": "smoke"},
            )
            data = ok.structured_content or {}
            check(
                not ok.is_error and data.get("rows") == [["Rome"], ["Oslo"]],
                "run_sql runs a SELECT",
                failures,
            )

            blocked = await client.call_tool(
                "run_sql", {"sql": "DROP TABLE city", "database": "smoke"}
            )
            data = blocked.structured_content or {}
            check(
                not blocked.is_error and data.get("blocked") is True and not data.get("ok"),
                "run_sql reports a DROP as blocked",
                failures,
            )

            unknown = await client.call_tool("run_sql", {"sql": "SELECT 1", "database": "nope"})
            check(unknown.is_error, "an unknown database is a tool error", failures)

            calc = await client.call_tool(
                "calculator", {"expression": "(150000 - 120000) / 120000 * 100"}
            )
            check(
                not calc.is_error and (calc.structured_content or {}).get("result") == 25,
                "calculator evaluates an expression",
                failures,
            )

            again = await client.call_tool(
                "run_sql", {"sql": "SELECT count(*) FROM city", "database": "smoke"}
            )
            check(
                (again.structured_content or {}).get("rows") == [[2]],
                "the table is intact after the blocked statement",
                failures,
            )
    return failures


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        databases_dir = Path(tmp) / "databases"
        databases_dir.mkdir()
        make_fixture(databases_dir)
        failures = anyio.run(run, databases_dir, Path(tmp) / "state")
    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("MCP stdio smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
