"""The MCP server object: tool and resource definitions on top of :class:`Service`."""

from typing import Annotated, Any

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from src.mcp_server.catalog import UnknownDatabaseError
from src.mcp_server.models import (
    AskResult,
    DatabaseList,
    ForgetResult,
    SqlResult,
    TeachResult,
)
from src.mcp_server.service import Service

MAX_SQL_CHARS = 20_000
ID_PATTERN = r"^[A-Za-z0-9_.@:+-]{1,128}$"

INSTRUCTIONS = (
    "Questions over SQLite databases. Call list_databases first, then describe_database to "
    "read a schema. Use ask for a question in natural language (it writes, checks and runs the "
    "SQL itself) and run_sql to run a query you wrote; every query goes through a read-only "
    "SQL guard that refuses anything but a single SELECT. teach and forget manage the "
    "definitions a user wants the agent to remember."
)

DatabaseName = Annotated[
    str, Field(description="Name of the database, as returned by list_databases", max_length=200)
]
UserId = Annotated[
    str,
    Field(
        description="Identifier of the user whose memories are used (letters, digits, _ . @ : + -)",
        pattern=ID_PATTERN,
    ),
]
_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def render_sql_result(result: SqlResult) -> str:
    """Short text form of a :class:`SqlResult` for clients that ignore structured content."""
    if result.error is not None:
        kind = "Blocked by the SQL guard" if result.blocked else "SQL error"
        return f"{kind}: {result.error}"
    lines = [f"{result.row_count} row(s)" + (" (truncated)" if result.truncated else "")]
    lines.append(" | ".join(result.columns))
    lines.extend(" | ".join("NULL" if v is None else str(v) for v in row) for row in result.rows)
    return "\n".join(lines)


def render_ask_result(result: AskResult) -> str:
    lines = [result.answer, "", f"SQL: {result.sql}"]
    if result.blocked:
        lines.append(f"Blocked by the SQL guard: {result.error}")
    elif result.error:
        lines.append(f"SQL error: {result.error}")
    else:
        lines.append(f"Rows: {result.row_count} ({len(result.rows)} returned)")
    lines.append(f"LLM calls: {result.llm_calls}, cost: ${result.cost_usd:.4f}")
    return "\n".join(lines)


def render_teach_result(result: TeachResult) -> str:
    if not result.stored:
        return "Nothing durable to remember was found in the text."
    return "Stored:\n" + "\n".join(f"- {d.term}: {d.definition}" for d in result.stored)


def render_forget_result(result: ForgetResult) -> str:
    if not result.deleted:
        return "Nothing to forget."
    return "Forgot: " + ", ".join(result.deleted)


def tool_result(model: BaseModel, text: str) -> CallToolResult:
    """Structured content plus a short text rendering."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=model.model_dump(mode="json"),
    )


def create_server(service: Service) -> MCPServer[Any]:
    """Register the tools of ``service`` on a new MCP server."""
    server: MCPServer[Any] = MCPServer("sql-agent", instructions=INSTRUCTIONS)

    def known(database: str) -> None:
        try:
            service.catalog.path(database)
        except UnknownDatabaseError as e:
            raise ToolError(str(e)) from None

    @server.tool(annotations=_READ_ONLY)
    async def list_databases() -> Annotated[CallToolResult, DatabaseList]:
        """List the databases this server can query, with their number of tables."""
        listing = await anyio.to_thread.run_sync(service.list_databases)
        text = "\n".join(f"{d.name} ({d.tables} tables)" for d in listing.databases)
        return tool_result(listing, text or "No databases.")

    @server.tool(annotations=_READ_ONLY)
    async def describe_database(database: DatabaseName) -> str:
        """Schema of a database as the agent sees it: DDL, undeclared keys, example values."""
        known(database)
        return await anyio.to_thread.run_sync(service.describe, database)

    @server.tool(annotations=_READ_ONLY)
    async def run_sql(
        sql: Annotated[
            str,
            Field(
                description="One SELECT statement (SQLite dialect); anything else is refused",
                max_length=MAX_SQL_CHARS,
            ),
        ],
        database: DatabaseName,
    ) -> Annotated[CallToolResult, SqlResult]:
        """Run a read-only query through the SQL guard (no language model involved).

        Returns at most 100 rows. A statement the guard refuses (writes, ATTACH, VACUUM INTO,
        several statements, ...) comes back with blocked=true and the reason.
        """
        known(database)
        result = await anyio.to_thread.run_sync(service.run_sql, sql, database)
        return tool_result(result, render_sql_result(result))

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=True
        )
    )
    async def ask(
        question: Annotated[
            str, Field(description="Question in natural language", min_length=1, max_length=2000)
        ],
        database: DatabaseName,
        evidence: Annotated[
            str,
            Field(
                description="Optional hint: a definition or formula the question depends on",
                max_length=4000,
            ),
        ] = "",
        thread_id: Annotated[
            str | None,
            Field(
                description="Questions that share a thread id form one conversation, so a "
                "follow-up can refer to the previous answer",
                pattern=ID_PATTERN,
            ),
        ] = None,
        user_id: Annotated[
            str | None,
            Field(
                description="Use the definitions this user taught with `teach` for this database",
                pattern=ID_PATTERN,
            ),
        ] = None,
    ) -> Annotated[CallToolResult, AskResult]:
        """Answer a question about a database with the SQL agent.

        The agent writes one SQL query, runs it under the read-only SQL guard, reviews the
        result (fixing the query when it is wrong) and writes the answer. Makes several
        language-model calls and costs money with a paid model; the result reports both.
        """
        known(database)
        result = await anyio.to_thread.run_sync(
            service.ask, question, database, evidence, thread_id, user_id
        )
        return tool_result(result, render_ask_result(result))

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        )
    )
    async def teach(
        text: Annotated[
            str,
            Field(
                description="What the user says about their data: what a term means, how a "
                "measure is computed, which values a label stands for",
                min_length=1,
                max_length=4000,
            ),
        ],
        database: DatabaseName,
        user_id: UserId,
    ) -> Annotated[CallToolResult, TeachResult]:
        """Remember definitions from text for one user and database (one language-model call).

        Later `ask` calls with the same user_id and database see them. Teaching a term again
        replaces its definition.
        """
        known(database)
        result = await anyio.to_thread.run_sync(service.teach, text, database, user_id)
        return tool_result(result, render_teach_result(result))

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
        )
    )
    async def forget(
        database: DatabaseName,
        user_id: UserId,
        term: Annotated[
            str | None,
            Field(
                description="Term to forget; omit to forget everything this user taught for the "
                "database",
                max_length=200,
            ),
        ] = None,
    ) -> Annotated[CallToolResult, ForgetResult]:
        """Delete one or all remembered definitions of a user for a database."""
        known(database)
        result = await anyio.to_thread.run_sync(service.forget, database, user_id, term)
        return tool_result(result, render_forget_result(result))

    @server.resource(
        "schema://{database}", mime_type="text/plain", description="Schema of a database"
    )
    async def schema(database: str) -> str:
        try:
            return await anyio.to_thread.run_sync(service.describe, database)
        except UnknownDatabaseError as e:
            raise ResourceError(str(e)) from None

    return server
