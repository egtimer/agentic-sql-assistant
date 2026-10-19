"""The MCP server object: tools, resources and the shared service behind them."""

import math
from typing import Annotated, Any

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from src.mcp_server.catalog import Catalog, UnknownDatabaseError
from src.mcp_server.models import DatabaseInfo, DatabaseList, SqlResult
from src.sql_agent.introspect import schema_context
from src.sql_guard import GuardPolicy, guarded_execute

SQL_MAX_ROWS = 100
SQL_TIMEOUT_S = 10.0
MAX_CELL_CHARS = 200
MAX_SQL_CHARS = 20_000

INSTRUCTIONS = (
    "Questions over SQLite databases. Call list_databases first, then describe_database to "
    "read a schema. Use ask for a question in natural language (it writes, checks and runs the "
    "SQL itself) and run_sql to run a query you wrote; every query goes through a read-only "
    "SQL guard that refuses anything but a single SELECT."
)

DatabaseName = Annotated[
    str, Field(description="Name of the database, as returned by list_databases", max_length=200)
]
_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def cell(value: object) -> object:
    """A JSON-safe, size-bounded copy of a database value."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return f"<blob, {len(value)} bytes>"
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "..."


def render_sql_result(result: SqlResult) -> str:
    """Short text form of a :class:`SqlResult` for clients that ignore structured content."""
    if result.error is not None:
        kind = "Blocked by the SQL guard" if result.blocked else "SQL error"
        return f"{kind}: {result.error}"
    lines = [f"{result.row_count} row(s)" + (" (truncated)" if result.truncated else "")]
    lines.append(" | ".join(result.columns))
    lines.extend(" | ".join("NULL" if v is None else str(v) for v in row) for row in result.rows)
    return "\n".join(lines)


def tool_result(model: BaseModel, text: str) -> CallToolResult:
    """Structured content plus a short text rendering."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=model.model_dump(mode="json"),
    )


class Service:
    """What the tools do, independent of the transport."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def list_databases(self) -> DatabaseList:
        return DatabaseList(
            databases=[
                DatabaseInfo(name=name, tables=self.catalog.table_count(name))
                for name in self.catalog.names
            ]
        )

    def describe(self, database: str) -> str:
        return schema_context(self.catalog.path(database))

    def run_sql(self, sql: str, database: str) -> SqlResult:
        path = self.catalog.path(database)
        policy = GuardPolicy(max_rows=SQL_MAX_ROWS, timeout_s=SQL_TIMEOUT_S)
        result = guarded_execute(path, sql, policy)
        if result.error is not None:
            return SqlResult(
                database=database, ok=False, blocked=result.blocked, error=result.error
            )
        rows = [[cell(v) for v in row] for row in result.rows]
        return SqlResult(
            database=database,
            ok=True,
            blocked=False,
            columns=result.columns,
            rows=rows,
            row_count=len(rows),
            truncated=result.truncated,
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

    @server.resource(
        "schema://{database}", mime_type="text/plain", description="Schema of a database"
    )
    async def schema(database: str) -> str:
        try:
            return await anyio.to_thread.run_sync(service.describe, database)
        except UnknownDatabaseError as e:
            raise ResourceError(str(e)) from None

    return server
