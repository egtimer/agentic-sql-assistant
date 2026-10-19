"""Typed outputs of the MCP tools (they become the tools' output schemas)."""

from pydantic import BaseModel, Field


class DatabaseInfo(BaseModel):
    name: str = Field(description="Name to pass as the `database` argument of the other tools")
    tables: int = Field(description="Number of tables")


class DatabaseList(BaseModel):
    databases: list[DatabaseInfo]


class SqlResult(BaseModel):
    """Outcome of `run_sql`: rows, or why nothing ran."""

    database: str
    ok: bool = Field(description="True when the query ran and returned a result")
    blocked: bool = Field(description="True when the SQL guard refused the statement")
    error: str | None = Field(default=None, description="Block reason or SQL error, else null")
    columns: list[str] = []
    rows: list[list[object]] = Field(default=[], description="At most 100 rows")
    row_count: int = Field(default=0, description="Rows returned (after the cap)")
    truncated: bool = Field(default=False, description="True when more rows existed than the cap")


class AskResult(BaseModel):
    """Outcome of `ask`: the agent's answer and the query behind it."""

    database: str
    answer: str = Field(description="Natural-language answer written from the query result")
    sql: str = Field(description="Final SQL query the agent chose")
    columns: list[str] = []
    rows: list[list[str]] = Field(default=[], description="First 20 rows, values truncated")
    row_count: str = Field(default="n/a", description="Rows the query returned (>1000 if capped)")
    blocked: bool = Field(description="True when the SQL guard refused the final query")
    error: str | None = Field(default=None, description="Error of the final query, else null")
    llm_calls: int = Field(description="Language-model calls made for this request")
    cost_usd: float = Field(description="Cost of those calls in US dollars")
    trace_id: str | None = Field(default=None, description="Trace id when tracing is configured")


class StoredDefinition(BaseModel):
    term: str
    definition: str


class TeachResult(BaseModel):
    database: str
    user_id: str
    stored: list[StoredDefinition] = Field(description="Definitions stored by this call")
    llm_calls: int
    cost_usd: float


class ForgetResult(BaseModel):
    database: str
    user_id: str
    deleted: list[str] = Field(description="Terms that were deleted")
