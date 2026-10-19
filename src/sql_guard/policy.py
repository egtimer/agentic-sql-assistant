"""Policy of the SQL guard: resource limits and the tables and columns that stay off limits."""

from pydantic import BaseModel, Field, field_validator


class GuardPolicy(BaseModel):
    """Limits and access rules applied to every model-written statement."""

    timeout_s: float = Field(default=30.0, gt=0, description="Wall-clock limit per statement")
    max_rows: int = Field(default=1000, ge=1, description="Rows returned; one more is fetched")
    max_value_bytes: int = Field(
        default=1_000_000, ge=1, description="SQLITE_LIMIT_LENGTH: longest string or blob"
    )
    max_sql_length: int = Field(
        default=100_000, ge=1, description="SQLITE_LIMIT_SQL_LENGTH and a pre-check"
    )
    max_memory_bytes: int = Field(
        default=1_073_741_824,
        ge=1,
        description="Hard heap limit of the SQLite library (process-wide), 0 turns it off",
    )
    denied_tables: set[str] = Field(default_factory=set)
    denied_columns: set[tuple[str, str]] = Field(default_factory=set)

    @field_validator("denied_tables")
    @classmethod
    def _lower_tables(cls, value: set[str]) -> set[str]:
        return {name.lower() for name in value}

    @field_validator("denied_columns")
    @classmethod
    def _lower_columns(cls, value: set[tuple[str, str]]) -> set[tuple[str, str]]:
        return {(table.lower(), column.lower()) for table, column in value}
