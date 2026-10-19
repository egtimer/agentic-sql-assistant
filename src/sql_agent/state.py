"""State and result types of the SQL agent."""

from typing import Literal

from pydantic import BaseModel, Field

from src.sql_agent.executor import Attempt
from src.sql_guard import GuardPolicy


class Verdict(BaseModel):
    """What one ``check`` call decided."""

    kind: Literal["ok", "fix", "unparsed"]
    sql: str = ""  # the corrected query, for "fix"
    note: str = ""  # free text of the reviewer (before the SQL block; 200 chars if unparsed)
    text: str = ""  # the full, untruncated answer of the reviewer

    @property
    def unparsed(self) -> bool:
        return self.kind == "unparsed"


class SqlAgentState(BaseModel):
    """Graph state: inputs, options and everything the nodes produce."""

    question: str
    evidence: str = ""
    db_path: str
    sample_values: bool = True
    use_check: bool = True
    max_fixes: int = 2
    skip_respond: bool = False
    policy: GuardPolicy = Field(default_factory=GuardPolicy)

    schema_context: str = ""
    candidate_sql: str = ""
    attempts: list[Attempt] = Field(default_factory=list)
    fixes_used: int = 0
    verdicts: list[Verdict] = Field(default_factory=list)
    final_sql: str = ""
    response: str = ""


class SqlAgentResult(BaseModel):
    """What :func:`run_sql_agent` returns."""

    final_sql: str
    attempts: list[Attempt]
    verdicts: list[Verdict]
    fixes_used: int
    response: str = ""
    schema_context: str = ""

    @property
    def n_unparsed(self) -> int:
        return sum(1 for v in self.verdicts if v.unparsed)
