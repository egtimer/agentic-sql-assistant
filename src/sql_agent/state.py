"""State and result types of the SQL agent."""

from typing import Literal

from pydantic import BaseModel, Field

from src.sql_agent.executor import Attempt
from src.sql_agent.memory import Memory
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


class Turn(BaseModel):
    """What the conversation remembers of one answered question (never the result rows)."""

    question: str
    final_sql: str
    row_count: int | None = None  # capped like ``Attempt.row_count``; None if the query failed
    columns: list[str] = Field(default_factory=list)


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
    user_id: str | None = None  # whose long-term memories ``recall`` reads (needs a store)

    schema_context: str = ""
    candidate_sql: str = ""
    attempts: list[Attempt] = Field(default_factory=list)
    fixes_used: int = 0
    verdicts: list[Verdict] = Field(default_factory=list)
    final_sql: str = ""
    response: str = ""
    recalled: list[Memory] = Field(default_factory=list)  # memories shown to generate/check
    n_memories: int = 0  # memories the user has for this database

    # Conversation memory: survives between the turns of one thread (checkpointer), and is
    # the only field that does. Everything above is reset explicitly on every turn.
    history: list[Turn] = Field(default_factory=list)


class SqlAgentResult(BaseModel):
    """What :func:`run_sql_agent` returns."""

    final_sql: str
    attempts: list[Attempt]
    verdicts: list[Verdict]
    fixes_used: int
    response: str = ""
    schema_context: str = ""
    history: list[Turn] = Field(default_factory=list)  # including the turn just answered
    recalled: list[Memory] = Field(default_factory=list)  # long-term memories that were shown
    n_memories: int = 0

    @property
    def n_unparsed(self) -> int:
        return sum(1 for v in self.verdicts if v.unparsed)
