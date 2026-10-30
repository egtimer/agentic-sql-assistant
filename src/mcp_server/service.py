"""What the MCP tools do, independent of the transport.

The service owns the database allowlist, the budget, the language-model client and the
persistent state (a SQLite checkpointer for conversations, a SQLite store for what users
taught). Failures a client can act on are raised as ``ToolError`` with a fixed message;
anything unexpected is logged on the server and reaches the client as a generic message, so
no stack trace and no file path leaves the process.
"""

import logging
import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from src.llm import (
    BudgetExceeded,
    Completion,
    LLMClient,
    LLMConfig,
    UsageTracker,
    build_client,
    reset_client,
    set_client,
)
from src.mcp_server.catalog import Catalog, UnknownDatabaseError
from src.mcp_server.models import (
    AskResult,
    DatabaseInfo,
    DatabaseList,
    ForgetResult,
    SqlResult,
    StoredDefinition,
    TeachResult,
)
from src.observability import tracing
from src.sql_agent.graph import open_checkpointer, run_sql_agent
from src.sql_agent.introspect import schema_context
from src.sql_agent.memory import (
    load_memories,
    namespace,
    normalize_term,
    open_memory_store,
    teach_text,
)
from src.sql_guard import GuardPolicy, guarded_execute
from src.tools.calculator import CalculatorResult, calculate
from src.tools.doc_search import DocSearchResult, doc_search

logger = logging.getLogger(__name__)

SQL_MAX_ROWS = 100
SQL_TIMEOUT_S = 10.0
MAX_CELL_CHARS = 200
CHECKPOINT_FILE = "checkpoints.sqlite"
MEMORY_FILE = "memories.sqlite"

LLM_FAILED = "The language model request failed. The server log has the details."
LLM_NOT_CONFIGURED = (
    "This server has no language model configured; only run_sql and the schema tools work."
)

DocSearch = Callable[..., DocSearchResult]
"""``doc_search(query, top_k=...)``; tests pass a fake that needs no Qdrant."""

ClientFactory = Callable[[UsageTracker], LLMClient]
"""Builds the client of a server from its usage tracker (which holds the budget). The default
builds the provider of the LLM config; tests pass a scripted one."""


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


@dataclass
class _Meter:
    """LLM calls and cost of one tool call."""

    calls: int = 0
    cost_usd: float = 0.0


_meter: ContextVar[_Meter | None] = ContextVar("mcp_meter", default=None)


class _MeteredClient:
    """Counts the calls and the cost of the request it runs in (not of the whole process)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        completion = self._inner.complete(system, user, tag=tag, prefix=prefix)
        meter = _meter.get()
        if meter is not None:
            meter.calls += 1
            meter.cost_usd += completion.usage.cost_usd
        return completion


class Service:
    """The tools' logic. Methods are blocking; the server runs them in worker threads."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        llm_config: LLMConfig | None = None,
        budget_usd: float | None = None,
        state_dir: Path | None = None,
        client_factory: ClientFactory | None = None,
        doc_search_fn: DocSearch = doc_search,
    ) -> None:
        self.catalog = catalog
        self._doc_search = doc_search_fn
        self.tracker = UsageTracker(budget_usd=budget_usd, keep_calls=False)
        self._state_dir = state_dir
        self._client: LLMClient | None = None
        if client_factory is not None:
            self._client = _MeteredClient(client_factory(self.tracker))
        elif llm_config is not None:
            self._client = _MeteredClient(build_client(llm_config, self.tracker))
        self._lock = threading.Lock()
        self._checkpointer: Any = None
        self._store: Any = None

    # --- state ----------------------------------------------------------------------------

    def _open_state(self, filename: str, opener: Callable[[Path], Any]) -> Any:
        if self._state_dir is None:
            raise ToolError("This server has no state directory; memory is not available.")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        return opener(self._state_dir / filename)

    def checkpointer(self) -> Any:
        with self._lock:
            if self._checkpointer is None:
                self._checkpointer = self._open_state(CHECKPOINT_FILE, open_checkpointer)
            return self._checkpointer

    def store(self) -> Any:
        with self._lock:
            if self._store is None:
                self._store = self._open_state(MEMORY_FILE, open_memory_store)
            return self._store

    def close(self) -> None:
        """Close the state files (the server calls it on shutdown)."""
        with self._lock:
            if self._checkpointer is not None:
                self._checkpointer.conn.close()
                self._checkpointer = None
            if self._store is not None:
                self._store.conn.close()
                self._store = None

    # --- schema and SQL (no language model) -----------------------------------------------

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

    # --- agent v1 tools (no language model) ----------------------------------------------

    def calculate(self, expression: str) -> CalculatorResult:
        return calculate(expression)

    def search_documents(self, query: str, top_k: int) -> DocSearchResult:
        return self._doc_search(query, top_k=top_k)

    # --- language-model tools -------------------------------------------------------------

    @contextmanager
    def _llm_scope(self) -> Iterator[_Meter]:
        """Make the client current for this call, refuse it when the budget is spent, and
        turn failures into messages that are safe to show."""
        if self._client is None:
            raise ToolError(LLM_NOT_CONFIGURED)
        try:
            self.tracker.check()
        except BudgetExceeded as e:
            raise ToolError(_budget_message(e)) from None
        meter = _Meter()
        meter_token = _meter.set(meter)
        client_token = set_client(self._client)
        try:
            yield meter
        except BudgetExceeded as e:
            raise ToolError(_budget_message(e)) from None
        except (ToolError, UnknownDatabaseError):
            raise
        except Exception:
            logger.exception("language-model request failed")
            raise ToolError(LLM_FAILED) from None
        finally:
            reset_client(client_token)
            _meter.reset(meter_token)

    def ask(
        self,
        question: str,
        database: str,
        evidence: str = "",
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> AskResult:
        path = self.catalog.path(database)
        with self._llm_scope() as meter:
            options: dict[str, Any] = {}
            if thread_id is not None:
                # One conversation per (database, thread id): history never crosses databases.
                options.update(
                    thread_id=f"{database}:{thread_id}", checkpointer=self.checkpointer()
                )
            if user_id is not None:
                options.update(user_id=user_id, store=self.store())
            with tracing.trace(
                "mcp_ask",
                input={"question": question, "evidence": evidence, "database": database},
                session_id=thread_id,
                user_id=user_id,
            ) as handle:
                result = run_sql_agent(question, evidence, path, **options)
                final = next(
                    (a for a in reversed(result.attempts) if a.sql == result.final_sql),
                    result.attempts[-1] if result.attempts else None,
                )
                out = AskResult(
                    database=database,
                    answer=result.response or "No answer was produced.",
                    sql=result.final_sql,
                    columns=final.columns if final else [],
                    rows=final.preview if final else [],
                    row_count=final.row_count_text if final else "n/a",
                    blocked=final.blocked if final else False,
                    error=final.error if final else None,
                    llm_calls=meter.calls,
                    cost_usd=round(meter.cost_usd, 6),
                    trace_id=handle.id,
                )
                handle.update(
                    output={"sql": out.sql, "blocked": out.blocked, "llm_calls": out.llm_calls}
                )
        if tracing.enabled():
            tracing.flush()
        return out

    def teach(self, text: str, database: str, user_id: str) -> TeachResult:
        self.catalog.path(database)
        with self._llm_scope() as meter:
            taught = teach_text(text, user_id=user_id, db_id=database, store=self.store())
        return TeachResult(
            database=database,
            user_id=user_id,
            stored=[
                StoredDefinition(term=m.term, definition=m.definition) for m in taught.memories
            ],
            llm_calls=meter.calls,
            cost_usd=round(meter.cost_usd, 6),
        )

    def forget(self, database: str, user_id: str, term: str | None = None) -> ForgetResult:
        self.catalog.path(database)
        store = self.store()
        memories = load_memories(store, user_id, database)
        if term is not None:
            key = normalize_term(term)
            memories = [m for m in memories if m.key == key]
        ns = namespace(user_id, database)
        for memory in memories:
            store.delete(ns, memory.key)
        return ForgetResult(database=database, user_id=user_id, deleted=[m.term for m in memories])


def _budget_message(error: BudgetExceeded) -> str:
    return (
        f"The budget of ${error.budget:.2f} for language-model calls is used up. "
        "ask and teach are disabled until the server restarts; run_sql, describe_database "
        "and list_databases still work."
    )
