"""A scripted language-model client for the MCP server tests (no network)."""

from pathlib import Path
from typing import Any

from mcp_types import CallToolResult, TextContent

from src.llm import CallUsage, Completion, UsageTracker
from src.mcp_server.catalog import Catalog
from src.mcp_server.service import Service


def fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


class ScriptedClient:
    """Replays answers per tag (the last one repeats) and records every call."""

    def __init__(
        self,
        tracker: UsageTracker,
        script: dict[str, list[str]],
        cost_per_call: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.tracker = tracker
        self.script = {k: list(v) for k, v in script.items()}
        self.cost = cost_per_call
        self.error = error
        self.calls: list[tuple[str, str]] = []  # (tag, prefix + user)

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        if self.error is not None:
            raise self.error
        self.calls.append((tag, (prefix or "") + user))
        queue = self.script.get(tag, [""])
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        usage = CallUsage(tag=tag, input_tokens=10, output_tokens=5, cost_usd=self.cost)
        self.tracker.record(usage)
        return Completion(text=text, usage=usage)

    def prompts(self, tag: str) -> list[str]:
        return [text for t, text in self.calls if t == tag]


DEFAULT_SCRIPT = {
    "generate": [fence("SELECT name, pop FROM city ORDER BY pop DESC LIMIT 1")],
    "check": ["Looks right.\n\nOK"],
    "respond": ["Rome is the biggest city."],
    "memorize": ['[{"term": "big city", "definition": "a city with pop above 5"}]'],
}


class Rig:
    """A service over the fixture databases with a scripted client."""

    def __init__(
        self,
        databases_dir: Path,
        state_dir: Path,
        *,
        script: dict[str, list[str]] | None = None,
        cost_per_call: float = 0.0,
        budget_usd: float | None = None,
        error: Exception | None = None,
    ) -> None:
        self.clients: list[ScriptedClient] = []

        def factory(tracker: UsageTracker) -> ScriptedClient:
            client = ScriptedClient(tracker, script or DEFAULT_SCRIPT, cost_per_call, error)
            self.clients.append(client)
            return client

        self.service = Service(
            Catalog.scan(databases_dir),
            budget_usd=budget_usd,
            state_dir=state_dir,
            client_factory=factory,
        )

    @property
    def llm(self) -> ScriptedClient:
        return self.clients[-1]


def text_of(result: CallToolResult) -> str:
    assert result.content and isinstance(result.content[0], TextContent)
    return result.content[0].text


def data_of(result: CallToolResult) -> dict[str, Any]:
    assert result.structured_content is not None
    return result.structured_content
