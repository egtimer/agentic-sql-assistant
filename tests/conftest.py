"""Markers for tests that need external resources, and the skip rules that apply them.

- ``requires_bird``: BIRD Mini-Dev data under ``data/external/`` (``make bird-data``). Skipped
  when the data is missing, which is always the case in CI.
- ``requires_ollama``: a running Ollama server. Runs only with ``RUN_OLLAMA_TESTS=1`` and
  never in CI.
- ``requires_api``: a paid LLM API (``ANTHROPIC_API_KEY``). Runs only with ``RUN_API_TESTS=1``
  and never in CI; API evaluations run in the manual/nightly workflow instead.
- ``requires_langfuse``: a running Langfuse server with the keys in the environment or
  ``.env`` (``make langfuse-up``). Runs only with ``RUN_LANGFUSE_TESTS=1`` and never in CI.
"""

import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from langfuse import Langfuse
from langfuse.serializer import EventSerializer

from evals.text2sql.data import db_path
from src.observability import tracing
from src.observability.tracing import LangfuseTracer

IN_CI = os.environ.get("CI", "").lower() == "true"


def _bird_available() -> bool:
    return db_path("student_club").exists()


def _skip_reason(marker: str) -> str | None:
    if marker == "requires_bird" and not _bird_available():
        return "requires_bird: BIRD Mini-Dev data missing (run `make bird-data`)"
    if marker == "requires_ollama":
        if IN_CI:
            return "requires_ollama: never runs in CI"
        if os.environ.get("RUN_OLLAMA_TESTS") != "1":
            return "requires_ollama: set RUN_OLLAMA_TESTS=1 with a running Ollama server"
    if marker == "requires_api":
        if IN_CI:
            return "requires_api: never runs in CI (see the manual/nightly API eval workflow)"
        if os.environ.get("RUN_API_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"):
            return "requires_api: set RUN_API_TESTS=1 and ANTHROPIC_API_KEY; spends money"
    if marker == "requires_langfuse":
        if IN_CI:
            return "requires_langfuse: never runs in CI"
        if os.environ.get("RUN_LANGFUSE_TESTS") != "1":
            return "requires_langfuse: set RUN_LANGFUSE_TESTS=1 with a running Langfuse server"
    return None


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [
        f"sqlite {sqlite3.sqlite_version} (python {sys.version.split()[0]})",
        f"CI={IN_CI} BIRD data={'present' if _bird_available() else 'missing'}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in ("requires_bird", "requires_ollama", "requires_api", "requires_langfuse"):
            if item.get_closest_marker(marker) is not None:
                reason = _skip_reason(marker)
                if reason is not None:
                    item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def events() -> Iterator[list[dict[str, Any]]]:
    """A LangfuseTracer on a real SDK client whose events are captured, not sent."""
    client = Langfuse(public_key="pk-test", secret_key="sk-test", host="http://127.0.0.1:1")
    captured: list[dict[str, Any]] = []

    def capture(event: dict[str, Any]) -> None:
        # what the SDK's ingestion consumer does before sending an event
        if hasattr(event["body"], "dict"):
            event["body"] = event["body"].dict(exclude_none=True)
        captured.append(json.loads(json.dumps(event, cls=EventSerializer)))

    client.task_manager.add_task = capture  # type: ignore[method-assign]
    with tracing.use_tracer(LangfuseTracer(client)):
        yield captured


@pytest.fixture(autouse=True)
def _tracing_off_by_default(request: pytest.FixtureRequest) -> Iterator[None]:
    """No test talks to a real Langfuse because of keys in .env; tests that want traces use
    the ``events`` fixture (captured, never sent) or the ``requires_langfuse`` marker."""
    if request.node.get_closest_marker("requires_langfuse") is not None:
        yield
        return
    with tracing.use_tracer(tracing.NoopTracer()):
        yield
