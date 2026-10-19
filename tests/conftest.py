"""Markers for tests that need external resources, and the skip rules that apply them.

- ``requires_bird``: BIRD Mini-Dev data under ``data/external/`` (``make bird-data``). Skipped
  when the data is missing, which is always the case in CI.
- ``requires_ollama``: a running Ollama server. Runs only with ``RUN_OLLAMA_TESTS=1`` and
  never in CI.
- ``requires_api``: a paid LLM API (``ANTHROPIC_API_KEY``). Runs only with ``RUN_API_TESTS=1``
  and never in CI; API evaluations run in the manual/nightly workflow instead.
"""

import os
import sqlite3
import sys

import pytest

from evals.text2sql.data import db_path

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
    return None


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [
        f"sqlite {sqlite3.sqlite_version} (python {sys.version.split()[0]})",
        f"CI={IN_CI} BIRD data={'present' if _bird_available() else 'missing'}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in ("requires_bird", "requires_ollama", "requires_api"):
            if item.get_closest_marker(marker) is not None:
                reason = _skip_reason(marker)
                if reason is not None:
                    item.add_marker(pytest.mark.skip(reason=reason))
