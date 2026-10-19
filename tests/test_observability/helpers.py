"""Helpers for the tracing tests."""

from typing import Any


def of_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Bodies of the captured Langfuse events of one type (see the ``events`` fixture)."""
    return [e["body"] for e in events if e["type"] == kind]
