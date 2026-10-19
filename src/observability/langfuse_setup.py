"""Compatibility layer for the agent v1 nodes: ``trace_node`` and ``flush``.

All tracing logic lives in :mod:`src.observability.tracing` (context-variable based).
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from src.observability import tracing
from src.observability.tracing import flush

__all__ = ["flush", "trace_node"]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def trace_node(node_name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that records a graph node as a span of the current trace.

    The span input is the node's state (without messages) and its output the dict the node
    returns. The span nests under whatever observation is open in the current context, so it
    works across LangGraph nodes without carrying anything on the state. No-op when no trace
    is active.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not tracing.active():
                return func(*args, **kwargs)
            state = args[0] if args else kwargs.get("state")
            with tracing.span(node_name, input=_state_input(state)) as handle:
                result = func(*args, **kwargs)
                handle.update(output=result)
                return result

        return wrapper

    return decorator


def _state_input(state: Any) -> dict[str, Any]:
    """Serializable fields of the agent state (messages left out)."""
    if state is None:
        return {}
    try:
        if hasattr(state, "model_dump"):
            return dict(state.model_dump(exclude={"messages"}, mode="json"))
    except Exception:
        logger.debug("could not serialize state", exc_info=True)
    return {"type": type(state).__name__}
