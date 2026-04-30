"""Langfuse tracing integration for agent observability."""

import functools
import logging
import time
from typing import Any, Callable, ParamSpec, TypeVar

from src.config import settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

_langfuse_client = None


def get_langfuse():
    """Return a shared Langfuse client instance, or None if not configured."""
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.debug("Langfuse not configured — tracing disabled")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse client initialized at %s", settings.langfuse_host)
        return _langfuse_client
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s", e)
        return None


def create_trace(name: str, query: str, metadata: dict[str, Any] | None = None):
    """Create a new Langfuse trace for an agent invocation.

    Args:
        name: Trace name (e.g., "agent_query").
        query: The user's input query.
        metadata: Optional metadata to attach to the trace.

    Returns:
        A Langfuse trace object, or None if tracing is disabled.
    """
    client = get_langfuse()
    if client is None:
        return None

    try:
        return client.trace(
            name=name,
            input={"query": query},
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning("Failed to create trace: %s", e)
        return None


def trace_node(node_name: str) -> Callable:
    """Decorator that wraps a graph node function with Langfuse span tracking.

    Each node execution is recorded as a span within the active trace,
    capturing input state, output, duration, and any errors.

    Args:
        node_name: Name for the span (e.g., "reason", "execute", "validate").

    Returns:
        Decorator function.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            state = args[0] if args else kwargs.get("state")

            trace = None
            if state is not None and hasattr(state, "__dict__"):
                trace = getattr(state, "_trace", None)

            span = None
            if trace is not None:
                try:
                    span = trace.span(
                        name=node_name,
                        input=_safe_serialize_state(state),
                    )
                except Exception as e:
                    logger.debug("Failed to create span for %s: %s", node_name, e)

            start_time = time.perf_counter()
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.debug("Node %s completed in %.1fms", node_name, duration_ms)

                if span is not None:
                    try:
                        span.end(
                            output=result if error is None else {"error": error},
                            level="ERROR" if error else "DEFAULT",
                        )
                    except Exception:
                        pass

        return wrapper
    return decorator


def _safe_serialize_state(state: Any) -> dict[str, Any]:
    """Extract serializable fields from agent state for tracing."""
    if state is None:
        return {}

    try:
        if hasattr(state, "model_dump"):
            data = state.model_dump(exclude={"messages"})
            return {k: v for k, v in data.items() if _is_serializable(v)}
    except Exception:
        pass

    return {"type": type(state).__name__}


def _is_serializable(value: Any) -> bool:
    """Check if a value can be safely serialized to JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_serializable(v) for k, v in value.items())
    return False


def flush() -> None:
    """Flush any pending Langfuse events. Call before shutdown."""
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse: %s", e)
