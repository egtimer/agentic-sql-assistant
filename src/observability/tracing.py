"""Tracing through a context variable, with a Langfuse implementation and a no-op default.

The current trace/observation lives in a ``ContextVar``, never on graph state (LangGraph does
not carry private attributes between nodes). Code calls the module functions:

- :func:`trace` opens a trace (a root); :func:`span` opens an observation nested under the
  current one; :func:`record_generation` logs one finished LLM call under the current one;
  :func:`score` attaches a score to a trace id; :func:`flush` sends pending events.
- With no trace active, ``span`` and ``record_generation`` do nothing, so library code can
  call them unconditionally. Without Langfuse keys the whole module is a no-op: no client,
  no network, no serialization. Check :func:`active` before building expensive payloads.
- A failure inside the tracer is logged and swallowed: tracing never breaks a request.

The Langfuse implementation uses the v2 SDK (``langfuse<3``): ``client.trace``,
``observation.span`` / ``observation.generation``, ``span.end``, ``client.score``.
"""

import contextlib
import logging
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from src.config import settings

logger = logging.getLogger(__name__)

MAX_SPAN_TEXT = 20_000  # characters kept per string in span inputs/outputs
MAX_DEPTH = 6


class Handle(Protocol):
    """What ``trace`` and ``span`` yield."""

    @property
    def id(self) -> str | None: ...

    def update(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Set fields that are sent when the observation ends (a trace: sent at once)."""
        ...


class _NoopHandle:
    id = None

    def update(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        return None


NOOP_HANDLE = _NoopHandle()


def jsonable(value: Any, *, max_str: int | None = MAX_SPAN_TEXT, _depth: int = 0) -> Any:
    """Best-effort conversion to JSON-friendly data (pydantic, messages, sets, objects)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if max_str is None or len(value) <= max_str else value[:max_str] + "..."
    if _depth >= MAX_DEPTH:
        return repr(value)[:200]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, max_str=max_str, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, max_str=max_str, _depth=_depth + 1) for v in value]
    if hasattr(value, "content") and hasattr(value, "type"):  # LangChain message
        return {
            "type": str(value.type),
            "content": jsonable(value.content, max_str=max_str, _depth=_depth + 1),
        }
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump(), max_str=max_str, _depth=_depth + 1)
        except Exception:
            return repr(value)[:200]
    if hasattr(value, "value") and isinstance(getattr(value, "value", None), (str, int)):
        return value.value  # enum
    return repr(value)[:200]


class Tracer(Protocol):
    """Tracer interface; see :class:`NoopTracer` and :class:`LangfuseTracer`."""

    enabled: bool

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> contextlib.AbstractContextManager[Handle]: ...

    def span(
        self, name: str, *, input: Any = None, metadata: Mapping[str, Any] | None = None
    ) -> contextlib.AbstractContextManager[Handle]: ...

    def record_generation(
        self,
        name: str,
        *,
        model: str | None,
        input: Any,
        output: Any,
        usage: Mapping[str, int] | None = None,
        cost_usd: float | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None: ...

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None: ...

    def flush(self) -> None: ...


class NoopTracer:
    """Does nothing; the default when Langfuse is not configured."""

    enabled = False

    def trace(self, name: str, **_: Any) -> contextlib.AbstractContextManager[Handle]:
        return contextlib.nullcontext(NOOP_HANDLE)

    def span(self, name: str, **_: Any) -> contextlib.AbstractContextManager[Handle]:
        return contextlib.nullcontext(NOOP_HANDLE)

    def record_generation(self, name: str, **_: Any) -> None:
        return None

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None:
        return None

    def flush(self) -> None:
        return None


@dataclass
class _Node:
    """The current observation: a Langfuse trace or span client plus the trace id."""

    client: Any
    trace_id: str


_current: ContextVar[_Node | None] = ContextVar("tracing_current", default=None)


class _LangfuseHandle:
    def __init__(self, node: _Node | None, is_trace: bool) -> None:
        self._node = node
        self._is_trace = is_trace
        self._fields: dict[str, Any] = {}

    @property
    def id(self) -> str | None:
        return self._node.trace_id if self._is_trace and self._node else None

    def update(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._node is None:
            return
        fields = {
            "output": None if output is None else jsonable(output),
            "metadata": None if metadata is None else jsonable(metadata),
            "level": level,
            "status_message": status_message,
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if self._is_trace:
            try:
                self._node.client.update(**{k: v for k, v in fields.items() if k != "level"})
                if "status_message" in fields:
                    self._node.client.update(metadata={"status_message": status_message})
            except Exception as e:
                logger.warning("tracing: trace update failed: %s", e)
        else:
            self._fields.update(fields)

    def end(self) -> None:
        if self._node is None or self._is_trace:
            return
        try:
            self._node.client.end(**self._fields)
        except Exception as e:
            logger.warning("tracing: span end failed: %s", e)


class LangfuseTracer:
    """Sends traces to Langfuse with the v2 Python SDK (``client`` is a ``Langfuse``)."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[Handle]:
        node: _Node | None = None
        try:
            trace_id = str(uuid.uuid4())
            kwargs: dict[str, Any] = {
                "id": trace_id,
                "name": name,
                "input": jsonable(input),
                "metadata": jsonable(metadata or {}),
            }
            if tags:
                kwargs["tags"] = list(tags)
            if session_id:
                kwargs["session_id"] = session_id
            if user_id:
                kwargs["user_id"] = user_id
            node = _Node(self._client.trace(**kwargs), trace_id)
        except Exception as e:
            logger.warning("tracing: could not create trace %s: %s", name, e)
        if node is None:
            yield NOOP_HANDLE
            return
        handle = _LangfuseHandle(node, is_trace=True)
        token = _current.set(node)
        try:
            yield handle
        except BaseException as e:
            handle.update(metadata={"error": f"{type(e).__name__}: {e}"})
            raise
        finally:
            _current.reset(token)

    @contextmanager
    def span(
        self, name: str, *, input: Any = None, metadata: Mapping[str, Any] | None = None
    ) -> Iterator[Handle]:
        parent = _current.get()
        node: _Node | None = None
        if parent is not None:
            try:
                kwargs: dict[str, Any] = {"name": name, "input": jsonable(input)}
                if metadata:
                    kwargs["metadata"] = jsonable(metadata)
                node = _Node(parent.client.span(**kwargs), parent.trace_id)
            except Exception as e:
                logger.warning("tracing: could not create span %s: %s", name, e)
        if node is None:
            yield NOOP_HANDLE
            return
        handle = _LangfuseHandle(node, is_trace=False)
        token = _current.set(node)
        try:
            yield handle
        except BaseException as e:
            handle.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
        finally:
            _current.reset(token)
            handle.end()

    def record_generation(
        self,
        name: str,
        *,
        model: str | None,
        input: Any,
        output: Any,
        usage: Mapping[str, int] | None = None,
        cost_usd: float | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        parent = _current.get()
        if parent is None:
            return
        try:
            kwargs: dict[str, Any] = {
                "name": name,
                "model": model,
                "input": jsonable(input, max_str=None),
                "output": jsonable(output, max_str=None),
                "start_time": start,
                "end_time": end or datetime.now(UTC),
                "metadata": jsonable(metadata or {}),
            }
            if usage is not None:
                kwargs["usage_details"] = {k: int(v) for k, v in usage.items()}
            if cost_usd is not None:
                kwargs["cost_details"] = {"total": float(cost_usd)}
            if level:
                kwargs["level"] = level
            if status_message:
                kwargs["status_message"] = status_message
            parent.client.generation(**kwargs)
        except Exception as e:
            logger.warning("tracing: could not record generation %s: %s", name, e)

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None:
        if not trace_id:
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception as e:
            logger.warning("tracing: could not post score %s: %s", name, e)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as e:
            logger.warning("tracing: flush failed: %s", e)


_tracer: Tracer | None = None
_lock = threading.Lock()


def configured() -> bool:
    """True when both Langfuse keys are set."""
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _build_default() -> Tracer:
    if not configured():
        return NoopTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (%s)", settings.langfuse_host)
        return LangfuseTracer(client)
    except Exception as e:
        logger.warning("Langfuse could not be initialised, tracing disabled: %s", e)
        return NoopTracer()


def get_tracer() -> Tracer:
    """The process-wide tracer, built from settings on first use."""
    global _tracer
    if _tracer is None:
        with _lock:
            if _tracer is None:
                _tracer = _build_default()
    return _tracer


def set_tracer(tracer: Tracer | None) -> Tracer | None:
    """Replace the tracer (None: rebuild from settings on next use). Returns the previous one."""
    global _tracer
    with _lock:
        previous, _tracer = _tracer, tracer
    return previous


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    """Use ``tracer`` in the whole process for the duration of the block."""
    previous = set_tracer(tracer)
    try:
        yield tracer
    finally:
        set_tracer(previous)


def enabled() -> bool:
    """True when a real tracer is configured (a trace may still not be open)."""
    return get_tracer().enabled


def active() -> bool:
    """True when an observation is open in this context: payloads are worth building."""
    return _current.get() is not None and get_tracer().enabled


def trace(
    name: str,
    *,
    input: Any = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> contextlib.AbstractContextManager[Handle]:
    """Open a trace (always a new root) for the duration of the block."""
    return get_tracer().trace(
        name, input=input, metadata=metadata, tags=tags, session_id=session_id, user_id=user_id
    )


def span(
    name: str, *, input: Any = None, metadata: Mapping[str, Any] | None = None
) -> contextlib.AbstractContextManager[Handle]:
    """Open an observation under the current one (no-op when no trace is active)."""
    tracer = get_tracer()
    if not tracer.enabled:
        return contextlib.nullcontext(NOOP_HANDLE)
    return tracer.span(name, input=input, metadata=metadata)


def trace_or_span(
    name: str,
    *,
    input: Any = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> contextlib.AbstractContextManager[Handle]:
    """A span when a trace is active, otherwise a new trace (entry points of the agents)."""
    if active():
        return span(name, input=input, metadata=metadata)
    return trace(
        name, input=input, metadata=metadata, tags=tags, session_id=session_id, user_id=user_id
    )


def record_generation(
    name: str,
    *,
    model: str | None,
    input: Any,
    output: Any,
    usage: Mapping[str, int] | None = None,
    cost_usd: float | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Log one finished LLM call under the current observation (no-op without a trace)."""
    get_tracer().record_generation(
        name,
        model=model,
        input=input,
        output=output,
        usage=usage,
        cost_usd=cost_usd,
        start=start,
        end=end,
        metadata=metadata,
        level=level,
        status_message=status_message,
    )


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach a numeric score to a trace."""
    get_tracer().score(trace_id, name, value, comment)


def flush() -> None:
    """Send pending events (call before the process exits)."""
    tracer = _tracer
    if tracer is not None:
        tracer.flush()
