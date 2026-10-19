"""Token usage, cost accounting and the hard budget cap."""

import contextlib
import json
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.llm.config import Pricing

_call_context: ContextVar[dict[str, Any] | None] = ContextVar("llm_call_context", default=None)


@dataclass
class CallUsage:
    """Usage and cost of a single LLM call."""

    tag: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    cost_usd: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-serializable dict; context keys (run_id, system, ...) are inlined."""
        record = asdict(self)
        context = record.pop("context")
        return {**context, **record}


class BudgetExceeded(Exception):  # noqa: N818  (name fixed by the spec)
    """Raised before an LLM call when spending has reached the budget."""

    def __init__(self, spent: float, budget: float) -> None:
        super().__init__(f"Budget of ${budget:.4f} reached (spent ${spent:.4f})")
        self.spent = spent
        self.budget = budget


def cost_of(usage: CallUsage, pricing: Pricing) -> float:
    """Cost in USD of one call. Long-context rates apply above the pricing threshold."""
    prompt_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    input_rate = pricing.input_per_mtok
    output_rate = pricing.output_per_mtok
    cache_read_rate = pricing.cache_read_per_mtok
    cache_write_rate = pricing.cache_write_per_mtok
    if (
        pricing.long_context_threshold is not None
        and prompt_tokens > pricing.long_context_threshold
    ):
        if pricing.long_input_per_mtok is not None:
            input_rate = pricing.long_input_per_mtok
        if pricing.long_output_per_mtok is not None:
            output_rate = pricing.long_output_per_mtok
        if pricing.long_cache_read_per_mtok is not None:
            cache_read_rate = pricing.long_cache_read_per_mtok
        if pricing.long_cache_write_per_mtok is not None:
            cache_write_rate = pricing.long_cache_write_per_mtok
    total = (
        usage.input_tokens * input_rate
        + usage.output_tokens * output_rate
        + usage.cache_read_tokens * cache_read_rate
        + usage.cache_write_tokens * cache_write_rate
    )
    return total / 1_000_000


@contextlib.contextmanager
def call_context(**values: Any) -> Iterator[None]:
    """Tag every call made in this thread/context with extra metadata (run_id, question_id...)."""
    merged = {**(_call_context.get() or {}), **values}
    token = _call_context.set(merged)
    try:
        yield
    finally:
        _call_context.reset(token)


def current_context() -> dict[str, Any]:
    """Return a copy of the metadata set by :func:`call_context`."""
    return dict(_call_context.get() or {})


class UsageTracker:
    """Thread-safe accumulator of call usage with an optional budget and JSONL sink.

    ``check()`` is called before each call and ``record()`` after it, so with several
    workers the overshoot is at most one in-flight call per worker.
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        sink_path: Path | str | None = None,
        keep_calls: bool = True,
    ) -> None:
        self.budget_usd = budget_usd
        self.sink_path = Path(sink_path) if sink_path is not None else None
        self._keep_calls = keep_calls
        self._lock = threading.Lock()
        self._calls: list[CallUsage] = []
        self._total_cost = 0.0

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def calls(self) -> list[CallUsage]:
        with self._lock:
            return list(self._calls)

    def check(self) -> None:
        """Raise BudgetExceeded if the budget has already been reached."""
        if self.budget_usd is None:
            return
        spent = self.total_cost
        if spent >= self.budget_usd:
            raise BudgetExceeded(spent, self.budget_usd)

    def record(self, call: CallUsage) -> None:
        """Add a finished call: tag it with the current context, sum its cost, append to sink."""
        call.context = {**current_context(), **call.context}
        with self._lock:
            self._total_cost += call.cost_usd
            if self._keep_calls:
                self._calls.append(call)
            if self.sink_path is not None:
                self.sink_path.parent.mkdir(parents=True, exist_ok=True)
                with self.sink_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(call.to_record(), ensure_ascii=False) + "\n")
