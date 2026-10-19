"""Provider-agnostic LLM client: one ``complete`` method with usage tracking."""

import logging
import threading
import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.llm.config import LLMConfig
from src.llm.usage import CallUsage, UsageTracker, cost_of
from src.observability import tracing

logger = logging.getLogger(__name__)

FAILURE_STOP_REASONS = frozenset({"refusal", "max_tokens"})


@dataclass
class Completion:
    """Text returned by the model plus the usage of the call that produced it."""

    text: str
    usage: CallUsage

    @property
    def stop_reason(self) -> str | None:
        return self.usage.stop_reason

    @property
    def is_failure(self) -> bool:
        """True when the model refused or was cut off, so the text is unreliable."""
        return self.usage.stop_reason in FAILURE_STOP_REASONS


def trace_completion(
    model: str,
    system: str | None,
    prefix: str | None,
    user: str,
    completion: Completion | None,
    start: datetime,
    *,
    tag: str,
    error: BaseException | None = None,
) -> None:
    """Record one finished (or failed) LLM call as a generation under the current trace.

    Cheap no-op when no trace is open. Input is the full prompt (system, then ``prefix +
    user``); usage carries the cache read/write tokens and the cost is ``cost_of`` (what the
    run accounts), not a Langfuse price table.
    """
    if not tracing.active():
        return
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": (prefix or "") + user})
    if completion is None:
        tracing.record_generation(
            tag,
            model=model,
            input=messages,
            output=None,
            start=start,
            end=datetime.now(UTC),
            level="ERROR",
            status_message=f"{type(error).__name__}: {error}",
        )
        return
    usage = completion.usage
    tracing.record_generation(
        tag,
        model=model,
        input=messages,
        output=completion.text,
        usage={
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cache_read": usage.cache_read_tokens,
            "cache_write": usage.cache_write_tokens,
        },
        cost_usd=usage.cost_usd,
        start=start,
        end=datetime.now(UTC),
        metadata={
            "tag": tag,
            "stop_reason": usage.stop_reason,
            "latency_ms": round(usage.latency_ms, 1),
        },
        level="WARNING" if completion.is_failure else None,
    )


class LLMClient(Protocol):
    """Anything that can complete a prompt."""

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        """Complete a prompt. The model sees ``prefix + user`` (plain concatenation).

        ``prefix`` is the part that is stable across calls; providers with explicit prompt
        caching may cache it, the others just send ``prefix + user``.
        """
        ...


class OllamaClient:
    """Local models through ``langchain_ollama.ChatOllama``."""

    def __init__(self, config: LLMConfig, tracker: UsageTracker | None = None) -> None:
        self.config = config
        self.tracker = tracker or UsageTracker()
        self._llm: ChatOllama | None = None

    def _get_llm(self) -> ChatOllama:
        if self._llm is None:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "temperature": 0 if self.config.temperature is None else self.config.temperature,
            }
            if self.config.base_url is not None:
                kwargs["base_url"] = self.config.base_url
            if self.config.num_ctx is not None:
                kwargs["num_ctx"] = self.config.num_ctx
            if self.config.max_tokens is not None:
                kwargs["num_predict"] = self.config.max_tokens
            self._llm = ChatOllama(**kwargs)
        return self._llm

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        user = (prefix or "") + user
        messages: list[BaseMessage] = []
        if system is not None:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user))

        started = datetime.now(UTC)
        start = time.perf_counter()
        try:
            response = self._get_llm().invoke(messages)
        except Exception as e:
            trace_completion(self.config.model, system, None, user, None, started, tag=tag, error=e)
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.content if isinstance(response.content, str) else str(response.content)
        meta: Mapping[str, Any] = response.usage_metadata or {}
        done_reason = response.response_metadata.get("done_reason")
        usage = CallUsage(
            tag=tag,
            input_tokens=int(meta.get("input_tokens", 0) or 0),
            output_tokens=int(meta.get("output_tokens", 0) or 0),
            latency_ms=latency_ms,
            stop_reason="max_tokens" if done_reason == "length" else done_reason,
        )
        usage.cost_usd = cost_of(usage, self.config.pricing)
        self.tracker.record(usage)
        completion = Completion(text=text, usage=usage)
        trace_completion(self.config.model, system, None, user, completion, started, tag=tag)
        return completion


class AnthropicClient:
    """Claude models through the official ``anthropic`` SDK.

    Credentials come from the environment (``ANTHROPIC_API_KEY``) and are never passed
    explicitly. ``temperature`` is not sent: non-default values are rejected by recent models.
    """

    def __init__(
        self,
        config: LLMConfig,
        tracker: UsageTracker | None = None,
        sdk_client: Any = None,
    ) -> None:
        self.config = config
        self.tracker = tracker or UsageTracker()
        self._sdk = sdk_client

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.Anthropic()
        return self._sdk

    def _user_content(self, user: str, prefix: str | None) -> str | list[dict[str, Any]]:
        """Message content: two blocks with the cache breakpoint after the stable prefix.

        ``cache_control`` goes only on the prefix block, never on the per-question part.
        """
        if self.config.prompt_caching and prefix:
            return [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user},
            ]
        return (prefix or "") + user

    def complete(
        self, system: str | None, user: str, *, tag: str, prefix: str | None = None
    ) -> Completion:
        self.tracker.check()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": self._user_content(user, prefix)}],
        }
        if system is not None:
            kwargs["system"] = system
        if self.config.effort:
            kwargs["output_config"] = {"effort": self.config.effort}

        started = datetime.now(UTC)
        start = time.perf_counter()
        try:
            response = self._get_sdk().messages.create(**kwargs)
        except Exception as e:
            trace_completion(
                self.config.model, system, prefix, user, None, started, tag=tag, error=e
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        raw = response.usage
        usage = CallUsage(
            tag=tag,
            input_tokens=raw.input_tokens or 0,
            output_tokens=raw.output_tokens or 0,
            cache_read_tokens=getattr(raw, "cache_read_input_tokens", None) or 0,
            cache_write_tokens=getattr(raw, "cache_creation_input_tokens", None) or 0,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
        )
        usage.cost_usd = cost_of(usage, self.config.pricing)
        self.tracker.record(usage)
        if usage.stop_reason in FAILURE_STOP_REASONS:
            logger.warning("LLM call %s ended with stop_reason=%s", tag, usage.stop_reason)
        completion = Completion(text=text, usage=usage)
        trace_completion(self.config.model, system, prefix, user, completion, started, tag=tag)
        return completion


def build_client(config: LLMConfig, tracker: UsageTracker | None = None) -> LLMClient:
    """Build the client for ``config.provider``."""
    if config.provider == "anthropic":
        return AnthropicClient(config, tracker)
    return OllamaClient(config, tracker)


_client_var: ContextVar[LLMClient | None] = ContextVar("llm_client", default=None)
_default_client: LLMClient | None = None
_default_lock = threading.Lock()


def _build_default_client() -> LLMClient:
    config = LLMConfig(
        provider="ollama",
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=settings.ollama_num_ctx,
    )
    return OllamaClient(config, UsageTracker(keep_calls=False))


def get_client() -> LLMClient:
    """Return the client for the current context, or the Ollama default from settings."""
    global _default_client
    client = _client_var.get()
    if client is not None:
        return client
    with _default_lock:
        if _default_client is None:
            _default_client = _build_default_client()
        return _default_client


def set_client(client: LLMClient | None) -> Token[LLMClient | None]:
    """Use ``client`` in the current thread/context (None restores the default)."""
    return _client_var.set(client)


def reset_client(token: Token[LLMClient | None]) -> None:
    """Undo a previous :func:`set_client`."""
    _client_var.reset(token)
