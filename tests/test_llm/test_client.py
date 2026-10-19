"""Tests for the Ollama and Anthropic clients (both backends mocked)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm import (
    AnthropicClient,
    BudgetExceeded,
    Completion,
    LLMConfig,
    OllamaClient,
    Pricing,
    UsageTracker,
    build_client,
    get_client,
    reset_client,
    set_client,
)
from src.llm.client import _build_default_client

PRICING = Pricing(
    input_per_mtok=0.10,
    output_per_mtok=0.50,
    cache_read_per_mtok=0.01,
    cache_write_per_mtok=0.125,
    long_context_threshold=100_000,
    long_input_per_mtok=0.50,
    long_output_per_mtok=2.50,
)


def anthropic_config(**overrides: Any) -> LLMConfig:
    params: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "max_tokens": 16000,
        "effort": "low",
        "pricing": PRICING,
    }
    params.update(overrides)
    return LLMConfig(**params)


def sdk_response(
    blocks: list[Any] | None = None,
    stop_reason: str = "end_turn",
    **usage: Any,
) -> SimpleNamespace:
    fields = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }
    fields.update(usage)
    return SimpleNamespace(
        content=blocks if blocks is not None else [SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(**fields),
        stop_reason=stop_reason,
    )


def fake_sdk(response: SimpleNamespace) -> MagicMock:
    sdk = MagicMock()
    sdk.messages.create.return_value = response
    return sdk


class TestAnthropicClient:
    def test_request_shape(self) -> None:
        sdk = fake_sdk(sdk_response())
        client = AnthropicClient(anthropic_config(), sdk_client=sdk)
        client.complete("be brief", "question?", tag="baseline")
        kwargs = sdk.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-5-5"
        assert kwargs["max_tokens"] == 16000
        assert kwargs["system"] == "be brief"
        assert kwargs["messages"] == [{"role": "user", "content": "question?"}]
        assert kwargs["output_config"] == {"effort": "low"}
        assert "temperature" not in kwargs

    def test_temperature_never_sent(self) -> None:
        sdk = fake_sdk(sdk_response())
        client = AnthropicClient(anthropic_config(temperature=0.7), sdk_client=sdk)
        client.complete(None, "q", tag="t")
        assert "temperature" not in sdk.messages.create.call_args.kwargs

    def test_system_and_effort_omitted_when_unset(self) -> None:
        sdk = fake_sdk(sdk_response())
        client = AnthropicClient(anthropic_config(effort=None), sdk_client=sdk)
        client.complete(None, "q", tag="t")
        kwargs = sdk.messages.create.call_args.kwargs
        assert "system" not in kwargs
        assert "output_config" not in kwargs

    def test_usage_mapping_and_cost(self) -> None:
        response = sdk_response(
            input_tokens=1000,
            output_tokens=2000,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=100,
        )
        tracker = UsageTracker()
        client = AnthropicClient(anthropic_config(), tracker, sdk_client=fake_sdk(response))
        completion = client.complete(None, "q", tag="reason")
        u = completion.usage
        assert (u.input_tokens, u.output_tokens) == (1000, 2000)
        assert (u.cache_read_tokens, u.cache_write_tokens) == (500, 100)
        assert u.tag == "reason"
        assert u.stop_reason == "end_turn"
        assert u.latency_ms >= 0
        expected = (1000 * 0.10 + 2000 * 0.50 + 500 * 0.01 + 100 * 0.125) / 1_000_000
        assert u.cost_usd == pytest.approx(expected)
        assert tracker.total_cost == pytest.approx(expected)
        assert tracker.calls == [u]

    def test_none_cache_fields_are_zero(self) -> None:
        client = AnthropicClient(anthropic_config(), sdk_client=fake_sdk(sdk_response()))
        u = client.complete(None, "q", tag="t").usage
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0

    def test_long_context_tier(self) -> None:
        response = sdk_response(input_tokens=150_000, output_tokens=1000)
        client = AnthropicClient(anthropic_config(), sdk_client=fake_sdk(response))
        u = client.complete(None, "q", tag="t").usage
        assert u.cost_usd == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1_000_000)

    def test_text_skips_thinking_blocks(self) -> None:
        blocks = [
            SimpleNamespace(type="thinking", thinking="hmm", text="SHOULD NOT APPEAR"),
            SimpleNamespace(type="text", text="SELECT "),
            SimpleNamespace(type="redacted_thinking", data="x"),
            SimpleNamespace(type="text", text="1"),
        ]
        client = AnthropicClient(anthropic_config(), sdk_client=fake_sdk(sdk_response(blocks)))
        completion = client.complete(None, "q", tag="t")
        assert completion.text == "SELECT 1"

    @pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
    def test_failure_stop_reasons_are_visible(self, reason: str) -> None:
        sdk = fake_sdk(sdk_response(stop_reason=reason))
        completion = AnthropicClient(anthropic_config(), sdk_client=sdk).complete(
            None, "q", tag="t"
        )
        assert completion.stop_reason == reason
        assert completion.is_failure is True
        assert completion.text == "hi"

    def test_normal_stop_is_not_failure(self) -> None:
        client = AnthropicClient(anthropic_config(), sdk_client=fake_sdk(sdk_response()))
        assert client.complete(None, "q", tag="t").is_failure is False

    def test_budget_abort_before_call(self) -> None:
        sdk = fake_sdk(sdk_response(input_tokens=1_000_000, output_tokens=1_000_000))
        tracker = UsageTracker(budget_usd=0.5)
        client = AnthropicClient(anthropic_config(), tracker, sdk_client=sdk)
        client.complete(None, "q", tag="t")  # costs 0.60 USD, reaches the budget
        with pytest.raises(BudgetExceeded):
            client.complete(None, "q", tag="t")
        assert sdk.messages.create.call_count == 1

    def test_sdk_built_lazily_from_env(self) -> None:
        with patch("anthropic.Anthropic") as ctor:
            ctor.return_value = fake_sdk(sdk_response())
            client = AnthropicClient(anthropic_config())
            ctor.assert_not_called()
            client.complete(None, "q", tag="t")
            ctor.assert_called_once_with()

    def test_build_client_dispatch(self) -> None:
        assert isinstance(build_client(anthropic_config()), AnthropicClient)
        ollama = LLMConfig(provider="ollama", model="m")
        assert isinstance(build_client(ollama), OllamaClient)


class TestPromptCaching:
    PREFIX = "SCHEMA...\n\n"

    def _sent(self, config: LLMConfig, prefix: str | None) -> Any:
        sdk = fake_sdk(sdk_response())
        AnthropicClient(config, sdk_client=sdk).complete(None, "Q?", tag="t", prefix=prefix)
        return sdk.messages.create.call_args.kwargs["messages"][0]["content"]

    def test_two_blocks_with_cache_control_on_prefix_only(self) -> None:
        content = self._sent(anthropic_config(prompt_caching=True), self.PREFIX)
        assert content == [
            {"type": "text", "text": self.PREFIX, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Q?"},
        ]
        assert "cache_control" not in content[1]

    def test_single_string_when_caching_disabled(self) -> None:
        assert self._sent(anthropic_config(), self.PREFIX) == self.PREFIX + "Q?"

    @pytest.mark.parametrize("prefix", [None, ""])
    def test_single_string_without_prefix(self, prefix: str | None) -> None:
        assert self._sent(anthropic_config(prompt_caching=True), prefix) == "Q?"

    def test_caching_is_off_by_default(self) -> None:
        assert anthropic_config().prompt_caching is False

    def test_cache_tokens_recorded_and_priced(self) -> None:
        sdk = fake_sdk(
            sdk_response(
                input_tokens=10, cache_read_input_tokens=1000, cache_creation_input_tokens=2000
            )
        )
        u = (
            AnthropicClient(anthropic_config(prompt_caching=True), sdk_client=sdk)
            .complete(None, "q", tag="t", prefix="p")
            .usage
        )
        assert (u.cache_read_tokens, u.cache_write_tokens) == (1000, 2000)
        assert u.cost_usd == pytest.approx(
            (10 * 0.10 + 50 * 0.50 + 1000 * 0.01 + 2000 * 0.125) / 1e6
        )

    def test_haiku_config_enables_caching(self) -> None:
        from pathlib import Path

        import yaml

        path = Path(__file__).parents[2] / "configs" / "llm" / "claude-haiku-5-5.yaml"
        config = LLMConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert config.prompt_caching is True


class TestOllamaClient:
    def test_prefix_and_user_are_concatenated(self) -> None:
        config = LLMConfig(provider="ollama", model="m")
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            OllamaClient(config).complete("sys", "Q?", tag="t", prefix="SCHEMA\n\n")
        sent = ctor.return_value.invoke.call_args.args[0]
        assert [m.content for m in sent] == ["sys", "SCHEMA\n\nQ?"]

    def _config(self, **overrides: Any) -> LLMConfig:
        params: dict[str, Any] = {
            "provider": "ollama",
            "model": "qwen2.5:3b",
            "base_url": "http://localhost:11434",
            "temperature": 0,
            "num_ctx": 8192,
            "max_tokens": 1024,
        }
        params.update(overrides)
        return LLMConfig(**params)

    def test_usage_text_and_messages(self) -> None:
        message = AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
            response_metadata={"done_reason": "stop"},
        )
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = message
            tracker = UsageTracker()
            client = OllamaClient(self._config(), tracker)
            completion = client.complete("sys", "usr", tag="reason")
        ctor.assert_called_once_with(
            model="qwen2.5:3b",
            temperature=0,
            base_url="http://localhost:11434",
            num_ctx=8192,
            num_predict=1024,
        )
        sent = ctor.return_value.invoke.call_args.args[0]
        assert isinstance(sent[0], SystemMessage) and sent[0].content == "sys"
        assert isinstance(sent[1], HumanMessage) and sent[1].content == "usr"
        assert completion.text == "answer"
        assert completion.usage.input_tokens == 12
        assert completion.usage.output_tokens == 7
        assert completion.usage.stop_reason == "stop"
        assert completion.usage.cost_usd == 0.0
        assert tracker.calls == [completion.usage]

    def test_no_system_message_when_none(self) -> None:
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            OllamaClient(self._config()).complete(None, "usr", tag="t")
        sent = ctor.return_value.invoke.call_args.args[0]
        assert len(sent) == 1 and isinstance(sent[0], HumanMessage)

    def test_missing_usage_metadata_is_zero(self) -> None:
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            completion = OllamaClient(self._config()).complete(None, "u", tag="t")
        assert completion.usage.input_tokens == 0
        assert completion.usage.output_tokens == 0

    def test_length_maps_to_max_tokens(self) -> None:
        message = AIMessage(content="x", response_metadata={"done_reason": "length"})
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = message
            completion = OllamaClient(self._config()).complete(None, "u", tag="t")
        assert completion.is_failure is True

    def test_optional_params_omitted(self) -> None:
        with patch("src.llm.client.ChatOllama") as ctor:
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            config = LLMConfig(provider="ollama", model="m")
            OllamaClient(config).complete(None, "u", tag="t")
        ctor.assert_called_once_with(model="m", temperature=0)

    def test_budget_abort(self) -> None:
        tracker = UsageTracker(budget_usd=0.0)
        with patch("src.llm.client.ChatOllama") as ctor, pytest.raises(BudgetExceeded):
            OllamaClient(self._config(), tracker).complete(None, "u", tag="t")
        ctor.return_value.invoke.assert_not_called()


class TestClientRegistry:
    def test_default_client_reproduces_current_behaviour(self) -> None:
        with (
            patch("src.llm.client.settings") as s,
            patch("src.llm.client.ChatOllama") as ctor,
        ):
            s.ollama_model = "qwen2.5:3b"
            s.ollama_base_url = "http://localhost:11434"
            s.ollama_num_ctx = None
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            client = _build_default_client()
            client.complete("sys", "usr", tag="t")
        ctor.assert_called_once_with(
            model="qwen2.5:3b", temperature=0, base_url="http://localhost:11434"
        )

    def test_default_client_passes_num_ctx_when_set(self) -> None:
        with (
            patch("src.llm.client.settings") as s,
            patch("src.llm.client.ChatOllama") as ctor,
        ):
            s.ollama_model = "m"
            s.ollama_base_url = "http://h"
            s.ollama_num_ctx = 4096
            ctor.return_value.invoke.return_value = AIMessage(content="x")
            _build_default_client().complete(None, "usr", tag="t")
        assert ctor.call_args.kwargs["num_ctx"] == 4096

    def test_set_and_reset_client(self) -> None:
        class Fake:
            def complete(
                self, system: str | None, user: str, *, tag: str, prefix: str | None = None
            ) -> Completion:
                raise NotImplementedError

        fake = Fake()
        token = set_client(fake)
        try:
            assert get_client() is fake
        finally:
            reset_client(token)
        assert get_client() is not fake
