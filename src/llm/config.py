"""Configuration models for LLM providers (loaded from YAML in ``configs/llm/``)."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class Pricing(BaseModel):
    """Token prices in USD per million tokens.

    When ``long_context_threshold`` is set and a call's prompt (input plus cache tokens)
    exceeds it, ``long_input_per_mtok`` and ``long_output_per_mtok`` replace the standard
    input and output rates for that call. Cache rates are the same in both tiers.
    """

    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    long_context_threshold: int | None = None
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None


class LLMConfig(BaseModel):
    """Provider, model and generation parameters for one LLM backend."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "anthropic"]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    effort: str | None = None
    num_ctx: int | None = None
    base_url: str | None = None
    prompt_caching: bool = False
    pricing: Pricing = Pricing()

    @model_validator(mode="after")
    def _anthropic_needs_max_tokens(self) -> Self:
        if self.provider == "anthropic" and self.max_tokens is None:
            raise ValueError("max_tokens is required for the anthropic provider")
        return self
