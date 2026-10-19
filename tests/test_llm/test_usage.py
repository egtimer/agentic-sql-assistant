"""Tests for cost math, the budget cap and the usage tracker."""

import json
import threading
from pathlib import Path

import pytest

from src.llm.config import LLMConfig, Pricing
from src.llm.usage import BudgetExceeded, CallUsage, UsageTracker, call_context, cost_of

HAIKU = Pricing(
    input_per_mtok=0.10,
    output_per_mtok=0.50,
    cache_read_per_mtok=0.01,
    cache_write_per_mtok=0.125,
    long_context_threshold=100_000,
    long_input_per_mtok=0.50,
    long_output_per_mtok=2.50,
    long_cache_read_per_mtok=0.05,
    long_cache_write_per_mtok=0.625,
)


class TestCostOf:
    def test_standard_rates(self) -> None:
        usage = CallUsage(tag="t", input_tokens=50_000, output_tokens=10_000)
        assert cost_of(usage, HAIKU) == pytest.approx((50_000 * 0.10 + 10_000 * 0.50) / 1e6)

    def test_cache_rates(self) -> None:
        usage = CallUsage(tag="t", cache_read_tokens=50_000, cache_write_tokens=10_000)
        assert cost_of(usage, HAIKU) == pytest.approx((50_000 * 0.01 + 10_000 * 0.125) / 1e6)

    def test_long_context_tier_above_threshold(self) -> None:
        usage = CallUsage(tag="t", input_tokens=100_001, output_tokens=1000)
        expected = (100_001 * 0.50 + 1000 * 2.50) / 1_000_000
        assert cost_of(usage, HAIKU) == pytest.approx(expected)

    def test_threshold_is_exclusive(self) -> None:
        usage = CallUsage(tag="t", input_tokens=100_000, output_tokens=1000)
        expected = (100_000 * 0.10 + 1000 * 0.50) / 1_000_000
        assert cost_of(usage, HAIKU) == pytest.approx(expected)

    def test_cache_tokens_count_towards_threshold(self) -> None:
        usage = CallUsage(tag="t", input_tokens=10, cache_read_tokens=100_000)
        expected = (10 * 0.50 + 100_000 * 0.05) / 1_000_000
        assert cost_of(usage, HAIKU) == pytest.approx(expected)

    def test_long_context_tier_reprices_cache_tokens(self) -> None:
        usage = CallUsage(tag="t", cache_read_tokens=90_000, cache_write_tokens=20_000)
        expected = (90_000 * 0.05 + 20_000 * 0.625) / 1_000_000
        assert cost_of(usage, HAIKU) == pytest.approx(expected)

    def test_zero_pricing(self) -> None:
        usage = CallUsage(tag="t", input_tokens=5000, output_tokens=5000)
        assert cost_of(usage, Pricing()) == 0.0


class TestConfig:
    def test_anthropic_requires_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(provider="anthropic", model="m")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig.model_validate({"provider": "ollama", "model": "m", "bogus": 1})


class TestUsageTracker:
    def test_record_accumulates(self) -> None:
        tracker = UsageTracker()
        tracker.record(CallUsage(tag="a", cost_usd=0.25))
        tracker.record(CallUsage(tag="b", cost_usd=0.5))
        assert tracker.total_cost == pytest.approx(0.75)
        assert [c.tag for c in tracker.calls] == ["a", "b"]

    def test_budget_abort(self) -> None:
        tracker = UsageTracker(budget_usd=1.0)
        tracker.check()
        tracker.record(CallUsage(tag="a", cost_usd=0.6))
        tracker.check()
        tracker.record(CallUsage(tag="b", cost_usd=0.6))
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check()
        assert exc.value.budget == 1.0
        assert exc.value.spent == pytest.approx(1.2)

    def test_budget_reached_exactly_aborts(self) -> None:
        tracker = UsageTracker(budget_usd=0.5)
        tracker.record(CallUsage(tag="a", cost_usd=0.5))
        with pytest.raises(BudgetExceeded):
            tracker.check()

    def test_no_budget_never_aborts(self) -> None:
        tracker = UsageTracker()
        tracker.record(CallUsage(tag="a", cost_usd=1e6))
        tracker.check()

    def test_context_is_attached(self) -> None:
        tracker = UsageTracker()
        with call_context(run_id="r1", system="baseline"):
            with call_context(question_id=7):
                tracker.record(CallUsage(tag="a"))
            tracker.record(CallUsage(tag="b"))
        tracker.record(CallUsage(tag="c"))
        assert tracker.calls[0].context == {"run_id": "r1", "system": "baseline", "question_id": 7}
        assert tracker.calls[1].context == {"run_id": "r1", "system": "baseline"}
        assert tracker.calls[2].context == {}

    def test_jsonl_sink(self, tmp_path: Path) -> None:
        sink = tmp_path / "sub" / "llm_calls.jsonl"
        tracker = UsageTracker(sink_path=sink)
        with call_context(run_id="r1"):
            tracker.record(CallUsage(tag="a", input_tokens=3, cost_usd=0.1, stop_reason="end_turn"))
        tracker.record(CallUsage(tag="b"))
        rows = [json.loads(line) for line in sink.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["run_id"] == "r1"
        assert rows[0]["tag"] == "a"
        assert rows[0]["input_tokens"] == 3
        assert rows[0]["stop_reason"] == "end_turn"
        assert "context" not in rows[0]

    def test_keep_calls_false(self) -> None:
        tracker = UsageTracker(keep_calls=False)
        tracker.record(CallUsage(tag="a", cost_usd=0.1))
        assert tracker.calls == []
        assert tracker.total_cost == pytest.approx(0.1)

    def test_thread_safety(self, tmp_path: Path) -> None:
        sink = tmp_path / "calls.jsonl"
        tracker = UsageTracker(sink_path=sink)

        def work() -> None:
            for _ in range(200):
                tracker.record(CallUsage(tag="x", cost_usd=0.001))

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(tracker.calls) == 1600
        assert tracker.total_cost == pytest.approx(1.6)
        assert len(sink.read_text().splitlines()) == 1600
