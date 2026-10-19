"""Tests for the bootstrap confidence intervals."""

import pytest

from evals.text2sql.run import format_ci, format_table
from evals.text2sql.stats import bootstrap_ci, bootstrap_meta, paired_bootstrap


class TestBootstrapCI:
    def test_known_small_input(self) -> None:
        # two outcomes, one correct: each resample mean is 0, 0.5 or 1
        low, high = bootstrap_ci([1, 0], resamples=2000, seed=1)  # type: ignore[misc]
        assert (low, high) == (0.0, 1.0)

    def test_deterministic_and_seed_dependent(self) -> None:
        data = [True] * 12 + [False] * 18
        assert bootstrap_ci(data, resamples=500, seed=3) == bootstrap_ci(
            data, resamples=500, seed=3
        )
        assert bootstrap_ci(data, resamples=500, seed=3) != bootstrap_ci(
            data, resamples=500, seed=4
        )

    def test_contains_point_estimate_and_ordered(self) -> None:
        data = [True] * 45 + [False] * 105
        ci = bootstrap_ci(data, resamples=2000, seed=0)
        assert ci is not None
        low, high = ci
        assert 0 <= low <= 45 / 150 <= high <= 1
        assert high - low < 0.25  # roughly +-7.5 points for n=150, p=0.3

    def test_all_correct_is_degenerate(self) -> None:
        assert bootstrap_ci([True] * 20, resamples=300) == (1.0, 1.0)
        assert bootstrap_ci([False] * 20, resamples=300) == (0.0, 0.0)

    def test_empty(self) -> None:
        assert bootstrap_ci([]) is None


class TestPaired:
    def test_known_input(self) -> None:
        a = {1: True, 2: True, 3: True, 4: False}
        b = {1: False, 2: False, 3: False, 4: False}
        r = paired_bootstrap(a, b, resamples=2000, seed=0)
        assert r is not None
        assert r["n"] == 4
        assert r["diff"] == pytest.approx(0.75)
        assert r["ci_low"] <= 0.75 <= r["ci_high"]
        assert r["frac_resamples_le_zero"] == pytest.approx(0.25**4, abs=0.01)  # all draw id 4

    def test_only_shared_ids_and_deterministic(self) -> None:
        a = {1: True, 2: False, 3: True}
        b = {2: True, 3: True, 9: False}
        r1 = paired_bootstrap(a, b, resamples=200, seed=5)
        assert r1 is not None and r1["n"] == 2
        assert r1 == paired_bootstrap(a, b, resamples=200, seed=5)

    def test_identical_systems(self) -> None:
        a = {i: i % 2 == 0 for i in range(10)}
        r = paired_bootstrap(a, dict(a), resamples=100)
        assert r is not None
        assert (r["diff"], r["ci_low"], r["ci_high"]) == (0.0, 0.0, 0.0)
        assert r["frac_resamples_le_zero"] == 1.0

    def test_no_overlap(self) -> None:
        assert paired_bootstrap({1: True}, {2: True}) is None


def test_meta_and_formatting() -> None:
    meta = bootstrap_meta(10000, 0)
    assert meta["B"] == 10000 and meta["seed"] == 0 and "bootstrap" in meta["method"]
    assert format_ci({"ex": 0.3, "ex_ci95": [0.12, 0.48]}) == "30.0% [12.0, 48.0]"
    assert format_ci({"ex": None}) == "-"
    assert format_ci({"ex": 0.5}) == "50.0%"


def test_table_shows_cis_and_comparison() -> None:
    block = {"n": 10, "correct": 3, "ex": 0.3, "ex_ci95": [0.1, 0.5]}
    system = {
        **block,
        "aborted": None,
        "failures": {},
        "by_difficulty": {d: block for d in ("simple", "moderate", "challenging")},
        "tokens": {"input_mean_per_question": 1, "output_mean_per_question": 1},
        "cost_usd": {"total": 0.0},
        "latency_ms": {"mean": 1.0},
    }
    summary = {
        "systems": {"baseline": system, "agent": system},
        "comparison": {
            "agent_minus_baseline": {
                "n": 10, "diff": 0.1, "ci_low": -0.1, "ci_high": 0.3, "frac_resamples_le_zero": 0.2
            }
        },
    }  # fmt: skip
    table = format_table(summary)
    assert "30.0% [10.0, 50.0]" in table
    assert "+10.0 pts [-10.0, +30.0]" in table
