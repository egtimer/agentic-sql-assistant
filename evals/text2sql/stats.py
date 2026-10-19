"""Bootstrap confidence intervals for execution accuracy (pure Python, no numpy)."""

import random
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
CONFIDENCE = 0.95
METHOD = "percentile bootstrap, questions resampled with replacement"
PAIRED_METHOD = "paired percentile bootstrap, question ids resampled with replacement"


def _percentile_interval(sorted_stats: list[float], confidence: float) -> tuple[float, float]:
    """Percentile interval of an already sorted list of resampled statistics."""
    b = len(sorted_stats)
    tail = (1 - confidence) / 2
    low = sorted_stats[int(b * tail)]
    high = sorted_stats[max(int(b * (1 - tail)) - 1, 0)]
    return low, high


def bootstrap_ci(
    outcomes: Sequence[bool | int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean of 0/1 outcomes; None when there are none."""
    n = len(outcomes)
    if n == 0:
        return None
    values = [int(o) for o in outcomes]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return _percentile_interval(means, confidence)


def paired_bootstrap(
    a: Mapping[int, bool],
    b: Mapping[int, bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, Any] | None:
    """Paired bootstrap of ``mean(a) - mean(b)`` over the question ids present in both.

    Returns the point estimate, its CI and the fraction of resamples with a difference
    <= 0; None when no question id is shared.
    """
    ids = sorted(a.keys() & b.keys())
    n = len(ids)
    if n == 0:
        return None
    diffs = [int(a[i]) - int(b[i]) for i in ids]
    rng = random.Random(seed)
    stats = [sum(rng.choices(diffs, k=n)) / n for _ in range(resamples)]
    low, high = _percentile_interval(sorted(stats), confidence)
    return {
        "n": n,
        "diff": sum(diffs) / n,
        "ci_low": low,
        "ci_high": high,
        "frac_resamples_le_zero": sum(1 for s in stats if s <= 0) / resamples,
    }


def bootstrap_meta(resamples: int, seed: int, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """What summary.json records about the bootstrap."""
    return {
        "B": resamples,
        "seed": seed,
        "confidence": confidence,
        "method": METHOD,
        "paired_method": PAIRED_METHOD,
    }
