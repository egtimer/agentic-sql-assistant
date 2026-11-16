"""Statistics for repeated eval runs: items x runs matrices of 0/1 verdicts.

A matrix ``m`` has one row per item (question) and one column per repetition; ``m[i, r]``
is 1 when the system got item ``i`` right in run ``r``. Everything here treats the item
as the cluster: intervals resample items, and within each resampled item its runs, so
they carry both the item sampling noise and the run-to-run noise (a two-stage, or
cluster, bootstrap). Paired comparisons use the same items and the same run indices for
both systems.
"""

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
CONFIDENCE = 0.95
TWO_STAGE_METHOD = (
    "two-stage percentile bootstrap: items resampled with replacement, then each "
    "resampled item's runs resampled with replacement"
)

Matrix = npt.NDArray[np.float64]
Index = npt.NDArray[np.int64]


def as_matrix(rows: list[list[bool]]) -> Matrix:
    m = np.asarray(rows, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        raise ValueError("need a non-empty items x runs matrix")
    return m


def _interval(stats: Matrix, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(stats, [tail, 1 - tail])
    return float(low), float(high)


def dispersion(m: Matrix) -> dict[str, Any]:
    """Accuracy of each run, and their mean, sample SD, min and max."""
    per_run = m.mean(axis=0)
    return {
        "n_items": int(m.shape[0]),
        "n_runs": int(m.shape[1]),
        "per_run": [float(x) for x in per_run],
        "mean": float(per_run.mean()),
        "sd": float(per_run.std(ddof=1)) if m.shape[1] > 1 else None,
        "min": float(per_run.min()),
        "max": float(per_run.max()),
    }


def flips(m: Matrix, ids: list[str]) -> dict[str, Any]:
    """Items whose verdict is not the same in every run."""
    item_mean = m.mean(axis=1)
    changing = [ids[i] for i in range(len(ids)) if 0 < item_mean[i] < 1]
    return {
        "n_items": len(ids),
        "n_changing": len(changing),
        "share_changing": len(changing) / len(ids),
        "n_always_right": int((item_mean == 1).sum()),
        "n_always_wrong": int((item_mean == 0).sum()),
        "changing_ids": changing,
    }


def variance_components(m: Matrix) -> dict[str, Any]:
    """Split the variance of a single 0/1 verdict into a between-item part (some items
    are harder than others) and a within-item part (the same item, run again).

    ``within`` is the mean over items of the sample variance of its runs; ``between`` is
    the variance of the item means minus the share of it the runs explain, floored at 0.
    ``icc`` is between / (between + within). The variance of the overall mean over n
    items and R runs is then about between / n + within / (n R).
    """
    n, r = m.shape
    if r < 2:
        raise ValueError("need at least two runs per item")
    within = float(m.var(axis=1, ddof=1).mean())
    between = max(0.0, float(m.mean(axis=1).var(ddof=1)) - within / r)
    total = between + within
    return {
        "between_item": between,
        "within_item": within,
        "icc": between / total if total > 0 else None,
        "se_of_mean_one_run": math.sqrt((between + within) / n),
        "se_of_mean_all_runs": math.sqrt(between / n + within / (n * r)),
    }


def _two_stage_means(m: Matrix, resamples: int, rng: np.random.Generator) -> Matrix:
    n, r = m.shape
    out = np.empty(resamples)
    for b in range(resamples):
        items: Index = np.asarray(rng.integers(0, n, size=n), dtype=np.int64)
        runs: Index = np.asarray(rng.integers(0, r, size=(n, r)), dtype=np.int64)
        out[b] = m[items[:, None], runs].mean()
    return out


def two_stage_ci(
    m: Matrix,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """Mean over items and runs with a CI that includes the run-to-run noise, next to the
    item-only CI of a single run (the first) for contrast."""
    rng = np.random.default_rng(seed)
    stats = _two_stage_means(m, resamples, rng)
    low, high = _interval(stats, confidence)
    first = m[:, 0]
    rng1 = np.random.default_rng(seed)
    single = np.array(
        [first[rng1.integers(0, len(first), size=len(first))].mean() for _ in range(resamples)]
    )
    s_low, s_high = _interval(single, confidence)
    return {
        "mean": float(m.mean()),
        "ci_low": low,
        "ci_high": high,
        "single_run": {"mean": float(first.mean()), "ci_low": s_low, "ci_high": s_high},
    }


def paired_two_stage(
    a: Matrix,
    b: Matrix,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """mean(a) - mean(b) on the same items and run indices, with a two-stage bootstrap
    that resamples items and, within each item, the same run indices for both."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    stats = _two_stage_means(d, resamples, rng)
    low, high = _interval(stats, confidence)
    per_run = d.mean(axis=0)
    return {
        "n_items": int(d.shape[0]),
        "n_runs": int(d.shape[1]),
        "diff": float(d.mean()),
        "ci_low": low,
        "ci_high": high,
        "frac_resamples_le_zero": float((stats <= 0).mean()),
        "per_run_diff": [float(x) for x in per_run],
        "per_run_sd": float(per_run.std(ddof=1)) if d.shape[1] > 1 else None,
    }


def _difference_components(d: Matrix) -> tuple[float, float]:
    """(between-item variance of the true per-item difference, within-item variance of a
    single run's difference)."""
    r = d.shape[1]
    within = float(d.var(axis=1, ddof=1).mean()) if r > 1 else 0.0
    between = max(0.0, float(d.mean(axis=1).var(ddof=1)) - (within / r if r > 1 else 0.0))
    return between, within


def items_needed(sd: float, delta: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired-difference sample size: n = ((z_{1-alpha/2} + z_power) sd / delta)^2."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return math.ceil((z * sd / delta) ** 2)


def simulated_power(
    d: Matrix,
    n: int,
    runs: int,
    delta: float,
    *,
    alpha: float = 0.05,
    simulations: int = 2000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Share of simulated studies of ``n`` items x ``runs`` runs that reject "no
    difference" (two-sided z test on the paired mean) when the true difference is
    ``delta``.

    Each simulated item is an observed item drawn with replacement: its effect is the
    observed per-item mean difference shrunk towards the grand mean so that the effects
    have the between-item variance (an observed item mean also carries run noise), moved
    so that their mean is ``delta``; each of its ``runs`` runs adds one of that item's
    own residuals (difference minus the item mean), drawn with replacement and scaled by
    sqrt(R / (R - 1)) so their variance is the within-item variance. With one observed
    run per item there is no residual and the observed differences are used as they are.
    """
    r_obs = d.shape[1]
    item_mean = d.mean(axis=1)
    grand = float(item_mean.mean())
    z = NormalDist().inv_cdf(1 - alpha / 2)
    rng = np.random.default_rng(seed)
    if r_obs > 1:
        between, _ = _difference_components(d)
        var_means = float(item_mean.var(ddof=1))
        shrink = math.sqrt(between / var_means) if var_means > 0 else 0.0
        effect = delta + shrink * (item_mean - grand)
        resid = (d - item_mean[:, None]) * math.sqrt(r_obs / (r_obs - 1))
    else:
        effect = d[:, 0] - grand + delta
        resid = np.zeros_like(d)
    rejected = 0
    for _ in range(simulations):
        items: Index = rng.integers(0, d.shape[0], size=n)
        cols: Index = np.asarray(rng.integers(0, r_obs, size=(n, runs)), dtype=np.int64)
        per_item = effect[items] + resid[items[:, None], cols].mean(axis=1)
        se = per_item.std(ddof=1) / math.sqrt(n)
        if se > 0 and abs(per_item.mean()) / se > z:
            rejected += 1
    return rejected / simulations


def power_analysis(
    a: Matrix,
    b: Matrix,
    *,
    delta: float = 0.03,
    alpha: float = 0.05,
    power: float = 0.8,
    runs_options: tuple[int, ...] = (1,),
    simulations: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Items needed to detect a paired difference of ``delta`` between two systems, with
    the variance components observed here, for each number of runs per item; each
    analytic n is checked by simulation."""
    d = a - b
    between, within = _difference_components(d)
    out: dict[str, Any] = {
        "delta": delta,
        "alpha": alpha,
        "power": power,
        "between_item_var": between,
        "within_item_var": within,
        "observed_diff": float(d.mean()),
        "by_runs": {},
    }
    for runs in runs_options:
        sd = math.sqrt(between + within / runs)
        n = items_needed(sd, delta, alpha=alpha, power=power)
        out["by_runs"][str(runs)] = {
            "sd_of_item_difference": sd,
            "items_needed": n,
            "simulated_power_at_n": simulated_power(
                d, n, runs, delta, alpha=alpha, simulations=simulations, seed=seed
            ),
            "total_samples": n * runs,
        }
    return out
