"""Cost and latency per question of a committed run, with bootstrap CIs.

For every system in ``predictions.jsonl``: mean cost per question and mean latency with
95% percentile bootstrap CIs (questions resampled with replacement, B = 10000, seed 0, as
in ``stats.py``), plus median and p95 latency. Latency is wall time per question as the
runner measured it, so it includes the concurrency of the run (``workers`` in its summary).
Writes ``cost_latency.json`` into the report directory and prints a table.

    uv run python -m evals.text2sql.cost_latency 2026-10-19-haiku-test-350
"""

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.text2sql.stats import (
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    _percentile_interval,
    bootstrap_meta,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, float] | None:
    """Mean of ``values`` and its percentile bootstrap CI; None when there are none."""
    n = len(values)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    low, high = _percentile_interval(means, confidence)
    return {"mean": sum(values) / n, "ci_low": low, "ci_high": high}


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of ``values`` (non-empty)."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(predictions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per system: n, cost and latency means with CIs, median and p95 latency (seconds)."""
    systems: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        systems.setdefault(p["system"], []).append(p)
    out: dict[str, dict[str, Any]] = {}
    for system, rows in systems.items():
        latency_s = [r["latency_ms"] / 1000 for r in rows]
        out[system] = {
            "n": len(rows),
            "cost_usd": mean_ci([r["cost_usd"] for r in rows]),
            "latency_s": mean_ci(latency_s),
            "latency_s_median": quantile(latency_s, 0.5),
            "latency_s_p95": quantile(latency_s, 0.95),
            "llm_calls": mean_ci([float(r["n_llm_calls"]) for r in rows]),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="report directory name under evals/reports/")
    args = parser.parse_args(argv)
    run_dir = REPORTS / args.report
    path = run_dir / "predictions.jsonl"
    with open(path, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    result = {
        "report": args.report,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "workers": summary.get("workers"),
        "bootstrap": bootstrap_meta(DEFAULT_RESAMPLES, DEFAULT_SEED),
        "systems": summarize(predictions),
    }
    (run_dir / "cost_latency.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.report} (workers {result['workers']})")
    for system, s in result["systems"].items():
        c, lat = s["cost_usd"], s["latency_s"]
        print(
            f"  {system:<18} n={s['n']:<4} cost ${c['mean']:.5f} [{c['ci_low']:.5f}, "
            f"{c['ci_high']:.5f}]  latency {lat['mean']:.2f} s [{lat['ci_low']:.2f}, "
            f"{lat['ci_high']:.2f}]  median {s['latency_s_median']:.2f} s  "
            f"p95 {s['latency_s_p95']:.2f} s"
        )


if __name__ == "__main__":
    main()
