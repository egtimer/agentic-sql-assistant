"""Run-to-run noise of repeated eval runs: dispersion, flips, intervals, paired
differences and power. No LLM call.

    uv run python -m evals.noise.analyze --eval tools --metric answer_correct \\
        --runs evals/reports/<dir>/run-r* --compare agent_v1:no_tools \\
        --reference evals/reports/2026-10-30-v1-tools-haiku-test --out <dir>/analysis.json

    uv run python -m evals.noise.analyze --eval text2sql --metric correct \\
        --runs evals/reports/<dir>/run-r* --compare agent_v2:baseline \\
        --reference evals/reports/2026-10-19-haiku-test-350 --out <dir>/analysis.json

Only items scored in every run enter a system's matrix (the tools set has queries
without a scored answer); the number left out is reported. ``--reference`` is a
published single run of the same configuration, shown next to the spread of the
repetitions; it is not part of it.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.noise import stats
from evals.noise.repeat import PROVIDER_ERROR_PATTERN

ID_KEY = {"tools": "id", "text2sql": "question_id"}
METRICS = {"tools": ("answer_correct", "exact"), "text2sql": ("correct",)}


def load_predictions(run_dir: Path, *, rescored: bool = False) -> list[dict[str, Any]]:
    name = "predictions_rescored.jsonl" if rescored else "predictions.jsonl"
    path = run_dir / name
    if rescored and not path.exists():
        path = run_dir / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_provider_failure(row: dict[str, Any]) -> bool:
    """An API failure recorded as a prediction (runs made before the runners raised
    ProviderError scored these as wrong answers)."""
    return bool(PROVIDER_ERROR_PATTERN.search(str(row.get("error") or "")))


def verdicts(rows: list[dict[str, Any]], kind: str, system: str, metric: str) -> dict[str, bool]:
    """item id -> verdict for one system. Items with no verdict (None) and API failures
    are left out: an answer that never came back is missing, not wrong."""
    key = ID_KEY[kind]
    return {
        str(r[key]): bool(r[metric])
        for r in rows
        if r["system"] == system and r.get(metric) is not None and not is_provider_failure(r)
    }


def matrix(
    runs: list[dict[str, bool]],
) -> tuple[list[str], stats.Matrix, int]:
    """Items present in every run, the items x runs matrix, and how many were dropped."""
    common = set(runs[0])
    every = set().union(*runs)
    for v in runs[1:]:
        common &= set(v)
    ids = sorted(common, key=lambda s: (len(s), s))
    m = stats.as_matrix([[v[i] for v in runs] for i in ids])
    return ids, m, len(every) - len(common)


def system_report(
    runs: list[dict[str, bool]], resamples: int, seed: int
) -> tuple[dict[str, Any], list[str], stats.Matrix]:
    ids, m, dropped = matrix(runs)
    report: dict[str, Any] = {
        "dropped_items": dropped,
        "dispersion": stats.dispersion(m),
        "flips": stats.flips(m, ids),
        "interval": stats.two_stage_ci(m, resamples=resamples, seed=seed),
    }
    if m.shape[1] > 1:
        report["variance_components"] = stats.variance_components(m)
    return report, ids, m


def reference_report(
    ref_dir: Path,
    kind: str,
    system: str,
    metric: str,
    ids: list[str],
    spread: dict[str, Any],
) -> dict[str, Any] | None:
    """The published run's accuracy on the same items as the repetitions (tools runs
    are read as rescored, the scorer the repetitions use)."""
    rows = load_predictions(ref_dir, rescored=kind == "tools")
    v = verdicts(rows, kind, system, metric)
    shared = [i for i in ids if i in v]
    if not shared:
        return None
    acc = sum(v[i] for i in shared) / len(shared)
    return {
        "run": ref_dir.name,
        "n_items": len(shared),
        "accuracy": acc,
        "within_repetition_range": spread["min"] <= acc <= spread["max"],
    }


def analyze(
    kind: str,
    run_dirs: list[Path],
    metric: str,
    comparisons: list[tuple[str, str]],
    *,
    reference: Path | None = None,
    delta: float = 0.03,
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
) -> dict[str, Any]:
    if metric not in METRICS[kind]:
        raise ValueError(f"metric {metric!r} is not one of {METRICS[kind]} for {kind}")
    per_run = [load_predictions(d) for d in run_dirs]
    systems = sorted({r["system"] for rows in per_run for r in rows})
    out: dict[str, Any] = {
        "eval": kind,
        "metric": metric,
        "runs": [str(d) for d in run_dirs],
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "confidence": stats.CONFIDENCE,
            "method": stats.TWO_STAGE_METHOD,
        },
        "systems": {},
        "comparisons": {},
    }
    matrices: dict[str, tuple[list[str], stats.Matrix]] = {}
    for system in systems:
        runs = [verdicts(rows, kind, system, metric) for rows in per_run]
        if not all(runs):
            continue
        report, ids, m = system_report(runs, resamples, seed)
        report["provider_failures_left_out"] = [
            sum(1 for r in rows if r["system"] == system and is_provider_failure(r))
            for rows in per_run
        ]
        if reference is not None:
            report["reference"] = reference_report(
                reference, kind, system, metric, ids, report["dispersion"]
            )
        out["systems"][system] = report
        matrices[system] = (ids, m)
    for a, b in comparisons:
        if a not in matrices or b not in matrices:
            raise ValueError(f"comparison {a}:{b} needs both systems in every run")
        ids_a, ma = matrices[a]
        ids_b, mb = matrices[b]
        shared = sorted(set(ids_a) & set(ids_b), key=lambda s: (len(s), s))
        ia = [ids_a.index(i) for i in shared]
        ib = [ids_b.index(i) for i in shared]
        pa, pb = ma[ia], mb[ib]
        r = pa.shape[1]
        out["comparisons"][f"{a}-{b}"] = {
            "paired": stats.paired_two_stage(pa, pb, resamples=resamples, seed=seed),
            "power": stats.power_analysis(
                pa, pb, delta=delta, runs_options=tuple(sorted({1, r})), seed=seed
            ),
        }
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{report['eval']} / {report['metric']}, {len(report['runs'])} runs"]
    for system, s in report["systems"].items():
        d, f, ci = s["dispersion"], s["flips"], s["interval"]
        sd = "n/a" if d["sd"] is None else f"{100 * d['sd']:.1f}"
        lines.append(
            f"{system:<12} n={d['n_items']:<4} mean {_pct(d['mean'])} sd {sd} "
            f"range [{_pct(d['min'])}, {_pct(d['max'])}]  CI with run noise "
            f"[{_pct(ci['ci_low'])}, {_pct(ci['ci_high'])}]  one run "
            f"[{_pct(ci['single_run']['ci_low'])}, {_pct(ci['single_run']['ci_high'])}]  "
            f"items changing {f['n_changing']}/{f['n_items']}"
        )
        ref = s.get("reference")
        if ref:
            inside = "inside" if ref["within_repetition_range"] else "OUTSIDE"
            lines.append(f"{'':<12} published run {_pct(ref['accuracy'])} ({inside} the range)")
    for name, c in report["comparisons"].items():
        p = c["paired"]
        lines.append(
            f"{name:<24} diff {_pct(p['diff'])} pts [{_pct(p['ci_low'])}, {_pct(p['ci_high'])}]"
            f"  share <= 0: {p['frac_resamples_le_zero']:.3f}"
        )
        for runs, pw in c["power"]["by_runs"].items():
            lines.append(
                f"{'':<24} to detect {_pct(c['power']['delta'])} pts with {runs} run(s)/item: "
                f"{pw['items_needed']} items (simulated power {pw['simulated_power_at_n']:.2f})"
            )
    return "\n".join(lines)


def _pair(value: str) -> tuple[str, str]:
    a, sep, b = value.partition(":")
    if not sep or not a or not b:
        raise argparse.ArgumentTypeError(f"expected SYSTEM_A:SYSTEM_B, got {value!r}")
    return a, b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval", choices=sorted(ID_KEY), required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--compare", type=_pair, action="append", default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(
            args.eval,
            sorted(args.runs),
            args.metric,
            args.compare,
            reference=args.reference,
            delta=args.delta,
            resamples=args.resamples,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
