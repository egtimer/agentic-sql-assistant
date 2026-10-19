"""Regression gate for the weekly API eval: compare a run with a committed reference report.

Both sides are directories with a ``predictions.jsonl`` (a report under ``evals/reports/`` and
a run under ``evals/results/text2sql/``). The comparison covers the question ids both sides
answered for ``--system``:

- EX: paired bootstrap of ``EX(current) - EX(reference)`` (95% CI, the runner's method)
- cost: mean cost per question of each side

Exit code 1 (regression) when the EX drop is larger than ``--max-drop-pts`` AND the CI upper
bound is below 0 (a drop of that size that noise could not explain), or when the cost per
question is above ``--max-cost-ratio`` times the reference. Exit code 2 for unusable input.
A markdown table goes to stdout and, when ``GITHUB_STEP_SUMMARY`` is set, is appended to it.

    uv run python -m evals.text2sql.check_regression \\
        --reference evals/reports/<report> --current evals/results/text2sql/<run id>
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evals.text2sql.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, paired_bootstrap

DEFAULT_SYSTEM = "agent_v2"
DEFAULT_MAX_DROP_PTS = 10.0
DEFAULT_MAX_COST_RATIO = 2.0


def load_records(run_dir: Path, system: str) -> dict[int, dict[str, Any]]:
    """Latest record per question id of ``system`` in ``run_dir/predictions.jsonl``."""
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    records: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["system"] == system:
                    records[int(record["question_id"])] = record
    if not records:
        raise SystemExit(f"no predictions for system {system!r} in {path}")
    return records


def evaluate(
    reference: dict[int, dict[str, Any]],
    current: dict[int, dict[str, Any]],
    *,
    max_drop_pts: float = DEFAULT_MAX_DROP_PTS,
    max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Numbers and verdict of the comparison on the shared question ids."""
    ids = sorted(reference.keys() & current.keys())
    if not ids:
        raise SystemExit("reference and current share no question id")
    ref_ok = {i: bool(reference[i]["correct"]) for i in ids}
    cur_ok = {i: bool(current[i]["correct"]) for i in ids}
    paired = paired_bootstrap(cur_ok, ref_ok, resamples=resamples, seed=seed)
    assert paired is not None
    n = len(ids)
    cost_ref = sum(float(reference[i]["cost_usd"]) for i in ids) / n
    cost_cur = sum(float(current[i]["cost_usd"]) for i in ids) / n
    diff_pts = paired["diff"] * 100
    drop_pts = -diff_pts
    ex_fail = drop_pts > max_drop_pts and paired["ci_high"] < 0
    cost_ratio = cost_cur / cost_ref if cost_ref > 0 else None
    cost_fail = cost_ratio is not None and cost_ratio > max_cost_ratio
    return {
        "n": n,
        "n_reference": len(reference),
        "n_current": len(current),
        "ex_reference": sum(ref_ok.values()) / n,
        "ex_current": sum(cur_ok.values()) / n,
        "diff_pts": diff_pts,
        "ci_low_pts": paired["ci_low"] * 100,
        "ci_high_pts": paired["ci_high"] * 100,
        "cost_reference": cost_ref,
        "cost_current": cost_cur,
        "cost_ratio": cost_ratio,
        "max_drop_pts": max_drop_pts,
        "max_cost_ratio": max_cost_ratio,
        "ex_regression": ex_fail,
        "cost_regression": cost_fail,
        "passed": not (ex_fail or cost_fail),
    }


def render(result: dict[str, Any], *, system: str, reference: str, current: str) -> str:
    """Markdown report: one table and a verdict line."""
    r = result
    ratio = "n/a (reference cost is 0)" if r["cost_ratio"] is None else f"{r['cost_ratio']:.2f}x"
    ex_status = "FAIL" if r["ex_regression"] else "ok"
    cost_status = "FAIL" if r["cost_regression"] else "ok"
    lines = [
        f"### Regression check: `{system}`",
        "",
        f"Reference `{reference}`, current `{current}`; {r['n']} shared questions "
        f"(reference {r['n_reference']}, current {r['n_current']}).",
        "",
        "| metric | reference | current | change | limit | status |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| EX | {r['ex_reference']:.1%} | {r['ex_current']:.1%} | "
        f"{r['diff_pts']:+.1f} pts [{r['ci_low_pts']:+.1f}, {r['ci_high_pts']:+.1f}] | "
        f"drop <= {r['max_drop_pts']:g} pts or CI not below 0 | {ex_status} |",
        f"| cost per question | ${r['cost_reference']:.5f} | ${r['cost_current']:.5f} | "
        f"{ratio} | <= {r['max_cost_ratio']:g}x | {cost_status} |",
        "",
        "**Result: " + ("pass" if r["passed"] else "regression") + "** "
        "(95% paired bootstrap of EX(current) - EX(reference), 10000 resamples, seed 0).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, required=True, help="committed report dir")
    parser.add_argument("--current", type=Path, required=True, help="run dir to check")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-drop-pts", type=float, default=DEFAULT_MAX_DROP_PTS)
    parser.add_argument("--max-cost-ratio", type=float, default=DEFAULT_MAX_COST_RATIO)
    args = parser.parse_args(argv)
    result = evaluate(
        load_records(args.reference, args.system),
        load_records(args.current, args.system),
        max_drop_pts=args.max_drop_pts,
        max_cost_ratio=args.max_cost_ratio,
    )
    text = render(
        result, system=args.system, reference=args.reference.name, current=args.current.name
    )
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
