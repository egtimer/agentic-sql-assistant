"""Paired comparison of two systems taken from committed run reports.

Each side is ``<report dir>:<system>``; the comparison uses the questions both sides answered
and prints EX of each side and the paired bootstrap of EX(B) - EX(A) (same method as the
runner's ``comparison`` block).

    uv run python -m evals.text2sql.compare_runs \
        2026-10-19-haiku-test-350:agent_v2 2026-10-19-sonnet-test-350:baseline
"""

import argparse
import json
from pathlib import Path

from evals.text2sql.stats import paired_bootstrap

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def load_outcomes(spec: str) -> dict[int, bool]:
    """``report:system`` -> {question_id: correct}."""
    report, system = spec.rsplit(":", 1)
    path = REPORTS / report / "predictions.jsonl"
    outcomes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["system"] == system:
                outcomes[int(record["question_id"])] = bool(record["correct"])
    if not outcomes:
        raise SystemExit(f"no predictions for system {system!r} in {path}")
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", help="reference side, report_dir:system")
    parser.add_argument("b", help="compared side, report_dir:system")
    args = parser.parse_args()
    a, b = load_outcomes(args.a), load_outcomes(args.b)
    common = sorted(set(a) & set(b))
    a = {q: a[q] for q in common}
    b = {q: b[q] for q in common}
    # paired_bootstrap(x, y) reports EX(x) - EX(y); B - A is wanted here.
    result = paired_bootstrap(b, a)
    print(f"n = {len(common)} common questions")
    print(f"A {args.a}: EX {sum(a.values()) / len(a):.1%}")
    print(f"B {args.b}: EX {sum(b.values()) / len(b):.1%}")
    print(
        f"B - A: {result['diff'] * 100:+.1f} pts "
        f"[{result['ci_low'] * 100:+.1f}, {result['ci_high'] * 100:+.1f}] "
        "(95% paired bootstrap)"
    )


if __name__ == "__main__":
    main()
