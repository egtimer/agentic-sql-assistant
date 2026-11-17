"""Compare two answer-step prompts on the same SQL: the judge's verdicts of final-answer passes
(``final_answer.py``) with ``--respond v1`` against passes with ``--respond v2``.

Each pass rebuilds agent v2's answer for the same committed run, so the SQL, EX and the gold
results are identical; only the answer prompt (and the run-to-run noise of the respond and
judge calls) differs. Questions without a reference are left out. For each subset the script
reports the share of answers the judge accepts per prompt, as a mean over passes with a
two-stage bootstrap CI (``evals/noise/stats.py``: questions resampled, then passes), and the
paired difference v2 - v1.

Subsets: all questions; EX correct; EX wrong; the questions whose final result has more rows
than the answer step is shown (``ANSWER_PREVIEW_ROWS``); and the failures the 2026-11-17
analysis put in ``answer_on_large_result`` and ``answer_misread``.

    uv run python -m evals.failures.compare_answers \\
        --v1 evals/results/failures/final-answer-haiku-r01 \\
             evals/results/failures/final-answer-haiku-r01-v1-rep2 \\
        --v2 evals/results/failures/final-answer-haiku-r01-v2-rep1 \\
             evals/results/failures/final-answer-haiku-r01-v2-rep2 \\
        --report 2026-11-16-noise-bird-haiku-test-350/run-r01 \\
        --classification evals/reports/2026-11-17-failure-analysis/classification.jsonl \\
        --out evals/reports/2026-11-17-fixes/answer-step.json
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.failures import taxonomy
from evals.failures.final_answer import NO_REFERENCE, REPORTS, load_predictions
from evals.memory.common import read_jsonl
from evals.noise import stats


def verdicts(run_dir: Path) -> dict[int, str]:
    return {int(r["question_id"]): r["judge_label"] for r in read_jsonl(run_dir / "answers.jsonl")}


def large_result_ids(predictions: Sequence[dict[str, Any]]) -> set[int]:
    """Questions whose final result has more rows than the answer step is shown."""
    out = set()
    for pred in predictions:
        final = next(
            (a for a in reversed(pred.get("attempts") or []) if a["sql"] == pred["pred_sql"]), None
        )
        if final is None:
            continue
        text = str(final.get("row_count"))
        rows = int(text[1:]) + 1 if text.startswith(">") else int(text) if text.isdigit() else 0
        if rows > taxonomy.ANSWER_PREVIEW_ROWS:
            out.add(int(pred["question_id"]))
    return out


def matrix(runs: Sequence[dict[int, str]], ids: Sequence[int]) -> stats.Matrix:
    return stats.as_matrix([[run[i] == "correct" for run in runs] for i in ids])


def compare(
    v1: Sequence[dict[int, str]],
    v2: Sequence[dict[int, str]],
    subsets: dict[str, set[int]],
) -> dict[str, Any]:
    """Accepted share per prompt and the paired difference, per subset."""
    if len(v1) != len(v2):
        raise ValueError("pass the same number of v1 and v2 passes (paired by index)")
    common = set.intersection(*(set(r) for r in [*v1, *v2]))
    scored = {i for i in common if all(r[i] != NO_REFERENCE for r in [*v1, *v2])}
    out: dict[str, Any] = {}
    for name, members in subsets.items():
        ids = sorted(members & scored)
        if not ids:
            continue
        a, b = matrix(v2, ids), matrix(v1, ids)
        out[name] = {
            "n_questions": len(ids),
            "v1": stats.two_stage_ci(b),
            "v2": stats.two_stage_ci(a),
            "v2_minus_v1": stats.paired_two_stage(a, b),
            "accepted_per_pass": {
                "v1": [int(b[:, j].sum()) for j in range(b.shape[1])],
                "v2": [int(a[:, j].sum()) for j in range(a.shape[1])],
            },
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--v1", type=Path, nargs="+", required=True)
    parser.add_argument("--v2", type=Path, nargs="+", required=True)
    parser.add_argument("--report", required=True, help="the BIRD run the passes rebuilt")
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    predictions = load_predictions(REPORTS / args.report)
    ex_ok = {int(p["question_id"]) for p in predictions if p["correct"]}
    all_ids = {int(p["question_id"]) for p in predictions}
    rows = [r for r in read_jsonl(args.classification) if r["source"] == "bird"]
    subsets = {
        "all": all_ids,
        "ex_correct": ex_ok,
        "ex_wrong": all_ids - ex_ok,
        "large_result": large_result_ids(predictions),
        "failures_answer_on_large_result": {
            int(r["item"]) for r in rows if r["primary"] == taxonomy.ANSWER_ON_LARGE_RESULT
        },
        "failures_answer_misread": {
            int(r["item"]) for r in rows if r["primary"] == taxonomy.ANSWER_MISREAD
        },
    }
    result = {
        "report": args.report,
        "v1_passes": [p.name for p in args.v1],
        "v2_passes": [p.name for p in args.v2],
        "subsets": compare([verdicts(p) for p in args.v1], [verdicts(p) for p in args.v2], subsets),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, s in result["subsets"].items():
        d = s["v2_minus_v1"]
        print(
            f"{name:<34} n={s['n_questions']:<4} v1 {s['v1']['mean']:.3f}  v2 {s['v2']['mean']:.3f}"
            f"  diff {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
