"""Import the hand review of the sampled failures and measure how well the rules agree with it.

The reviewer fills ``human_category`` in a copy of ``review/labels_template.csv`` (one of
``taxonomy.CATEGORIES``; ``not_a_failure`` and ``benchmark_issue`` only a reviewer can give)
and may add a ``note``. This script checks the file against the sample, then writes
``review/labels.csv`` (the validated labels, committed) and ``review/agreement.json``:

- rule vs human over the reviewed items: accuracy and Cohen's kappa over the categories, each
  with a 95% percentile bootstrap CI (items resampled, B = 10000, seed 0, as in
  ``evals/judge/agreement.py``), and the confusion matrix;
- how many items the reviewer did not count as agent failures (``not_a_failure``,
  ``benchmark_issue``), by rule category;
- per rule category: reviewed, agreed.

    uv run python -m evals.failures.review --report evals/reports/2026-11-17-failure-analysis \\
        --labels path/to/filled.csv --reviewer "the author"
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.failures import taxonomy
from evals.failures.classify import REVIEW_COLUMNS
from evals.judge.agreement import compare

REVIEWER_ONLY = (taxonomy.NOT_A_FAILURE, taxonomy.BENCHMARK_ISSUE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def validate(
    template: Sequence[Mapping[str, str]],
    filled: Sequence[Mapping[str, str]],
    *,
    allow_partial: bool = False,
) -> list[dict[str, str]]:
    """The filled rows in template order. Refuses unknown ids, a changed rule category, an
    unknown human category and, unless ``allow_partial``, an empty one."""
    expected = {row["review_id"]: row for row in template}
    by_id: dict[str, dict[str, str]] = {}
    for row in filled:
        rid = row.get("review_id", "")
        if rid not in expected:
            raise ValueError(f"unknown review id {rid!r}")
        if rid in by_id:
            raise ValueError(f"review id {rid} appears twice")
        ref = expected[rid]
        for key in ("failure_id", "rule_category"):
            if row.get(key) != ref[key]:
                raise ValueError(f"{rid}: {key} changed ({row.get(key)!r} != {ref[key]!r})")
        human = row.get("human_category", "")
        if human and human not in taxonomy.CATEGORIES:
            raise ValueError(f"{rid}: unknown category {human!r}")
        by_id[rid] = {k: row.get(k, "") for k in REVIEW_COLUMNS}
    missing = [rid for rid in expected if not by_id.get(rid, {}).get("human_category")]
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} items without a human category: {', '.join(missing)}")
    return [by_id[rid] for rid in expected if rid in by_id]


def agreement(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Rule vs human over the rows that have a human category."""
    done = [r for r in rows if r["human_category"]]
    rule = {r["review_id"]: r["rule_category"] for r in done}
    human = {r["review_id"]: r["human_category"] for r in done}
    used = [c for c in taxonomy.CATEGORIES if c in set(rule.values()) | set(human.values())]
    report = compare(rule, human, used)
    report.pop("share_" + used[0], None)  # meaningful for two labels only
    report["confusion"]["rows"] = "rule"
    report["confusion"]["columns"] = "human"
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"reviewed": 0, "agreed": 0})
    for r in done:
        entry = per_category[r["rule_category"]]
        entry["reviewed"] += 1
        entry["agreed"] += int(r["rule_category"] == r["human_category"])
    not_agent = Counter(r["rule_category"] for r in done if r["human_category"] in REVIEWER_ONLY)
    return {
        "n_reviewed": len(done),
        "n_in_sample": len(rows),
        "rule_vs_human": report,
        "per_rule_category": {c: per_category[c] for c in taxonomy.CATEGORIES if c in per_category},
        "human_categories": dict(Counter(r["human_category"] for r in done).most_common()),
        "not_agent_failures_by_rule_category": dict(not_agent.most_common()),
    }


def _fmt(stat: Mapping[str, Any]) -> str:
    if stat["value"] is None:
        return "undefined"
    if stat["ci_low"] is None:
        return f"{stat['value']:.2f}"
    return f"{stat['value']:.2f} [{stat['ci_low']:.2f}, {stat['ci_high']:.2f}]"


def write_labels(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True, help="classify.py output folder")
    parser.add_argument("--labels", type=Path, required=True, help="filled copy of the template")
    parser.add_argument("--reviewer", required=True, help="who reviewed, recorded as is")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    review_dir = args.report / "review"
    template = read_csv(review_dir / "labels_template.csv")
    try:
        rows = validate(template, read_csv(args.labels), allow_partial=args.allow_partial)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    write_labels(review_dir / "labels.csv", rows)
    out = {"reviewer": args.reviewer, **agreement(rows)}
    (review_dir / "agreement.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"{out['n_reviewed']}/{out['n_in_sample']} reviewed; "
        f"accuracy {_fmt(out['rule_vs_human']['accuracy'])}; "
        f"kappa {_fmt(out['rule_vs_human']['kappa'])}"
    )


if __name__ == "__main__":
    main()
