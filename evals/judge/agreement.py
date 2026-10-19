"""Agreement between the judge, the human labels and execution accuracy (EX).

Three comparisons, each with accuracy (share of items where both say the same), Cohen's kappa
and the confusion matrix, every number with a 95% percentile bootstrap CI (items resampled
with replacement, B = 10000, seed 0; the same style as ``evals/text2sql/stats.py``):

- judge vs human: items the human did not mark ``unsure`` and the judge could read
- human vs EX:    does the SQL-level metric agree with a human reading of the answer?
- judge vs EX:    the same question for the judge, over every item the judge could read

A resample in which kappa is undefined (chance agreement 1: one label only on both sides) has
no value: it is left out of the interval and counted in ``frac_undefined``.

    uv run python -m evals.judge.agreement \\
        --items evals/results/judge/<items run> --judgements evals/results/judge/<judge run>
"""

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.judge.judge import CORRECT, INCORRECT, LABELS
from evals.memory.common import read_jsonl
from evals.text2sql.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, _percentile_interval

UNSURE = "unsure"
Metric = Callable[[Sequence[str], Sequence[str]], float | None]


def accuracy(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Share of positions where the two raters give the same label; None for no items."""
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa ``(po - pe) / (1 - pe)`` of two raters; None when ``pe`` is 1.

    ``po`` is the observed agreement, ``pe`` the agreement expected from the two label
    distributions. ``pe == 1`` (both raters used one and the same label throughout) leaves the
    ratio undefined.
    """
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[label] * cb[label] for label in ca.keys() | cb.keys()) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def confusion(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Counts with ``a`` on the rows and ``b`` on the columns, in the order of ``labels``."""
    counts = Counter(zip(a, b, strict=True))
    return {
        "labels": list(labels),
        "rows": "first rater",
        "columns": "second rater",
        "matrix": [[counts[(x, y)] for y in labels] for x in labels],
    }


def bootstrap_metric(
    a: Sequence[str],
    b: Sequence[str],
    metric: Metric,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """``metric`` on the items plus its percentile bootstrap CI (items resampled in pairs)."""
    n = len(a)
    value = metric(a, b)
    out: dict[str, Any] = {"value": value, "ci_low": None, "ci_high": None, "frac_undefined": None}
    if n == 0:
        return out
    rng = random.Random(seed)
    idx = range(n)
    stats: list[float] = []
    undefined = 0
    for _ in range(resamples):
        pick = rng.choices(idx, k=n)
        stat = metric([a[i] for i in pick], [b[i] for i in pick])
        if stat is None:
            undefined += 1
        else:
            stats.append(stat)
    if stats:
        out["ci_low"], out["ci_high"] = _percentile_interval(sorted(stats), confidence)
    out["frac_undefined"] = undefined / resamples if resamples else None
    return out


def compare(
    first: Mapping[str, str],
    second: Mapping[str, str],
    labels: Sequence[str] = LABELS,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Agreement of two raters over the item ids both have (sorted by id)."""
    ids = sorted(first.keys() & second.keys())
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    positive = labels[0]
    return {
        "n": len(ids),
        "accuracy": bootstrap_metric(a, b, accuracy, resamples=resamples, seed=seed),
        "kappa": bootstrap_metric(a, b, cohens_kappa, resamples=resamples, seed=seed),
        "confusion": confusion(a, b, labels),
        f"share_{positive}": {
            "first": sum(x == positive for x in a) / len(a) if a else None,
            "second": sum(y == positive for y in b) / len(b) if b else None,
        },
    }


def ex_labels(items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """EX of the final SQL as a label, for the items that have one."""
    return {
        i["item_id"]: CORRECT if i["ex_correct"] else INCORRECT
        for i in items
        if i.get("ex_correct") is not None
    }


def agreement_report(
    items: Sequence[Mapping[str, Any]],
    judgements: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """All three comparisons plus the counts of excluded items."""
    ids = {i["item_id"] for i in items}
    human_all = {label["item_id"]: label["label"] for label in labels}
    human = {i: v for i, v in human_all.items() if v in LABELS and i in ids}
    judge_all = {j["item_id"]: j["label"] for j in judgements}
    judge = {i: v for i, v in judge_all.items() if v in LABELS and i in ids}
    ex = ex_labels(items)
    options = {"resamples": resamples, "seed": seed}
    return {
        "counts": {
            "items": len(ids),
            "human_labelled": len(human_all),
            "human_unsure": sum(1 for v in human_all.values() if v == UNSURE),
            "judged": len(judge_all),
            "judge_unparsed": sum(1 for v in judge_all.values() if v not in LABELS),
            "judge_by_label": dict(Counter(judge_all.values())),
        },
        "judge_vs_human": compare(judge, human, **options),
        "human_vs_ex": compare(human, ex, **options),
        "judge_vs_ex": compare(judge, ex, **options),
        "bootstrap": {
            "B": resamples,
            "seed": seed,
            "confidence": 0.95,
            "method": "percentile bootstrap, items resampled with replacement",
        },
    }


def _ci(stat: Mapping[str, Any], digits: int = 2) -> str:
    if stat["value"] is None:
        return "undefined"
    text = f"{stat['value']:.{digits}f}"
    if stat["ci_low"] is not None:
        text += f" [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"
    return text


def format_report(report: Mapping[str, Any]) -> str:
    """Short text table of the report."""
    c = report["counts"]
    lines = [
        f"items {c['items']}  human unsure {c['human_unsure']}  judge unparsed "
        f"{c['judge_unparsed']}  (judge labels {c['judge_by_label']})",
        "",
        f"{'comparison':<18}{'n':>4}  {'accuracy [95% CI]':<24}{'kappa [95% CI]':<24}",
    ]
    for key, name in (
        ("judge_vs_human", "judge vs human"),
        ("human_vs_ex", "human vs EX"),
        ("judge_vs_ex", "judge vs EX"),
    ):
        r = report[key]
        lines.append(f"{name:<18}{r['n']:>4}  {_ci(r['accuracy']):<24}{_ci(r['kappa']):<24}")
    lines.append("")
    for key in ("judge_vs_human", "human_vs_ex", "judge_vs_ex"):
        m = report[key]["confusion"]
        lines.append(f"{key} (rows: first, columns: second; labels {m['labels']}): {m['matrix']}")
    return "\n".join(lines)


def run_dir_file(path: Path, name: str) -> Path:
    """``path`` itself when it is a file, else ``path/name``."""
    return path if path.is_file() else path / name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=Path, required=True, help="items run directory")
    parser.add_argument("--judgements", type=Path, required=True, help="judge run directory")
    parser.add_argument("--labels", type=Path, help="labels.jsonl (default: in the items run)")
    parser.add_argument("--out", type=Path, help="agreement.json (default: in the judge run)")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    items_path = args.items / "items.jsonl"
    items = read_jsonl(items_path)
    if not items:
        raise SystemExit(f"no items in {items_path}")
    labels_path = args.labels or args.items / "labels.jsonl"
    labels = read_jsonl(labels_path)
    if not labels:
        raise SystemExit(f"no labels in {labels_path} (run import_labels first)")
    judgements = read_jsonl(run_dir_file(args.judgements, "judgements.jsonl"))
    report = agreement_report(items, judgements, labels, resamples=args.resamples, seed=args.seed)
    report["items_run"] = args.items.name
    report["judge_run"] = args.judgements.name
    report["labeler"] = labels[0].get("labeler")
    out = args.out or run_dir_file(args.judgements, "agreement.json").with_name("agreement.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
