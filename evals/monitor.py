"""Run the production detectors on a run and, with a reference, check accuracy (ADR-012).

    uv run python -m evals.monitor <run dir> [--reference <report dir>] [--system agent_v2]
        [--config configs/alerts.yaml] [--out <alerts.json>]

``<run dir>`` holds a ``predictions.jsonl`` from ``evals.text2sql.run`` or ``evals.tools.run``.
Detectors from ``src/monitoring/detectors.py`` run on the records of ``--system`` (default:
``agent_v2`` for text-to-SQL runs, ``agent_v1`` for tool runs). With ``--reference``, accuracy
(EX, or the answer check of the tool eval) is compared on the shared questions with a paired
bootstrap; it alerts when it dropped by more than ``accuracy_drop.max_drop_pts`` and the 95% CI
is below zero, the rule of ``evals.text2sql.check_regression``.

Writes ``alerts.json`` (next to the predictions unless ``--out``) and a markdown summary to
stdout and ``GITHUB_STEP_SUMMARY``. Exit code 1 when any detector alerts, 2 on unusable input.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.text2sql.stats import paired_bootstrap
from src.monitoring.detectors import DEFAULT_CONFIG, Finding, load_config, record_id, run_detectors


def load_records(run_dir: Path, system: str | None) -> tuple[str, list[dict[str, Any]]]:
    """``(system, records)`` from ``run_dir/predictions.jsonl``; the last record per id wins."""
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise ValueError(f"{path} not found")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if system is None:
        system = "agent_v1" if rows and "steps" in rows[0] else "agent_v2"
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("system") == system:
            by_id[record_id(row)] = row
    if not by_id:
        raise ValueError(f"no records for system {system!r} in {path}")
    return system, list(by_id.values())


def correct(record: Mapping[str, Any]) -> bool | None:
    if "correct" in record:
        return bool(record["correct"])
    value = record.get("answer_correct")
    return None if value is None else bool(value)


def accuracy_drop(
    current: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    max_drop_pts: float,
) -> Finding:
    cur = {record_id(r): correct(r) for r in current}
    ref = {record_id(r): correct(r) for r in reference}
    shared = sorted(k for k in cur.keys() & ref.keys() if cur[k] is not None and ref[k] is not None)
    if not shared:
        raise ValueError("run and reference share no scored question")
    index = {k: i for i, k in enumerate(shared)}
    paired = paired_bootstrap(
        {index[k]: bool(cur[k]) for k in shared}, {index[k]: bool(ref[k]) for k in shared}
    )
    assert paired is not None
    drop = -paired["diff"] * 100 + 0.0
    alert = drop > max_drop_pts and paired["ci_high"] < 0
    worse = [k for k in shared if ref[k] and not cur[k]]
    return Finding(
        "accuracy_drop",
        drop,
        max_drop_pts,
        alert,
        f"{len(shared)} shared questions: {sum(bool(ref[k]) for k in shared)} right in the "
        f"reference, {sum(bool(cur[k]) for k in shared)} now; change "
        f"{paired['diff'] * 100:+.1f} pts [{paired['ci_low'] * 100:+.1f}, "
        f"{paired['ci_high'] * 100:+.1f}]",
        worse[:20],
    )


def render(system: str, run: str, findings: Sequence[Finding]) -> str:
    alerts = [f for f in findings if f.alert]
    lines = [
        f"### Monitor: `{system}` in `{run}`",
        "",
        "| detector | value | limit | status | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in findings:
        status = "ALERT" if f.alert else "ok"
        lines.append(
            f"| {f.detector} | {f.value + 0.0:.4g} | {f.limit:g} | {status} | {f.detail} |"
        )
    lines += ["", f"**{len(alerts)} alert(s).**" if alerts else "**No alerts.**"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run", type=Path, help="run directory with predictions.jsonl")
    parser.add_argument("--reference", type=Path, help="committed report to compare accuracy")
    parser.add_argument("--system", help="records to check (default: by the eval's shape)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="alerts.json path (default: in the run dir)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        system, records = load_records(args.run, args.system)
        findings = run_detectors(records, config)
        if args.reference is not None:
            _, reference = load_records(args.reference, system)
            limit = float(config.get("accuracy_drop", {}).get("max_drop_pts", 10.0))
            findings.append(accuracy_drop(records, reference, limit))
    except (OSError, ValueError, KeyError) as e:
        print(f"monitor: {e}", file=sys.stderr)
        return 2
    out = args.out or args.run / "alerts.json"
    payload = {
        "run": str(args.run),
        "system": system,
        "n_records": len(records),
        "reference": None if args.reference is None else str(args.reference),
        "config": str(args.config),
        "alerts": sum(f.alert for f in findings),
        "findings": [f.as_dict() for f in findings],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = render(system, args.run.name, findings)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if payload["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
