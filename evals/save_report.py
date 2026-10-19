"""Copy a finished run directory into a committed report directory.

Copies ``summary.json``, ``run.json``, ``llm_calls.jsonl`` and, when present, ``teaching.jsonl``
and ``trace_report.json``; rewrites ``predictions.jsonl`` without the ``gold_sql`` field (BIRD
gold queries are CC BY-SA 4.0 content and are not redistributed in this repository). Refuses an
existing report directory unless ``--force``.

    uv run python -m evals.save_report evals/results/text2sql/<run id> evals/reports/<date>-<name>
"""

import argparse
import json
import shutil
from pathlib import Path

COPIED = ("summary.json", "run.json", "llm_calls.jsonl", "teaching.jsonl", "trace_report.json")
DROPPED_FIELDS = ("gold_sql",)


def save_report(run_dir: Path, report_dir: Path, *, force: bool = False) -> list[str]:
    """Copy the publishable files of ``run_dir`` into ``report_dir``; return their names."""
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"{run_dir} has no summary.json (unfinished or wrong directory)")
    if report_dir.exists() and any(report_dir.iterdir()) and not force:
        raise FileExistsError(f"{report_dir} exists; use --force to overwrite")
    report_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in COPIED:
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, report_dir / name)
            written.append(name)
    predictions = run_dir / "predictions.jsonl"
    if predictions.exists():
        with (
            open(predictions, encoding="utf-8") as src,
            open(report_dir / "predictions.jsonl", "w", encoding="utf-8") as dst,
        ):
            for line in src:
                if line.strip():
                    record = json.loads(line)
                    for field in DROPPED_FIELDS:
                        record.pop(field, None)
                    dst.write(json.dumps(record) + "\n")
        written.append("predictions.jsonl")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    written = save_report(args.run_dir, args.report_dir, force=args.force)
    print(f"{args.report_dir}: {', '.join(written)}")


if __name__ == "__main__":
    main()
