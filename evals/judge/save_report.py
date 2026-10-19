"""Copy a judge calibration (items run, human labels, judge runs) into a committed report.

Leaves out what can quote BIRD gold rows (CC BY-SA content kept out of committed files): the
gold result preview of each item, the judges' rationales and the labeler's notes. Keeps what
every agreement number is computed from: per item the public fields plus the EX of the final
SQL (labeling is over, so EX no longer has to be hidden), the human label and each judge label.

    uv run python -m evals.judge.save_report --items evals/results/judge/judge-set-dev-50 \\
        --judge haiku=evals/results/judge/judge-haiku-dev-50 \\
        --judge sonnet=evals/results/judge/judge-sonnet-dev-50 \\
        --out evals/reports/<date>-judge-calibration-dev-50

Layout of the report: ``items.jsonl``, ``labels.jsonl``, ``items_run/`` (summary, run.json,
LLM calls of the answer set) and one directory per judge (``judgements.jsonl``, summary,
run.json, LLM calls, ``agreement.json``). Refuses a non-empty ``--out`` unless ``--force``.
"""

import argparse
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from evals.memory.common import read_jsonl

ITEM_FIELDS = (
    "item_id",
    "question_id",
    "db_id",
    "difficulty",
    "question",
    "evidence",
    "answer",
    "final_sql",
    "result_preview",
    "ex_correct",
)
LABEL_FIELDS = ("item_id", "label", "labeler")
JUDGEMENT_DROPPED = ("rationale",)
RUN_FILES = ("summary.json", "run.json", "llm_calls.jsonl")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_run_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def save_report(
    items_run: Path, judge_runs: Mapping[str, Path], out: Path, *, force: bool = False
) -> list[str]:
    """Write the report into ``out``; return the paths written, relative to ``out``."""
    if not (items_run / "labels.jsonl").exists():
        raise FileNotFoundError(f"{items_run} has no labels.jsonl (run import_labels first)")
    for name, run in judge_runs.items():
        if not (run / "agreement.json").exists():
            raise FileNotFoundError(f"judge {name}: {run} has no agreement.json")
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists; use --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(items_run / "items.jsonl")
    _write_jsonl(out / "items.jsonl", ({k: i.get(k) for k in ITEM_FIELDS} for i in items))
    labels = read_jsonl(items_run / "labels.jsonl")
    _write_jsonl(out / "labels.jsonl", ({k: lab.get(k) for k in LABEL_FIELDS} for lab in labels))
    _copy_run_files(items_run, out / "items_run")
    written = ["items.jsonl", "labels.jsonl", "items_run/"]

    for name, run in judge_runs.items():
        dst = out / name
        _copy_run_files(run, dst)
        shutil.copy(run / "agreement.json", dst / "agreement.json")
        judgements = read_jsonl(run / "judgements.jsonl")
        _write_jsonl(
            dst / "judgements.jsonl",
            ({k: v for k, v in j.items() if k not in JUDGEMENT_DROPPED} for j in judgements),
        )
        written.append(f"{name}/")
    return written


def _judge_arg(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=RUN_DIR")
    return name, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=Path, required=True, help="items run directory")
    parser.add_argument(
        "--judge", type=_judge_arg, action="append", required=True, help="NAME=RUN_DIR"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    written = save_report(args.items, dict(args.judge), args.out, force=args.force)
    print(f"{args.out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
