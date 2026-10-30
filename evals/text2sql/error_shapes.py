"""Classify wrong predictions of a committed run by the shape of their result vs the gold one.

Re-executes the predicted and gold SQL (gold SQL comes from the local BIRD data, run
``make bird-data`` first) and puts every wrong-result prediction in one bucket:
more columns than gold (and whether the gold values are all contained in the prediction),
fewer columns, empty result, different row count, or same shape with different values.
Errors, timeouts and missing SQL keep their failure type.

    uv run python -m evals.text2sql.error_shapes 2026-10-19-haiku-test-350 [--write]
"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from evals.text2sql.data import db_path, load_questions
from evals.text2sql.scorer import execute

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def _n_columns(database: Path, sql: str) -> int | None:
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return len(conn.execute(sql).description or [])
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def classify(report: str) -> dict[str, Counter[str]]:
    """{system: Counter(bucket -> count)} over the wrong predictions of ``report``."""
    questions = {q.question_id: q for q in load_questions()}
    buckets: dict[str, Counter[str]] = {}
    with open(REPORTS / report / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            counter = buckets.setdefault(p["system"], Counter())
            if p["correct"]:
                continue
            if p["failure_type"] != "wrong_result":
                counter[p["failure_type"]] += 1
                continue
            q = questions[int(p["question_id"])]
            database = db_path(q.db_id)
            pred, gold = execute(database, p["pred_sql"], 30), execute(database, q.sql, 30)
            if pred.rows is None or gold.rows is None:
                counter["re-execution failed"] += 1
                continue
            pred_cols, gold_cols = _n_columns(database, p["pred_sql"]), _n_columns(database, q.sql)
            if pred_cols is None or gold_cols is None:
                counter["re-execution failed"] += 1
            elif pred_cols > gold_cols:
                gold_values = {v for row in gold.rows for v in row}
                pred_values = {v for row in pred.rows for v in row}
                contained = gold_values <= pred_values
                counter["more columns" + (" (gold values contained)" if contained else "")] += 1
            elif pred_cols < gold_cols:
                counter["fewer columns"] += 1
            elif not pred.rows:
                counter["empty result"] += 1
            elif len(pred.rows) != len(gold.rows):
                counter["different row count"] += 1
            else:
                counter["same shape, different values"] += 1
    return buckets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="directory name under evals/reports/")
    parser.add_argument(
        "--write", action="store_true", help="also write error_shapes.json into the report"
    )
    args = parser.parse_args(argv)
    buckets = classify(args.report)
    for system, counter in buckets.items():
        print(f"{system}: {sum(counter.values())} wrong")
        for bucket, count in counter.most_common():
            print(f"  {bucket}: {count}")
    if args.write:
        out = {
            system: {"wrong": sum(c.values()), "buckets": dict(c.most_common())}
            for system, c in buckets.items()
        }
        path = REPORTS / args.report / "error_shapes.json"
        path.write_text(json.dumps({"report": args.report, "systems": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
