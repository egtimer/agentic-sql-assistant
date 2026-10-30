"""Build the agent v1 tool-selection set from ``specs.py`` (free, no LLM).

Seeds the demo database into a temporary directory (``src.db.seed``, fixed seed 42),
resolves every ``sql_*`` check to the values of its gold query, splits the queries into
dev and test, stratified by category (``--dev-fraction``, seed 0), and writes into
``--out``:

- ``queries.jsonl``     one query per line with its split, tools and resolved checks
- ``review_sheet.md``   the queries with expected tools and gold answers, for the human
                        review before any run (no model output in it)
- ``review.csv``        the same, with empty ``review`` and ``note`` columns

    uv run python -m evals.tools.build_set --out evals/tools/set
"""

import argparse
import csv
import json
import random
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.tools.specs import SPECS
from src.db.seed import seed_database

DEFAULT_OUT = Path(__file__).resolve().parent / "set"
TOOLS = ("sql_query", "doc_search", "calculator")


def resolve_checks(specs: list[dict[str, Any]], db: Path) -> list[dict[str, Any]]:
    """Copies of ``specs`` whose ``sql_*`` checks carry the gold ``values``."""
    conn = sqlite3.connect(db)
    try:
        out = []
        for spec in specs:
            checks = []
            for check in spec["checks"]:
                check = dict(check)
                if check["kind"] in ("sql_number", "sql_text"):
                    rows = conn.execute(check["sql"]).fetchall()
                    if not rows or rows[0][0] is None:
                        raise ValueError(f"{spec['id']}: gold query returned nothing")
                    values = [row[0] for row in rows]
                    check["values"] = (
                        values if check["kind"] == "sql_number" else [str(v) for v in values]
                    )
                checks.append(check)
            out.append({**spec, "checks": checks})
        return out
    finally:
        conn.close()


def assign_splits(specs: list[dict[str, Any]], dev_fraction: float, seed: int) -> dict[str, str]:
    """``{id: "dev" | "test"}``, stratified by category, independent of spec order."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        by_category[spec["category"]].append(spec["id"])
    rng = random.Random(seed)
    splits: dict[str, str] = {}
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        rng.shuffle(ids)
        n_dev = round(len(ids) * dev_fraction)
        for i, qid in enumerate(ids):
            splits[qid] = "dev" if i < n_dev else "test"
    return splits


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def gold_text(spec: dict[str, Any]) -> str:
    parts = []
    for check in spec["checks"]:
        values = check["values"]
        parts.append(", ".join(_fmt(v) for v in values))
    return "; ".join(parts) or "(not scored)"


def gold_source(spec: dict[str, Any]) -> str:
    sources = [c["sql"] for c in spec["checks"] if "sql" in c]
    if spec.get("source"):
        sources.append(spec["source"])
    if spec["category"] == "calc":
        sources.append("arithmetic in the question")
    return " | ".join(sources)


def write_outputs(queries: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    lines = [
        "# Agent v1 tool-selection set: review sheet",
        "",
        "For every query, check the expected tools and the gold answer, then write `ok` or",
        "what to change in `review.csv` (same folder). `required`: tools the query needs;",
        "`optional`: allowed but not needed (arithmetic that SQL can also do); every other",
        "tool counts as wrong. Answers are scored by the numbers and names in `gold`",
        "(numbers by absolute value, with 1% relative tolerance; text without accents).",
        "`none` queries score tool selection only.",
        "No model has been run on these queries.",
        "",
        "| id | split | category | query | required | optional | gold | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for q in queries:
        source = gold_source(q).replace("|", "/")
        lines.append(
            f"| {q['id']} | {q['split']} | {q['category']} | {q['query']} | "
            f"{', '.join(q['required']) or '(none)'} | {', '.join(q['optional'])} | "
            f"{gold_text(q)} | {f'`{source}`' if source else ''} |"
        )
    (out / "review_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "review.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "split", "category", "query", "required", "optional", "gold", "review", "note"]
        )
        for q in queries:
            writer.writerow(
                [q["id"], q["split"], q["category"], q["query"], " ".join(q["required"]),
                 " ".join(q["optional"]), gold_text(q), "", ""]
            )  # fmt: skip


def build(dev_fraction: float = 0.4, seed: int = 0) -> list[dict[str, Any]]:
    for spec in SPECS:
        unknown = set(spec["required"]) | set(spec.get("optional", []))
        if unknown - set(TOOLS):
            raise ValueError(f"{spec['id']}: unknown tools {unknown - set(TOOLS)}")
    with tempfile.TemporaryDirectory() as tmp:
        db = seed_database(Path(tmp) / "sample.db")
        resolved = resolve_checks(SPECS, db)
    splits = assign_splits(resolved, dev_fraction, seed)
    return [
        {
            "id": s["id"],
            "split": splits[s["id"]],
            "category": s["category"],
            "query": s["query"],
            "required": list(s["required"]),
            "optional": list(s.get("optional", [])),
            "checks": s["checks"],
            "source": s.get("source"),
        }
        for s in resolved
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dev-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    queries = build(args.dev_fraction, args.seed)
    write_outputs(queries, args.out)
    n_dev = sum(q["split"] == "dev" for q in queries)
    print(f"{len(queries)} queries ({n_dev} dev, {len(queries) - n_dev} test) in {args.out}")


if __name__ == "__main__":
    main()
