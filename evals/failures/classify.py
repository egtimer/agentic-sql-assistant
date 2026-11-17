"""Classify the failures of committed runs with the rules of ``taxonomy.py`` and prepare the
hand review. Free: no LLM calls.

Sources:

- one BIRD run of agent v2 (``--bird-report``), joined with its final-answer pass
  (``--answers``, from ``final_answer.py``); the final and gold SQL are re-executed read-only
  to compare the shape of their results (needs ``make bird-data``);
- any number of agent v1 tool-eval runs (``--tools-report``, repeatable; each a directory with
  ``run-r*/predictions.jsonl`` or one ``predictions.jsonl``).

Writes into ``--out`` (committed):

- ``classification.jsonl``  one row per failure: source, item, primary category, all tags
- ``detectors.json``        what each rule finds on every item, failures or not (a tool error
                            the agent recovered from still counts here)
- ``traces/<failure id>.json``  the trace of every failure, small and readable on GitHub: the
                            question, the SQL or tool steps, the answer, the labels. No gold SQL,
                            no gold rows, no judge rationale (it may quote gold rows)
- ``review/review_sheet.md`` and ``review/labels_template.csv``: the stratified sample for the
  hand review (about 5 per category, ``--review-n`` in total, seed 0)

and, gitignored, ``evals/results/failures/<answers run>/review_private.md``: the same sample
with the reference result next to each answer, which the reviewer needs and which is BIRD
content that is not committed.

    uv run python -m evals.failures.classify \\
        --bird-report 2026-11-16-noise-bird-haiku-test-350/run-r01 \\
        --answers evals/results/failures/final-answer-haiku-r01 \\
        --tools-report 2026-11-16-noise-tools-haiku-test \\
        --tools-report 2026-11-16-noise-tools-sonnet-test \\
        --out evals/reports/2026-11-17-failure-analysis
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.failures import taxonomy
from evals.judge.judge import format_preview
from evals.memory.common import read_jsonl
from evals.text2sql.data import REPO_ROOT, Question, db_path, load_questions
from evals.text2sql.scorer import MAX_RESULT_ROWS
from src.sql_guard import GuardPolicy, guarded_execute

REPORTS = REPO_ROOT / "evals" / "reports"
TOOL_QUERIES = REPO_ROOT / "evals" / "tools" / "set" / "queries.jsonl"
DEFAULT_REVIEW_N = 40
PER_CATEGORY = 5
SEED = 0
GOLD_FAILED = "gold sql failed"
REVIEW_COLUMNS = ("review_id", "failure_id", "rule_category", "human_category", "note")


def result_shape(database: Path, sql: str, timeout_s: float = 30) -> tuple[int | None, int | None]:
    """``(columns, rows)`` of ``sql`` under the scorer's guard; ``(None, None)`` on error."""
    policy = GuardPolicy(timeout_s=timeout_s, max_rows=MAX_RESULT_ROWS)
    result = guarded_execute(database, sql, policy)
    if result.error is not None:
        return None, None
    rows = None if result.truncated else len(result.rows)
    return len(result.columns), rows


def bird_shape(question: Question, pred_sql: str, databases_dir: Path | None) -> dict[str, Any]:
    database = db_path(question.db_id, databases_dir)
    pred_columns, pred_rows = result_shape(database, pred_sql)
    gold_columns, gold_rows = result_shape(database, question.sql)
    return {
        "pred_columns": pred_columns,
        "gold_columns": gold_columns,
        "pred_rows": pred_rows,
        "gold_rows": gold_rows,
    }


def _model(report: str) -> str:
    for name in ("haiku", "sonnet", "opus"):
        if name in report:
            return name
    return "model"


def classify_bird(
    report: str,
    answers_dir: Path,
    questions: Mapping[int, Question],
    *,
    databases_dir: Path | None = None,
    reports_root: Path = REPORTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Failures of the ``agent_v2`` rows of a BIRD run, and the detector counts on all rows."""
    predictions = [
        r
        for r in read_jsonl(reports_root / report / "predictions.jsonl")
        if r["system"] == "agent_v2"
    ]
    answers = {int(a["question_id"]): a for a in read_jsonl(answers_dir / "answers.jsonl")}
    failures: list[dict[str, Any]] = []
    detectors: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    for pred in sorted(predictions, key=lambda r: int(r["question_id"])):
        qid = int(pred["question_id"])
        if str(pred.get("error") or "").startswith(GOLD_FAILED):
            excluded["gold query failed (not scored)"] += 1
            continue
        answer = answers.get(qid)
        if answer is None:
            excluded["no final answer"] += 1
            continue
        record = {
            **pred,
            "answer": answer["answer"],
            "judge_label": answer["judge_label"],
        }
        q = questions[qid]
        is_failure = taxonomy.bird_is_failure(record)
        shape = bird_shape(q, pred["pred_sql"], databases_dir) if not pred["correct"] else None
        tags = taxonomy.bird_tags(record, shape)
        detectors.update(tags)
        if not is_failure:
            continue
        failures.append(
            {
                "failure_id": f"bird-{_model(report)}-{qid}",
                "source": "bird",
                "report": report,
                "item": qid,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "primary": taxonomy.primary(tags),
                "tags": tags,
                "ex_correct": bool(pred["correct"]),
                "judge_label": answer["judge_label"],
                "shape": shape,
                "_trace": {
                    "question": q.question,
                    "evidence": q.evidence,
                    "attempts": pred.get("attempts"),
                    "verdicts": [
                        {"kind": v["kind"], "text": v["text"]} for v in pred.get("verdicts") or []
                    ],
                    "fixes_used": pred.get("fixes_used"),
                    "final_sql": pred["pred_sql"],
                    "answer": answer["answer"],
                },
                "_reference": answer.get("gold_preview"),
            }
        )
    stats = {
        "items": len(predictions),
        "excluded": dict(excluded),
        "failures": len(failures),
        "tags_on_all_items": dict(detectors.most_common()),
    }
    return failures, stats


def _tool_runs(report_dir: Path) -> list[Path]:
    runs = sorted(report_dir.glob("run-r*/predictions.jsonl"))
    return runs or [report_dir / "predictions.jsonl"]


def classify_tools(
    reports: Sequence[str],
    checks_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reports_root: Path = REPORTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Failures of the ``agent_v1`` rows of tool-eval runs, and detector counts on all rows."""
    failures: list[dict[str, Any]] = []
    detectors: Counter[str] = Counter()
    recovered: Counter[str] = Counter()
    n_items = 0
    for report in reports:
        for path in _tool_runs(reports_root / report):
            run = path.parent.name if path.parent.name.startswith("run-") else "run"
            for record in read_jsonl(path):
                if record["system"] != "agent_v1":
                    continue
                n_items += 1
                tags = taxonomy.tools_tags(record, checks_by_id.get(record["id"]))
                detectors.update(tags)
                if not taxonomy.tools_is_failure(record):
                    recovered.update(tags)
                    continue
                failures.append(
                    {
                        "failure_id": f"tools-{_model(report)}-{run}-{record['id']}",
                        "source": "tools",
                        "report": f"{report}/{run}",
                        "item": record["id"],
                        "category": record.get("category"),
                        "primary": taxonomy.primary(tags),
                        "tags": tags,
                        "answer_correct": record.get("answer_correct"),
                        "exact": record.get("exact"),
                        "_trace": {
                            "steps": record.get("steps"),
                            "response": record.get("response"),
                            "wrong_tools": record.get("wrong_tools"),
                            "missing_tools": record.get("missing_tools"),
                        },
                        "_reference": None,
                    }
                )
    stats = {
        "items": n_items,
        "failures": len(failures),
        "tags_on_all_items": dict(detectors.most_common()),
        "tags_on_items_answered_correctly": dict(recovered.most_common()),
    }
    return failures, stats


def sample_for_review(
    failures: Sequence[Mapping[str, Any]], n: int = DEFAULT_REVIEW_N, seed: int = SEED
) -> list[Mapping[str, Any]]:
    """Up to ``PER_CATEGORY`` failures of every primary category (seeded), then more from the
    largest categories until ``n``. The same question is not sampled twice in a category (the
    tool runs repeat each query). Ordered by a seeded shuffle so categories are mixed."""
    rng = random.Random(seed)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for f in sorted(failures, key=lambda f: f["failure_id"]):
        by_category[f["primary"]].append(f)
    for members in by_category.values():
        rng.shuffle(members)
    for category, members in by_category.items():  # one failure per question and category
        seen: set[tuple[str, Any]] = set()
        unique = []
        for f in members:
            if (f["source"], f["item"]) not in seen:
                seen.add((f["source"], f["item"]))
                unique.append(f)
        by_category[category] = unique
    chosen = [f for members in by_category.values() for f in members[:PER_CATEGORY]]
    leftovers = sorted(
        ((c, m[PER_CATEGORY:]) for c, m in by_category.items() if len(m) > PER_CATEGORY),
        key=lambda cm: -len(cm[1]),
    )
    while len(chosen) < n and any(rest for _, rest in leftovers):
        for _, rest in leftovers:
            if rest and len(chosen) < n:
                chosen.append(rest.pop(0))
    chosen = chosen[:n] if len(chosen) > n else chosen
    rng.shuffle(chosen)
    return chosen


def public_trace(failure: Mapping[str, Any]) -> dict[str, Any]:
    keys = [k for k in failure if not k.startswith("_")]
    return {**{k: failure[k] for k in keys}, "trace": failure["_trace"]}


def _evidence(failure: Mapping[str, Any]) -> str:
    """One line a reviewer can check the rule against."""
    if failure["source"] == "bird":
        shape = failure.get("shape") or {}
        bits = [
            f"EX {'ok' if failure['ex_correct'] else 'wrong'}",
            f"judge {failure['judge_label']}",
            f"fixes {failure['_trace']['fixes_used']}",
        ]
        if shape:
            bits.append(
                f"columns {shape['pred_columns']} vs gold {shape['gold_columns']}, "
                f"rows {shape['pred_rows']} vs gold {shape['gold_rows']}"
            )
        return "; ".join(bits)
    steps = failure["_trace"]["steps"] or []
    calls = ", ".join(f"{s['tool']}({'ok' if s['success'] else 'failed'})" for s in steps)
    return f"answer check {failure['answer_correct']}; exact {failure['exact']}; {calls}"


def _md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def review_sheet(sample: Sequence[Mapping[str, Any]], *, private: bool) -> str:
    """The sheet the reviewer reads: one section per sampled failure."""
    lines = [
        "# Failure review sheet" + (" (private: holds BIRD reference rows)" if private else ""),
        "",
        "For each item, read the question, what the agent did and its answer, and write in "
        "`labels.csv` the category you would give it (one of the list below, or "
        "`not_a_failure`). The rule's category is shown so you can disagree with it; "
        "the reference result is only in the private copy.",
        "",
        "| Category | Meaning |",
        "|---|---|",
        *(f"| `{c}` | {_md(taxonomy.DESCRIPTIONS[c])} |" for c in taxonomy.CATEGORIES),
        "",
    ]
    for n, f in enumerate(sample, start=1):
        trace = f["_trace"]
        lines += [
            f"## r-{n:02d} · `{f['failure_id']}` · rule: `{f['primary']}`",
            "",
            f"Tags: {', '.join(f'`{t}`' for t in f['tags']) or '(none)'}. {_evidence(f)}.",
            "",
        ]
        if f["source"] == "bird":
            lines += [f"**Question:** {_md(trace['question'])}", ""]
            if trace["evidence"]:
                lines += [f"**External knowledge:** {_md(trace['evidence'])}", ""]
            lines += ["**Final SQL:**", "", "```sql", trace["final_sql"].strip(), "```", ""]
            if trace["fixes_used"]:
                lines += [f"Earlier attempts: {len(trace['attempts']) - 1}.", ""]
            lines += [f"**Answer:** {_md(trace['answer'])}", ""]
            if private and f.get("_reference"):
                lines += [
                    "**Reference result:**",
                    "",
                    "```",
                    format_preview(f["_reference"]),
                    "```",
                ]
                lines.append("")
        else:
            lines.append("**Steps:**")
            lines.append("")
            for s in trace["steps"] or []:
                status = "ok" if s["success"] else f"failed: {s['error']}"
                lines.append(f"- `{s['tool']}` ({status}): `{_md(s['input'])[:200]}`")
            for n_step, s in enumerate(trace["steps"] or [], start=1):
                output = str(s.get("output") or "")
                cut = taxonomy.AGENT_TOOL_OUTPUT_CHARS
                shown = output[:cut] + "\n[--- the agent's prompt stops here ---]\n" + output[cut:]
                lines += ["", f"Output of step {n_step} (as recorded, at most 2,000 characters):"]
                lines += ["", "```", shown if len(output) > cut else output, "```"]
            lines += ["", f"**Answer:** {_md(trace['response'])}", ""]
        lines.append(f"[trace](../traces/{f['failure_id']}.json)" if not private else "")
        lines.append("")
    return "\n".join(lines)


def write_outputs(
    out: Path,
    failures: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    sample: Sequence[Mapping[str, Any]],
    private_dir: Path | None,
) -> None:
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    rows = [public_trace(f) for f in failures]
    with open(out / "classification.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            slim = {k: v for k, v in row.items() if k != "trace"}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    for row in rows:
        path = out / "traces" / f"{row['failure_id']}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "detectors.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out / "review" / "review_sheet.md").write_text(
        review_sheet(sample, private=False), encoding="utf-8"
    )
    with open(out / "review" / "labels_template.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(REVIEW_COLUMNS)
        for n, f in enumerate(sample, start=1):
            writer.writerow([f"r-{n:02d}", f["failure_id"], f["primary"], "", ""])
    if private_dir is not None:
        (private_dir / "review_private.md").write_text(
            review_sheet(sample, private=True), encoding="utf-8"
        )


def counts(failures: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """{source: {primary category: count}}, categories in ``PRIORITY`` order."""
    out: dict[str, dict[str, int]] = {}
    for source in sorted({f["source"] for f in failures}):
        c = Counter(f["primary"] for f in failures if f["source"] == source)
        out[source] = {k: c[k] for k in taxonomy.PRIORITY if c[k]}
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--bird-report", required=True, help="BIRD run under evals/reports/")
    parser.add_argument("--answers", type=Path, required=True, help="final_answer.py run dir")
    parser.add_argument("--tools-report", action="append", default=[], help="repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-n", type=int, default=DEFAULT_REVIEW_N)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    questions = {q.question_id: q for q in load_questions()}
    checks = {q["id"]: q["checks"] for q in read_jsonl(TOOL_QUERIES)}
    bird, bird_stats = classify_bird(args.bird_report, args.answers, questions)
    tools, tools_stats = classify_tools(args.tools_report, checks)
    failures = [*bird, *tools]
    sample = sample_for_review(failures, args.review_n)
    stats = {
        "bird": {"report": args.bird_report, "answers": args.answers.name, **bird_stats},
        "tools": {"reports": args.tools_report, **tools_stats},
        "primary_counts": counts(failures),
        "review_sample": {
            "n": len(sample),
            "per_category_max": PER_CATEGORY,
            "seed": SEED,
            "by_rule_category": dict(Counter(f["primary"] for f in sample)),
        },
    }
    write_outputs(args.out, failures, stats, sample, args.answers)
    print(json.dumps(stats["primary_counts"], indent=2))
    print(f"{len(failures)} failures, {len(sample)} sampled for review; written to {args.out}")


if __name__ == "__main__":
    main()
