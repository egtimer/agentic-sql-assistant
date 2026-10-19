"""Build the item set for the answer judge: dev questions answered by agent v2 with the
respond step ON.

N dev questions are drawn stratified by database and difficulty (``evals/text2sql/sampling``,
fixed seed) and run through ``run_sql_agent`` (generate, check, respond). For each question
the run records the agent's answer text, its final SQL and result preview (columns and up to
10 rows), and, kept only in gitignored files, the gold result preview (the gold SQL run through
the scorer's guarded path) and the EX correctness of the final SQL.

Output in ``evals/results/judge/<run-id>/`` (gitignored):

- ``items.jsonl``         full records, with the gold result preview and EX (BIRD gold content)
- ``items_public.jsonl``  no gold SQL, no gold rows, no EX: what may be committed
- ``run.json`` (provenance), ``llm_calls.jsonl`` (usage), ``summary.json``, ``gold_cache.json``

Item ids ``j-01``... follow a seeded shuffle of the sample, so the order does not group by
database or difficulty.

Spends money with the Anthropic config (about 3 LLM calls per item, at most 4); the owner runs it:

    uv run python -m evals.judge.build_set --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --run-id judge-set-dev-50 --budget-usd 0.50 --launched-by owner
"""

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.memory.common import (
    ClientFactory,
    Engine,
    make_client,
    new_tracker,
    provenance,
    read_jsonl,
    timed_calls,
)
from evals.provenance import LAUNCHED_BY_ENV, git_info, sigterm_as_interrupt
from evals.text2sql.data import REPO_ROOT, Question, db_path, load_questions, load_split
from evals.text2sql.run import (
    DEFAULT_MAX_FIXES,
    PredictionWriter,
    load_llm_config,
    sampling_meta,
    select_questions,
)
from evals.text2sql.scorer import MAX_RESULT_ROWS, GoldCache, score_prediction
from src.llm import LLMConfig
from src.sql_agent.executor import Attempt, _cell
from src.sql_agent.graph import run_sql_agent
from src.sql_guard import GuardPolicy, guarded_execute

RESULTS_ROOT = REPO_ROOT / "evals" / "results" / "judge"
SPLIT = "dev"
DEFAULT_N = 50
PREVIEW_ROWS = 10
PUBLIC_FIELDS = (
    "item_id",
    "question_id",
    "db_id",
    "difficulty",
    "question",
    "evidence",
    "answer",
    "final_sql",
    "result_preview",
)


def assign_ids(questions: list[Question], seed: int) -> list[tuple[str, Question]]:
    """``j-01``... for the questions, in a seeded shuffle of the sorted ids."""
    order = sorted(questions, key=lambda q: q.question_id)
    random.Random(seed).shuffle(order)
    width = max(2, len(str(len(order))))
    return [(f"j-{n:0{width}d}", q) for n, q in enumerate(order, start=1)]


def attempt_preview(attempt: Attempt | None) -> dict[str, Any]:
    """Columns, row count and up to ``PREVIEW_ROWS`` rows of an agent attempt."""
    if attempt is None or not attempt.ok:
        return {"columns": [], "rows": [], "row_count": "n/a", "error": attempt and attempt.error}
    return {
        "columns": list(attempt.columns),
        "rows": [list(row) for row in attempt.preview[:PREVIEW_ROWS]],
        "row_count": attempt.row_count_text,
    }


def gold_preview(database: Path, gold_sql: str, timeout_s: float) -> dict[str, Any]:
    """Gold result through the scorer's guard policy: columns, count, first rows (or error)."""
    policy = GuardPolicy(timeout_s=timeout_s, max_rows=MAX_RESULT_ROWS)
    result = guarded_execute(database, gold_sql, policy)
    if result.error is not None or result.truncated:
        return {"columns": [], "rows": [], "row_count": "n/a", "error": result.error or "too large"}
    return {
        "columns": list(result.columns),
        "rows": [[_cell(v) for v in row] for row in result.rows[:PREVIEW_ROWS]],
        "row_count": str(len(result.rows)),
    }


def public_record(item: dict[str, Any]) -> dict[str, Any]:
    return {k: item[k] for k in PUBLIC_FIELDS}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    tmp.replace(path)


def build_set(
    *,
    config: LLMConfig,
    questions: list[Question],
    run_dir: Path,
    run_id: str,
    budget_usd: float | None = None,
    workers: int = 1,
    seed: int = 0,
    llm_config_name: str | None = None,
    client_factory: ClientFactory | None = None,
    databases_dir: Path | None = None,
    timeout_s: float = 30,
    max_fixes: int = DEFAULT_MAX_FIXES,
    extra_meta: dict[str, Any] | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
) -> dict[str, Any]:
    """Run the agent on ``questions`` and write the items (the run directory must be new)."""
    box: dict[str, Any] = {}
    with provenance(
        run_dir,
        run_id=run_id,
        argv=argv,
        llm_config_name=llm_config_name,
        config=config,
        budget_usd=budget_usd,
        launched_by=launched_by,
        summary_box=box,
    ):
        started_at = datetime.now(UTC).isoformat()
        tracker = new_tracker(budget_usd, run_dir)
        client = make_client(client_factory, config, tracker)
        gold_cache = GoldCache(run_dir / "gold_cache.json")
        sink = PredictionWriter(run_dir / "items.jsonl")
        engine = Engine(workers)
        planned = assign_ids(questions, seed)

        def case(entry: tuple[str, Question]) -> dict[str, Any]:
            item_id, q = entry
            database = db_path(q.db_id, databases_dir)
            result = None
            with timed_calls(client, run_id, "judge_set", q.question_id) as call:
                result = run_sql_agent(
                    q.question, q.evidence, database, max_fixes=max_fixes, skip_respond=False
                )
            final = None
            ex_correct, error, failure = False, call.error, "exception" if call.error else None
            if result is not None and call.error is None:
                final = next(
                    (a for a in reversed(result.attempts) if a.sql == result.final_sql), None
                )
                score = score_prediction(
                    q.question_id, database, result.final_sql, q.sql, gold_cache, timeout_s
                )
                ex_correct, error = score.correct, score.error
            usages = call.counting.usages
            gold = gold_preview(database, q.sql, timeout_s)
            return {
                "item_id": item_id,
                "question_id": q.question_id,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "question": q.question,
                "evidence": q.evidence,
                "answer": result.response if result is not None else "",
                "final_sql": result.final_sql if result is not None else "",
                "result_preview": attempt_preview(final),
                "gold_preview": gold,  # gitignored only
                "ex_correct": ex_correct,  # gitignored only
                "error": error,
                "failure_type": failure,
                "n_llm_calls": len(usages),
                "cost_usd": sum(u.cost_usd for u in usages),
                "latency_ms": call.latency_ms,
            }

        try:
            engine.run_pass("items", planned, case, sink)
        except KeyboardInterrupt:
            print("Interrupted. Saving partial results.")
            engine.stop("interrupted")

        items = sorted(read_jsonl(run_dir / "items.jsonl"), key=lambda r: r["item_id"])
        write_jsonl(run_dir / "items.jsonl", items)
        write_jsonl(run_dir / "items_public.jsonl", [public_record(i) for i in items])
        summary: dict[str, Any] = {
            "run_id": run_id,
            "kind": "judge_set",
            "split": SPLIT,
            "n_planned": len(planned),
            "n_items": len(items),
            "n_failed": sum(1 for i in items if i["failure_type"] == "exception"),
            "n_ex_correct": sum(1 for i in items if i["ex_correct"]),
            "n_empty_answer": sum(1 for i in items if not i["answer"].strip()),
            "n_gold_error": sum(1 for i in items if i["gold_preview"].get("error")),
            "totals": {
                "cost_usd": sum(i["cost_usd"] for i in items),
                "n_llm_calls": sum(i["n_llm_calls"] for i in items),
            },
            "seed": seed,
            "sampling": sampling_meta(questions, len(questions), None, seed),
            "aborted": engine.reason,
            "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
            "workers": workers,
            "max_fixes": max_fixes,
            "llm_config_file": llm_config_name,
            "llm_config": config.model_dump(),
            "git": git_info(),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            **(extra_meta or {}),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        box["summary"] = summary
    out: dict[str, Any] = box["summary"]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument("--run-id", required=True, help="new folder under evals/results/judge/")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="dev questions to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-usd", type=float, help="hard cap on spend (anthropic: required)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-fixes", type=int, default=DEFAULT_MAX_FIXES)
    parser.add_argument(
        "--launched-by", help=f"free text recorded in run.json (default: env {LAUNCHED_BY_ENV})"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv if argv is None else argv)
    args = parse_args(argv)
    config = load_llm_config(args.llm_config)
    if config.provider == "anthropic" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for the anthropic provider")
    questions = select_questions(load_questions(), load_split(SPLIT), args.n, None, args.seed)
    run_dir = RESULTS_ROOT / args.run_id
    print(f"Run {args.run_id}: {len(questions)} {SPLIT} questions (seed {args.seed})")
    try:
        with sigterm_as_interrupt():
            summary = build_set(
                config=config,
                questions=questions,
                run_dir=run_dir,
                run_id=args.run_id,
                budget_usd=args.budget_usd,
                workers=args.workers,
                seed=args.seed,
                llm_config_name=str(args.llm_config),
                max_fixes=args.max_fixes,
                argv=raw_argv,
                launched_by=args.launched_by,
            )
    except FileExistsError as e:
        raise SystemExit(str(e)) from e
    print(
        f"{summary['n_items']}/{summary['n_planned']} items, {summary['n_ex_correct']} EX-correct, "
        f"{summary['n_failed']} failed, cost ${summary['totals']['cost_usd']:.4f}"
    )
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main()
