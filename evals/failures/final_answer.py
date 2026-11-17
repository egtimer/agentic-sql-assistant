"""Final-answer pass over a committed BIRD run: agent v2's answer, written from the SQL the run
already chose, graded by the answer judge.

The noise repetitions ran agent v2 with the respond step on, but the runner of the time dropped
the answer (fixed in 7c93f44). This pass rebuilds it without re-running the agent: for every
``agent_v2`` prediction of the run it re-executes ``pred_sql`` read-only (the same
``run_query`` the agent uses, so the result text is the one the respond step saw), makes the
respond call with the agent's own prompt (``RESPOND_SYSTEM``, ``respond_user``, tag
``respond``), and grades the answer with the judge of ``evals/judge`` against the gold result.
Two LLM calls per question. The SQL is not touched, so EX stays the run's.

Output in ``evals/results/failures/<run-id>/`` (gitignored):

- ``answers.jsonl``         full records, with the gold preview and the judge's rationale
- ``answers_public.jsonl``  question id, db, difficulty, EX, answer, judge label: may be committed
- ``run.json``, ``llm_calls.jsonl``, ``summary.json``

Spends money with the Anthropic config; the owner runs it:

    uv run python -m evals.failures.final_answer \\
        --report 2026-11-16-noise-bird-haiku-test-350/run-r01 \\
        --llm-config configs/llm/claude-haiku-5-5.yaml --run-id final-answer-haiku-r01 \\
        --budget-usd 0.15 --launched-by owner
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.judge.build_set import gold_preview, write_jsonl
from evals.judge.judge import JUDGE_SYSTEM, judge_answer
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
from evals.text2sql.data import REPO_ROOT, Question, db_path, load_questions
from evals.text2sql.run import PredictionWriter, load_llm_config
from src.llm import CallUsage, LLMConfig, ProviderError
from src.sql_agent.executor import run_query
from src.sql_agent.prompts import RESPOND_SYSTEM, respond_user

REPORTS = REPO_ROOT / "evals" / "reports"
RESULTS_ROOT = REPO_ROOT / "evals" / "results" / "failures"
SYSTEM = "agent_v2"
NO_REFERENCE = "no_reference"
ERROR = "error"
PUBLIC_FIELDS = ("question_id", "db_id", "difficulty", "ex_correct", "answer", "judge_label")


def load_predictions(report_dir: Path, system: str = SYSTEM) -> list[dict[str, Any]]:
    """The ``system`` rows of a report's ``predictions.jsonl``, sorted by question id."""
    rows = [r for r in read_jsonl(report_dir / "predictions.jsonl") if r["system"] == system]
    if not rows:
        raise SystemExit(f"no {system} predictions in {report_dir}")
    return sorted(rows, key=lambda r: int(r["question_id"]))


def _usage_totals(usages: list[CallUsage]) -> dict[str, Any]:
    return {
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "cost_usd": sum(u.cost_usd for u in usages),
    }


def _stop_on_provider_error(engine: Engine, error: str | None) -> None:
    """The provider could not answer (e.g. no credit): stop instead of judging empty answers."""
    if error is not None and error.startswith(ProviderError.__name__):
        print(f"Provider error: {error}. Stopping and saving partial results.")
        engine.stop("provider_error")


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in PUBLIC_FIELDS}


def run_final_answer(
    *,
    config: LLMConfig,
    predictions: list[dict[str, Any]],
    questions: dict[int, Question],
    report: str,
    run_dir: Path,
    run_id: str,
    budget_usd: float | None = None,
    workers: int = 1,
    timeout_s: float = 30,
    databases_dir: Path | None = None,
    llm_config_name: str | None = None,
    client_factory: ClientFactory | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
) -> dict[str, Any]:
    """Answer and judge every prediction (the run directory must be new); write the outputs."""
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
        sink = PredictionWriter(run_dir / "answers.jsonl")
        engine = Engine(workers)

        def case(pred: dict[str, Any]) -> dict[str, Any]:
            q = questions[int(pred["question_id"])]
            database = db_path(q.db_id, databases_dir)
            attempt = run_query(database, pred["pred_sql"], timeout_s)
            record: dict[str, Any] = {
                "question_id": q.question_id,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "ex_correct": bool(pred["correct"]),
                "pred_sql": pred["pred_sql"],
                "result_error": attempt.error,
                "answer": "",
                "judge_label": ERROR,
                "judge_rationale": "",
                "gold_preview": None,  # gitignored only
                "failure_type": None,
                "error": None,
            }
            answer = None
            with timed_calls(client, run_id, "respond", q.question_id) as respond:
                completion = respond.counting.complete(
                    RESPOND_SYSTEM, respond_user(q.question, attempt), tag="respond"
                )
                answer = completion.text.strip()
            record["respond"] = _usage_totals(respond.counting.usages)
            if respond.error is not None or answer is None:
                record.update(failure_type="exception", error=respond.error)
                record["judge"] = _usage_totals([])
                _stop_on_provider_error(engine, respond.error)
                return record
            record["answer"] = answer
            gold = gold_preview(database, q.sql, timeout_s)
            record["gold_preview"] = gold
            if gold.get("error"):
                record.update(judge_label=NO_REFERENCE, judge_rationale=f"gold: {gold['error']}")
                record["judge"] = _usage_totals([])
                return record
            verdict = None
            with timed_calls(client, run_id, "judge", q.question_id) as judging:
                verdict = judge_answer(q.question, q.evidence, answer, gold, judging.counting)
            record["judge"] = _usage_totals(judging.counting.usages)
            if judging.error is not None or verdict is None:
                record.update(failure_type="exception", error=judging.error)
                _stop_on_provider_error(engine, judging.error)
            else:
                record.update(judge_label=verdict.label, judge_rationale=verdict.rationale)
            return record

        try:
            engine.run_pass("final answer", predictions, case, sink)
        except KeyboardInterrupt:
            print("Interrupted. Saving partial results.")
            engine.stop("interrupted")

        records = sorted(read_jsonl(run_dir / "answers.jsonl"), key=lambda r: r["question_id"])
        write_jsonl(run_dir / "answers.jsonl", records)
        write_jsonl(run_dir / "answers_public.jsonl", [public_record(r) for r in records])
        labels = Counter(r["judge_label"] for r in records)
        cost = {part: sum(r[part]["cost_usd"] for r in records) for part in ("respond", "judge")}
        summary: dict[str, Any] = {
            "run_id": run_id,
            "kind": "final_answer",
            "report": report,
            "system": SYSTEM,
            "n_planned": len(predictions),
            "n_done": len(records),
            "judge_labels": dict(labels),
            "ex_vs_judge": dict(
                Counter(
                    f"ex_{'ok' if r['ex_correct'] else 'wrong'}/judge_{r['judge_label']}"
                    for r in records
                )
            ),
            "n_empty_answer": sum(1 for r in records if not r["answer"].strip()),
            "totals": {**cost, "cost_usd": cost["respond"] + cost["judge"]},
            "judge_prompt_chars": len(JUDGE_SYSTEM),
            "aborted": engine.reason,
            "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
            "workers": workers,
            "llm_config_file": llm_config_name,
            "llm_config": config.model_dump(),
            "git": git_info(),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        box["summary"] = summary
    out: dict[str, Any] = box["summary"]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--report", required=True, help="run directory under evals/reports/")
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument("--run-id", required=True, help="new folder under evals/results/failures/")
    parser.add_argument("--budget-usd", type=float, help="hard cap on spend (anthropic: required)")
    parser.add_argument("--workers", type=int, default=1)
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
    predictions = load_predictions(REPORTS / args.report)
    questions = {q.question_id: q for q in load_questions()}
    run_dir = RESULTS_ROOT / args.run_id
    print(f"Run {args.run_id}: final answer for {len(predictions)} predictions of {args.report}")
    try:
        with sigterm_as_interrupt():
            summary = run_final_answer(
                config=config,
                predictions=predictions,
                questions=questions,
                report=args.report,
                run_dir=run_dir,
                run_id=args.run_id,
                budget_usd=args.budget_usd,
                workers=args.workers,
                llm_config_name=str(args.llm_config),
                argv=raw_argv,
                launched_by=args.launched_by,
            )
    except FileExistsError as e:
        raise SystemExit(str(e)) from e
    print(f"labels {summary['judge_labels']}, cost ${summary['totals']['cost_usd']:.4f}")
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main()
