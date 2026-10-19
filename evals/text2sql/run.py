"""Text-to-SQL runner: baseline and agent on BIRD Mini-Dev with token and cost logging.

Example:
    python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --system both --split dev --limit 20 --budget-usd 1.00 --run-id smoke

Systems run in separate passes (all baseline questions, then all agent questions) and
questions are ordered by (db_id, question_id), so a local server can reuse the cached
schema prefix of the prompts. ``--workers > 1`` is meant for the API provider.
"""

import argparse
import json
import statistics
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.text2sql.agent_system import predict_agent
from evals.text2sql.baseline import CountingClient, predict_baseline
from evals.text2sql.data import (
    DIFFICULTIES,
    REPO_ROOT,
    Question,
    db_path,
    load_questions,
    load_split,
)
from evals.text2sql.sampling import stratified_sample, stratum_label
from evals.text2sql.scorer import GoldCache, score_prediction
from src.llm import BudgetExceeded, LLMClient, LLMConfig, UsageTracker, build_client, call_context

RESULTS_ROOT = REPO_ROOT / "evals" / "results" / "text2sql"
FAILURE_STOP_REASONS = ("refusal", "max_tokens")
MAX_CONSECUTIVE_EXCEPTIONS = 3

Predictor = Callable[..., tuple[str, int]]
PREDICTORS: dict[str, Predictor] = {"baseline": predict_baseline, "agent": predict_agent}
ClientFactory = Callable[[LLMConfig, UsageTracker], LLMClient]


def load_llm_config(path: Path) -> LLMConfig:
    """Load an LLM config from YAML."""
    with open(path, encoding="utf-8") as f:
        return LLMConfig.model_validate(yaml.safe_load(f))


def git_info() -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def execution_order(questions: list[Question]) -> list[Question]:
    """Order used inside each system pass: by database, then question id."""
    return sorted(questions, key=lambda q: (q.db_id, q.question_id))


def select_questions(
    questions: list[Question],
    split_ids: list[int],
    limit: int | None = None,
    question_ids: list[int] | None = None,
    seed: int = 0,
) -> list[Question]:
    """Questions of a split.

    ``question_ids`` restricts to those ids and bypasses sampling. Otherwise ``limit``
    draws a stratified sample (by database and difficulty) with ``seed``.
    """
    by_id = {q.question_id: q for q in questions}
    wanted = sorted(split_ids)
    if question_ids is not None:
        outside = [i for i in question_ids if i not in set(split_ids)]
        if outside:
            raise ValueError(f"question ids not in the selected split: {outside}")
        return [by_id[i] for i in sorted(set(question_ids))]
    if limit is not None:
        wanted = stratified_sample([by_id[i] for i in wanted], limit, seed)
    return [by_id[i] for i in wanted]


def sampling_meta(
    selected: list[Question], limit: int | None, question_ids: list[int] | None, seed: int
) -> dict[str, Any]:
    """What summary.json records about how the questions were chosen."""
    strata = Counter(stratum_label((q.db_id, q.difficulty)) for q in selected)
    stratified = limit is not None and question_ids is None
    return {
        "method": "stratified" if stratified else "all" if question_ids is None else "question_ids",
        "seed": seed if stratified else None,
        "n_sampled": len(selected),
        "per_stratum": dict(sorted(strata.items())),
    }


def classify(correct: bool, error: str | None, stop_reasons: list[str]) -> str | None:
    """Failure type of a prediction, or None when correct."""
    for reason in FAILURE_STOP_REASONS:
        if reason in stop_reasons:
            return reason
    if correct:
        return None
    if error == "no sql":
        return "no_sql"
    if error == "timeout":
        return "timeout"
    if error is not None:
        return "sql_error"
    return "wrong_result"


def _is_done(done: dict[tuple[int, str], dict[str, Any]], q: Question, system: str) -> bool:
    """A pair counts as done unless its last attempt ended in an exception (retried)."""
    record = done.get((q.question_id, system))
    return record is not None and record["failure_type"] != "exception"


def read_predictions(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """Latest record per (question_id, system)."""
    records: dict[tuple[int, str], dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[(record["question_id"], record["system"])] = record
    return records


class PredictionWriter:
    """Appends prediction records to predictions.jsonl from several threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_question(
    question: Question,
    system: str,
    client: LLMClient,
    gold_cache: GoldCache,
    *,
    run_id: str,
    databases_dir: Path | None,
    timeout_s: float,
) -> dict[str, Any]:
    """Predict and score one (question, system) pair. Raises BudgetExceeded on the cap."""
    database = db_path(question.db_id, databases_dir)
    recorder = CountingClient(client)
    pred_sql = ""
    n_calls = 0
    error: str | None = None
    exception = False
    start = time.perf_counter()
    with call_context(run_id=run_id, system=system, question_id=question.question_id):
        try:
            pred_sql, n_calls = PREDICTORS[system](question, recorder, database=database)
        except BudgetExceeded:
            raise
        except Exception as e:  # API or agent failure: record it and keep going
            exception = True
            error = f"{type(e).__name__}: {e}"
    latency_ms = (time.perf_counter() - start) * 1000

    stop_reasons = [u.stop_reason for u in recorder.usages if u.stop_reason]
    if exception:
        correct, failure_type = False, "exception"
    else:
        score = score_prediction(
            question.question_id, database, pred_sql, question.sql, gold_cache, timeout_s
        )
        error = score.error
        failure_type = classify(score.correct, score.error, stop_reasons)
        correct = score.correct and failure_type is None

    usages = recorder.usages
    return {
        "run_id": run_id,
        "question_id": question.question_id,
        "db_id": question.db_id,
        "difficulty": question.difficulty,
        "system": system,
        "pred_sql": pred_sql,
        "gold_sql": question.sql,
        "correct": correct,
        "error": error,
        "failure_type": failure_type,
        "stop_reasons": stop_reasons,
        "n_llm_calls": n_calls or len(usages),
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "cache_read_tokens": sum(u.cache_read_tokens for u in usages),
        "cache_write_tokens": sum(u.cache_write_tokens for u in usages),
        "cost_usd": sum(u.cost_usd for u in usages),
        "latency_ms": latency_ms,
        "llm_latency_ms": sum(u.latency_ms for u in usages),
    }


def summarize_system(
    records: list[dict[str, Any]], n_planned: int, aborted: str | None
) -> dict[str, Any]:
    """Aggregate prediction records of one system."""
    n = len(records)

    def mean(key: str) -> float | None:
        return sum(r[key] for r in records) / n if n else None

    def ex(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(1 for r in rows if r["correct"])
        return {"n": len(rows), "correct": correct, "ex": correct / len(rows) if rows else None}

    latencies = [r["latency_ms"] for r in records]
    input_total = sum(r["input_tokens"] for r in records)
    cache_read = sum(r["cache_read_tokens"] for r in records)
    cache_write = sum(r["cache_write_tokens"] for r in records)
    prompt_total = input_total + cache_read + cache_write
    return {
        "n": n,
        "n_planned": n_planned,
        "aborted": aborted,
        **ex(records),
        "by_difficulty": {
            d: ex([r for r in records if r["difficulty"] == d]) for d in DIFFICULTIES
        },
        "tokens": {
            "input_total": input_total,
            "output_total": sum(r["output_tokens"] for r in records),
            "cache_read_total": cache_read,
            "cache_write_total": cache_write,
            # cache_read / (input + cache_read + cache_write) over all calls of the system
            "cache_hit_ratio": cache_read / prompt_total if prompt_total else None,
            "input_mean_per_question": mean("input_tokens"),
            "output_mean_per_question": mean("output_tokens"),
        },
        "cost_usd": {
            "total": sum(r["cost_usd"] for r in records),
            "mean_per_question": mean("cost_usd"),
        },
        "llm_calls_mean_per_question": mean("n_llm_calls"),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "failures": dict(Counter(r["failure_type"] for r in records if r["failure_type"])),
    }


def format_table(summary: dict[str, Any]) -> str:
    """Short text table of the per-system results."""
    header = (
        f"{'system':<10}{'n':>5}{'EX':>8}{'simple':>9}{'moderate':>10}{'challeng.':>11}"
        f"{'cost $':>10}{'in tok/q':>10}{'out tok/q':>11}{'lat s/q':>9}"
    )
    lines = [header, "-" * len(header)]

    def pct(block: dict[str, Any]) -> str:
        return f"{block['ex']:.1%}" if block["ex"] is not None else "-"

    for name, s in summary["systems"].items():
        tokens = s["tokens"]
        latency = s["latency_ms"]["mean"]
        lines.append(
            f"{name:<10}{s['n']:>5}{pct(s):>8}"
            f"{pct(s['by_difficulty']['simple']):>9}{pct(s['by_difficulty']['moderate']):>10}"
            f"{pct(s['by_difficulty']['challenging']):>11}"
            f"{s['cost_usd']['total']:>10.4f}"
            f"{(tokens['input_mean_per_question'] or 0):>10.0f}"
            f"{(tokens['output_mean_per_question'] or 0):>11.0f}"
            f"{(latency or 0) / 1000:>9.1f}"
        )
        if s["aborted"]:
            lines.append(f"  {name}: aborted ({s['aborted']}), {s['n']}/{s['n_planned']} done")
        if s["failures"]:
            lines.append(f"  {name} failures: {s['failures']}")
    return "\n".join(lines)


def run_experiment(
    *,
    config: LLMConfig,
    systems: list[str],
    questions: list[Question],
    run_dir: Path,
    run_id: str,
    split: str,
    budget_usd: float | None = None,
    workers: int = 1,
    resume: bool = False,
    llm_config_name: str | None = None,
    client_factory: ClientFactory = build_client,
    databases_dir: Path | None = None,
    timeout_s: float = 30,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the systems over the questions and write predictions, LLM calls and summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.exists() and predictions_path.stat().st_size > 0 and not resume:
        raise FileExistsError(f"{predictions_path} exists; use --resume or another --run-id")

    started_at = datetime.now(UTC).isoformat()
    tracker = UsageTracker(budget_usd=budget_usd, sink_path=run_dir / "llm_calls.jsonl")
    client = client_factory(config, tracker)
    writer = PredictionWriter(predictions_path)
    gold_cache = GoldCache(run_dir / "gold_cache.json")
    ordered = execution_order(questions)
    abort_reason: dict[str, str] = {}
    abort = threading.Event()
    n_planned = len(ordered)

    def run_one(question: Question, system: str) -> bool:
        """Returns True when the pair ended in an exception."""
        if abort.is_set():
            return False
        try:
            record = evaluate_question(
                question,
                system,
                client,
                gold_cache,
                run_id=run_id,
                databases_dir=databases_dir,
                timeout_s=timeout_s,
            )
        except BudgetExceeded as e:
            abort_reason.setdefault(system, "budget")
            abort.set()
            print(f"Budget reached: {e}. Stopping and saving partial results.")
            return False
        writer.write(record)
        return bool(record["failure_type"] == "exception")

    aborted_systems: dict[str, str] = {}
    try:
        for system in systems:  # one pass per system, never interleaved
            done = read_predictions(predictions_path)
            pending = [q for q in ordered if not (resume and _is_done(done, q, system))]
            print(f"[{system}] {len(pending)} to run, {n_planned - len(pending)} already done")
            consecutive_exceptions = 0
            if workers <= 1:
                for q in pending:
                    if abort.is_set():
                        break
                    consecutive_exceptions = consecutive_exceptions + 1 if run_one(q, system) else 0
                    if consecutive_exceptions >= MAX_CONSECUTIVE_EXCEPTIONS:
                        abort_reason.setdefault(system, "errors")
                        abort.set()
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(lambda q, s=system: run_one(q, s), pending))
            if system in abort_reason:
                aborted_systems[system] = abort_reason[system]
            if abort.is_set():
                break
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results.")
        for system in systems:
            aborted_systems.setdefault(system, "interrupted")

    final = read_predictions(predictions_path)
    selected_ids = {q.question_id for q in questions}
    system_summaries: dict[str, Any] = {}
    for system in systems:
        records = sorted(
            (r for (qid, s), r in final.items() if s == system and qid in selected_ids),
            key=lambda r: (r["db_id"], r["question_id"]),
        )
        aborted = aborted_systems.get(system)
        if aborted is None and len(records) < n_planned and abort.is_set():
            aborted = next(iter(abort_reason.values()), "interrupted")
        system_summaries[system] = summarize_system(records, n_planned, aborted)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "n_questions": n_planned,
        "systems": system_summaries,
        "aborted": next(iter(aborted_systems.values()), None),
        "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
        "resumed": resume,
        "workers": workers,
        "llm_config_file": llm_config_name,
        "llm_config": config.model_dump(),
        "git": git_info(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        **(extra_meta or {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument("--system", choices=["baseline", "agent", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--limit",
        type=int,
        help="stratified sample of N questions of the split (by database and difficulty)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the --limit sampling")
    parser.add_argument("--question-ids", type=int, nargs="+", help="restrict to these ids")
    parser.add_argument(
        "--budget-usd",
        type=float,
        help="hard cap on spend for this invocation (required for the anthropic provider)",
    )
    parser.add_argument("--run-id", help="output folder name under evals/results/text2sql/")
    parser.add_argument("--workers", type=int, default=1, help="threads; intended for the API")
    parser.add_argument("--resume", action="store_true", help="skip pairs already predicted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_llm_config(args.llm_config)
    if config.provider == "anthropic" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for the anthropic provider")
    systems = ["baseline", "agent"] if args.system == "both" else [args.system]
    run_id = args.run_id or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{args.llm_config.stem}"
    questions = select_questions(
        load_questions(), load_split(args.split), args.limit, args.question_ids, args.seed
    )
    print(f"Run {run_id}: {len(questions)} questions, systems {systems}, split {args.split}")
    summary = run_experiment(
        config=config,
        systems=systems,
        questions=questions,
        run_dir=RESULTS_ROOT / run_id,
        run_id=run_id,
        split=args.split,
        budget_usd=args.budget_usd,
        workers=args.workers,
        resume=args.resume,
        llm_config_name=str(args.llm_config),
        extra_meta={
            "limit": args.limit,
            "question_ids": args.question_ids,
            "sampling": sampling_meta(questions, args.limit, args.question_ids, args.seed),
        },
    )
    print()
    print(format_table(summary))
    print(f"\nResults in {RESULTS_ROOT / run_id}")


if __name__ == "__main__":
    main()
