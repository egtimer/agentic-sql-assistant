"""Machinery shared by the two memory evaluations (long-term and follow-up).

Both runners reuse the pieces of the text-to-SQL runner: LLM client, usage tracker and budget,
execution-accuracy scorer, bootstrap statistics, sampling, provenance (``run.json``) and the
result layout (``predictions.jsonl`` without gold SQL, ``llm_calls.jsonl``, ``summary.json``).
They differ from it in one way: a run directory is always new, there is no ``--resume``
(a half-finished run holds memories and threads that cannot be rebuilt cheaply).
"""

import contextlib
import json
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from evals.provenance import finish_run, start_run, status_from_abort
from evals.text2sql.baseline import CountingClient
from evals.text2sql.data import Question
from evals.text2sql.run import (
    MAX_CONSECUTIVE_EXCEPTIONS,
    PredictionWriter,
    classify,
)
from evals.text2sql.scorer import GoldCache, score_prediction
from evals.text2sql.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, _percentile_interval
from src.llm import (
    BudgetExceeded,
    CallUsage,
    LLMClient,
    LLMConfig,
    UsageTracker,
    build_client,
    call_context,
    reset_client,
    set_client,
)
from src.observability import tracing
from src.observability.tracing import NoopTracer
from src.sql_agent.state import SqlAgentResult

T = TypeVar("T")
ClientFactory = Callable[[LLMConfig, UsageTracker], LLMClient]
RESULTS_ROOT_NAME = ("evals", "results", "memory")
TEST_SPLIT_WARNING = (
    "WARNING: the test split is for the single final run, once, with the owner's OK. "
    "Do not tune prompts on it."
)


def check_new_run_dir(run_dir: Path) -> None:
    """Memory runs are never resumed: an existing, non-empty directory is refused."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} exists; memory runs have no --resume: use a new --run-id")


@contextlib.contextmanager
def provenance(
    run_dir: Path,
    *,
    run_id: str,
    argv: list[str] | None,
    llm_config_name: str | None,
    config: LLMConfig,
    budget_usd: float | None,
    launched_by: str | None,
    summary_box: dict[str, Any],
) -> Iterator[None]:
    """Write ``run.json`` before the first case and finish it with a status afterwards.

    The caller stores the summary in ``summary_box['summary']`` before leaving the block;
    its ``aborted`` field decides the status. Tracing is switched off for the whole run.
    """
    check_new_run_dir(run_dir)
    run_file = start_run(
        run_dir,
        run_id=run_id,
        argv=argv,
        llm_config_path=llm_config_name,
        config=config,
        budget_usd=budget_usd,
        launched_by=launched_by,
    )
    try:
        with tracing.use_tracer(NoopTracer()):
            yield
    except BaseException as e:
        finish_run(
            run_file,
            "interrupted" if isinstance(e, KeyboardInterrupt) else f"failed:{type(e).__name__}",
        )
        raise
    finish_run(run_file, status_from_abort(summary_box["summary"]["aborted"]))


class Engine:
    """Runs the cases of a pass (sequentially or in a thread pool) with the runner's abort
    rules: the budget stops everything, three exceptions in a row stop the run."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.abort = threading.Event()
        self.reason: str | None = None
        self._lock = threading.Lock()
        self._consecutive = 0

    def _stop(self, reason: str) -> None:
        with self._lock:
            self.reason = self.reason or reason
        self.abort.set()

    def _one(
        self, item: T, fn: Callable[[T], dict[str, Any]], sink: PredictionWriter | None
    ) -> None:
        if self.abort.is_set():
            return
        try:
            record = fn(item)
        except BudgetExceeded as e:
            print(f"Budget reached: {e}. Stopping and saving partial results.")
            self._stop("budget")
            return
        if sink is not None:
            sink.write(record)
        with self._lock:
            self._consecutive = (
                self._consecutive + 1 if record["failure_type"] == "exception" else 0
            )
            too_many = self._consecutive >= MAX_CONSECUTIVE_EXCEPTIONS
        if too_many:
            self._stop("errors")

    def stop(self, reason: str) -> None:
        self._stop(reason)

    def run_pass(
        self,
        label: str,
        items: Sequence[T],
        fn: Callable[[T], dict[str, Any]],
        sink: PredictionWriter | None,
    ) -> None:
        print(f"[{label}] {len(items)} to run")
        if self.workers <= 1:
            for item in items:
                self._one(item, fn, sink)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(lambda item: self._one(item, fn, sink), items))


@dataclass
class Call:
    """What :func:`timed_calls` measured around a group of agent calls."""

    counting: CountingClient
    results: list[SqlAgentResult] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0


@contextlib.contextmanager
def timed_calls(client: LLMClient, run_id: str, system: str, question_id: int) -> Iterator[Call]:
    """Count and time the LLM calls made inside the block; failures other than the budget
    are recorded in ``Call.error`` (the run goes on), the budget propagates."""
    call = Call(counting=CountingClient(client))
    token = set_client(call.counting)
    start = time.perf_counter()
    try:
        with call_context(run_id=run_id, system=system, question_id=question_id):
            try:
                yield call
            except BudgetExceeded:
                raise
            except Exception as e:  # API or agent failure
                call.error = f"{type(e).__name__}: {e}"
    finally:
        call.latency_ms = (time.perf_counter() - start) * 1000
        reset_client(token)


def agent_details(result: SqlAgentResult) -> dict[str, Any]:
    """Attempts, verdicts and fixes of one agent run, as the text-to-SQL runner records them."""
    return {
        "attempts": [
            {
                "sql": a.sql,
                "error": a.error,
                "row_count": a.row_count_text if a.ok else None,
                "n_columns": len(a.columns),
            }
            for a in result.attempts
        ],
        "verdicts": [
            {"kind": v.kind, "unparsed": v.unparsed, "note": v.note, "text": v.text}
            for v in result.verdicts
        ],
        "fixes_used": result.fixes_used,
    }


def scored_record(
    *,
    run_id: str,
    question: Question,
    condition: str,
    call: Call,
    result: SqlAgentResult | None,
    database: Path,
    gold_cache: GoldCache,
    timeout_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score ``result`` (the scored turn) against the gold SQL of ``question`` and build the
    record of ``predictions.jsonl``. Cost, tokens and calls cover every call in ``call``;
    the gold SQL is not written out."""
    usages: list[CallUsage] = call.counting.usages
    stop_reasons = [u.stop_reason for u in usages if u.stop_reason]
    pred_sql = result.final_sql if result is not None else ""
    error = call.error
    details: dict[str, Any] = {}
    if call.error is not None or result is None:
        correct, failure_type = False, "exception"
    else:
        score = score_prediction(
            question.question_id, database, pred_sql, question.sql, gold_cache, timeout_s
        )
        error = score.error
        failure_type = classify(score.correct, score.error, stop_reasons)
        correct = score.correct and failure_type is None
        first_sql = result.attempts[0].sql if result.attempts else ""
        if first_sql == pred_sql:
            first_correct = score.correct
        else:
            first_correct = score_prediction(
                question.question_id, database, first_sql, question.sql, gold_cache, timeout_s
            ).correct
        details = {
            **agent_details(result),
            "first_sql": first_sql,
            "first_sql_correct": first_correct,
        }
    return {
        "run_id": run_id,
        "question_id": question.question_id,
        "db_id": question.db_id,
        "difficulty": question.difficulty,
        "system": condition,
        "pred_sql": pred_sql,
        "correct": correct,
        "error": error,
        "failure_type": failure_type,
        "stop_reasons": stop_reasons,
        "n_llm_calls": len(usages),
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "cache_read_tokens": sum(u.cache_read_tokens for u in usages),
        "cache_write_tokens": sum(u.cache_write_tokens for u in usages),
        "cost_usd": sum(u.cost_usd for u in usages),
        "latency_ms": call.latency_ms,
        "llm_latency_ms": sum(u.latency_ms for u in usages),
        **details,
        **(extra or {}),
    }


def new_tracker(budget_usd: float | None, run_dir: Path) -> UsageTracker:
    return UsageTracker(budget_usd=budget_usd, sink_path=run_dir / "llm_calls.jsonl")


def make_client(factory: ClientFactory | None, config: LLMConfig, tracker: UsageTracker) -> Any:
    return (factory or build_client)(config, tracker)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def gap_closed(
    no_memory: dict[int, bool],
    memory: dict[int, bool],
    oracle: dict[int, bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    """Share of the oracle's gain over ``no_memory`` that ``memory`` recovers:
    ``(EX(memory) - EX(no_memory)) / (EX(oracle) - EX(no_memory))``.

    Questions are resampled together (paired across the three conditions). A resample whose
    denominator is 0 has no ratio: it is left out of the interval and counted in
    ``frac_undefined``; the point estimate is None when the full-sample denominator is 0.
    None when no question was run under all three conditions.
    """
    ids = sorted(no_memory.keys() & memory.keys() & oracle.keys())
    n = len(ids)
    if n == 0:
        return None
    a = [int(no_memory[i]) for i in ids]
    b = [int(memory[i]) for i in ids]
    c = [int(oracle[i]) for i in ids]

    def ratio(idx: Sequence[int]) -> float | None:
        den = sum(c[i] - a[i] for i in idx)
        return None if den == 0 else sum(b[i] - a[i] for i in idx) / den

    rng = random.Random(seed)
    everyone = range(n)
    ratios: list[float] = []
    undefined = 0
    for _ in range(resamples):
        value = ratio(rng.choices(everyone, k=n))
        if value is None:
            undefined += 1
        else:
            ratios.append(value)
    interval = _percentile_interval(sorted(ratios), confidence) if ratios else None
    return {
        "n": n,
        "gap_closed": ratio(list(everyone)),
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "frac_undefined": undefined / resamples if resamples else None,
        "resamples": resamples,
        "method": "paired percentile bootstrap, questions resampled together; "
        "resamples with a zero denominator are excluded",
    }


def fmt_pct(value: float | None) -> str:
    return "undefined" if value is None else f"{value * 100:.1f}%"
