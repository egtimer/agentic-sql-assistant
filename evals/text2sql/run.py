"""Text-to-SQL runner: baseline and agent on BIRD Mini-Dev with token and cost logging.

Example:
    python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --system baseline,agent_v2 --split dev --limit 20 --budget-usd 1.00 --run-id smoke

Systems (comma-separated, run in the order given): baseline, baseline_ctx, agent_v1,
agent_v2_nocheck, agent_v2. ``agent`` is an alias of agent_v1 and ``both`` of
``baseline,agent_v1``.

Systems run in separate passes (all questions of one system, then the next) and
questions are ordered by (db_id, question_id), so a local server can reuse the cached
schema prefix of the prompts. ``--workers > 1`` is meant for the API provider.
"""

import argparse
import contextlib
import json
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.provenance import (
    LAUNCHED_BY_ENV,
    finish_run,
    git_info,
    sigterm_as_interrupt,
    start_run,
    status_from_abort,
    update_run,
)
from evals.text2sql.agent_system import (
    predict_agent,
    predict_agent_v2,
    predict_agent_v2_nocheck,
)
from evals.text2sql.baseline import CountingClient, predict_baseline, predict_baseline_ctx
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
from evals.text2sql.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_ci,
    bootstrap_meta,
    paired_bootstrap,
)
from src.config import settings
from src.llm import BudgetExceeded, LLMClient, LLMConfig, UsageTracker, build_client, call_context
from src.observability import tracing
from src.observability.tracing import NoopTracer


class TracingNotConfiguredError(RuntimeError):
    """``trace=True`` without Langfuse keys."""


RESULTS_ROOT = REPO_ROOT / "evals" / "results" / "text2sql"
FAILURE_STOP_REASONS = ("refusal", "max_tokens")
MAX_CONSECUTIVE_EXCEPTIONS = 3

# A predictor returns ``(sql, n_llm_calls)`` or ``(sql, n_llm_calls, details)``.
Predictor = Callable[..., tuple[Any, ...]]
PREDICTORS: dict[str, Predictor] = {
    "baseline": predict_baseline,
    "baseline_ctx": predict_baseline_ctx,
    "agent": predict_agent,  # alias of agent_v1 (name used by older summaries)
    "agent_v1": predict_agent,
    "agent_v2_nocheck": predict_agent_v2_nocheck,
    "agent_v2": predict_agent_v2,
}
V1_SYSTEMS = frozenset({"agent", "agent_v1"})
V2_SYSTEMS = frozenset({"agent_v2", "agent_v2_nocheck"})
SYSTEM_ALIASES = {"agent": "agent_v1", "both": "baseline,agent_v1"}
DEFAULT_MAX_FIXES = 2


def parse_systems(value: str) -> list[str]:
    """``--system`` value -> ordered, de-duplicated list of system names."""
    names: list[str] = []
    for raw in value.split(","):
        name = raw.strip()
        for part in SYSTEM_ALIASES.get(name, name).split(","):
            if part not in PREDICTORS or part == "agent":
                raise ValueError(
                    f"unknown system {name!r}; choose from {sorted(set(PREDICTORS) - {'agent'})}"
                )
            if part not in names:
                names.append(part)
    return names


def system_options(system: str, *, skip_respond: bool, max_fixes: int) -> dict[str, Any]:
    """Keyword options passed to the predictor of ``system``."""
    if system in V1_SYSTEMS:
        return {"skip_respond": True} if skip_respond else {}
    if system in V2_SYSTEMS:
        return {"skip_respond": skip_respond, "max_fixes": max_fixes}
    return {}


ClientFactory = Callable[[LLMConfig, UsageTracker], LLMClient]


def load_llm_config(path: Path) -> LLMConfig:
    """Load an LLM config from YAML."""
    with open(path, encoding="utf-8") as f:
        return LLMConfig.model_validate(yaml.safe_load(f))


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
    skip_respond: bool = False,
    max_fixes: int = DEFAULT_MAX_FIXES,
    trace: bool = False,
    split: str | None = None,
) -> dict[str, Any]:
    """Predict and score one (question, system) pair. Raises BudgetExceeded on the cap.

    With ``trace`` the pair runs inside one Langfuse trace (session = ``run_id``, tags =
    system, database, difficulty and split); after scoring the trace gets the scores
    ``execution_accuracy`` (0/1) and ``cost_usd``, and the record gets a ``trace_id``.
    """
    if not trace:
        return _evaluate_pair(
            question,
            system,
            client,
            gold_cache,
            run_id=run_id,
            databases_dir=databases_dir,
            timeout_s=timeout_s,
            skip_respond=skip_respond,
            max_fixes=max_fixes,
        )
    tags = [system, question.db_id, question.difficulty] + ([split] if split else [])
    with tracing.trace(
        f"text2sql/{system}",
        input={"question": question.question, "evidence": question.evidence},
        metadata={"question_id": question.question_id, "run_id": run_id, "system": system},
        tags=tags,
        session_id=run_id,
    ) as handle:
        record = _evaluate_pair(
            question,
            system,
            client,
            gold_cache,
            run_id=run_id,
            databases_dir=databases_dir,
            timeout_s=timeout_s,
            skip_respond=skip_respond,
            max_fixes=max_fixes,
        )
        handle.update(
            output={
                "pred_sql": record["pred_sql"],
                "correct": record["correct"],
                "error": record["error"],
                "failure_type": record["failure_type"],
            }
        )
        tracing.score(
            handle.id,
            "execution_accuracy",
            1.0 if record["correct"] else 0.0,
            record["failure_type"],
        )
        tracing.score(handle.id, "cost_usd", record["cost_usd"])
    return {**record, "trace_id": handle.id}


def _evaluate_pair(
    question: Question,
    system: str,
    client: LLMClient,
    gold_cache: GoldCache,
    *,
    run_id: str,
    databases_dir: Path | None,
    timeout_s: float,
    skip_respond: bool,
    max_fixes: int,
) -> dict[str, Any]:
    database = db_path(question.db_id, databases_dir)
    recorder = CountingClient(client)
    pred_sql = ""
    n_calls = 0
    error: str | None = None
    details: dict[str, Any] | None = None
    exception = False
    start = time.perf_counter()
    with call_context(run_id=run_id, system=system, question_id=question.question_id):
        try:
            options = system_options(system, skip_respond=skip_respond, max_fixes=max_fixes)
            out = PREDICTORS[system](question, recorder, database=database, **options)
            pred_sql, n_calls = out[0], out[1]
            details = out[2] if len(out) > 2 else None
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

    extra: dict[str, Any] = {}
    if details is not None and not exception:
        first_sql = details["attempts"][0]["sql"] if details["attempts"] else ""
        if first_sql == pred_sql:
            first_correct = score.correct
        else:
            first_correct = score_prediction(
                question.question_id, database, first_sql, question.sql, gold_cache, timeout_s
            ).correct
        extra = {**details, "first_sql": first_sql, "first_sql_correct": first_correct}

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
        **extra,
    }


def summarize_v2(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Agent v2 extras: what the fix loop changed, relative to the first generated SQL."""
    rows = [r for r in records if "first_sql_correct" in r]
    if not rows:
        return None
    flips = {"wrong_to_right": 0, "right_to_wrong": 0, "wrong_to_wrong": 0, "right_to_right": 0}
    for r in rows:
        if r["first_sql"] == r["pred_sql"]:
            continue
        before = "right" if r["first_sql_correct"] else "wrong"
        after = "right" if r["correct"] else "wrong"
        flips[f"{before}_to_{after}"] += 1
    n = len(rows)
    first_correct = sum(1 for r in rows if r["first_sql_correct"])
    return {
        "n": n,
        "flips": flips,
        "n_final_differs_from_first": sum(flips.values()),
        "frac_with_fix": sum(1 for r in rows if r["fixes_used"] >= 1) / n,
        "llm_calls_mean": sum(r["n_llm_calls"] for r in rows) / n,
        "unparsed_checks": sum(1 for r in rows for v in r["verdicts"] if v["unparsed"]),
        "first_sql_correct": first_correct,
        "first_sql_ex": first_correct / n,
    }


def summarize_system(
    records: list[dict[str, Any]],
    n_planned: int,
    aborted: str | None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Aggregate prediction records of one system."""
    n = len(records)

    def mean(key: str) -> float | None:
        return sum(r[key] for r in records) / n if n else None

    def ex(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(1 for r in rows if r["correct"])
        ci = bootstrap_ci([r["correct"] for r in rows], resamples=resamples, seed=bootstrap_seed)
        return {
            "n": len(rows),
            "correct": correct,
            "ex": correct / len(rows) if rows else None,
            "ex_ci95": list(ci) if ci else None,
        }

    latencies = [r["latency_ms"] for r in records]
    input_total = sum(r["input_tokens"] for r in records)
    cache_read = sum(r["cache_read_tokens"] for r in records)
    cache_write = sum(r["cache_write_tokens"] for r in records)
    prompt_total = input_total + cache_read + cache_write
    v2 = summarize_v2(records)
    return {
        "n": n,
        "n_planned": n_planned,
        "aborted": aborted,
        **ex(records),
        **({"v2": v2} if v2 else {}),
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


def compare_systems(
    system_records: dict[str, list[dict[str, Any]]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
) -> dict[str, Any] | None:
    """Paired bootstrap of EX(j) - EX(i) for every pair i < j, in the order the systems ran.

    Keys look like ``agent_v2_minus_baseline``; only questions both systems ran are used.
    """
    names = list(system_records)
    outcomes = {s: {r["question_id"]: bool(r["correct"]) for r in system_records[s]} for s in names}
    comparison: dict[str, Any] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            result = paired_bootstrap(
                outcomes[second], outcomes[first], resamples=resamples, seed=bootstrap_seed
            )
            if result is not None:
                comparison[f"{second}_minus_{first}"] = result
    return comparison or None


def format_ci(block: dict[str, Any]) -> str:
    """``30.0% [12.0, 48.0]`` (percent); "-" when there are no questions."""
    if block["ex"] is None:
        return "-"
    ci = block.get("ex_ci95")
    if not ci:
        return f"{block['ex']:.1%}"
    return f"{block['ex']:.1%} [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def format_table(summary: dict[str, Any]) -> str:
    """Short text table of the per-system results."""
    header = (
        f"{'system':<18}{'n':>5}{'EX [95% CI]':>22}{'simple':>9}{'moderate':>10}{'challeng.':>11}"
        f"{'cost $':>10}{'in tok/q':>10}{'out tok/q':>11}{'lat s/q':>9}"
    )
    lines = [header, "-" * len(header)]

    def pct(block: dict[str, Any]) -> str:
        return f"{block['ex']:.1%}" if block["ex"] is not None else "-"

    for name, s in summary["systems"].items():
        tokens = s["tokens"]
        latency = s["latency_ms"]["mean"]
        lines.append(
            f"{name:<18}{s['n']:>5}{format_ci(s):>22}"
            f"{pct(s['by_difficulty']['simple']):>9}{pct(s['by_difficulty']['moderate']):>10}"
            f"{pct(s['by_difficulty']['challenging']):>11}"
            f"{s['cost_usd']['total']:>10.4f}"
            f"{(tokens['input_mean_per_question'] or 0):>10.0f}"
            f"{(tokens['output_mean_per_question'] or 0):>11.0f}"
            f"{(latency or 0) / 1000:>9.1f}"
        )
        if s["aborted"]:
            lines.append(f"  {name}: aborted ({s['aborted']}), {s['n']}/{s['n_planned']} done")
        by_difficulty = ", ".join(
            f"{d} {format_ci(s['by_difficulty'][d])}" for d in s["by_difficulty"]
        )
        lines.append(f"  {name} EX by difficulty [95% CI]: {by_difficulty}")
        if s["failures"]:
            lines.append(f"  {name} failures: {s['failures']}")
    for name, s in summary["systems"].items():
        v2 = s.get("v2")
        if v2:
            f = v2["flips"]
            lines.append(
                f"  {name} first-SQL EX: {v2['first_sql_ex']:.1%}; final differs from first in "
                f"{v2['n_final_differs_from_first']} (wrong->right {f['wrong_to_right']}, "
                f"right->wrong {f['right_to_wrong']}, wrong->wrong {f['wrong_to_wrong']}, "
                f"right->right {f['right_to_right']}); with a fix: {v2['frac_with_fix']:.1%}; "
                f"LLM calls/q: {v2['llm_calls_mean']:.2f}; unparsed checks: {v2['unparsed_checks']}"
            )
    for key, paired in (summary.get("comparison") or {}).items():
        label = key.replace("_minus_", " - ")
        lines.append(
            f"{label} EX: {paired['diff'] * 100:+.1f} pts "
            f"[{paired['ci_low'] * 100:+.1f}, {paired['ci_high'] * 100:+.1f}] "
            f"(95% paired bootstrap, n={paired['n']}, "
            f"resamples with diff <= 0: {paired['frac_resamples_le_zero']:.1%})"
        )
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
    skip_respond: bool = False,
    max_fixes: int = DEFAULT_MAX_FIXES,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
    extra_meta: dict[str, Any] | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
    trace: bool = False,
) -> dict[str, Any]:
    """Run the systems over the questions and write predictions, LLM calls and summary.

    ``run.json`` (see ``evals.provenance``) is written before the first question runs. An
    existing, non-empty ``run_dir`` is refused unless ``resume`` is set; a resume appends an
    entry to the existing ``run.json``.

    With ``trace`` every (question, system) pair is one Langfuse trace in the session
    ``run_id`` (see :func:`evaluate_question`); it needs Langfuse to be configured. Without
    it tracing is switched off for the whole run, even when the keys are set.
    """
    if trace and not tracing.enabled():
        raise TracingNotConfiguredError(
            "--trace needs Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "(and LANGFUSE_HOST) in the environment or .env, e.g. after `make langfuse-up`"
        )
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"{run_dir} exists; use --resume or another --run-id")
    run_file = start_run(
        run_dir,
        run_id=run_id,
        argv=argv,
        llm_config_path=llm_config_name,
        config=config,
        budget_usd=budget_usd,
        launched_by=launched_by,
        resume=resume,
    )
    if trace:
        update_run(
            run_file,
            langfuse={"host": settings.langfuse_host, "session_id": run_id, "traced": True},
        )
    try:
        scope = contextlib.nullcontext() if trace else tracing.use_tracer(NoopTracer())
        with scope:
            summary = _run_experiment(
                config=config,
                systems=systems,
                questions=questions,
                run_dir=run_dir,
                run_id=run_id,
                split=split,
                budget_usd=budget_usd,
                workers=workers,
                resume=resume,
                llm_config_name=llm_config_name,
                client_factory=client_factory,
                databases_dir=databases_dir,
                timeout_s=timeout_s,
                skip_respond=skip_respond,
                max_fixes=max_fixes,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                extra_meta=extra_meta,
                trace=trace,
            )
    except BaseException as e:
        finish_run(
            run_file,
            "interrupted" if isinstance(e, KeyboardInterrupt) else f"failed:{type(e).__name__}",
        )
        raise
    finish_run(run_file, status_from_abort(summary["aborted"]))
    return summary


def _run_experiment(
    *,
    config: LLMConfig,
    systems: list[str],
    questions: list[Question],
    run_dir: Path,
    run_id: str,
    split: str,
    budget_usd: float | None,
    workers: int,
    resume: bool,
    llm_config_name: str | None,
    client_factory: ClientFactory,
    databases_dir: Path | None,
    timeout_s: float,
    skip_respond: bool,
    max_fixes: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    extra_meta: dict[str, Any] | None,
    trace: bool,
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
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
                skip_respond=skip_respond,
                max_fixes=max_fixes,
                trace=trace,
                split=split,
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

    if trace:
        tracing.flush()
    final = read_predictions(predictions_path)
    selected_ids = {q.question_id for q in questions}
    system_summaries: dict[str, Any] = {}
    system_records: dict[str, list[dict[str, Any]]] = {}
    for system in systems:
        records = sorted(
            (r for (qid, s), r in final.items() if s == system and qid in selected_ids),
            key=lambda r: (r["db_id"], r["question_id"]),
        )
        aborted = aborted_systems.get(system)
        if aborted is None and len(records) < n_planned and abort.is_set():
            aborted = next(iter(abort_reason.values()), "interrupted")
        system_records[system] = records
        system_summaries[system] = summarize_system(
            records,
            n_planned,
            aborted,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "n_questions": n_planned,
        "systems": system_summaries,
        "comparison": compare_systems(
            system_records, resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
        ),
        "bootstrap": bootstrap_meta(bootstrap_resamples, bootstrap_seed),
        "aborted": next(iter(aborted_systems.values()), None),
        "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
        "resumed": resume,
        "workers": workers,
        "skip_respond": skip_respond,
        "max_fixes": max_fixes,
        "systems_order": systems,
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
    parser.add_argument(
        "--system",
        default="both",
        help="comma-separated systems, run in this order: baseline, baseline_ctx, agent_v1,\n"
        "agent_v2_nocheck, agent_v2 (agent = agent_v1; both = baseline,agent_v1)",
    )
    parser.add_argument(
        "--max-fixes",
        type=int,
        default=DEFAULT_MAX_FIXES,
        help="agent_v2*: maximum number of query fixes after the first SQL (default 2)",
    )
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
    parser.add_argument(
        "--skip-respond",
        action="store_true",
        help="agents: skip the final natural-language answer step (EX does not score it);\n"
        "its cost is then excluded from the reported cost; recorded in summary.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run dir (skip pairs already predicted); without it an\n"
        "existing run dir is refused",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="send one Langfuse trace per (question, system) pair (session = run id) with\n"
        "execution_accuracy and cost_usd scores; needs LANGFUSE_* keys (make langfuse-up)",
    )
    parser.add_argument(
        "--launched-by",
        help=f"free text recorded in run.json (default: env {LAUNCHED_BY_ENV})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv if argv is None else argv)
    args = parse_args(argv)
    config = load_llm_config(args.llm_config)
    if config.provider == "anthropic" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for the anthropic provider")
    try:
        systems = parse_systems(args.system)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if args.trace and not tracing.enabled():
        raise SystemExit(
            "--trace needs Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and "
            "LANGFUSE_HOST) in the environment or .env; `make langfuse-up` starts a local one"
        )
    run_id = args.run_id or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{args.llm_config.stem}"
    questions = select_questions(
        load_questions(), load_split(args.split), args.limit, args.question_ids, args.seed
    )
    print(f"Run {run_id}: {len(questions)} questions, systems {systems}, split {args.split}")
    with sigterm_as_interrupt():
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
            skip_respond=args.skip_respond,
            max_fixes=args.max_fixes,
            llm_config_name=str(args.llm_config),
            extra_meta={
                "limit": args.limit,
                "question_ids": args.question_ids,
                "sampling": sampling_meta(questions, args.limit, args.question_ids, args.seed),
            },
            argv=raw_argv,
            launched_by=args.launched_by,
            trace=args.trace,
        )
    print()
    print(format_table(summary))
    print(f"\nResults in {RESULTS_ROOT / run_id}")
    if args.trace:
        print(
            f"Traces: {settings.langfuse_host} (session {run_id}); check with "
            f"`python -m evals.observability.verify_traces --session {run_id} "
            f"--predictions {RESULTS_ROOT / run_id}`"
        )


if __name__ == "__main__":
    main()
