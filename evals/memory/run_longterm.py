"""Long-term memory evaluation, built from BIRD's own external knowledge.

For every selected question with non-empty evidence, a teaching turn tells the agent that
evidence as a user would ("For my questions about the <db> database: <evidence>"); ``teach``
extracts definitions into a store (one fresh store per database, user ``eval-user``, all the
evidences of the database accumulate). Then each question is answered, each in a fresh thread
(no conversation memory), under three conditions:

- ``no_memory``  agent v2 without the evidence
- ``memory``     agent v2 without the evidence, recalling from the store
- ``oracle``     agent v2 with the evidence (today's system)

This measures how much of the value of the evidence survives the trip through ``teach`` and
``recall``. The teaching text is the dataset's own evidence, not a user's free phrasing.

Example (spends money with the Anthropic config; the owner runs it):

    python -m evals.memory.run_longterm --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --split dev --limit 30 --seed 0 --skip-respond --workers 4 --budget-usd 1.00 \\
        --run-id lt-dev-30 --launched-by owner
"""

import argparse
import functools
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore

from evals.memory.common import (
    RESULTS_ROOT_NAME,
    TEST_SPLIT_WARNING,
    ClientFactory,
    Engine,
    fmt_pct,
    gap_closed,
    make_client,
    mean_or_none,
    new_tracker,
    provenance,
    read_jsonl,
    scored_record,
    timed_calls,
)
from evals.provenance import LAUNCHED_BY_ENV, git_info, sigterm_as_interrupt
from evals.text2sql.data import REPO_ROOT, Question, db_path, load_questions, load_split
from evals.text2sql.run import (
    DEFAULT_MAX_FIXES,
    PredictionWriter,
    compare_systems,
    execution_order,
    format_table,
    load_llm_config,
    read_predictions,
    sampling_meta,
    select_questions,
    summarize_system,
)
from evals.text2sql.scorer import GoldCache
from evals.text2sql.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_meta
from src.llm import LLMConfig
from src.sql_agent.graph import run_sql_agent
from src.sql_agent.memory import load_memories, teach_text

USER_ID = "eval-user"
CONDITIONS = ("no_memory", "memory", "oracle")
RESULTS_ROOT = REPO_ROOT.joinpath(*RESULTS_ROOT_NAME)
TEACH_TEMPLATE = "For my questions about the {db_id} database: {evidence}"


def teaching_text(db_id: str, evidence: str) -> str:
    return TEACH_TEMPLATE.format(db_id=db_id, evidence=evidence.strip())


def select_longterm_questions(
    questions: list[Question], split_ids: list[int], limit: int | None = None, seed: int = 0
) -> list[Question]:
    """Questions of the split that have evidence; ``limit`` draws a stratified sample of them."""
    wanted = set(split_ids)
    with_evidence = [q for q in questions if q.question_id in wanted and q.evidence.strip()]
    return select_questions(
        with_evidence, [q.question_id for q in with_evidence], limit=limit, seed=seed
    )


def summarize_teaching(records: list[dict[str, Any]], stores: dict[str, InMemoryStore]) -> Any:
    n = len(records)
    stored = [r["n_memories"] for r in records]
    # distinct memory keys per database, from the teaching log (a later definition of the same
    # term replaces the earlier one in the store); this also works after a resume, when the
    # in-process stores of the first invocation are gone
    keys_by_db: dict[str, set[str]] = defaultdict(set)
    for r in records:
        keys_by_db[r["db_id"]].update(r.get("keys") or [])
    in_store = sum(len(k) for k in keys_by_db.values())
    del stores
    return {
        "n_turns": n,
        "n_errors": sum(1 for r in records if r["error"]),
        "memories_extracted": sum(stored),
        "memories_in_store_after_dedup": in_store,
        "turns_with_zero_memories": sum(1 for m in stored if m == 0),
        "frac_turns_with_zero_memories": (sum(1 for m in stored if m == 0) / n) if n else None,
        "mean_memories_per_turn": mean_or_none(stored),
        "items_parsed": sum(r["n_parsed"] for r in records),
        "items_rejected_term_not_in_text": sum(r["n_rejected"] for r in records),
        "cost_usd": sum(r["cost_usd"] for r in records),
        "n_llm_calls": sum(r["n_llm_calls"] for r in records),
        "latency_ms_mean": mean_or_none([r["latency_ms"] for r in records]),
    }


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How well ``recall`` found each question's own definitions (``memory`` condition)."""
    n = len(records)
    hits = [r for r in records if r.get("retrieval_hit")]
    stored = [r for r in records if r.get("own_memory_stored")]
    return {
        "n": n,
        "retrieval_recall": len(hits) / n if n else None,
        "n_with_own_memory": len(stored),
        "retrieval_recall_given_stored": (len(hits) / len(stored)) if stored else None,
        "mean_injected": mean_or_none([r["n_recalled"] for r in records]),
        "mean_available": mean_or_none([r["n_available"] for r in records]),
        "note": "a question's own memory is the one stored with source_ref = its id; a term "
        "taught by two questions keeps only the later one, so the earlier question has none",
    }


def run_longterm(
    *,
    config: LLMConfig,
    questions: list[Question],
    run_dir: Path,
    run_id: str,
    split: str,
    budget_usd: float | None = None,
    workers: int = 1,
    llm_config_name: str | None = None,
    client_factory: ClientFactory | None = None,
    databases_dir: Path | None = None,
    timeout_s: float = 30,
    skip_respond: bool = False,
    max_fixes: int = DEFAULT_MAX_FIXES,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
    extra_meta: dict[str, Any] | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Teach, then answer ``questions`` under the three conditions; write the run directory.

    The run directory must be new unless ``resume`` is set. ``run.json`` is written first.
    A resume skips every (question, condition) already in ``predictions.jsonl`` and the
    teaching of databases whose teaching is complete; it refuses to start when questions of
    the ``memory`` condition are still pending for a database whose teaching already ran,
    because that store lived in the process that stopped and teaching again would give
    different memories.
    """
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
        resume=resume,
    ):
        box["summary"] = _run(
            config=config,
            questions=questions,
            run_dir=run_dir,
            run_id=run_id,
            split=split,
            budget_usd=budget_usd,
            workers=workers,
            llm_config_name=llm_config_name,
            client_factory=client_factory,
            databases_dir=databases_dir,
            timeout_s=timeout_s,
            skip_respond=skip_respond,
            max_fixes=max_fixes,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            extra_meta=extra_meta,
            resume=resume,
        )
    summary: dict[str, Any] = box["summary"]
    return summary


def _run(
    *,
    config: LLMConfig,
    questions: list[Question],
    run_dir: Path,
    run_id: str,
    split: str,
    budget_usd: float | None,
    workers: int,
    llm_config_name: str | None,
    client_factory: ClientFactory | None,
    databases_dir: Path | None,
    timeout_s: float,
    skip_respond: bool,
    max_fixes: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    extra_meta: dict[str, Any] | None,
    resume: bool = False,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    tracker = new_tracker(budget_usd, run_dir)
    client = make_client(client_factory, config, tracker)
    predictions_path = run_dir / "predictions.jsonl"
    teaching_path = run_dir / "teaching.jsonl"
    predictions, teaching = PredictionWriter(predictions_path), PredictionWriter(teaching_path)
    gold_cache = GoldCache(run_dir / "gold_cache.json")
    engine = Engine(workers)
    ordered = execution_order(questions)
    by_db: dict[str, list[Question]] = defaultdict(list)
    for q in ordered:
        by_db[q.db_id].append(q)
    stores: dict[str, InMemoryStore] = {}
    done = set(read_predictions(predictions_path)) if resume else set()
    taught = {r["question_id"] for r in read_jsonl(teaching_path)} if resume else set()
    to_teach = sorted(
        db for db, qs in by_db.items() if any(q.question_id not in taught for q in qs)
    )
    partial = [db for db in to_teach if any(q.question_id in taught for q in by_db[db])]
    if partial:
        raise RuntimeError(
            f"cannot resume: the teaching of {partial} stopped half way and its store is gone"
        )
    pending = {c: [q for q in ordered if (q.question_id, c) not in done] for c in CONDITIONS}
    lost = sorted({q.db_id for q in pending["memory"] if q.db_id not in to_teach})
    if lost:
        raise RuntimeError(
            f"cannot resume: memory questions are pending for {lost}, whose teaching ran in the "
            "stopped process; its store is gone and teaching again would give different "
            "memories. Start a new run id."
        )
    if resume:
        print(
            "resume: "
            + ", ".join(
                f"{c} {len(ordered) - len(pending[c])}/{len(ordered)} done" for c in CONDITIONS
            )
            + f"; teaching to run for {len(to_teach)} database(s)"
        )

    def teach_database(db_id: str) -> dict[str, Any]:
        """One teaching turn per question of the database, in id order, into a fresh store."""
        store = stores[db_id] = InMemoryStore()
        failed = False
        for q in by_db[db_id]:
            result = None
            with timed_calls(client, run_id, "teaching", q.question_id) as call:
                result = teach_text(
                    teaching_text(db_id, q.evidence),
                    user_id=USER_ID,
                    db_id=db_id,
                    store=store,
                    source_ref=str(q.question_id),
                )
            usages = call.counting.usages
            failed = failed or call.error is not None
            teaching.write(
                {
                    "run_id": run_id,
                    "question_id": q.question_id,
                    "db_id": db_id,
                    "n_memories": len(result.memories) if result else 0,
                    "keys": [m.key for m in result.memories] if result else [],
                    "n_parsed": result.n_parsed if result else 0,
                    "n_rejected": result.n_rejected if result else 0,
                    "error": call.error,
                    "n_llm_calls": len(usages),
                    "input_tokens": sum(u.input_tokens for u in usages),
                    "output_tokens": sum(u.output_tokens for u in usages),
                    "cost_usd": sum(u.cost_usd for u in usages),
                    "latency_ms": call.latency_ms,
                }
            )
        return {"failure_type": "exception" if failed else None}

    def answer(q: Question, condition: str) -> dict[str, Any]:
        database = db_path(q.db_id, databases_dir)
        store = stores.get(q.db_id) if condition == "memory" else None
        result = None
        with timed_calls(client, run_id, condition, q.question_id) as call:
            result = run_sql_agent(
                q.question,
                q.evidence if condition == "oracle" else "",
                database,
                skip_respond=skip_respond,
                max_fixes=max_fixes,
                store=store,
                user_id=USER_ID if store is not None else None,
            )
        extra: dict[str, Any] = {}
        if condition == "memory" and store is not None:
            own = {
                m.key
                for m in load_memories(store, USER_ID, q.db_id)
                if m.source_ref == str(q.question_id)
            }
            injected = [m.key for m in result.recalled] if result else []
            extra = {
                "recalled_keys": injected,
                "n_recalled": len(injected),
                "n_available": result.n_memories if result else 0,
                "own_memory_stored": bool(own),
                "retrieval_hit": bool(own & set(injected)),
            }
        return scored_record(
            run_id=run_id,
            question=q,
            condition=condition,
            call=call,
            result=result,
            database=database,
            gold_cache=gold_cache,
            timeout_s=timeout_s,
            extra=extra,
        )

    try:
        engine.run_pass("teaching", to_teach, teach_database, None)
        for condition in CONDITIONS:
            if engine.abort.is_set():
                break
            engine.run_pass(
                condition,
                pending[condition],
                functools.partial(answer, condition=condition),
                predictions,
            )
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results.")
        engine.stop("interrupted")

    final = read_predictions(predictions_path)
    selected = {q.question_id for q in questions}
    n_planned = len(ordered)
    system_records: dict[str, list[dict[str, Any]]] = {}
    systems: dict[str, Any] = {}
    for condition in CONDITIONS:
        records = sorted(
            (r for (qid, s), r in final.items() if s == condition and qid in selected),
            key=lambda r: (r["db_id"], r["question_id"]),
        )
        aborted = engine.reason if len(records) < n_planned else None
        system_records[condition] = records
        systems[condition] = summarize_system(
            records,
            n_planned,
            aborted,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
    outcomes = {
        c: {r["question_id"]: bool(r["correct"]) for r in rs} for c, rs in system_records.items()
    }
    teaching_records = read_jsonl(teaching_path)
    teach_summary = summarize_teaching(teaching_records, stores)
    condition_cost = sum(s["cost_usd"]["total"] for s in systems.values())
    condition_calls = sum(r["n_llm_calls"] for rs in system_records.values() for r in rs)
    memory_n = systems["memory"]["n"]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "kind": "longterm_memory",
        "split": split,
        "n_questions": n_planned,
        "user_id": USER_ID,
        "systems": systems,
        "comparison": compare_systems(
            system_records, resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
        ),
        "gap_closed": gap_closed(
            outcomes["no_memory"],
            outcomes["memory"],
            outcomes["oracle"],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "teaching": teach_summary,
        "retrieval": summarize_retrieval(system_records["memory"]),
        "totals": {
            "cost_usd": teach_summary["cost_usd"] + condition_cost,
            "cost_usd_teaching": teach_summary["cost_usd"],
            "cost_usd_conditions": condition_cost,
            "cost_usd_memory_with_teaching_per_question": (
                (teach_summary["cost_usd"] + systems["memory"]["cost_usd"]["total"]) / memory_n
                if memory_n
                else None
            ),
            "n_llm_calls": teach_summary["n_llm_calls"] + condition_calls,
        },
        "notes": [
            "The teaching text is BIRD's own evidence wrapped in one sentence, so this measures "
            "storing and recalling definitions, not a user's free-form phrasing.",
            "no_memory and memory run without the evidence; oracle runs with it.",
        ],
        "bootstrap": bootstrap_meta(bootstrap_resamples, bootstrap_seed),
        "aborted": engine.reason,
        "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
        "workers": workers,
        "resumed": resume,
        "skip_respond": skip_respond,
        "max_fixes": max_fixes,
        "conditions_order": list(CONDITIONS),
        "llm_config_file": llm_config_name,
        "llm_config": config.model_dump(),
        "git": git_info(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        **(extra_meta or {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_report(summary: dict[str, Any]) -> str:
    """Text report: the usual table plus gap closed, teaching, retrieval and cost."""
    lines = [format_table(summary), ""]
    gap = summary["gap_closed"]
    if gap:
        ci = (
            f" [{fmt_pct(gap['ci_low'])}, {fmt_pct(gap['ci_high'])}]"
            if gap["ci_low"] is not None
            else ""
        )
        lines.append(
            f"gap closed (memory-no_memory)/(oracle-no_memory): {fmt_pct(gap['gap_closed'])}{ci}; "
            f"resamples with a zero denominator: {gap['frac_undefined']:.1%}"
        )
    t = summary["teaching"]
    lines.append(
        f"teaching: {t['n_turns']} turns, {t['memories_extracted']} memories extracted "
        f"({t['memories_in_store_after_dedup']} after dedup), {t['turns_with_zero_memories']} "
        f"turns with none, {t['mean_memories_per_turn'] or 0:.2f} per turn, "
        f"{t['items_rejected_term_not_in_text']} items rejected, cost ${t['cost_usd']:.4f}"
    )
    r = summary["retrieval"]
    lines.append(
        f"retrieval recall: {fmt_pct(r['retrieval_recall'])} of questions "
        f"({fmt_pct(r['retrieval_recall_given_stored'])} of those with a stored memory); "
        f"mean injected {r['mean_injected'] or 0:.1f} of {r['mean_available'] or 0:.1f} available"
    )
    totals = summary["totals"]
    lines.append(
        f"total cost ${totals['cost_usd']:.4f} (teaching ${totals['cost_usd_teaching']:.4f}), "
        f"{totals['n_llm_calls']} LLM calls"
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument(
        "--run-id", required=True, help="new output folder name under evals/results/memory/"
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="stratified sample of N questions with evidence")
    parser.add_argument("--seed", type=int, default=0, help="seed of the --limit sampling")
    parser.add_argument("--budget-usd", type=float, help="hard cap on spend (anthropic: required)")
    parser.add_argument("--workers", type=int, default=1, help="threads; intended for the API")
    parser.add_argument("--skip-respond", action="store_true", help="skip the answer sentence")
    parser.add_argument("--max-fixes", type=int, default=DEFAULT_MAX_FIXES)
    parser.add_argument(
        "--launched-by", help=f"free text recorded in run.json (default: env {LAUNCHED_BY_ENV})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a stopped run with the same --run-id and options: skips every finished "
        "(question, condition) and complete teaching; refuses if memory questions are pending "
        "for a database already taught (its store is gone)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv if argv is None else argv)
    args = parse_args(argv)
    config = load_llm_config(args.llm_config)
    if config.provider == "anthropic" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for the anthropic provider")
    if args.split == "test":
        print(TEST_SPLIT_WARNING, file=sys.stderr)
    run_dir = RESULTS_ROOT / args.run_id
    questions = select_longterm_questions(
        load_questions(), load_split(args.split), args.limit, args.seed
    )
    print(f"Run {args.run_id}: {len(questions)} questions with evidence, split {args.split}")
    try:
        with sigterm_as_interrupt():
            summary = run_longterm(
                config=config,
                questions=questions,
                run_dir=run_dir,
                run_id=args.run_id,
                split=args.split,
                budget_usd=args.budget_usd,
                workers=args.workers,
                llm_config_name=str(args.llm_config),
                skip_respond=args.skip_respond,
                max_fixes=args.max_fixes,
                extra_meta={
                    "limit": args.limit,
                    "sampling": sampling_meta(questions, args.limit, None, args.seed),
                },
                argv=raw_argv,
                launched_by=args.launched_by,
                resume=args.resume,
            )
    except (FileExistsError, FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e)) from e
    print()
    print(format_report(summary))
    print(f"\nResults in {run_dir}")


if __name__ == "__main__":
    main()
