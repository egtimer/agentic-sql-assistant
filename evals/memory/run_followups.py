"""Short-term (conversation) memory evaluation on two-turn follow-ups.

``evals/memory/followups.jsonl`` holds 30 conversations written from BIRD Mini-Dev dev
questions: turn 1 sets a context, turn 2 is an elliptical follow-up. The score is always the
execution accuracy of one SQL against the gold SQL of the source question. Conditions, each
with the source question's evidence:

- ``turn2_alone``   turn 2 only, in a fresh thread
- ``with_history``  one thread: turn 1, then turn 2 (turn 2 is scored; turn 1 is not, but its
                    cost and LLM calls count)
- ``original``      the source question itself, in a fresh thread (the ceiling)

n is 30 and every pair comes from the dev split: read the intervals, not the point estimates.

Example (spends money with the Anthropic config; the owner runs it):

    python -m evals.memory.run_followups --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --skip-respond --workers 4 --budget-usd 0.50 --run-id fu-dev-30 --launched-by owner
"""

import argparse
import functools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.memory.common import (
    RESULTS_ROOT_NAME,
    ClientFactory,
    Engine,
    make_client,
    new_tracker,
    provenance,
    scored_record,
    timed_calls,
)
from evals.provenance import LAUNCHED_BY_ENV, git_info, sigterm_as_interrupt
from evals.text2sql.data import REPO_ROOT, Question, db_path, load_questions, load_split
from evals.text2sql.run import (
    DEFAULT_MAX_FIXES,
    PredictionWriter,
    compare_systems,
    format_table,
    load_llm_config,
    read_predictions,
    summarize_system,
)
from evals.text2sql.scorer import GoldCache
from evals.text2sql.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_meta
from src.llm import LLMConfig
from src.sql_agent.graph import memory_checkpointer, run_sql_agent

CONDITIONS = ("turn2_alone", "with_history", "original")
FOLLOWUPS_PATH = Path(__file__).with_name("followups.jsonl")
RESULTS_ROOT = REPO_ROOT.joinpath(*RESULTS_ROOT_NAME)
SPLIT = "dev"


class FollowUp(BaseModel):
    """One two-turn conversation and the dev question it was written from."""

    id: str
    source_question_id: int
    db_id: str
    difficulty: str
    turn1: str
    turn2: str


def load_followups(path: Path | None = None) -> list[FollowUp]:
    with open(path or FOLLOWUPS_PATH, encoding="utf-8") as f:
        return [FollowUp.model_validate_json(line) for line in f if line.strip()]


def pair_questions(
    followups: list[FollowUp], questions: list[Question], split_ids: list[int]
) -> list[tuple[FollowUp, Question]]:
    """Each follow-up with its source question; ordered by (database, source id).

    Raises when ids repeat, a source question is missing, is outside the split, or belongs
    to another database than the follow-up says.
    """
    by_id = {q.question_id: q for q in questions}
    allowed = set(split_ids)
    if len({f.id for f in followups}) != len(followups):
        raise ValueError("duplicate follow-up ids")
    if len({f.source_question_id for f in followups}) != len(followups):
        raise ValueError("two follow-ups share a source question")
    pairs = []
    for f in followups:
        source = by_id.get(f.source_question_id)
        if source is None:
            raise ValueError(f"{f.id}: source question {f.source_question_id} not found")
        if source.question_id not in allowed:
            raise ValueError(f"{f.id}: source question {source.question_id} is not in {SPLIT}")
        if source.db_id != f.db_id:
            raise ValueError(f"{f.id}: database {f.db_id!r} != {source.db_id!r} of the source")
        pairs.append((f, source))
    return sorted(pairs, key=lambda p: (p[1].db_id, p[1].question_id))


EVIDENCE_MODES = ("source", "none")


def run_followups(
    *,
    config: LLMConfig,
    pairs: list[tuple[FollowUp, Question]],
    run_dir: Path,
    run_id: str,
    budget_usd: float | None = None,
    workers: int = 1,
    llm_config_name: str | None = None,
    client_factory: ClientFactory | None = None,
    databases_dir: Path | None = None,
    timeout_s: float = 30,
    skip_respond: bool = False,
    max_fixes: int = DEFAULT_MAX_FIXES,
    evidence: str = "source",
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_SEED,
    extra_meta: dict[str, Any] | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
) -> dict[str, Any]:
    """Run the three conditions over the pairs; write the run directory (must be new)."""
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
        box["summary"] = _run(
            config=config,
            pairs=pairs,
            run_dir=run_dir,
            run_id=run_id,
            budget_usd=budget_usd,
            workers=workers,
            llm_config_name=llm_config_name,
            client_factory=client_factory,
            databases_dir=databases_dir,
            timeout_s=timeout_s,
            skip_respond=skip_respond,
            max_fixes=max_fixes,
            evidence=evidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            extra_meta=extra_meta,
        )
    summary: dict[str, Any] = box["summary"]
    return summary


def _run(
    *,
    config: LLMConfig,
    pairs: list[tuple[FollowUp, Question]],
    run_dir: Path,
    run_id: str,
    budget_usd: float | None,
    workers: int,
    llm_config_name: str | None,
    client_factory: ClientFactory | None,
    databases_dir: Path | None,
    timeout_s: float,
    skip_respond: bool,
    max_fixes: int,
    evidence: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    extra_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    tracker = new_tracker(budget_usd, run_dir)
    client = make_client(client_factory, config, tracker)
    predictions_path = run_dir / "predictions.jsonl"
    predictions = PredictionWriter(predictions_path)
    gold_cache = GoldCache(run_dir / "gold_cache.json")
    engine = Engine(workers)
    options: dict[str, Any] = {"skip_respond": skip_respond, "max_fixes": max_fixes}
    if evidence not in EVIDENCE_MODES:
        raise ValueError(f"evidence must be one of {EVIDENCE_MODES}")

    def case(pair: tuple[FollowUp, Question], condition: str) -> dict[str, Any]:
        follow, source = pair
        ev = source.evidence if evidence == "source" else ""
        database = db_path(source.db_id, databases_dir)
        result = None
        extra: dict[str, Any] = {"pair_id": follow.id, "n_turns": 1}
        with timed_calls(client, run_id, condition, source.question_id) as call:
            if condition == "original":
                result = run_sql_agent(source.question, ev, database, **options)
            elif condition == "turn2_alone":
                result = run_sql_agent(follow.turn2, ev, database, **options)
            else:
                thread = {
                    "thread_id": f"{run_id}/{follow.id}",
                    "checkpointer": memory_checkpointer(),
                }
                first = run_sql_agent(follow.turn1, ev, database, **thread, **options)
                extra = {"pair_id": follow.id, "n_turns": 2, "turn1_sql": first.final_sql}
                extra["turn1_cost_usd"] = sum(u.cost_usd for u in call.counting.usages)
                extra["turn1_llm_calls"] = len(call.counting.usages)
                result = run_sql_agent(follow.turn2, ev, database, **thread, **options)
        return scored_record(
            run_id=run_id,
            question=source,
            condition=condition,
            call=call,
            result=result,
            database=database,
            gold_cache=gold_cache,
            timeout_s=timeout_s,
            extra=extra,
        )

    try:
        for condition in CONDITIONS:
            if engine.abort.is_set():
                break
            engine.run_pass(
                condition, pairs, functools.partial(case, condition=condition), predictions
            )
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results.")
        engine.stop("interrupted")

    final = read_predictions(predictions_path)
    selected = {source.question_id for _, source in pairs}
    n_planned = len(pairs)
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
    summary: dict[str, Any] = {
        "run_id": run_id,
        "kind": "followup_memory",
        "split": SPLIT,
        "n_questions": n_planned,
        "systems": systems,
        "comparison": compare_systems(
            system_records, resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
        ),
        "totals": {
            "cost_usd": sum(s["cost_usd"]["total"] for s in systems.values()),
            "n_llm_calls": sum(r["n_llm_calls"] for rs in system_records.values() for r in rs),
        },
        "notes": [
            f"n = {n_planned} conversations and every one comes from the {SPLIT} split: "
            "the intervals are wide, treat the point estimates as indicative.",
            "with_history scores turn 2 only; turn 1 cost and calls are in its totals.",
            (
                "Every condition gets the evidence of the source question; the history is the "
                "only variable between turn2_alone and with_history. The evidence often names "
                "the entity that turn 1 introduces, which makes turn 2 less elliptical."
                if evidence == "source"
                else "No condition gets any evidence, so the history is the only source of "
                "the context that turn 1 introduces; the ceiling (original) is lower than with "
                "evidence."
            ),
        ],
        "bootstrap": bootstrap_meta(bootstrap_resamples, bootstrap_seed),
        "aborted": engine.reason,
        "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
        "workers": workers,
        "skip_respond": skip_respond,
        "max_fixes": max_fixes,
        "evidence": evidence,
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
    lines = [format_table(summary), ""]
    lines.append(f"total cost ${summary['totals']['cost_usd']:.4f}")
    lines.extend(f"note: {n}" for n in summary["notes"][:1])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument(
        "--run-id", required=True, help="new output folder name under evals/results/memory/"
    )
    parser.add_argument("--followups", type=Path, default=FOLLOWUPS_PATH)
    parser.add_argument("--budget-usd", type=float, help="hard cap on spend (anthropic: required)")
    parser.add_argument("--workers", type=int, default=1, help="threads; intended for the API")
    parser.add_argument("--skip-respond", action="store_true", help="skip the answer sentence")
    parser.add_argument("--max-fixes", type=int, default=DEFAULT_MAX_FIXES)
    parser.add_argument(
        "--evidence",
        choices=EVIDENCE_MODES,
        default="source",
        help="source: every condition gets the source question's evidence; none: no condition "
        "does, so turn 2 is fully elliptical and the history is the only context",
    )
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
    pairs = pair_questions(load_followups(args.followups), load_questions(), load_split(SPLIT))
    run_dir = RESULTS_ROOT / args.run_id
    print(f"Run {args.run_id}: {len(pairs)} conversations from the {SPLIT} split")
    try:
        with sigterm_as_interrupt():
            summary = run_followups(
                config=config,
                pairs=pairs,
                run_dir=run_dir,
                run_id=args.run_id,
                budget_usd=args.budget_usd,
                workers=args.workers,
                llm_config_name=str(args.llm_config),
                skip_respond=args.skip_respond,
                max_fixes=args.max_fixes,
                evidence=args.evidence,
                argv=raw_argv,
                launched_by=args.launched_by,
            )
    except FileExistsError as e:
        raise SystemExit(str(e)) from e
    print()
    print(format_report(summary))
    print(f"\nResults in {run_dir}")


if __name__ == "__main__":
    main()
