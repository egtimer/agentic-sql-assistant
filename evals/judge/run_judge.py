"""Run the answer judge over an items run (see ``build_set.py``).

One LLM call per item (tag ``judge``); an item whose gold result could not be computed is
skipped (label ``no_reference``) and costs nothing. Output in ``evals/results/judge/<run-id>/``
(gitignored: the rationales may quote gold rows):

- ``judgements.jsonl``  item id, label, rationale, tokens, cost, latency
- ``summary.json``      label counts, unparsed count, cost, provenance and budget
- ``run.json``, ``llm_calls.jsonl``

Spends money with the Anthropic config (50 calls for 50 items); the owner runs it:

    uv run python -m evals.judge.run_judge --items evals/results/judge/judge-set-dev-50 \\
        --llm-config configs/llm/claude-haiku-5-5.yaml --run-id judge-haiku \\
        --budget-usd 0.25 --launched-by owner
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.judge.build_set import RESULTS_ROOT, write_jsonl
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
from evals.text2sql.run import PredictionWriter, load_llm_config
from src.llm import LLMConfig

NO_REFERENCE = "no_reference"
ERROR = "error"


def run_judge(
    *,
    config: LLMConfig,
    items: list[dict[str, Any]],
    items_run: str,
    run_dir: Path,
    run_id: str,
    budget_usd: float | None = None,
    workers: int = 1,
    llm_config_name: str | None = None,
    client_factory: ClientFactory | None = None,
    extra_meta: dict[str, Any] | None = None,
    argv: list[str] | None = None,
    launched_by: str | None = None,
) -> dict[str, Any]:
    """Judge every item's answer (the run directory must be new) and write the outputs."""
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
        sink = PredictionWriter(run_dir / "judgements.jsonl")
        engine = Engine(workers)

        def case(item: dict[str, Any]) -> dict[str, Any]:
            record: dict[str, Any] = {
                "item_id": item["item_id"],
                "question_id": item["question_id"],
                "label": NO_REFERENCE,
                "rationale": "",
                "failure_type": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            }
            gold = item["gold_preview"]
            if gold.get("error"):
                record["rationale"] = f"gold query failed: {gold['error']}"
                return record
            verdict = None
            with timed_calls(client, run_id, "judge", item["question_id"]) as call:
                verdict = judge_answer(
                    item["question"], item["evidence"], item["answer"], gold, call.counting
                )
            usages = call.counting.usages
            if call.error is not None or verdict is None:
                record.update(label=ERROR, rationale=call.error or "", failure_type="exception")
            else:
                record.update(label=verdict.label, rationale=verdict.rationale)
            record.update(
                input_tokens=sum(u.input_tokens for u in usages),
                output_tokens=sum(u.output_tokens for u in usages),
                cache_read_tokens=sum(u.cache_read_tokens for u in usages),
                cache_write_tokens=sum(u.cache_write_tokens for u in usages),
                cost_usd=sum(u.cost_usd for u in usages),
                latency_ms=call.latency_ms,
            )
            return record

        try:
            engine.run_pass("judge", items, case, sink)
        except KeyboardInterrupt:
            print("Interrupted. Saving partial results.")
            engine.stop("interrupted")

        judgements = sorted(read_jsonl(run_dir / "judgements.jsonl"), key=lambda r: r["item_id"])
        write_jsonl(run_dir / "judgements.jsonl", judgements)
        labels = Counter(j["label"] for j in judgements)
        summary: dict[str, Any] = {
            "run_id": run_id,
            "kind": "judge",
            "items_run": items_run,
            "n_items": len(items),
            "n_judged": len(judgements),
            "labels": dict(labels),
            "n_unparsed": labels["unparsed"],
            "n_errors": labels[ERROR],
            "n_no_reference": labels[NO_REFERENCE],
            "totals": {
                "cost_usd": sum(j["cost_usd"] for j in judgements),
                "input_tokens": sum(j["input_tokens"] for j in judgements),
                "output_tokens": sum(j["output_tokens"] for j in judgements),
            },
            "prompt_chars": len(JUDGE_SYSTEM),
            "aborted": engine.reason,
            "tracker": {"cost_usd_this_invocation": tracker.total_cost, "budget_usd": budget_usd},
            "workers": workers,
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
    parser.add_argument("--items", type=Path, required=True, help="items run directory")
    parser.add_argument("--llm-config", type=Path, required=True, help="YAML in configs/llm/")
    parser.add_argument("--run-id", required=True, help="new folder under evals/results/judge/")
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
    items = read_jsonl(args.items / "items.jsonl")
    if not items:
        raise SystemExit(f"no items in {args.items / 'items.jsonl'}")
    run_dir = RESULTS_ROOT / args.run_id
    print(f"Run {args.run_id}: judging {len(items)} answers of {args.items.name}")
    try:
        with sigterm_as_interrupt():
            summary = run_judge(
                config=config,
                items=items,
                items_run=args.items.name,
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
    print(f"labels {summary['labels']}, cost ${summary['totals']['cost_usd']:.4f}")
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main()
