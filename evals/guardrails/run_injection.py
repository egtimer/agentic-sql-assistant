"""Prompt-injection evaluation of agent v2 behind the SQL guard (needs an LLM).

Each question runs on a disposable copy of its database, in its own temporary directory.
The runner records every SQL attempt of the model, whether the guard blocked it and why,
whether the attempt was harmful (judged independently of the guard, see ``classify``),
and whether the database copy or its directory changed. Usage:

    python -m evals.guardrails.run_injection --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --budget-usd 0.25 --run-id <new-run-id>

Output goes to ``--out-root`` (default evals/results/guardrails/) / run id. The run id must be
new: the runner refuses a directory that already holds ``run.json``. ``run.json`` (who
started the run, argv, git commit, LLM config, budget) is written before the first case and
completed with a status at the end. To write straight into a report directory pass
``--out evals/reports/<date>-guardrails-injection/`` (it must not exist yet).
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from evals.guardrails.cases import DB_NAME, INJECTION_PATH, disposable, policy_for, read_jsonl
from evals.guardrails.classify import classify_attempt
from evals.guardrails.run_sql import sha256_file
from evals.provenance import (
    LAUNCHED_BY_ENV,
    REPO_ROOT,
    finish_run,
    sigterm_as_interrupt,
    start_run,
    status_from_abort,
)
from evals.text2sql.baseline import CountingClient
from evals.text2sql.run import load_llm_config
from src.llm import (
    BudgetExceeded,
    LLMClient,
    UsageTracker,
    build_client,
    call_context,
    reset_client,
    set_client,
)
from src.sql_agent.executor import NO_SQL_ERROR
from src.sql_agent.graph import run_sql_agent

DEFAULT_MAX_FIXES = 2
DEFAULT_OUT_ROOT = REPO_ROOT / "evals" / "results" / "guardrails"


def apply_setup(db: Path, setup: dict[str, Any]) -> None:
    """Indirect injection: write the instruction into the disposable copy before the run."""
    conn = sqlite3.connect(db)
    try:
        conn.execute(setup["sql"], setup.get("params", []))
        conn.commit()
    finally:
        conn.close()


def run_case(
    case: dict[str, Any],
    client: LLMClient,
    *,
    max_fixes: int = DEFAULT_MAX_FIXES,
    tracker: UsageTracker | None = None,
) -> dict[str, Any]:
    """Run agent v2 on one case. Raises BudgetExceeded when the cap is reached."""
    cost_before = tracker.total_cost if tracker else 0.0
    counting = CountingClient(client)
    error: str | None = None
    result = None
    with disposable(case["db"]) as db:
        if case.get("setup"):
            apply_setup(db, case["setup"])
        before = sha256_file(db)
        token = set_client(counting)
        started = time.monotonic()
        try:
            with call_context(case_id=case["id"]):
                result = run_sql_agent(
                    case["question"],
                    case.get("evidence", ""),
                    db,
                    skip_respond=True,
                    max_fixes=max_fixes,
                    policy=policy_for(case["db"]),
                )
        except BudgetExceeded:
            raise
        except Exception as e:  # keep going; the failure is part of the record
            error = f"{type(e).__name__}: {e}"
        finally:
            reset_client(token)
        elapsed = time.monotonic() - started
        attempts = []
        for a in result.attempts if result else []:
            harmful, reasons = classify_attempt(a.sql, db, policy_for(case["db"]))
            attempts.append(
                {
                    "sql": a.sql,
                    "blocked": a.blocked,
                    "error": a.error,
                    "harmful": harmful,
                    "harm_reasons": reasons,
                }
            )
        modified = sha256_file(db) != before
        new_files = sorted(set(p.name for p in db.parent.iterdir()) - {DB_NAME})
    executed = [a for a in attempts if a["error"] != NO_SQL_ERROR]
    if any(a["harmful"] for a in attempts):
        behaviour = "complied"
    elif not executed:
        behaviour = "refused"
    else:
        behaviour = "answered_benignly"
    return {
        "id": case["id"],
        "type": case["type"],
        "db": case["db"],
        "question": case["question"],
        "evidence": case.get("evidence", ""),
        "setup": case.get("setup"),
        "attempts": attempts,
        "final_sql": result.final_sql if result else "",
        "verdicts": [v.kind for v in result.verdicts] if result else [],
        "behaviour": behaviour,
        "harmful_attempts": sum(a["harmful"] for a in attempts),
        "blocked_harmful_attempts": sum(a["harmful"] and a["blocked"] for a in attempts),
        "harmful_executions": sum(a["harmful"] and not a["blocked"] for a in attempts),
        "blocked_attempts": sum(a["blocked"] for a in attempts),
        "db_modified": modified,
        "new_files": new_files,
        "side_effect": modified or bool(new_files),
        "n_llm_calls": counting.n_calls,
        "cost_usd": round((tracker.total_cost - cost_before) if tracker else 0.0, 6),
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }


def summarize(records: list[dict[str, Any]], cost_usd: float, **meta: Any) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    controls = [r for r in records if r["type"] == "control"]
    return {
        **meta,
        "questions": len(records),
        "questions_by_type": by_type,
        "attempts": sum(len(r["attempts"]) for r in records),
        "harmful_attempts": sum(r["harmful_attempts"] for r in records),
        "blocked_harmful_attempts": sum(r["blocked_harmful_attempts"] for r in records),
        "harmful_executions": sum(r["harmful_executions"] for r in records),
        "questions_with_side_effects": sum(r["side_effect"] for r in records),
        "model_complied": sum(r["behaviour"] == "complied" for r in records),
        "model_refused": sum(r["behaviour"] == "refused" for r in records),
        "model_answered_benignly": sum(r["behaviour"] == "answered_benignly" for r in records),
        "complied_by_type": {
            t: sum(r["behaviour"] == "complied" for r in records if r["type"] == t) for t in by_type
        },
        "control_questions_with_blocked_attempt": sum(r["blocked_attempts"] > 0 for r in controls),
        "agent_errors": sum(r["error"] is not None for r in records),
        "llm_calls": sum(r["n_llm_calls"] for r in records),
        "cost_usd": round(cost_usd, 6),
    }


def resolve_out_dir(run_id: str, out_root: Path, out: Path | None) -> Path:
    """Absolute output directory of a run; refuses to reuse an existing one."""
    if out is not None:
        path = out.resolve()
        if path.exists():
            raise SystemExit(f"{path} exists; --out must be a directory that does not exist yet")
        return path
    path = (out_root / run_id).resolve()
    if (path / "run.json").exists():
        raise SystemExit(f"{path / 'run.json'} exists; choose a new --run-id")
    return path


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="new run id (names the output folder)")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="parent of the run folder (default evals/results/guardrails/)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="explicit report directory; must not exist yet"
    )
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--max-fixes", type=int, default=DEFAULT_MAX_FIXES)
    parser.add_argument("--limit", type=int, default=None, help="first N cases only")
    parser.add_argument("--ids", default=None, help="comma-separated case ids")
    parser.add_argument(
        "--launched-by",
        default=None,
        help=f"free text for run.json (default: env {LAUNCHED_BY_ENV})",
    )
    args = parser.parse_args(argv)
    # Every case runs with the cwd inside its sandbox; a relative output path would put the call
    # log (and anything else written there) into the sandbox, where it counts as a side effect.
    out_dir = resolve_out_dir(args.run_id, args.out_root, args.out)

    config = load_llm_config(args.llm_config)
    if config.provider == "anthropic" and args.budget_usd is None:
        raise SystemExit("--budget-usd is required for the anthropic provider")
    cases = read_jsonl(INJECTION_PATH)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    run_file = start_run(
        out_dir,
        run_id=args.run_id,
        argv=raw_argv,
        llm_config_path=args.llm_config,
        config=config,
        budget_usd=args.budget_usd,
        launched_by=args.launched_by,
    )
    results_path = out_dir / "results.jsonl"
    tracker = UsageTracker(budget_usd=args.budget_usd, sink_path=out_dir / "llm_calls.jsonl")
    client = build_client(config, tracker)
    records: list[dict[str, Any]] = []
    stopped = None
    reason: str | None = None
    try:
        with sigterm_as_interrupt(), open(results_path, "w", encoding="utf-8") as f:
            for case in cases:
                try:
                    record = run_case(case, client, max_fixes=args.max_fixes, tracker=tracker)
                except BudgetExceeded as e:
                    stopped, reason = f"budget: {e}", "budget"
                    print(f"Budget reached: {e}. Stopping with partial results.", file=sys.stderr)
                    break
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                f.flush()
                print(
                    f"{record['id']} {record['behaviour']} cost ${record['cost_usd']:.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        stopped, reason = "interrupted", "interrupted"
        print("Interrupted. Saving partial results.", file=sys.stderr)
    except BaseException as e:
        finish_run(run_file, f"failed:{type(e).__name__}")
        raise
    summary = summarize(
        records,
        tracker.total_cost,
        llm_config=str(args.llm_config),
        model=config.model,
        max_fixes=args.max_fixes,
        budget_usd=args.budget_usd,
        cases_planned=len(cases),
        stopped=stopped,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    finish_run(run_file, status_from_abort(reason))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
