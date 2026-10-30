"""Run agent v1 and its baselines on the tool-selection set, and score them. Spends money.

Systems:

- ``agent_v1``       the agent the API serves (default profile, its tool-selection prompt)
- ``simple_prompt``  the same graph with a minimal tool-selection prompt: tool names, one
                     line each and the table columns; no usage hints, no examples
- ``no_tools``       one LLM call that answers without data or documents

Each answered query is appended to ``predictions.jsonl`` at once (``--resume`` continues
a run). ``--budget-usd`` is a hard cap checked before every LLM call. At the end
``summary.json`` holds, per system and per category, tool precision, recall and exact
match, answer accuracy, cost, latency and LLM calls with 95% bootstrap CIs, and paired
differences of each baseline against ``agent_v1``. Needs Qdrant with the documents
ingested (``make up``, ``make ingest``) and the seeded demo database (``make seed``).

    uv run python -m evals.tools.run --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --split dev --budget-usd 0.15 --run-id <new id>
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from evals.provenance import git_info
from evals.text2sql.run import load_llm_config
from evals.text2sql.stats import paired_bootstrap
from evals.tools.metrics import aggregate, check_answer, tool_scores
from src.agent import prompts
from src.agent.graph import run_agent
from src.config import settings
from src.llm import (
    BudgetExceeded,
    UsageTracker,
    build_client,
    call_context,
    reset_client,
    set_client,
)
from src.rag.embedding import embedding_runtime, index_runtime

SET = Path(__file__).resolve().parent / "set" / "queries.jsonl"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
SYSTEMS = ("agent_v1", "simple_prompt", "no_tools")

SIMPLE_TOOL_SELECTION_PROMPT = """\
Tools:
- sql_query: runs a SQLite SELECT on tables products (id, name, category, price), sales (id, \
product_id, customer_id, quantity, amount, date, quarter, region), customers (id, name, region, \
tier)
- doc_search: searches the company policy documents
- calculator: evaluates a math expression

Reply with a JSON array of {{"tool", "input", "reason"}} objects, or [] if no tool is needed.

User question: {query}"""

NO_TOOLS_SYSTEM = (
    "You are a business intelligence assistant for a company that sells electronics, office "
    "products and software. You have no access to its sales data or policy documents. Answer "
    "the question as well as you can, in the language of the question."
)


@contextmanager
def tool_selection_prompt(template: str) -> Iterator[None]:
    """Use ``template`` as the default profile's tool-selection prompt inside the block."""
    original = prompts.TOOL_SELECTION_PROMPT
    prompts.TOOL_SELECTION_PROMPT = template
    try:
        yield
    finally:
        prompts.TOOL_SELECTION_PROMPT = original


def answer(system: str, query: str, client: Any) -> tuple[str, list[str]]:
    """The final answer and the tools executed, in order."""
    if system == "no_tools":
        return client.complete(NO_TOOLS_SYSTEM, query, tag="answer").text, []
    if system == "simple_prompt":
        with tool_selection_prompt(SIMPLE_TOOL_SELECTION_PROMPT):
            result = run_agent(query)
    else:
        result = run_agent(query)
    return result["response"], list(result["tools_used"])


def score(q: dict[str, Any], system: str, response: str, tools: list[str]) -> dict[str, Any]:
    return {
        "id": q["id"],
        "category": q["category"],
        "system": system,
        "tools_executed": tools,
        **tool_scores(tools, q["required"], q["optional"]),
        "answer_correct": check_answer(response, q["checks"]),
        "response": response,
    }


def run_query(
    q: dict[str, Any], system: str, client: Any, tracker: UsageTracker, run_id: str
) -> dict[str, Any]:
    before = len(tracker.calls)
    token = set_client(client)
    start = time.perf_counter()
    error = None
    try:
        with call_context(run_id=run_id, system=system, query_id=q["id"]):
            response, tools = answer(system, q["query"], client)
    except BudgetExceeded:
        raise
    except Exception as e:  # recorded, the run goes on
        response, tools, error = "", [], f"{type(e).__name__}: {e}"
    finally:
        reset_client(token)
    calls = tracker.calls[before:]
    return {
        **score(q, system, response, tools),
        "error": error,
        "n_llm_calls": len(calls),
        "cost_usd": sum(c.cost_usd for c in calls),
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "latency_s": time.perf_counter() - start,
    }


def summarize(rows: list[dict[str, Any]], systems: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"systems": {}, "paired_vs_agent_v1": {}}
    for system in systems:
        mine = [r for r in rows if r["system"] == system]
        categories = sorted({r["category"] for r in mine})
        out["systems"][system] = {
            "all": aggregate(mine),
            "by_category": {
                c: aggregate([r for r in mine if r["category"] == c]) for c in categories
            },
        }
    base = {r["id"]: r for r in rows if r["system"] == "agent_v1"}
    for system in systems:
        if system == "agent_v1" or not base:
            continue
        other = {r["id"]: r for r in rows if r["system"] == system}
        out["paired_vs_agent_v1"][system] = {
            metric: paired_bootstrap(
                {i: bool(r[metric]) for i, r in other.items() if r[metric] is not None},
                {i: bool(r[metric]) for i, r in base.items() if r[metric] is not None},
            )
            for metric in ("exact", "answer_correct")
        }
    return out


def provenance(config_path: Path, argv: list[str]) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    db = Path(settings.sqlite_db_path)
    client = QdrantClient(url=settings.qdrant_url)
    packages = {}
    for pkg in ("anthropic", "langgraph", "qdrant-client", "sentence-transformers"):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None
    return {
        "command": " ".join(argv),
        "git": git_info(),
        "llm_config_file": str(config_path),
        "set_sha256": hashlib.sha256(SET.read_bytes()).hexdigest(),
        "demo_db_sha256": hashlib.sha256(db.read_bytes()).hexdigest(),
        "embedding_runtime": embedding_runtime(),
        "index_runtime": index_runtime(client, settings.qdrant_collection),
        "qdrant_server": client.info().version,
        "packages": packages,
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv if argv is None else ["run.py", *argv])
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if unknown := set(systems) - set(SYSTEMS):
        parser.error(f"unknown systems: {sorted(unknown)}")

    run_dir = REPORTS / args.run_id
    if run_dir.exists() and not args.resume:
        print(f"error: {run_dir} exists (pass --resume to continue it)", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_llm_config(args.llm_config)
    queries = [q for q in map(json.loads, SET.read_text().splitlines()) if q["split"] == args.split]

    pred_path = run_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {(r["system"], r["id"]) for r in map(json.loads, pred_path.read_text().splitlines())}
    run_json = run_dir / "run.json"
    if not run_json.exists():
        meta = {
            "run_id": args.run_id,
            "split": args.split,
            "systems": systems,
            "budget_usd": args.budget_usd,
            "started_at": datetime.now(UTC).isoformat(),
            "llm_config": config.model_dump(),
            "provenance": provenance(args.llm_config, raw_argv),
        }
        run_json.write_text(json.dumps(meta, indent=2) + "\n")

    tracker = UsageTracker(budget_usd=args.budget_usd, sink_path=run_dir / "llm_calls.jsonl")
    client = build_client(config, tracker)
    stopped = None
    try:
        for system in systems:
            for q in queries:
                if (system, q["id"]) in done:
                    continue
                row = run_query(q, system, client, tracker, args.run_id)
                with open(pred_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = f"exact={row['exact']} answer={row['answer_correct']}"
                print(f"{system:<14} {q['id']:<10} {status} ${row['cost_usd']:.5f}", flush=True)
    except BudgetExceeded as e:
        stopped = str(e)
        print(f"stopped: {e}", file=sys.stderr)

    rows = [json.loads(line) for line in pred_path.read_text().splitlines()]
    summary = {
        "run_id": args.run_id,
        "split": args.split,
        "n_queries": len(queries),
        "stopped": stopped,
        "cost_usd_this_invocation": tracker.total_cost,
        "cost_usd_total": sum(r["cost_usd"] for r in rows),
        "finished_at": datetime.now(UTC).isoformat(),
        **summarize(rows, systems),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for system in systems:
        a = summary["systems"][system]["all"]

        def fmt(stat: dict[str, float] | None, pct: bool = True) -> str:
            if stat is None:
                return "n/a"
            k = 100 if pct else 1
            return f"{k * stat['mean']:.1f} [{k * stat['ci_low']:.1f}, {k * stat['ci_high']:.1f}]"

        print(f"{system:<14} exact {fmt(a['tool_exact'])}  precision {fmt(a['tool_precision'])}  "
              f"answer {fmt(a['answer_correct'])}  cost ${a['cost_usd']['mean']:.5f}  "
              f"latency {fmt(a['latency_s'], pct=False)} s")  # fmt: skip
    print(f"cost of this run so far: ${summary['cost_usd_total']:.4f}")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
