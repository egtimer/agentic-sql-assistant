"""Check that a traced run reached Langfuse complete and well formed.

Reads the Langfuse public API (basic auth with the project keys) and verifies, per trace of
a session (the run id of ``python -m evals.text2sql.run --trace``):

- the expected node spans exist (agent v1: reason, respond; agent v2: introspect, generate,
  execute, respond, and one check span per reviewer verdict when the predictions are given);
- every observation belongs to the trace and is nested under it: the parent of each one is
  another observation of the same trace, and agent systems have exactly one root span;
- the number of generations equals the LLM calls recorded for that question in
  ``predictions.jsonl`` and every generation has a model, token usage and a cost;
- the scores ``execution_accuracy`` and ``cost_usd`` exist (and match the prediction record).

Usage:

    python -m evals.observability.verify_traces --session <run id> \\
        [--predictions evals/results/text2sql/<run id>] [--out report.json]

Keys and host come from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``
(environment or ``.env``). Writes a JSON report (default: ``trace_report.json`` in the
predictions directory, else the current directory), prints a summary, exits 1 on failures.
"""

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from src.config import settings

PAGE_SIZE = 100
COST_TOLERANCE = 1e-9
V1_SYSTEMS = frozenset({"agent", "agent_v1"})
V2_SYSTEMS = frozenset({"agent_v2", "agent_v2_nocheck"})
ROOT_SPANS = {"agent_v1": "agent_query", "agent_v2": "sql_agent_v2"}

Fetch = Callable[[str, dict[str, Any]], dict[str, Any]]


def make_fetch(host: str, public_key: str, secret_key: str) -> Fetch:
    """GET helper for the Langfuse public API."""
    client = httpx.Client(base_url=host.rstrip("/"), auth=(public_key, secret_key), timeout=30)

    def fetch(path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = client.get(path, params=params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    return fetch


def fetch_all(fetch: Fetch, path: str, **params: Any) -> list[dict[str, Any]]:
    """All rows of a paginated list endpoint."""
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        body = fetch(path, {**params, "page": page, "limit": PAGE_SIZE})
        rows.extend(body["data"])
        if page >= int(body.get("meta", {}).get("totalPages", 1)):
            return rows
        page += 1


def load_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    """predictions.jsonl records that have a trace id, keyed by trace id."""
    path = run_dir / "predictions.jsonl"
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("trace_id"):
                records[record["trace_id"]] = record
    return records


def system_of(trace: dict[str, Any], record: dict[str, Any] | None) -> str:
    if record is not None:
        return str(record["system"])
    name = str(trace.get("name") or "")
    return name.split("/", 1)[1] if name.startswith("text2sql/") else name


def check_trace(
    trace: dict[str, Any],
    observations: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    record: dict[str, Any] | None,
    *,
    skip_respond: bool | None = None,
) -> dict[str, Any]:
    """Run every check on one trace; returns its report entry (``problems`` empty = ok)."""
    problems: list[str] = []
    system = system_of(trace, record)
    tid = trace["id"]
    spans = [o for o in observations if o["type"] == "SPAN"]
    generations = [o for o in observations if o["type"] == "GENERATION"]
    by_id = {o["id"]: o for o in observations}
    span_counts = Counter(o["name"] for o in spans)

    # Nesting: same trace, parents exist, a single root for the agent systems.
    for o in observations:
        if o.get("traceId") not in (None, tid):
            problems.append(f"{o['type'].lower()} {o['name']!r} belongs to another trace")
        parent = o.get("parentObservationId")
        if parent is not None and parent not in by_id:
            problems.append(f"{o['type'].lower()} {o['name']!r} has an unknown parent")
    roots = [o for o in observations if o.get("parentObservationId") is None]
    agent = "agent_v1" if system in V1_SYSTEMS else "agent_v2" if system in V2_SYSTEMS else None
    if agent is not None:
        if [r["name"] for r in roots] != [ROOT_SPANS[agent]]:
            problems.append(
                f"expected one root span {ROOT_SPANS[agent]!r}, found {[r['name'] for r in roots]}"
            )
        else:
            for o in observations:
                if o is not roots[0] and not _under(o, roots[0]["id"], by_id):
                    problems.append(f"{o['name']!r} is not nested under the root span")
    elif spans:
        problems.append(f"{system} should have no node spans, found {sorted(span_counts)}")

    # Expected node spans.
    exception = record is not None and record.get("failure_type") == "exception"
    expected: Counter[str] = Counter()
    if agent == "agent_v1":
        expected.update({"reason": 1})
        if skip_respond is False:
            expected.update({"respond": 1})
    elif agent == "agent_v2":
        expected.update({"introspect": 1, "generate": 1, "execute": 1, "respond": 1})
        if record is not None and "verdicts" in record:
            expected["execute"] = len(record["attempts"])
            expected["check"] = len(record["verdicts"])
    if not exception:
        for name, n in expected.items():
            got = span_counts.get(name, 0)
            exact = agent == "agent_v2" and record is not None and name in ("execute", "check")
            if (exact and got != n) or got < n:
                problems.append(f"span {name!r}: found {got}, expected {n}")

    # Generations: count against the recorded LLM calls, then usage and cost on each.
    if record is not None and not exception and len(generations) != record["n_llm_calls"]:
        problems.append(
            f"generations: found {len(generations)}, predictions record {record['n_llm_calls']} "
            "LLM calls"
        )
    for g in generations:
        usage = g.get("usage") or {}
        if not g.get("model"):
            problems.append(f"generation {g['name']!r} has no model")
        if not usage.get("input") or usage.get("output") is None or not usage.get("total"):
            problems.append(f"generation {g['name']!r} has no token usage: {usage}")
        if g.get("calculatedTotalCost") is None:
            problems.append(f"generation {g['name']!r} has no cost")
    cost = sum(g.get("calculatedTotalCost") or 0.0 for g in generations)
    if record is not None and not exception and abs(cost - record["cost_usd"]) > COST_TOLERANCE:
        problems.append(f"generation costs sum to {cost}, predictions record {record['cost_usd']}")

    # Scores.
    by_name = {s["name"]: s for s in scores}
    for name in ("execution_accuracy", "cost_usd"):
        if name not in by_name:
            problems.append(f"score {name!r} is missing")
    accuracy = by_name.get("execution_accuracy")
    wanted = None if record is None else (1.0 if record["correct"] else 0.0)
    if accuracy is not None and wanted is not None and accuracy["value"] != wanted:
        problems.append("execution_accuracy score differs from the prediction record")

    return {
        "trace_id": tid,
        "system": system,
        "question_id": None if record is None else record["question_id"],
        "spans": dict(sorted(span_counts.items())),
        "generations": len(generations),
        "generation_cost_usd": cost,
        "scores": {name: s["value"] for name, s in sorted(by_name.items())},
        "ok": not problems,
        "problems": problems,
    }


def _under(obs: dict[str, Any], root_id: str, by_id: dict[str, dict[str, Any]]) -> bool:
    """True when ``root_id`` is among the ancestors of ``obs``."""
    seen: set[str] = set()
    parent = obs.get("parentObservationId")
    while parent is not None and parent not in seen:
        if parent == root_id:
            return True
        seen.add(parent)
        parent = by_id.get(parent, {}).get("parentObservationId")
    return False


def verify(
    fetch: Fetch,
    session: str,
    predictions_dir: Path | None = None,
    *,
    wait_s: float = 30.0,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    """Fetch the session from Langfuse and check every trace. Returns the report."""
    records = load_predictions(predictions_dir) if predictions_dir else {}
    skip_respond: bool | None = None
    if predictions_dir and (predictions_dir / "summary.json").exists():
        summary = json.loads((predictions_dir / "summary.json").read_text(encoding="utf-8"))
        skip_respond = bool(summary.get("skip_respond"))

    deadline = time.monotonic() + wait_s
    while True:  # ingestion is asynchronous: wait for the traces (and their content)
        traces = fetch_all(fetch, "/api/public/traces", sessionId=session)
        found = {t["id"] for t in traces}
        complete = bool(traces) and (not records or set(records) <= found)
        if complete or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)

    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    if not traces:
        problems.append(f"no traces found for session {session!r}")
    for tid in sorted(set(records) - found):
        problems.append(f"trace {tid} of question {records[tid]['question_id']} not found")
    for trace in sorted(traces, key=lambda t: t["id"]):
        record = records.get(trace["id"])
        if records and record is None:
            continue  # a retried pair left an older trace behind: not part of this run
        while True:
            # The detail endpoint returns the trace with its observations and scores. On
            # server 2.95 the traceId filter of /api/public/scores is ignored (it lists the
            # whole project), so the list endpoints are not used for them.
            detail = fetch(f"/api/public/traces/{trace['id']}", {})
            entry = check_trace(
                trace,
                detail.get("observations", []),
                detail.get("scores", []),
                record,
                skip_respond=skip_respond,
            )
            if entry["ok"] or time.monotonic() >= deadline:
                break
            time.sleep(poll_s)
        entries.append(entry)
    n_ok = sum(1 for e in entries if e["ok"])
    return {
        "session": session,
        "n_traces_found": len(traces),
        "n_traces_expected": len(records) or None,
        "n_checked": len(entries),
        "n_ok": n_ok,
        "passed": not problems and n_ok == len(entries) and bool(entries),
        "problems": problems,
        "totals": {
            "generations": sum(e["generations"] for e in entries),
            "generation_cost_usd": sum(e["generation_cost_usd"] for e in entries),
            "by_system": dict(Counter(e["system"] for e in entries)),
        },
        "traces": entries,
    }


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"session {report['session']}: {report['n_traces_found']} traces found"
        + (
            f" ({report['n_traces_expected']} expected)"
            if report["n_traces_expected"] is not None
            else ""
        )
        + f", {report['n_ok']}/{report['n_checked']} traces ok",
        f"  by system: {report['totals']['by_system']}; "
        f"{report['totals']['generations']} generations, "
        f"cost ${report['totals']['generation_cost_usd']:.6f}",
    ]
    lines += [f"  PROBLEM: {p}" for p in report["problems"]]
    for e in report["traces"]:
        if not e["ok"]:
            lines.append(f"  trace {e['trace_id']} ({e['system']}, q{e['question_id']}):")
            lines += [f"    - {p}" for p in e["problems"]]
    lines.append("RESULT: " + ("PASSED" if report["passed"] else "FAILED"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", required=True, help="Langfuse session id (= run id)")
    parser.add_argument("--predictions", type=Path, help="run dir with predictions.jsonl")
    parser.add_argument("--out", type=Path, help="JSON report path")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds to wait for ingestion")
    parser.add_argument("--host", default=settings.langfuse_host)
    args = parser.parse_args(argv)
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        raise SystemExit("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (environment or .env)")
    fetch = make_fetch(args.host, settings.langfuse_public_key, settings.langfuse_secret_key)
    try:
        report = verify(fetch, args.session, args.predictions, wait_s=args.wait)
    except httpx.HTTPError as e:
        raise SystemExit(f"cannot query Langfuse at {args.host}: {e}") from e
    out = args.out or (args.predictions or Path.cwd()) / "trace_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_summary(report))
    print(f"report: {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
