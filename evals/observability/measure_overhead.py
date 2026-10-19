"""Tracing overhead per question with the scripted client (no LLM, no cost).

Runs the smoke fixture (4 questions, systems ``baseline`` and ``agent_v2``) through
``evaluate_question`` repeatedly with tracing off (no-op tracer) and on (Langfuse, needs the
keys and a server). Passes alternate between the two modes so drift hits both equally. For
each mode it reports the mean and spread (sample standard deviation, min, max) of the time
per (question, system) pair on the calling thread, and for "on" the time ``flush`` needs to
deliver the events (the SDK sends them from a background thread). The scripted client answers
instantly, so this is the worst case: a real LLM call takes seconds. Usage:

    python -m evals.observability.measure_overhead [--passes 20]
"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from evals.text2sql.run import evaluate_question
from evals.text2sql.scorer import GoldCache
from evals.text2sql.smoke import ScriptedClient, build_fixture
from src.llm import UsageTracker
from src.observability import tracing
from src.observability.tracing import NoopTracer

SYSTEMS = ("baseline", "agent_v2")


def one_pass(trace: bool, run_id: str, root: Path) -> tuple[list[float], float]:
    """Times (ms) of every pair in one pass and the flush time (ms, 0 when off)."""
    databases_dir, questions = build_fixture(root)
    client = ScriptedClient(UsageTracker(keep_calls=False))
    gold = GoldCache(root / "gold.json")
    times: list[float] = []
    for system in SYSTEMS:
        for q in questions:
            start = time.perf_counter()
            evaluate_question(
                q,
                system,
                client,
                gold,
                run_id=run_id,
                databases_dir=databases_dir,
                timeout_s=30,
                skip_respond=True,
                trace=trace,
                split="dev",
            )
            times.append((time.perf_counter() - start) * 1000)
    flush_ms = 0.0
    if trace:
        start = time.perf_counter()
        tracing.flush()
        flush_ms = (time.perf_counter() - start) * 1000
    return times, flush_ms


def describe(label: str, values: list[float]) -> str:
    return (
        f"{label}: mean {statistics.fmean(values):.3f} ms, sd {statistics.stdev(values):.3f}, "
        f"min {min(values):.3f}, max {max(values):.3f} (n={len(values)})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passes", type=int, default=20, help="passes per mode")
    args = parser.parse_args(argv)
    if not tracing.enabled():
        raise SystemExit("set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and start Langfuse")
    live = tracing.get_tracer()
    off: list[float] = []
    on: list[float] = []
    flushes: list[float] = []
    one_pass(False, "warmup", Path(tempfile.mkdtemp(prefix="ovh-")))  # imports, caches
    session = f"overhead-{int(time.time())}"
    for i in range(args.passes):
        for mode in ("off", "on") if i % 2 == 0 else ("on", "off"):
            with tempfile.TemporaryDirectory(prefix="ovh-") as tmp:
                if mode == "off":
                    with tracing.use_tracer(NoopTracer()):
                        off += one_pass(False, session, Path(tmp))[0]
                else:
                    with tracing.use_tracer(live):
                        times, flush_ms = one_pass(True, session, Path(tmp))
                    on += times
                    flushes.append(flush_ms)
    pairs = len(SYSTEMS) * 4
    print(f"{args.passes} passes x {pairs} pairs per mode (alternating order)")
    print(describe("tracing off", off))
    print(describe("tracing on ", on))
    extra = [b - a for a, b in zip(off, on, strict=True)]
    print(describe("on - off (paired by position)", extra))
    print(describe("flush per pass (ms, delivery of ~8 traces)", flushes))
    print(f"flush per pair: mean {statistics.fmean(flushes) / pairs:.3f} ms")
    print(f"Langfuse session: {session}")


if __name__ == "__main__":
    main()
