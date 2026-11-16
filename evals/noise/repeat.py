"""Run an existing eval N times with nothing changed, to measure run-to-run noise. Spends money.

Each repetition is a separate invocation of the eval's own runner, in a subprocess, with
its own run id (``<prefix>-r01`` ...) and its own ``--budget-usd`` hard cap. Nothing else
differs between repetitions: same config, split, systems, sample and code. A repetition
whose summary already exists and was not stopped is skipped, so the command can be
re-issued after an interruption (an interrupted repetition is continued with
``--resume``). The loop stops at the first repetition that hits its budget cap.

    uv run python -m evals.noise.repeat --eval tools --n 10 --prefix <id> \\
        --budget-usd 0.03 -- --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --split test --systems agent_v1,no_tools

    uv run python -m evals.noise.repeat --eval text2sql --n 5 --prefix <id> \\
        --budget-usd 0.30 -- --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --system baseline,agent_v2 --split test --skip-respond --workers 4

Writes ``repeat.json`` (commands, run ids, cost and status of each repetition) to
``evals/results/noise/<prefix>/``.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
NOISE_ROOT = REPO_ROOT / "evals" / "results" / "noise"
NEAR_CAP = 0.8
MIN_BUDGET_USD = 0.001
# An API failure as recorded in a prediction's ``error`` before the runners raised
# ProviderError (a missing answer that was scored as a wrong one).
PROVIDER_ERROR_PATTERN = re.compile(
    r"ProviderError|BadRequestError|APIStatusError|APIConnectionError|APITimeoutError|"
    r"RateLimitError|InternalServerError|OverloadedError|Error code: \d{3}"
)
RESUMABLE = ("partial", "provider_error", "interrupted")


@dataclass(frozen=True)
class EvalKind:
    module: str
    out_root: Path
    stop_key: str  # summary.json key that is non-empty when the run was cut short

    def run_dir(self, run_id: str) -> Path:
        return self.out_root / run_id


EVALS = {
    "tools": EvalKind("evals.tools.run", REPO_ROOT / "evals" / "reports", "stopped"),
    "text2sql": EvalKind(
        "evals.text2sql.run", REPO_ROOT / "evals" / "results" / "text2sql", "aborted"
    ),
}


def run_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}-r{i:02d}" for i in range(1, n + 1)]


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def spent_usd(run_dir: Path) -> float:
    """What a run directory has cost so far, over all its invocations."""
    path = run_dir / "llm_calls.jsonl"
    if not path.exists():
        return 0.0
    return sum(
        float(json.loads(line)["cost_usd"]) for line in path.read_text().splitlines() if line
    )


def provider_error_rows(run_dir: Path) -> int:
    """Predictions whose recorded error is an API failure (older runs scored them)."""
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return 0
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return sum(1 for r in rows if PROVIDER_ERROR_PATTERN.search(str(r.get("error") or "")))


def status_of(kind: EvalKind, run_dir: Path) -> str:
    """One of:

    - ``new``: nothing there yet;
    - ``partial``, ``interrupted``, ``provider_error``: cut short by something that is not
      the run's fault; continued with --resume;
    - ``stopped``: hit its budget cap or aborted on repeated agent errors; left alone;
    - ``contaminated``: finished, but some predictions are API failures scored as wrong
      answers; must be moved aside and run again, never analysed;
    - ``done``.
    """
    summary = read_summary(run_dir)
    if summary is None:
        return "partial" if run_dir.exists() else "new"
    reason = str(summary.get(kind.stop_key) or "")
    if reason.startswith("provider_error"):
        return "provider_error"
    if reason == "interrupted":
        return "interrupted"
    if reason:
        return "stopped"
    return "contaminated" if provider_error_rows(run_dir) else "done"


def command(
    kind: EvalKind, run_id: str, budget: float, passthrough: list[str], resume: bool
) -> list[str]:
    cmd = [sys.executable, "-m", kind.module, *passthrough]
    cmd += ["--run-id", run_id, "--budget-usd", str(budget)]
    return [*cmd, "--resume"] if resume else cmd


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    own, passthrough = (
        (raw[: raw.index("--")], raw[raw.index("--") + 1 :]) if "--" in raw else (raw, [])
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval", choices=sorted(EVALS), required=True)
    parser.add_argument("--n", type=int, required=True, help="number of repetitions")
    parser.add_argument("--prefix", required=True, help="run ids are <prefix>-r01 ...")
    parser.add_argument("--budget-usd", type=float, required=True, help="cap per repetition")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args(own)
    if args.n < 1 or args.budget_usd <= 0:
        parser.error("--n must be >= 1 and --budget-usd > 0")
    if any(a in passthrough for a in ("--run-id", "--budget-usd", "--resume")):
        parser.error("--run-id, --budget-usd and --resume are set by this script")
    kind = EVALS[args.eval]

    manifest_dir = NOISE_ROOT / args.prefix
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "repeat.json"
    manifest: dict[str, Any] = {
        "eval": args.eval,
        "n": args.n,
        "prefix": args.prefix,
        "budget_usd_per_run": args.budget_usd,
        "passthrough": passthrough,
        "runs": [],
    }
    exit_code = 0
    for run_id in run_ids(args.prefix, args.n):
        run_dir = kind.run_dir(run_id)
        before = status_of(kind, run_dir)
        entry: dict[str, Any] = {"run_id": run_id, "run_dir": str(run_dir.relative_to(REPO_ROOT))}
        if before == "done":
            entry["status"] = "done (skipped)"
        elif before in ("stopped", "contaminated"):
            entry["status"] = f"{before} earlier"
            exit_code = 1
        else:
            remaining = args.budget_usd - spent_usd(run_dir)
            if remaining < MIN_BUDGET_USD:
                entry["status"] = "cap already spent"
                manifest["runs"].append(entry)
                exit_code = 1
                break
            cmd = command(
                kind, run_id, round(remaining, 6), passthrough, resume=before in RESUMABLE
            )
            entry["command"] = " ".join(cmd[1:])
            if args.dry_run:
                print(" ".join(cmd))
                entry["status"] = "dry run"
            else:
                entry["started_at"] = datetime.now(UTC).isoformat()
                entry["returncode"] = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
                entry["finished_at"] = datetime.now(UTC).isoformat()
                entry["status"] = status_of(kind, run_dir)
        summary = read_summary(run_dir)
        if summary is not None:
            cost = spent_usd(run_dir)
            entry["cost_usd"] = cost
            entry["share_of_cap"] = cost / args.budget_usd
            flag = "  NEAR CAP" if cost >= NEAR_CAP * args.budget_usd else ""
            print(f"{run_id}: {entry['status']}, ${cost:.4f} of ${args.budget_usd}{flag}")
        manifest["runs"].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if entry["status"] not in ("done", "done (skipped)", "dry run"):
            print(f"{run_id} did not finish ({entry['status']}); stopping", file=sys.stderr)
            exit_code = 1
            break
    total = sum(r.get("cost_usd", 0.0) for r in manifest["runs"])
    manifest["cost_usd_total"] = total
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"total ${total:.4f} over {len(manifest['runs'])} repetitions; {manifest_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
