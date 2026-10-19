"""Provenance of eval runs that can spend money: ``run.json`` is written before the first case.

The file records who started the run (argv, user, host, pid, parent process and its command
line, git commit, LLM config and budget) so a partial output can always be traced, and is
updated when the run ends with a status: ``completed``, ``aborted:budget``,
``aborted:errors``, ``interrupted`` or ``failed:<ExceptionType>``.
"""

import getpass
import json
import os
import platform
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from src.llm import LLMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = "run.json"
LAUNCHED_BY_ENV = "EVAL_LAUNCHED_BY"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def git_info() -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def _user() -> str | None:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return os.environ.get("USER")


def parent_cmdline(ppid: int) -> str | None:
    """Command line of process ``ppid`` from ``/proc`` (None when unavailable)."""
    try:
        raw = Path(f"/proc/{ppid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip() or None


def resolve_launched_by(flag: str | None) -> str | None:
    return flag or os.environ.get(LAUNCHED_BY_ENV) or None


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def start_run(
    out_dir: Path,
    *,
    run_id: str,
    argv: list[str] | None,
    llm_config_path: Path | str | None,
    config: LLMConfig,
    budget_usd: float | None,
    launched_by: str | None = None,
    resume: bool = False,
) -> Path:
    """Write ``run.json`` in ``out_dir`` (created if needed) and return its path.

    A fresh run refuses to start when ``run.json`` already exists. With ``resume`` an
    existing file is kept and a resume entry (time, argv, pid, ...) is appended to it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RUN_FILE
    ppid = os.getppid()
    argv = list(sys.argv if argv is None else argv)
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use a new run id (or --resume)")
    entry = {
        "resumed_at": now_utc(),
        "argv": argv,
        "pid": os.getpid(),
        "ppid": ppid,
        "parent_cmdline": parent_cmdline(ppid),
        "launched_by": resolve_launched_by(launched_by),
    }
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("resumes", []).append(entry)
        data["status"] = "running"
        data["finished_at"] = None
    else:
        data = {
            "run_id": run_id,
            "argv": argv,
            "started_at": now_utc(),
            "finished_at": None,
            "status": "running",
            "git": git_info(),
            "user": _user(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "ppid": ppid,
            "parent_cmdline": parent_cmdline(ppid),
            "python": platform.python_version(),
            "llm_config": {
                "path": None if llm_config_path is None else str(llm_config_path),
                "provider": config.provider,
                "model": config.model,
                "budget_usd": budget_usd,
            },
            "launched_by": resolve_launched_by(launched_by),
            "resumes": [entry] if resume else [],
        }
    _write(path, data)
    return path


def finish_run(path: Path, status: str) -> None:
    """Record the end time and final ``status`` in ``run.json``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["finished_at"] = now_utc()
    data["status"] = status
    _write(path, data)


def status_from_abort(reason: str | None) -> str:
    """``summary['aborted']`` (None, budget, errors, interrupted) -> run status."""
    if reason is None:
        return "completed"
    return "interrupted" if reason == "interrupted" else f"aborted:{reason}"


@contextmanager
def sigterm_as_interrupt() -> Iterator[None]:
    """Turn SIGTERM into KeyboardInterrupt so a killed run still records ``interrupted``."""

    def handler(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
