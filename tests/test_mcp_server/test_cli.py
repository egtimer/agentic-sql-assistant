"""Command line, environment and the two transports of the real server process."""

import socket
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client

from src.mcp_server import smoke
from src.mcp_server.config import (
    DEFAULT_LLM_CONFIG,
    DEFAULT_STATE_DIR,
    parse_config,
)
from src.mcp_server.demo_data import CHINOOK_DIR, REPO_ROOT

HAIKU = REPO_ROOT / "configs" / "llm" / "claude-haiku-5-5.yaml"


def test_defaults(monkeypatch: pytest.MonkeyPatch):
    for name in ("TRANSPORT", "HOST", "PORT", "DATABASES_DIR", "LLM_CONFIG", "BUDGET_USD"):
        monkeypatch.delenv(f"SQL_MCP_{name}", raising=False)
    monkeypatch.delenv("SQL_MCP_STATE_DIR", raising=False)
    config = parse_config([])
    assert config.transport == "stdio"
    assert (config.host, config.port) == ("127.0.0.1", 8765)
    assert config.databases_dir == CHINOOK_DIR
    assert config.llm_config_path == DEFAULT_LLM_CONFIG
    assert config.llm_config.provider == "ollama"
    assert config.budget_usd is None
    assert config.state_dir == DEFAULT_STATE_DIR


def test_flags_and_relative_llm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    config = parse_config(
        [
            "--transport=http",
            "--port=9000",
            "--databases-dir",
            str(tmp_path),
            "--llm-config",
            "configs/llm/claude-haiku-5-5.yaml",
            "--budget-usd=0.5",
            "--state-dir",
            str(tmp_path / "s"),
        ]
    )
    assert config.transport == "http" and config.port == 9000
    assert config.llm_config_path == HAIKU and config.llm_config.provider == "anthropic"
    assert config.budget_usd == 0.5 and config.state_dir == tmp_path / "s"


def test_environment_sets_defaults_and_flags_win(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SQL_MCP_BUDGET_USD", "2")
    monkeypatch.setenv("SQL_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SQL_MCP_LLM_CONFIG", str(HAIKU))
    config = parse_config([])
    assert config.budget_usd == 2.0 and config.state_dir == tmp_path
    assert parse_config(["--budget-usd", "3"]).budget_usd == 3.0


def test_anthropic_requires_a_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("SQL_MCP_BUDGET_USD", raising=False)
    with pytest.raises(SystemExit) as info:
        parse_config(["--llm-config", str(HAIKU)])
    assert info.value.code == 2
    assert "--budget-usd is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [["--budget-usd=-1"], ["--port=0"], ["--llm-config=missing.yaml"], ["--transport=sse"]]
)
def test_invalid_settings_exit_with_status_2(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SQL_MCP_BUDGET_USD", raising=False)
    with pytest.raises(SystemExit) as info:
        parse_config(argv)
    assert info.value.code == 2


def test_empty_databases_dir_is_a_clear_startup_error(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "src.mcp_server", "--databases-dir", str(tmp_path / "none")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode != 0
    assert "make chinook-data" in result.stderr
    assert "Traceback" not in result.stderr


def test_stdio_smoke_passes():
    assert smoke.main() == 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_streamable_http_transport(databases_dir: Path, tmp_path: Path):
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.mcp_server",
            "--transport",
            "http",
            "--port",
            str(port),
            "--databases-dir",
            str(databases_dir),
            "--state-dir",
            str(tmp_path / "state"),
        ],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("the HTTP server did not start")
                await anyio.sleep(0.2)
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            listed = await client.call_tool("list_databases", {})
            blocked = await client.call_tool(
                "run_sql", {"sql": "DROP TABLE city", "database": "toy"}
            )
        assert listed.structured_content == {
            "databases": [{"name": "other", "tables": 2}, {"name": "toy", "tables": 2}]
        }
        assert (blocked.structured_content or {}).get("blocked") is True
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
