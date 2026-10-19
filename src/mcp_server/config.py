"""Command line and environment configuration of the MCP server."""

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.llm import LLMConfig
from src.mcp_server.demo_data import CHINOOK_DIR, REPO_ROOT

DEFAULT_LLM_CONFIG = REPO_ROOT / "configs" / "llm" / "ollama-qwen2.5-3b.yaml"
DEFAULT_STATE_DIR = REPO_ROOT / "data" / "state"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
ENV_PREFIX = "SQL_MCP_"


@dataclass(frozen=True)
class ServerConfig:
    transport: str
    host: str
    port: int
    databases_dir: Path
    llm_config_path: Path
    llm_config: LLMConfig
    budget_usd: float | None
    state_dir: Path


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.mcp_server",
        description="MCP server for the SQL agent v2 and its SQL guard. Each flag can also be "
        f"set with the environment variable {ENV_PREFIX}<FLAG> (for example "
        f"{ENV_PREFIX}BUDGET_USD); a flag wins.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=_env("TRANSPORT", "stdio"),
        help="stdio (default) or http (streamable HTTP, endpoint /mcp)",
    )
    parser.add_argument("--host", default=_env("HOST", DEFAULT_HOST), help="http only")
    parser.add_argument(
        "--port", type=int, default=int(_env("PORT", str(DEFAULT_PORT))), help="http only"
    )
    parser.add_argument(
        "--databases-dir",
        type=Path,
        default=Path(_env("DATABASES_DIR", str(CHINOOK_DIR))),
        help="directory scanned for *.sqlite and *.db files; the file stem is the database name "
        "(default: the Chinook demo directory, see `make chinook-data`)",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=Path(_env("LLM_CONFIG", str(DEFAULT_LLM_CONFIG))),
        help="YAML in configs/llm/ (default: the Ollama config)",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=float(_env("BUDGET_USD", "0")) or None,
        help="cap on language-model spending of this server process; required for anthropic",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(_env("STATE_DIR", str(DEFAULT_STATE_DIR))),
        help="where the conversation checkpoints and the user memories are stored "
        "(default: data/state/)",
    )
    return parser


def resolve_llm_config_path(path: Path) -> Path:
    """``path`` as given, or relative to the repository root (clients start servers anywhere)."""
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def load_llm_config(path: Path) -> LLMConfig:
    with open(path, encoding="utf-8") as f:
        return LLMConfig.model_validate(yaml.safe_load(f))


def parse_config(argv: Sequence[str] | None = None) -> ServerConfig:
    """Parse the command line. Exits with status 2 and a message on invalid settings."""
    parser = build_parser()
    args = parser.parse_args(argv)
    llm_path = resolve_llm_config_path(args.llm_config)
    try:
        llm_config = load_llm_config(llm_path)
    except (OSError, ValueError) as e:
        parser.error(f"cannot load --llm-config {args.llm_config}: {e}")
    if args.budget_usd is not None and args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    if llm_config.provider == "anthropic" and args.budget_usd is None:
        parser.error("--budget-usd is required for an anthropic config (it spends real money)")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return ServerConfig(
        transport=args.transport,
        host=args.host,
        port=args.port,
        databases_dir=args.databases_dir,
        llm_config_path=llm_path,
        llm_config=llm_config,
        budget_usd=args.budget_usd,
        state_dir=args.state_dir,
    )
