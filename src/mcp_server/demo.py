"""Scripted MCP session against the real server over stdio, written out as a transcript.

Starts ``python -m src.mcp_server`` as a subprocess (the same way an MCP client would), runs a
fixed scenario on the Chinook demo database and writes a Markdown transcript with every tool
call, its structured result and the cost reported by the server:

1. ``list_databases`` and ``run_sql`` (no model);
2. ``ask`` a question, then a follow-up in the same ``thread_id`` (short-term memory);
3. ``teach`` a definition for a user, then ``ask`` with that ``user_id`` (long-term memory);
4. ``run_sql`` with a statement the guard refuses;
5. ``forget`` the definition.

Spends money with an Anthropic config (a few LLM calls); the budget flag caps it.

    uv run python -m src.mcp_server.demo --llm-config configs/llm/claude-haiku-5-5.yaml \\
        --budget-usd 0.10 --out docs/examples/mcp-demo.md
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from src.mcp_server.demo_data import REPO_ROOT

DATABASE = "Chinook"
USER = "demo-user"
THREAD = "demo-thread"
TIMEOUT_S = 600

SCENARIO: list[tuple[str, str, dict[str, Any]]] = [
    ("Which databases can I use?", "list_databases", {}),
    (
        "Run my own read-only query",
        "run_sql",
        {"sql": "SELECT COUNT(*) AS tracks FROM Track", "database": DATABASE},
    ),
    (
        "Ask a question",
        "ask",
        {
            "question": "Which 3 genres have the most tracks?",
            "database": DATABASE,
            "thread_id": THREAD,
        },
    ),
    (
        "Follow-up in the same thread",
        "ask",
        {
            "question": "And how many tracks does the second one have?",
            "database": DATABASE,
            "thread_id": THREAD,
        },
    ),
    (
        "Teach a definition",
        "teach",
        {
            "text": "For me, a long track is a track that lasts more than 6 minutes "
            "(Milliseconds > 360000).",
            "database": DATABASE,
            "user_id": USER,
        },
    ),
    (
        "Ask with the taught definition, in a new session",
        "ask",
        {"question": "How many long tracks are there?", "database": DATABASE, "user_id": USER},
    ),
    (
        "A statement the guard refuses",
        "run_sql",
        {"sql": "DELETE FROM Track", "database": DATABASE},
    ),
    ("Forget the definition", "forget", {"database": DATABASE, "user_id": USER}),
]


def _server_params(llm_config: str, budget: float, state_dir: Path) -> StdioServerParameters:
    env = {"PYTHONPATH": str(REPO_ROOT), "PATH": os.environ.get("PATH", "")}
    if "ANTHROPIC_API_KEY" in os.environ:  # passed through, never printed or written
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "src.mcp_server",
            "--llm-config",
            llm_config,
            "--budget-usd",
            str(budget),
            "--state-dir",
            str(state_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
    )


async def run(llm_config: str, budget: float) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mcp-demo-") as tmp:
        params = _server_params(llm_config, budget, Path(tmp))
        with anyio.fail_after(TIMEOUT_S):
            async with Client(params) as client:
                for title, tool, arguments in SCENARIO:
                    result = await client.call_tool(tool, arguments)
                    text = " ".join(
                        getattr(c, "text", "") for c in result.content if getattr(c, "text", "")
                    )
                    steps.append(
                        {
                            "title": title,
                            "tool": tool,
                            "arguments": arguments,
                            "is_error": bool(result.is_error),
                            "structured": result.structured_content,
                            "text": text,
                        }
                    )
    return steps


def render(steps: list[dict[str, Any]], llm_config: str) -> str:
    total = sum(float((s["structured"] or {}).get("cost_usd") or 0.0) for s in steps)
    calls = sum(int((s["structured"] or {}).get("llm_calls") or 0) for s in steps)
    lines = [
        "# MCP server demo on Chinook",
        "",
        "A scripted MCP client session against the real server over stdio "
        f"(`python -m src.mcp_server.demo`, LLM config `{llm_config}`). Every tool call and its "
        "structured result as returned to the client. "
        f"Total: {calls} LLM calls, ${total:.4f}.",
        "",
    ]
    for i, s in enumerate(steps, 1):
        lines += [
            f"## {i}. {s['title']}",
            "",
            f"`{s['tool']}` " + json.dumps(s["arguments"], ensure_ascii=False),
            "",
            "```json",
            json.dumps(
                s["structured"] if s["structured"] is not None else {"text": s["text"]},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            "```",
            "",
        ]
        if s["is_error"]:
            lines += [f"Tool error: {s['text']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llm-config", default="configs/llm/claude-haiku-5-5.yaml")
    parser.add_argument("--budget-usd", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=Path("docs/examples/mcp-demo.md"))
    args = parser.parse_args(argv)
    steps = anyio.run(run, args.llm_config, args.budget_usd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(steps, args.llm_config) + "\n", encoding="utf-8")
    failed = [s["title"] for s in steps if s["is_error"]]
    print(f"wrote {args.out}; {len(steps)} steps; tool errors: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
