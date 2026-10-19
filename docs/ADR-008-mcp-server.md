# ADR-008: MCP server for the SQL agent and its guard

**Status:** Accepted (implementation and offline tests done; real-model demo pending)
**Date:** 2026-10-19

## Context

Agent v2 (ADR-004), its SQL guard (ADR-005), tracing (ADR-006) and memory (ADR-007) are used today through Python calls and the eval runners. Anything that wants to use them from outside this repository (an editor plugin, a chat client, another agent) would need its own integration. The Model Context Protocol (MCP) is the common interface those clients already speak, so one server makes the agent and, separately, the guarded SQL tool available to all of them.

Two uses are different and are kept apart as tools:

- **Ask in natural language.** The server's own model writes, runs and reviews the SQL (`ask`). The client does not need to know SQL.
- **Run SQL the client wrote.** A client model that already knows the schema can send a query and get rows back (`run_sql`), with no language model on the server side. The value here is the guard: a client that is itself a model is as untrusted as the model inside `ask`.

## Decision

### SDK and API

The official Python SDK, package `mcp` 2.x (`mcp>=2.2,<3`; 2.3.0 is locked). In 2.x `FastMCP` is gone; the high-level server is `mcp.server.mcpserver.MCPServer` with `@server.tool()` and `@server.resource()` decorators, and tool/resource failures raised as `ToolError`/`ResourceError`. The client is `mcp.Client`, which takes an `MCPServer` instance (in-process, used by the tests), a URL (streamable HTTP) or `StdioServerParameters` (subprocess). Tool results are `mcp_types.CallToolResult`. Input schemas come from the function signature (`Annotated[..., Field(description=...)]`); output schemas from `Annotated[CallToolResult, Model]`, which lets a tool return structured content (the model, validated against the output schema) together with its own short text rendering.

### Code layout (`src/mcp_server/`)

| Module | Role |
|---|---|
| `server.py` | `create_server(service)`: tool and resource definitions, input constraints, text renderings |
| `service.py` | `Service`: the logic behind the tools (catalog, budget, client, state); blocking, run in worker threads |
| `catalog.py` | the database allowlist |
| `models.py` | pydantic models of the tool outputs |
| `config.py`, `__main__.py` | flags, environment, startup, transports |
| `demo_data.py` | Chinook download with a pinned sha256 |
| `smoke.py` | stdio end-to-end smoke test |

### Tools

Names are verbs a model can pick from the list without reading much: `list_databases`, `describe_database`, `ask`, `run_sql`, `teach`, `forget`. Every input has a description and constraints (lengths, an id pattern); every tool returns structured content plus a short text.

| Tool | Does | LLM |
|---|---|---|
| `list_databases()` | names and table counts | no |
| `describe_database(database)` | the agent's schema context: DDL, keys not declared in the DDL, example values of text columns | no |
| `ask(question, database, evidence="", thread_id=None, user_id=None)` | `run_sql_agent` with the respond step on; returns answer, final SQL, columns, the first 20 rows (values cut at 60 characters), row count, `blocked`, the error of the final query, number of LLM calls and cost of this request, trace id | yes |
| `run_sql(sql, database)` | `guarded_execute` with all guard layers, 10 s timeout, at most 100 rows (values cut at 200 characters); a refused statement comes back as `blocked: true` with the reason | no |
| `teach(text, database, user_id)` | `teach_text` into the persistent store; returns the definitions it stored | yes |
| `forget(database, user_id, term=None)` | deletes one definition (by normalised term) or all of that user for that database | no |

Resource `schema://{database}` returns the same text as `describe_database`. There are no prompts.

A guard block or a SQL error is a normal **result** (`ok: false`, `blocked`, `error`), not a tool error: the client model can read the reason and correct itself, and a block is expected behaviour, not a failure of the server. Tool errors (`isError`) are for things the client cannot fix by rewriting the query: unknown database, invalid arguments, budget used up, model unavailable.

### Transports

`python -m src.mcp_server` speaks stdio by default (the client starts the process). `--transport http --host 127.0.0.1 --port 8765` serves streamable HTTP at `/mcp` and binds to localhost; for a loopback host the SDK also enables its DNS-rebinding protection (Host header allowlist). A non-loopback `--host` prints a warning. Nothing is written to stdout except protocol messages (logs go to stderr).

Configuration is by flag or environment variable `SQL_MCP_<FLAG>` (a flag wins): `--databases-dir` (default `data/external/chinook/`), `--llm-config` (default `configs/llm/ollama-qwen2.5-3b.yaml`; a relative path is also tried against the repository root, because clients start servers from any directory), `--budget-usd`, `--state-dir` (default `data/state/`, gitignored).

### Databases are addressed by name only

At startup the server scans `--databases-dir` for `*.sqlite` and `*.db` files directly inside it and builds a map from file stem to resolved path. Two files with the same stem stop the server. A tool receives a name and looks it up in that map; a name that is not in it (including anything that looks like a path, an extension or `..`) is a tool error that lists the valid names. No client-supplied string is ever turned into a file path. The agent derives the long-term memory's database id from the file stem, which is the same name.

### State

Persistent state lives in `--state-dir`, opened on first use: `checkpoints.sqlite` (the LangGraph SQLite checkpointer, `open_checkpointer`) and `memories.sqlite` (the SQLite store, `open_memory_store`). With `thread_id`, `ask` runs the graph with the checkpointer under the key `<database>:<thread_id>`, so a thread never mixes history from two databases, and conversations survive a server restart. With `user_id`, `ask` passes the store so the `recall` step reads that user's definitions; `teach` writes them and `forget` removes them, so a user controls what is kept. Without these arguments nothing is read or written and the prompts are the frozen ones of ADR-004. Result rows are never stored (ADR-007).

### Budget and cost

The server owns one `UsageTracker` with the `--budget-usd` cap. `ask` and `teach` check it before the first call and the LLM client checks it before every call, so a request that reaches the cap halfway fails with a tool error and the server keeps serving `run_sql`, `describe_database` and `list_databases`. The cap is per server process (it restarts at zero), and `--budget-usd` is required for `anthropic` configs, as in the eval runners. The client the tools use is wrapped so each request counts its own LLM calls and cost; concurrent requests do not mix their numbers.

### Safety model

- **The client is untrusted.** Both the SQL a client sends to `run_sql` and the SQL the server's model writes in `ask` (which can be steered by the question, the evidence or database contents) go through `guarded_execute`: lexical check, authorizer allowlist, read-only connection with engine limits, timeout and row cap (ADR-005). The server opens no other connection for SQL a model or client wrote. The schema context is built with a trusted read-only connection and fixed PRAGMAs, as for the agent.
- **No paths.** Databases by name; state paths come from the flag, not from the client; errors never contain them.
- **Errors.** `ToolError` messages are fixed strings the client can act on. Any other exception is logged on the server (stderr) and reaches the client as the SDK's generic `Error executing tool <name>` or, for model failures, as "The language model request failed". Provider error texts, stack traces and file paths do not reach the client. Tests check this with a model client that raises an error containing a path.
- **Input limits.** Question 2000 characters, evidence and teach text 4000, SQL 20000, ids 128 characters from `[A-Za-z0-9_.@:+-]`.
- **Memory isolation** is that of ADR-007: namespaces per `(user_id, database)`, definitions only from `teach`.
- **Tracing.** When Langfuse is configured, each `ask` is one trace named `mcp_ask` (the agent's `sql_agent_v2` trace becomes a span below it) with the thread id as session and the user id; the trace id is in the result.

### Chinook demo

Chinook (a digital media store: artists, albums, tracks, invoices, customers; 11 tables) by Luis Rocha, MIT License, Copyright (c) 2008-2024 Luis Rocha. `make chinook-data` (`python -m src.mcp_server.demo_data`) downloads the SQLite file of release v1.4.5 into `data/external/chinook/` (gitignored), checks the sha256 and refuses a mismatch (a bad download is deleted; an existing file that does not match is reported and kept). It is public data and the demo needs no BIRD download. The database has a single file stem, `Chinook`, which is the database name.

## Out of scope

- **Authentication and authorisation for remote HTTP deployments.** The HTTP transport is meant for localhost. Exposing it needs a reverse proxy or the SDK's auth hooks, TLS and per-client quotas; none of that is done here, and the startup warning says so. With stdio the client is the process owner.
- **Multi-tenant isolation beyond `user_id`.** `user_id` and `thread_id` are labels the client chooses. They separate memories and conversations of well-behaved clients; they are not credentials. A client that knows or guesses another user's id can read or delete that user's memories, and the server cannot tell. One server process, one budget and one state directory serve all its clients.
- Concurrent requests on the same `thread_id` (the checkpointer would interleave their turns): clients should send them in order.
- Writes to the databases, cancellation of a running `ask`, progress notifications, MCP prompts, and a deny policy for tables and columns (the guard supports one, the server uses the default policy like the evals).
- The guard's own limits (SQLite-specific, not a sandbox against a malicious host; ADR-005).

## Testing

`tests/test_mcp_server/` (64 tests, all offline): an in-process `mcp.Client` against `create_server(...)` for tool list and schemas, `list_databases`/`describe_database`/the resource, `run_sql` (rows, row and value caps, DROP, DELETE, ATTACH, VACUUM INTO, multiple statements, PRAGMA and `load_extension` reported as results, the table intact afterwards), unknown names and path-like names, and `ask`/`teach`/`forget` with a scripted client injected through the `client_factory` argument of `Service` (an internal hook of the object; there is no environment variable or flag for it): per-request cost and call counts, evidence, thread memory across two asks and across a service restart, per-database threads, teach then recall for the right user only, forget of one term and of all, budget exhausted before and during a request while `run_sql` still works, error texts without traces or paths, one trace per `ask`. The CLI tests start the real process over stdio and streamable HTTP. `python -m src.mcp_server.smoke` is the same stdio check as a standalone command and runs in the `evals-offline` CI job; it calls no LLM tool.

## Limitations

- The offline tests prove the plumbing, not answer quality; quality is what ADR-004 and ADR-007 measured on BIRD Mini-Dev. A real-model session on Chinook is recorded in the README when it has been run.
- `ask` blocks one worker thread for the whole agent run (several LLM calls). The default thread pool is large enough for a local server; there is no queue or per-client concurrency limit.
- `describe_database` shows example values from the data; for a sensitive database that is data leaving through the schema tool, which is as visible to the client as `run_sql` already makes it.
- The budget counts what the tracker records; a call that fails after the provider charged for it is not counted.
