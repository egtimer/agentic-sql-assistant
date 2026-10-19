# ADR-006: Observability with Langfuse

**Status:** Accepted
**Date:** 2026-10-19

## Context

Langfuse tracing existed but did not work for the agents, and evaluation runs could not be inspected question by question.

- **Per-node spans were lost.** Agent v1 carried the trace on the graph state as a private attribute (`state._trace`). LangGraph builds a new state object for every node, so the attribute never reached the nodes and no span was created.
- **Agent v2 and the LLM client had no tracing at all.** An eval run produced numbers but no way to open the question that failed: the prompt, the model's answer, the SQL that ran, what the reviewer said.
- **In the container the app looked for Langfuse on `localhost`**, which inside the app container is the app itself.

## Decision

### Context-variable tracing (`src/observability/tracing.py`)

The current trace or observation lives in a `contextvars.ContextVar`, never in graph state. Code calls module functions; the nesting follows the call stack:

| Function | Effect |
|---|---|
| `trace(name, input, metadata, tags, session_id, user_id)` | context manager, opens a root; the handle has `.id` and `.update(output=...)` |
| `span(name, input, metadata)` | context manager, nests under the current observation; records the output, sets level `ERROR` and a status message on an exception and re-raises |
| `trace_or_span(...)` | a span when a trace is active, otherwise a new trace (entry points of the agents) |
| `record_generation(name, model, input, output, usage, cost_usd, start, end, metadata)` | one finished LLM call under the current observation |
| `score(trace_id, name, value, comment)` | numeric score on a trace |
| `flush()` | send pending events |

Two implementations: `NoopTracer` (the default when `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` is empty) and `LangfuseTracer`. Rules:

- **Without a trace nothing happens.** `span` and `record_generation` are no-ops when no trace is open in the current context, so library code calls them unconditionally; callers check `tracing.active()` before building payloads. With no keys there is no client, no network and no serialization.
- **Tracing never breaks a request.** Every SDK call is wrapped; a failure is logged as a warning and the request goes on.
- **LangGraph keeps the context.** Nodes run in the context of the `invoke` call (verified by the tests, which open a trace around a real graph run and check the nesting), so a span opened inside a node nests under the span or trace of its caller. A thread pool does not copy the context: the eval runner opens each trace inside the worker thread.
- `trace_node` (agent v1) keeps its name and now opens a span through the tracer. The old `create_trace` and `get_langfuse` are gone.

### What is traced

| Where | Trace / observation |
|---|---|
| `POST /agent/query` | one trace `api.agent_query` per request (optional `session_id`, `user_id` request fields); the response has a new optional `trace_id`; `agent_query` span below it |
| Agent v1 | span per node: `reason`, `execute`, `validate`, `respond`; input is the state without messages, output the node's update |
| Agent v2 | trace `sql_agent_v2` when none is active, otherwise a span; spans `introspect`, `generate`, `execute`, `check`, `respond`. `execute`: input the SQL; output columns, row count, truncated, error, blocked, block reason, guard layer, elapsed seconds. `check`: output verdict kind, fix SQL, reviewer note |
| LLM client (`complete`) | a generation per call under the current observation, named by its tag: model, input (system and `prefix + user`, full text), output text, token usage, cost from `cost_of`, start and end time, metadata with stop reason, latency and the cache split. A failed call is recorded with level `ERROR` |
| Eval runner `--trace` | one trace per (question, system), see below |

`Attempt` (agent v2) gained `block_reason`, `layer` and `elapsed_s` for the `execute` span; predictions and reports are unchanged.

### Linking evaluations and traces

`python -m evals.text2sql.run --trace` (off by default, refused with an error when Langfuse is not configured) opens one trace per (question, system) named `text2sql/<system>`:

- `session_id` is the run id, so a run is one session in the UI;
- `tags` are `[system, db_id, difficulty, split]`, metadata has `question_id` and `run_id` (the gold SQL is never sent);
- after scoring the trace gets the scores `execution_accuracy` (0 or 1, the failure type as comment) and `cost_usd`;
- `trace_id` is written into every record of `predictions.jsonl`, and `run.json` records the Langfuse host and session.

Without `--trace` the runner switches tracing off for the whole run even if the keys are set. `evals/text2sql/smoke.py` has the same flag (its scripted client reports its calls through the same hook as the real providers).

`python -m evals.observability.verify_traces --session <run id> --predictions <run dir>` checks a traced run against the Langfuse API: the expected node spans exist (one `execute` span per attempt and one `check` span per verdict for agent v2), every observation is nested under the single root span, the number of generations equals `n_llm_calls` of the prediction record, every generation has a model, token usage and a cost (and the costs add up to the record's `cost_usd`), and both scores exist and match. It writes a JSON report and exits 1 on any failure. It reads each trace through `GET /api/public/traces/{id}`: on server 2.95.11 the `traceId` filter of `GET /api/public/scores` is ignored (the endpoint returns the whole project), which the verifier caught on its first run.

### Why SDK v2 and server v2

The code uses the v2 Python SDK (`langfuse==2.60.10`, pinned `<3` in `pyproject.toml`) and the server image `langfuse/langfuse:2.95.11` (pinned in `docker-compose.yml`, the last 2.x tag). The v3 server needs ClickHouse, Redis and blob storage besides Postgres, which is too much infrastructure for a local portfolio setup. Consequences that matter:

- The SDK calls used are `Langfuse.trace(id, name, input, metadata, tags, session_id, user_id)`, `trace.update(output, metadata)`, `observation.span(name, input, metadata)`, `span.end(output, metadata, level, status_message)`, `observation.generation(name, model, input, output, usage, start_time, end_time, metadata, level, status_message)`, `Langfuse.score(trace_id, name, value, data_type, comment)` and `Langfuse.flush()`. Trace ids are generated by the tracer (UUID) so the id is known before any event is sent.
- **Usage and cost use the legacy `usage` object** (`input`, `output`, `total`, `unit`, `totalCost`). The SDK accepts `usage_details` and `cost_details`, but server 2.95.11 stores neither (a first version sent them and the UI showed zero tokens). `input` counts every prompt token (uncached, cache read, cache write); the cache split is in the generation metadata. `totalCost` is the run's own `cost_of` figure, not a Langfuse price table, so the cost shown equals the cost in `llm_calls.jsonl`.
- The server supports headless initialisation (`LANGFUSE_INIT_*`, present in the 2.95.11 image), so `docker compose up -d langfuse-db langfuse` creates the organisation, project, API keys and a user with no clicks. The defaults are for local use only (keys `pk-lf-local-dev` / `sk-lf-local-dev`, user `dev@example.com`); they are the same variables the app and the evals read, so one `.env` configures both ends. Never reuse them anywhere reachable from outside.

## Local setup

```bash
cp .env.example .env     # local-only keys that match the Langfuse project below
make langfuse-up         # Postgres + Langfuse 2.95.11 on http://localhost:3000
# login: dev@example.com / local-dev-password (project "agentic-sql-assistant")
uv run python -m evals.text2sql.smoke --run-id smoke-1 --trace --out /tmp/smoke-1
uv run python -m evals.observability.verify_traces --session smoke-1 --predictions /tmp/smoke-1
make langfuse-down       # stops and removes the containers, keeps the data volume
```

Inside docker compose the app gets `LANGFUSE_HOST=http://langfuse:3000`.

## Overhead

Measured with `python -m evals.observability.measure_overhead --passes 20` (scripted client, so no LLM latency; the 4 smoke questions through `baseline` and `agent_v2` with `skip_respond`, 8 pairs per pass; passes alternate between tracing off and on; local Langfuse in the same codespace). Time per (question, system) pair on the calling thread, including prediction and scoring:

| Mode | mean | sd | min | max |
|---|---|---|---|---|
| tracing off (no-op tracer) | 2.59 ms | 1.73 | 0.80 | 7.20 |
| tracing on | 7.71 ms | 4.06 | 2.99 | 15.84 |
| difference, paired by position | 5.12 ms | 2.46 | 1.79 | 11.16 |

n=160 pairs per mode. Events are sent by the SDK's background thread; `flush()` after a pass of 8 traces (14 generations and 16 scores) took 1.06 s on average (sd 0.11 s, 1.0 to 1.5 s), mostly the SDK's 0.5 s flush interval and one HTTP round trip. The scripted client answers instantly, so these are the worst relative figures: a real single-shot call to Haiku took about 2 s per question in the dev pilot reported in the README. The tracing-off figure is the no-op path (a context-variable read and a check per node).

Limits: one machine, a local server, a small fixture (short prompts: the cost of serialising a large schema context in `introspect` and `generate` inputs is not in these numbers; span strings are truncated at 20,000 characters, generation inputs are not); the runner flushes only at the end, so with a remote server the final `flush` grows with the number of traces.

## Limitations

- Server and SDK v2 are in maintenance; moving to v3 means ClickHouse, Redis and blob storage, and the SDK's observation API changes (context-managed `start_as_current_span`), which would stay behind `tracing.py`.
- Generation input is stored in full, including the schema context of every call: a large database means large traces. Do not send traces of private data to a shared Langfuse.
- `execute` in agent v1 is one span for the node; the tool call inside it is not a separate observation.
- Traces are sent by a background thread; a process killed with SIGKILL loses the unsent events (`flush()` runs at the end of eval runs and on API shutdown).
- Real-model verification of a traced run: see the README, Observability.
