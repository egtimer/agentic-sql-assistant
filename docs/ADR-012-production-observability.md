# ADR-012: OpenTelemetry GenAI traces next to Langfuse, and production detectors

**Status:** Accepted
**Date:** 2026-11-17

## Context

Since ADR-006 every agent node and LLM call goes to Langfuse (self-hosted v2) through `src/observability/tracing.py`. That works for this project, but it ties the traces to one vendor's data model, and most production stacks collect traces with OpenTelemetry. The GenAI semantic conventions now name what an LLM application should record (agent runs, model calls, tool calls, token usage), so a collector, a vendor or a dashboard can read the traces without knowing this code.

The failure analysis of 2026-11-17 also showed which problems a person had to find by reading traces: tool calls that fail, failures the answer does not mention, calls repeated with the same input, and cost. Those can be checked automatically on every run.

## Decision

### Traces: one API, two backends

- `src/observability/otel.py` implements the same `Tracer` interface as the Langfuse tracer. `tracing._build_default` builds Langfuse when its keys are set, OpenTelemetry when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, and both through `MultiTracer` when both are. Without either, tracing stays a no-op. The agents' code does not change: they keep calling `tracing.trace`, `tracing.span` and `tracing.record_generation`.
- **Spans** (GenAI conventions, `gen_ai` attributes of `opentelemetry-semantic-conventions` 0.66b1, still incubating):
  - `invoke_agent agent_v1` / `invoke_agent agent_v2` for an agent run: `gen_ai.operation.name`, `gen_ai.agent.name`.
  - `chat {model}` for every LLM call: `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` (every prompt token, cached or not, as the conventions define it), `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.response.finish_reasons`.
  - `execute_tool {tool}` for each tool call: `gen_ai.tool.name`. Tool calls had no span of their own; they now do in both backends (`tool:calculator`, `tool:run_query` in Langfuse), and a failed call marks its span as an error.
  - Graph nodes, `recall` and eval roots keep their names, with `app.span.kind = internal`.
- **Cost** is not in the conventions. It goes in `app.llm.cost_usd`, the same figure the run's budget tracker accounts (`cost_of`, with cache prices), so the trace and the eval report agree. The call's purpose (`generate`, `check`, `respond`, `fill`) goes in `app.llm.tag`. The `app.` prefix keeps both apart from any future convention.
- **No prompt or answer text in OTel spans.** The conventions make message content opt-in because it can hold personal or client data. The content already goes to Langfuse, which runs on the owner's server. An OTLP endpoint may be a third-party service, so it gets structure, tokens, cost and errors only. A test checks that no prompt text reaches any span attribute.
- **Errors**: status ERROR plus `error.type` (the exception class, or `QueryError` / `ToolError` for a tool that reported a failure).
- **Export**: OTLP over HTTP with the SDK's batch processor, only when the endpoint variable is set; the standard `OTEL_*` variables configure it. Sampling is the SDK default (every trace). Tests use the in-memory exporter; nothing needs a collector.
- **Why keep Langfuse:** it holds the full prompts and answers, the eval scores attached to traces (`execution_accuracy`, `cost_usd`) and the trace verification of ADR-006. OTel adds the portable, content-free view. Scores stay Langfuse-only.
- Dependencies: `opentelemetry-api`, `-sdk` and `-exporter-otlp-proto-http`, bounded `>=1.30,<2`.

### Detectors (`src/monitoring/detectors.py`, `configs/alerts.yaml`)

They read the per-question records the evals write (and a service could log) for either agent:

| Detector | Measures | Limit |
|---|---|---|
| `tool_failure_rate` | failed tool calls (SQL attempts, tool steps) over all calls | 10% |
| `silent_failures` | answered questions where a tool call failed and the answer does not say so | any |
| `loops` | questions with more than 6 tool calls, or the same call (tool and input) twice | 2% |
| `cost_mean`, `cost_item` | mean cost per question; any single question | $0.01; $0.05 |
| `accuracy_drop` (`evals/monitor.py`, needs a reference) | EX or answer check against a committed report, paired bootstrap | drop over 10 points with the CI below 0 |

The limits come from the committed runs: on all 39 of them (noise repetitions and the ADR-011 before and after runs, both agents) none alerts; the highest tool failure rate is 4.8%, the highest mean cost $0.0070 per question. The `silent_failures` rule is the one the failure analysis used, now shared from `src/monitoring/` so the analysis and production apply the same text.

`python -m evals.monitor <run> [--reference <report>]` writes `alerts.json`, prints a markdown table (also to the GitHub step summary) and exits with 1 on an alert, 2 on unusable input. The weekly API eval runs it after the regression gate, on agent v2's records.

## Consequences

- One more span per tool call in Langfuse; the trace verification counts `tool:run_query` against the attempts.
- A deployment can send traces to any OTLP backend by setting one variable, without exposing prompts or answers there.
- The detectors catch failures that announce themselves. They would not have caught the calculator bug of ADR-011, whose wrong results ran without error: that took gold values and a person reading traces. Checking tool outputs against known answers is a job for the evals, not the detectors.
- The thresholds are set from runs that all passed; they have not yet seen a real incident, so they may need tuning once one happens.
- The OTel conventions for GenAI are still incubating; attribute names may change and are pinned to the semconv version above.
