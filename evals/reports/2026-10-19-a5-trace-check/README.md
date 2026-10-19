# Traced run with a real model, checked against Langfuse

Verification of the A5 observability (ADR-006) with Claude Haiku 5.5 and a local Langfuse server
(2.95.11, `make langfuse-up`): every question and system becomes one trace, with a span per agent
node and one generation per LLM call, and the run's numbers must match what Langfuse stores.

- Run id `a5-trace-check-1`, 2026-10-19, 5 stratified dev questions (`--limit 5 --seed 0`),
  systems `baseline` and `agent_v2`, `--skip-respond`, 1 worker, budget $0.05 (spent $0.00805,
  15 LLM calls). Not an accuracy result (n=5).
- `verify_traces`: 10 traces found of 10 expected, 10/10 ok; 15 generations with usage and cost;
  total cost in Langfuse $0.008050, equal to `llm_calls.jsonl`; `execution_accuracy` and
  `cost_usd` scores on every trace (`trace_report.json`).
- Spot check of one `agent_v2` trace through the Langfuse public API: `introspect`, `generate`,
  `execute`, `check` and `respond` spans under `sql_agent_v2`; the `generate` and `check`
  generations sit under their node spans with model `claude-haiku-5-5`, input/output tokens,
  cache read/write tokens in the metadata, and cost (for `generate`: 100 uncached input tokens at
  $0.10, 424 output tokens at $0.50 and 3,418 cache-write tokens at $0.125 per million =
  $0.00064925, as stored).
- `run.json` is the provenance record written before the first call.

Commands:

```
make langfuse-up
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,agent_v2 --split dev --limit 5 --seed 0 --skip-respond --budget-usd 0.05 \
  --workers 1 --run-id a5-trace-check-1 --trace
uv run python -m evals.observability.verify_traces --session a5-trace-check-1 \
  --predictions evals/results/text2sql/a5-trace-check-1
make langfuse-down
```

Files: `summary.json`, `run.json`, `trace_report.json`, `predictions.jsonl` (gold SQL removed,
BIRD content is CC BY-SA 4.0), `llm_calls.jsonl`.
