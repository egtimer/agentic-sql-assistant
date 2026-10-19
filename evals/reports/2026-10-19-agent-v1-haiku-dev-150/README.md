# Agent v1 vs single-shot baseline, BIRD Mini-Dev dev split (150 questions)

Historical reference for the README case study: the first LangGraph agent (plan once with a
JSON tool plan, no error recovery) against a single-shot text-to-SQL baseline.

- Run id `haiku-dev-150`, 2026-10-19, code at commit `ec33f8e` (clean tree).
- Model `claude-haiku-5-5`, effort `low`, prompt caching on the schema prefix, `--skip-respond`.
- Split `dev` (150 question ids in `evals/text2sql/splits/dev.json`), 1 worker.
- Metric: execution accuracy (EX), 95% percentile bootstrap CIs (B=10000, seed 0), paired
  bootstrap for the difference.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system both --split dev --skip-respond --budget-usd 0.50 --workers 1 --run-id haiku-dev-150
```

Files: `summary.json` (as written by the runner), `predictions.jsonl` (per question and system,
with the `gold_sql` field removed: gold queries are BIRD content, CC BY-SA 4.0, and are not
redistributed here), `llm_calls.jsonl` (tokens, cost and latency of every LLM call).
