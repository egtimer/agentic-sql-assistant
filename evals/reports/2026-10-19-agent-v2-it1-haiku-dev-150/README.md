# SQL agent v2 ablation, iteration 1, BIRD Mini-Dev dev split (150 questions)

First (and final) prompt iteration of the SQL agent v2 (ADR-004). Prompts were frozen after this
run: the success criterion was met and the remaining errors showed no systematic pattern that a
prompt change could fix without tuning to dev.

- Run id `haiku-dev-150-v2-it1`, 2026-10-19, code at commit `b615179` (clean tree).
- Model `claude-haiku-5-5`, effort `low`, prompt caching on the schema prefix, `--skip-respond`,
  4 workers, `--max-fixes 2`.
- Systems: `baseline` (single shot, DDL), `baseline_ctx` (single shot, v2 introspection context
  with example values), `agent_v2_nocheck` (v2 generate prompt; check only on execution errors),
  `agent_v2` (full: generate, execute, check, up to 2 fixes).
- Metric: execution accuracy (EX), 95% percentile bootstrap CIs (B=10000, seed 0), paired
  bootstrap for every pair of systems.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,baseline_ctx,agent_v2_nocheck,agent_v2 --split dev --skip-respond \
  --workers 4 --budget-usd 0.45 --max-fixes 2 --run-id haiku-dev-150-v2-it1
```

Files: `summary.json`, `predictions.jsonl` (gold SQL removed, BIRD content is CC BY-SA 4.0),
`llm_calls.jsonl`.
