# SQL agent v2 ablation, replication with frozen prompts, BIRD Mini-Dev dev split (150 questions)

Same systems, prompts, model and settings as iteration 1
(`../2026-10-19-agent-v2-it1-haiku-dev-150/`), run again without any change to measure how much
the results move with model sampling (thinking is on, so outputs are not deterministic). The
bootstrap CIs only cover sampling over questions; this replication covers sampling of the model.

- Run id `haiku-dev-150-v2-rep`, 2026-10-19. `summary.json` records commit `8b82a01`; the code
  is identical to `b615179` (the commit in between only added the iteration 1 report).
- Model `claude-haiku-5-5`, effort `low`, prompt caching, `--skip-respond`, 4 workers,
  `--max-fixes 2`.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,baseline_ctx,agent_v2_nocheck,agent_v2 --split dev --skip-respond \
  --workers 4 --budget-usd 0.45 --max-fixes 2 --run-id haiku-dev-150-v2-rep
```

Files: `summary.json`, `predictions.jsonl` (gold SQL removed, BIRD content is CC BY-SA 4.0),
`llm_calls.jsonl`.
