# Test split, Claude Haiku 5.5, four systems (350 questions)

The single run on the BIRD Mini-Dev test split with the prompts frozen after dev iteration 1
(ADR-004). Run once; nothing was tuned on these results.

- Run id `haiku-test-350-v2`, 2026-10-19. Model `claude-haiku-5-5`, effort `low`, prompt
  caching, `--skip-respond`, 4 workers, `--max-fixes 2`, total budget $1.00.
- The run was interrupted by a codespace restart after 213 baseline questions and finished with
  `--resume` (same command plus `--resume --budget-usd 0.95`); `summary.json` covers all 350
  questions of every system. It records `dirty: true` because an untracked file
  (`configs/llm/claude-sonnet-5-5.yaml`) existed when the run resumed; the code is commit
  `b514e0c`.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,baseline_ctx,agent_v2_nocheck,agent_v2 --split test --skip-respond \
  --workers 4 --budget-usd 1.00 --run-id haiku-test-350-v2
```

Files: `summary.json`, `predictions.jsonl` (gold SQL removed, BIRD content is CC BY-SA 4.0),
`llm_calls.jsonl`.
