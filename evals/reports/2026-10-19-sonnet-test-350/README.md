# Test split, Claude Sonnet 5.5, baseline and agent v2 (350 questions)

Same frozen prompts, split and settings as the Haiku test run, with a stronger model, to get
the cost/accuracy trade-off. Run once.

- Run id `sonnet-test-350-v2`, 2026-10-19. Model `claude-sonnet-5-5`, effort `low`, prompt
  caching, `--skip-respond`, 4 workers, `--max-fixes 2`, hard budget $3.00 (spent $2.51).
- Prices (USD per million tokens, Anthropic pricing page, checked 2026-10-19): input 2.00,
  5-minute cache write 2.50, cache hit 0.10, output 10.00. Refusal fallbacks were not enabled,
  so every answer comes from this model; no call ended in a refusal.
- `summary.json` records `dirty: true` because the Sonnet config was not committed yet; the
  code is commit `b514e0c`.
- The check step of agent v2 on Sonnet usually explains its review before ending with "OK";
  198 check answers did not start with "OK" and were counted as `unparsed`, which the code
  treats as OK. A fix is recognised only in a ```sql block. The full check text was not stored,
  so a fix written in another format cannot be ruled out; the numbers describe agent v2 as run.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-sonnet-5-5.yaml \
  --system baseline,agent_v2 --split test --skip-respond --workers 4 --budget-usd 3.00 \
  --run-id sonnet-test-350-v2
```

Files: `summary.json`, `predictions.jsonl` (gold SQL removed), `llm_calls.jsonl`.
