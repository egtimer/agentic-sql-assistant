# Long-term memory, test split, single run (Claude Haiku 5.5)

The one run of the long-term memory evaluation on the BIRD Mini-Dev test split, with the prompts
frozen after the first dev iteration (ADR-007). Nothing was tuned on these results.

- Run id `a6-longterm-test`, 2026-10-19, code at commit `3d57772` (clean tree), effort `low`,
  prompt caching, `--skip-respond`, 4 workers, hard budget $0.80, spent $0.5009 (2,524 LLM calls;
  teaching $0.0358).
- The 349 test questions with external knowledge. Teaching: 349 turns, 751 definitions extracted
  (614 distinct per database), 3 turns with none, 2 items rejected because their term was not in
  the user's text.
- Result: no memory 54.4% [49.0, 59.6], memory 60.7% [55.6, 65.9], oracle 62.8% [57.6, 67.9];
  memory - no memory = +6.3 pts [+2.6, +10.0]; oracle - memory = +2.0 pts [-0.6, +4.6]; gap
  closed 75.9% [44.4%, 110.0%]; retrieval recall 92.3% (98.8% of the questions whose own
  definition was stored), 12.0 definitions shown of 59.9 stored per database on average.

Command:

```
uv run python -m evals.memory.run_longterm --llm-config configs/llm/claude-haiku-5-5.yaml \
  --split test --skip-respond --workers 4 --budget-usd 0.80 --run-id a6-longterm-test
```

Files: `summary.json`, `run.json`, `teaching.jsonl`, `predictions.jsonl` (no gold SQL; BIRD
content is CC BY-SA 4.0), `llm_calls.jsonl`.
