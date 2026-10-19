# Long-term memory, dev split, iteration 1 (Claude Haiku 5.5)

Definitions taught in one session, recalled in a later one (ADR-007). The 149 dev questions with
external knowledge: every evidence of a database is taught first (one `teach` turn each, user
`eval-user`), then every question is answered in a fresh thread under three conditions:
`no_memory` (evidence removed), `memory` (evidence removed, definitions recalled from the store),
`oracle` (evidence in the prompt). Prompts were frozen after this iteration.

- Run id `a6-longterm-dev-it1`, 2026-10-19, code at commit `bc05247`, effort `low`, prompt
  caching, `--skip-respond`, 4 workers, budget $0.40, spent $0.2192 (1,082 LLM calls; teaching
  $0.0153).
- Result: `memory` - `no_memory` = +7.4 pts [+0.7, +14.8]; `oracle` - `memory` = +0.0 pts
  [-4.7, +4.7]; retrieval recall 96.6%. Full numbers in `summary.json`.

Command:

```
uv run python -m evals.memory.run_longterm --llm-config configs/llm/claude-haiku-5-5.yaml \
  --split dev --skip-respond --workers 4 --budget-usd 0.40 --run-id a6-longterm-dev-it1
```

Files: `summary.json`, `run.json` (provenance), `teaching.jsonl` (one row per teaching turn:
memories extracted, keys, rejections, cost), `predictions.jsonl` (no gold SQL; BIRD content is
CC BY-SA 4.0), `llm_calls.jsonl`.
