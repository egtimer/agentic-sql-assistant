# Pilot, Claude Haiku 5.5 on 10 dev questions (A2)

First run of the text-to-SQL runner, used to measure real token usage and cost before the
full dev run. Not a result to quote: n=10, and `--limit` then took the first 10 question ids
(only two databases); sampling is now stratified. Agent v1 ran with its final answer step,
which later runs skip.

- Run id `pilot-haiku-10`, 2026-10-18, code at commit `e231c05`, budget $0.10 (spent $0.0118).

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system both --split dev --limit 10 --budget-usd 0.10 --run-id pilot-haiku-10
```
