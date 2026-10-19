# Pilot, Claude Sonnet 5.5 on 20 stratified dev questions

Used only to measure Sonnet's real token usage before the test run (the estimate from Haiku's
token profile was above the owner's $3 threshold). Not a result to quote: n=20 and a small run
pays a cache write on almost every call, so its cost per question is higher than in a full run.

- Run id `sonnet-dev-20-pilot`, 2026-10-19, `--limit 20 --seed 0`, 1 worker, budget $0.50
  (spent $0.30). Code at commit `b514e0c` (`dirty: true`: the Sonnet config was not committed).

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-sonnet-5-5.yaml \
  --system baseline,agent_v2 --split dev --limit 20 --seed 0 --skip-respond --workers 1 \
  --budget-usd 0.50 --run-id sonnet-dev-20-pilot
```
