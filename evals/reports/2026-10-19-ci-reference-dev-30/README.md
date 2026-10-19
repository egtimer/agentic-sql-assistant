# Reference run for the weekly API eval

The committed reference that `.github/workflows/evals-api.yml` compares each weekly or manual run
against (`evals/text2sql/check_regression.py`, paired on the same questions). It uses exactly the
weekly configuration: dev split, 30 questions stratified by database and difficulty
(`--limit 30 --seed 0`), `baseline` and `agent_v2`, `--skip-respond`, Claude Haiku 5.5.

- Run id `ci-reference-dev-30`, 2026-10-19, code at commit `0f8ba03`; `summary.json` records
  `dirty: true` because files unrelated to the runner were being written in the working tree at
  the time. Budget $0.10, spent $0.0238.
- Result: baseline 60.0%, agent_v2 66.7% [50.0, 83.3]. With n = 30 the gate fails only on a drop
  larger than 10 points whose paired 95% CI excludes 0, or on a cost per question above twice this
  run's. Checking this run against itself passes (exit 0).

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/claude-haiku-5-5.yaml \
  --system baseline,agent_v2 --split dev --limit 30 --seed 0 --skip-respond --workers 4 \
  --budget-usd 0.10 --run-id ci-reference-dev-30
uv run python -m evals.save_report evals/results/text2sql/ci-reference-dev-30 \
  evals/reports/2026-10-19-ci-reference-dev-30
```
