# Follow-up conversations without evidence (dev, Claude Haiku 5.5)

The 30 two-turn conversations of `evals/memory/followups.jsonl` (written from dev questions;
turn 1 sets a context, turn 2 is elliptical and is scored against the source question's gold
SQL). No condition gets any external knowledge (`--evidence none`), so the conversation history
is the only way turn 2 can know what turn 1 introduced.

- Run id `a6-followups-dev-noev`, 2026-10-19, commit `83ed9a4`, budget $0.15, spent $0.0506.
- Result: turn 2 alone 6.7% (2/30), with history 46.7% [30.0, 63.3], original question 56.7%
  [40.0, 73.3]; with history - turn 2 alone = +40.0 pts [+23.3, +56.7]; original - with history
  = +10.0 pts [-3.3, +23.3]. n = 30, all from dev: indicative.

Command:

```
uv run python -m evals.memory.run_followups --llm-config configs/llm/claude-haiku-5-5.yaml \
  --skip-respond --workers 4 --budget-usd 0.15 --evidence none --run-id a6-followups-dev-noev
```

Files: `summary.json`, `run.json`, `predictions.jsonl` (no gold SQL), `llm_calls.jsonl`.
