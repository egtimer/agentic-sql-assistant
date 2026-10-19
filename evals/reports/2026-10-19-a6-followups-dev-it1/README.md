# Follow-up conversations with the source evidence (dev, Claude Haiku 5.5) — superseded

The 30 two-turn conversations of `evals/memory/followups.jsonl`, with the source question's
external knowledge given in every condition (`--evidence source`).

Caveat, found after this run: in 18 of the 30 pairs the evidence names the entity that turn 1
introduces (for example the molecule id or the team name), so turn 2 alone is not really
elliptical and the effect of the history is understated. The evidence-free run
(`../2026-10-19-a6-followups-dev-noev/`) is the one to read; this run is kept because every run
is reported.

- Run id `a6-followups-dev-it1`, 2026-10-19, commit `bc05247`, budget $0.15, spent $0.0569.
- Result: turn 2 alone 50.0%, with history 60.0%, original 60.0%; with history - turn 2 alone
  = +10.0 pts [-3.3, +23.3].

Command:

```
uv run python -m evals.memory.run_followups --llm-config configs/llm/claude-haiku-5-5.yaml \
  --skip-respond --workers 4 --budget-usd 0.15 --run-id a6-followups-dev-it1
```
