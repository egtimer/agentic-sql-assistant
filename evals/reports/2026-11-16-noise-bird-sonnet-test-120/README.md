# Run-to-run noise: bird-sonnet-test-120

Method, the credit incident during the first attempt, and the results of all four noise
reports: [`../2026-11-16-noise-README.md`](../2026-11-16-noise-README.md).

Needs the BIRD data. The cap was first $1.00 per repetition, estimated from the per-question cost of the 350-question run; r01 hit it with `agent_v2` at 100/120 (a 120-question sample over 11 databases writes the cached schema prefix far more often per call). The cap was raised to $1.25 and r01 resumed (`--resume-stopped`). r03 cost less ($0.80) than r01 and r02 ($1.15, $1.11), with more cache reads.

Cost per repetition (from each `llm_calls.jsonl`, every invocation counted): $1.1469, 1.1142, 0.8003 (total $3.0614).

Run:

```
uv run python -m evals.noise.repeat --eval text2sql --n 3 --prefix 2026-11-16-noise-bird-sonnet-test-120/run \
    --budget-usd 1.25 --resume-stopped -- --llm-config configs/llm/claude-sonnet-5-5.yaml \
    --system baseline,agent_v2 --split test --limit 120 --seed 0 --skip-respond --workers 4
for i in 1 2 3; do uv run python -m evals.save_report \
    evals/results/text2sql/2026-11-16-noise-bird-sonnet-test-120/run-r0$i \
    evals/reports/2026-11-16-noise-bird-sonnet-test-120/run-r0$i; done
```

Analyse:

```
uv run python -m evals.noise.analyze --eval text2sql --metric correct \
    --runs evals/reports/2026-11-16-noise-bird-sonnet-test-120/run-r* --compare agent_v2:baseline \
    --reference evals/reports/2026-10-19-sonnet-test-350 \
    --out evals/reports/2026-11-16-noise-bird-sonnet-test-120/analysis.json
```

Files: `run-rNN/` one repetition each (the runner's own files; BIRD predictions without
`gold_sql`), `repeat.json` the driver's record of the last invocation, `analysis*.json`.
