# Run-to-run noise: bird-haiku-test-350

Method, the credit incident during the first attempt, and the results of all four noise
reports: [`../2026-11-16-noise-README.md`](../2026-11-16-noise-README.md).

Needs the BIRD data (`make bird-data`). The respond step runs after the final SQL, so it does not change what EX scores; r01's cost includes it (about $0.02). r01 and r02 were damaged by the credit incident and their API-failure rows re-run with `--resume`.

Cost per repetition (from each `llm_calls.jsonl`, every invocation counted): $0.2284, 0.2109, 0.2051, 0.2032, 0.2100 (total $1.0576).

Run:

```
# r01 ran with the respond step (its answers were generated but not stored: fixed in 7c93f44)
uv run python -m evals.noise.repeat --eval text2sql --n 1 --prefix 2026-11-16-noise-bird-haiku-test-350/run \
    --budget-usd 0.30 -- --llm-config configs/llm/claude-haiku-5-5.yaml --system baseline,agent_v2 \
    --split test --workers 4
uv run python -m evals.noise.repeat --eval text2sql --n 5 --prefix 2026-11-16-noise-bird-haiku-test-350/run \
    --budget-usd 0.30 -- --llm-config configs/llm/claude-haiku-5-5.yaml --system baseline,agent_v2 \
    --split test --skip-respond --workers 4
for i in 1 2 3 4 5; do uv run python -m evals.save_report \
    evals/results/text2sql/2026-11-16-noise-bird-haiku-test-350/run-r0$i \
    evals/reports/2026-11-16-noise-bird-haiku-test-350/run-r0$i; done
```

Analyse:

```
uv run python -m evals.noise.analyze --eval text2sql --metric correct \
    --runs evals/reports/2026-11-16-noise-bird-haiku-test-350/run-r* --compare agent_v2:baseline \
    --reference evals/reports/2026-10-19-haiku-test-350 \
    --out evals/reports/2026-11-16-noise-bird-haiku-test-350/analysis.json
```

Files: `run-rNN/` one repetition each (the runner's own files; BIRD predictions without
`gold_sql`), `repeat.json` the driver's record of the last invocation, `analysis*.json`.
