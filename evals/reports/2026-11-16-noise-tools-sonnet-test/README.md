# Run-to-run noise: tools-sonnet-test

Method, the credit incident during the first attempt, and the results of all four noise
reports: [`../2026-11-16-noise-README.md`](../2026-11-16-noise-README.md).

Needs Qdrant with the policy documents ingested and the seeded demo database, as the tool eval does. r01 to r03 were damaged by the credit incident and their API-failure rows re-run with `--resume`; r01's cost includes both attempts.

Cost per repetition (from each `llm_calls.jsonl`, every invocation counted): $0.4057, 0.4073, 0.4014, 0.4088, 0.4052 (total $2.0284).

Run:

```
uv run python -m evals.noise.repeat --eval tools --n 5 --prefix 2026-11-16-noise-tools-sonnet-test/run \
    --budget-usd 0.50 -- --llm-config configs/llm/claude-sonnet-5-5.yaml --split test --systems agent_v1,no_tools
```

Analyse:

```
for m in answer_correct exact; do uv run python -m evals.noise.analyze --eval tools --metric $m \
    --runs evals/reports/2026-11-16-noise-tools-sonnet-test/run-r* --compare agent_v1:no_tools \
    --reference evals/reports/2026-10-30-v1-tools-sonnet-test \
    --out evals/reports/2026-11-16-noise-tools-sonnet-test/analysis_$m.json; done
```

Files: `run-rNN/` one repetition each (the runner's own files; BIRD predictions without
`gold_sql`), `repeat.json` the driver's record of the last invocation, `analysis*.json`.
