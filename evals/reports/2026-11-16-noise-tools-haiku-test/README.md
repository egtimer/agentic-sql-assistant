# Run-to-run noise: tools-haiku-test

Method, the credit incident during the first attempt, and the results of all four noise
reports: [`../2026-11-16-noise-README.md`](../2026-11-16-noise-README.md).

Needs Qdrant with the policy documents ingested and the seeded demo database, as the tool eval does. r04 to r06 were damaged by the credit incident and their API-failure rows re-run with `--resume`.

Cost per repetition (from each `llm_calls.jsonl`, every invocation counted): $0.0155, 0.0154, 0.0152, 0.0153, 0.0156, 0.0158, 0.0156, 0.0152, 0.0154, 0.0154 (total $0.1545).

Run:

```
uv run python -m evals.noise.repeat --eval tools --n 10 --prefix 2026-11-16-noise-tools-haiku-test/run \
    --budget-usd 0.03 -- --llm-config configs/llm/claude-haiku-5-5.yaml --split test --systems agent_v1,no_tools
```

Analyse:

```
for m in answer_correct exact; do uv run python -m evals.noise.analyze --eval tools --metric $m \
    --runs evals/reports/2026-11-16-noise-tools-haiku-test/run-r* --compare agent_v1:no_tools \
    --reference evals/reports/2026-10-30-v1-tools-haiku-test \
    --out evals/reports/2026-11-16-noise-tools-haiku-test/analysis_$m.json; done
```

Files: `run-rNN/` one repetition each (the runner's own files; BIRD predictions without
`gold_sql`), `repeat.json` the driver's record of the last invocation, `analysis*.json`.
