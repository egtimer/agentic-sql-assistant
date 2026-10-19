# Local model, Ollama qwen2.5:3b on CPU, 20 stratified dev questions

Baseline and agent v1 (the system is named `agent` in this run) with a 3B model served by
Ollama 0.5.7 on a 4-core CPU (AMD EPYC 7763, no GPU), `num_ctx` 8192, temperature 0. Measures
time and accuracy of a fully local setup; cost is zero.

- Run id `ollama-dev-20`, 2026-10-19, `--limit 20 --seed 0`, `--skip-respond`, 1 worker.
- Interrupted by a codespace restart after 20 baseline and 6 agent questions, finished with
  `--resume`; `summary.json` covers all 40 predictions and records the commit of the resumed
  run (`b615179`). Agent v1 and the baseline did not change between the two commits.
- Latency is per question and is dominated by Ollama's prompt processing on CPU. Measured
  separately with one `/api/generate` call on the formula_1 schema: 1,253 prompt tokens took
  92.6 s (about 14 tokens/s) on a cold prefix and the next question on the same schema took
  6.8 s, because Ollama reuses the cached prefix; the runner orders questions by database.

Command:

```
uv run python -m evals.text2sql.run --llm-config configs/llm/ollama-qwen2.5-3b.yaml \
  --system both --split dev --limit 20 --seed 0 --skip-respond --run-id ollama-dev-20
```
