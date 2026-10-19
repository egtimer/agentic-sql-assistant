# Long-term memory, dev split, replication with frozen prompts (Claude Haiku 5.5)

Same eval, prompts and settings as iteration 1 (`../2026-10-19-a6-longterm-dev-it1/`), run again
without changes to see how much the result moves with model sampling.

- Run id `a6-longterm-dev-rep`, 2026-10-19, started at commit `83ed9a4`. The codespace restarted
  while the run was in progress (after teaching, `no_memory`, `memory` and 86 of 149 `oracle`
  questions; $0.1891 spent); it was continued with `--resume` at commit `84ff843`, which only
  adds the resume code (agent code identical). The resume ran the 63 missing `oracle` questions
  and nothing else; `run.json` records the resume. `llm_calls.jsonl` has 2 calls more than the
  predictions account for ($0.0005): calls in flight when the process was killed, whose question
  was answered again after the resume.
- Budget $0.40 in total (0.40 first invocation, 0.21 for the resume), spent $0.2147 (1,082 calls).
- Result: `memory` - `no_memory` = +8.1 pts [+1.3, +15.4]; `oracle` - `memory` = +1.3 pts
  [-4.7, +7.4]; retrieval recall 98.0%.

Commands:

```
uv run python -m evals.memory.run_longterm --llm-config configs/llm/claude-haiku-5-5.yaml \
  --split dev --skip-respond --workers 4 --budget-usd 0.40 --run-id a6-longterm-dev-rep
uv run python -m evals.memory.run_longterm --llm-config configs/llm/claude-haiku-5-5.yaml \
  --split dev --skip-respond --workers 4 --budget-usd 0.21 --run-id a6-longterm-dev-rep --resume
```

Files as in iteration 1.
