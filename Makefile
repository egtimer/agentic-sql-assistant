.PHONY: setup seed ingest run serve test cov eval eval-sql bird-data lint typecheck format clean docker-build up down

setup:
	uv sync

seed:
	uv run python -m src.db.seed

ingest:
	uv run python -m src.rag.ingest

run:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest -v --tb=short

cov:
	uv run pytest --cov=src --cov-report=term-missing

eval:
	uv run python -m evals.run_eval

# Text-to-SQL eval on BIRD Mini-Dev. Override on the command line, e.g.
#   make eval-sql LLM_CONFIG=configs/llm/claude-haiku-5-5.yaml BUDGET_USD=1 RUN_ID=smoke LIMIT=20
LLM_CONFIG ?= configs/llm/ollama-qwen2.5-3b.yaml
SYSTEM ?= both
SPLIT ?= dev
eval-sql:
	uv run python -m evals.text2sql.run --llm-config $(LLM_CONFIG) --system $(SYSTEM) --split $(SPLIT) \
		$(if $(LIMIT),--limit $(LIMIT)) $(if $(BUDGET_USD),--budget-usd $(BUDGET_USD)) \
		$(if $(RUN_ID),--run-id $(RUN_ID)) $(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(SEED),--seed $(SEED)) $(if $(SKIP_RESPOND),--skip-respond) $(if $(MAX_FIXES),--max-fixes $(MAX_FIXES)) $(if $(RESUME),--resume) $(if $(QUESTION_IDS),--question-ids $(QUESTION_IDS))

bird-data:
	uv run python -m evals.text2sql.prepare

lint:
	uv run ruff check src tests evals
	uv run ruff format --check src tests evals
	uv run mypy src

typecheck:
	uv run mypy src

format:
	uv run ruff format src tests evals

docker-build:
	docker build -t agentic-sql-assistant:dev .

up:
	docker compose up -d

down:
	docker compose down

clean:
	rm -f src/db/sample.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
