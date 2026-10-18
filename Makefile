.PHONY: setup seed ingest run serve test cov eval bird-data lint typecheck format clean docker-build up down

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
