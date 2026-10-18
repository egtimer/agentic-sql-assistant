.PHONY: setup seed ingest run serve test eval lint format clean

setup:
	uv sync

seed:
	python -m src.db.seed

ingest:
	python -m src.rag.ingest

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -v --tb=short

eval:
	python -m eval.run_eval

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

clean:
	rm -f src/db/sample.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
