.PHONY: help install api ui eval eval-compare test lint fmt

help:
	@echo "install       Install deps        | api          Run FastAPI on :8000"
	@echo "ui            Run Streamlit UI (needs the API running)"
	@echo "eval          Run eval harness (NO_RAGAS=1 to skip Ragas)"
	@echo "eval-compare  Benchmark vector vs hybrid retrieval (A/B)"
	@echo "test          Run keyless tests   | lint  ruff check   | fmt  ruff --fix"

install:
	pip install -r requirements.txt

api:
	uvicorn src.rag_lc.api:app --reload --port 8000

ui:
	streamlit run ui/streamlit_app.py

eval:
	python -m eval.run_eval $(if $(NO_RAGAS),--no-ragas,)

eval-compare:
	python -m eval.run_eval --compare

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff check --fix .
