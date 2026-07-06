# Lean image for the FastAPI service (UI and eval deps intentionally excluded).
FROM python:3.11-slim
WORKDIR /app

# Runtime deps incl. the OpenAI (langchain-openai) and pgvector (langchain-postgres
# + psycopg) providers, python-multipart for /upload, and pypdf/bs4 for uploads.
# Eval-only libs (ragas, datasets) and the UI (streamlit) are excluded -- see
# requirements-api.txt.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY data/ ./data/

ENV LLM_PROVIDER=fake \
    EMBEDDING_PROVIDER=fake \
    VECTOR_BACKEND=memory

EXPOSE 8000
# Ingest first, then serve -- so the image is self-sufficient however it's
# launched (parity with the from-scratch API image). This populates pgvector
# before serving; on the memory backend it's a cheap no-op (no DB, no embedding
# -- the pipeline builds its in-process index lazily on the first request).
CMD ["sh", "-c", "python -m src.rag_lc.ingest && uvicorn src.rag_lc.api:app --host 0.0.0.0 --port 8000"]
