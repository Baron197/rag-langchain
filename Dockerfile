FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    "langchain>=0.3" "langchain-core>=0.3" "langchain-community>=0.3" \
    "langchain-text-splitters>=0.3" "langchain-classic>=1.0" "rank-bm25>=0.2" \
    "pydantic>=2.6" "pydantic-settings>=2.2" \
    "fastapi>=0.111" "uvicorn[standard]>=0.30"
COPY src/ ./src/
COPY data/ ./data/
ENV LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake
EXPOSE 8000
CMD ["uvicorn", "src.rag_lc.api:app", "--host", "0.0.0.0", "--port", "8000"]
