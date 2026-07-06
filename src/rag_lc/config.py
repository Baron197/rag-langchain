"""Typed configuration (pydantic-settings), read from environment / `.env`.

Defaults keep the app **keyless and offline** (a deterministic `fake` embedding +
chat model, in-memory vector store), so tests and CI run with no API key. Switch
providers to `openai` or `hf` (local Hugging Face) for real models.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Providers: "fake" (default, offline) | "openai" (paid) | "hf" (local, free).
    llm_provider: str = "fake"
    embedding_provider: str = "fake"

    # OpenAI (langchain-openai)
    openai_api_key: str | None = None
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Hugging Face (langchain-huggingface; local, free, no key)
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Retrieval / chunking
    retrieval_mode: str = "vector"   # "vector" | "hybrid" (EnsembleRetriever: BM25 + vector)
    top_k: int = 4
    chunk_size: int = 1500           # characters (RecursiveCharacterTextSplitter)
    chunk_overlap: int = 150

    # Vector store: "memory" (default, in-process InMemoryVectorStore) or "pgvector"
    # (Postgres + pgvector via langchain-postgres). pgvector PERSISTS the embeddings,
    # so you ingest once and queries connect to the collection without re-embedding.
    vector_backend: str = "memory"
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    pg_collection: str = "rag_lc_docs"

    # Answer cache (LRU): repeated questions are served from memory at zero cost.
    enable_cache: bool = True
    cache_size: int = 256

    # API auth (optional): if set, the cost/mutation endpoints (/query, /ingest,
    # /upload) require a matching `X-API-Key` header. Empty (default) = open, so
    # keyless/local use is unchanged. The Streamlit UI forwards it automatically.
    api_key: str = ""

    docs_dir: Path = Path("data/docs")
    trace_dir: Path = Path("traces")  # per-query JSONL traces (observability / metrics)
    # Where the eval harness writes reports and /eval-results reads them; anchored
    # to the repo root (not the process CWD) so reader and writer always agree.
    eval_results_dir: Path = Path(__file__).resolve().parents[2] / "eval" / "results"


def get_settings() -> Settings:
    """Return a fresh Settings (re-reads env/.env)."""
    return Settings()
