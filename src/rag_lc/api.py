"""FastAPI service exposing the LangChain RAG pipeline.

Endpoints:
  GET  /health       -> liveness + active providers + index size
  POST /ingest       -> rebuild the in-memory index from DOCS_DIR
  POST /upload       -> save uploaded file(s) into DOCS_DIR, then re-index
  POST /query        -> grounded answer with citations, tokens, cost and latency
  GET  /metrics      -> aggregate cost/latency/throughput from recorded traces
  GET  /analytics    -> per-query trace records for the analytics dashboard
  GET  /eval-results -> evaluation reports (retrieval/Ragas/A-B) for the eval dashboard

The pipeline (and its in-memory index) is built once at first use and reused; it
is rebuilt on /ingest and /upload. Access is guarded by a lock (FastAPI runs sync
handlers in a threadpool, so build/reset could otherwise race).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipelineLC

logger = logging.getLogger("rag_lc.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="RAG Knowledge Assistant (LangChain variant)", version="0.4.0")
# Lazily-built singleton pipeline, shared across requests (rebuilt on /ingest &
# /upload). Guarded by a lock against threadpool races.
_pipeline: RAGPipelineLC | None = None
_pipeline_lock = threading.Lock()

# Documents the loader can read (see ingest._read_file).
ALLOWED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024      # per-file cap
MAX_REQUEST_BYTES = 50 * 1024 * 1024     # whole-request cap


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject oversized requests up front (HTTP 413) before the body is parsed."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request too large (max {MAX_REQUEST_BYTES // (1024 * 1024)} MB)."},
        )
    return await call_next(request)


def get_pipeline() -> RAGPipelineLC:
    """Return the shared pipeline, building it on first use (thread-safe)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipelineLC()
        return _pipeline


def _set_pipeline(p: RAGPipelineLC) -> None:
    """Swap in a freshly-built pipeline (thread-safe)."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = p


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    n: int
    source: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    n_contexts: int
    retrieval_mode: str
    latency_ms: float
    cost_usd: float
    tokens: dict
    timings_ms: dict
    cached: bool


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {
        "status": "ok",
        "framework": "langchain",
        "llm_provider": p.settings.llm_provider,
        "embedding_provider": p.settings.embedding_provider,
        "retrieval_mode": p.settings.retrieval_mode,
        "vector_backend": p.settings.vector_backend,
        "indexed_chunks": len(p.docs),
    }


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Optional API-key gate for the cost/mutation endpoints (/query, /ingest,
    /upload). No-op when `api_key` is unset (the default) -- keyless/local use is
    unchanged. When set, requests must carry a matching `X-API-Key` header;
    otherwise 401. Read-only endpoints and /health stay open."""
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def run_ingest() -> dict:
    """Rebuild the in-memory index from DOCS_DIR (re-load + re-split + re-embed)."""
    try:
        # reset_index=True re-embeds and (on pgvector) rewrites the collection.
        p = RAGPipelineLC(reset_index=True)  # sync handler -> runs in FastAPI's threadpool
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    _set_pipeline(p)
    return {"indexed_chunks": len(p.docs)}


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload(files: Annotated[list[UploadFile], File(...)]) -> dict:
    """Save uploaded document(s) into DOCS_DIR, then rebuild the in-memory index.

    Filenames are reduced to a bare name (no directory components) to prevent path
    traversal; oversized or unsupported files are skipped. The rebuild runs in the
    threadpool so this async handler never stalls the event loop.
    """
    docs_dir = Path(get_settings().docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    skipped: list[str] = []
    for f in files:
        name = Path(f.filename or "").name  # strip any path -> just the filename
        if not name or Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            skipped.append(f.filename or "(unnamed)")
            continue
        content = await f.read(MAX_UPLOAD_BYTES + 1)  # bounded read (request already capped)
        if len(content) > MAX_UPLOAD_BYTES:
            skipped.append(f"{name} (over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit)")
            continue
        (docs_dir / name).write_bytes(content)
        saved.append(name)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files uploaded. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    try:
        p = await run_in_threadpool(lambda: RAGPipelineLC(reset_index=True))  # rebuild off event loop
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion after upload failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    _set_pipeline(p)
    return {"saved": saved, "skipped": skipped, "indexed_chunks": len(p.docs)}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    try:
        ans = get_pipeline().answer(req.question, req.k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(
        question=ans.question,
        answer=ans.answer,
        citations=[CitationModel(n=c.n, source=c.source, snippet=c.snippet) for c in ans.citations],
        n_contexts=ans.n_contexts,
        retrieval_mode=ans.retrieval_mode,
        latency_ms=ans.latency_ms,
        cost_usd=ans.cost_usd,
        tokens=ans.tokens,
        timings_ms=ans.timings_ms,
        cached=ans.cached,
    )


@app.get("/metrics")
def metrics() -> dict:
    """Aggregate cost/latency/throughput across all recorded query traces."""
    return get_pipeline().tracer.aggregate()


@app.get("/analytics")
def analytics(limit: int = 2000) -> dict:
    """Per-query trace records (most recent, bounded by `limit`) for the analytics
    dashboard. Only uncached queries are traced, so this reflects real work."""
    limit = max(1, min(limit, 5000))
    rows = get_pipeline().tracer.records(limit)
    queries = [
        {
            "ts": r.get("ts"),
            "question": r.get("question", ""),
            "answer_preview": (r.get("answer", "") or "")[:240],
            "timings_ms": r.get("timings_ms", {}),
            "tokens": r.get("tokens", {}),
            "cost_usd": r.get("cost_usd", 0.0),
            "n_contexts": r.get("n_contexts", 0),
            "sources": r.get("sources", []),
        }
        for r in rows
    ]
    return {"count": len(queries), "queries": queries}


@app.get("/eval-results")
def eval_results(limit: int = 50) -> dict:
    """Evaluation reports written by the eval harness (from `eval_results_dir`),
    newest first, for the Evaluation dashboard. Read-only."""
    limit = max(1, min(limit, 200))
    results_dir = Path(get_settings().eval_results_dir)
    evals: list[dict] = []
    compares: list[dict] = []
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            data["_name"] = path.name
            if path.name.startswith("compare-"):
                compares.append(data)
            elif path.name.startswith("eval-"):
                evals.append(data)
    return {
        "results_dir": str(results_dir),
        "eval_runs": evals[:limit],
        "compare_runs": compares[:limit],
    }
