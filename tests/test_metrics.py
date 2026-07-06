"""Observability tests: each answered query is traced with tokens/cost/timings,
and GET /metrics aggregates throughput, latency and cost. Keyless fake path;
traces are written to a temp dir so the repo is never touched.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lc.config import Settings  # noqa: E402
from src.rag_lc.pipeline import RAGPipelineLC  # noqa: E402


def test_answer_carries_cost_tokens_timings(tmp_path):
    """Every answer reports (zero) cost, nonzero token counts, and stage timings."""
    s = Settings(
        llm_provider="fake",
        embedding_provider="fake",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        top_k=4,
    )
    ans = RAGPipelineLC(settings=s).answer("What does the Free plan include?")
    assert ans.cost_usd == 0.0                       # keyless fake path is free
    assert ans.tokens["prompt"] > 0                  # approximated from text
    assert set(ans.timings_ms) == {"retrieval", "generation"}


def _client(tmp_path, monkeypatch) -> TestClient:
    """A TestClient whose pipeline writes traces to a temp dir (keyless fake)."""
    monkeypatch.setenv("DOCS_DIR", "data/docs")
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    from src.rag_lc import api
    importlib.reload(api)  # rebuild module-level app + singleton with the new env
    return TestClient(api.app)


def test_metrics_aggregates_after_queries(tmp_path, monkeypatch):
    """After answering queries, /metrics reports throughput, latency and cost."""
    client = _client(tmp_path, monkeypatch)
    for q in [
        "What does the Free plan include?",
        "What is the per-second rate limit on the Growth plan?",
    ]:
        assert client.post("/query", json={"question": q}).status_code == 200
    m = client.get("/metrics").json()
    assert m["queries"] == 2
    assert m["avg_latency_ms"] >= 0
    assert "p95_latency_ms" in m
    assert m["total_cost_usd"] == 0.0        # keyless path
    assert m["avg_contexts"] >= 1
