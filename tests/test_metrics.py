"""Observability tests: each answered query is traced with tokens/cost/timings,
and GET /metrics aggregates throughput, latency and cost. Keyless fake path;
traces are written to a temp dir so the repo is never touched.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lc.config import Settings  # noqa: E402
from src.rag_lc.observability import cost_usd  # noqa: E402
from src.rag_lc.pipeline import RAGPipelineLC  # noqa: E402


class _UsageReportingChat(BaseChatModel):
    """Stands in for ChatOpenAI: a canned reply that REPORTS its token usage on the
    message (`usage_metadata`), the way langchain-openai does."""

    @property
    def _llm_type(self) -> str:
        return "usage-stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = AIMessage(
            content="Stub answer [1].",
            usage_metadata={"input_tokens": 321, "output_tokens": 12, "total_tokens": 333},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


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


def test_reported_token_usage_is_billed_exactly(tmp_path, monkeypatch):
    """A provider that reports usage (OpenAI does) is billed on its real token counts,
    not the word-count fallback -- on every langchain-core line, including the 0.2 line
    the Ragas eval stack is pinned to."""
    monkeypatch.setattr("src.rag_lc.pipeline.get_chat_model", lambda s: _UsageReportingChat())
    s = Settings(
        llm_provider="openai",            # priced as gpt-4o-mini; the model itself is stubbed
        embedding_provider="fake",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        enable_cache=False,
    )
    ans = RAGPipelineLC(settings=s).answer("What does the Free plan include?")
    assert ans.tokens["prompt"] == 321
    assert ans.tokens["completion"] == 12
    assert ans.cost_usd == pytest.approx(cost_usd("gpt-4o-mini", 321, 12))
    assert ans.cost_usd > 0


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
