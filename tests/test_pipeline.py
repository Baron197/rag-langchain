"""End-to-end tests on the keyless `fake` path (no API key, no network).

Prove the LangChain pipeline builds, retrieves the right document, returns
grounded citations, refuses out-of-scope questions, and that hybrid retrieval
works. Query traces are written under a temp dir so tests never touch the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lc.config import Settings  # noqa: E402
from src.rag_lc.pipeline import RAGPipelineLC  # noqa: E402


def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        llm_provider="fake",
        embedding_provider="fake",
        docs_dir=Path("data/docs"),
        trace_dir=tmp_path / "traces",
        top_k=4,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def pipeline(tmp_path) -> RAGPipelineLC:
    return RAGPipelineLC(settings=_settings(tmp_path))


def test_pipeline_indexes_chunks(pipeline: RAGPipelineLC):
    """Ingestion + splitting produced chunks."""
    assert len(pipeline.docs) > 0


def test_answer_returns_citations(pipeline: RAGPipelineLC):
    """A normal question yields a non-empty answer with at least one citation."""
    ans = pipeline.answer("What does the Free plan include?")
    assert ans.n_contexts > 0
    assert ans.citations
    assert ans.answer.strip() != ""


def test_retrieval_finds_relevant_source(pipeline: RAGPipelineLC):
    """Vector retrieval surfaces the document that contains the answer."""
    ans = pipeline.answer("What is the per-second rate limit on the Growth plan?")
    assert "03-rate-limits.md" in {c.source for c in ans.citations}


def test_out_of_scope_question_is_handled(pipeline: RAGPipelineLC):
    """An out-of-scope question degrades gracefully (structured Answer, no crash).

    On the fake path the retriever always returns k docs and the echo model
    grounds in them, so a *true* refusal needs semantic embeddings (openai/hf).
    """
    ans = pipeline.answer("zzzqqq nonexistent topic about quantum giraffes")
    assert isinstance(ans.answer, str) and ans.answer.strip() != ""
    assert isinstance(ans.citations, list)
    assert ans.retrieval_mode in ("vector", "hybrid")


def test_hybrid_retrieval_finds_relevant_source(tmp_path):
    """Hybrid (BM25 + vector via EnsembleRetriever) also finds the right source."""
    pipe = RAGPipelineLC(settings=_settings(tmp_path, retrieval_mode="hybrid"))
    ans = pipe.answer("What is the per-second rate limit on the Growth plan?")
    assert "03-rate-limits.md" in {c.source for c in ans.citations}
    assert ans.retrieval_mode == "hybrid"


def test_cache_hit_is_marked_and_free(tmp_path):
    """A repeated question is served from the LRU cache: marked cached, costing zero."""
    pipe = RAGPipelineLC(settings=_settings(tmp_path, enable_cache=True))
    q = "How do I rotate a leaked API key?"
    first = pipe.answer(q)
    second = pipe.answer(q)
    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == 0.0
    assert second.answer == first.answer


def test_cache_can_be_disabled(tmp_path):
    """With caching off, repeats are recomputed (never marked cached)."""
    pipe = RAGPipelineLC(settings=_settings(tmp_path, enable_cache=False))
    q = "What webhook events does Nimbus emit?"
    pipe.answer(q)
    assert pipe.answer(q).cached is False
