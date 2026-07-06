"""The RAG pipeline, assembled with LangChain (LCEL).

`answer()` retrieves the top passages, formats them into a numbered context
block, and runs the LCEL generation chain (`prompt | chat_model | parser`) to
produce a grounded, cited answer (or a refusal). It also records a structured
trace per query (retrieval vs generation timing, tokens, USD cost) so the same
"production" observability story as the from-scratch repo holds here. Returns a
structured `Answer` with citations, latency, tokens and cost.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .components import (
    build_retriever,
    build_vectorstore,
    get_chat_model,
    get_embeddings,
)
from .config import Settings, get_settings
from .ingest import load_and_split
from .observability import Trace, Tracer, cost_usd
from .prompts import RAG_PROMPT
from .schemas import Answer, Citation


class RAGPipelineLC:
    """LangChain RAG pipeline: ingest -> retrieve -> LCEL generate -> trace -> Answer."""

    def __init__(self, settings: Settings | None = None, *, reset_index: bool = False) -> None:
        self.settings = settings or get_settings()
        self.embeddings = get_embeddings(self.settings)
        self.docs = load_and_split(self.settings)
        # On the pgvector backend, `reset_index=True` (ingest) re-embeds and rewrites
        # the collection; the default (query) just connects to the existing one.
        self.vectorstore = build_vectorstore(
            self.settings, self.embeddings, self.docs, reset=reset_index
        )
        self.retriever = build_retriever(self.settings, self.vectorstore, self.docs)
        self.chat_model = get_chat_model(self.settings)
        # The LCEL generation chain: prompt -> chat model -> string output.
        self.gen_chain = RAG_PROMPT | self.chat_model | StrOutputParser()
        self.tracer = Tracer(self.settings.trace_dir)
        # LRU cache of recent answers, guarded by a lock for thread-safety.
        self._cache: OrderedDict[tuple, Answer] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: tuple) -> Answer | None:
        with self._cache_lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def _cache_put(self, key: tuple, ans: Answer) -> None:
        with self._cache_lock:
            self._cache[key] = ans
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)

    def answer(self, question: str, k: int | None = None) -> Answer:
        """Retrieve context, generate a grounded answer, and record a trace."""
        timings: dict[str, float] = {}
        top = k or self.settings.top_k

        if self.settings.enable_cache:
            hit = self._cache_get((question, top))
            if hit is not None:
                # Served from cache: free and instant. Not traced, so /metrics and
                # /analytics reflect real (uncached) query cost and latency.
                return replace(hit, cost_usd=0.0, cached=True,
                               latency_ms=0.0, timings_ms={"cache_hit": 0.0})

        t0 = perf_counter()
        docs = list(self.retriever.invoke(question))[:top]
        timings["retrieval"] = round((perf_counter() - t0) * 1000, 2)

        context = self._format(docs) if docs else "(no relevant context found)"
        inputs = {"context": context, "question": question}

        t1 = perf_counter()
        text, prompt_tokens, completion_tokens = self._generate(inputs)
        timings["generation"] = round((perf_counter() - t1) * 1000, 2)
        text = text.strip()

        cost = cost_usd(self._model_label(), prompt_tokens, completion_tokens)
        citations = self._citations(docs)
        latency_ms = round(sum(timings.values()), 2)

        self.tracer.record(
            Trace(
                question=question,
                timings_ms=timings,
                tokens={"prompt": prompt_tokens, "completion": completion_tokens},
                cost_usd=cost,
                n_contexts=len(docs),
                retrieval_mode=self.settings.retrieval_mode,
                sources=[c.source for c in citations],
                answer=text,
            )
        )
        ans = Answer(
            question=question,
            answer=text,
            citations=citations,
            n_contexts=len(docs),
            retrieval_mode=self.settings.retrieval_mode,
            latency_ms=latency_ms,
            cost_usd=cost,
            tokens={"prompt": prompt_tokens, "completion": completion_tokens},
            timings_ms=timings,
            cached=False,
        )
        if self.settings.enable_cache:
            self._cache_put((question, top), ans)
        return ans

    def _generate(self, inputs: dict) -> tuple[str, int, int]:
        """Run the LCEL chain and return (text, prompt_tokens, completion_tokens).

        Real token usage is captured when the provider reports it (e.g. OpenAI) via
        LangChain's usage-metadata callback; on the keyless `fake`/local `hf` paths
        no usage is reported, so we fall back to a word-count approximation (which
        still costs $0). This keeps the LCEL chain itself unchanged.
        """
        try:
            from langchain_core.callbacks import get_usage_metadata_callback

            with get_usage_metadata_callback() as cb:
                text = self.gen_chain.invoke(inputs)
            usage = cb.usage_metadata  # {model_name: {input_tokens, output_tokens, ...}}
        except Exception:  # noqa: BLE001 -- older langchain-core without the callback
            text = self.gen_chain.invoke(inputs)
            usage = {}

        prompt_tokens = sum(u.get("input_tokens", 0) for u in usage.values())
        completion_tokens = sum(u.get("output_tokens", 0) for u in usage.values())
        if prompt_tokens == 0 and completion_tokens == 0:
            # No usage reported (fake/hf) -> approximate from text so counts are nonzero.
            prompt_tokens = len(RAG_PROMPT.invoke(inputs).to_string().split())
            completion_tokens = len(text.split())
        return text, prompt_tokens, completion_tokens

    def _model_label(self) -> str:
        """The model name used for pricing (keyless/local -> priced at $0)."""
        if self.settings.llm_provider == "openai":
            return self.settings.openai_llm_model
        if self.settings.llm_provider == "hf":
            return self.settings.hf_llm_model
        return "fake-llm"

    @staticmethod
    def _format(docs: list[Document]) -> str:
        """Number the retrieved passages for citation, e.g. `[1] (source: x)`."""
        return "\n\n".join(
            f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )

    @staticmethod
    def _citations(docs: list[Document]) -> list[Citation]:
        out = []
        for i, d in enumerate(docs, start=1):
            snippet = d.page_content.strip().replace("\n", " ")[:200]
            out.append(Citation(n=i, source=d.metadata.get("source", "?"), snippet=snippet))
        return out
