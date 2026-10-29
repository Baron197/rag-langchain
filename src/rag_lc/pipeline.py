"""The RAG pipeline, assembled with LangChain (LCEL).

`answer()` retrieves the top passages, formats them into a numbered context
block, and runs the LCEL generation chain (`prompt | chat_model | parser`) to
produce a grounded, cited answer (or a refusal). It also records a structured
trace per query (retrieval vs generation timing, tokens, USD cost) so the same
"production" observability story as the from-scratch repo holds here. Returns a
structured `Answer` with citations, latency, tokens and cost.

Follow-up questions are supported by passing the recent chat `history`: a second
LCEL chain (`CONDENSE_PROMPT | chat_model | parser`) rewrites the follow-up into
a standalone question before retrieval (see conversation.py). History is never
used as evidence for the answer.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import replace
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .components import (
    FakeGroundedChatModel,
    build_retriever,
    build_vectorstore,
    get_chat_model,
    get_embeddings,
)
from .config import Settings, get_settings
from .conversation import (
    MAX_REWRITE_CHARS,
    clean_rewrite,
    heuristic_rewrite,
    history_key,
    needs_rewrite,
    normalize_history,
    to_messages,
)
from .ingest import load_and_split
from .observability import Trace, Tracer, cost_usd
from .prompts import CONDENSE_PROMPT, RAG_PROMPT
from .schemas import Answer, Citation

logger = logging.getLogger("rag_lc.pipeline")


class _UsageCounter(BaseCallbackHandler):
    """Sums the token usage a chat model reports (`AIMessage.usage_metadata`) during
    one chain run.

    Same idea as langchain-core's `get_usage_metadata_callback`, which only exists
    from 0.3.49 -- but the eval stack is pinned to the 0.2 line by Ragas 0.1.x, so a
    plain callback handler keeps real OpenAI token counts on every supported line.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.prompt_tokens += usage.get("input_tokens", 0)
                self.completion_tokens += usage.get("output_tokens", 0)


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
        # The follow-up rewrite chain: history + question -> standalone question.
        self.condense_chain = CONDENSE_PROMPT | self.chat_model | StrOutputParser()
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

    def clear_cache(self) -> None:
        """Drop every cached answer (e.g. between eval runs, for clean cost/latency)."""
        with self._cache_lock:
            self._cache.clear()

    def _retriever_for(self, top: int):
        """The shared retriever, or a per-query one when `k` overrides the default
        `top_k` -- so a larger k actually retrieves more (matching the from-scratch
        product) instead of being silently capped by the pre-built retriever."""
        if top == self.settings.top_k:
            return self.retriever
        return build_retriever(
            self.settings.model_copy(update={"top_k": top}), self.vectorstore, self.docs
        )

    def answer(
        self, question: str, k: int | None = None, history: list[dict] | None = None
    ) -> Answer:
        """(Rewrite a follow-up,) retrieve context, generate a grounded answer, trace.

        `history` is the recent conversation, oldest first, as
        ``[{"role": "user" | "assistant", "content": str}, ...]``. It only turns a
        follow-up into a standalone question; the answer is grounded in the
        retrieved passages alone.
        """
        timings: dict[str, float] = {}
        top = k or self.settings.top_k
        turns = normalize_history(history, self.settings.history_window)
        # Only a question that leans on earlier turns ("does deleting IT ...?") is
        # rewritten; a standalone one ignores the history (no rewrite call, and it
        # shares the single-turn cache entry). A follow-up's key includes the raw
        # history, so nothing is spent before the lookup and every hit is free.
        follow_up = needs_rewrite(question, turns)
        key = (question, top, history_key(turns) if follow_up else ())

        if self.settings.enable_cache:
            hit = self._cache_get(key)
            if hit is not None:
                # Served from cache: free and instant. Not traced, so /metrics and
                # /analytics reflect real (uncached) query cost and latency.
                return replace(hit, cost_usd=0.0, cached=True,
                               latency_ms=0.0, timings_ms={"cache_hit": 0.0})

        search_question, rewrite_pt, rewrite_ct = question, 0, 0
        if follow_up:
            tr = perf_counter()
            search_question, rewrite_pt, rewrite_ct = self._rewrite(question, turns)
            timings["rewrite"] = round((perf_counter() - tr) * 1000, 2)

        t0 = perf_counter()
        docs = list(self._retriever_for(top).invoke(search_question))[:top]
        timings["retrieval"] = round((perf_counter() - t0) * 1000, 2)

        context = self._format(docs) if docs else "(no relevant context found)"
        inputs = {"context": context, "question": search_question}

        t1 = perf_counter()
        text, prompt_tokens, completion_tokens = self._generate(inputs)
        timings["generation"] = round((perf_counter() - t1) * 1000, 2)
        text = text.strip()

        # The rewrite ran on the same chat model, so it is priced the same way.
        cost = cost_usd(self._model_label(), prompt_tokens, completion_tokens) + cost_usd(
            self._model_label(), rewrite_pt, rewrite_ct
        )
        tokens = {"prompt": prompt_tokens, "completion": completion_tokens}
        if follow_up:
            tokens["rewrite"] = rewrite_pt + rewrite_ct
        citations = self._citations(docs)
        latency_ms = round(sum(timings.values()), 2)

        self.tracer.record(
            Trace(
                question=question,
                timings_ms=timings,
                tokens=tokens,
                cost_usd=cost,
                n_contexts=len(docs),
                retrieval_mode=self.settings.retrieval_mode,
                sources=[c.source for c in citations],
                answer=text,
                standalone_question=search_question,
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
            tokens=tokens,
            timings_ms=timings,
            cached=False,
            standalone_question=search_question,
        )
        if self.settings.enable_cache:
            self._cache_put(key, ans)
        return ans

    def _generate(self, inputs: dict) -> tuple[str, int, int]:
        """Run the generation chain: (answer_text, prompt_tokens, completion_tokens)."""
        return self._invoke_counted(self.gen_chain, RAG_PROMPT, inputs)

    def _rewrite(self, question: str, turns: list[dict[str, str]]) -> tuple[str, int, int]:
        """Resolve a follow-up: (standalone_question, prompt_tokens, completion_tokens).

        The keyless fake chat model can't rewrite, so it uses the deterministic
        heuristic (no call, 0 tokens). A real model runs the LCEL condense chain;
        an empty/overlong reply falls back to the heuristic (keeping the tokens it
        cost), and a provider error falls back too, so a rewrite never fails a query.
        """
        if isinstance(self.chat_model, FakeGroundedChatModel):
            return heuristic_rewrite(question, turns), 0, 0
        inputs = {"chat_history": to_messages(turns), "question": question}
        try:
            text, pt, ct = self._invoke_counted(self.condense_chain, CONDENSE_PROMPT, inputs)
        except Exception:  # noqa: BLE001 -- a failed rewrite must not fail the whole query
            logger.warning("follow-up rewrite failed; using the heuristic", exc_info=True)
            return heuristic_rewrite(question, turns), 0, 0
        standalone = clean_rewrite(text)
        if not standalone or len(standalone) > MAX_REWRITE_CHARS:
            return heuristic_rewrite(question, turns), pt, ct
        return standalone, pt, ct

    @staticmethod
    def _invoke_counted(chain, prompt, inputs: dict) -> tuple[str, int, int]:
        """Run an LCEL chain and return (text, prompt_tokens, completion_tokens).

        Real token usage is captured when the provider reports it (e.g. OpenAI) by a
        callback handler (`_UsageCounter`) attached to this one run; on the keyless
        `fake`/local `hf` paths no usage is reported, so we fall back to a word-count
        approximation (which still costs $0). The LCEL chains themselves are unchanged,
        and a provider error from `invoke` propagates once (never re-invoked).
        """
        counter = _UsageCounter()
        text = chain.invoke(inputs, config={"callbacks": [counter]})
        prompt_tokens, completion_tokens = counter.prompt_tokens, counter.completion_tokens
        if prompt_tokens == 0 and completion_tokens == 0:
            # No usage reported (fake/hf) -> approximate from text so counts are nonzero.
            prompt_tokens = len(prompt.invoke(inputs).to_string().split())
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
