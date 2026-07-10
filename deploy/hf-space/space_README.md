---
title: RAG Knowledge Assistant (LangChain)
emoji: 🔗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LangChain (LCEL) RAG over docs — keyless demo (FastAPI + Streamlit)
---

# RAG Knowledge Assistant — LangChain variant (live demo)

A keyless (`fake`-provider) demo of a RAG service built **idiomatically with
LangChain (LCEL)** — LangChain retrievers, an `InMemoryVectorStore`, an LCEL
generation chain, and hybrid (vector + BM25) retrieval. Grounded answers with
inline citations, plus per-query cost / latency / token telemetry, served by a
FastAPI backend behind a multipage Streamlit UI.

On this keyless path the **retrieval, citations and telemetry are real**; the
answer *text* is a deterministic stand-in (the fake chat model echoes the top
passage). Real generated answers and Ragas metrics are on the OpenAI path.

**Source & docs:** https://github.com/Baron197/rag-langchain
**Framework-free sibling:** https://github.com/Baron197/rag-knowledge-assistant

---

*Want real generated answers here?* Uncomment the `hf` block in the `Dockerfile`,
set `LLM_PROVIDER=hf` and `EMBEDDING_PROVIDER=hf` in **Settings → Variables**, and
restart. It runs a small local open-source model (Qwen2.5-0.5B) — real answers, no
API key, but slower on the free CPU tier.
