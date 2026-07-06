"""LangChain variant of the RAG Knowledge Assistant.

A separate, standalone re-implementation of the from-scratch RAG service, built
idiomatically with **LangChain** (LCEL chains, LangChain retrievers, vector
stores, and chat models). Same product and same behaviour (grounded, cited
answers with a refusal path; keyless `fake` mode; vector or hybrid retrieval),
but expressed with the industry-standard framework instead of hand-written
components.
"""

__version__ = "0.1.0"
