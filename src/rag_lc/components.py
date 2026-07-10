"""LangChain component factories, with keyless fake implementations.

Everything the pipeline needs is a real LangChain object:

- **Embeddings**: a keyless `KeywordHashEmbeddings` (a proper
  `langchain_core.embeddings.Embeddings` subclass, deterministic, keyword-based),
  or real `OpenAIEmbeddings` / `HuggingFaceEmbeddings`.
- **Chat model**: a keyless `FakeGroundedChatModel` (a real
  `SimpleChatModel` that echoes the retrieved context and refuses when there is
  none), or real `ChatOpenAI` / `ChatHuggingFace`.
- **Vector store**: `InMemoryVectorStore` (default, zero external services) or a
  persistent Postgres/pgvector store (`PGVector`) when `VECTOR_BACKEND=pgvector`.
- **Retriever**: the vector store's retriever, or an `EnsembleRetriever` that
  fuses BM25 (keyword) with the vector retriever for hybrid search.

Real/paid/heavy providers are imported lazily so the keyless path stays light.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class KeywordHashEmbeddings(Embeddings):
    """Deterministic, keyless embeddings: a hashed bag-of-words (L2-normalised).

    Captures keyword overlap (so vector retrieval finds the right document in
    tests) but not deep semantics -- the keyless analogue of a real embedding
    model. Implements the standard LangChain `Embeddings` interface, so it drops
    straight into any vector store.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeGroundedChatModel(SimpleChatModel):
    """Keyless chat model that mimics grounded RAG behaviour.

    A real `SimpleChatModel`: it reads the incoming prompt, echoes the first
    numbered context passage as a stand-in "grounded" answer (with a `[1]`
    citation), and returns the exact refusal sentence when no context was found.
    Lets the whole LCEL chain run with no API key.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-grounded"

    def _call(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> str:
        human = ""
        for m in messages:
            if m.type == "human":
                human = m.content if isinstance(m.content, str) else str(m.content)
        # Match the full "no context" block the prompt emits (HUMAN_TEMPLATE), not a
        # bare substring — otherwise a question or passage that happens to contain
        # this phrase would spuriously trigger the refusal.
        if "Context passages:\n\n(no relevant context found)\n\n" in human:
            return "I don't have enough information in the documentation to answer that."
        snippet = ""
        if "[1]" in human:
            after = human.split("[1]", 1)[1]
            body = after.split("\n", 1)[1] if "\n" in after else after
            snippet = body.split("\n\n", 1)[0].strip()[:280]
        return f"{snippet} [1]" if snippet else "Based on the documentation. [1]"


def get_embeddings(settings: Settings) -> Embeddings:
    """Factory: the LangChain embeddings implementation named in settings."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    if settings.embedding_provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    return KeywordHashEmbeddings()


def get_chat_model(settings: Settings):
    """Factory: the LangChain chat model named in settings."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_llm_model, temperature=0, api_key=settings.openai_api_key
        )
    if settings.llm_provider == "hf":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        llm = HuggingFacePipeline.from_model_id(
            model_id=settings.hf_llm_model,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512, "do_sample": False, "return_full_text": False},
        )
        return ChatHuggingFace(llm=llm)
    return FakeGroundedChatModel()


def _pg_connection_string(dsn: str) -> str:
    """Normalise a Postgres DSN to the SQLAlchemy + psycopg(v3) form langchain-postgres wants."""
    for prefix in ("postgresql://", "postgres://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix):]
    return dsn  # already has a +driver, or a non-standard scheme -> pass through


def build_vectorstore(
    settings: Settings, embeddings: Embeddings, docs: list[Document], *, reset: bool = False
) -> VectorStore:
    """Build the vector store named by `settings.vector_backend`.

    - "memory"   -> an `InMemoryVectorStore` built fresh from the chunked docs.
    - "pgvector" -> a persistent Postgres/pgvector store (`langchain_postgres.PGVector`).
      When `reset` is True (ingest) it embeds the docs and REWRITES the collection;
      otherwise (query) it just CONNECTS to the existing collection, so queries never
      re-embed -- the whole point of a database backend (parity with the from-scratch repo).
    """
    if settings.vector_backend == "pgvector":
        from langchain_postgres import PGVector  # lazy: only needed on the pgvector path

        conn = _pg_connection_string(settings.pg_dsn)
        if reset:
            return PGVector.from_documents(
                docs,
                embedding=embeddings,
                collection_name=settings.pg_collection,
                connection=conn,
                use_jsonb=True,
                pre_delete_collection=True,
            )
        return PGVector(
            embeddings=embeddings,
            collection_name=settings.pg_collection,
            connection=conn,
            use_jsonb=True,
        )
    return InMemoryVectorStore.from_documents(docs, embeddings)


def build_retriever(settings: Settings, vectorstore: VectorStore, docs: list[Document]):
    """Return a retriever: the vector retriever, or a hybrid EnsembleRetriever.

    Hybrid fuses BM25 (keyword) with the vector retriever using LangChain's
    `EnsembleRetriever` (reciprocal-rank-fusion style weighting).
    """
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    # `not docs`: an empty corpus can't build a BM25 index -- fall back to the
    # (empty) vector retriever, which returns [] and drives the standard refusal.
    if settings.retrieval_mode != "hybrid" or not docs:
        return vector_retriever

    # EnsembleRetriever moved to `langchain_classic` in LangChain v1; support both.
    try:
        from langchain.retrievers import EnsembleRetriever  # LangChain 0.3
    except ImportError:  # pragma: no cover
        from langchain_classic.retrievers import EnsembleRetriever  # LangChain 1.x
    # BM25Retriever still ships in langchain-community (sunset in v1) and works;
    # EnsembleRetriever lives in langchain-classic. Both are pinned in requirements.
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = settings.top_k
    return EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])
