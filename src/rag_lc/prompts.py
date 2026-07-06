"""The grounding contract, expressed as a LangChain `ChatPromptTemplate`.

Same rules as the from-scratch version: answer only from the numbered context,
cite the passages used, and refuse with a fixed sentence when the context does
not contain the answer.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a precise documentation assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Rules:\n"
    "1. Ground every claim in the context. Cite the passages you used inline, e.g. [1], [2].\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the documentation to answer that.\" "
    "Do not use outside knowledge.\n"
    "3. Be concise and specific. Prefer steps and concrete details over fluff."
)

HUMAN_TEMPLATE = (
    "Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"
)

# The chat prompt used by the LCEL generation chain (prompt | llm | parser).
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
)
