"""The grounding contract, expressed as a LangChain `ChatPromptTemplate`.

Same rules as the from-scratch version: answer only from the numbered context,
cite the passages used, and refuse with a fixed sentence when the context does
not contain the answer.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# Follow-up rewriting (see conversation.py). The model only RESOLVES references in
# the latest question using the conversation; it must not answer or add facts, and
# the history is never shown to the answering step as evidence.
CONDENSE_SYSTEM_PROMPT = (
    "You rewrite the user's latest question into a standalone question for a "
    "documentation search engine. Use the conversation ONLY to resolve references "
    "(pronouns such as 'it', 'they' or 'that', and omitted subjects). Rules:\n"
    "1. Do not answer the question.\n"
    "2. Do not add details the conversation does not state.\n"
    "3. If the question already stands on its own, return it unchanged.\n"
    "4. Reply with the standalone question only: one line, no quotes, no preamble."
)

# The prior turns arrive as real chat messages via MessagesPlaceholder -- the same
# shape LangChain's create_history_aware_retriever uses.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "Latest question: {question}\n\nStandalone question:"),
    ]
)
