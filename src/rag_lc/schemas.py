"""Small result records returned by the pipeline / API."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """A numbered source reference shown with an answer."""

    n: int
    source: str
    snippet: str


@dataclass
class Answer:
    """The structured result of one query."""

    question: str
    answer: str
    citations: list[Citation]
    n_contexts: int
    retrieval_mode: str
    latency_ms: float
    cost_usd: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)      # prompt / completion
    timings_ms: dict[str, float] = field(default_factory=dict)  # retrieval / generation
    cached: bool = False  # True when served from the LRU answer cache (cost 0)
