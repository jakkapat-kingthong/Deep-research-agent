"""LangGraph state — the single source of truth passed between nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from deep_research.schemas import Claim, Source, SubQuestion
from deep_research.tools.rerank import Chunk


class ResearchState(TypedDict, total=False):
    """State container passed between LangGraph nodes.

    `total=False` means all keys are optional — nodes fill them progressively.
    """

    # Input
    query: str
    budget_usd: float

    # Planner output
    sub_questions: list[SubQuestion]

    # Searcher + fetcher output
    chunks: list[Chunk]
    sources: list[Source]

    # Synthesizer output
    claims: list[Claim]
    summary: str

    # Critic loop state
    iteration: int
    critic_feedback: str

    # Cost + telemetry
    cost_usd: float
    tokens_in: int
    tokens_out: int

    # LangGraph message channel (for optional chat UI)
    messages: Annotated[list, add_messages]
