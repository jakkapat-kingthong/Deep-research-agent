"""Planner node: decompose query into sub-questions."""

from __future__ import annotations

from loguru import logger

from deep_research.llm.gemini_llm import GeminiLLM
from deep_research.schemas import PlannerOutput
from deep_research.state import ResearchState

PLANNER_SYSTEM = """You are a research planner. Given a complex question,
decompose it into 2-5 distinct, non-overlapping sub-questions that together
fully cover the original question.

Rules:
- Each sub-question must be independently searchable on the web.
- Avoid redundancy — no two sub-questions should seek the same information.
- Favor specificity: include entities, timeframes, and metrics where relevant.
- Output MUST match the provided JSON schema."""


async def planner_node(state: ResearchState) -> ResearchState:
    """Decompose `state.query` into sub-questions."""
    query = state["query"]
    logger.info(f"Planner: decomposing query: {query!r}")

    llm = GeminiLLM()
    user_msg = f"Research question:\n{query}\n\nDecompose into sub-questions."

    output, tokens_in, tokens_out = await llm.structured_complete(
        system=PLANNER_SYSTEM,
        user=user_msg,
        response_model=PlannerOutput,
    )

    logger.info(f"Planner: generated {len(output.sub_questions)} sub-questions")
    for i, sq in enumerate(output.sub_questions, 1):
        logger.info(f"  {i}. {sq.text}")

    # Approx cost: Gemini 1.5 Pro (Prompts > 128k) = $1.25/MTok in, $5.00/MTok out
    # For smaller prompts it's even cheaper, but we'll use the standard rate
    cost = (tokens_in * 1.25 + tokens_out * 5.00) / 1_000_000

    return {
        "sub_questions": output.sub_questions,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "tokens_in": state.get("tokens_in", 0) + tokens_in,
        "tokens_out": state.get("tokens_out", 0) + tokens_out,
    }
