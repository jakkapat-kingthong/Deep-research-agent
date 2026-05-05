"""Planner node: decompose user query into sub-questions."""

from __future__ import annotations

from loguru import logger

from deep_research.llm.groq_llm import GroqLLM
from deep_research.schemas import PlannerOutput
from deep_research.state import ResearchState

PLANNER_SYSTEM = """You are a research planning assistant. Your goal is to take a 
complex research question and decompose it into 3-5 distinct, non-overlapping 
sub-questions that can be answered by searching the web.

Each sub-question should be specific enough to result in high-quality search 
results but broad enough to cover a significant part of the original query."""


async def planner_node(state: ResearchState) -> ResearchState:
    """Decompose query into sub-questions using Groq Llama 3."""
    query = state["query"]
    logger.info(f"Planner: decomposing query: {query!r}")

    # เรียกใช้ Groq Llama 3 70b เพื่อการวางแผนที่ฉลาดที่สุด
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    user_msg = f"Research question:\n{query}\n\nDecompose into sub-questions."

    output, tokens_in, tokens_out = await llm.structured_complete(
        system=PLANNER_SYSTEM,
        user=user_msg,
        response_model=PlannerOutput,
    )

    logger.info(f"Planner: generated {len(output.sub_questions)} sub-questions")
    for i, sq in enumerate(output.sub_questions, 1):
        logger.info(f"  {i}. {sq.text}")

    # คำนวณราคาจำลอง (อัตรา Groq)
    cost = (tokens_in * 0.05 + tokens_out * 0.10) / 1_000_000

    return {
        "sub_questions": output.sub_questions,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "tokens_in": state.get("tokens_in", 0) + tokens_in,
        "tokens_out": state.get("tokens_out", 0) + tokens_out,
    }
