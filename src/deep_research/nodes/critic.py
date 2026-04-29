"""Critic node: inspect report, decide if another iteration is needed."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from deep_research.llm.groq_llm import GroqLLM
from deep_research.state import ResearchState

MAX_ITERATIONS = 2

CRITIC_SYSTEM = """You are a research critic. Given a query and a draft report,
decide if the report adequately answers the query.

If the report has significant gaps OR cites weak/indirect sources for important
claims, set needs_more_research=true and describe the gap in <=100 words.
Otherwise, set needs_more_research=false.

Be conservative — only request more research if there's a real gap, not just
imperfection. The goal is a good answer, not a perfect one."""


class CriticOutput(BaseModel):
    needs_more_research: bool
    feedback: str = Field(default="", max_length=500)
    additional_queries: list[str] = Field(default_factory=list, max_length=3)


async def critic_node(state: ResearchState) -> ResearchState:
    """Evaluate draft report using Groq Llama 3 8b."""
    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        logger.info(f"Critic: max iterations ({MAX_ITERATIONS}) reached, stopping")
        return {"iteration": iteration, "critic_feedback": "max_iterations"}

    query = state["query"]
    claims = state.get("claims", [])
    summary = state.get("summary", "")

    claims_str = "\n".join(f"- {c.text} (sources: {c.source_ids})" for c in claims)
    user_msg = f"Query: {query}\n\nSummary: {summary}\n\nClaims:\n{claims_str}\n\nEvaluate."

    # ใช้ Llama 3 8b รุ่นเล็กก็พอสำหรับการตรวจงาน
    llm = GroqLLM(model="llama-3.1-8b-instant")
    output, t_in, t_out = await llm.structured_complete(
        system=CRITIC_SYSTEM,
        user=user_msg,
        response_model=CriticOutput,
    )

    cost = (t_in * 0.05 + t_out * 0.10) / 1_000_000

    if output.needs_more_research:
        logger.info(f"Critic: needs more research — {output.feedback}")
    else:
        logger.info("Critic: report is adequate")

    return {
        "iteration": iteration + 1,
        "critic_feedback": output.feedback if output.needs_more_research else "done",
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "tokens_in": state.get("tokens_in", 0) + t_in,
        "tokens_out": state.get("tokens_out", 0) + t_out,
    }
