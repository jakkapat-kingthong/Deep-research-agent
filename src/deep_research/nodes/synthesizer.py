"""Synthesizer node: produce grounded Report from chunks."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from deep_research.llm.groq_llm import GroqLLM
from deep_research.schemas import Claim, Report
from deep_research.state import ResearchState

SYNTHESIZER_SYSTEM = """You are a research synthesizer. Given a research question
and a set of source chunks, produce a structured report where EVERY claim is
grounded in specific source_ids.

Rules:
1. Every Claim.source_ids MUST contain only source_ids that appear in the
   provided Sources list. Do NOT invent source_ids.
2. If a claim cannot be supported by the provided sources, DO NOT include it.
3. Prefer multiple supporting sources where available (confidence > 0.9).
4. Summary should be 2-4 sentences synthesizing the key findings.
5. Output at least 3 claims, maximum 12.

Source format in the user message:
  [src_XX] (title) ... chunk text ...

The output format will strictly be a JSON object matching the defined schema."""


class _ReportStub(BaseModel):
    """Pydantic stub — summary + claims only (sources set by node)"""

    summary: str = Field(min_length=20, max_length=1500)
    claims: list[Claim] = Field(min_length=1, max_length=12)


async def synthesizer_node(state: ResearchState) -> ResearchState:
    """Generate grounded Report from query + chunks + sources using Groq."""
    query = state["query"]
    chunks = state["chunks"]
    sources = state["sources"]

    # Build user message: question + labeled source chunks
    source_block = "\n\n".join(
        f"[{_source_id_from_chunk_id(c.chunk_id)}] ({c.title})\n{c.text}" for c in chunks
    )
    user_msg = (
        f"Research question:\n{query}\n\n"
        f"Available sources:\n{source_block}\n\n"
        "Produce a grounded Report."
    )

    # Inline available source_ids for model convenience
    available_ids = sorted({_source_id_from_chunk_id(c.chunk_id) for c in chunks})
    user_msg += f"\n\nValid source_ids to cite: {available_ids}"

    # ใช้ Llama 3 70b เพื่อการเขียนที่ไหลลื่นและถูกต้อง
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    report_stub, t_in, t_out = await llm.structured_complete(
        system=SYNTHESIZER_SYSTEM,
        user=user_msg,
        response_model=_ReportStub,
        max_tokens=4096,
    )

    report = Report(
        query=query,
        summary=report_stub.summary,
        claims=report_stub.claims,
        sources=sources,
    )

    missing = report.validate_grounding()
    if missing:
        logger.warning(
            f"Synthesizer produced claims citing unknown sources {missing}. Filtering them out."
        )
        report.claims = [
            c for c in report.claims if all(sid not in missing for sid in c.source_ids)
        ]

    cost = (t_in * 0.05 + t_out * 0.10) / 1_000_000

    return {
        "claims": report.claims,
        "summary": report.summary,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
        "tokens_in": state.get("tokens_in", 0) + t_in,
        "tokens_out": state.get("tokens_out", 0) + t_out,
    }


def _source_id_from_chunk_id(chunk_id: str) -> str:
    """Extract 'src_XX' from 'src_XX__rest_of_id'."""
    return chunk_id.split("__", 1)[0]
