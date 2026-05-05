"""FastMCP server — exposes Deep Research Agent as a tool for MCP clients."""

from __future__ import annotations

from uuid import uuid4

from fastmcp import FastMCP
from loguru import logger

from deep_research.graph import get_graph
from deep_research.logging_config import configure_logging

configure_logging()

mcp = FastMCP("Deep Research Agent")


@mcp.tool()
async def research(query: str, budget_usd: float = 0.30) -> dict:
    """Run a grounded research query and return a report with verified citations.

    Every claim in the output is backed by specific source_ids from the
    sources list. No unsupported claims are included.

    Args:
        query: The research question.
        budget_usd: Maximum spend per query (default $0.30).

    Returns:
        Dict with keys: summary, claims, sources, cost_usd, latency_ms.
    """
    logger.info(f"MCP tool called: {query!r}")
    graph = get_graph()
    result = await graph.ainvoke(
        {"query": query, "budget_usd": budget_usd},
        config={"configurable": {"thread_id": str(uuid4())}},
    )
    return {
        "summary": result.get("summary", ""),
        "claims": [c.model_dump(mode="json") for c in result.get("claims", [])],
        "sources": [s.model_dump(mode="json") for s in result.get("sources", [])],
        "cost_usd": result.get("cost_usd", 0.0),
    }


if __name__ == "__main__":
    mcp.run()
