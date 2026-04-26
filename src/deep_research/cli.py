"""Typer CLI for driving the research agent from terminal."""

from __future__ import annotations

import asyncio

import typer
from loguru import logger
from rich.console import Console
from rich.pretty import pprint

from deep_research.config import settings
from deep_research.graph import get_graph
from deep_research.logging_config import configure_logging

app = typer.Typer(help="Deep Research Agent CLI")
console = Console()


@app.command()
def ask(
    query: str = typer.Argument(..., help="Research question"),
    budget: float = typer.Option(settings.DEFAULT_BUDGET_USD, help="Budget cap (USD)"),
) -> None:
    """Run the research agent on a query."""
    configure_logging()
    asyncio.run(_run(query, budget))


async def _run(query: str, budget: float) -> None:
    graph = get_graph()
    result = await graph.ainvoke(
        {"query": query, "budget_usd": budget},
        config={"configurable": {"thread_id": "cli-session"}},
    )
    console.rule("[bold green]Result")
    pprint(result, max_depth=3)
    logger.info(f"Total cost: ${result.get('cost_usd', 0.0):.4f}")


if __name__ == "__main__":
    app()
