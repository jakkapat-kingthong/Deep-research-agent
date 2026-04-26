"""Integration tests — hit real Tavily + real URLs. Run manually."""

from __future__ import annotations

import pytest

from deep_research.tools.fetch import FetchTool
from deep_research.tools.search import SearchTool


@pytest.mark.asyncio
async def test_search_returns_results() -> None:
    tool = SearchTool()
    results = await tool.search("Model Context Protocol Anthropic", max_results=3)
    assert len(results) >= 1
    assert all(r.url.startswith("http") for r in results)


@pytest.mark.asyncio
async def test_fetch_wikipedia() -> None:
    tool = FetchTool()
    articles = await tool.fetch_many(["https://en.wikipedia.org/wiki/Model_Context_Protocol"])
    assert len(articles) == 1
    assert articles[0].word_count > 100
