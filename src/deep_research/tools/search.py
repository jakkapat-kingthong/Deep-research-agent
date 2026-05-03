"""Web search tool: Tavily primary, DuckDuckGo fallback."""

from __future__ import annotations

from typing import Literal

from duckduckgo_search import AsyncDDGS
from loguru import logger
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from deep_research.config import settings


class SearchResult(BaseModel):
    """Single search result with provenance."""

    title: str
    url: str
    snippet: str = Field(default="")
    source: Literal["tavily", "duckduckgo"] = "tavily"
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchTool:
    """Web search with Tavily primary + DDG fallback."""

    def __init__(self) -> None:
        self._tavily = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY.get_secret_value())

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search web, return top results. Falls back to DDG on Tavily failure."""
        try:
            return await self._tavily_search(query, max_results)
        except Exception as exc:
            logger.warning(f"Tavily failed, falling back to DDG: {exc}")
            return await self._ddg_search(query, max_results)

    async def _tavily_search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = await self._tavily.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r["url"],
                snippet=r.get("content", ""),
                source="tavily",
                score=float(r.get("score", 0.0)),
            )
            for r in resp.get("results", [])
        ]

    async def _ddg_search(self, query: str, max_results: int) -> list[SearchResult]:
        async with AsyncDDGS() as ddgs:
            results = []
            responses = await ddgs.atext(query, max_results=max_results)
            for r in responses:
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r["href"],
                        snippet=r.get("body", ""),
                        source="duckduckgo",
                    )
                )
            return results
