"""Async web fetch + article extraction via trafilatura."""

from __future__ import annotations

import asyncio
import json

import httpx
import trafilatura
from loguru import logger
from pydantic import BaseModel, Field


class FetchedArticle(BaseModel):
    """Cleaned article content with metadata."""

    url: str
    title: str = ""
    text: str
    author: str = ""
    date: str = ""
    word_count: int = Field(default=0, ge=0)


class FetchTool:
    """Async fetch + trafilatura extraction, handles timeouts and retries."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout = timeout_s
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; DeepResearchAgent/0.1; "
                "+https://github.com/jakkapat-kingthong/deep-research-agent)"
            )
        }

    async def fetch_many(self, urls: list[str]) -> list[FetchedArticle]:
        """Fetch multiple URLs concurrently, drop failures silently."""
        tasks = [self._fetch_one(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        articles: list[FetchedArticle] = []
        for url, result in zip(urls, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(f"Fetch failed for {url}: {result}")
                continue
            if result is not None:
                articles.append(result)
        return articles

    async def _fetch_one(self, url: str) -> FetchedArticle | None:
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers=self._headers,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            with_metadata=True,
            output_format="json",
        )
        if not extracted:
            return None

        meta = json.loads(extracted)
        text = meta.get("text", "") or ""
        if len(text) < 200:  # drop tiny pages
            return None

        return FetchedArticle(
            url=url,
            title=meta.get("title", "") or "",
            text=text,
            author=meta.get("author", "") or "",
            date=meta.get("date", "") or "",
            word_count=len(text.split()),
        )
