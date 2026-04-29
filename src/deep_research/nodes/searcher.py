"""Searcher node: parallel search across sub-questions + fetch + chunk."""

from __future__ import annotations

import asyncio

from loguru import logger

from deep_research.config import settings
from deep_research.schemas import Source
from deep_research.state import ResearchState
from deep_research.tools.fetch import FetchTool
from deep_research.tools.rerank import Reranker, chunk_text
from deep_research.tools.search import SearchTool


async def searcher_node(state: ResearchState) -> ResearchState:
    """Search web for each sub-question, fetch articles, chunk + rerank."""
    query = state["query"]
    sub_questions = state["sub_questions"]

    search = SearchTool()
    fetch = FetchTool()

    # 1. Parallel search across sub-questions
    search_tasks = [search.search(sq.text, max_results=5) for sq in sub_questions]
    search_results_per_sq = await asyncio.gather(*search_tasks)

    # Dedupe URLs across all sub-questions
    all_urls: dict[str, str] = {}  # url -> title
    for results in search_results_per_sq:
        for r in results:
            all_urls.setdefault(r.url, r.title)

    logger.info(f"Searcher: {len(all_urls)} unique URLs to fetch")

    # 2. Parallel fetch
    articles = await fetch.fetch_many(list(all_urls.keys()))
    logger.info(f"Searcher: {len(articles)} articles successfully fetched")

    # 3. Chunk all articles
    all_chunks = []
    sources = []
    for i, article in enumerate(articles, 1):
        source_id = f"src_{i:02d}"
        sources.append(
            Source(
                source_id=source_id,
                url=article.url,
                title=article.title,
                author=article.author,
                date=article.date,
            )
        )
        chunks = chunk_text(
            article.text,
            url=article.url,
            title=article.title,
        )
        # prepend source_id to chunk_id so synthesizer can map back
        for c in chunks:
            c.chunk_id = f"{source_id}__{c.chunk_id}"
        all_chunks.extend(chunks)

    logger.info(f"Searcher: {len(all_chunks)} chunks total")

    # 4. Rerank, keep top-k
    reranker = Reranker()
    top_chunks = reranker.rerank(
        query,
        all_chunks,
        top_k=settings.MAX_SOURCES_PER_QUERY,
    )
    logger.info(f"Searcher: reranked to top-{len(top_chunks)} chunks")

    return {
        "chunks": top_chunks,
        "sources": sources,
    }
