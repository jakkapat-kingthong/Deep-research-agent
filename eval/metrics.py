"""Custom evaluation metrics for Deep Research Agent."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from loguru import logger

from deep_research.graph import get_graph
from deep_research.schemas import Claim, Source


@dataclass
class EvalCase:
    """One benchmark case."""

    id: str
    query: str
    expected_keywords: list[str]
    min_claims: int


@dataclass
class EvalResult:
    """Result for one case."""

    case_id: str
    query: str
    num_claims: int
    num_sources: int
    citation_accuracy: float
    keyword_coverage: float
    has_min_claims: bool
    latency_ms: int
    cost_usd: float
    error: str | None = None


def compute_citation_accuracy(claims: list[Claim], sources: list[Source]) -> float:
    """% of claims where all source_ids exist in sources."""
    if not claims:
        return 0.0
    source_ids = {s.source_id for s in sources}
    valid = sum(1 for c in claims if all(sid in source_ids for sid in c.source_ids))
    return valid / len(claims)


def compute_keyword_coverage(
    summary: str, claims: list[Claim], expected_keywords: list[str]
) -> float:
    """% of expected keywords present in summary+claims."""
    text = (summary + " " + " ".join(c.text for c in claims)).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords) if expected_keywords else 0.0


async def run_one_case(case: EvalCase, budget: float = 0.30) -> EvalResult:
    """Execute the agent on a single case, compute metrics."""
    graph = get_graph()
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"query": case.query, "budget_usd": budget},
                config={"configurable": {"thread_id": f"eval-{case.id}"}},
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        return EvalResult(
            case_id=case.id,
            query=case.query,
            num_claims=0,
            num_sources=0,
            citation_accuracy=0.0,
            keyword_coverage=0.0,
            has_min_claims=False,
            latency_ms=120_000,
            cost_usd=0.0,
            error="timeout",
        )
    except Exception as exc:
        logger.exception(f"Case {case.id} failed")
        return EvalResult(
            case_id=case.id,
            query=case.query,
            num_claims=0,
            num_sources=0,
            citation_accuracy=0.0,
            keyword_coverage=0.0,
            has_min_claims=False,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=0.0,
            error=str(exc),
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    claims = result.get("claims", [])
    sources = result.get("sources", [])
    summary = result.get("summary", "")

    return EvalResult(
        case_id=case.id,
        query=case.query,
        num_claims=len(claims),
        num_sources=len(sources),
        citation_accuracy=compute_citation_accuracy(claims, sources),
        keyword_coverage=compute_keyword_coverage(summary, claims, case.expected_keywords),
        has_min_claims=len(claims) >= case.min_claims,
        latency_ms=latency_ms,
        cost_usd=result.get("cost_usd", 0.0),
    )
