"""FastAPI server with SSE streaming and Observability for Deep Research Agent."""

from __future__ import annotations

import json
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from loguru import logger
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from deep_research.config import settings
from deep_research.graph import get_graph
from deep_research.logging_config import configure_logging
from deep_research.observability import setup_tracing

# 1. Setup Logging & Tracing
configure_logging()
setup_tracing()

app = FastAPI(
    title="Deep Research Agent",
    version="0.1.0",
    description=("Agentic research system with schema-enforced verifiable citations"),
)

# 2. Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=5, max_length=500)
    budget_usd: float = Field(default=settings.DEFAULT_BUDGET_USD, ge=0.01, le=5.00)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check API health status."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/v1/research")
async def research_stream(req: ResearchRequest) -> EventSourceResponse:
    """Stream research progress via SSE (Server-Sent Events)."""
    graph = get_graph()

    thread_id = str(uuid4())

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            async for event in graph.astream(
                {"query": req.query, "budget_usd": req.budget_usd},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    yield {
                        "event": "node_update",
                        "data": json.dumps(
                            {
                                "node": node_name,
                                "output": _jsonify(node_output),
                            },
                            default=str,
                        ),
                    }
            yield {"event": "done", "data": "{}"}
        except Exception as exc:
            logger.exception("Research stream encountered an error")
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


@app.post("/v1/research/sync")
async def research_sync(req: ResearchRequest) -> dict:
    """Non-streaming endpoint — waits for the full report and returns it."""
    graph = get_graph()
    try:
        result = await graph.ainvoke(
            {"query": req.query, "budget_usd": req.budget_usd},
            config={"configurable": {"thread_id": str(uuid4())}},
        )
        return _jsonify(result)
    except Exception as exc:
        logger.exception("Synchronous research task failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _jsonify(obj: object) -> dict:
    """Helper to convert Pydantic models and complex dicts to JSON-safe format."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {
            k: _jsonify_val(v)
            for k, v in obj.items()
            if k != "messages"  # Skip internal langgraph state
        }
    return {"value": str(obj)}


def _jsonify_val(v: object) -> object:
    """Recursive helper for JSON conversion."""
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    if isinstance(v, list):
        return [_jsonify_val(x) for x in v]
    return v
