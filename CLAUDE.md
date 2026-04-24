# Deep Research Agent

## Architecture

Agentic research system: **planner → search → synthesizer → critic** loop.

```
User query
    │
    ▼
Planner          Decomposes the query into focused sub-questions
    │
    ▼
Search           Retrieves sources for each sub-question (Tavily + Google)
    │
    ▼
Synthesizer      Merges retrieved evidence into a draft answer with citations
    │
    ▼
Critic           Evaluates quality, flags gaps; loops back to Search if needed
    │
    ▼
Final answer     Structured response with inline citations and a source list
```

## Tech Stack

| Layer       | Library                          |
|-------------|----------------------------------|
| Orchestration | LangGraph                      |
| API serving | FastAPI                          |
| Config      | pydantic-settings v2             |
| Validation  | pydantic v2                      |
| Logging     | loguru                           |
| LLM         | Anthropic SDK (Claude 3.5 Sonnet)|
| Search      | Tavily API, Google Search API    |

## Invariants

- **Strict citations** — every factual claim in the final answer must reference a source index that appears in the returned source list. No unsourced assertions.
- **Budget enforcement** — cumulative LLM spend is tracked via token counts and the run is halted if it would exceed `DEFAULT_BUDGET_USD`. Budget is checked before each LLM call.
- **Deterministic sub-question count** — the planner emits at most `MAX_SUBQUESTIONS` sub-questions; the critic cannot expand this budget.
- **Source cap per query** — each search call returns at most `MAX_SOURCES_PER_QUERY` results; deduplication happens before synthesis.

## Key Files

```
src/deep_research/
    config.py          # All settings via pydantic-settings; single source of truth
    logging_config.py  # loguru setup; call setup_logging() once at startup
    __main__.py        # Entrypoint — python -m deep_research
```

## Development

```bash
cp .env.example .env          # fill in API keys
uv sync                        # install deps
pre-commit install             # wire ruff hooks
python -m deep_research        # smoke test startup
```
