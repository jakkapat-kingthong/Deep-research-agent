"""Pytest-runnable eval harness — fails if avg metrics drop below threshold."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from eval.metrics import EvalCase, run_one_case


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    path = Path(__file__).parents[2] / "eval" / "dataset.jsonl"
    loaded = []
    with path.open() as f:
        for line in f:
            loaded.append(EvalCase(**json.loads(line)))
    return loaded[:5]  # smoke: first 5 cases only for pytest run


def test_benchmark_smoke(cases: list[EvalCase]) -> None:
    """Quick smoke test — 5 cases, assert aggregate thresholds."""

    async def _run() -> list:
        return [await run_one_case(c) for c in cases]

    results = asyncio.run(_run())
    succ = [r for r in results if r.error is None]

    assert len(succ) >= 3, f"Too many failures: {len(results) - len(succ)}"

    avg_citation = sum(r.citation_accuracy for r in succ) / len(succ)
    assert avg_citation >= 0.70, f"Citation accuracy {avg_citation:.2f} < 0.70"
