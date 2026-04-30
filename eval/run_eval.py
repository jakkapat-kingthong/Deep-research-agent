"""Run the full benchmark and output a metrics report."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from deep_research.logging_config import configure_logging
from eval.metrics import EvalCase, EvalResult, run_one_case


async def main() -> None:
    configure_logging()
    console = Console()

    # Load dataset
    dataset_path = Path(__file__).parent / "dataset.jsonl"
    cases: list[EvalCase] = []
    with dataset_path.open() as f:
        for line in f:
            data = json.loads(line)
            cases.append(EvalCase(**data))

    logger.info(f"Loaded {len(cases)} eval cases")

    # Run sequentially (parallel can exceed rate limits)
    results: list[EvalResult] = []
    for case in cases:
        logger.info(f"Running case {case.id}: {case.query[:60]}...")
        result = await run_one_case(case)
        results.append(result)

    # Write JSON report
    report_path = Path(__file__).parent / "latest_report.json"
    with report_path.open("w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, default=str)

    # Print summary table
    table = Table(title="Deep Research Agent — Eval Results")
    table.add_column("ID")
    table.add_column("Claims")
    table.add_column("Citation")
    table.add_column("Keyword")
    table.add_column("Latency")
    table.add_column("Cost")
    table.add_column("Error", style="red")

    for r in results:
        table.add_row(
            r.case_id,
            f"{r.num_claims}",
            f"{r.citation_accuracy:.2f}",
            f"{r.keyword_coverage:.2f}",
            f"{r.latency_ms}ms",
            f"${r.cost_usd:.4f}",
            r.error or "",
        )
    console.print(table)

    # Aggregates
    succ = [r for r in results if r.error is None]
    if succ:
        avg_cit = sum(r.citation_accuracy for r in succ) / len(succ)
        avg_kw = sum(r.keyword_coverage for r in succ) / len(succ)
        avg_lat = sum(r.latency_ms for r in succ) / len(succ)
        avg_cost = sum(r.cost_usd for r in succ) / len(succ)
        console.rule("[bold green]Aggregate")
        console.print(f"Cases passed       : {len(succ)}/{len(results)}")
        console.print(f"Citation accuracy  : {avg_cit:.3f}")
        console.print(f"Keyword coverage   : {avg_kw:.3f}")
        console.print(f"Mean latency       : {avg_lat:.0f} ms")
        console.print(f"Mean cost per query: ${avg_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
