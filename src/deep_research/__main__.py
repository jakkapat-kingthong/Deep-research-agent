from loguru import logger

from deep_research.config import settings
from deep_research.logging_config import setup_logging


def main() -> None:
    setup_logging(settings.LOG_LEVEL)
    logger.info("Deep Research Agent starting up")
    logger.info("Planner model: {}", settings.PLANNER_MODEL)
    logger.info("Max sub-questions: {}", settings.MAX_SUBQUESTIONS)
    logger.info("Budget cap: ${:.2f}", settings.DEFAULT_BUDGET_USD)


if __name__ == "__main__":
    main()
