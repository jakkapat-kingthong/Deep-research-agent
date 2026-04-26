"""Logging configuration for the project."""

import sys
from loguru import logger
from deep_research.config import settings


def configure_logging() -> None:
    """Configure loguru with settings from .env."""
    # ลบการตั้งค่าพื้นฐานทิ้งก่อน
    logger.remove()
    # ตั้งค่าใหม่ให้แสดงผลผ่านหน้าจอ Terminal ตามระดับ LOG_LEVEL ที่เราตั้งไว้
    logger.add(sys.stderr, level=settings.LOG_LEVEL)
