"""Centralized logging. Use loguru — colored stderr + rotating file sink."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


_CONFIGURED = False


def setup_logging(level: str = "INFO", sink: str | None = None, json_logs: bool = False) -> None:
    """Idempotent. Call once at the top of every entrypoint."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, serialize=json_logs)
    if sink:
        Path(sink).parent.mkdir(parents=True, exist_ok=True)
        logger.add(sink, level=level, rotation="100 MB", retention="14 days", serialize=json_logs)
    _CONFIGURED = True


__all__ = ["logger", "setup_logging"]
