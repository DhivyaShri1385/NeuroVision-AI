"""
Structured logging setup using loguru.

Provides a single pre-configured logger instance imported across the project.
Automatically creates the log directory and rotates files to avoid disk bloat.
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger


def setup_logger(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "30 days",
    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    ),
) -> None:
    """Configure loguru with console + rotating file sinks.

    Call once at application startup (run_phase1.py or FastAPI lifespan).
    All subsequent `from src.utils.logger import logger` calls share the same
    configured instance.
    """
    _logger.remove()  # Remove default stderr sink

    # --- Console sink (colorized) ---
    _logger.add(
        sys.stdout,
        level=level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # --- File sink (rotating, no color codes) ---
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_path / "neurovision_{time:YYYY-MM-DD}.log",
            level=level,
            format=log_format,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            colorize=False,
        )


# Module-level logger — import this everywhere:
#   from src.utils.logger import logger
logger = _logger
