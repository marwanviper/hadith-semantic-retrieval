"""
Centralized logging configuration.

All modules call `get_logger(__name__)` to obtain a named logger.
This ensures:
  - Consistent format across the codebase.
  - Log level controlled from one place (config or env var).
  - No raw print() calls in library code.

Log format example:
    2026-08-29 19:32:11 | INFO     | src.indexes.hnsw | Index built in 2.41s

Usage:
    from src.logging.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting search")
    logger.debug("Query vector shape: %s", query.shape)
"""

import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "{asctime} | {levelname:<8} | {name} | {message}"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(
    level: str = "INFO",
    fmt: str = _LOG_FORMAT,
    datefmt: str = _DATE_FORMAT,
    stream=None,
) -> None:
    """
    Configure the root logger.  Call once at application startup.

    Args:
        level:   Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt:     Log format string (str.format style with style='{').
        datefmt: Date format for timestamps.
        stream:  Output stream (defaults to sys.stderr).
    """
    global _configured
    if stream is None:
        stream = sys.stderr

    # Honour environment variable override
    level = os.environ.get("HADITH_LOG_LEVEL", level).upper()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt=datefmt, style="{")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    If logging has not been configured yet, applies a sensible default
    so that library users get useful output even without explicit setup.

    Args:
        name:  Module name (pass __name__ from the calling module).
        level: Optional override for this specific logger's level.

    Returns:
        logging.Logger instance.
    """
    if not _configured:
        configure_logging()

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level.upper())
    return logger
