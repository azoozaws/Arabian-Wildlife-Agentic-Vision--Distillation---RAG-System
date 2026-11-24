"""Consistent logging configuration across all modules and pipelines."""

from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger without duplicating root handlers."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
