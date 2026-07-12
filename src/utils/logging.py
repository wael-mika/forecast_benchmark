"""Small logging helpers for readable console output across scripts and modules.

This module keeps logger setup intentionally simple: one helper that returns a
plain console logger with a consistent format and without duplicate handlers.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured console logger with a readable one-line format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
