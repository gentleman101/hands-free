"""Structured JSON logging via python-json-logger."""
from __future__ import annotations

import logging
import os
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging() -> None:
    """Configure root logger for structured JSON output to stderr."""
    level = os.environ.get("HF_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level"},
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
