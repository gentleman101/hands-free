"""Structured JSON logging via python-json-logger."""
from __future__ import annotations

import logging
import os
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging() -> None:
    """Configure root logger.

    By default logs JSON to stderr. If HF_LOG_FILE is set (e.g. by scripts/start.py
    for CLI-UI mode), logs are routed to that file instead so the terminal stays
    clean for the live status display.
    """
    level = os.environ.get("HF_LOG_LEVEL", "INFO").upper()
    log_file = os.environ.get("HF_LOG_FILE", "").strip()

    if log_file:
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={"asctime": "ts", "levelname": "level"},
            )
        )
    else:
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
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
