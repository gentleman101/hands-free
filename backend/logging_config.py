"""Structured JSON logging for Hands-Free pipeline."""
import logging
import sys
from datetime import UTC, datetime

from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]


class _UTCJsonFormatter(JsonFormatter):
    def add_fields(
        self,
        log_record: dict,  # type: ignore[type-arg]
        record: logging.LogRecord,
        message_dict: dict,  # type: ignore[type-arg]
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["ts"] = datetime.now(UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record.pop("levelname", None)
        log_record.pop("name", None)


def configure_logging(level: str = "INFO") -> None:
    """Call once at app startup. Sets structured JSON on root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _UTCJsonFormatter("%(ts)s %(level)s %(logger)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
