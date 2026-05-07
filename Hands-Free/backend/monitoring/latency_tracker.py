"""
@track_latency decorator + LatencySpan context manager.

Wraps any async fn; emits a row to the metrics DB on completion.
Stage names match those in CONVENTIONS.md (stt, llm_total, tts_total, pipeline_ttfa).
"""
from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, TypeVar

from backend.database import fire_and_forget, insert_event

log = logging.getLogger(__name__)

T = TypeVar("T")


def track_latency(stage: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator: emits a latency_events row after the async fn returns.

    Usage:
        @track_latency("stt")
        async def transcribe(...): ...
    """
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            request_id = kwargs.get("request_id") or str(uuid.uuid4())
            t0 = time.perf_counter()
            success = True
            try:
                return await fn(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                fire_and_forget(insert_event(
                    request_id=request_id,
                    stage=stage,
                    duration_ms=duration_ms,
                    timestamp=datetime.now(UTC).isoformat(),
                    pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                    success=success,
                ))
        return wrapper
    return decorator


class LatencySpan:
    """
    Context manager for tracking a span of code (e.g. inside a stream loop).

    Usage:
        async with LatencySpan("llm_ttft", request_id=rid) as span:
            ...
            span.duration_ms      # available after exit
    """
    def __init__(self, stage: str, *, request_id: str, language: str | None = None) -> None:
        self.stage = stage
        self.request_id = request_id
        self.language = language
        self.t0: float = 0.0
        self.duration_ms: int = 0

    async def __aenter__(self) -> "LatencySpan":
        self.t0 = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = int((time.perf_counter() - self.t0) * 1000)
        fire_and_forget(insert_event(
            request_id=self.request_id,
            stage=self.stage,
            duration_ms=self.duration_ms,
            timestamp=datetime.now(UTC).isoformat(),
            language=self.language,
            pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
            success=(exc is None),
        ))
