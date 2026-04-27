"""
Latency tracking for every pipeline stage.

Usage:
    @track_latency("stt")
    async def transcribe(...): ...

    async with LatencySpan("llm_ttft", request_id=rid, language="en") as span:
        result = await first_token()
        span.set_meta(model="gemma4:e2b")
"""
import functools
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from backend.database import insert_event

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _pipeline_phase() -> str:
    return os.environ.get("HF_PIPELINE_PHASE", "raw")


def _model_version() -> str:
    return os.environ.get("HF_OLLAMA_MODEL", "gemma4:e2b")


def track_latency(stage: str) -> Callable:  # type: ignore[type-arg]
    """
    Decorator for async functions. Records duration_ms to SQLite + structured log.
    Injects request_id as first kwarg if the function accepts it.
    """

    def decorator(fn: Callable) -> Callable:  # type: ignore[type-arg]
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = kwargs.pop("request_id", str(uuid.uuid4()))
            language = kwargs.get("language")
            t0 = time.perf_counter()
            success = True
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                ts = _now_iso()
                log.info(
                    "latency",
                    extra={
                        "stage": stage,
                        "duration_ms": duration_ms,
                        "request_id": request_id,
                        "language": language,
                        "success": success,
                        "pipeline_phase": _pipeline_phase(),
                    },
                )
                await insert_event(
                    request_id=request_id,
                    stage=stage,
                    duration_ms=duration_ms,
                    timestamp=ts,
                    language=language,
                    model_version=_model_version(),
                    pipeline_phase=_pipeline_phase(),
                    success=success,
                )

        return wrapper

    return decorator


class LatencySpan:
    """
    Async context manager for sub-stage timing (e.g. TTFT, TTFA).

    async with LatencySpan("llm_ttft", request_id=rid, language="en") as span:
        chunk = await first_token()
        span.set_meta(tokens=1)
    """

    def __init__(
        self,
        stage: str,
        *,
        request_id: str | None = None,
        language: str | None = None,
        ttfa: bool = False,
    ) -> None:
        self.stage = stage
        self.request_id = request_id or str(uuid.uuid4())
        self.language = language
        self.ttfa = ttfa
        self._meta: dict[str, Any] = {}
        self._t0: float = 0.0
        self._success: bool = True

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def mark_failure(self) -> None:
        self._success = False

    @asynccontextmanager
    async def __call__(self) -> AsyncGenerator["LatencySpan", None]:
        self._t0 = time.perf_counter()
        try:
            yield self
        except Exception:
            self._success = False
            raise
        finally:
            duration_ms = int((time.perf_counter() - self._t0) * 1000)
            ttfa_ms = duration_ms if self.ttfa else None
            ts = _now_iso()
            meta_str = json.dumps(self._meta) if self._meta else None
            log.info(
                "latency",
                extra={
                    "stage": self.stage,
                    "duration_ms": duration_ms,
                    "ttfa_ms": ttfa_ms,
                    "request_id": self.request_id,
                    "language": self.language,
                    "success": self._success,
                    "pipeline_phase": _pipeline_phase(),
                    **self._meta,
                },
            )
            await insert_event(
                request_id=self.request_id,
                stage=self.stage,
                duration_ms=duration_ms,
                ttfa_ms=ttfa_ms,
                timestamp=ts,
                language=self.language,
                model_version=_model_version(),
                pipeline_phase=_pipeline_phase(),
                success=self._success,
                metadata_json=meta_str,
            )
