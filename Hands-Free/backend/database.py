"""
Async SQLite store for latency events.

Schema is created on first init_db() call. Writes are fire-and-forget
so the pipeline never blocks on metrics.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)

_DB_PATH = os.environ.get("HF_METRICS_DB", "data/metrics.db")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS latency_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,
    stage        TEXT NOT NULL,            -- stt, llm_ttft, llm_total, tts_ttfa, tts_total, pipeline_ttfa
    duration_ms  INTEGER NOT NULL,
    ttfa_ms      INTEGER,
    timestamp    TEXT NOT NULL,            -- ISO 8601 UTC
    language     TEXT,                     -- en | ar
    model_version TEXT,
    pipeline_phase TEXT,                   -- raw | prompted | lora_v1 | ...
    success      INTEGER NOT NULL DEFAULT 1,
    extra        TEXT                      -- JSON blob for misc fields
);

CREATE INDEX IF NOT EXISTS idx_request_id ON latency_events(request_id);
CREATE INDEX IF NOT EXISTS idx_stage ON latency_events(stage);
"""


async def init_db() -> None:
    """Create schema if missing. Idempotent."""
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    log.info("metrics db ready", extra={"path": _DB_PATH})


async def insert_event(
    *,
    request_id: str,
    stage: str,
    duration_ms: int,
    timestamp: str,
    ttfa_ms: int | None = None,
    language: str | None = None,
    model_version: str | None = None,
    pipeline_phase: str | None = None,
    success: bool = True,
    **extra: Any,
) -> None:
    """Fire-and-forget insert. Swallows DB errors so the pipeline keeps moving."""
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO latency_events
                (request_id, stage, duration_ms, ttfa_ms, timestamp, language,
                 model_version, pipeline_phase, success, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    request_id, stage, duration_ms, ttfa_ms, timestamp, language,
                    model_version, pipeline_phase, int(success),
                    None if not extra else __import__("json").dumps(extra),
                ),
            )
            await db.commit()
    except Exception:
        log.exception("metrics insert failed", extra={"request_id": request_id, "stage": stage})


def fire_and_forget(coro) -> None:
    """Schedule a coroutine without awaiting (use for non-blocking metric writes)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)
