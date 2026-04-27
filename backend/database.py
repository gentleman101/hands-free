"""SQLite connection and schema for Hands-Free latency metrics."""
import logging
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "metrics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS latency_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    duration_ms     INTEGER NOT NULL,
    ttfa_ms         INTEGER,
    timestamp       TEXT    NOT NULL,
    language        TEXT,
    model_version   TEXT,
    pipeline_phase  TEXT,
    success         INTEGER NOT NULL DEFAULT 1,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_stage_ts
    ON latency_events(stage, timestamp);

CREATE INDEX IF NOT EXISTS idx_pipeline_phase
    ON latency_events(pipeline_phase);
"""


async def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    log.info("database initialised", extra={"path": str(_DB_PATH)})


async def insert_event(
    *,
    request_id: str,
    stage: str,
    duration_ms: int,
    ttfa_ms: int | None = None,
    timestamp: str,
    language: str | None = None,
    model_version: str | None = None,
    pipeline_phase: str | None = None,
    success: bool = True,
    metadata_json: str | None = None,
) -> None:
    """Append one latency event row. Fire-and-forget safe (exceptions logged)."""
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            await db.execute(
                """INSERT INTO latency_events
                   (request_id, stage, duration_ms, ttfa_ms, timestamp,
                    language, model_version, pipeline_phase, success, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    stage,
                    duration_ms,
                    ttfa_ms,
                    timestamp,
                    language,
                    model_version,
                    pipeline_phase,
                    int(success),
                    metadata_json,
                ),
            )
            await db.commit()
    except Exception:
        log.exception("failed to write latency event", extra={"stage": stage})
