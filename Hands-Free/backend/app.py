"""
FastAPI aggregator for the Windows (client) side.

Loads config, sets up logging + DB, registers routers. Service modules
are loaded lazily on first request so `import backend.app` is fast.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend import config as cfg
from backend.api import health, voice
from backend.database import init_db
from backend.logging_config import setup_logging
from backend.services import llm_client, tts as tts_svc

setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup: init metrics DB. Shutdown: close httpx clients."""
    config = cfg.load()
    log.info("starting hands-free", extra={"role": config.machine_role})
    await init_db()
    yield
    log.info("shutting down")
    await llm_client.aclose()
    await tts_svc.aclose()


app = FastAPI(title="Hands-Free", lifespan=lifespan)
app.include_router(health.router)
app.include_router(voice.router)
