"""
Hands-Free FastAPI application aggregator.
Registers routers, configures logging, initialises database on startup.
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.api import health, voice
from backend.database import init_db
from backend.logging_config import configure_logging

configure_logging(os.environ.get("HF_LOG_LEVEL", "INFO"))


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield
    # audio_capture stopped per-request in CLI mode


app = FastAPI(
    title="Hands-Free Voice Pipeline",
    description="Bilingual (Arabic + English) voice-to-voice AI — POC",
    version="0.1.0",
    lifespan=_lifespan,
)

app.include_router(health.router)
app.include_router(voice.router)
