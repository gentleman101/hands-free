"""Health endpoints: /healthz (liveness) and /readyz (full readiness check)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend import config as cfg
from backend.services import llm_client, tts as tts_svc

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness — process is up. No external dep checks."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict:
    """Readiness — Mac LLM reachable + Fish Speech reachable (Windows client)."""
    config = cfg.load()
    checks: dict[str, bool] = {}

    if config.is_client:
        checks["llm_remote"] = await llm_client.health_check()
        checks["tts_local"] = await tts_svc.health_check()
    else:
        # Server mode: simply confirm config is set
        checks["llm_model_path_set"] = bool(config.llm_model_path)

    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})

    return {"status": "ready", "checks": checks, "role": config.machine_role}
