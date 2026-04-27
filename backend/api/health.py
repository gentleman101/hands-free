"""Health endpoints: /healthz (liveness) and /readyz (readiness)."""
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    ollama: bool
    whisper: bool


@router.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Always returns 200 if the process is running."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    """
    Returns 200 only when:
    - Ollama is reachable and gemma4:e2b is available
    - Whisper model is loaded (lazy-loads on first call if not yet done)
    """
    ollama_ok = await _check_ollama()
    whisper_ok = _check_whisper()

    if not ollama_ok or not whisper_ok:
        raise HTTPException(
            status_code=503,
            detail=ReadinessResponse(
                status="not ready", ollama=ollama_ok, whisper=whisper_ok
            ).model_dump(),
        )
    return ReadinessResponse(status="ready", ollama=ollama_ok, whisper=whisper_ok)


async def _check_ollama() -> bool:
    base = os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(base.replace("/v1", "/api/tags"))
            return resp.status_code == 200
    except Exception:
        return False


def _check_whisper() -> bool:
    from backend.services.stt import _load_model
    import os

    try:
        _load_model(os.environ.get("HF_WHISPER_MODEL_SIZE", "small"))
        return True
    except Exception:
        return False
