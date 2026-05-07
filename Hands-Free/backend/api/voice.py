"""
Voice pipeline API endpoints.

POST /api/voice/pipeline        — full STT→LLM→TTS round-trip; returns metadata + audio
POST /api/voice/pipeline/stream — same but streams audio chunks as SSE/chunked
POST /api/voice/transcribe      — STT only (debug/eval)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.services import stt as stt_svc
from backend.services.pipeline import PipelineResult, run_realtime

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice")


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    """STT only. Useful for debugging and eval pipelines."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="empty audio")
    result = await stt_svc.transcribe(audio_bytes)
    return asdict(result)


@router.post("/pipeline/stream")
async def pipeline_stream(audio: UploadFile = File(...)) -> StreamingResponse:
    """Run full pipeline and stream audio chunks back to caller."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="empty audio")

    rid = str(uuid.uuid4())
    out = PipelineResult()

    async def gen():
        async for chunk in run_realtime(audio_bytes, request_id=rid, out=out):
            yield chunk

    return StreamingResponse(gen(), media_type="audio/wav")
