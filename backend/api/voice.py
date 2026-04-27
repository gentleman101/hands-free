"""
Voice pipeline API endpoints.

POST /api/voice/pipeline         — full STT→LLM→TTS, returns audio + metadata
POST /api/voice/pipeline/stream  — full pipeline, streams audio bytes (chunked)
POST /api/voice/transcribe       — STT only, returns transcription
"""
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services import pipeline as pipeline_svc
from backend.services import stt as stt_svc

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])


class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration_ms: int


class PipelineResponse(BaseModel):
    transcription: str
    language: str
    response_text: str
    audio_b64: str           # base64-encoded WAV audio
    latency: dict            # type: ignore[type-arg]


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    """STT only. Accepts WAV audio, returns transcription."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    result = await stt_svc.transcribe(audio_bytes)
    return TranscribeResponse(
        text=result.text,
        language=result.language,
        confidence=result.confidence,
        duration_ms=result.duration_ms,
    )


@router.post("/pipeline")
async def pipeline_run(audio: UploadFile = File(...)) -> PipelineResponse:
    """
    Full STT→LLM→TTS pipeline. Non-streaming.
    Returns JSON with transcription, response text, and base64 audio.
    """
    import base64

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    result = await pipeline_svc.run(audio_bytes)

    return PipelineResponse(
        transcription=result.transcription,
        language=result.language,
        response_text=result.response_text,
        audio_b64=base64.b64encode(result.audio_bytes).decode(),
        latency={
            "stt_ms": result.latency.stt_ms,
            "llm_ttft_ms": result.latency.llm_ttft_ms,
            "llm_total_ms": result.latency.llm_total_ms,
            "tts_ttfa_ms": result.latency.tts_ttfa_ms,
            "tts_total_ms": result.latency.tts_total_ms,
            "pipeline_ttfa_ms": result.latency.pipeline_ttfa_ms,
            "total_ms": result.latency.total_ms,
        },
    )


@router.post("/pipeline/stream")
async def pipeline_stream(audio: UploadFile = File(...)) -> StreamingResponse:
    """
    Full pipeline with streamed audio output.
    Returns audio/wav chunks as they arrive from Fish Speech.
    Client can begin playback on first chunk.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    async def _generate():  # type: ignore[return]
        async for chunk in pipeline_svc.run_realtime(audio_bytes):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="audio/wav",
        headers={"X-Content-Type-Options": "nosniff"},
    )
