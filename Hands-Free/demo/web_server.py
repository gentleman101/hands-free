"""
Rammas web demo — phone / browser frontend over LAN via WebRTC.

Replaces LocalAudioTransport with SmallWebRTCTransport so any device on
the LAN can use the assistant through a browser. The Pipecat pipeline is
otherwise identical to demo.py.

Browser echo-cancellation is handled by the WebRTC stack (built-in AEC),
so RNNoiseFilter is not applied here — the mic track is already clean.

Setup (one-time):
    venv-pipecat\\Scripts\\pip install fastapi "uvicorn[standard]" "pipecat-ai[webrtc]" python-multipart

    # Generate self-signed TLS cert so the phone can access the mic:
    venv-pipecat\\Scripts\\python demo\\gen_cert.py

Run:
    set PYTHONIOENCODING=utf-8
    set PYTHONUTF8=1
    set HF_HUB_DISABLE_SYMLINKS=1
    venv-pipecat\\Scripts\\python demo\\web_server.py

Then open  https://<windows-lan-ip>:8443  on your phone.
Accept the self-signed cert warning once.
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\Users\dewac\Documents\Naman\Hands-Free")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequestHandler,
    IceCandidate,
    ConnectionMode,
)

from bot_speaking_gate import BotSpeakingGate
from confident_stt import RammasSTTService
from language_router import LanguageRouter
from silma_tts_service import SilmaTTSService, _VOICES_DIR, _REF_FILES, _REF_TEXTS
from mac_llm_service import MacLLMService
from stage_logger import StageLogger

# ─── Config ─────────────────────────────────────────────────────────────────
_MAC_LLM_URL    = os.environ.get("HF_LLM_SERVICE_URL", "http://192.168.137.237:8080")
_HOST           = os.environ.get("HF_WEB_HOST", "0.0.0.0")
_PORT           = int(os.environ.get("HF_WEB_PORT", "8443"))
_CERT_FILE      = Path(__file__).parent / "certs" / "cert.pem"
_KEY_FILE       = Path(__file__).parent / "certs" / "key.pem"
_STATIC_DIR     = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)

log = logging.getLogger("web_server")

# Pre-load SILMA once at startup (Vocos cuDNN requirement)
log.info("preloading SILMA...")
from silma_tts.api import SilmaTTS
_silma_instance = SilmaTTS()
log.info("SILMA ready")

# Warm reference audio cache for both languages so every real turn skips
# "Converting ref audio..." and goes straight to synthesis.
_noop_fn = lambda *a, **k: None  # noqa: E731
for _lang in ("en", "ar"):
    log.info("silma.warmup lang=%s", _lang)
    _silma_instance.infer(
        ref_file=str(_VOICES_DIR / _REF_FILES[_lang]),
        ref_text=_REF_TEXTS[_lang],
        gen_text="Hello." if _lang == "en" else "مرحبا.",
        file_wave=None,
        speed=1.0,
        force_tashkeel=(_lang == "ar"),
        normalize_numbers=(_lang == "ar"),
        nfe_step=16,
        show_info=_noop_fn,
        progress=_noop_fn,
    )
    log.info("silma.warmup done lang=%s", _lang)
log.info("SILMA cache warm for EN + AR")

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Rammas — DEWA Voice Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

webrtc_handler = SmallWebRTCRequestHandler(
    connection_mode=ConnectionMode.MULTIPLE,  # allow multiple phones simultaneously
)

# Serve the static frontend
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/offer")
async def handle_offer(body: dict):
    """WebRTC signaling — phone sends SDP offer, we return SDP answer."""
    request = SmallWebRTCRequest.from_dict(body)
    answer = await webrtc_handler.handle_web_request(request, _on_new_connection)
    return JSONResponse(content=answer)


@app.patch("/api/offer")
async def handle_patch(body: dict):
    """ICE candidate trickle."""
    candidates = [
        IceCandidate(
            candidate=c["candidate"],
            sdp_mid=c["sdpMid"],
            sdp_mline_index=c["sdpMLineIndex"],
        )
        for c in body.get("candidates", [])
    ]
    patch = SmallWebRTCPatchRequest(pc_id=body["pc_id"], candidates=candidates)
    await webrtc_handler.handle_patch_request(patch)
    return JSONResponse(content={"status": "ok"})


async def _on_new_connection(webrtc_connection) -> None:
    """Called once per new WebRTC peer connection. Builds and runs a pipeline."""
    log.info(f"new WebRTC connection: {webrtc_connection.pc_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    # VAD
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.95,   # raised from 0.92 — echo was scoring above 0.92 on iPhone AEC
                start_secs=0.2,
                stop_secs=0.4,
                min_volume=0.30,
            )
        ),
    )

    # STT — EN/AR auto-detect, beam_size=1, no double-VAD
    stt = RammasSTTService(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        compute_type="int8_float16",
    )

    # LLM — Mac Mini Gemma 4 E4B
    llm = MacLLMService(
        base_url=_MAC_LLM_URL,
        max_tokens=200,
        language="en",
    )

    # TTS — SILMA with comma-aware sentence splitting
    tts = SilmaTTSService(language=Language.EN, speed=1.0)
    tts._silma = _silma_instance  # inject pre-loaded instance

    # Context
    context = LLMContext(messages=[{"role": "system", "content": ""}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
            )
        ),
    )

    # Language router
    language_router = LanguageRouter(context, llm, tts, filler_enabled=False)

    # Pipeline — same as demo.py, transport swapped
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        StageLogger(),   # captures: VAD events, interruptions, bot speaking
        stt,
        StageLogger(),   # captures: STT transcriptions, LLM start/done, TTS frames
        language_router,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    # Run in background so the HTTP handler returns the SDP answer immediately
    asyncio.create_task(_run_pipeline(task, webrtc_connection.pc_id))


async def _run_pipeline(task: PipelineTask, pc_id: str) -> None:
    try:
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
    except Exception:
        log.exception(f"pipeline error for {pc_id}")
    finally:
        log.info(f"pipeline ended for {pc_id}")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket

    # Print LAN IP so user knows where to point their phone
    lan_ip = socket.gethostbyname(socket.gethostname())

    print()
    print("=" * 60)
    print("Rammas — DEWA AI Voice Assistant (WebRTC / Phone)")
    print(f"LLM: {_MAC_LLM_URL}")
    print()

    ssl_ctx = None
    if _CERT_FILE.exists() and _KEY_FILE.exists():
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(_CERT_FILE), str(_KEY_FILE))
        print(f"[HTTPS] self-signed cert loaded")
        print(f"[PHONE] open:  https://{lan_ip}:{_PORT}")
        print(f"        accept the cert warning once on your device")
    else:
        print(f"[WARN] no TLS cert — mic will be blocked on phone browsers")
        print(f"[PHONE] open:  http://{lan_ip}:{_PORT}")
        print(f"        run  venv-pipecat\\Scripts\\python demo\\gen_cert.py  first")

    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host=_HOST,
        port=_PORT,
        ssl_certfile=str(_CERT_FILE) if ssl_ctx else None,
        ssl_keyfile=str(_KEY_FILE) if ssl_ctx else None,
        log_level="warning",
    )
