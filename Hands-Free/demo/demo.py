"""
Pipecat voice-to-voice demo for Hands-Free — Rammas (DEWA AI Assistant).

Full production pipeline on Windows (RTX A3000):
    mic ─► SileroVAD ─► RammasSTT ─► LanguageRouter ─► MacLLM ─► SILMA ─► speakers

All manual pipeline features now wired in:
- STT confidence gates (no_speech, logprob, compression, hallucination denylist, RMS)
- Language detection (EN/AR auto) + script validation
- Language-specific system prompts (EXP-017: saves ~80 tokens/call)
- Dynamic language switching for LLM + TTS per utterance
- Filler audio (context-aware, plays during LLM wait) — set HF_FILLER_ENABLED=1
- VAD threshold 0.80 (EXP-011: reduces HVAC/breath false fires)
- Barge-in / interruptions (Pipecat built-in)
- Smart Turn v3 semantic endpointing (Pipecat built-in)
- Conversation memory (Pipecat LLMContext built-in)

Run from worktree root:
    set PYTHONIOENCODING=utf-8
    set PYTHONUTF8=1
    set HF_HUB_DISABLE_SYMLINKS=1
    venv-pipecat\\Scripts\\python.exe demo\\demo.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# nemo_text_processing in this venv imports backend.services.text_normalizer
# from the main Hands-Free repo — add it so the import resolves.
sys.path.insert(0, r"C:\Users\dewac\Documents\Naman\Hands-Free")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from confident_stt import RammasSTTService
from language_router import LanguageRouter
from silma_tts_service import SilmaTTSService
from mac_llm_service import MacLLMService
from stage_logger import StageLogger


# ─── Configuration ──────────────────────────────────────────────────────────
_MAC_LLM_URL   = os.environ.get("HF_LLM_SERVICE_URL", "http://192.168.137.237:8080")
_FILLER_ENABLED = os.environ.get("HF_FILLER_ENABLED", "0") != "0"


async def main() -> None:
    # DEBUG MODE: full logging to diagnose silent exit. Tighten once working.
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    log = logging.getLogger("demo")

    # Pre-load SILMA on main thread before Pipecat starts (Vocos cuDNN requirement)
    log.info("preloading SILMA...")
    from silma_tts.api import SilmaTTS
    from silma_tts_service import _VOICES_DIR, _REF_FILES, _REF_TEXTS
    silma_instance = SilmaTTS()
    log.info("SILMA ready")

    # Warm reference audio cache for both languages so every real turn skips
    # "Converting ref audio..." and goes straight to synthesis.
    # One dummy infer() per language is enough — SILMA caches the preprocessed
    # mel features after the first call; subsequent calls with the same ref file
    # show "Using cached preprocessed reference audio..." instead.
    _noop = lambda *a, **k: None  # noqa: E731
    for _lang in ("en", "ar"):
        log.info("silma.warmup lang=%s", _lang)
        silma_instance.infer(
            ref_file=str(_VOICES_DIR / _REF_FILES[_lang]),
            ref_text=_REF_TEXTS[_lang],
            gen_text="Hello." if _lang == "en" else "مرحبا.",
            file_wave=None,
            speed=1.0,
            force_tashkeel=(_lang == "ar"),
            normalize_numbers=(_lang == "ar"),
            nfe_step=16,
            show_info=_noop,
            progress=_noop,
        )
        log.info("silma.warmup done lang=%s", _lang)
    log.info("SILMA cache warm for EN + AR")

    # Transport: mic + speakers. NOTE: vad_analyzer in TransportParams is IGNORED
    # by LocalAudioTransport (only Daily.co uses it). For local VAD we add a
    # VADProcessor as its own stage in the pipeline below.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            audio_in_filter=RNNoiseFilter(),   # office noise suppression
        )
    )

    # Standalone VAD processor — Silero with tuned params
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.70,   # laptop: RNNoise handles noise, no echo issue
                start_secs=0.2,
                stop_secs=0.4,
                min_volume=0.30,
            )
        ),
    )

    # STT — full confidence pipeline (replaces bare WhisperSTTService)
    stt = RammasSTTService(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        compute_type="int8_float16",
        # language=None is default in RammasSTTService (auto-detect)
    )

    # LLM — Mac Mini Gemma 4 E4B
    llm = MacLLMService(
        base_url=_MAC_LLM_URL,
        max_tokens=200,
        language="en",  # LanguageRouter will update this per utterance
    )

    # TTS — SILMA EN+AR, inject pre-loaded instance
    tts = SilmaTTSService(language=Language.EN, speed=1.0)
    tts._silma = silma_instance

    # Conversation context — system prompt starts as EN, LanguageRouter swaps per turn
    context = LLMContext(messages=[{"role": "system", "content": ""}])

    # Replace Smart Turn v3 (default, adds ~150ms ONNX) with a simple speech-timeout
    # strategy: 300ms silence after VAD-stop = turn end. Sufficient for declarative
    # DEWA queries; re-enable Smart Turn if false-positives appear.
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
            )
        ),
    )

    # LanguageRouter: per-utterance language detection + prompt swap + optional filler
    language_router = LanguageRouter(
        context,
        llm,
        tts,
        filler_enabled=_FILLER_ENABLED,
    )

    # Pipeline order:
    #   transport.input()    → mic PCM + VAD events
    #   stt                  → TranscriptionFrame (with language + confidence gates)
    #   language_router      → swaps prompt, updates LLM/TTS language, optional filler
    #   ctx.user()           → appends user msg → LLMContextFrame
    #   llm                  → streams LLMTextFrame tokens
    #   tts                  → sentence-level synthesis → TTSAudioRawFrame
    #   transport.output()   → speakers
    #   ctx.assistant()      → appends assistant msg to history
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

    print()
    print("=" * 60)
    print("Rammas — DEWA AI Voice Assistant (Pipecat)")
    print(f"LLM: {_MAC_LLM_URL}")
    print("Speak in English or Arabic. Ctrl+C to stop.")
    print("=" * 60)
    print()
    print("🎤  listening for speech...", flush=True)

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
