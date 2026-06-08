"""Smoke test: build the pipeline + verify everything wires together.

Does NOT start the audio loop or contact the Mac — just imports, instantiates,
and links the pipeline. Catches all the obvious wiring bugs in < 30s.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step(label):
    print(f"[smoke] {label} ... ", end="", flush=True)


def ok():
    print("OK")


async def main():
    logging.basicConfig(level=logging.WARNING)

    step("import pipecat core")
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    ok()

    step("import our services")
    from silma_tts_service import SilmaTTSService
    from stub_llm_service import StubLLMService
    from mac_llm_service import MacLLMService
    ok()

    step("instantiate LocalAudioTransport (mic + speakers + Silero VAD)")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.7, stop_secs=0.6)),
        )
    )
    ok()

    step("instantiate WhisperSTTService (large-v3-turbo CUDA, lazy load)")
    stt = WhisperSTTService(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        compute_type="float16",
        language=None,
    )
    ok()

    step("instantiate StubLLMService")
    stub_llm = StubLLMService()
    ok()

    step("instantiate MacLLMService (no network call)")
    mac_llm = MacLLMService(base_url="http://192.168.137.237:8080")
    ok()

    step("instantiate SilmaTTSService (model load deferred to first run_tts)")
    tts = SilmaTTSService(language=Language.EN, speed=1.0)
    ok()

    step("build LLMContext + LLMContextAggregatorPair")
    context = LLMContext(messages=[{"role": "system", "content": "You are a test bot."}])
    context_aggregator = LLMContextAggregatorPair(context)
    ok()

    step("link Pipeline([input, stt, ctx.user, llm, tts, output, ctx.assistant])")
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        stub_llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    ok()

    step("build PipelineTask with allow_interruptions + metrics")
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )
    ok()

    step("verify reference voice files exist on disk")
    repo = Path(r"C:\Users\dewac\Documents\Naman\Hands-Free")
    for fname in ("f5_basic_ref_en.wav", "arabic_f5_ref.wav"):
        p = repo / "assets" / "voices" / fname
        assert p.exists(), f"missing: {p}"
    ok()

    print()
    print("[smoke] ALL CHECKS PASSED")
    print("[smoke] To run live demo:")
    print("[smoke]   set PYTHONIOENCODING=utf-8")
    print("[smoke]   set PYTHONUTF8=1")
    print("[smoke]   venv-pipecat\\Scripts\\python.exe demo\\demo.py")


if __name__ == "__main__":
    asyncio.run(main())
