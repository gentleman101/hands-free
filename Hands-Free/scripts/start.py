"""
Windows entry point: starts API server + mic loop.

Usage:
  set HF_MACHINE_ROLE=client
  set HF_LLM_SERVICE_URL=http://mac-mini.local:11434
  python scripts/start.py

This:
  1. Verifies role is "client"
  2. Health-checks Mac LLM server (fail-fast on office network)
  3. Starts FastAPI in a background task
  4. Runs the mic loop: capture → pipeline → speaker
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf    # type: ignore[import-untyped]
import io


async def _play_wav(wav_bytes: bytes) -> None:
    """Play WAV bytes through default output device."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    try:
        await asyncio.get_event_loop().run_in_executor(None, sd.wait)
    except asyncio.CancelledError:
        sd.stop()
        raise


async def _mic_loop(log: logging.Logger) -> None:
    from backend.services.audio_capture import AudioCapture
    from backend.services.filler import FillerPlayer
    from backend.services.pipeline import PipelineResult, run_realtime
    from backend import config as cfg

    config = cfg.load()
    capture = AudioCapture(
        vad_backend=config.vad_backend,
        device_env=config.audio_input_device,
    )
    filler = FillerPlayer(play_fn=_play_wav)

    log.info("ready — speak into the mic")
    try:
        async for utterance in capture.utterances():
            out = PipelineResult()
            print("\n🎤  utterance captured — processing...")
            chunks: list[bytes] = []
            async for chunk in run_realtime(
                utterance, filler=filler, out=out
            ):
                chunks.append(chunk)
            if not chunks:
                print(f"  (no response — transcription empty: '{out.transcription}')")
                continue
            print(f"  📝 {out.language}: {out.transcription}")
            print(f"  💬 {out.response_text}")
            print(
                f"  ⏱  pipeline_ttfa={out.latency.pipeline_ttfa_ms}ms, "
                f"total={out.latency.total_ms}ms"
            )
            # Play assembled audio
            wav = b"".join(chunks)
            await _play_wav(wav)
    finally:
        capture.stop()


async def _serve_api() -> None:
    """Run FastAPI in the same event loop."""
    import uvicorn
    from backend import config as cfg

    config = cfg.load()
    server_config = uvicorn.Config(
        "backend.app:app",
        host="127.0.0.1",
        port=int(__import__("os").environ.get("HF_API_PORT", "8000")),
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def _main() -> int:
    from backend.logging_config import setup_logging
    from backend import config as cfg
    from backend.services import llm_client

    setup_logging()
    log = logging.getLogger("scripts.start")
    config = cfg.load()

    if config.machine_role != "client":
        log.error(
            "scripts/start.py is for the client. Mac uses scripts/start_llm_server.py.",
            extra={"role": config.machine_role},
        )
        return 1

    # Strict startup: fail fast if Mac is not reachable (office network)
    log.info("checking Mac LLM server", extra={"url": config.llm_service_url})
    if not await llm_client.health_check():
        log.error(
            "Mac LLM server is not reachable. Start it on the Mac:",
            extra={"url": config.llm_service_url},
        )
        log.error("  ssh mac-mini")
        log.error("  cd hands-free && python scripts/start_llm_server.py")
        return 2
    log.info("Mac LLM server is reachable")

    api_task = asyncio.create_task(_serve_api())
    mic_task = asyncio.create_task(_mic_loop(log))

    try:
        await asyncio.gather(api_task, mic_task)
    except KeyboardInterrupt:
        log.info("interrupted by user")
        api_task.cancel()
        mic_task.cancel()
    return 0


def main() -> int:
    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    # Ensure project root is on sys.path when running directly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
