#!/usr/bin/env python3
"""
Hands-Free — single entry point.

Starts:
  1. FastAPI server (uvicorn, background thread)
  2. Live mic loop (VAD → STT → LLM → TTS → speaker)

Usage:
    python scripts/start.py
    python scripts/start.py --no-api     # mic loop only, no HTTP server
    python scripts/start.py --phase prompted
"""
import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "HF_OLLAMA_BASE_URL": "http://localhost:11434/v1",
    "HF_OLLAMA_MODEL": "gemma4:e2b",
    "HF_WHISPER_MODEL_SIZE": "small",
    "HF_FISH_SPEECH_URL": "http://localhost:8001",
    "HF_LOG_LEVEL": "WARNING",   # keep console clean during voice use
    "HF_PIPELINE_PHASE": "raw",
    "HF_MAX_TOKENS": "2048",             # thinking model needs room to reason before responding
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",  # allow Metal ops to fall back to CPU if unsupported
}


def _apply_defaults() -> None:
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)


# ── preflight checks ──────────────────────────────────────────────────────────

def _check_ollama() -> bool:
    import urllib.request
    base = os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    url = base.replace("/v1", "/api/tags")
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def _check_fish() -> bool:
    import urllib.request
    url = os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001")
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def start_fish_speech() -> None:
    """Launch Fish Speech API server as a background subprocess."""
    import subprocess
    fish_dir = ROOT / "fish-speech"
    server_script = fish_dir / "tools" / "api_server.py"
    if not server_script.exists():
        print("  Fish Speech server script not found — skipping auto-start")
        return

    log_path = ROOT / "data" / "fish_speech.log"
    log_file = open(log_path, "w")  # noqa: SIM115 — intentional long-lived file handle

    python = sys.executable
    proc = subprocess.Popen(
        [python, str(server_script), "--listen", "0.0.0.0:8001", "--device", "mps"],
        cwd=str(fish_dir),
        stdout=log_file,
        stderr=log_file,
    )
    print(f"  Fish Speech starting (pid {proc.pid}) — logs: data/fish_speech.log")
    print("  Waiting for model load", end="", flush=True)
    # Model loading on Mac can take 60s+
    for _ in range(90):
        time.sleep(1)
        print(".", end="", flush=True)
        if proc.poll() is not None:
            print(f"\n  ERROR: Fish Speech exited (code {proc.returncode})")
            print(f"  Check logs: tail -50 {log_path}")
            return
        if _check_fish():
            print(" ✓")
            return
    print("\n  WARNING: Fish Speech did not bind in 90s — TTS may fail")


def preflight() -> None:
    print("\n  Checking dependencies...")

    ollama_ok = _check_ollama()
    fish_ok = _check_fish()

    print(f"  Ollama      {'✓' if ollama_ok else '✗ NOT RUNNING'}")
    print(f"  Fish Speech {'✓' if fish_ok else '○ not running — will start'}")

    if not ollama_ok:
        print("\n  ERROR: Ollama is not running.")
        print("  Start it with:  ollama serve")
        print("  Then pull model: ollama pull gemma4:e2b\n")
        sys.exit(1)

    if not fish_ok:
        start_fish_speech()

    # Warm up Whisper so first utterance has no cold-start delay
    print("  Loading Whisper model...", end="", flush=True)
    from backend.services.stt import preload as preload_whisper
    preload_whisper()
    print(" ✓")


# ── FastAPI server (background thread) ───────────────────────────────────────

def _run_server(port: int) -> None:
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )


def start_server(port: int) -> threading.Thread:
    t = threading.Thread(target=_run_server, args=(port,), daemon=True)
    t.start()
    # Give uvicorn a moment to bind
    time.sleep(1.5)
    return t


# ── audio playback ────────────────────────────────────────────────────────────

async def _play(audio_bytes: bytes) -> None:
    import io
    import wave

    import numpy as np
    import sounddevice as sd  # type: ignore[import-untyped]

    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    except wave.Error:
        # Fish Speech streaming returns raw PCM (int16, 44100Hz) without WAV header
        sr = 44100
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # Normalize to target peak — avoids clipping distortion from fixed gain
    peak = float(np.max(np.abs(pcm)))
    if peak > 0:
        target = float(os.environ.get("HF_AUDIO_TARGET_PEAK", "0.9"))
        pcm = pcm * (target / peak)
    sd.play(pcm, samplerate=sr, blocking=True)


# ── macOS say fallback ────────────────────────────────────────────────────────

async def _say(text: str, language: str) -> None:
    """Use macOS built-in TTS as fallback when Fish Speech is unavailable."""
    # Remove Fish Speech prosody tags before passing to say
    import re
    clean = re.sub(r"\[(?:laugh|whisper|emphasis|sigh|excited|angry)\]", "", text).strip()
    voice = "Layla" if language == "ar" else "Samantha"
    proc = await asyncio.create_subprocess_exec(
        "say", "-v", voice, clean,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


# ── mic loop ──────────────────────────────────────────────────────────────────

async def mic_loop(fish_available: bool) -> None:
    from backend.database import init_db
    from backend.services.audio_capture import AudioCapture, list_input_devices, _pick_input_device
    from backend.services.pipeline import run
    import os

    await init_db()

    # Show input device selection
    devices = list_input_devices()
    device_env = os.environ.get("HF_AUDIO_INPUT_DEVICE")
    if device_env is not None:
        chosen_id = int(device_env)
        chosen_name = next((n for i, n in devices if i == chosen_id), f"device {device_env}")
        mic_label = f"#{chosen_id} {chosen_name}"
    elif devices:
        _chosen = _pick_input_device(devices)
        chosen_name = next((n for i, n in devices if i == _chosen), "unknown") if _chosen is not None else "none"
        mic_label = f"#{_chosen} {chosen_name}" if _chosen is not None else "none found"
    else:
        mic_label = "none found"

    tts_mode = "Fish Speech" if fish_available else "macOS say (fallback)"
    capture = AudioCapture()
    await capture.start()

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Hands-Free is listening                │")
    print("  │  Speak → pause 300ms → response plays   │")
    print("  │  Ctrl+C to quit                         │")
    print(f"  │  MIC: {mic_label:<35}│")
    print(f"  │  TTS: {tts_mode:<35}│")
    print("  └─────────────────────────────────────────┘\n")

    # Conversation history — keeps context across turns
    history: list[dict[str, str]] = []

    try:
        async for utterance_wav in capture.utterances():
            try:
                result = await run(utterance_wav, conversation_history=history)
                if not result.transcription.strip():
                    continue  # noise/silence — stay silent, keep listening
                print("  ▶ processing...", end="", flush=True)
                lat = result.latency
                lang = result.language.upper()
                turn = len(history) // 2 + 1
                # Truncate long texts for display
                heard = result.transcription[:80] + ("…" if len(result.transcription) > 80 else "")
                said = result.response_text[:80] + ("…" if len(result.response_text) > 80 else "")
                print(
                    f"\r  ── Turn {turn} [{lang}] ────────────────────────────────\n"
                    f"  STT  {lat.stt_ms:>5}ms  \"{heard}\"\n"
                    f"  LLM  {lat.llm_ttft_ms:>5}ms ttft  /  {lat.llm_total_ms:>6}ms total\n"
                    f"  TTS  {lat.tts_ttfa_ms:>5}ms ttfa  /  {lat.tts_total_ms:>6}ms total\n"
                    f"       \"{said}\"\n"
                    f"  TTFA {lat.pipeline_ttfa_ms:>5}ms  |  total {lat.total_ms}ms\n"
                )

                # Append this turn to history for next round
                history.append({"role": "user", "content": result.transcription})
                history.append({"role": "assistant", "content": result.response_text})

                # Keep last 10 turns (20 messages) to avoid context overflow
                if len(history) > 20:
                    history = history[-20:]

                if fish_available and result.audio_bytes:
                    await _play(result.audio_bytes)
                else:
                    await _say(result.response_text, result.language)
            except Exception as exc:
                print(f"\r  [error] {exc}\n")
    except KeyboardInterrupt:
        print("\n  Goodbye.\n")
    finally:
        await capture.stop()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hands-Free — start everything")
    parser.add_argument("--no-api", action="store_true", help="Skip HTTP API server")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--phase", default=None, help="Pipeline phase (raw/prompted/lora_v1…)")
    args = parser.parse_args()

    _apply_defaults()
    if args.phase:
        os.environ["HF_PIPELINE_PHASE"] = args.phase

    preflight()

    if not args.no_api:
        start_server(args.port)
        print(f"  API server   http://localhost:{args.port}")
        print(f"  Health       http://localhost:{args.port}/readyz")

    asyncio.run(mic_loop(fish_available=_check_fish()))


if __name__ == "__main__":
    main()
