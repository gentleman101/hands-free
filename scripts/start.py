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
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ── shared display state (mic_loop → Gradio thread) ──────────────────────────
_display_queue: queue.Queue = queue.Queue()
_pipeline_status: str = "listening"  # "listening" | "processing" | "speaking"

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "HF_OLLAMA_BASE_URL": "http://localhost:11434/v1",
    "HF_OLLAMA_MODEL": "gemma4:e2b",
    "HF_WHISPER_MODEL_SIZE": "small",
    "HF_FISH_SPEECH_URL": "http://localhost:8001",
    "HF_LOG_LEVEL": "WARNING",   # keep console clean during voice use
    "HF_PIPELINE_PHASE": "raw",
    "HF_MAX_TOKENS": "200",              # short conversational replies — reasoning is OFF, no need for room
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",  # allow Metal ops to fall back to CPU if unsupported
    "HF_TTS_ENGINE": "say",              # "say" = Apple built-in (fast) | "fish" = Fish Speech (quality)
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
    # Override PYTORCH_ENABLE_MPS_FALLBACK=0 for Fish Speech — the parent process sets it to 1
    # for Whisper, but for Fish Speech it causes silent CPU fallback on every unsupported MPS op,
    # which kills throughput (8 tok/sec instead of 300+ tok/sec on Apple Silicon).
    fish_env = os.environ.copy()
    fish_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    proc = subprocess.Popen(
        [python, str(server_script), "--listen", "0.0.0.0:8001", "--device", "mps", "--half", "--compile"],
        cwd=str(fish_dir),
        stdout=log_file,
        stderr=log_file,
        env=fish_env,
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

    using_fish = os.environ.get("HF_TTS_ENGINE", "say") == "fish"
    print(f"  Ollama      {'✓' if ollama_ok else '✗ NOT RUNNING'}")
    if using_fish:
        print(f"  Fish Speech {'✓' if fish_ok else '○ not running — will start'}")

    if not ollama_ok:
        print("\n  ERROR: Ollama is not running.")
        print("  Start it with:  ollama serve")
        print("  Then pull model: ollama pull gemma4:e2b\n")
        sys.exit(1)

    if os.environ.get("HF_TTS_ENGINE", "say") == "fish":
        if not fish_ok:
            start_fish_speech()
    else:
        print("  TTS engine      Apple say (HF_TTS_ENGINE=fish to switch)")

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
    """Play a complete audio buffer (WAV or raw PCM). Used by macOS say fallback."""
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


async def _play_realtime_stream(gen) -> None:  # type: ignore[no-untyped-def]
    """
    Consume an async generator of raw PCM int16 chunks (44100Hz) from Fish Speech
    and play them gaplessly via a continuous sounddevice OutputStream.

    Normalisation is applied to the whole response after collection — streaming
    normalisation would require a look-ahead which adds latency. We accept
    natural Fish Speech gain here (already reasonable) and apply a mild ceiling.
    """
    import numpy as np
    import sounddevice as sd  # type: ignore[import-untyped]

    SR = 44100

    with sd.OutputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        async for raw in gen:
            if not raw:
                continue
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            # Write immediately — OutputStream buffers and plays gaplessly
            stream.write(pcm)
        # Context manager __exit__ calls stream.stop() + close(), draining the buffer


# ── status line ──────────────────────────────────────────────────────────────

def _status(msg: str) -> None:
    """Overwrite the current terminal line with a status message (no newline)."""
    print(f"\r  {msg:<50}", end="", flush=True)


# ── turn display ─────────────────────────────────────────────────────────────

def _fmt_turn(turn: int, lang: str, heard: str, said: str, ctx: dict) -> str:
    stt     = ctx.get("stt_ms", 0)
    ttft    = ctx.get("llm_ttft_ms", 0)
    llm_tot = ctx.get("llm_total_ms", 0)
    ttfa    = ctx.get("tts_ttfa_ms", 0)
    tts_tot = ctx.get("tts_total_ms", 0)
    p_ttfa  = ctx.get("pipeline_ttfa_ms", 0)
    total   = ctx.get("total_ms", 0)
    return (
        f"  ── Turn {turn} [{lang}] ──────────────────────────────\n"
        f"  Whisper  {stt:>6,} ms  \"{heard}\"\n"
        f"  LLM TTFT {ttft:>6,} ms  →  total {llm_tot:>7,} ms\n"
        f"  TTS TTFA {ttfa:>6,} ms  →  total {tts_tot:>7,} ms\n"
        f"  {'─' * 47}\n"
        f"  TTFA     {p_ttfa:>6,} ms  |  total  {total:>7,} ms\n"
        f"       \"{said}\"\n"
    )


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
    from backend.services.pipeline import run_realtime
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

    tts_engine = os.environ.get("HF_TTS_ENGINE", "say")
    tts_mode = "Fish Speech (streaming)" if tts_engine == "fish" else "Apple say (EN: Samantha, AR: Layla)"
    capture = AudioCapture()
    await capture.start()

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Hands-Free is listening                │")
    print("  │  Speak → pause → response plays         │")
    print("  │  Ctrl+C to quit                         │")
    print(f"  │  MIC: {mic_label:<35}│")
    print(f"  │  TTS: {tts_mode:<35}│")
    print("  └─────────────────────────────────────────┘\n")

    # Conversation history — keeps context across turns
    history: list[dict[str, str]] = []

    _status("🎤  Listening...")

    try:
        async for utterance_wav in capture.utterances():
            try:
                ctx: dict = {}

                _status("⏳  Processing — please wait...")
                global _pipeline_status
                _pipeline_status = "processing"
                capture.mute()   # block mic from queuing while we process + play
                try:
                    if tts_engine == "fish" and fish_available:
                        # Streaming path — gapless playback via OutputStream
                        # run_realtime populates ctx after generator exhausts
                        await _play_realtime_stream(
                            run_realtime(utterance_wav, conversation_history=history, out=ctx)
                        )
                    else:
                        # Apple say path — skip Fish Speech TTS entirely
                        async for _ in run_realtime(
                            utterance_wav,
                            conversation_history=history,
                            out=ctx,
                            skip_tts=True,
                        ):
                            pass  # nothing yielded; ctx populated after exhaustion
                        if ctx.get("response_text"):
                            _pipeline_status = "speaking"
                            await _say(ctx["response_text"], ctx.get("language", "en"))
                    # Settle delay — speaker physically rings ~200ms after OutputStream closes;
                    # keeping mic muted a little longer prevents self-pickup
                    await asyncio.sleep(0.4)
                finally:
                    _pipeline_status = "listening"
                    capture.unmute()  # always re-open mic, even on error

                transcription = ctx.get("transcription", "")
                if not transcription.strip():
                    _status("🎤  Listening...")
                    continue  # noise/silence — stay silent, keep listening

                response_text = ctx.get("response_text", "")
                language = ctx.get("language", "en")
                turn = len(history) // 2 + 1

                heard = transcription[:80] + ("…" if len(transcription) > 80 else "")
                said = response_text[:80] + ("…" if len(response_text) > 80 else "")
                print()   # end the status line before printing the turn block
                print(_fmt_turn(turn, language.upper(), heard, said, ctx))

                # Push turn to Gradio display
                _display_queue.put({
                    "turn": turn,
                    "lang": language.upper(),
                    "heard": transcription,
                    "said": response_text,
                    "stt_ms": ctx.get("stt_ms", 0),
                    "llm_ttft_ms": ctx.get("llm_ttft_ms", 0),
                    "total_ms": ctx.get("total_ms", 0),
                })

                # Append this turn to history for next round
                history.append({"role": "user", "content": transcription})
                history.append({"role": "assistant", "content": response_text})

                # Keep last 10 turns (20 messages) to avoid context overflow
                if len(history) > 20:
                    history = history[-20:]

                _status("🎤  Listening...")

            except Exception as exc:
                print(f"\r  [error] {exc}\n")
                _status("🎤  Listening...")
    except KeyboardInterrupt:
        print("\n\n  Goodbye.\n")
    finally:
        await capture.stop()


# ── Gradio display ───────────────────────────────────────────────────────────

def _waveform_html(state: str) -> str:
    """Return animated waveform HTML for the given pipeline state."""
    colours = {
        "listening":  ("#00c853", "1.4s"),   # green, slow pulse
        "processing": ("#ff6d00", "0.5s"),   # orange, fast pulse
        "speaking":   ("#00b0ff", "0.6s"),   # blue, medium pulse
    }
    colour, speed = colours.get(state, colours["listening"])
    label = {"listening": "🎤  Listening…", "processing": "⏳  Processing…", "speaking": "🔊  Speaking…"}.get(state, "")
    bars = "".join(
        f'<span style="animation-duration:{speed};animation-delay:{i*0.1:.1f}s"></span>'
        for i in range(10)
    )
    return f"""
<style>
  .waveform {{ display:flex; align-items:center; gap:4px; height:48px; }}
  .waveform span {{
    display:inline-block; width:6px; border-radius:3px;
    background:{colour};
    animation: pulse {speed} ease-in-out infinite alternate;
  }}
  @keyframes pulse {{
    from {{ height:6px; opacity:0.4; }}
    to   {{ height:40px; opacity:1.0; }}
  }}
  .wlabel {{ margin-left:14px; font-size:15px; color:#555; font-family:sans-serif; }}
</style>
<div style="display:flex;align-items:center;padding:12px 0">
  <div class="waveform">{bars}</div>
  <span class="wlabel">{label}</span>
</div>"""


def start_gradio_display(port: int = 7860) -> None:
    """Launch Gradio display window in a daemon thread."""
    import gradio as gr

    _turns: list[dict] = []

    def _refresh():
        while not _display_queue.empty():
            _turns.append(_display_queue.get_nowait())

        wave = _waveform_html(_pipeline_status)

        if not _turns:
            conv = "<p style='color:#aaa;font-family:sans-serif;padding:16px'>No turns yet — start speaking.</p>"
        else:
            parts = []
            for t in reversed(_turns):
                lang_badge = f"<span style='background:#007A4D;color:#fff;border-radius:4px;padding:2px 7px;font-size:11px;font-family:sans-serif'>{t['lang']}</span>"
                metrics = f"<span style='color:#aaa;font-size:11px;font-family:monospace'>STT {t['stt_ms']}ms · LLM {t['llm_ttft_ms']}ms · Total {t['total_ms']}ms</span>"
                parts.append(f"""
<div style='border:1px solid #e0e0e0;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-family:sans-serif'>
  <div style='margin-bottom:8px'>Turn {t['turn']} &nbsp;{lang_badge}&nbsp;&nbsp;{metrics}</div>
  <div style='margin-bottom:4px'><b>You:</b> {t['heard']}</div>
  <div style='color:#007A4D'><b>Rammas:</b> {t['said']}</div>
</div>""")
            conv = "".join(parts)

        return wave, conv

    def _clear():
        _turns.clear()
        return _waveform_html("listening"), ""

    with gr.Blocks(title="Rammas — DEWA AI Voice Assistant", theme=gr.themes.Default()) as demo:
        gr.Markdown("## Rammas · DEWA AI Voice Assistant")
        waveform  = gr.HTML(value=_waveform_html("listening"))
        conv_html = gr.HTML(value="<p style='color:#aaa;font-family:sans-serif;padding:16px'>No turns yet — start speaking.</p>")
        gr.Button("Clear conversation", size="sm").click(_clear, outputs=[waveform, conv_html])
        demo.load(_refresh, outputs=[waveform, conv_html], every=0.5)

    def _run():
        demo.launch(server_port=port, quiet=True, prevent_thread_lock=True, show_api=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(2)  # give Gradio a moment to bind
    print(f"  Display UI   http://localhost:{port}")


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
    start_gradio_display()

    if not args.no_api:
        start_server(args.port)
        print(f"  API server   http://localhost:{args.port}")
        print(f"  Health       http://localhost:{args.port}/readyz")

    tts_engine = os.environ.get("HF_TTS_ENGINE", "say")
    asyncio.run(mic_loop(fish_available=tts_engine == "fish" and _check_fish()))


if __name__ == "__main__":
    main()
