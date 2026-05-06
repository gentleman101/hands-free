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
    "HF_HUB_OFFLINE": "1",              # use cached HuggingFace models — no network calls on startup
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

_STATE_CFG = {
    "listening":  {"colour": "#00c853", "speed": "1.4s", "label": "Listening",  "emoji": "🎤"},
    "processing": {"colour": "#ff6d00", "speed": "0.5s", "label": "Processing", "emoji": "⏳"},
    "speaking":   {"colour": "#00b0ff", "speed": "0.6s", "label": "Speaking",   "emoji": "🔊"},
}

def _build_ui_html(state: str, turns: list[dict]) -> str:
    """Full-page dark UI: waveform background, status, conversation, clear button — all in one HTML block."""
    cfg = _STATE_CFG.get(state, _STATE_CFG["listening"])
    colour, speed = cfg["colour"], cfg["speed"]

    # 32 bars, evenly spaced edge-to-edge
    bars = "".join(
        f'<div class="bar" style="animation-duration:{speed};animation-delay:{i * 0.08:.2f}s;left:{i * 3.125:.2f}%"></div>'
        for i in range(32)
    )

    # Conversation cards (newest first)
    if not turns:
        cards = """
<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:300px;gap:12px'>
  <div style='font-size:48px;opacity:0.3'>🎤</div>
  <div style='color:rgba(255,255,255,0.3);font-size:15px;letter-spacing:0.5px'>Start speaking to begin</div>
</div>"""
    else:
        parts = []
        for t in reversed(turns):
            lang_badge = (
                f"<span style='background:{colour}22;border:1px solid {colour}66;"
                f"color:{colour};border-radius:4px;padding:2px 8px;font-size:11px;"
                f"font-weight:700;letter-spacing:0.5px'>{t['lang']}</span>"
            )
            metrics = (
                f"<span style='color:rgba(255,255,255,0.35);font-size:11px;font-family:monospace'>"
                f"STT {t['stt_ms']}ms &nbsp;·&nbsp; LLM {t['llm_ttft_ms']}ms &nbsp;·&nbsp; Total <b style='color:rgba(255,255,255,0.6)'>{t['total_ms']}ms</b></span>"
            )
            parts.append(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:16px 18px;margin-bottom:10px;color:#fff;
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif'>
  <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
    <span style='color:rgba(255,255,255,0.3);font-size:11px;letter-spacing:0.5px'>TURN {t['turn']}</span>
    {lang_badge}
    <span style='flex:1'></span>
    {metrics}
  </div>
  <div style='margin-bottom:6px;font-size:14px;line-height:1.5'>
    <span style='color:rgba(255,255,255,0.4);font-size:12px;text-transform:uppercase;
                 letter-spacing:0.8px;margin-right:8px'>You</span>{t['heard']}
  </div>
  <div style='font-size:14px;line-height:1.5'>
    <span style='color:{colour};font-size:12px;text-transform:uppercase;
                 letter-spacing:0.8px;margin-right:8px'>Rammas</span>{t['said']}
  </div>
</div>""")
        cards = "".join(parts)

    return f"""
<style>
  /* hide Gradio chrome */
  footer, .gradio-container > .top-panel, #component-0 > .label {{display:none!important}}
  body, .gradio-container, .main {{
    background:#06090f!important; margin:0!important; padding:0!important;
    min-height:100vh!important;
  }}
  .rmwrap {{
    position:relative; min-height:100vh; overflow:hidden;
    background:#06090f;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    padding:28px 32px 32px;
    box-sizing:border-box;
  }}
  .rm-bg {{
    position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden;
  }}
  .bar {{
    position:absolute; bottom:0; width:2.8%; border-radius:3px 3px 0 0;
    background:{colour}; opacity:0.10;
    animation: barpulse {speed} ease-in-out infinite alternate;
  }}
  @keyframes barpulse {{
    from {{ height:5%; }}
    to   {{ height:80%; }}
  }}
  .rm-fg {{ position:relative; z-index:1; max-width:860px; margin:0 auto; }}
  .rm-header {{
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:28px; padding-bottom:18px;
    border-bottom:1px solid rgba(255,255,255,0.07);
  }}
  .rm-title {{
    font-size:18px; font-weight:700; color:#fff; letter-spacing:0.2px;
  }}
  .rm-title span {{ color:rgba(255,255,255,0.35); font-weight:400; }}
  .rm-status {{
    display:flex; align-items:center; gap:8px;
    font-size:13px; font-weight:700; color:{colour};
    letter-spacing:1.5px; text-transform:uppercase;
  }}
  .rm-dot {{
    width:8px; height:8px; border-radius:50%; background:{colour};
    box-shadow:0 0 8px {colour};
    animation: dotpulse {speed} ease-in-out infinite alternate;
  }}
  @keyframes dotpulse {{
    from {{ opacity:0.4; transform:scale(0.8); }}
    to   {{ opacity:1.0; transform:scale(1.2); }}
  }}
  .rm-conv {{ padding-bottom:16px; }}
  .rm-clear {{
    margin-top:8px; cursor:pointer;
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
    color:rgba(255,255,255,0.4); border-radius:8px; padding:8px 20px;
    font-size:12px; letter-spacing:0.5px; font-family:inherit;
    transition:all 0.2s;
  }}
  .rm-clear:hover {{ background:rgba(255,255,255,0.1); color:#fff; }}
</style>
<div class="rmwrap">
  <div class="rm-bg">{bars}</div>
  <div class="rm-fg">
    <div class="rm-header">
      <span class="rm-title">Rammas &nbsp;<span>· DEWA AI Voice Assistant</span></span>
      <div class="rm-status">
        <div class="rm-dot"></div>
        {cfg['label']}
      </div>
    </div>
    <div class="rm-conv">{cards}</div>
    <button class="rm-clear" onclick="this.dispatchEvent(new CustomEvent('clear-conv',{{bubbles:true}}))">
      Clear conversation
    </button>
  </div>
</div>"""


def start_gradio_display(port: int = 7860) -> None:
    """Launch Gradio display window in a daemon thread."""
    import gradio as gr

    _turns: list[dict] = []

    def _refresh():
        while not _display_queue.empty():
            _turns.append(_display_queue.get_nowait())
        return _build_ui_html(_pipeline_status, _turns)

    def _clear():
        _turns.clear()
        return _build_ui_html("listening", [])

    with gr.Blocks(title="Rammas — DEWA AI Voice Assistant") as demo:
        panel = gr.HTML(value=_build_ui_html("listening", []))
        clr   = gr.Button("__clear__", visible=False)
        clr.click(_clear, outputs=[panel])
        gr.Timer(value=0.5).tick(_refresh, outputs=[panel])

    def _run():
        demo.launch(server_port=port, quiet=True, prevent_thread_lock=True)

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
