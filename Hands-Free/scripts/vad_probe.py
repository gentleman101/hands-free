"""
Live VAD diagnostic — shows real-time probabilities, speech/silence transitions,
and utterance durations. Use to verify your mic + VAD config before running start.py.

Output:
  prob=0.12 silence
  prob=0.34 silence
  prob=0.87 SPEECH ← start
  prob=0.91 SPEECH
  prob=0.04 silence (600ms triggers end)
  >>> utterance #1: 1.42s (44 speech chunks)

Usage:
  cd C:\\Users\\dewac\\Documents\\Naman\\Hands-Free
  venv\\Scripts\\activate
  python scripts/vad_probe.py
  python scripts/vad_probe.py --threshold 0.5
  python scripts/vad_probe.py --device 1
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# Project root on path + load .env
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import io
import wave

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

# Match the same constants the real pipeline uses
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_MS = int(CHUNK_SAMPLES / SAMPLE_RATE * 1000)
SILENCE_DURATION_MS = 600
SILENCE_CHUNKS = SILENCE_DURATION_MS // CHUNK_MS
MIN_SPEECH_CHUNKS = 8
PREFIX_CHUNKS = 4


def pick_device(name_filter: str | None, device_id: int | None) -> int:
    if device_id is not None:
        return device_id
    devices = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0]
    if name_filter:
        for i, n in devices:
            if name_filter.lower() in n.lower():
                return i
    # default: first openable
    for i, n in devices:
        try:
            with sd.InputStream(device=i, channels=1, samplerate=SAMPLE_RATE):
                pass
            return i
        except Exception:
            continue
    raise RuntimeError("no openable input device")


def make_vad(threshold: float):
    try:
        from ten_vad import TenVad
        v = TenVad()
        print(f"VAD backend: TEN VAD (threshold {threshold})")

        def is_speech(chunk: np.ndarray) -> float:
            pcm16 = (chunk * 32767).astype(np.int16)
            best = 0.0
            for i in (0, 256):
                p, _ = v.process(pcm16[i:i + 256])
                best = max(best, p)
            return best

        return is_speech
    except ImportError:
        print(f"VAD backend: energy RMS fallback (threshold {threshold})")

        def is_speech_rms(chunk: np.ndarray) -> float:
            return float(np.sqrt(np.mean(chunk ** 2)))

        return is_speech_rms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--name", type=str, default=None,
                        help="substring match on device name (e.g. 'microphone array')")
    parser.add_argument("--verbose", action="store_true",
                        help="print probability for every chunk (default: only on transitions)")
    args = parser.parse_args()

    dev = pick_device(args.name, args.device)
    devname = sd.query_devices(dev)["name"]
    print(f"device: {dev} — {devname}")
    print(f"sample rate: {SAMPLE_RATE} Hz, chunk size: {CHUNK_SAMPLES} ({CHUNK_MS} ms)")
    print(f"silence-to-end: {SILENCE_DURATION_MS} ms ({SILENCE_CHUNKS} chunks)")
    print(f"min speech: {MIN_SPEECH_CHUNKS} chunks (~{MIN_SPEECH_CHUNKS * CHUNK_MS} ms)")
    print(f"pre-roll: {PREFIX_CHUNKS} chunks (~{PREFIX_CHUNKS * CHUNK_MS} ms)")
    print(f"\nSpeak. Ctrl+C to exit.\n")

    is_speech = make_vad(args.threshold)
    import queue
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print(f"  (status: {status})")
        q.put(indata[:, 0].copy())

    state = "idle"
    speech_count = 0
    silence_count = 0
    utterance_start = 0.0
    utt_num = 0
    max_prob = 0.0
    prefix: deque = deque(maxlen=PREFIX_CHUNKS)
    utterance_buf: list[np.ndarray] = []
    save_dir = ROOT / "data" / "vad_captures"
    save_dir.mkdir(parents=True, exist_ok=True)

    def save_wav(samples: np.ndarray, path: Path) -> None:
        pcm = (samples * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    with sd.InputStream(device=dev, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=CHUNK_SAMPLES, dtype="float32", callback=cb):
        try:
            while True:
                chunk = q.get()
                prob = is_speech(chunk)
                is_sp = prob >= args.threshold

                if state == "idle":
                    prefix.append(chunk)
                    if args.verbose:
                        print(f"  prob={prob:.2f} silence")
                    if is_sp:
                        state = "speech"
                        speech_count = 1
                        silence_count = 0
                        max_prob = prob
                        utterance_start = time.perf_counter()
                        utterance_buf = list(prefix) + [chunk]
                        print(f"  prob={prob:.2f} SPEECH-START")
                else:
                    utterance_buf.append(chunk)
                    if args.verbose:
                        print(f"  prob={prob:.2f} {'SPEECH' if is_sp else 'silence'}")
                    if is_sp:
                        speech_count += 1
                        silence_count = 0
                        max_prob = max(max_prob, prob)
                    else:
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS:
                            dur = time.perf_counter() - utterance_start
                            ms = int(dur * 1000)
                            if speech_count >= MIN_SPEECH_CHUNKS:
                                utt_num += 1
                                samples = np.concatenate(utterance_buf)
                                wav_path = save_dir / f"utterance_{utt_num:03d}.wav"
                                save_wav(samples, wav_path)
                                rms = float(np.sqrt(np.mean(samples ** 2)))
                                peak = float(np.max(np.abs(samples)))
                                print(f"\n  >>> utterance #{utt_num}: {ms} ms, "
                                      f"{speech_count} speech chunks, peak prob={max_prob:.2f}")
                                print(f"      audio: rms={rms:.3f} peak={peak:.3f}")
                                print(f"      saved: {wav_path.relative_to(ROOT)}\n")
                            else:
                                print(f"  (skipped: only {speech_count} speech chunks, "
                                      f"need {MIN_SPEECH_CHUNKS})")
                            state = "idle"
                            speech_count = 0
                            silence_count = 0
                            max_prob = 0.0
                            prefix.clear()
                            utterance_buf = []
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
