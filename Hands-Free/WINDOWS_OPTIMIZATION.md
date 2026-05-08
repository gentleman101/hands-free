# WINDOWS_OPTIMIZATION.md — Hands-Free on RTX A3000

> Windows 11 + Intel i7-11850H + RTX A3000 6GB VRAM + 32GB RAM
> Documents decisions, benchmarks, and tuning applied.
>
> Last Updated: 2026-05-08

---

## Hardware Reality

| Resource | Available | Notes |
|---|---|---|
| System RAM | 32 GB | Abundant |
| GPU VRAM | 6 GB | RTX A3000 (not 4GB — corrected) |
| GPU | CUDA-capable, RTX A3000 | Used for STT + Fish Speech (WSL2) |
| CPU | i7-11850H (8c/16t) | Runs VAD, intent classifier, filler loading |

---

## VRAM Allocation

| Service | VRAM | Notes |
|---|---|---|
| Fish Speech 1.5 (WSL2) | ~1.84 GB | In WSL2; torch.compile active |
| faster-whisper large-v3-turbo INT8 | ~1 GB | CTranslate2 CUDA |
| **Total** | **~2.84 GB** | ~3 GB headroom remaining |

---

## VAD — TEN VAD

**Chosen over:** Silero VAD (built into faster-whisper)

**Why:** TEN VAD has faster state transitions and lower false-positive rate in office environments (HVAC, typing). Silero VAD is kept as automatic fallback if `ten-vad` is not installed.

**Tuning (from EXP-011):**

| Parameter | Value | Rationale |
|---|---|---|
| Speech threshold | 0.80 | 0.70 fired on HVAC/breath; 0.80 requires intentional speech |
| Silence duration | 600ms | Optimal end-of-speech; 300ms cuts off Arabic trailing vowels |
| Pre-roll chunks | 4 (~128ms) | Prevents first-word clipping |
| Min speech chunks | 8 (~256ms) | Ignores short noise bursts |

---

## STT — faster-whisper large-v3-turbo

**Model:** `large-v3-turbo` (INT8 CUDA)
- Near-large quality at ~6× the speed of large-v3
- ~1 GB VRAM (INT8 quantization via CTranslate2)

**Key optimizations:**

| Setting | Value | Reason |
|---|---|---|
| `beam_size` | 1 | Greedy decoding; 80-200ms faster; quality fine for short utterances (EXP-009) |
| `vad_filter` | False | VAD already done by audio_capture.py; double-VAD wastes time (EXP-008) |
| Language constraint | EN or AR only | Eliminates French/Portuguese misdetections on short phrases (EXP-002) |
| Confidence gate | `no_speech_prob < 0.30` | Discards silence/noise segments |
| Logprob gate | `avg_logprob > -1.0` | Discards low-confidence transcriptions |
| Compression ratio | `< 2.4` | Discards hallucinated repetitive tokens |
| Script validation | Arabic chars ≥ 30% for AR; Latin ≥ 50% for EN | Catches cross-language misclassification |

---

## TTS — Fish Speech 1.5 in WSL2

### Why WSL2?

`torch.compile` requires Triton, which is Linux-only. Without compile, Fish Speech runs at ~10 tok/sec (too slow). In WSL2 with compile:

| Mode | Speed | TTFA |
|---|---|---|
| Windows (no compile) | ~10 tok/sec | ~13s |
| WSL2 + torch.compile | ~65 tok/sec | ~2.1s |

**5× speedup from torch.compile alone.**

### WSL2 Setup Notes

- Venv lives at `~/fish-speech-wsl/venv` (not on `/mnt/c/` — Windows FS causes ensurepip errors)
- PyTorch: `torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128` (CUDA 13.2 requires cu128)
- torchaudio: `torchaudio==2.5.1` pinned (API differences in newer versions)
- WSL2 IP: `172.29.211.12` (find with `wsl hostname -I`)

### Voice Cloning

| Voice | Reference File | Source |
|---|---|---|
| English | `assets/voices/brian_en.wav` | Real English voice MP3, 27s offset, 15s |
| Arabic | `assets/voices/hamdan_ar.wav` | Real Emirati Arabic news MP3, 7s offset, 15s |

Reference audio is base64-encoded and sent per TTS request. Fish Speech LRU-caches the VQGAN encoding keyed by audio bytes — first request costs ~12s, cached requests cost ~20ms. `tts.warmup()` pre-populates the cache at startup.

### Payload Optimizations

| Setting | Value | Effect |
|---|---|---|
| `chunk_length` | 200 | Smaller chunks = lower TTFA (vs default 300) |
| `use_memory_cache` | `"on"` | Server-side caching hint |
| `streaming` | `True` | Yields chunks as generated |

---

## Intent Classifier — TF-IDF

| Metric | Value |
|---|---|
| Inference time | 0.46ms |
| GPU required | No |
| Languages | EN + AR (char n-gram handles both scripts) |
| Intents | 7: billing, location, outage, account, service_request, sustainability, general |
| Confidence threshold | 0.15 (cosine similarity) |
| Fallback | "general" |

**Why not LLM-based intent?** Adds 200ms+ latency and a network hop. TF-IDF is deterministic, instant, and accurate enough for 7 well-separated intents.

---

## Filler Audio

84 WAVs pre-generated at setup time. Loaded into memory at first call.
Yield is `AsyncIterator[bytes]` — same interface as `tts.synthesize_stream`.
Pipeline yields filler bytes FIRST (instant from memory), then real TTS bytes.
This hides the real pipeline latency entirely.

---

## Requirements Split

**No torch in Windows project venv.** Fish Speech is isolated in WSL2.

| File | Contents |
|---|---|
| `requirements-shared.txt` | fastapi, uvicorn, pydantic, httpx, aiosqlite, dotenv, jsonlogger |
| `requirements-windows.txt` | faster-whisper, sounddevice, soundfile, numpy, scikit-learn, ormsgpack |
| `requirements-mac.txt` | llama-cpp-python (Metal) |
