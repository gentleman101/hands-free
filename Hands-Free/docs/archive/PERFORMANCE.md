# Performance Report — Hands-Free
> Last updated: 2026-05-08 (Fish Speech WSL2 benchmarks; end-to-end pending Mac setup)

---

## Current Status

End-to-end pipeline (STT→LLM→TTS) is **not yet benchmarked** — Mac Mini LLM server pending deployment.

TTS-only benchmarks below are from live Fish Speech 1.5 (WSL2, torch.compile, RTX A3000).

---

## TTS Benchmarks — Fish Speech 1.5 WSL2 (2026-05-08)

*Measured with `scripts/prosody_bench.py`. VQGAN cache warm (after warmup()).*

| Sample | TTFA | Total | Audio |
|---|---|---|---|
| EN with prosody tags | 2175ms | 2499ms | 568 KB |
| AR with prosody tags | 1813ms | 2141ms | 524 KB |
| EN plain (baseline) | 1510ms | 1836ms | 436 KB |

**VQGAN encode (reference audio):**
- Cold (first call after server restart): ~12s
- Warm (LRU cache hit): ~20ms
- `tts.warmup()` called at startup to pre-populate cache

**Why TTS is still ~1.8s:**
Fish Speech generates the full utterance before streaming the first chunk despite `streaming=True`.
Chunk length = 200 tokens reduces TTFA slightly vs default 300. Target is < 500ms with shorter
input sentences (SentenceBuffer min_chars=8 helps).

---

## Expected End-to-End Latency (Estimates — not yet measured)

| Stage | Estimate | Notes |
|---|---|---|
| VAD (TEN VAD) | < 1ms | Fast turn transitions |
| STT (turbo INT8 CUDA) | 150–300ms | EXP-008/009: beam=1, no VAD filter |
| Network RPC (Windows→Mac) | 5–15ms | Connection pool, local WiFi |
| LLM TTFT (JAIS-2 + KV cache) | 50–100ms | KV cache saves 200-500ms/req |
| TTS first chunk (Fish Speech) | ~1.5–2s | Improving; short sentence = shorter wait |
| Filler audio | ~0ms perceived | Plays during STT+LLM+TTS processing |
| **Pipeline TTFA (perceived)** | **~0ms** | Filler hides real latency |
| **Pipeline TTFA (real)** | **~2–3s** | Will improve with sentence length tuning |

---

## Distillation Progress

| Phase | Model | Status |
|---|---|---|
| `raw` | JAIS-2 8B Q4 | [Pending — Mac not set up] |
| `prompted` | JAIS-2 8B Q4 | [Pending] |
| `lora_v1` | JAIS-2 8B + LoRA | [Pending — Phase 4] |

---

## How to Run Benchmarks

```bash
# TTS only (Fish Speech must be running in WSL2):
venv\Scripts\python scripts/prosody_bench.py

# VQGAN codec timing:
venv\Scripts\python scripts/test_voice_codec.py

# Full pipeline (after Mac LLM server is running):
# (benchmark script TBD — will measure pipeline_ttfa_ms from SQLite)
```
