# CODE_STRUCTURE — Hands-Free

> Canonical, always-current map of every meaningful file.
> Update whenever you create, rename, move, or delete a file.
>
> Last Updated: 2026-05-08

---

## Current Project State

**Architecture:** Distributed — Windows (STT + TTS) ↔ Mac Mini (LLM)
**Windows status:** All services complete, 35 tests passing
**Mac status:** Code complete — pending first deployment (llama.cpp + JAIS-2)
**End-to-end status:** Pending Mac setup
**Target pipeline TTFA:** < 600ms real (perceived ~0ms with filler audio)

---

## Directory Tree

```
Hands-Free/
│
├── backend/
│   ├── __init__.py
│   ├── app.py                          [COMPLETE] FastAPI lifespan; loads config, DB, TTS warmup; registers routers
│   ├── config.py                       [COMPLETE] HF_* env vars → Config dataclass; is_client / is_server helpers
│   ├── database.py                     [COMPLETE] aiosqlite; init_db(); insert_event(); fire_and_forget()
│   ├── logging_config.py               [COMPLETE] Structured JSON logging via python-json-logger
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py                   [COMPLETE] GET /healthz (liveness), /readyz (LLM + TTS reachability)
│   │   └── voice.py                    [COMPLETE] POST /api/voice/transcribe, /pipeline/stream (SSE)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   │
│   │   │   ── Windows (client) ──
│   │   ├── audio_capture.py            [COMPLETE] sounddevice + TEN VAD (Silero fallback); device priority; pre-roll; utterances()
│   │   ├── stt.py                      [COMPLETE] faster-whisper large-v3-turbo CUDA INT8; EN/AR gate; confidence filters; script validation
│   │   ├── intent_classifier.py        [COMPLETE] TF-IDF char n-gram; 7 intents; bilingual EN+AR; 0.46ms inference
│   │   ├── filler.py                   [COMPLETE] Loads 84 WAVs from assets/fillers/; stream_filler() yields chunks
│   │   ├── tts.py                      [COMPLETE] Fish Speech 1.5 HTTP stream (WSL2); HTTPX singleton; voice cloning; warmup()
│   │   ├── pipeline.py                 [COMPLETE] STT→intent→filler→LLM→SentenceBuffer→TTS; yields audio bytes
│   │   ├── prompt_builder.py           [COMPLETE] Two language-specific prompts EN/AR; <20 words system prompt
│   │   ├── llm_client.py               [COMPLETE] HTTP RPC to Mac; HTTPX pool; 3-retry; streaming token yield
│   │   │
│   │   │   ── Mac Mini (server) ──
│   │   └── llm_server.py               [COMPLETE] FastAPI + llama.cpp Metal; JAIS-2 8B; KV cache; NDJSON stream
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── latency_tracker.py          [COMPLETE] @track_latency decorator; fire-and-forget SQLite writes
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py                 [COMPLETE] silent_wav_bytes fixture; patch_metrics autouse stub
│       ├── services/
│       │   ├── __init__.py
│       │   ├── test_intent_classifier.py  [COMPLETE] 10 tests: EN+AR billing/location/outage/account/service_request
│       │   ├── test_pipeline.py           [COMPLETE] SentenceBuffer: EN/AR boundaries, comma, flush, min_chars
│       │   ├── test_prompt_builder.py     [COMPLETE] Language selection, system first, user last, history
│       │   ├── test_stt_helpers.py        [COMPLETE] _script_matches_language: Arabic/Latin detection, edge cases
│       │   └── test_tts.py                [COMPLETE] strip_prosody_tags: tags, closing tags, Arabic pass-through
│       ├── api/
│       │   └── __init__.py
│       └── integration/
│           └── __init__.py
│
├── assets/
│   ├── fillers/
│   │   ├── en/                         [COMPLETE] 42 WAVs — 7 intents × 6 variants, synthesized with brian_en voice
│   │   │   └── {intent}_{1-6}.wav
│   │   └── ar/                         [COMPLETE] 42 WAVs — 7 intents × 6 variants, synthesized with hamdan_ar voice
│   │       └── {intent}_{1-6}.wav
│   └── voices/
│       ├── brian_en.wav                [COMPLETE] English reference — real voice from MP3, 27s offset, 15s clip
│       └── hamdan_ar.wav               [COMPLETE] Arabic reference — real Emirati voice from MP3, 7s offset, 15s clip
│
├── scripts/
│   ├── start.py                        [COMPLETE] Windows entry: health-check Mac → FastAPI + mic loop
│   ├── start_llm_server.py             [COMPLETE] Mac entry: role=server → uvicorn llm_server.app
│   ├── generate_fillers.py             [COMPLETE] Synthesize 84 filler WAVs via Fish Speech (run once)
│   ├── generate_reference_voices.py    [COMPLETE] Generate reference WAVs via edge-tts (run once, optional)
│   ├── prosody_bench.py                [COMPLETE] EN+AR prosody latency benchmark; saves + plays WAVs
│   └── test_voice_codec.py             [COMPLETE] VQGAN encode reference WAV; measure cache hit vs cold timing
│
├── fish-speech/                        [WSL2 ONLY — not committed to git]
│   └── (Fish Speech v1.5.1 clone, venv at ~/fish-speech-wsl/venv)
│
├── requirements-shared.txt             [COMPLETE] fastapi, uvicorn, pydantic, httpx, aiosqlite, dotenv, jsonlogger
├── requirements-windows.txt            [COMPLETE] faster-whisper, sounddevice, soundfile, numpy, scikit-learn, ormsgpack
├── requirements-mac.txt                [COMPLETE] llama-cpp-python (Metal)
├── requirements-dev.txt                [COMPLETE] pytest, pytest-asyncio, ruff, mypy
├── .env.example                        [COMPLETE] All HF_* vars with defaults
├── .gitignore                          [EXISTS]
│
├── CODE_STRUCTURE.md                   [THIS FILE]
├── PROJECT.md                          [EXISTS] PMBOK project log
├── CONVENTIONS.md                      [EXISTS] Style + pipeline rules
├── INSTRUCTIONS.md                     [EXISTS] Setup guide + bug log
├── PERFORMANCE.md                      [EXISTS] Latency benchmarks
├── EXPERIMENTS.md                      [EXISTS] EXP-001 through EXP-017+
├── WINDOWS_OPTIMIZATION.md             [EXISTS] Windows-specific tuning notes
├── STT_OPTIMIZATION.md                 [EXISTS] Whisper EN/AR tuning
└── DISTILLATION.md                     [EXISTS] Dataset + fine-tuning roadmap
```

---

## Service Contracts

### `backend/services/audio_capture.py`
```python
class AudioCapture:
    def __init__(self, vad_backend: str = "ten", device_env: str = "") -> None
    async def utterances(self) -> AsyncIterator[bytes]  # yields WAV bytes per utterance
    def stop(self) -> None
```

### `backend/services/stt.py`
```python
@dataclass
class TranscribeResult:
    text: str
    language: str       # "en" | "ar" | "unknown"
    confidence: float
    duration_ms: int

async def transcribe(audio_bytes: bytes, *, request_id: str | None = None) -> TranscribeResult
```

### `backend/services/intent_classifier.py`
```python
def classify(text: str) -> str
# Returns one of: billing | location | outage | account | service_request | sustainability | general
# Bilingual (EN+AR). Inference: ~0.46ms. Falls back to "general" on low confidence (<0.15).
```

### `backend/services/filler.py`
```python
async def stream_filler(intent: str, language: str, chunk_size: int = 4096) -> AsyncIterator[bytes]
# Yields WAV bytes from memory. Same interface as tts.synthesize_stream.
# 84 pre-generated WAVs loaded at first call. Falls back to "general" if intent missing.
```

### `backend/services/tts.py`
```python
@dataclass
class SynthesizeResult:
    audio_bytes: bytes
    sample_rate: int    # 44100
    ttfa_ms: int
    duration_ms: int

async def synthesize_stream(text: str, language: str = "en", *, request_id: str | None = None) -> AsyncIterator[bytes]
async def synthesize(text: str, language: str = "en", *, request_id: str | None = None) -> SynthesizeResult
async def warmup() -> bool         # pre-encode reference audio into Fish Speech LRU cache
async def health_check() -> bool
```

### `backend/services/llm_client.py`  *(Windows side)*
```python
async def generate_stream(messages, *, language, max_tokens, request_id) -> AsyncIterator[str]
async def generate(messages, *, language, max_tokens, request_id) -> GenerateResult
async def health_check(timeout_s: float = 2.0) -> bool
```

### `backend/services/llm_server.py`  *(Mac side — FastAPI app)*
```
POST /api/llm/generate  → StreamingResponse (NDJSON: {"token": str, "done": bool})
GET  /healthz           → {"status": "ok", "model": str}
GET  /readyz            → {"status": "ready"} or 503
```

### `backend/services/pipeline.py`
```python
@dataclass
class PipelineResult:
    transcription: str
    language: str
    intent: str
    response_text: str
    latency: PipelineLatency   # stt_ms, llm_ttft_ms, pipeline_ttfa_ms, total_ms

async def run_realtime(
    utterance_bytes: bytes,
    *,
    request_id: str | None = None,
    conversation_history: list | None = None,
    out: PipelineResult | None = None,
) -> AsyncIterator[bytes]
# Yields: filler WAV bytes first (instant from memory), then real TTS audio chunks
```

---

## API Endpoints

| Method | Path | Host | Description |
|---|---|---|---|
| POST | `/api/voice/transcribe` | Windows | STT only → text + language |
| POST | `/api/voice/pipeline/stream` | Windows | Full pipeline → streaming audio (SSE) |
| GET | `/healthz` | Both | Liveness probe |
| GET | `/readyz` | Both | Readiness (checks external deps) |
| POST | `/api/llm/generate` | Mac | LLM streaming NDJSON (internal) |

---

## Environment Variables (`.env.example`)

```
# Machine role
HF_MACHINE_ROLE=client            # "client" (Windows) or "server" (Mac)

# LLM — Mac Mini endpoint
HF_LLM_SERVICE_URL=http://mac-mini.local:8080
HF_LLM_TIMEOUT_S=10.0
HF_LLM_MAX_RETRIES=3
HF_MAX_TOKENS=200
HF_LLM_TEMPERATURE=0.7

# LLM — Mac server config
HF_LLM_MODEL_PATH=/Users/naman/models/jais-2-8b-q4.gguf
HF_LLM_MODEL_NAME=jais-2-8b-q4
HF_LLM_PORT=8080
HF_LLM_N_GPU_LAYERS=99
HF_LLM_N_CTX=2048

# STT — Windows
HF_WHISPER_MODEL=large-v3-turbo
HF_WHISPER_DEVICE=cuda
HF_WHISPER_COMPUTE_TYPE=int8

# TTS — Fish Speech in WSL2
HF_FISH_SPEECH_URL=http://172.29.211.12:8001
HF_TTS_ENGINE=fish

# Audio — Windows
HF_VAD_BACKEND=ten
HF_AUDIO_INPUT_DEVICE=

# Metrics
HF_PIPELINE_PHASE=raw
HF_LOG_LEVEL=INFO
```

---

## Status Markers

`[PLANNED]` `[EXISTS]` `[COMPLETE]` `[DEPRECATED]`

| Event | Action |
|---|---|
| File created | `[PLANNED]` → `[EXISTS]` |
| File complete + tested | `[EXISTS]` → `[COMPLETE]` |
| File deleted | Remove from tree; note in PROJECT.md Change Log |
| Service signature changed | Update Service Contracts |
| New endpoint | Update API Endpoints table |
| New env var | Update Environment Variables |
