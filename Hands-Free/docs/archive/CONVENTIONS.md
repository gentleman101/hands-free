# CONVENTIONS — Hands-Free (Bilingual Voice AI)

> Claude reads this before any work. Keep it short, specific, and current.
> Style rulebook — see PROJECT.md for narrative, CODE_STRUCTURE.md for live file map.
>
> Last Updated: 2026-05-08

---

## Naming

- **Python modules:** `snake_case.py`
- **Python classes:** `PascalCase`
- **Python functions / vars:** `snake_case`
- **Env vars:** `SCREAMING_SNAKE_CASE`, prefixed `HF_` (e.g. `HF_LLM_SERVICE_URL`)

## File Structure

See `CODE_STRUCTURE.md` — that file is the canonical, always-current map.

## Soft File Size Caps

| Unit | Cap |
|---|---|
| Function | 50 lines |
| Backend service / route | 400 lines |
| Aggregator (`app.py`) | 200 lines |
| Test file | 500 lines |

Before appending to any file, check its current line count. If adding would breach the cap, extract first.

---

## Machine Roles

| Role | Machine | Services |
|---|---|---|
| `client` | Windows (RTX A3000) | audio_capture, stt, intent_classifier, filler, tts, pipeline, llm_client, app |
| `server` | Mac Mini (M4) | llm_server |

Set via `HF_MACHINE_ROLE` in `.env`. The `is_client` / `is_server` helpers in `config.py` guard role-specific code paths.

---

## Pipeline Services

| Service | Library / Backend | Key detail |
|---|---|---|
| VAD | TEN VAD (`ten-vad`) + Silero fallback | Threshold 0.80; 600ms silence; 128ms pre-roll |
| STT | `faster-whisper` large-v3-turbo CUDA INT8 | Beam=1; EN/AR gate (>0.30 prob); confidence gates; script validation |
| Intent | TF-IDF char n-gram (scikit-learn) | 7 intents; bilingual; 0.46ms; threshold 0.15 |
| Filler | Pre-generated WAVs in memory | 84 clips (7×2×6); yields bytes like TTS; no network call |
| LLM | JAIS-2 8B Q4 via llama.cpp (Mac) | HTTP RPC; pooled httpx; NDJSON stream; 3-retry |
| TTS | Fish Speech 1.5 (WSL2 via HTTP) | torch.compile CUDA; voice cloning; use_memory_cache=on; warmup() at startup |
| Audio | `sounddevice` | 16kHz capture; 44100Hz playback |

Each service is a standalone module in `backend/services/` — no service imports another.
`pipeline.py` is the sole orchestrator.

---

## Pipeline Data Flow

```
utterance bytes
  → stt.transcribe()           → TranscribeResult (text, language, confidence)
  → intent_classifier.classify() → intent string
  → filler.stream_filler()     → WAV bytes yielded immediately (from memory)
  → llm_client.generate_stream() → token stream
  → SentenceBuffer.feed()      → sentence strings (boundary: . ? ! , ، ؟, min 8 chars)
  → tts.synthesize_stream()    → WAV bytes yielded per sentence
```

---

## Filler Audio Rules

- Always yield filler FIRST (before any TTS) — pipeline.py handles this
- Filler selection: `intent_classifier.classify(transcript)` → `filler.stream_filler(intent, language)`
- Falls back to `general` intent if intent-specific clips missing
- 6 variants per intent per language (3 formal, 3 casual) — random selection each turn

---

## Language / Bilingual Rules

- Language detection: faster-whisper `detect_language()` → EN or AR (force one; reject if max prob < 0.30)
- System prompt: two separate prompts (EN-only vs AR-only) — `prompt_builder.py`
- Arabic: respond in the same dialect/register the user used
- Fallback: if detection fails → "unknown" → pipeline skips (no empty LLM calls)

---

## Fish Speech (WSL2)

- Runs in WSL2: `~/fish-speech-wsl/fish-speech/` with separate venv at `~/fish-speech-wsl/venv`
- Start command: `python tools/api_server.py --llama-checkpoint-path checkpoints/fish-speech-1.5 --decoder-checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth --listen 0.0.0.0:8001 --device cuda --compile`
- Windows accesses it at `http://172.29.211.12:8001` (WSL2 IP — check with `wsl hostname -I`)
- **Do NOT install torch or fish-speech in the main Windows project venv**
- `tts.warmup()` must be called at startup to pre-populate the VQGAN LRU cache (avoids 12s penalty on first real request)

---

## API Patterns

- All endpoints are async
- Streaming endpoints use `StreamingResponse` (FastAPI)
- Errors: raise `HTTPException` with appropriate status code
- Never block the event loop — use `asyncio.get_event_loop().run_in_executor` for sync calls

---

## Dependency Rules

| Package | Where |
|---|---|
| `faster-whisper`, `sounddevice`, `soundfile`, `numpy`, `scikit-learn`, `ormsgpack` | `requirements-windows.txt` only |
| `llama-cpp-python` | `requirements-mac.txt` only |
| `fastapi`, `uvicorn`, `pydantic`, `httpx`, `aiosqlite`, `python-dotenv`, `python-json-logger` | `requirements-shared.txt` |
| `torch` | WSL2 venv only — never in project venv |

---

## Development Rules

**Rule 1 (B001):** Any standalone script using `insert_event` must call `await init_db()` first. Do not assume FastAPI lifespan has run.

**Rule 2:** Never mark a bug resolved without a regression test.

**Rule 3:** Before any push — `venv\Scripts\python -m pytest backend/tests/ -q` must be green.
