# CONVENTIONS — Hands-Free (Bilingual Voice AI)

> Claude reads this before any work. Keep it short, specific, and current.
> This is the style rulebook — see `PROJECT.md` for narrative history, `CODE_STRUCTURE.md` for live file map.

## Naming

- **Python modules:** `snake_case.py`
- **Python classes:** `PascalCase`
- **Python functions / vars:** `snake_case`
- **Vue components:** `PascalCase.vue` (e.g. `VoicePanel.vue`)
- **Vue composables:** `useX.js` (e.g. `useVoicePipeline.js`)
- **Vue views:** `PascalCase.vue` in `src/views/`
- **CSS classes:** `kebab-case`, scoped styles only
- **Env vars:** `SCREAMING_SNAKE_CASE`, prefixed `HF_` (e.g. `HF_GEMMA_MODEL_PATH`)

## File Structure

See `CODE_STRUCTURE.md` — that file is the canonical, always-current map. This section intentionally defers to it.

## Soft File Size Caps

- Function: 50 lines
- Vue SFC: 250 lines
- Composable / util: 200 lines
- Backend service / route: 400 lines
- Aggregator (`app.py`, `main.py`): 200 lines
- Test file: 500 lines

Before appending to any file, check its current line count. If adding would breach the cap, extract first.

## Models & Hardware

- LLM: Ollama serving `gemma4:e2b` — `ollama pull gemma4:e2b` before first run
- LiteRT local model (`/Users/naman/.saltpepper/models/gemma-4-E2B-it.litertlm`) is NOT used in pipeline (Ollama replaces it; kept for reference)
- Hardware: Mac Mini Apple Silicon — Metal/MLX via Ollama; Fish Speech on MPS (slow, acceptable for POC)
- Do NOT load any model at import time; load lazily on first request or `/readyz` warmup
- Production (Azure): same Ollama Docker image, CUDA GPU for Fish Speech real-time

## Pipeline Services

| Service | Library | Key detail |
|---|---|---|
| STT | `faster-whisper` | Silero VAD built-in; use utterance-triggered mode, not streaming |
| LLM | Ollama `gemma4:e2b` | OpenAI SDK `stream=True`; `base_url=HF_OLLAMA_BASE_URL` |
| TTS | `fish-speech` 1.5 | HTTP chunked streaming; DAC decoder only; MPS on Mac |
| Audio | `sounddevice` | 16kHz capture, 22050Hz playback |

Each service is a standalone module in `backend/services/`; no service imports another (pipeline.py orchestrates).
`audio_capture.py` owns VAD + buffering. `pipeline.py` owns `SentenceBuffer` + orchestration.

## Language / Bilingual Rules

- Default language detection: auto (Whisper detects it)
- System prompt must explicitly instruct: respond in the same language as the user
- Arabic: support both MSA and Gulf/UAE dialect
- Fallback: if language detection fails, default to English

## API Patterns

- REST verbs + resource paths: `POST /api/voice/transcribe`, `POST /api/voice/respond`, `GET /api/pipeline/status`
- All request/response bodies validated with Pydantic
- Errors: `HTTPException(status_code=..., detail=...)` — never `{"error": "..."} + 200`
- Streaming response for TTS audio (chunked)
- Structured log on every non-2xx

## Testing

- **Backend:** `pytest` in `backend/tests/`. Mirror source: `services/stt.py` → `tests/services/test_stt.py`
- **Frontend:** `vitest` adjacent to components
- **Every new module:** golden-path test + one edge case minimum
- **Pipeline integration test:** must test the full STT → LLM → TTS path with a fixture audio file

## Commits

- Conventional-commit: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `data:`, `model:`
- One logical change per commit
- Message explains *why*, not *what*

## Pre-Push Checklist

Run `scripts/pre_push_check.sh` before every push. It verifies:
1. `mypy backend/` — type checker passes
2. `ruff check backend/` — linter clean
3. `pytest backend/tests/` — all tests pass
4. No secrets in diff (`git diff HEAD` scan)
5. `CODE_STRUCTURE.md` is up to date (last modified within current session)
6. `PROJECT.md` `Last Updated` date matches today

If any check fails: fix it, don't skip it.

## Correction Loop (Bug Non-Recurrence Protocol)

When a bug is found:
1. **Log it immediately** in `INSTRUCTIONS.md` under `## Bug & Correction Log`
2. Fix the bug
3. **Update the log** with the root cause and fix
4. **Add a regression test** that would have caught it
5. **Add a rule** to this CONVENTIONS.md or INSTRUCTIONS.md to prevent recurrence

A bug that happens twice is a process failure, not a code failure.

## Project-Specific Conventions

- Timezone: Asia/Dubai
- Audio sample rate in: 16000 Hz (Whisper requirement)
- Audio sample rate out: 22050 Hz (Fish Speech 1.5 default)
- VAD silence threshold: 300ms before utterance is considered complete
- Max LLM response: 150 tokens (latency constraint — enforced in prompt_builder)
- Language detection confidence threshold: 0.7 (below → ask user to repeat)
- Model calls never on main thread — use `asyncio.run_in_executor` for blocking calls
- Primary latency metric: **pipeline_ttfa_ms** (time from utterance end to first audio chunk)
- All latency events written to `data/metrics.db` via `@track_latency` decorator
- `PERFORMANCE.md` is auto-generated — never edit manually
- `pipeline_phase` values: `raw` | `prompted` | `lora_v1` | `lora_v2` | `jais_fallback`
