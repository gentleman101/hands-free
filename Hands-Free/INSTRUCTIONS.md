# INSTRUCTIONS — Hands-Free: Bilingual Voice-to-Voice AI

> **For Claude Code:** Read this file at the start of every session.
> This document is the master build guide + living correction log.
> When a bug is found: log it here immediately, even before fixing it.

---

## ⚠️ Mandatory Pre-Work Rule

Before installing, downloading, or using ANY model, library, API, or framework:

1. **Search the internet for the latest version and release notes**
2. **Check hardware compatibility** (Mac Mini, Apple Silicon, Metal)
3. **Check for breaking changes** since the last time you used it

This is not optional. AI tooling changes weekly. Outdated integration = wasted hours.

---

## Locked Stack (Final — Do Not Change Without PROJECT.md Change Log Entry)

| Layer | Technology | Notes |
|---|---|---|
| STT | faster-whisper | Silero VAD built-in; utterance-triggered, not streaming mid-sentence |
| LLM | Ollama → gemma4:e2b | Metal/MLX; OpenAI SDK stream=True; `ollama pull gemma4:e2b` |
| TTS | Fish Speech 1.5 | HTTP chunked streaming; DAC decoder only (no voice cloning codec) |
| Pipelining | Sentence-level | Each complete sentence from LLM → TTS immediately; don't wait for full response |
| Audio I/O | sounddevice | Capture 16kHz mono; playback on first TTS chunk |
| VAD | Silero (via faster-whisper) | 300ms silence → utterance complete → trigger STT |
| Backend | FastAPI | Async; SSE streaming endpoint; thread pool for blocking calls |
| Logging | Structured JSON + SQLite | Every stage timed; TTFA is primary metric |
| Benchmark | scripts/benchmark.py | Auto-regenerates PERFORMANCE.md |
| Distillation TTS | F5-TTS + F5-TTS-Arabic | Phase 4+ only; better prosody via voice cloning |

**Target TTFA (time-to-first-audio after you stop speaking): < 800ms**

---

## ⚠️ Pre-Push Gate (Non-Negotiable)

Run `scripts/pre_push_check.sh` before EVERY push. Do not bypass with `--no-verify`.

The script checks:
```
[ ] mypy backend/        — type checker passes
[ ] ruff check backend/  — linter clean
[ ] pytest backend/tests/ — all tests pass
[ ] No secrets in diff
[ ] CODE_STRUCTURE.md updated this session
[ ] PROJECT.md Last Updated = today
```

If any check fails: **fix it**. Do not push a failing state.

---

## Phase Overview

```
Phase 1 → Base Pipeline (STT + LLM + TTS)
Phase 2 → Behavior Control (prompts)
Phase 3 → Dataset Creation
Phase 4 → LoRA Fine-Tuning (distillation)
Phase 5 → Evaluation + Iteration
Phase 6 → Optional enhancements (RAG, voice cloning, streaming)
```

Fallback: If Gemma Arabic quality is insufficient → switch to **Jais** (Arabic-first model).

---

## Phase 1: Base Pipeline

### Streaming Architecture (5 Concurrent Layers)

```
Layer 1  sounddevice → 512ms PCM chunks → audio_queue
Layer 2  Silero VAD → detects utterance end (300ms silence) → stt_queue
Layer 3  faster-whisper.transcribe(utterance) → (text, language) → llm_queue   [~150-300ms]
Layer 4  Ollama stream=True → SentenceBuffer → emits complete sentences → tts_queue [TTFT ~200ms]
Layer 5  Fish Speech HTTP stream → audio chunks → sounddevice plays first chunk  [TTFA ~300ms]
```

### Task Sequence

1. Create Python venv + requirements.txt + .env.example
2. `ollama pull gemma4:e2b` — verify Ollama serves it
3. `backend/logging_config.py` — structured JSON
4. `backend/database.py` — SQLite schema (latency_events)
5. `backend/monitoring/latency_tracker.py` — `@track_latency` decorator
6. `backend/services/stt.py` — utterance bytes → TranscribeResult
7. `backend/services/llm.py` — generate() + generate_stream()
8. `backend/services/tts.py` — Fish Speech synthesize() + synthesize_stream()
9. `backend/services/prompt_builder.py` — bilingual system prompt
10. `backend/services/audio_capture.py` — VAD + mic buffering → utterances()
11. `backend/services/pipeline.py` — SentenceBuffer + run() + run_realtime()
12. `backend/api/voice.py` + `health.py` + `backend/app.py`
13. `fixtures/audio/` — 4 WAV files (en/ar × greeting/hr_query)
14. `scripts/benchmark.py` + `scripts/generate_report.py` → first `PERFORMANCE.md`
15. `DISTILLATION.md`

### Latency Targets

| Stage | Target | Metric |
|---|---|---|
| STT | < 300ms | duration_ms |
| LLM time-to-first-token | < 300ms | ttft_ms |
| TTS time-to-first-audio | < 400ms | ttfa_ms |
| **Pipeline TTFA** | **< 800ms** | **pipeline_ttfa_ms** |
| Total round-trip | < 2000ms | total_ms |

### Audio Spec

- Input: 16000 Hz, mono, 16-bit PCM (Whisper native)
- Output: 22050 Hz (Fish Speech 1.5 default), playback via sounddevice
- VAD silence threshold: 300ms (tune if needed)

---

## Phase 2: Behavior Control

### System Prompt Template

```
You are a helpful bilingual voice assistant. Respond ONLY in the language the user speaks.
If the user speaks Arabic, respond in Arabic. If English, respond in English.
Keep responses short and conversational — maximum 2–3 sentences.
Do not explain what you are doing. Just answer.
```

### Bilingual Rules

- Language detection confidence threshold: 0.7
- If confidence < 0.7: ask user to repeat
- Never mix languages in a single response
- Gulf/UAE Arabic dialect preferred when responding in Arabic

---

## Phase 3: Dataset Creation

### Format

```json
{
  "instruction": "Task description for the assistant",
  "input": "User query (Arabic or English)",
  "output": "Ideal assistant response"
}
```

### Dataset Targets

| Stage | Size |
|---|---|
| POC | 500–1,000 samples |
| Production | 3,000–10,000 samples |

### Quality Rules

- Max response length: 3 sentences
- Must include follow-up questions where appropriate
- Must include refusal/guardrail cases
- Remove robotic phrasing
- Remove inconsistent tone

### Teacher Model

Use OpenAI API for generating ideal responses. This is a one-time cost, not runtime.
Store API key in `.env` as `HF_OPENAI_API_KEY`. Never commit to git.

---

## Phase 4: LoRA Fine-Tuning

### Config Constraints

- Max 1–2 epochs (avoid overfit)
- QLoRA (4-bit quantization) for Mac memory constraints
- Target modules: `q_proj`, `v_proj`
- Rank: 8–16
- Learning rate: 2e-4

### Output

- LoRA adapters: `models/lora/hands-free-v1/`
- Merged + quantized: rebuild as GGUF for inference

---

## Phase 5: Evaluation Loop

### Failure Capture Protocol

1. Run real conversations, note failures
2. For each failure: log in "Bug & Correction Log" below
3. Generate correct response using teacher model
4. Add to dataset
5. Retrain (1 epoch, validate loss improvement)
6. Retest the specific failure case

---

## Phase 6: Optional Enhancements

- RAG: inject knowledge base at inference time (LlamaIndex or Haystack — check latest)
- Voice cloning: Fish Speech supports it — evaluate after Phase 1
- Dialect handling: add Gulf dialect examples to dataset in Phase 3
- Streaming: FastAPI StreamingResponse for TTS chunks

---

## Fallback Strategy

If Gemma Arabic quality is insufficient after Phase 2:

**Switch to Jais**
- Arabic-first training, strong Gulf dialect understanding
- Download from HuggingFace: search for latest Jais release before downloading
- Drop-in replacement for `backend/services/llm.py`

Trigger conditions for fallback:
- Arabic comprehension accuracy < 70% on test set
- Consistent bilingual mixing in responses
- Weak contextual handling in Arabic

---

## Stack Versions (Check Before Use)

> Last checked: 2026-04-24. Always reverify before installing.

| Library | Purpose | Source to check |
|---|---|---|
| faster-whisper | STT + Silero VAD | GitHub: SYSTRAN/faster-whisper |
| ollama (Python) | LLM client | PyPI: ollama |
| openai | LLM streaming client | PyPI: openai (used against Ollama base_url) |
| fish-speech | TTS | GitHub: fishaudio/fish-speech |
| sounddevice | Audio I/O | PyPI: sounddevice |
| FastAPI | Backend | PyPI: fastapi |
| pydantic | Validation | PyPI: pydantic |
| torch | Training (Phase 4) | pytorch.org — check Metal/MPS support version |
| transformers | Training (Phase 4) | Hugging Face — check latest |
| peft | LoRA (Phase 4) | Hugging Face: peft |

**Fish Speech note:** Uses DAC (Descript Audio Codec) internally — bundled, no separate install.
For TTS without voice cloning: only decoder runs (tokens → audio). Encoder (cloning) is Phase 6.
Fish Speech requires CUDA for real-time on production. On Mac Mini with MPS: ~15-25s/sentence.
POC accepts this latency — PERFORMANCE.md will show real numbers. Azure deploy = real-time.

---

## Bug & Correction Log

> **Protocol:** When a bug is discovered, add a row immediately (before fixing).
> Update with resolution after fixing. Add a regression test. Add a prevention rule.
> **A bug that recurs is a process failure.**

| ID | Date | Phase | Description | Root Cause | Fix | Regression Test | Prevention Rule | Status |
|---|---|---|---|---|---|---|---|---|
| B001 | 2026-04-24 | 1 | `latency_events` table not found when benchmark.py runs standalone | `init_db()` is only called via FastAPI lifespan; benchmark.py skips that path | Call `await init_db()` at the top of `run_benchmark()` in benchmark.py | `test_benchmark_inits_db` (added) | Never call `insert_event` before `init_db()` is awaited; scripts that bypass FastAPI must call it explicitly | [Resolved] |

### Bug Log Instructions (for Claude Code)

When you find a bug:
1. **Immediately** add a row with ID, date, phase, description — leave root cause/fix blank
2. Investigate and fix
3. Update root cause and fix columns
4. Write a test that would have caught it
5. Add a prevention rule to `CONVENTIONS.md` or this section
6. Mark status `[Resolved]`

Never mark a bug resolved without a regression test.
Never close this log — historical record is permanent.

---

## LLM Prosody Instructions for Fish Speech

The system prompt must teach Gemma to write text that Fish Speech renders naturally.

**Fish Speech inline tags (embed in response text):**
```
[laugh]      — natural laugh at that position
[whisper]    — lower to whisper
[emphasis]   — stress the following word
[sigh]       — audible sigh before sentence
[excited]    — high energy delivery
[angry]      — assertive/urgent tone
```

**Punctuation as prosody:**
```
,     — short breath pause
...   — thinking pause (longer)
—     — abrupt mid-thought cut
?     — rising intonation
!     — energy increase
،     — Arabic comma pause
؟     — Arabic question mark (rising)
```

**System prompt addition (append to all prompts):**
```
Format responses for natural spoken delivery via text-to-speech.
Use [laugh], [emphasis], or [sigh] where a real speaker would naturally use them.
Use commas for breath pauses and ... for a thinking pause.
Keep responses to 1-2 sentences maximum. Never write lists or bullet points.
Use contractions (it's, you're, that's) — never formal full forms in speech.
Never start with "Certainly", "Of course", or "Great question".
Respond in the exact same language the user spoke.
```

---

## Development Rules (Derived from Corrections)

> This section grows as bugs are resolved. Each rule here prevented at least one real bug.

**Rule 1 (from B001):** Any script or CLI tool that uses `insert_event` must call `await init_db()` explicitly before its first database write. Do not assume FastAPI lifespan has run. Add `await init_db()` as the first line of any standalone async `main()` function.

---

## Architecture Reference

See `CODE_STRUCTURE.md` for the always-current file map.
See `PROJECT.md` for decisions, risks, and status.
See `CONVENTIONS.md` for naming, size caps, and commit rules.
