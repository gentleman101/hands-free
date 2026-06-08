# INSTRUCTIONS — Hands-Free: Bilingual Voice-to-Voice AI

> Master setup guide + living correction log.
> Read this at the start of every session.
> When a bug is found: log it immediately, even before fixing it.
>
> Last Updated: 2026-05-08

---

## Architecture

```
Windows (RTX A3000, CUDA)              Mac Mini (M4, Metal)
─────────────────────────              ────────────────────
sounddevice (mic in)                   llm_server.py
→ TEN VAD                              → llama.cpp Metal
→ stt.py (faster-whisper CUDA)         → JAIS-2 8B Q4
→ intent_classifier.py                 → streaming NDJSON
→ filler.py (from memory) ─────────────────────────────────┐
→ llm_client.py ──────────→ HTTP RPC ──────────────────────┤
→ pipeline.py / SentenceBuffer                             │
→ tts.py → Fish Speech (WSL2 CUDA)                        │
→ sounddevice (speaker out) ◄──────────────────────────────┘

Fish Speech lives in WSL2 (Linux, torch.compile CUDA enabled).
Windows accesses it at http://172.29.211.12:8001.
```

---

## Quick Start

### Windows (client)

```powershell
# 1. Clone and enter
git clone <repo> && cd Hands-Free

# 2. Python 3.11 venv
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install
pip install -r requirements-shared.txt -r requirements-windows.txt

# 4. Configure
copy .env.example .env
# Edit .env: set HF_MACHINE_ROLE=client, HF_LLM_SERVICE_URL=http://<mac-ip>:8080

# 5. Start Fish Speech in WSL2 (separate terminal)
wsl
cd ~/fish-speech-wsl/fish-speech
source ~/fish-speech-wsl/venv/bin/activate
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/fish-speech-1.5 \
  --decoder-checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
  --listen 0.0.0.0:8001 --device cuda --compile

# 6. Run
python scripts/start.py
```

### Mac Mini (server)

```bash
# 1. Clone and enter
git clone <repo> && cd Hands-Free

# 2. Python 3.11
brew install python@3.11
python3.11 -m venv venv && source venv/bin/activate

# 3. Install
pip install -r requirements-shared.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# 4. Download model
mkdir -p ~/models
# Download JAIS-2 8B Q4 GGUF from HuggingFace → ~/models/jais-2-8b-q4.gguf
# Check: https://huggingface.co/core42 for latest JAIS GGUF
# Fallback: Qwen2.5-7B-Instruct-Q4_K_M.gguf (strong Arabic support, widely available)

# 5. Configure
cp .env.example .env
# Edit .env:
# HF_MACHINE_ROLE=server
# HF_LLM_MODEL_PATH=/Users/naman/models/jais-2-8b-q4.gguf
# HF_LLM_MODEL_NAME=jais-2-8b-q4
# HF_LLM_PORT=8080
# HF_LLM_N_GPU_LAYERS=99
# HF_LLM_N_CTX=2048

# 6. Run
python scripts/start_llm_server.py

# 7. Verify from Windows
curl http://<mac-ip>:8080/healthz
```

---

## Fish Speech WSL2 Setup (one-time)

```bash
# In WSL2 terminal
sudo apt update && sudo apt install -y python3.11 python3.11-venv portaudio19-dev ffmpeg

mkdir -p ~/fish-speech-wsl && cd ~/fish-speech-wsl
git clone https://github.com/fishaudio/fish-speech fish-speech
cd fish-speech && git checkout v1.5.1

python3.11 -m venv ~/fish-speech-wsl/venv
source ~/fish-speech-wsl/venv/bin/activate

pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[stable]"

# Download checkpoints
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5

# Test
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/fish-speech-1.5 \
  --decoder-checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
  --listen 0.0.0.0:8001 --device cuda --compile
```

### WSL2 IP
Fish Speech binds to `0.0.0.0:8001` in WSL2. Find the IP:
```bash
wsl hostname -I   # e.g. 172.29.211.12
```
Set `HF_FISH_SPEECH_URL=http://172.29.211.12:8001` in `.env`.

---

## Regenerating Assets (one-time scripts)

```powershell
# Generate reference voice WAVs via Edge TTS (only if voices/ is missing)
venv\Scripts\python scripts/generate_reference_voices.py

# Generate all 84 filler WAVs (Fish Speech must be running)
venv\Scripts\python scripts/generate_fillers.py

# Benchmark TTS
venv\Scripts\python scripts/prosody_bench.py
```

---

## Running Tests

```powershell
venv\Scripts\python -m pytest backend/tests/ -q
# Expected: 35 passed
```

---

## Phase Overview

```
Phase 1 → Windows Pipeline (COMPLETE)
Phase 2 → Mac Mini LLM Setup (IN PROGRESS)
Phase 3 → Behavior Control (prompts, DEWA vocab)
Phase 4 → Dataset Creation (DEWA domain)
Phase 5 → LoRA Fine-Tuning
Phase 6 → Evaluation + Iteration
```

---

## Latency Targets

| Stage | Target |
|---|---|
| STT | < 300ms |
| LLM TTFT | < 100ms |
| TTS first chunk | < 500ms (improving) |
| **Pipeline TTFA (real)** | **< 600ms** |
| **Pipeline TTFA (perceived)** | **~0ms** (filler hides it) |

---

## Stack Versions (Last checked: 2026-05-08)

| Library | Purpose | Notes |
|---|---|---|
| `faster-whisper` | STT | `large-v3-turbo` model; check SYSTRAN/faster-whisper for updates |
| `fish-speech` v1.5.1 | TTS | WSL2 only; pinned at v1.5.1 tag |
| `llama-cpp-python` | LLM (Mac) | Build with Metal: `CMAKE_ARGS="-DGGML_METAL=on"` |
| `sounddevice` | Audio I/O | Windows native; no driver install needed |
| `ten-vad` | VAD | `pip install ten-vad`; better than Silero for turn-taking |
| `scikit-learn` | Intent classifier | TF-IDF vectorizer |
| `ormsgpack` | Fish Speech VQGAN | Used only in `tts.warmup()` |

---

## LLM Prosody Instructions for Fish Speech

Embed these tags in LLM responses for natural TTS delivery:

```
[laugh]      — natural laugh
[whisper]    — lower to whisper
[emphasis]   — stress the following word
[sigh]       — audible sigh
[excited]    — high energy
```

Punctuation as prosody: `,` breath pause, `...` thinking pause, `?` rising intonation.

System prompts already include Fish Speech prosody rules — see `prompt_builder.py`.

---

## Bug & Correction Log

| ID | Date | Phase | Description | Root Cause | Fix | Regression Test | Status |
|---|---|---|---|---|---|---|---|
| B001 | 2026-04-24 | 1 | `latency_events` table not found in standalone scripts | `init_db()` not called outside FastAPI lifespan | Call `await init_db()` in standalone async `main()` | `test_benchmark_inits_db` | [Resolved] |
| B002 | 2026-05-08 | 1 | `scripts/start.py` imported `FillerPlayer` class which no longer exists | `filler.py` was refactored to `stream_filler()` generator; `start.py` not updated | Removed `FillerPlayer` import; pipeline handles fillers internally | Covered by pipeline tests | [Resolved] |

---

## Development Rules

**Rule 1 (B001):** Any standalone script using `insert_event` must call `await init_db()` before first DB write.

**Rule 2 (B002):** When refactoring a service interface, grep for all callers before committing.

**Rule 3:** Never mark a bug resolved without a regression test.

**Rule 4:** `pytest backend/tests/ -q` must be green before every push.
