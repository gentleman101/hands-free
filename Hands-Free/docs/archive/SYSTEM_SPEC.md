# SYSTEM_SPEC.md — Hands-Free / Rammas

> Exhaustive system specification, performance contract, concurrency model,
> and systematic preflight check.
>
> Last Updated: 2026-05-13

---

## 1. System Topology

```
┌──────────────────────────────────────┐        ┌──────────────────────────┐
│ Windows laptop (RTX A3000, 6GB VRAM) │        │ Mac Mini (M4, 16GB)      │
│                                      │        │                          │
│  ┌────────┐  ┌──────┐  ┌──────────┐  │        │  ┌────────────────────┐  │
│  │  mic   │→ │ VAD  │→ │   STT    │  │        │  │  Ollama daemon     │  │
│  │ 16kHz  │  │ TEN  │  │ Whisper  │  │        │  │  :11434            │  │
│  └────────┘  └──────┘  └──────────┘  │        │  └─────────┬──────────┘  │
│                            │         │  HTTP  │            │             │
│  ┌──────────┐  ┌──────────┐│         │  RPC   │  ┌─────────┴──────────┐  │
│  │ Intent   │← │ Filler   ││         │ ◄──────┼──┤ FastAPI bridge     │  │
│  │ TF-IDF   │  │ 84 WAVs  ││         │        │  │ :8080              │  │
│  └────┬─────┘  └────┬─────┘│         │        │  │  /api/llm/generate │  │
│       │             │      ▼         │        │  └────────────────────┘  │
│       │        ┌────┴──────────┐     │        └──────────────────────────┘
│       │        │ SentenceBuffer│     │                ▲
│       └───────►│  Pipeline     │─────┼────────────────┘
│                └────┬──────────┘     │
│                     ▼                │
│       ┌────────────────────┐         │     ┌──────────────────────────┐
│       │ TTS                │─────────┼────►│ Fish Speech (WSL2)       │
│       │ orchestrator       │  HTTP   │     │ :8001  CUDA + compile    │
│       └────────┬───────────┘         │     └──────────────────────────┘
│                ▼                     │
│       ┌────────────────────┐         │
│       │ speaker (44.1kHz)  │         │
│       └────────────────────┘         │
└──────────────────────────────────────┘
```

---

## 2. End-to-End Specification Table

### 2.1 Audio Capture

| Param | Value | Notes |
|---|---|---|
| Sample rate (capture) | 16,000 Hz | Whisper-native |
| Channels | 1 (mono) | |
| Block size | 512 samples (32ms) | Matches Silero default |
| Dtype | float32 | |
| Max utterance | 30 s | Hard cap |
| Device priority | BT/USB → built-in → first | Skips virtual/HDMI |
| Device fallback | Skip if unopenable | Tries next candidate |

### 2.2 Voice Activity Detection

| Param | Value | Notes |
|---|---|---|
| Backend (primary) | TEN VAD | Faster, fewer false fires |
| Backend (fallback) | Silero VAD | Auto if TEN unavailable |
| Backend (last resort) | Energy RMS | Threshold 0.02 |
| Speech threshold | 0.80 | EXP-011: 0.70 fires on HVAC |
| Silence to end | 600 ms | LiveKit-style turn-taking |
| Min speech | 8 chunks (~256ms) | Filter cough/click |
| Pre-roll | 4 chunks (~128ms) | First-word capture |

### 2.3 STT (faster-whisper)

| Param | Value | Notes |
|---|---|---|
| Model | large-v3-turbo | 6× speed of large-v3 |
| Device | CUDA | RTX A3000 |
| Compute type | INT8 | ~1GB VRAM |
| Beam size | 1 (greedy) | EXP-009: 80-200ms faster |
| Internal VAD filter | OFF | EXP-008: capture already VADs |
| Lang detection threshold | 0.30 | Below → "unknown" |
| Allowed languages | EN, AR only | EXP-002 |
| `no_speech_prob` gate | <0.30 | Discard non-speech |
| `avg_logprob` gate | >−1.0 | Discard low confidence |
| `compression_ratio` gate | <2.4 | Discard hallucinations |
| Script validation (AR) | ≥30% Arabic chars | Catches misclassification |
| Script validation (EN) | ≥50% Latin chars | Catches misclassification |

### 2.4 Intent Classifier

| Param | Value | Notes |
|---|---|---|
| Algorithm | TF-IDF char n-gram (2,4) | scikit-learn |
| Similarity | Cosine | |
| Threshold | 0.15 | Below → "general" |
| Intents | 7 | billing, location, outage, account, service_request, sustainability, general |
| Bilingual | Yes | Same model EN+AR |
| Inference | 0.46 ms | CPU |

### 2.5 Filler Audio

| Param | Value | Notes |
|---|---|---|
| Total WAVs | 84 | 7 × 2 × 6 |
| Variants per intent | 6 | 3 formal + 3 casual |
| Selection | Random | Per turn |
| Source | Voice-cloned (Fish Speech) | Same voice as TTS |
| Sample rate | 44.1 kHz | |
| Load timing | First call (lazy) | Cached in memory |
| Stitching to LLM | YES | `pick_filler()` → text → `prefix=true` |

### 2.6 LLM (Server-side: Mac Mini)

| Param | Value | Notes |
|---|---|---|
| Backend | Ollama | Native `/api/chat` |
| Model | gemma4:e4b | MoE, ~1B active params |
| Quantization | Q8_0 | |
| Port (Ollama) | 11434 | Internal |
| Port (bridge) | 8080 | Exposed to client |
| `think` | False | Suppresses reasoning |
| `temperature` | 0.7 | |
| `num_predict` (max tokens) | 200 | EXP-007 |
| `n_ctx` | 2048 | |
| `n_gpu_layers` | 99 | All layers on Metal |
| Filler prefix injection | YES | `messages[-1]={role:"assistant", prefix:True}` |

### 2.7 LLM Client (Windows)

| Param | Value | Notes |
|---|---|---|
| Transport | HTTPX async | Connection pool |
| Max connections | 10 | |
| Keepalive | 5 | TCP reuse |
| Connect timeout | 2 s | |
| Total timeout | 120 s | Bumped for streaming |
| Retries | 3 | 0.5s backoff |
| Health check | `/healthz` GET | Pre-flight on startup |

### 2.8 TTS (Fish Speech, WSL2)

| Param | Value | Notes |
|---|---|---|
| Engine | Fish Speech 1.5 | |
| Backend | CUDA + torch.compile | WSL2 only |
| Speed | ~65 tok/sec | 5× Windows-native |
| TTFA (warm) | ~1.9 s | First chunk |
| Sample rate | 44,100 Hz | firefly-gan-vq output |
| Chunk length | 200 tokens | Smaller = lower TTFA |
| Streaming | True | Raw PCM (no per-chunk WAV header) |
| Voice cloning | YES | brian_en.wav / hamdan_ar.wav |
| Memory cache | "on" | Server-side reuse |
| VQGAN warmup | YES | `tts.warmup()` at startup |

### 2.9 SentenceBuffer

| Param | Value | Notes |
|---|---|---|
| Min sentence chars | 8 | EXP-005 |
| Boundary chars | `.?!\n,،؟！,` | EN + AR punctuation |
| First-clause trigger | Comma | Saves 200-500ms TTFA |

### 2.10 Pipeline TTFA Budget (target)

| Stage | Real (ms) | Hidden by filler? |
|---|---|---|
| VAD end-of-speech | 0 | — |
| STT | 300-500 | YES |
| Intent classify | 0.5 | YES |
| Filler emit (memory) | <10 | This IS the filler |
| LLM TTFT | 300-800 | YES |
| First sentence ready | 800-1500 | YES |
| TTS TTFA | 1800-2200 | YES |
| **Total real pipeline** | **~3.5-4.0 s** | |
| **Perceived TTFA** | **~10 ms** | Filler plays instantly |

---

## 3. Concurrency Model

### 3.1 Why Threading > Multiprocessing Here

**Multiprocessing is NOT useful** for this system because:
- LLM is on a remote machine (HTTP I/O bound)
- TTS is on a remote machine (HTTP I/O bound, WSL2 sidecar)
- VAD/Intent are sub-millisecond — no parallelism benefit
- Pickling overhead between processes would be larger than the savings
- The model objects (Whisper, VAD) don't survive `fork` cleanly on Windows

**Threading IS useful in three places**:
1. **Audio capture callback** — sounddevice runs its own audio thread; we read from a thread-safe queue
2. **STT inference** — `faster-whisper` releases GIL (CTranslate2 is C++); already done via `run_in_executor`
3. **VAD inference** — also C++/native; runs inside the asyncio loop (fast enough)

**asyncio handles everything else** since the remaining bottlenecks are HTTP I/O.

### 3.2 Punctuation-Triggered Concurrency

> **Yes** — when SentenceBuffer detects a boundary, that sentence can be sent to TTS
> immediately while the LLM continues generating the next one. Sentences are
> processed *concurrently* but played *in order*.

**Current behavior (sequential):**
```
LLM: tok1 tok2 tok3 . (sentence 1 done)
                        TTS sentence 1 → wait → audio chunks
                                                            LLM: tok4 tok5 tok6 .
                                                                                   TTS sentence 2 → ...
```

**Improved behavior (concurrent fanout, ordered playback):**
```
LLM: tok1 tok2 tok3 . (boundary)
                        ├─ spawn TTS-task-1 (background)
                        └─ continue receiving LLM tokens
                                                LLM: tok4 tok5 .
                                                                  ├─ spawn TTS-task-2
                                                                  └─ continue
playback: drain task-1 chunks → drain task-2 chunks → ...
```

Implementation: `asyncio.Queue` per sentence + ordered consumer.
Win: 200-500 ms saved per *additional* sentence (current pipeline only matters
for replies that span 2+ sentences, which is rare given the brevity prompt
but still worth it).

### 3.3 Threading Map

| Component | Mechanism | Notes |
|---|---|---|
| Audio callback | Native thread (PortAudio) | Queue-handoff to asyncio |
| VAD inference | asyncio loop (sync call) | <1ms per chunk |
| STT inference | `run_in_executor` (default ThreadPool) | Releases GIL |
| LLM stream | asyncio (httpx) | I/O bound |
| TTS stream | asyncio (httpx) | I/O bound |
| Sentence TTS fanout (NEW) | `asyncio.create_task` per boundary | Parallel HTTP |
| Audio playback | asyncio + sd.wait in executor | |

---

## 4. Systematic System Check (Preflight)

A new script `scripts/system_check.py` will validate everything before `start.py` is launched.

| # | Check | Pass criterion | Failure mode |
|---|---|---|---|
| 1 | Python version | 3.11.x | exit 1 |
| 2 | venv active | sys.prefix != base | warn |
| 3 | `.env` exists | file present | warn (use defaults) |
| 4 | HF_MACHINE_ROLE | "client" or "server" | exit 1 |
| 5 | Required env vars | all set | exit 1 |
| 6 | Required packages | imports succeed | print install hint |
| 7 | Whisper model cache | downloaded | offer to preload |
| 8 | Input devices listed | ≥1 openable | exit 1 |
| 9 | Fish Speech URL reachable | GET /v1/health 200 | exit 2 |
| 10 | Mac LLM URL reachable | GET /healthz 200 | exit 2 |
| 11 | Mac LLM model loaded | GET /readyz returns model | exit 2 |
| 12 | Filler WAVs present | 84 files | warn |
| 13 | Reference voices present | brian_en, hamdan_ar | warn |
| 14 | CUDA available (Windows) | torch / CTranslate2 sees GPU | warn |
| 15 | Sample LLM call | <5 s round-trip | warn |
| 16 | Sample TTS call (TTFA) | <3 s | warn |

Exit codes:
- 0: All green
- 1: Configuration error (fix and rerun)
- 2: Service unreachable (start dependency)
- 3: Soft warnings only

---

## 5. Code Changes (this commit)

### 5.1 Fix
- `backend/config.py` — defaults to `mac-mini.local:11434` (wrong port). Change to env-required with no misleading default.
- `backend/services/pipeline.py` — add sentence-parallel TTS via task queue.

### 5.2 Add
- `scripts/system_check.py` — preflight (16 checks above).
- `backend/services/tts_orchestrator.py` — manages sentence-parallel TTS with ordered playback.

### 5.3 Remove
- None.

---
