# WINDOWS_OPTIMIZATION.md — Hands-Free for RTX A3000

> Strategy to adapt Hands-Free for Windows 11 + Intel i7-11850H + RTX A3000 (4GB VRAM) + 32GB system RAM
> Focus: Research optimization via quantization, pruning, and clever inference patterns

**Target:** Prove bilingual voice pipeline < 2s latency on constrained hardware

---

## Hardware Baseline

| Resource | Available | Constraint |
|----------|-----------|-----------|
| System RAM | 32 GB | ✅ Abundant |
| GPU VRAM | 4 GB | ❌ Tight (need 5.1 GB for Gemma E2B full) |
| CPU | i7-11850H (8c/16t, 2.5 GHz) | ⚠️ Can run models, but slow |
| GPU | RTX A3000 (CUDA-capable) | ✅ Good for inference (not training) |

---

## Module-by-Module Strategy

### 1. VAD (Voice Activity Detection) — Silero VAD

**Current:** `audio_capture.py` uses Silero VAD with `sounddevice`

**Status:** ✅ **No changes needed**
- Silero VAD: ~5 MB, runs entirely on CPU
- `sounddevice`: Windows-native audio API supported
- No GPU requirement

**Optimization:**
- Keep as-is; it's the lightest component

---

### 2. STT (Speech-to-Text) — Whisper

**Current Project:** mlx-whisper (Mac/Metal-only)

**Problem:** MLX framework is Mac-exclusive; need Windows alternative

#### Option A: Faster-Whisper + CUDA (Recommended)

```yaml
Library: faster-whisper (CTranslate2 backend)
Backend: CUDA (RTX A3000 supported)
Model: whisper-small-int8 or tiny (quantized)
Estimated VRAM: 500 MB – 1 GB
Latency: 150–500ms (small-int8 on RTX A3000)
```

**Setup:**
```bash
pip install faster-whisper
# CTranslate2 auto-detects CUDA; no extra config needed
```

**Code change** (`backend/services/stt.py`):
```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="int8")
segments, info = model.transcribe(audio_np, language="en")
```

#### Option B: Whisper.cpp + CUDA (Lower VRAM)

```yaml
Library: whisper.cpp (C++ runtime)
Backend: CUDA
Model: ggml-tiny.bin or ggml-small.bin (GGML quantized)
Estimated VRAM: 150–400 MB
Latency: 300–800ms (depends on model)
Pros: Very low VRAM; can run on CPU fallback
Cons: Python bindings less mature; requires C++ compilation
```

**Setup:**
```bash
pip install whisper-cpp-python
# or build from: https://github.com/stlutz/whisper-cpp-python
```

#### Option C: OpenAI Whisper (CUDA via PyTorch)

```yaml
Library: openai-whisper
Backend: CUDA via PyTorch
Model: base, small, tiny (auto-quantized)
Estimated VRAM: 1–2 GB
Latency: 500–1500ms
Pros: Simple API; official model
Cons: Slower than faster-whisper due to no CTranslate2 optimization
```

**Recommendation:** **Option A (faster-whisper + CUDA)** — best balance of speed and VRAM

---

### 3. LLM (Language Model) — Gemma 4 E2B

**Current:** Ollama serving gemma4:e2b (5.1 GB full precision)

**Problem:** 5.1 GB > 4 GB VRAM; need quantization

#### Gemma E2B Quantization Analysis

| Quantization | Size | VRAM | Latency | Quality Loss |
|---|---|---|---|---|
| FP16 (half precision) | 3.0 GB | 3.2 GB | ~80 ms/token | None |
| Q8 (8-bit) | 2.0 GB | 2.2 GB | ~100 ms/token | Minimal (<0.5% perplexity) |
| Q4 (4-bit) | 1.2 GB | 1.4 GB | ~150 ms/token | Moderate (~3–5% perplexity) |
| Q3 (3-bit) | 900 MB | 1.1 GB | ~200 ms/token | High (10%+ perplexity) |

#### Option A: Ollama + FP16 (Recommended)

**Setup:**
```bash
# Create Modelfile with precision control
cat > Modelfile <<EOF
FROM gemma:2b
PARAMETER num_ctx 2048
PARAMETER num_gpu 99  # Use all available GPU
EOF

ollama create gemma-2b-fp16 -f Modelfile
ollama run gemma-2b-fp16
```

**Problem:** Gemma 4 E2B doesn't have an official FP16 GGUF yet; only full precision (5.1 GB)

**Workaround:** Use **Gemma 2 (smaller)** instead
```bash
ollama pull gemma:2b
# Gemma 2B: 1.6 GB, fits in 4 GB VRAM with room for other models
```

#### Option B: Ollama + GPTQ Quantization (4-bit)

**Setup:**
```bash
# Download GPTQ-quantized Gemma from HuggingFace
# https://huggingface.co/TheBloke/Gemma-2B-Instruct-GPTQ
ollama create gemma-2b-gptq --from gemma:2b
```

**Code:**
```python
# Same Ollama API — no code change needed
client.chat.completions.create(
    model="gemma-2b-gptq",
    messages=messages,
)
```

#### Option C: Local Inference with `transformers` + bitsandbytes (8-bit)

**Setup:**
```bash
pip install transformers torch bitsandbytes
```

**Code** (replaces Ollama):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 8-bit quantization: loads in 1.6 GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto",
)

def generate(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])
```

**Latency:** ~150–200 ms/token (8-bit quantization on RTX A3000)

#### Option D: INT4 LoRA (Extreme Compression)

**Research approach:** Use QLoRA on top of quantized base model

```yaml
Base: Gemma 2B FP16 (1.6 GB)
LoRA: 4-bit quantized, rank=8 (50–100 MB)
Total VRAM: ~2.2 GB
Expected latency: 100–150 ms/token
Quality: Can match or exceed base with good training data
```

**Tools:** [`unsloth`](https://github.com/unslothai/unsloth) library (optimized LoRA training)

#### ⚠️ Question: Can Reasoning Be Offloaded?

**Your Question:** "reasoning is turned off can it be offloaded as well?"

**Answer:** 
- EXP-001 in the project disables `think=false` in Ollama → saves 20–32s reasoning tokens
- **Can reasoning run on CPU while inference runs on GPU?** Partially:
  - Ollama doesn't expose per-layer device placement
  - `transformers` library *can* split layers: some on GPU, some on CPU (slow, no latency benefit)
  - **Better approach:** Run two separate models:
    - Small thinking model on CPU (analyze utterance)
    - Small output model on GPU (generate response)
    - But this adds complexity with no clear latency win

**Recommendation:** Keep reasoning disabled; focus on inference speed instead.

---

### 4. TTS (Text-to-Speech) — Fish Speech 1.5

**Current:** Fish Speech 1.5 HTTP streaming (Python package or Docker)

**Problem:** Fish Speech is slow on MPS (8 tok/sec on Mac); worse on CUDA without proper optimization

#### Option A: Fish Speech + CUDA + Optimization (Current Stack)

**Setup:**
```bash
pip install fish-speech
python scripts/start.py --device cuda --half --compile
```

**Expected:**
- Baseline: 8 tok/sec on MPS → potentially 20–30 tok/sec on CUDA RTX A3000
- VRAM: 2–3 GB for inference
- Latency: 1–3s per sentence (depends on sentence length)

#### Option B: Coqui TTS (Lightweight Alternative)

```yaml
Library: Coqui TTS (VITS or Glow-TTS)
VRAM: 500 MB – 1 GB
Latency: 100–500 ms per sentence
Arabic support: Native (Arabic VITS models available)
Pros: Lightweight, fast, open-source
Cons: Quality < Fish Speech; no voice cloning (Phase 6)
```

**Setup:**
```bash
pip install TTS
python -m TTS.server --device cuda
```

**Code:**
```python
from TTS.api import TTS
model = TTS(model_name="tts_models/ar/cv/glow-tts", gpu=True)
wav = model.tts("مرحبا بك")
```

#### Option C: SILMA TTS (Bilingual Specialist)

```yaml
Library: SILMA TTS v1 (F5-TTS-based)
VRAM: 2–3 GB
Latency: 800–1500 ms per sentence
Arabic + English: Native support
Pros: Designed for bilingual; high quality
Cons: Not yet released / research stage
Status: Monitor HuggingFace for release
```

#### Option D: Hybrid (Local STT + Cloud TTS)

```yaml
Local: Whisper (STT) on GPU
Cloud: OpenAI / Google TTS (streaming audio chunks)
Latency: STT local, TTS remote (add network delay)
VRAM: Only for STT (~1 GB)
Pros: Best quality TTS; reduces local VRAM pressure
Cons: Privacy concern; adds network dependency
```

**Recommendation:** **Option A (Fish Speech + CUDA)** — same architecture, test CUDA performance gain first

---

## Integration Plan

### Phase 1: Proof of Concept (Week 1)

1. **STT:** Implement faster-whisper + CUDA (Option A)
   - Replace `backend/services/stt.py` imports
   - Test on fixture audio files
   - Benchmark latency

2. **LLM:** Try Gemma 2B (smaller than E2B) with Ollama
   - Create quantized Modelfile
   - Keep Ollama API — no code changes
   - Test `curl http://localhost:11434/api/chat`

3. **TTS:** Keep Fish Speech; test CUDA backend
   - `python scripts/start.py --device cuda`
   - Measure latency on sample text

4. **VAD:** No changes needed

### Phase 2: Full Pipeline Integration (Week 2)

1. Adapt `backend/services/` for Windows paths (`os.name == "nt"`)
2. Update `.env.example` with CUDA model names
3. Run full benchmark: `python scripts/benchmark.py --iterations 5`
4. Measure real TTFA (target: <2000ms on constrained hardware)

### Phase 3: Optimization (Week 3+)

1. **Profile** each stage with `cProfile` → identify bottlenecks
2. **Quantize further** if needed (Q8 → Q4 LLM)
3. **Batch optimization:** Can we pipeline STT → LLM → TTS with overlap?
4. **Memory pooling:** Reuse allocations across requests

---

## Feasibility Summary

| Component | Feasible? | Confidence | VRAM | Latency Est. |
|-----------|-----------|-----------|------|--------------|
| **VAD** | ✅ Yes | 100% | 5 MB | <50 ms |
| **STT (Whisper small)** | ✅ Yes | 95% | 1 GB | 150–500 ms |
| **LLM (Gemma 2B Q8)** | ✅ Yes | 80% | 2 GB | 150–200 ms/token |
| **TTS (Fish Speech)** | ⚠️ Slow | 70% | 2–3 GB | 2–5s / sentence |
| **Full Pipeline** | ✅ Possible | 75% | ~4–5 GB total | 2–8s round-trip |

**Conclusion:** You can build a working bilingual voice pipeline on RTX A3000, but **TTS latency will be the bottleneck**. Fish Speech may not be ideal; consider Coqui TTS as a faster alternative.

---

## Research Questions to Explore

1. **Batch inference:** Can Whisper + Gemma + Fish Speech be pipelined (STT running while LLM generates)?
2. **Quantization transfer:** Can a LoRA trained on full-precision Gemma E2B be applied to quantized Gemma 2B?
3. **Streaming override:** Can Fish Speech stream single tokens to TTS while LLM still decoding (reduce perceived latency)?
4. **Memory mapping:** Can models be loaded from disk on-demand (offload to NVMe between requests)?

---

## Next Actions

- [ ] Test faster-whisper CUDA + measure latency on Windows
- [ ] Create Ollama quantized Modelfile for Gemma 2B
- [ ] Benchmark TTS latency with `--device cuda` on Fish Speech
- [ ] Profile memory during full pipeline run
- [ ] Document findings in PERFORMANCE.md for Windows variant

