# DISTILLATION.md — Hands-Free

> Tracks dataset inventory, training runs, evaluation results, and the failure capture loop.
> Updated after every training run, eval session, or dataset change.
> Never edit `pipeline_phase` values without updating INSTRUCTIONS.md and CONVENTIONS.md too.

---

## Pipeline Phases

| Phase ID | Description | Model | Dataset | Status |
|---|---|---|---|---|
| `raw` | Off-the-shelf Gemma 4 E2B, no tuning | `gemma4:e2b` | — | **[Active — Phase 1]** |
| `prompted` | Bilingual system prompt + Fish Speech prosody instructions | `gemma4:e2b` | — | [Pending — Phase 2] |
| `lora_v1` | LoRA fine-tuned on 500-sample POC dataset | `gemma4:e2b` + LoRA | `v1.jsonl` | [Pending — Phase 4] |
| `lora_v2` | LoRA fine-tuned on 3,000-sample full dataset | `gemma4:e2b` + LoRA | `v2.jsonl` | [Pending — Phase 4] |
| `jais_fallback` | Arabic-first model, activated if Arabic quality < 70% | `jais` (HuggingFace) | — | [Fallback] |

To activate a phase: set `HF_PIPELINE_PHASE=<phase_id>` in `.env`.

---

## Dataset Inventory

### v1 — POC Dataset (target: 500–1,000 samples)

| Field | Value |
|---|---|
| Status | [Not started] |
| Target size | 500–1,000 samples |
| Languages | English + Arabic (Gulf/UAE dialect) |
| Domains | HR queries, leave policy, task workflows, conversational greetings |
| Format | JSONL — `{"instruction": ..., "input": ..., "output": ...}` |
| Output path | `datasets/processed/v1.jsonl` (gitignored) |
| Teacher model | OpenAI API (`HF_OPENAI_API_KEY` in `.env`) |

### v2 — Production Dataset (target: 3,000–10,000 samples)

| Field | Value |
|---|---|
| Status | [Not started] |
| Target size | 3,000–10,000 samples |
| Planned after | Phase 5 evaluation loop identifies failure cases |
| Output path | `datasets/processed/v2.jsonl` (gitignored) |

---

## Dataset Quality Rules

1. **Max response length:** 3 sentences (voice assistant — brevity is non-negotiable)
2. **No lists or bullet points** — voice output cannot render them
3. **Include follow-up questions** where appropriate to keep conversation natural
4. **Include refusal/guardrail cases** — at least 10% of dataset
5. **No robotic phrasing** — remove "Certainly", "Of course", "Great question"
6. **Consistent tone** — conversational, not formal report writing
7. **Dialect requirement** — Arabic responses use Gulf/UAE dialect, not MSA where possible
8. **Fish Speech prosody tags** — include `[laugh]`, `[emphasis]`, `[sigh]` in ~20% of samples where natural

---

## Training Configuration

### LoRA Hyperparameters (locked)

```yaml
method: QLoRA
quantization: 4-bit
target_modules: [q_proj, v_proj]
rank: 8-16
learning_rate: 2e-4
epochs: 1-2
batch_size: 4  # Mac Mini memory constraint
```

### Hardware Constraints (Mac Mini)

- Max batch size: 4 (16GB RAM, Apple Silicon)
- Use `device_map="mps"` for MPS acceleration
- Monitor temperature — Metal accelerated training runs hot
- `torch.mps.empty_cache()` between epochs

### Output Paths

```
models/lora/hands-free-v1/     ← LoRA adapters (v1 training)
models/lora/hands-free-v2/     ← LoRA adapters (v2 training)
models/merged/hands-free-v1/   ← Merged base + LoRA (before GGUF conversion)
```

---

## Training Runs

> Add a row after each training run. Never delete historical runs.

| Run ID | Date | Phase | Dataset | Epochs | Train Loss | Val Loss | TTFA p50 | Notes | Status |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | No runs yet | — |

---

## Evaluation Results

### Bilingual Quality Thresholds

| Metric | Target | Jais Fallback Trigger |
|---|---|---|
| Arabic comprehension accuracy | > 70% | < 70% |
| Consistent language response | > 95% | < 85% |
| No language mixing in response | > 99% | < 90% |

### Evaluation History

> Add a row after each eval session.

| Date | Phase | EN Accuracy | AR Accuracy | Language Mixing | TTFA p50 | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | No evals yet |

---

## Failure Capture Log

> Protocol: when a real conversation produces a bad response, log it here immediately.
> Generate correct response with teacher model. Add to dataset. Retrain 1 epoch. Retest.

| ID | Date | Phase | Utterance (EN/AR) | Bad Response | Correct Response | Root Cause | Added to Dataset | Status |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | No failures yet |

---

## Teacher Model Usage

- **Model:** OpenAI `gpt-4o` (check latest at time of use)
- **API key env var:** `HF_OPENAI_API_KEY` (never commit)
- **Cost estimate:** ~$0.02–0.05 per sample (Phase 3 one-time)
- **Total estimated cost:** ~$10–50 for v1 dataset (500–1,000 samples)

Teacher prompt template (append to each generation call):
```
Generate a response for a bilingual Arabic/English voice assistant.
Rules:
- Max 2-3 sentences
- No lists or bullet points
- Use [laugh], [emphasis], or [sigh] where natural
- Respond in the exact same language as the input
- Gulf/UAE Arabic dialect if responding in Arabic
- Use contractions (it's, you're, that's) in English
- Never start with "Certainly", "Of course", "Great question"
```

---

## Jais Fallback Activation Procedure

If Arabic comprehension accuracy falls below 70% after Phase 2 evaluation:

1. Search HuggingFace for latest Jais release (search `inceptionai/jais`)
2. Verify Mac Mini compatibility (check GGUF availability)
3. `ollama pull jais` (or load GGUF via Ollama Modelfile)
4. Update `HF_OLLAMA_MODEL=jais` in `.env`
5. Set `HF_PIPELINE_PHASE=jais_fallback`
6. Re-run benchmark: `python scripts/benchmark.py --iterations 10`
7. Update this file with eval results
8. Log the switch decision in `PROJECT.md` §9 Change Log

---

## F5-TTS Upgrade Path (Phase 4+)

Fish Speech 1.5 is the POC TTS. For Phase 4+ (post-LoRA), consider upgrading to F5-TTS for better prosody and voice cloning.

| Property | Fish Speech 1.5 | F5-TTS |
|---|---|---|
| Latency on Mac MPS | ~15–25s/sentence | ~30–60s/sentence (diffusion) |
| Arabic support | Native | `F5-TTS-Arabic` fine-tune (HuggingFace) |
| Voice cloning | Yes (Phase 6) | Yes |
| Streaming | HTTP chunked | Limited |
| POC decision | **Selected** | Phase 4+ |

To upgrade: search HuggingFace for `SWivid/F5-TTS` and `F5-TTS-Arabic` latest release before installing. The diffusion model requires CUDA for real-time — Azure deployment recommended for production.

---

## Update Instructions (for Claude Code)

| Trigger | Section to update |
|---|---|
| Dataset file created | Dataset Inventory — update size + status |
| Training run complete | Training Runs — add row |
| Eval session complete | Evaluation History — add row |
| Failure captured in conversation | Failure Capture Log — add row immediately |
| Failure resolved | Failure Capture Log — update status |
| Phase activated | Pipeline Phases — update status column |
| Jais activated | Jais Fallback Activation — document the switch |
