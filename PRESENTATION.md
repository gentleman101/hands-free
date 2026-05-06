# Rammas — POC to Production: Presentation Spec

**Format:** Single slide infographic (PowerPoint / Keynote / Figma)
**Audience:** Technical + management (mixed)
**Message:** Built with limited resources. Clear path to something much better.

---

## SLIDE STRUCTURE

Three vertical sections + one footer strip.

```
┌────────────────────────────────────────────────────────────────────────┐
│  RAMMAS — DEWA AI Voice Assistant                        [DEWA Logo]  │
│  "Built with limited resources. Ready to scale."                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│                          │                                             │
│   WHERE WE ARE           │   WHERE WE CAN GO                          │
│   (POC Today)            │   (More Time & Resources)                  │
│                          │                                             │
│   [Pipeline diagram]     │   [Pipeline diagram — same flow, better]  │
│                          │                                             │
│   [3 component cards]    │   [3 component cards — upgraded]           │
│                          │                                             │
├──────────────────────────┴─────────────────────────────────────────────┤
│   FUTURE HORIZON — Theoretically possible with dedicated ML resources  │
│   [Distillation flow diagram]                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## HEADER

**Title:** Rammas — DEWA AI Voice Assistant
**Subtitle:** Built with limited resources. Ready to scale.
**Top right:** DEWA logo

---

## LEFT COLUMN — WHERE WE ARE (POC Today)

**Background:** Dark navy `#0D1B2A`

### Pipeline Diagram (horizontal flow):
```
🎤  →  Voice Detection  →  Speech Recognition  →  AI Thinking  →  Voice Output  →  🔊
          Silero VAD          Whisper Small           Gemma4 E2B       Apple say
```

### Component Cards (3 cards side by side):

**Card 1 — HEARING**
- Model: Whisper Small on Apple Silicon GPU
- Bilingual: English + Arabic auto-detected
- Speed: 380ms
- ⚠️ Small model — occasional false positives on Arabic; Gulf dialect accuracy ~85%

**Card 2 — THINKING**
- Model: Gemma4 E2B — untuned, general knowledge
- Responds in detected language automatically
- Speed: 430ms to first word
- ⚠️ No DEWA-specific knowledge — answers from general world training only

**Card 3 — SPEAKING**
- Engine: macOS built-in voice (Apple say)
- Works in Arabic + English
- Speed: ~50ms (instant)
- ⚠️ Robotic — no natural prosody, no pauses, no warmth. Voice cannot be cloned.

### Key Metric (bold, large):
> ⏱ First response heard: **~814ms** after user stops speaking
> 🔒 100% on-premise — no data leaves DEWA network

---

## RIGHT COLUMN — WHERE WE CAN GO

**Background:** White with DEWA green `#007A4D` accents

### Pipeline Diagram (same flow, upgraded nodes highlighted in green):
```
🎤  →  Voice Detection  →  Speech Recognition  →  AI Thinking  →  Voice Output  →  🔊
          Silero VAD          Whisper Large-v3    Fine-tuned LLM   Fish Speech / Neural TTS
```

### Component Cards (3 cards side by side):

**Card 1 — BETTER HEARING**
- Model: Whisper Large-v3 on Apple Silicon GPU
- Arabic accuracy: ~97% (vs ~85% today)
- Gulf dialect + UAE accent support
- ✅ Near-zero false positives

**Card 2 — SMARTER THINKING**
- Fine-tuned on DEWA knowledge base
  (billing, outages, service requests, policies)
- Captures Rammas personality and response style
- Speed: ~200ms (prompt cache on known DEWA queries)
- ✅ No hallucinations on DEWA-specific topics
- Roadmap already designed: 500 samples → 3,000 samples → production
- Arabic-first fallback model (Jais) available if Arabic quality needs boost

**Card 3 — HUMAN-LIKE VOICE**
- Engine: Fish Speech / Neural TTS
- Natural prosody — rhythm, warmth, pauses
- Supports inline expressions (emphasis, soft tone)
- ✅ Voice cloning — clone any DEWA agent's voice for brand identity
- ✅ Barge-in — user can interrupt mid-response, just like a real conversation

### Key Metric (bold, large):
> ⏱ Target first response: **~600ms**
> 🗣 Natural back-and-forth — interruption supported

---

## FOOTER — FUTURE HORIZON

**Background:** Light grey `#F0F0F0`
**Label:** THEORETICALLY POSSIBLE — With dedicated ML resources (3–6 months)

### Distillation Flow Diagram (compact, horizontal):
```
[ChatGPT / OpenAI API]  ──►  [Curated DEWA Conversations]  ──►  [Rammas Own Model]
   Teacher model                Billing · Outages · HR             Runs 100% on-premise
                                Service requests · FAQs             Zero API dependency
                                ~3,000–10,000 examples              Full DEWA personality
```

### Three Impact Points (icon + one line each):
- 💰 **Cost:** Replace ongoing API fees with a model DEWA owns outright — near-zero cost per query at scale
- 🔒 **Privacy:** Every conversation stays inside DEWA infrastructure — always, by design
- 🎯 **Accuracy:** A model trained exclusively on DEWA data outperforms general AI on every DEWA-specific query

**Footnote (small italic):**
*Knowledge distillation from existing LLM APIs is an established production technique — not experimental research. The dataset format, training pipeline, and LoRA hyperparameters are already designed in this project.*

---

## DESIGN NOTES

**Colours:**
- Left column: dark navy `#0D1B2A`, white text — "current/limited"
- Right column: white background, DEWA green `#007A4D` accents — "potential"
- Footer strip: light grey `#F0F0F0` — future/theoretical
- Divider line between columns: DEWA green, 2px

**Typography:**
- Title: Bold, 28pt
- Section labels: Small caps, muted, 10pt
- Card headlines: Bold, 13pt
- Body text: 11pt
- Key stats (814ms, 600ms): 32pt bold, accent colour — numbers are the headline
- Arabic text: Cairo or Noto Arabic font

**Diagram style:**
- Pipeline flows: flat coloured rectangles + right-pointing arrows. No shadows, no 3D.
- ⚠️ on every left-column card (limitation), ✅ on every right-column card (upgrade)
- Cards: rounded corners, icon top-left, speed metric bottom-right
- Footer diagram: single-line boxes + arrows

**Guiding principle:**
> Numbers and icons tell the story. Every card shows exactly: what it does, how fast, and what's missing or better. One glance = full picture.
