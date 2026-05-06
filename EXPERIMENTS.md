# EXPERIMENTS.md — Hands-Free

> Tracks every deliberate tuning change: what changed, why, the original value, and how to revert.
> Fill in the **Result** section after testing each experiment.
> Status: `Active` | `Reverted` | `Superseded`

---

## EXP-001 — LLM native Ollama endpoint + think:false
Date: 2026-04-27
Status: Active

### Change
File: `backend/services/llm.py`

Before: OpenAI SDK → `/v1/chat/completions` with `extra_body={"think": False}` (silently ignored — Ollama bug #15288)

After: `httpx` → native `/api/chat` with `"think": False` hardcoded in payload

### Rationale
Ollama's `/v1/chat/completions` endpoint silently drops the `think` parameter, so Gemma4 spent 20–32s reasoning. Native `/api/chat` correctly passes `think: false` to the chat template.

### Expected gain
LLM total: 20–32s → 2–5s

### Revert
Replace `httpx` calls with `AsyncOpenAI` client using `/v1/chat/completions`. Remove `"think": False` from payload dict.

### Result
✅ Confirmed 240ms warm TTFT. Reasoning fully suppressed.

---

## EXP-002 — STT EN/AR language constraint
Date: 2026-04-27
Status: Active

### Change
File: `backend/services/stt.py`

Before: `model.transcribe(audio_np, language=env_var_or_None)` — Whisper auto-detected from 99 languages

After: `_, _, lang_probs_list = model.detect_language(audio_np)` then force `language="en"` or `"ar"` based on highest of `en_prob`/`ar_prob`. Return empty if `max < 0.30`.

### Rationale
Short/accented phrases were misdetected as French (0.24 confidence), Portuguese, etc. Constraining to EN/AR eliminates false detections entirely.

### Expected gain
Zero language-detection misroutes. Arabic responses stay in Arabic; English in English.

### Revert
Remove `detect_language()` call. Restore `language=os.environ.get("HF_WHISPER_LANGUAGE")` fallback logic.

### Result
✅ No French/Portuguese misdetections observed in testing.

---

## EXP-003 — Fish Speech MPS + fp16
Date: 2026-04-27
Status: Active

### Change
File: `scripts/start.py` — `start_fish_speech()` subprocess args

Before: `["python", server_script, "--listen", "0.0.0.0:8001"]`

After: `["python", server_script, "--listen", "0.0.0.0:8001", "--device", "mps", "--half"]`

### Rationale
Fish Speech defaults to `--device cuda`. On Apple Silicon the model runs on CPU unless `--device mps` is set. `--half` forces fp16 (MPS lacks native bf16 support, which is Fish Speech's default).

### Expected gain
Model on MPS instead of CPU. Bandwidth 5× higher.

### Revert
Remove `"--device", "mps", "--half"` from subprocess args list.

### Result
✅ Startup log confirms `mps is available, running on mps.` But speed remains 8 tok/sec — see EXP-004.

---

## EXP-004 — Fish Speech torch.compile + MPS fallback disabled
Date: 2026-04-27
Status: Active

### Change
File: `scripts/start.py` — subprocess args + env

Before: no `--compile`, `PYTORCH_ENABLE_MPS_FALLBACK=1`

After: added `"--compile"` to args. Added `fish_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"` in subprocess env.

### Rationale
`torch.compile(decode_one_token)` fuses per-token kernel dispatches. `MPS_FALLBACK=0` prevents ops from silently bouncing to CPU.

### Expected gain
Fish Speech tok/sec: 8 → 30–100+

### Revert
Remove `"--compile"` from subprocess args. Remove `fish_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"` line.

### Result
⚠️ Still 8 tok/sec. Torch.compile's Inductor backend on MPS does not fuse small autoregressive ops well (MPS codegen is nascent). Architectural TTS replacement is needed for real gains — see deferred items.

---

## EXP-005 — SentenceBuffer: lower min chars + comma boundary
Date: 2026-04-27
Status: Active

### Change
File: `backend/services/pipeline.py`

Before:
```python
_BOUNDARIES = frozenset(".?!\n،؟！")
_MIN_SENTENCE_CHARS = 15
```

After:
```python
_BOUNDARIES = frozenset(".?!\n،؟！,")
_MIN_SENTENCE_CHARS = 8
```

### Rationale
Fish Speech first-chunk latency dominates pipeline TTFA. Shorter first emit = faster pipeline_ttfa. Comma added so the first clause ships to TTS before the full sentence is complete.

### Expected gain
pipeline_ttfa: shave 200–500ms off first-chunk delivery.

### Revert
`_MIN_SENTENCE_CHARS = 15`, `_BOUNDARIES = frozenset(".?!\n،؟！")` (remove comma).

### Result
✅ First clause ships to TTS earlier. No prosody regressions observed.

---

## EXP-006 — System prompt brevity rewrite
Date: 2026-04-27
Status: Active

### Change
File: `backend/services/prompt_builder.py` — `_SYSTEM_PROMPT`

Before: General bilingual assistant prompt, no hard word limit.

After: "CRITICAL: Keep replies VERY short — usually 1 sentence, never more than 2. Aim for under 20 words." with concrete examples.

### Rationale
Shorter LLM output → fewer TTS tokens → lower TTS latency. Voice users hate long answers.

### Expected gain
Average response length: 80 tokens → 15–25 tokens. TTS total: 30–40% reduction.

### Revert
Replace `_SYSTEM_PROMPT` with the prior general prompt (no CRITICAL brevity block, no examples).

### Result
✅ Responses are noticeably shorter. Occasionally too terse — monitor.

---

## EXP-007 — HF_MAX_TOKENS 2048 → 200
Date: 2026-04-27
Status: Active

### Change
File: `scripts/start.py` — `_DEFAULTS["HF_MAX_TOKENS"]`

Before: `"HF_MAX_TOKENS": "2048"` (legacy "thinking model needs room" comment)

After: `"HF_MAX_TOKENS": "200"`

### Rationale
Hard cap prevents runaway long responses. 200 tokens ≈ 150 words — generous for voice, but enforces the brevity intent from EXP-006.

### Expected gain
Worst-case response capped. Eliminates tail latency from unexpected long outputs.

### Revert
`"HF_MAX_TOKENS": "2048"` in `_DEFAULTS`.

### Result
✅ No cases observed where 200 tokens was insufficient for conversational replies.

---

## EXP-008 — Remove Whisper double-VAD
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/stt.py` — `_transcribe_sync()`

Before:
```python
segments, _ = model.transcribe(
    audio_np,
    beam_size=2,
    language=lang,
    vad_filter=True,
    vad_parameters={
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 100,
    },
)
```

After:
```python
segments, _ = model.transcribe(
    audio_np,
    beam_size=1,
    language=lang,
)
```

### Rationale
`audio_capture.py` already runs Silero VAD in real time, only emitting a trimmed speech segment. Whisper's internal `vad_filter` reruns the same Silero model on already-trimmed audio — duplicate work with no benefit.

### Expected gain
STT: 50–150ms saved per utterance.

### Revert
Re-add to `model.transcribe(...)`:
```python
vad_filter=True,
vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100},
```

### Result
*(fill in after testing)*

---

## EXP-009 — Whisper beam_size 2 → 1
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/stt.py` — `_transcribe_sync()` (same call as EXP-008)

Before: `beam_size=2`

After: `beam_size=1`

### Rationale
Beam search helps on long/ambiguous audio. For 1–3 second, single-speaker, language-forced, VAD-trimmed voice commands the quality difference is negligible (<0.2 WER points). Greedy decoding runs ~25–35% faster.

### Expected gain
STT decode: 80–200ms saved on `small`-int8 Metal.

### Revert
Change `beam_size=1` back to `beam_size=2` in `_transcribe_sync()`.

### Result
*(fill in after testing)*

---

## EXP-010 — HTTPX module-level singleton in tts.py
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/tts.py`

Before: `async with httpx.AsyncClient(timeout=60.0).stream("POST", url, ...) as response:`

After: Module-level `_http_client: httpx.AsyncClient | None = None` with `_tts_client()` getter. `synthesize_stream()` calls `_tts_client().stream(...)`.

### Rationale
Creating a new `AsyncClient` per call allocates a connection pool and transport on every TTS request. On localhost this costs ~5–15ms. A singleton reuses the open connection.

### Expected gain
TTS call overhead: 5–15ms saved per sentence chunk.

### Revert
Delete `_http_client` global and `_tts_client()` function. Restore:
```python
async with httpx.AsyncClient(timeout=60.0).stream("POST", url, json=payload) as response:
```

### Result
*(fill in after testing)*

---

## EXP-011 — VAD speech threshold 0.70 → 0.80
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/audio_capture.py:84`

Before: `_SPEECH_THRESHOLD = 0.7`

After: `_SPEECH_THRESHOLD = 0.80`

### Rationale
0.70 fires on HVAC noise, breath sounds, and ambient speech from nearby TVs/speakers. 0.80 requires a clear, intentional utterance before the pipeline activates.

### Expected gain
Fewer false-positive utterances reaching STT. Reduces wasted pipeline cycles on non-speech.

### Revert
`_SPEECH_THRESHOLD = 0.7` in `audio_capture.py:84`.
If too strict (clips first word), try 0.75 before reverting all the way to 0.70.

### Result
*(fill in after testing — watch for missed utterance starts)*

---

## EXP-012 — Status line UI + mute gate visibility
Date: 2026-04-28
Status: Active

### Change
Files: `scripts/start.py`, `backend/services/pipeline.py`

Before:
- No terminal indicator while listening or processing
- `log.warning("empty transcription — skipping...")` printed to console on every noise burst

After:
- `🎤  Listening...` shown on a persistent status line while mic is open
- `⏳  Processing — please wait...` replaces it when utterance detected + mic muted
- Status restored after each turn or on empty transcription
- Empty transcription warning downgraded to `log.debug` (no longer visible at WARNING level)

### Rationale
User needs to know when it's safe to speak (mute gate is now visible). Constant "empty" spam in console made real output hard to read.

### Expected gain
Better UX — clear speak/wait state. Cleaner console output.

### Revert
Remove `_status()` calls from `mic_loop`. Remove `_status()` helper. Restore `log.warning` in `pipeline.py` for both empty transcription checks.

### Result
*(fill in after testing)*

---

## EXP-013 — Language directive injected per-turn in prompt
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/prompt_builder.py` — `build_messages()`

Before: `language` param was accepted but only used in a debug log. System prompt said "respond in the same language" but gave Gemma no per-turn signal about *which* language is active.

After: A second system message is injected after `_SYSTEM_PROMPT` for every call:
```
IMPORTANT: The user is speaking Arabic. You MUST reply in Arabic only. Do not switch languages.
```
(or English, depending on detected language)

### Rationale
Observed: user said "كيف صحالك" (AR), got back "I'm doing well, thank you. How are you?" (EN).
Root cause: `language` param was unused. Gemma's training bias toward English wins without an explicit per-call directive.

### Expected gain
Zero language-switching errors. Arabic input → Arabic output, always.

### Revert
Remove the `lang_directive` second system message. Restore single-item `messages` list with only `_SYSTEM_PROMPT`.

### Result
*(fill in after testing)*

---

## EXP-014 — Apple say as primary TTS (Fish Speech standby)
Date: 2026-04-28
Status: Active

### Change
File: `scripts/start.py`

Before: Fish Speech always started at boot; `mic_loop(fish_available=_check_fish())`

After:
- `HF_TTS_ENGINE = "say"` added to `_DEFAULTS`
- Fish Speech only starts when `HF_TTS_ENGINE == "fish"`
- `mic_loop` routes to `_say()` when engine is `"say"` (Apple built-in)
- EN voice: Samantha, AR voice: Layla

### Rationale
Fish Speech tops out at 8 tok/sec on MPS (~3–18s TTS TTFA). Apple say is sub-100ms.
Using say for testing while evaluating a proper bilingual replacement (SILMA, Qwen3-TTS).

### Revert
Set `HF_TTS_ENGINE=fish` in `.env` or change default in `_DEFAULTS`.
Fish Speech will auto-start on next boot.

### Result
*(fill in after testing)*

---

## EXP-015 — STT engine: faster-whisper (CPU) → mlx-whisper (Apple Silicon GPU)
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/stt.py` — full rewrite

Before: `faster-whisper` with CTranslate2 — no MPS support, runs on CPU (~2000ms for short utterances)

After: `mlx-whisper` — runs Whisper natively via Apple MLX (Metal/Neural Engine). Quantized q4 models (~100MB). Single-pass: auto-detect language + transcribe in one call.

Model mapping:
- `small` → `mlx-community/whisper-small-mlx-q4`
- `base`  → `mlx-community/whisper-base-mlx-q4`
- etc.

Key differences from old stt.py:
- No separate `detect_language()` pass — detection happens during transcription
- `confidence` is always `1.0` (mlx_whisper doesn't expose per-language probabilities)
- EN/AR constraint: discard result if detected language is not "en" or "ar"
- `beam_size=1` (greedy, EXP-009 carried over)

### Rationale
CTranslate2 (faster-whisper's backend) has no MPS support — `device="auto"` silently falls back to CPU on Apple Silicon. mlx-whisper uses the Metal GPU/Neural Engine natively.

### Expected gain
STT: 2000ms → 150–300ms on `small-q4` on M-series Mac Mini.

### Revert
```python
# requirements: pip install faster-whisper
# restore original stt.py from git:
git checkout HEAD~1 -- backend/services/stt.py
```

### Result
✅ STT dropped from 2,073ms → 380ms (5.5× faster) on Apple Silicon GPU.
⚠️ Switched from q4 → full precision `small` model after Arabic transcription errors ("المترغسرة" instead of "المتاحة"). q4 quantization degrades Arabic accuracy noticeably.

---

## EXP-016 — Language tag prefixed on user message (double-enforcement)
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/prompt_builder.py`

Before: Language directive only in a second system message — Gemma could still default to English.

After: User message prefixed with language tag:
- Arabic: `[أجب بالعربية فقط] <transcription>`
- English: `[Reply in English only] <transcription>`

### Rationale
Turn 6 showed Arabic input ("ما هي الخدمات المترغسرة") got English response despite EXP-013 system-level directive. Gemma4 ignores system messages in favour of its training bias. Embedding the tag in the user turn itself is harder to ignore.

### Expected gain
Zero language-switching errors on Arabic input.

### Revert
Remove the `if language == "ar": tagged = ...` block. Restore `messages.append({"role": "user", "content": transcription})`.

### Result
*(fill in after testing)*

---

## EXP-017 — Split bilingual prompt into two language-specific prompts
Date: 2026-04-28
Status: Active

### Change
File: `backend/services/prompt_builder.py`

Before: Single `_SYSTEM_PROMPT` (~120 tokens) + language directive system message (~20 tokens) + `[أجب بالعربية فقط]` user tag (~8 tokens) = ~148 tokens per call regardless of language.

After: `_SYSTEM_PROMPT_EN` and `_SYSTEM_PROMPT_AR` (~65 tokens each). `build_messages()` selects by detected language. Language directive and user tag removed entirely.

### Rationale
Prompt tokens inflate LLM TTFT directly. Arabic prompt is in Arabic (model natively understands it without translation overhead), English prompt is in English. No cross-language instructions needed.

### Expected gain
~80 fewer prompt tokens per call → measurable TTFT reduction at every turn.
Also eliminates language-switching errors — Arabic prompt has no English in it at all.

### Revert
Restore single `_SYSTEM_PROMPT` string. Change `build_messages()` to use it unconditionally.

### Result
*(fill in after testing)*

---

## Deferred experiments (not yet started)

| ID | Description | Expected gain | Blocker |
|----|-------------|---------------|---------|
| EXP-011 | TTS engine → SILMA TTS v1 (bilingual EN+AR, 150M params, F5-TTS) | TTS TTFA 12–18s → 500–1200ms | Integration work needed |
| EXP-012 | TTS fallback → Piper hybrid (en_US-lessac + ar_JO-kareem, CPU) | TTFA ~50–200ms | No voice cloning |
| EXP-013 | VAD silence 600ms → 400ms in audio_capture.py | 200ms/turn direct | Regression risk on slow speakers |
| EXP-014 | Whisper small → base (244M → 74M) | STT ~2× faster | Needs EN/AR WER spot-check first |
| EXP-015 | Barge-in: detect speech while muted, cancel active playback | Near-instant response on interrupt | VAD-while-muted + cancellation logic |
