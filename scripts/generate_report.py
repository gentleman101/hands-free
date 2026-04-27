#!/usr/bin/env python3
"""
PERFORMANCE.md generator.

Reads data/metrics.db (live request latency) and data/benchmark_results.jsonl
(benchmark history) and regenerates PERFORMANCE.md with current numbers.

Run:
    python scripts/generate_report.py

Called automatically by benchmark.py after each benchmark run.
PERFORMANCE.md is always machine-generated — never edit it manually.
"""
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
DB_PATH = ROOT / "data" / "metrics.db"
BENCHMARK_LOG = ROOT / "data" / "benchmark_results.jsonl"
REPORT_PATH = ROOT / "PERFORMANCE.md"
SNAPSHOT_PATH = ROOT / "data" / "performance-snapshot.json"

TARGETS = {
    "stt": 300,
    "llm_ttft": 300,
    "tts_ttfa": 400,
    "pipeline_ttfa": 800,
    "total": 2000,
}

DISTILLATION_PHASES = [
    ("raw",          "gemma4:e2b",  "—",    "—",    "—"),
    ("prompted",     "gemma4:e2b",  "—",    "—",    "—"),
    ("lora_v1",      "gemma4:e2b",  "500",  "—",    "—"),
    ("lora_v2",      "gemma4:e2b",  "3000", "—",    "—"),
    ("jais_fallback","jais",        "—",    "—",    "—"),
]


def _pct(values: list[float], p: float) -> str:
    if not values:
        return "—"
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    return f"{sorted_v[min(idx, len(sorted_v)-1)]:.0f}ms"


def _load_db_stats(stage: str, last_n: int = 100) -> list[float]:
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT duration_ms FROM latency_events WHERE stage=? AND success=1 ORDER BY timestamp DESC LIMIT ?",
            (stage, last_n),
        ).fetchall()
        conn.close()
        return [float(r[0]) for r in rows]
    except Exception:
        return []


def _load_db_by_language(stage: str, language: str, last_n: int = 100) -> list[float]:
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT duration_ms FROM latency_events WHERE stage=? AND language=? AND success=1 ORDER BY timestamp DESC LIMIT ?",
            (stage, language, last_n),
        ).fetchall()
        conn.close()
        return [float(r[0]) for r in rows]
    except Exception:
        return []


def _load_phase_stats(phase: str) -> dict:  # type: ignore[type-arg]
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT stage, duration_ms FROM latency_events WHERE pipeline_phase=? AND success=1",
            (phase,),
        ).fetchall()
        conn.close()
        by_stage: dict[str, list[float]] = {}  # type: ignore[type-arg]
        for stage, dur in rows:
            by_stage.setdefault(stage, []).append(float(dur))
        return by_stage
    except Exception:
        return {}


def _load_benchmark_history() -> list[dict]:  # type: ignore[type-arg]
    if not BENCHMARK_LOG.exists():
        return []
    rows = []
    with BENCHMARK_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-20:]  # last 20 benchmark runs


def _status_icon(p50_str: str, target_ms: int) -> str:
    if p50_str == "—":
        return "⚪"
    try:
        val = int(p50_str.replace("ms", ""))
        return "🟢" if val <= target_ms else "🔴"
    except ValueError:
        return "⚪"


def generate() -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # --- Current latency table ---
    stage_map = {
        "stt":           ("STT (Whisper)",        "stt"),
        "llm_ttft":      ("LLM Time-to-First-Token", "llm_ttft"),
        "tts_ttfa":      ("TTS Time-to-First-Audio", "tts_ttfa"),
        "pipeline_ttfa": ("Pipeline TTFA ★",      "pipeline_ttfa"),
        "total":         ("Total Round-Trip",      "pipeline_total"),
    }

    current_rows = []
    for key, (label, db_stage) in stage_map.items():
        vals = _load_db_stats(db_stage)
        p50 = _pct(vals, 50)
        p95 = _pct(vals, 95)
        p99 = _pct(vals, 99)
        target = f"<{TARGETS[key]}ms"
        icon = _status_icon(p50, TARGETS[key])
        current_rows.append(f"| {icon} {label} | {p50} | {p95} | {p99} | {target} |")

    # --- Benchmark history table ---
    history = _load_benchmark_history()
    history_rows = []
    for run in reversed(history[-10:]):
        ts = run.get("timestamp", "")[:10]
        phase = run.get("pipeline_phase", "—")
        model = run.get("model", "—")
        stt = f"{run.get('stt', {}).get('p50', 0):.0f}ms"
        ttft = f"{run.get('llm_ttft', {}).get('p50', 0):.0f}ms"
        ttfa = f"{run.get('tts_ttfa', {}).get('p50', 0):.0f}ms"
        pttfa = f"{run.get('pipeline_ttfa', {}).get('p50', 0):.0f}ms"
        total = f"{run.get('total', {}).get('p50', 0):.0f}ms"
        history_rows.append(f"| {ts} | {phase} | {model} | {stt} | {ttft} | {ttfa} | **{pttfa}** | {total} |")

    if not history_rows:
        history_rows = ["| — | — | — | — | — | — | — | — |"]

    # --- Distillation progress table ---
    dist_rows = []
    current_phase = os.environ.get("HF_PIPELINE_PHASE", "raw")
    for phase, model, dataset, wer_en, wer_ar in DISTILLATION_PHASES:
        phase_stats = _load_phase_stats(phase)
        ttfa_vals = phase_stats.get("pipeline_ttfa", [])
        ttfa_p50 = _pct(ttfa_vals, 50) if ttfa_vals else "—"
        status = "[Active]" if phase == current_phase else "[Pending]"
        if phase == "jais_fallback":
            status = "[Fallback]"
        marker = "**" if phase == current_phase else ""
        dist_rows.append(
            f"| {marker}{phase}{marker} | {model} | {dataset} | {wer_en} | {wer_ar} | {ttfa_p50} | {status} |"
        )

    # --- Per-language breakdown ---
    lang_rows = []
    for lang, label in [("en", "English"), ("ar", "Arabic")]:
        stt_v = _load_db_by_language("stt", lang)
        llm_v = _load_db_by_language("llm_ttft", lang)
        tts_v = _load_db_by_language("tts_ttfa", lang)
        pttfa_v = _load_db_by_language("pipeline_ttfa", lang)
        lang_rows.append(
            f"| {label} | {len(stt_v)} reqs | {_pct(stt_v,50)} | {_pct(llm_v,50)} | {_pct(tts_v,50)} | {_pct(pttfa_v,50)} |"
        )

    # --- Assemble report ---
    report = f"""# Performance Report — Hands-Free
> Auto-generated by `scripts/generate_report.py` — **do not edit manually**.
> Last updated: {now}

---

## Current Latency (last 100 requests)

| Stage | p50 | p95 | p99 | Target |
|---|---|---|---|---|
{chr(10).join(current_rows)}

★ **Pipeline TTFA** = time from utterance end to first audio chunk. This is the primary user-perceived metric.

---

## Benchmark History (last 10 runs)

| Date | Phase | Model | STT p50 | LLM TTFT | TTS TTFA | **Pipeline TTFA** | Total p50 |
|---|---|---|---|---|---|---|---|
{chr(10).join(history_rows)}

---

## Distillation Progress

| Phase | Model | Dataset | WER EN | WER AR | TTFA p50 | Status |
|---|---|---|---|---|---|---|
{chr(10).join(dist_rows)}

**Phases:**
- `raw` — Off-the-shelf Gemma 4 E2B, no prompt engineering
- `prompted` — Bilingual system prompt + Fish Speech prosody instructions
- `lora_v1` — LoRA fine-tuned on 500-sample dataset (Phase 4)
- `lora_v2` — LoRA fine-tuned on 3000-sample dataset (Phase 4)
- `jais_fallback` — Arabic-first model (activate if Arabic quality < 70%)

---

## Per-Language Breakdown

| Language | Requests | STT p50 | LLM TTFT | TTS TTFA | Pipeline TTFA |
|---|---|---|---|---|---|
{chr(10).join(lang_rows)}

---

## How to Update This Report

```bash
# After running benchmarks:
python scripts/benchmark.py --iterations 10

# Or generate from live request data:
python scripts/generate_report.py
```
"""

    REPORT_PATH.write_text(report)
    print(f"  PERFORMANCE.md written ({len(report)} chars)")

    # Write snapshot JSON
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "generated_at": now,
        "pipeline_phase": current_phase,
        "targets": TARGETS,
        "benchmark_history": history[-5:],
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    generate()
