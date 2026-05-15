"""Smoke test for semantic filler matching — measures per-call latency."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

print("Loading filler module (downloads MiniLM model if not cached ~118 MB)...")
t0 = time.perf_counter()
from backend.services import filler as _filler_mod
from backend.services.filler import pick_filler_semantic

# Wait for background pre-warm thread to finish (model load + index build)
print("Waiting for pre-warm thread...")
_filler_mod._preload_thread.join(timeout=120)
load_ms = int((time.perf_counter() - t0) * 1000)

from backend.services.filler import _sbert, _sem_embeddings
print(f"Import + index build: {load_ms}ms")
print(f"SBERT loaded: {_sbert is not None}")
print(f"Languages indexed: {list(_sem_embeddings.keys())}")
for lang, embs in _sem_embeddings.items():
    print(f"  {lang}: {embs.shape[0]} fillers, {embs.shape[1]}-dim embeddings")
print()

tests = [
    ("my electricity bill seems too high this month", "en"),
    ("where is the nearest DEWA happiness center?", "en"),
    ("the power went out in my building", "en"),
    ("I want to apply for a new water connection", "en"),
    ("can I install solar panels on my villa?", "en"),
    ("hello, what can you help me with?", "en"),
    ("كم فاتورتي هذا الشهر", "ar"),
    ("انقطع التيار الكهربائي عن منطقتي منذ ساعة", "ar"),
    ("أين أقرب مركز خدمة؟", "ar"),
]

print(f"{'Query':<50} {'Lang':>4} {'ms':>5}  Filler selected")
print("-" * 100)
for text, lang in tests:
    t0 = time.perf_counter()
    wav, filler_text = pick_filler_semantic(text, lang)
    ms = int((time.perf_counter() - t0) * 1000)
    wav_ok = "ok" if wav else "MISSING"
    print(f"{text[:50]:<50} {lang:>4} {ms:>4}ms  [{wav_ok}] {filler_text}")
