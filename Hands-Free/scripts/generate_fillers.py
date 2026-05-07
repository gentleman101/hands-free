"""
One-time script: generate filler audio WAVs via Fish Speech.

Run on Windows (or any machine with Fish Speech sidecar running) to populate
assets/fillers/ with 5+ files per language.

Usage:
  # Start Fish Speech sidecar first (separate terminal):
  python -m fish_speech.server --listen 0.0.0.0:8001 --device cuda

  # Then generate the fillers:
  python scripts/generate_fillers.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


FILLERS = {
    "en": {
        "en_let_me_check.wav": "Let me check that for you.",
        "en_one_moment.wav": "One moment please.",
        "en_looking_into_it.wav": "Looking into that.",
        "en_just_a_second.wav": "Just a second.",
        "en_processing.wav": "Processing your request.",
    },
    "ar": {
        "ar_lahza_min_fadlek.wav": "لحظة من فضلك",
        "ar_da3ni_atahaqaq.wav": "دعني أتحقق من ذلك",
        "ar_thaniya_wahida.wav": "ثانية واحدة",
        "ar_jari_almoaalja.wav": "جاري المعالجة",
        "ar_intazher_qaleelan.wav": "انتظر قليلاً",
    },
}


async def main() -> int:
    from backend.logging_config import setup_logging
    from backend.services import tts

    setup_logging()
    log = logging.getLogger(__name__)

    out_dir = Path("assets/fillers")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fish Speech health check
    if not await tts.health_check():
        url = os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001")
        log.error(
            "Fish Speech sidecar not reachable. Start it first.",
            extra={"url": url},
        )
        return 1

    total = 0
    for lang, files in FILLERS.items():
        for fname, text in files.items():
            out_path = out_dir / fname
            if out_path.exists():
                log.info("skipping existing", extra={"file": fname})
                continue
            log.info("generating", extra={"file": fname, "lang": lang, "text": text})
            result = await tts.synthesize(text, language=lang)
            out_path.write_bytes(result.audio_bytes)
            log.info("wrote", extra={"file": str(out_path), "bytes": len(result.audio_bytes)})
            total += 1

    log.info("done", extra={"generated": total})
    await tts.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
