"""
Test filler-stitched prompts against the live LLM.
Prints the full assembled system prompt, user message, exact LLM output, and timing.

Run:
    venv\Scripts\python scripts/test_filler_prompt.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.services import prompt_builder
from backend.services import llm_client

CASES = [
    {
        "label": "EN billing",
        "text": "my electricity bill seems very high this month",
        "lang": "en",
        "filler": "Let me pull that up,",
    },
    {
        "label": "EN outage",
        "text": "the power went out in my building an hour ago",
        "lang": "en",
        "filler": "Aaah... yes... got it,",
    },
    {
        "label": "AR location",
        "text": "أين أقرب مركز خدمة ديوا؟",
        "lang": "ar",
        "filler": "آه... نعم، على الفور،",
    },
]

W = 72


async def run_case(i: int, case: dict) -> None:
    messages = prompt_builder.build_messages(
        case["text"],
        language=case["lang"],
        filler_text=case["filler"],
    )

    system_prompt = messages[0]["content"]
    user_msg = messages[-1]["content"]

    print(f"\n{'=' * W}")
    print(f"  [{i}] {case['label']}")
    print(f"{'=' * W}")

    print("\n>>> SYSTEM PROMPT (full)\n")
    print(system_prompt)

    print(f"\n{'─' * W}")
    print(f"\n>>> USER MESSAGE\n")
    print(user_msg)

    print(f"\n{'─' * W}")
    print(f"\n>>> LLM RAW OUTPUT  (streaming tokens as they arrive)\n")

    t0 = time.perf_counter()
    chunks: list[str] = []
    ttft_ms = 0
    first = True

    async for token in llm_client.generate_stream(messages, language=case["lang"]):
        if first:
            ttft_ms = int((time.perf_counter() - t0) * 1000)
            first = False
        chunks.append(token)
        print(token, end="", flush=True)

    total_ms = int((time.perf_counter() - t0) * 1000)
    response = "".join(chunks).strip()

    print(f"\n\n{'─' * W}")
    print(f"\n>>> STITCHED  (filler + llm output)\n")
    print(f'"{case["filler"]} {response}"')

    print(f"\n{'─' * W}")
    print(f"  TTFT : {ttft_ms} ms    Total : {total_ms} ms    Tokens : {len(chunks)}")


async def main() -> None:
    import os
    print(f"gemma4:e4b  |  {os.environ.get('HF_LLM_SERVICE_URL', 'endpoint not set')}")

    for i, case in enumerate(CASES, 1):
        await run_case(i, case)

    await llm_client.aclose()
    print(f"\n{'=' * W}\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
