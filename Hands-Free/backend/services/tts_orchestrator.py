"""
Sentence-parallel TTS orchestrator.

When the LLM emits multiple sentences, this fans out TTS calls concurrently
(one asyncio.Task per sentence) while preserving playback order.

Pattern:
  sentence 1 boundary → spawn TTS task 1
  sentence 2 boundary → spawn TTS task 2  (parallel with task 1)
  consumer drains task 1's chunks → then task 2's chunks → ...

Why asyncio.Task and not multiprocessing:
  - TTS calls are HTTP I/O bound (Fish Speech sidecar is on WSL2)
  - asyncio gives us cheap concurrency with zero pickling overhead
  - Ordering is enforced by a queue of queues (one inner queue per sentence)

Gain: saves (N-1) × TTS_TTFA ms for N-sentence replies. Typical Rammas reply
is 1–2 sentences (brevity prompt), so the real-world win is modest, but the
infrastructure is in place for longer replies later.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from backend.services import tts as tts_svc

log = logging.getLogger(__name__)


# Sentinel to mark "this sentence is done"
_DONE = object()


async def _synth_into_queue(
    sentence: str, language: str, request_id: str, q: asyncio.Queue
) -> None:
    """Background task: stream a sentence's TTS chunks into a queue."""
    try:
        async for chunk in tts_svc.synthesize_stream(
            sentence, language, request_id=request_id
        ):
            await q.put(chunk)
    except Exception as e:
        log.error("tts task failed", extra={"err": str(e), "sentence": sentence[:40]})
    finally:
        await q.put(_DONE)


class TTSOrchestrator:
    """
    Submit sentences as they become available; consume audio chunks in order.

    Usage:
        orch = TTSOrchestrator(language="en", request_id=rid)
        # producer side:
        await orch.submit("First sentence.")
        await orch.submit("Second sentence.")
        await orch.close()  # no more sentences coming
        # consumer side:
        async for chunk in orch.stream():
            yield chunk
    """

    def __init__(self, language: str, request_id: str) -> None:
        self._language = language
        self._rid = request_id
        self._queues: asyncio.Queue[asyncio.Queue | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def submit(self, sentence: str) -> None:
        """Spawn a TTS task for this sentence. Non-blocking."""
        if not sentence.strip():
            return
        inner: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(
            _synth_into_queue(sentence, self._language, self._rid, inner)
        )
        self._tasks.append(task)
        await self._queues.put(inner)

    async def close(self) -> None:
        """Signal that no further sentences will be submitted."""
        await self._queues.put(None)

    async def stream(self) -> AsyncIterator[bytes]:
        """
        Yield audio chunks in submission order.
        Blocks waiting for each sentence's chunks to arrive.
        """
        while True:
            inner = await self._queues.get()
            if inner is None:
                break
            while True:
                item = await inner.get()
                if item is _DONE:
                    break
                yield item  # type: ignore[misc]

        # Ensure all tasks completed (errors already logged)
        for t in self._tasks:
            if not t.done():
                await t
