"""
LLM service using Ollama native /api/chat endpoint (NOT the OpenAI-compatible /v1).

Why native endpoint: Ollama bug #15288 — the /v1/chat/completions endpoint silently
drops the `think` parameter, so `think: false` has no effect there. The native
/api/chat endpoint correctly suppresses gemma4's reasoning chain.

`think: false` is hardcoded — reasoning adds 20-32s latency with no benefit for
a voice assistant answering short conversational queries.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    text: str
    tokens_generated: int
    ttft_ms: int      # time-to-first-token
    duration_ms: int  # total generation time


def _ollama_base() -> str:
    """Base Ollama URL — strips /v1 suffix so HF_OLLAMA_BASE_URL works unchanged."""
    url = os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return url.removesuffix("/v1").removesuffix("/")


def _model() -> str:
    return os.environ.get("HF_OLLAMA_MODEL", "gemma4:e2b")


def _max_tokens() -> int:
    return int(os.environ.get("HF_MAX_TOKENS", "2048"))


async def generate_stream(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """
    Stream tokens from Ollama via native /api/chat with think=false.
    Yields plain text chunks as they arrive.
    """
    rid = request_id or str(uuid.uuid4())
    max_tok = max_tokens or _max_tokens()

    payload = {
        "model": _model(),
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "num_predict": max_tok,
            "temperature": 0.7,
        },
    }

    t0 = time.perf_counter()
    ttft_recorded = False

    log.debug("llm stream start", extra={"request_id": rid, "model": _model()})

    url = f"{_ollama_base()}/api/chat"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                delta = data.get("message", {}).get("content", "")
                if delta:
                    if not ttft_recorded:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                        ttft_recorded = True
                        from backend.database import insert_event
                        from datetime import UTC, datetime

                        await insert_event(
                            request_id=rid,
                            stage="llm_ttft",
                            duration_ms=ttft_ms,
                            timestamp=datetime.now(UTC).isoformat(),
                            model_version=_model(),
                            pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
                            success=True,
                        )
                        log.info("llm ttft", extra={"ttft_ms": ttft_ms, "request_id": rid})
                    yield delta

                if data.get("done"):
                    break


@track_latency("llm_total")
async def generate(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    language: str | None = None,
) -> GenerateResult:
    """
    Non-streaming generation. Collects full response + timing.
    Use generate_stream for pipeline; use this for testing/benchmarking.
    """
    t0 = time.perf_counter()
    rid = str(uuid.uuid4())
    tokens = 0
    ttft_ms = 0
    chunks: list[str] = []
    first = True

    async for chunk in generate_stream(messages, max_tokens=max_tokens, request_id=rid):
        if first:
            ttft_ms = int((time.perf_counter() - t0) * 1000)
            first = False
        chunks.append(chunk)
        tokens += 1

    text = "".join(chunks)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    log.info(
        "llm complete",
        extra={
            "chars": len(text),
            "tokens_approx": tokens,
            "ttft_ms": ttft_ms,
            "duration_ms": duration_ms,
            "language": language,
        },
    )
    return GenerateResult(
        text=text,
        tokens_generated=tokens,
        ttft_ms=ttft_ms,
        duration_ms=duration_ms,
    )
