"""
LLM service using Ollama via OpenAI-compatible SDK.

Supports both streaming (generate_stream) and non-streaming (generate).
Streaming yields text chunks as tokens arrive — first token ~200ms.
"""
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from backend.monitoring.latency_tracker import track_latency

log = logging.getLogger(__name__)

@dataclass
class GenerateResult:
    text: str
    tokens_generated: int
    ttft_ms: int      # time-to-first-token
    duration_ms: int  # total generation time


def _client() -> AsyncOpenAI:
    """Build a fresh AsyncOpenAI client per call (cheap; ensures correct event loop binding)."""
    return AsyncOpenAI(
        base_url=os.environ.get("HF_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # Ollama ignores this but SDK requires a value
    )


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
    Stream tokens from Ollama as they are generated.
    First token arrives in ~200ms. Each yield is a partial text string.
    TTFT is recorded internally.
    """
    rid = request_id or str(uuid.uuid4())
    max_tok = max_tokens or _max_tokens()
    client = _client()

    t0 = time.perf_counter()
    ttft_recorded = False

    log.debug("llm stream start", extra={"request_id": rid, "model": _model()})

    stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(  # type: ignore[attr-defined, assignment]
        model=_model(),
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tok,
        stream=True,
        temperature=0.7,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
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
