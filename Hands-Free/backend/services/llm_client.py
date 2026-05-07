"""
LLM client (Windows side): HTTP RPC to Mac LLM server.

Optimizations:
- Module-level singleton httpx.AsyncClient with connection pooling
  (reuses TCP connections; saves 50-100ms/request on office WiFi)
- 3-retry with 0.5s backoff for transient network failures
- Health check on startup (HF_MACHINE_ROLE=client only)
- Streaming response: yields tokens as they arrive over the wire
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncIterator

import httpx

from backend import config as cfg
from backend.database import fire_and_forget, insert_event

log = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    text: str
    tokens_generated: int
    ttft_ms: int
    duration_ms: int


# ─── Singleton httpx client with connection pooling ─────────────────────────
_client: httpx.AsyncClient | None = None


def _http_client() -> httpx.AsyncClient:
    """Lazy-init httpx client with connection pooling.

    Limits chosen for office network: keep up to 5 connections warm, max 10
    total. Pooling saves 50-100ms/request vs creating a new connection each
    time (TCP handshake + httpx setup).
    """
    global _client
    if _client is None:
        config = cfg.load()
        _client = httpx.AsyncClient(
            base_url=config.llm_service_url,
            timeout=httpx.Timeout(config.llm_timeout_s, connect=2.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _client


async def aclose() -> None:
    """Close the httpx client (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ─── Health check ───────────────────────────────────────────────────────────
async def health_check(timeout_s: float = 2.0) -> bool:
    """Return True if Mac LLM server is reachable. False on any error."""
    try:
        resp = await _http_client().get("/healthz", timeout=timeout_s)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError):
        return False
    except Exception:
        log.exception("unexpected health_check error")
        return False


# ─── Streaming generation ───────────────────────────────────────────────────
async def generate_stream(
    messages: list[dict[str, str]],
    *,
    language: str = "en",
    max_tokens: int | None = None,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """
    Stream tokens from Mac LLM server.

    Yields plain text chunks (deltas). On first token, emits a llm_ttft event
    to the metrics DB.

    Retries up to llm_max_retries times on transient network failures with
    0.5s backoff between attempts.
    """
    config = cfg.load()
    rid = request_id or str(uuid.uuid4())
    payload = {
        "messages": messages,
        "language": language,
        "max_tokens": max_tokens,
        "request_id": rid,
    }

    last_exc: Exception | None = None
    for attempt in range(config.llm_max_retries):
        try:
            t0 = time.perf_counter()
            ttft_ms: int | None = None

            async with _http_client().stream(
                "POST", "/api/llm/generate", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise httpx.HTTPError(f"LLM server returned {resp.status_code}: {body[:200]}")

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("bad llm stream line", extra={"line": line[:100]})
                        continue

                    if data.get("done"):
                        break

                    token = data.get("token", "")
                    if not token:
                        continue

                    # Record TTFT on first token
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                        log.info("llm ttft", extra={"ttft_ms": ttft_ms, "request_id": rid})
                        fire_and_forget(insert_event(
                            request_id=rid,
                            stage="llm_ttft",
                            duration_ms=ttft_ms,
                            timestamp=datetime.now(UTC).isoformat(),
                            language=language,
                            model_version=config.llm_model_name,
                            pipeline_phase=config.pipeline_phase,
                            success=True,
                        ))

                    yield token
            return  # success

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exc = e
            if attempt < config.llm_max_retries - 1:
                log.warning(
                    "llm rpc transient failure",
                    extra={"attempt": attempt + 1, "err": str(e), "request_id": rid},
                )
                await asyncio.sleep(0.5)
            else:
                log.error(
                    "llm rpc failed after retries",
                    extra={"retries": config.llm_max_retries, "err": str(e), "request_id": rid},
                )

    if last_exc is not None:
        raise last_exc


# ─── Non-streaming convenience ──────────────────────────────────────────────
async def generate(
    messages: list[dict[str, str]],
    *,
    language: str = "en",
    max_tokens: int | None = None,
    request_id: str | None = None,
) -> GenerateResult:
    """Collect full response. Use generate_stream in pipeline; this for tests."""
    rid = request_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    chunks: list[str] = []
    ttft_ms = 0
    first = True

    async for token in generate_stream(
        messages, language=language, max_tokens=max_tokens, request_id=rid
    ):
        if first:
            ttft_ms = int((time.perf_counter() - t0) * 1000)
            first = False
        chunks.append(token)

    text = "".join(chunks)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    return GenerateResult(
        text=text,
        tokens_generated=len(chunks),
        ttft_ms=ttft_ms,
        duration_ms=duration_ms,
    )
