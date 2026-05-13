"""
LLM server (Mac side): FastAPI + Ollama + Gemma 4 E4B Q8_0.

Runs as a standalone service on Mac Mini. Windows client calls /api/llm/generate.

Key optimizations:
- Ollama Metal backend — native Apple Silicon GPU via llama.cpp built into Ollama
- Gemma 4 E4B: ~1B active params per token (MoE), fast on M4
- think:false via /api/chat (NOT /v1/chat/completions — that silently drops it, EXP-001)
- Streaming HTTP response (newline-delimited JSON) so client TTFT = first token
- Filler stitching: last assistant message used as open generation prefix
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import config as cfg
from backend.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

CONFIG = cfg.load()

# Ollama native API (not OpenAI-compat) — required for think:false to work
_OLLAMA_BASE = "http://localhost:11434"
_OLLAMA_CHAT_URL = f"{_OLLAMA_BASE}/api/chat"
_OLLAMA_TAGS_URL = f"{_OLLAMA_BASE}/api/tags"


# ─── Schema ─────────────────────────────────────────────────────────────────
class LLMRequest(BaseModel):
    messages: list[dict[str, str]]
    language: str = Field(default="en", description="en | ar (for logging)")
    max_tokens: int | None = None
    temperature: float | None = None
    request_id: str | None = None


# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Hands-Free LLM Server")


@app.get("/healthz")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "model": CONFIG.llm_model_name}


@app.get("/readyz")
async def ready() -> dict:
    """Readiness probe — checks Ollama is up and model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(_OLLAMA_TAGS_URL)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model_name = CONFIG.llm_model_name
            loaded = any(model_name in m for m in models)
            if not loaded:
                raise HTTPException(status_code=503, detail=f"Model {model_name} not found in Ollama. Run: ollama pull {model_name}")
            return {"status": "ready", "model": model_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


async def _stream_tokens(req: LLMRequest) -> AsyncIterator[bytes]:
    """Yield NDJSON token events from Ollama /api/chat stream."""
    max_tokens = req.max_tokens or CONFIG.max_tokens
    temperature = req.temperature if req.temperature is not None else CONFIG.llm_temperature
    rid = req.request_id or "no-rid"

    log.info(
        "llm request",
        extra={
            "request_id": rid,
            "language": req.language,
            "msg_count": len(req.messages),
            "max_tokens": max_tokens,
        },
    )

    messages = list(req.messages)

    # Filler stitching: seed open assistant turn so model continues naturally
    filler_prefix: str | None = None
    if messages and messages[-1]["role"] == "assistant":
        filler_prefix = messages.pop()["content"]

    # Build Ollama /api/chat payload — think:false disables Gemma 4 reasoning
    payload: dict = {
        "model": CONFIG.llm_model_name,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    # Inject filler as partial assistant prefix if present
    if filler_prefix:
        payload["messages"].append({"role": "assistant", "content": filler_prefix, "prefix": True})

    t0 = time.perf_counter()
    ttft_ms: int | None = None
    token_count = 0

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", _OLLAMA_CHAT_URL, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                log.error("ollama error", extra={"status": response.status_code, "body": body.decode()})
                raise HTTPException(status_code=502, detail="Ollama request failed")

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("message", {}).get("content", "")
                if not delta:
                    continue

                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - t0) * 1000)
                    log.info("llm ttft", extra={"ttft_ms": ttft_ms, "request_id": rid})

                token_count += 1
                out: dict = {"token": delta, "done": False}
                if token_count == 1:
                    out["ttft_ms"] = ttft_ms
                yield (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")

                if chunk.get("done"):
                    break

    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "llm complete",
        extra={
            "request_id": rid,
            "tokens": token_count,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
        },
    )
    yield (json.dumps({"token": "", "done": True, "duration_ms": duration_ms}) + "\n").encode("utf-8")


@app.post("/api/llm/generate")
async def generate(req: LLMRequest) -> StreamingResponse:
    """Stream LLM tokens as newline-delimited JSON."""
    return StreamingResponse(
        _stream_tokens(req),
        media_type="application/x-ndjson",
    )
