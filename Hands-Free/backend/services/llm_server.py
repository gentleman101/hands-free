"""
LLM server (Mac side): FastAPI + llama.cpp Metal + JAIS-2 8B + KV cache reuse.

Runs as a standalone service on Mac Mini. Windows client calls /api/llm/generate.

Key optimizations:
- llama.cpp Metal backend (Apple Silicon GPU)
- KV cache reuse (`cache_type="reuse"`) — saves 200-500ms/req by reusing the
  system-prompt KV cache across calls
- Streaming HTTP response (newline-delimited JSON) so client TTFT = first token
- think:false equivalent: JAIS doesn't have reasoning by default; n_predict cap
  via HF_MAX_TOKENS prevents runaway responses (EXP-007)
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import config as cfg
from backend.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

CONFIG = cfg.load()


# ─── Lazy model load ────────────────────────────────────────────────────────
_model = None


def _get_model():
    """Load llama.cpp model on first call. Caches for subsequent calls."""
    global _model
    if _model is None:
        try:
            import llama_cpp  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: "
                'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
            ) from e

        if not CONFIG.llm_model_path:
            raise RuntimeError("HF_LLM_MODEL_PATH not set in environment")

        log.info(
            "loading llm model",
            extra={
                "path": CONFIG.llm_model_path,
                "n_gpu_layers": CONFIG.llm_n_gpu_layers,
                "n_ctx": CONFIG.llm_n_ctx,
            },
        )

        _model = llama_cpp.Llama(
            model_path=CONFIG.llm_model_path,
            n_gpu_layers=CONFIG.llm_n_gpu_layers,
            n_ctx=CONFIG.llm_n_ctx,
            verbose=False,
        )
        log.info("llm model ready", extra={"model": CONFIG.llm_model_name})
    return _model


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
    """Liveness probe — does NOT load the model."""
    return {"status": "ok", "model": CONFIG.llm_model_name}


@app.get("/readyz")
async def ready() -> dict:
    """Readiness probe — ensures model loads."""
    try:
        _get_model()
        return {"status": "ready", "model": CONFIG.llm_model_name}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Convert chat messages to a single prompt string for llama.cpp.

    Most GGUF chat templates handle this internally via create_chat_completion,
    but for streaming we format manually for full control.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        elif role == "user":
            parts.append(f"<user>\n{content}\n</user>")
        elif role == "assistant":
            parts.append(f"<assistant>\n{content}\n</assistant>")
    parts.append("<assistant>\n")
    return "\n".join(parts)


async def _stream_tokens(req: LLMRequest) -> AsyncIterator[bytes]:
    """Yield NDJSON token events."""
    model = _get_model()
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

    t0 = time.perf_counter()
    ttft_ms: int | None = None
    token_count = 0

    # Use create_chat_completion for proper chat template handling
    stream = model.create_chat_completion(
        messages=req.messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not delta:
            continue
        if ttft_ms is None:
            ttft_ms = int((time.perf_counter() - t0) * 1000)
            log.info("llm ttft", extra={"ttft_ms": ttft_ms, "request_id": rid})
        token_count += 1
        payload = {"token": delta, "done": False}
        if ttft_ms is not None and token_count == 1:
            payload["ttft_ms"] = ttft_ms
        yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

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
