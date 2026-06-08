"""
Custom LLMService for the distributed Mac LLM endpoint.

The Mac runs Ollama behind a thin FastAPI wrapper that exposes:
    POST http://192.168.137.237:8080/api/llm/generate
    payload: {"messages": [...], "language": "en", "max_tokens": N,
              "request_id": "..."}
    stream:  JSON-lines, one per chunk:
             {"token": "...", "done": false, "ttft_ms": 8297}
             {"token": "",    "done": true,  "duration_ms": 8710}

This is NOT OpenAI-compatible (Ollama port 11434 isn't exposed externally),
so Pipecat's OllamaLLMService can't talk to it directly. We subclass
LLMService and implement the streaming ourselves.

NOT WIRED INTO demo.py YET — instantiate from the demo script when you're
ready to flip the LLM live.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

log = logging.getLogger(__name__)


class MacLLMService(LLMService):
    """Streams from the Mac Mini Ollama-backed FastAPI wrapper."""

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.137.237:8080",
        max_tokens: int = 200,
        language: str = "en",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._language = language
        self._session: aiohttp.ClientSession | None = None

    def set_language(self, language: str) -> None:
        """Hot-swap language for next LLM request (called by LanguageRouter)."""
        self._language = language

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120, connect=2)
            )

    async def _stream_tokens(
        self, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        await self._ensure_session()
        rid = str(uuid.uuid4())
        payload = {
            "messages": messages,
            "language": self._language,
            "max_tokens": self._max_tokens,
            "request_id": rid,
        }
        url = f"{self._base_url}/api/llm/generate"

        log.info("mac_llm.request", extra={"rid": rid, "messages": len(messages)})

        assert self._session is not None
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("mac_llm.bad_json", extra={"line": line[:120]})
                    continue
                tok = obj.get("token", "")
                if tok:
                    yield tok
                if obj.get("done"):
                    return

    async def health_check(self) -> bool:
        """GET /healthz on the Mac wrapper."""
        try:
            await self._ensure_session()
            assert self._session is not None
            async with self._session.get(
                f"{self._base_url}/healthz", timeout=aiohttp.ClientTimeout(total=2)
            ) as r:
                return r.status == 200
        except Exception:
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.get_messages()
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            tok_count = 0
            first_token_logged = False
            try:
                async for tok in self._stream_tokens(messages):
                    if not first_token_logged:
                        print(f"   ↳ first token received", flush=True)
                        first_token_logged = True
                    tok_count += 1
                    await self.push_frame(LLMTextFrame(tok), direction)
                print(f"   ↳ LLM done ({tok_count} tokens)", flush=True)
            except Exception as e:
                print(f"   ↳ LLM stream failed: {e}", flush=True)
                log.exception("mac_llm.stream_failed")
                await self.push_frame(
                    ErrorFrame(error=f"MacLLM stream failed: {e}"), direction
                )
            finally:
                await self.push_frame(LLMFullResponseEndFrame(), direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        await super().cleanup()
