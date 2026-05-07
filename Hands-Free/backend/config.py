"""
Environment-driven configuration.

All settings come from environment variables (HF_ prefix) so the same
codebase runs on Windows (client) and Mac (server) with different .env files.

EXP-007: HF_MAX_TOKENS=200 caps response length (voice users hate long answers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Snapshot of HF_* env vars at startup."""

    # Role
    machine_role: str  # "client" | "server"

    # LLM (client side: where Mac is reachable)
    llm_service_url: str
    llm_timeout_s: float
    llm_max_retries: int

    # LLM (server side: model file + port)
    llm_model_path: str
    llm_model_name: str
    llm_port: int
    llm_n_gpu_layers: int
    llm_n_ctx: int

    # Generation
    max_tokens: int
    llm_temperature: float

    # STT
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    # TTS
    fish_speech_url: str
    tts_engine: str

    # VAD
    vad_backend: str
    audio_input_device: str

    # Metrics tagging
    pipeline_phase: str

    @property
    def is_client(self) -> bool:
        return self.machine_role == "client"

    @property
    def is_server(self) -> bool:
        return self.machine_role == "server"


def load() -> Config:
    """Load Config from env. Call once at startup."""
    return Config(
        machine_role=os.environ.get("HF_MACHINE_ROLE", "client"),
        llm_service_url=os.environ.get("HF_LLM_SERVICE_URL", "http://mac-mini.local:11434"),
        llm_timeout_s=float(os.environ.get("HF_LLM_TIMEOUT_S", "10")),
        llm_max_retries=int(os.environ.get("HF_LLM_MAX_RETRIES", "3")),
        llm_model_path=os.environ.get("HF_LLM_MODEL_PATH", ""),
        llm_model_name=os.environ.get("HF_LLM_MODEL_NAME", "jais-2-8b"),
        llm_port=int(os.environ.get("HF_LLM_PORT", "11434")),
        llm_n_gpu_layers=int(os.environ.get("HF_LLM_N_GPU_LAYERS", "99")),
        llm_n_ctx=int(os.environ.get("HF_LLM_N_CTX", "2048")),
        max_tokens=int(os.environ.get("HF_MAX_TOKENS", "200")),
        llm_temperature=float(os.environ.get("HF_LLM_TEMPERATURE", "0.7")),
        whisper_model=os.environ.get("HF_WHISPER_MODEL", "large-v3-turbo"),
        whisper_device=os.environ.get("HF_WHISPER_DEVICE", "cuda"),
        whisper_compute_type=os.environ.get("HF_WHISPER_COMPUTE_TYPE", "int8"),
        fish_speech_url=os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001"),
        tts_engine=os.environ.get("HF_TTS_ENGINE", "fish"),
        vad_backend=os.environ.get("HF_VAD_BACKEND", "ten"),
        audio_input_device=os.environ.get("HF_AUDIO_INPUT_DEVICE", ""),
        pipeline_phase=os.environ.get("HF_PIPELINE_PHASE", "raw"),
    )
