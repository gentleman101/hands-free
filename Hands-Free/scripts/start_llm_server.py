"""
Mac Mini entry point: start the LLM HTTP server.

Run on Mac Mini after pip install -r requirements-shared.txt -r requirements-mac.txt
and downloading JAIS-2 8B GGUF to HF_LLM_MODEL_PATH.

  export HF_MACHINE_ROLE=server
  export HF_LLM_MODEL_PATH=/Users/naman/models/jais-2-8b-q4.gguf
  python scripts/start_llm_server.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    # Late import so config + logging set up first
    import uvicorn

    from backend import config as cfg
    from backend.logging_config import setup_logging

    setup_logging()
    config = cfg.load()

    if not config.is_server:
        print(
            "ERROR: HF_MACHINE_ROLE must be 'server' to run the LLM server. "
            f"Current: {config.machine_role}",
            file=sys.stderr,
        )
        return 1

    # Ollama manages model loading — no prewarm needed here

    uvicorn.run(
        "backend.services.llm_server:app",
        host="0.0.0.0",
        port=config.llm_port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
