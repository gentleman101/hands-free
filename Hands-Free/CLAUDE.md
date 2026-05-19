# Hands-Free — Claude Context

## Machine Setup

### Windows (client)
- Role: VAD + STT + TTS + pipeline orchestration
- GPU: RTX A3000 (CUDA)
- Entry point: `venv\Scripts\python scripts/start.py`

### Mac Mini M4 (server)
- Role: LLM only
- IP: `192.168.137.28:8080`
- Health check: `GET /healthz` → `{"status":"ok","model":"gemma4:e4b-it-q8_0"}`
- LLM backend: Ollama (serves Gemma 4 E4B via FastAPI wrapper on port 8080)
- Repo path: `~/distributed-system/hands-free/Hands-Free`
- Start command:
  ```bash
  cd ~/distributed-system/hands-free/Hands-Free && HF_MACHINE_ROLE=server PYTHONPATH=. venv/bin/python scripts/start_llm_server.py
  ```
- Full start command (starts Ollama + FastAPI together):
  ```bash
  cd ~/distributed-system/hands-free/Hands-Free && ollama serve &> /tmp/ollama.log & sleep 3 && source venv/bin/activate && set -a && source .env && set +a && PYTHONPATH=. python scripts/start_llm_server.py
  ```

### WSL2 (Fish Speech TTS)
- Runs on Windows machine inside WSL2
- Accessible from Windows at `localhost:8001`
- Install path: `~/fish-speech-wsl/fish-speech`
- Start command:
  ```bash
  source ~/fish-speech-wsl/venv/bin/activate && cd ~/fish-speech-wsl/fish-speech && python -m tools.api_server --listen 0.0.0.0:8001
  ```

## Startup Order
1. **Mac**: start LLM server
2. **WSL2**: start Fish Speech (command above)
3. **Windows**: `venv\Scripts\python scripts/start.py`

## Audio
- Input device: follows Windows default input (Sound Settings → Input)
- No hardcoded device — set default in Windows and pipeline follows it
- AirPods work when set as default input (activates HFP profile)
- Built-in mic: Intel SST Microphone Array (appears after restart if missing)

## Key env vars (.env)
- `HF_LLM_SERVICE_URL=http://192.168.137.28:8080`
- `HF_FISH_SPEECH_URL=http://localhost:8001`
- `HF_LLM_MODEL_NAME=gemma4:e4b`
- `HF_AUDIO_INPUT_DEVICE=` (empty = use Windows default)
