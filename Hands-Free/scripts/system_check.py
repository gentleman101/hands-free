"""
Systematic preflight check — run before scripts/start.py.

Validates every component the pipeline depends on, with exit codes:
  0 — all green
  1 — configuration error (fix and rerun)
  2 — dependency service unreachable (start dependency)
  3 — soft warnings only

Usage:
  venv\\Scripts\\python scripts/system_check.py
  venv\\Scripts\\python scripts/system_check.py --quick  (skip sample calls)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

# Project root on path + load .env
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")


# ── Output helpers ──────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_results: list[tuple[str, str, str]] = []  # (status, name, detail)


def ok(name: str, detail: str = "") -> None:
    _results.append(("ok", name, detail))
    print(f"  {_GREEN}[OK]{_RESET}    {name}{(' — ' + detail) if detail else ''}")


def warn(name: str, detail: str = "") -> None:
    _results.append(("warn", name, detail))
    print(f"  {_YELLOW}[WARN]{_RESET}  {name}{(' — ' + detail) if detail else ''}")


def fail(name: str, detail: str = "") -> None:
    _results.append(("fail", name, detail))
    print(f"  {_RED}[FAIL]{_RESET}  {name}{(' — ' + detail) if detail else ''}")


def header(text: str) -> None:
    print(f"\n{_DIM}-- {text} --{_RESET}")


# ── Checks ──────────────────────────────────────────────────────────────────
def check_python_version() -> None:
    v = sys.version_info
    if v.major == 3 and v.minor == 11:
        ok("Python version", f"{v.major}.{v.minor}.{v.micro}")
    elif v.major == 3 and v.minor in (10, 12):
        warn("Python version", f"{v.major}.{v.minor} (3.11 recommended)")
    else:
        fail("Python version", f"{v.major}.{v.minor} (need 3.11)")


def check_venv() -> None:
    if sys.prefix != sys.base_prefix:
        ok("venv active", sys.prefix)
    else:
        warn("venv active", "running in system Python — activate venv first")


def check_env_file() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        ok(".env present", str(env_path.relative_to(ROOT)))
    else:
        warn(".env present", "using built-in defaults; copy .env.example → .env")


def check_machine_role() -> str | None:
    role = os.environ.get("HF_MACHINE_ROLE", "")
    if role == "client":
        ok("HF_MACHINE_ROLE", "client (Windows)")
        return role
    if role == "server":
        ok("HF_MACHINE_ROLE", "server (Mac Mini)")
        return role
    fail("HF_MACHINE_ROLE", f"must be 'client' or 'server' (got '{role}')")
    return None


def check_required_env_vars(role: str | None) -> None:
    if role == "client":
        required = ["HF_LLM_SERVICE_URL", "HF_FISH_SPEECH_URL"]
    elif role == "server":
        required = ["HF_LLM_MODEL_NAME", "HF_LLM_PORT"]
    else:
        return
    for key in required:
        val = os.environ.get(key, "")
        if val:
            ok(key, val)
        else:
            fail(key, "not set")


def check_packages(role: str | None) -> None:
    shared = ["fastapi", "uvicorn", "httpx", "pydantic", "aiosqlite", "dotenv"]
    client_only = ["faster_whisper", "sounddevice", "soundfile", "numpy",
                   "sklearn", "ormsgpack", "ten_vad"]
    server_only = ["llama_cpp"]

    pkgs = shared + (client_only if role == "client" else server_only if role == "server" else [])

    for name in pkgs:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            ok(f"pkg {name}", version)
        except ImportError:
            soft = name in {"ten_vad", "llama_cpp"}
            (warn if soft else fail)(f"pkg {name}", "not installed")


def check_input_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        warn("input devices", "sounddevice not installed")
        return
    try:
        devs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
        if not devs:
            fail("input devices", "none found")
            return
        # Try opening at least one
        for i, name in devs:
            try:
                with sd.InputStream(device=i, channels=1, samplerate=16000):
                    pass
                ok("input devices", f"{len(devs)} found, device {i} openable: {name[:50]}")
                return
            except Exception:
                continue
        fail("input devices", f"{len(devs)} found, none openable")
    except Exception as e:
        fail("input devices", str(e))


def check_filler_wavs() -> None:
    fillers = ROOT / "assets" / "fillers"
    if not fillers.exists():
        warn("filler WAVs", "directory missing; run scripts/generate_fillers.py")
        return
    count = sum(1 for _ in fillers.rglob("*.wav"))
    expected = 84  # 7 intents × 2 langs × 6 variants
    if count >= expected:
        ok("filler WAVs", f"{count} files")
    else:
        warn("filler WAVs", f"only {count}/{expected} — regenerate via scripts/generate_fillers.py")


def check_reference_voices() -> None:
    voices = ROOT / "assets" / "voices"
    en = voices / "brian_en.wav"
    ar = voices / "hamdan_ar.wav"
    if en.exists() and ar.exists():
        ok("reference voices", "brian_en.wav + hamdan_ar.wav")
    else:
        missing = [p.name for p in (en, ar) if not p.exists()]
        warn("reference voices", f"missing: {', '.join(missing)}")


def check_cuda() -> None:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            ok("CUDA (CTranslate2)", f"{ctranslate2.get_cuda_device_count()} device(s)")
        else:
            warn("CUDA (CTranslate2)", "no CUDA device — STT will run on CPU (slow)")
    except ImportError:
        warn("CUDA", "ctranslate2 not installed")
    except Exception as e:
        warn("CUDA", str(e))


async def check_fish_speech() -> bool:
    url = os.environ.get("HF_FISH_SPEECH_URL", "http://localhost:8001")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{url}/v1/health")
        if r.status_code == 200:
            ok("Fish Speech reachable", url)
            return True
        fail("Fish Speech reachable", f"{url} returned {r.status_code}")
    except Exception as e:
        fail("Fish Speech reachable", f"{url} — {str(e)[:60]}")
    return False


async def check_llm_server() -> bool:
    url = os.environ.get("HF_LLM_SERVICE_URL", "http://localhost:8080")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/healthz")
        if r.status_code == 200:
            ok("LLM server reachable", f"{url} — {r.json().get('model', '?')}")
            # readyz check
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    rz = await c.get(f"{url}/readyz")
                if rz.status_code == 200:
                    ok("LLM model loaded", rz.json().get("model", "?"))
                else:
                    warn("LLM model loaded", f"readyz returned {rz.status_code}")
            except Exception as e:
                warn("LLM model loaded", str(e)[:60])
            return True
        fail("LLM server reachable", f"{url} returned {r.status_code}")
    except Exception as e:
        fail("LLM server reachable", f"{url} — {str(e)[:60]}")
    return False


async def sample_llm_call() -> None:
    url = os.environ.get("HF_LLM_SERVICE_URL", "")
    if not url:
        return
    try:
        import httpx
        t0 = time.perf_counter()
        ttft = None
        async with httpx.AsyncClient(timeout=30.0) as c:
            async with c.stream(
                "POST",
                f"{url}/api/llm/generate",
                json={"messages": [{"role": "user", "content": "Say hi."}], "language": "en"},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.strip():
                        ttft = int((time.perf_counter() - t0) * 1000)
                        break
        total = int((time.perf_counter() - t0) * 1000)
        if ttft is not None and ttft < 3000:
            ok("sample LLM call", f"TTFT {ttft} ms, header {total} ms")
        else:
            warn("sample LLM call", f"TTFT {ttft} ms (>3 s)")
    except Exception as e:
        warn("sample LLM call", str(e)[:60])


async def sample_tts_call() -> None:
    url = os.environ.get("HF_FISH_SPEECH_URL", "")
    if not url:
        return
    try:
        import httpx
        t0 = time.perf_counter()
        ttfa = None
        async with httpx.AsyncClient(timeout=60.0) as c:
            async with c.stream(
                "POST",
                f"{url}/v1/tts",
                json={"text": "Hello.", "streaming": True, "format": "wav",
                      "chunk_length": 200, "normalize": True},
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(chunk_size=4096):
                    if chunk:
                        ttfa = int((time.perf_counter() - t0) * 1000)
                        break
        if ttfa is not None and ttfa < 3500:
            ok("sample TTS call", f"TTFA {ttfa} ms")
        else:
            warn("sample TTS call", f"TTFA {ttfa} ms (>3.5 s)")
    except Exception as e:
        warn("sample TTS call", str(e)[:60])


# ── Main ────────────────────────────────────────────────────────────────────
async def run(quick: bool) -> int:
    print(f"\n{_DIM}Hands-Free preflight — {ROOT}{_RESET}")

    header("Environment")
    check_python_version()
    check_venv()
    check_env_file()
    role = check_machine_role()
    check_required_env_vars(role)

    header("Packages")
    check_packages(role)

    if role == "client":
        header("Hardware / Assets")
        check_input_devices()
        check_cuda()
        check_filler_wavs()
        check_reference_voices()

        header("Services")
        fish_ok = await check_fish_speech()
        llm_ok = await check_llm_server()

        if not quick and fish_ok and llm_ok:
            header("Sample round-trip")
            await sample_llm_call()
            await sample_tts_call()

    header("Summary")
    fails = sum(1 for r in _results if r[0] == "fail")
    warns = sum(1 for r in _results if r[0] == "warn")
    oks = sum(1 for r in _results if r[0] == "ok")
    print(f"  {_GREEN}{oks} OK{_RESET}, {_YELLOW}{warns} warn{_RESET}, {_RED}{fails} fail{_RESET}\n")

    if fails:
        config_fails = any("HF_" in r[1] or "Python" in r[1] or "pkg" in r[1]
                           for r in _results if r[0] == "fail")
        service_fails = any("reachable" in r[1] or "loaded" in r[1] or "devices" in r[1]
                            for r in _results if r[0] == "fail")
        if service_fails and not config_fails:
            return 2
        return 1
    if warns:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hands-Free preflight check")
    parser.add_argument("--quick", action="store_true",
                        help="Skip sample round-trip calls")
    args = parser.parse_args()
    return asyncio.run(run(quick=args.quick))


if __name__ == "__main__":
    sys.exit(main())
