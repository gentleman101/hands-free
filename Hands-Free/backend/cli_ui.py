"""
CLI status display for the live mic loop.

Renders a single status line that updates in place as the pipeline moves
through stages. History (transcripts + responses) is printed above the
status line so the terminal stays readable.

Stages:
    listening   — mic open, no speech
    capturing   — speech detected, accumulating
    transcribing — STT running
    thinking    — LLM streaming
    speaking    — TTS streaming + playback
    idle        — between turns
"""
from __future__ import annotations

import sys
import threading
import time

# ANSI codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CLEAR_LINE = "\r\033[K"

_COLOR = {
    "listening":    "\033[96m",  # cyan
    "capturing":    "\033[93m",  # yellow
    "transcribing": "\033[94m",  # blue
    "thinking":     "\033[95m",  # magenta
    "speaking":     "\033[92m",  # green
    "idle":         "\033[90m",  # gray
    "error":        "\033[91m",  # red
}

_SPINNER = "|/-\\"


class StatusUI:
    """
    Single-line status indicator with a background spinner thread.

    Thread-safe: stage()/info() can be called from any context; the spinner
    redraws at ~10 Hz to keep the line visible.
    """

    def __init__(self) -> None:
        self._stage = "idle"
        self._detail = ""
        self._started = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0

    # ── public API ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None
        # Clear line on exit
        sys.stdout.write(_CLEAR_LINE)
        sys.stdout.flush()

    def stage(self, stage: str, detail: str = "") -> None:
        """Set the current stage. Resets the per-stage timer."""
        with self._lock:
            self._stage = stage
            self._detail = detail
            self._started = time.perf_counter()

    def info(self, line: str) -> None:
        """Print a permanent line above the status (transcript, response, etc)."""
        with self._lock:
            sys.stdout.write(_CLEAR_LINE)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._draw()

    def banner(self, title: str) -> None:
        """Print a separator + title for a new turn."""
        with self._lock:
            sys.stdout.write(_CLEAR_LINE)
            sys.stdout.write(f"\n{_DIM}{'-' * 60}{_RESET}\n")
            sys.stdout.write(f"{_BOLD}{title}{_RESET}\n")
            sys.stdout.flush()
            self._draw()

    # ── internals ───────────────────────────────────────────────────────────
    def _draw(self) -> None:
        with_lock = True  # caller holds lock
        stage = self._stage
        detail = self._detail
        elapsed = time.perf_counter() - self._started
        spinner = _SPINNER[self._frame % len(_SPINNER)]
        color = _COLOR.get(stage, _COLOR["idle"])

        # active stages get spinner; idle gets static dot
        marker = spinner if stage not in ("idle", "error") else "·"

        label = stage.upper()
        if detail:
            text = f"  {color}{marker} {label}{_RESET}  {_DIM}{detail}{_RESET}  {_DIM}{elapsed:5.1f}s{_RESET}"
        else:
            text = f"  {color}{marker} {label}{_RESET}  {_DIM}{elapsed:5.1f}s{_RESET}"

        sys.stdout.write(_CLEAR_LINE + text)
        sys.stdout.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._draw()
                self._frame += 1
            time.sleep(0.1)


# Module-level singleton for convenience
_ui: StatusUI | None = None


def get() -> StatusUI:
    global _ui
    if _ui is None:
        _ui = StatusUI()
    return _ui
