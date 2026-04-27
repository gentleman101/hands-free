"""Tests for backend/services/audio_capture.py — SentenceBuffer + VAD utils"""
import io
import wave

import numpy as np

from backend.services.audio_capture import _pcm_to_wav, _SAMPLE_RATE


class TestPcmToWav:
    def test_produces_valid_wav(self) -> None:
        pcm = np.zeros(1600, dtype=np.float32)
        wav_bytes = _pcm_to_wav(pcm)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == _SAMPLE_RATE
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 1600

    def test_non_zero_audio(self) -> None:
        pcm = np.ones(1600, dtype=np.float32) * 0.5
        wav_bytes = _pcm_to_wav(pcm)
        assert len(wav_bytes) > 0
        # WAV header is 44 bytes; audio follows
        assert len(wav_bytes) > 44

    def test_round_trip_amplitude(self) -> None:
        """PCM values should survive WAV encode/decode approximately."""
        import soundfile as sf  # type: ignore[import-untyped]

        pcm = np.sin(np.linspace(0, 2 * np.pi, 1600)).astype(np.float32)
        wav_bytes = _pcm_to_wav(pcm)
        buf = io.BytesIO(wav_bytes)
        recovered, sr = sf.read(buf, dtype="float32")
        assert sr == _SAMPLE_RATE
        np.testing.assert_allclose(recovered, pcm, atol=1e-3)
