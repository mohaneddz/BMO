import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from constants import settings
from main import app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    # Keep uploads, chat logs and generated audio out of the repo.
    for name in ("ASSETS_STT_DIR", "ASSETS_CV_DIR", "ASSETS_LLM_DIR", "TTS_LOCAL_OUT_DIR"):
        target = tmp_path / name.lower()
        target.mkdir()
        monkeypatch.setattr(settings, name, target)
    return TestClient(app)


def api(module_prefix: str, route: str) -> str:
    return f"{settings.API_PREFIX}{module_prefix}{route}"


def make_wav_bytes(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()
