import numpy as np
import pytest
import soundfile as sf

import TTS.local_pipeline as tts_pipeline
from conftest import api
from constants import settings


class FakeEngine:
    sample_rate = 24000
    load_seconds = 0.0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize_to_numpy(self, text, clip_path):
        self.calls.append({"text": text, "clip": clip_path.name})
        audio = np.sin(np.linspace(0, 200, self.sample_rate // 4)).astype(np.float32) * 0.2
        return audio, {"state_prepare_s": 0.0, "inference_s": 0.01, "max_tokens": 120.0}


@pytest.fixture
def engine(monkeypatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr(tts_pipeline, "_get_engine", lambda: fake)
    return fake


def test_synthesize_writes_wav_and_serves_it(client, engine):
    response = client.post(
        api(settings.TTS_ROUTE_PREFIX, settings.TTS_SYNTHESIZE_ROUTE),
        json={"text": "  BMO says hello  ", "mode": "medium"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "medium"
    assert body["voice"] == settings.TTS_POCKET_CLIP_MEDIUM_PATH.name
    assert engine.calls == [{"text": "BMO says hello", "clip": settings.TTS_POCKET_CLIP_MEDIUM_PATH.name}]

    audio, sample_rate = sf.read(body["file_path"])
    assert sample_rate == FakeEngine.sample_rate
    assert len(audio) == FakeEngine.sample_rate // 4

    download = client.get(body["file_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "audio/wav"
    assert download.content[:4] == b"RIFF"


def test_voice_selects_sample_by_number(client, engine):
    response = client.post(
        api(settings.TTS_ROUTE_PREFIX, settings.TTS_SYNTHESIZE_ROUTE),
        json={"text": "Who wants to play video games?", "voice": "4"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["voice"] == "4.wav"
    assert engine.calls[0]["clip"] == "4.wav"


def test_unknown_voice_is_a_client_error(client, engine):
    response = client.post(
        api(settings.TTS_ROUTE_PREFIX, settings.TTS_SYNTHESIZE_ROUTE),
        json={"text": "Hello", "voice": "99"},
    )
    assert response.status_code == 422
    assert "99" in response.json()["detail"]
    assert engine.calls == []


def test_engine_failure_is_explicit(client, monkeypatch):
    def unavailable():
        raise tts_pipeline.LocalTTSPipelineError("Could not load PocketTTS model config 'b6369a24'.")

    monkeypatch.setattr(tts_pipeline, "_get_engine", unavailable)

    response = client.post(
        api(settings.TTS_ROUTE_PREFIX, settings.TTS_SYNTHESIZE_ROUTE),
        json={"text": "Hello"},
    )
    assert response.status_code == 503
    assert "PocketTTS" in response.json()["detail"]


def test_download_rejects_paths_outside_output_dir(client):
    response = client.get(api(settings.TTS_ROUTE_PREFIX, "/files/..%2F..%2Fmain.py"))
    assert response.status_code == 404
