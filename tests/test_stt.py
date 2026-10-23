from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

import STT.core as stt_core
from conftest import api, make_wav_bytes
from constants import settings


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, path, language=None, beam_size=None, temperature=None):
        self.calls.append({"path": path, "language": language, "beam_size": beam_size})
        segments = iter(
            [
                SimpleNamespace(start=0.0, end=0.4, text=" Hello from"),
                SimpleNamespace(start=0.4, end=0.9, text=" BMO. "),
            ]
        )
        info = SimpleNamespace(language=language or "en", language_probability=0.98, duration=0.9)
        return segments, info


def test_transcribe_returns_model_output(client, monkeypatch):
    fake = FakeWhisperModel()
    monkeypatch.setattr(stt_core, "_get_model", lambda: fake)

    response = client.post(
        api(settings.STT_ROUTE_PREFIX, settings.STT_TRANSCRIBE_ROUTE),
        files={"file": ("hello.wav", make_wav_bytes(), "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["transcription"] == "Hello from BMO."
    assert [s["text"] for s in body["segments"]] == ["Hello from", "BMO."]
    assert body["language"] == "en"
    assert body["duration_seconds"] == 0.9
    assert Path(body["saved_to"]).read_bytes()[:4] == b"RIFF"
    assert fake.calls[0]["path"] == body["saved_to"]
    assert fake.calls[0]["beam_size"] == settings.STT_BEAM_SIZE


def test_auto_language_lets_whisper_detect(client, monkeypatch):
    fake = FakeWhisperModel()
    monkeypatch.setattr(stt_core, "_get_model", lambda: fake)

    response = client.post(
        api(settings.STT_ROUTE_PREFIX, settings.STT_TRANSCRIBE_ROUTE),
        files={"file": ("hello.wav", make_wav_bytes(), "audio/wav")},
        data={"language": "auto"},
    )

    assert response.status_code == 200, response.text
    assert fake.calls[0]["language"] is None
    assert response.json()["language_probability"] == 0.98


def test_empty_upload_is_rejected(client):
    response = client.post(
        api(settings.STT_ROUTE_PREFIX, settings.STT_TRANSCRIBE_ROUTE),
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_model_load_failure_is_explicit(client, monkeypatch):
    def unavailable():
        raise HTTPException(status_code=503, detail="Could not load Whisper model 'base'.")

    monkeypatch.setattr(stt_core, "_get_model", unavailable)

    response = client.post(
        api(settings.STT_ROUTE_PREFIX, settings.STT_TRANSCRIBE_ROUTE),
        files={"file": ("hello.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 503
    assert "Whisper model" in response.json()["detail"]
