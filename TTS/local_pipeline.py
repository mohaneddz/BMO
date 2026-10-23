from __future__ import annotations

import io
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import soundfile as sf

from constants import settings


PipelineMode = Literal["small", "medium", "large"]

MODE_SETTINGS: dict[PipelineMode, dict[str, str]] = {
    "small": {"clip": "small"},
    "medium": {"clip": "medium"},
    "large": {"clip": "large"},
}


class LocalTTSPipelineError(RuntimeError):
    pass


class LocalTTSInputError(LocalTTSPipelineError):
    pass


@dataclass(slots=True)
class LocalTTSPipelineResult:
    text: str
    mode: PipelineMode
    clip_path: Path
    output_wav_path: Path
    output_format: str = "wav"
    pipeline: str = "pocket-tts"
    model_name: str = "pocket-tts"
    timings_seconds: dict[str, float] | None = None


@dataclass(slots=True)
class LocalTTSPlaybackResult:
    text: str
    mode: PipelineMode
    clip_path: Path
    pipeline: str
    output_format: str
    wav_bytes: bytes
    sample_rate: int
    model_name: str = "pocket-tts"
    timings_seconds: dict[str, float] | None = None


def list_voice_samples() -> list[Path]:
    voice_dir = settings.TTS_POCKET_VOICE_DIR
    if not voice_dir.exists():
        return []
    return sorted([p for p in voice_dir.glob("*.wav") if p.is_file()])


class _WarmLocalTTSEngine:
    def __init__(self) -> None:
        load_started = time.perf_counter()
        try:
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise LocalTTSPipelineError(
                "TTS needs pocket-tts and torch. Install them with `pip install -r requirements.txt`."
            ) from exc

        torch.set_num_threads(settings.TTS_POCKET_TORCH_THREADS)
        self._lock = threading.Lock()
        try:
            self.model = TTSModel.load_model(
                config=settings.TTS_POCKET_MODEL_CONFIG,
                temp=settings.TTS_POCKET_TEMPERATURE,
                lsd_decode_steps=settings.TTS_POCKET_LSD_DECODE_STEPS,
                noise_clamp=settings.TTS_POCKET_NOISE_CLAMP,
                eos_threshold=settings.TTS_POCKET_EOS_THRESHOLD,
            )
        except Exception as exc:
            raise LocalTTSPipelineError(
                f"Could not load PocketTTS model config '{settings.TTS_POCKET_MODEL_CONFIG}'. "
                "The first run downloads the weights from https://huggingface.co/kyutai/pocket-tts, "
                f"which needs internet access and `hf auth login`. Cause: {exc}"
            ) from exc
        self.sample_rate = int(self.model.sample_rate)
        self.states: dict[str, dict] = {}
        self.load_seconds = time.perf_counter() - load_started

    def _resolve_max_tokens(self, text: str) -> int:
        if not settings.TTS_POCKET_DYNAMIC_MAX_TOKENS:
            return settings.TTS_POCKET_MAX_TOKENS
        estimated = int(len(text) * settings.TTS_POCKET_TOKENS_PER_CHAR)
        return max(settings.TTS_POCKET_MIN_TOKENS, min(settings.TTS_POCKET_MAX_TOKENS, estimated))

    def warm_states(self) -> float:
        warm_started = time.perf_counter()
        for mode in MODE_SETTINGS:
            self._get_or_create_state(get_clip_path_for_mode(mode))
        return time.perf_counter() - warm_started

    def _get_or_create_state(self, clip_path: Path) -> dict:
        key = str(clip_path.resolve())
        existing = self.states.get(key)
        if existing is not None:
            return existing

        try:
            state = self.model.get_state_for_audio_prompt(str(clip_path))
        except ValueError as exc:
            if "voice cloning" in str(exc).lower():
                raise LocalTTSPipelineError(
                    "PocketTTS voice cloning is not enabled in this environment. "
                    "Accept terms on https://huggingface.co/kyutai/pocket-tts and run `hf auth login`, "
                    "then rerun synthesis."
                ) from exc
            raise
        self.states[key] = state
        return state

    def synthesize_to_numpy(self, text: str, clip_path: Path) -> tuple[np.ndarray, dict[str, float]]:
        with self._lock:
            state_started = time.perf_counter()
            state = self._get_or_create_state(clip_path)
            state_ended = time.perf_counter()
            max_tokens = self._resolve_max_tokens(text)

            gen_started = time.perf_counter()
            audio_tensor = self.model.generate_audio(
                model_state=state,
                text_to_generate=text,
                max_tokens=max_tokens,
                frames_after_eos=settings.TTS_POCKET_FRAMES_AFTER_EOS,
                copy_state=True,
            )
            gen_ended = time.perf_counter()

        audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if audio_np.size == 0:
            raise LocalTTSPipelineError("PocketTTS returned empty audio.")

        return audio_np, {
            "state_prepare_s": state_ended - state_started,
            "inference_s": gen_ended - gen_started,
            "max_tokens": float(max_tokens),
        }


_ENGINE_LOCK = threading.Lock()
_ENGINE: _WarmLocalTTSEngine | None = None


def resolve_mode(mode: str | None) -> PipelineMode:
    resolved = (mode or settings.TTS_LOCAL_DEFAULT_MODE).strip().lower()
    if resolved not in MODE_SETTINGS:
        raise LocalTTSInputError(
            f"Invalid TTS mode '{resolved}'. Expected one of: {', '.join(MODE_SETTINGS)}."
        )
    return resolved  # type: ignore[return-value]


def load_text_source(text_or_path: str | None) -> str:
    if text_or_path is None:
        text = settings.TTS_LOCAL_DEFAULT_TEXT
    else:
        candidate = Path(text_or_path)
        if candidate.exists() and candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
        else:
            text = text_or_path

    normalized = text.strip()
    if not normalized:
        raise LocalTTSPipelineError("Input text is empty after trimming whitespace.")
    return normalized


def _find_voice_sample_by_label(label: str) -> Path | None:
    normalized = label.strip().lower()
    if not normalized:
        return None

    samples = list_voice_samples()
    if not samples:
        return None

    numeric_target: int | None = None
    if normalized.startswith("voice-"):
        normalized = normalized[6:]
    if normalized.endswith(".wav"):
        normalized = normalized[:-4]
    if normalized.isdigit():
        numeric_target = int(normalized)

    for path in samples:
        stem_lower = path.stem.lower()
        name_lower = path.name.lower()
        if normalized in {stem_lower, name_lower}:
            return path
        if numeric_target is not None and stem_lower.isdigit() and int(stem_lower) == numeric_target:
            return path
    return None


def get_clip_path_for_mode(mode: PipelineMode, voice: str | None = None) -> Path:
    if voice:
        matched = _find_voice_sample_by_label(voice)
        if matched is None:
            raise LocalTTSInputError(
                f"Voice sample '{voice}' was not found in {settings.TTS_POCKET_VOICE_DIR}. "
                "Use a wav filename, stem, or number (for example: 1, 2, 3)."
            )
        return matched

    clip_map = {
        "small": settings.TTS_POCKET_CLIP_SMALL_PATH,
        "medium": settings.TTS_POCKET_CLIP_MEDIUM_PATH,
        "large": settings.TTS_POCKET_CLIP_LARGE_PATH,
    }
    clip_path = clip_map[mode]
    if not clip_path.exists():
        raise LocalTTSPipelineError(
            f"PocketTTS clip file not found for mode '{mode}': {clip_path}. "
            "The clips ship in TTS/assets/voice; restore them with `git checkout -- TTS/assets/voice` "
            "or point TTS_POCKET_CLIP_*_PATH in .env at your own wav files."
        )
    return clip_path


def _build_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = uuid4().hex[:8]
    output_path = settings.TTS_LOCAL_OUT_DIR / f"tts_pocket_{timestamp}_{token}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _get_engine() -> _WarmLocalTTSEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _WarmLocalTTSEngine()
        return _ENGINE


def _audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    scaled = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (scaled * 32767.0).astype(np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return wav_buffer.getvalue()


def prewarm_local_tts_engine() -> float:
    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    for mode in MODE_SETTINGS:
        get_clip_path_for_mode(mode)

    warm_started = time.perf_counter()
    engine = _get_engine()
    state_warm_seconds = engine.warm_states()
    return (time.perf_counter() - warm_started) + state_warm_seconds


def run_local_tts_playback(
    text: str,
    mode: str | None = None,
    voice: str | None = None,
    *,
    realtime: bool = False,
) -> LocalTTSPlaybackResult:
    del realtime

    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    total_started = time.perf_counter()
    resolved_mode = resolve_mode(mode)
    clip_path = get_clip_path_for_mode(resolved_mode, voice=voice)

    engine_prepare_started = time.perf_counter()
    engine = _get_engine()
    engine_prepare_seconds = time.perf_counter() - engine_prepare_started

    audio_np, timings_seconds = engine.synthesize_to_numpy(text=text, clip_path=clip_path)
    wav_bytes = _audio_to_wav_bytes(audio_np, engine.sample_rate)

    timings_seconds["engine_prepare_s"] = engine_prepare_seconds
    timings_seconds["engine_load_s"] = engine.load_seconds
    timings_seconds["total_s"] = time.perf_counter() - total_started

    return LocalTTSPlaybackResult(
        text=text,
        mode=resolved_mode,
        clip_path=clip_path,
        pipeline="pocket-tts",
        output_format="wav",
        wav_bytes=wav_bytes,
        sample_rate=engine.sample_rate,
        timings_seconds=timings_seconds,
    )


def run_local_tts_pipeline(
    text: str,
    mode: str | None = None,
    voice: str | None = None,
) -> LocalTTSPipelineResult:
    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    total_started = time.perf_counter()
    resolved_mode = resolve_mode(mode)
    clip_path = get_clip_path_for_mode(resolved_mode, voice=voice)

    output_wav_path = _build_output_path()

    engine_prepare_started = time.perf_counter()
    engine = _get_engine()
    engine_prepare_seconds = time.perf_counter() - engine_prepare_started

    audio_np, timings_seconds = engine.synthesize_to_numpy(text=text, clip_path=clip_path)
    sf.write(output_wav_path, audio_np, samplerate=engine.sample_rate)

    timings_seconds["engine_prepare_s"] = engine_prepare_seconds
    timings_seconds["engine_load_s"] = engine.load_seconds
    timings_seconds["total_s"] = time.perf_counter() - total_started

    return LocalTTSPipelineResult(
        text=text,
        mode=resolved_mode,
        clip_path=clip_path,
        output_wav_path=output_wav_path,
        model_name=settings.TTS_MODEL_NAME,
        timings_seconds=timings_seconds,
    )
