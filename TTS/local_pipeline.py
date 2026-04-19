from __future__ import annotations

import importlib.util
import io
import numpy as np
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from constants import settings


PipelineMode = Literal["small", "medium", "large"]

MODE_SETTINGS: dict[PipelineMode, dict[str, str]] = {
    "small": {
        "method": "pm",
        "index_rate": "0.5",
        "filter_radius": "3",
        "protect": "0.3",
    },
    "medium": {
        "method": "harvest",
        "index_rate": "0.7",
        "filter_radius": "5",
        "protect": "0.4",
    },
    "large": {
        "method": "rmvpe",
        "index_rate": "0.8",
        "filter_radius": "7",
        "protect": "0.5",
    },
}

REALTIME_MODE_SETTINGS: dict[str, str] = {
    "method": "pm",
    "index_rate": "0.0",
    "filter_radius": "1",
    "protect": "0.0",
}


class LocalTTSPipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class LocalTTSPipelineResult:
    text: str
    mode: PipelineMode
    piper_model: str
    piper_model_path: Path
    piper_config_path: Path
    rvc_model_path: Path
    rvc_index_path: Path
    temp_wav_path: Path
    output_wav_path: Path
    output_format: str = "wav"
    pipeline: str = "piper-rvc"
    timings_seconds: dict[str, float] | None = None


@dataclass(slots=True)
class LocalTTSPlaybackResult:
    text: str
    mode: PipelineMode
    pipeline: str
    output_format: str
    wav_bytes: bytes
    sample_rate: int
    timings_seconds: dict[str, float] | None = None


class _WarmLocalTTSEngine:
    def __init__(
        self,
        piper_model_path: Path,
        piper_config_path: Path,
        rvc_model_path: Path,
        rvc_index_path: Path,
        device: str,
    ) -> None:
        load_started = time.perf_counter()
        from piper.voice import PiperVoice
        from rvc_python.infer import RVCInference

        self._lock = threading.Lock()
        self.voice = PiperVoice.load(
            model_path=piper_model_path,
            config_path=piper_config_path,
            use_cuda=device.startswith("cuda"),
        )
        self.rvc = RVCInference(
            device=device,
            model_path=str(rvc_model_path),
            index_path=str(rvc_index_path),
        )
        self.load_seconds = time.perf_counter() - load_started

    def _resolve_mode_settings(self, mode: PipelineMode, realtime: bool) -> dict[str, str]:
        if realtime:
            return REALTIME_MODE_SETTINGS
        return MODE_SETTINGS[mode]

    @staticmethod
    def _trim_pcm16_silence(
        pcm16: np.ndarray,
        sample_rate: int,
        threshold: int = 280,
        pad_ms: int = 40,
    ) -> np.ndarray:
        if pcm16.size == 0:
            return pcm16

        non_silent = np.flatnonzero(np.abs(pcm16) > threshold)
        if non_silent.size == 0:
            return pcm16

        pad_samples = int(sample_rate * pad_ms / 1000)
        start = max(0, int(non_silent[0]) - pad_samples)
        end = min(pcm16.size, int(non_silent[-1]) + pad_samples + 1)
        return pcm16[start:end]

    def _synthesize_piper_pcm16(self, text: str, *, trim_silence: bool = False) -> tuple[int, np.ndarray]:
        sample_rate: int | None = None
        pcm_chunks: list[bytes] = []
        for audio_chunk in self.voice.synthesize(text):
            if sample_rate is None:
                sample_rate = int(audio_chunk.sample_rate)
            pcm_chunks.append(audio_chunk.audio_int16_bytes)

        if not pcm_chunks or sample_rate is None:
            raise LocalTTSPipelineError("Piper synthesis did not produce audio.")

        pcm16 = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
        if trim_silence:
            pcm16 = self._trim_pcm16_silence(pcm16, sample_rate)
        return sample_rate, pcm16

    def _rvc_convert_from_audio(
        self,
        input_audio_path: str,
        mode_settings: dict[str, str],
    ) -> tuple[int, np.ndarray]:
        self.rvc.set_params(
            f0method=mode_settings["method"],
            index_rate=float(mode_settings["index_rate"]),
            filter_radius=int(mode_settings["filter_radius"]),
            protect=float(mode_settings["protect"]),
        )

        file_index = ""
        if float(mode_settings["index_rate"]) > 0 and self.rvc.current_model:
            file_index = self.rvc.models.get(self.rvc.current_model, {}).get("index", "") or ""

        wav_opt = self.rvc.vc.vc_single(
            sid=0,
            input_audio_path=input_audio_path,
            f0_up_key=self.rvc.f0up_key,
            f0_file="",
            f0_method=self.rvc.f0method,
            file_index=file_index,
            file_index2="",
            index_rate=self.rvc.index_rate,
            filter_radius=self.rvc.filter_radius,
            resample_sr=self.rvc.resample_sr,
            rms_mix_rate=self.rvc.rms_mix_rate,
            protect=self.rvc.protect,
        )

        if not isinstance(wav_opt, np.ndarray):
            raise LocalTTSPipelineError(f"RVC conversion failed: {wav_opt}")

        target_sr = int(self.rvc.vc.tgt_sr)
        return target_sr, wav_opt.astype(np.int16, copy=False)

    @staticmethod
    def _pcm16_to_wav_bytes(sample_rate: int, pcm16: np.ndarray) -> bytes:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return wav_buffer.getvalue()

    def synthesize(
        self,
        text: str,
        mode: PipelineMode,
        temp_wav_path: Path,
        output_wav_path: Path,
    ) -> dict[str, float]:
        mode_settings = MODE_SETTINGS[mode]

        with self._lock:
            piper_started = time.perf_counter()
            with wave.open(str(temp_wav_path), "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            piper_ended = time.perf_counter()

            self.rvc.set_params(
                f0method=mode_settings["method"],
                index_rate=float(mode_settings["index_rate"]),
                filter_radius=int(mode_settings["filter_radius"]),
                protect=float(mode_settings["protect"]),
            )
            self.rvc.infer_file(str(temp_wav_path), str(output_wav_path))
            rvc_ended = time.perf_counter()

        return {
            "piper_s": piper_ended - piper_started,
            "rvc_s": rvc_ended - piper_ended,
            "inference_s": rvc_ended - piper_started,
        }

    def synthesize_to_wav_bytes(
        self,
        text: str,
        mode: PipelineMode,
        *,
        realtime: bool = False,
        temp_wav_path: Path | None = None,
    ) -> tuple[bytes, dict[str, float], int]:
        mode_settings = self._resolve_mode_settings(mode, realtime)

        with self._lock:
            piper_started = time.perf_counter()
            piper_sample_rate, piper_pcm16 = self._synthesize_piper_pcm16(
                text,
                trim_silence=realtime,
            )
            piper_ended = time.perf_counter()

            rvc_started = time.perf_counter()
            if temp_wav_path is None:
                raise LocalTTSPipelineError("temp_wav_path is required in playback mode.")
            temp_wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(temp_wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(piper_sample_rate)
                wav_file.writeframes(piper_pcm16.tobytes())

            output_sample_rate, output_pcm16 = self._rvc_convert_from_audio(
                input_audio_path=str(temp_wav_path),
                mode_settings=mode_settings,
            )
            rvc_ended = time.perf_counter()

            serialization_started = time.perf_counter()
            wav_bytes = self._pcm16_to_wav_bytes(sample_rate=output_sample_rate, pcm16=output_pcm16)
            serialization_ended = time.perf_counter()

        return (
            wav_bytes,
            {
                "piper_s": piper_ended - piper_started,
                "rvc_s": rvc_ended - rvc_started,
                "serialize_s": serialization_ended - serialization_started,
                "inference_s": rvc_ended - piper_started,
            },
            output_sample_rate,
        )


_ENGINE_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple[str, str, str, str, str], _WarmLocalTTSEngine] = {}


def resolve_mode(mode: str | None) -> PipelineMode:
    resolved = (mode or settings.TTS_LOCAL_DEFAULT_MODE).strip().lower()
    if resolved not in MODE_SETTINGS:
        raise LocalTTSPipelineError(
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


def _ensure_runtime_support() -> None:
    missing_modules: list[str] = []
    for module_name in ("piper", "rvc_python"):
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)

    if missing_modules:
        raise LocalTTSPipelineError(
            "Missing runtime dependencies for local TTS pipeline: "
            + ", ".join(missing_modules)
            + ". Install the required packages in the active environment."
        )

    for required_path, label in (
        (settings.TTS_RVC_MODEL_PATH, "RVC model"),
        (settings.TTS_RVC_INDEX_PATH, "RVC index"),
    ):
        if not required_path.exists():
            raise LocalTTSPipelineError(f"{label} file was not found: {required_path}")


def _resolve_piper_model_files() -> tuple[str, Path, Path]:
    configured = settings.TTS_LOCAL_PIPER_MODEL.strip()
    configured_path = Path(configured)

    if configured_path.suffix == ".onnx":
        model_path = configured_path if configured_path.is_absolute() else (Path.cwd() / configured_path)
        config_path = model_path.with_suffix(model_path.suffix + ".json")
        if not model_path.exists():
            raise LocalTTSPipelineError(f"Piper model file was not found: {model_path}")
        if not config_path.exists():
            raise LocalTTSPipelineError(f"Piper config file was not found: {config_path}")
        return configured, model_path, config_path

    voice_id = configured
    download_dir = settings.TTS_LOCAL_PIPER_DOWNLOAD_DIR
    model_path = download_dir / f"{voice_id}.onnx"
    config_path = download_dir / f"{voice_id}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        _run_command(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                voice_id,
                "--download-dir",
                str(download_dir),
            ],
            f"Piper voice download ({voice_id})",
        )

    if not model_path.exists() or not config_path.exists():
        raise LocalTTSPipelineError(
            f"Piper voice download did not produce expected files: {model_path} and {config_path}"
        )

    return voice_id, model_path, config_path


def _run_command(command: list[str], step_name: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        return

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    details = stderr or stdout or "No subprocess output was captured."
    raise LocalTTSPipelineError(f"{step_name} failed with exit code {completed.returncode}: {details}")


def _build_paths() -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = uuid4().hex[:8]
    base_name = f"tts_rvc_{timestamp}_{token}"
    temp_wav_path = settings.TTS_LOCAL_TMP_DIR / f"{base_name}_tts.wav"
    output_wav_path = settings.TTS_LOCAL_OUT_DIR / f"{base_name}.wav"
    temp_wav_path.parent.mkdir(parents=True, exist_ok=True)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    return temp_wav_path, output_wav_path


def _engine_cache_key(
    piper_model_path: Path,
    piper_config_path: Path,
    rvc_model_path: Path,
    rvc_index_path: Path,
    device: str,
) -> tuple[str, str, str, str, str]:
    return (
        str(piper_model_path.resolve()),
        str(piper_config_path.resolve()),
        str(rvc_model_path.resolve()),
        str(rvc_index_path.resolve()),
        device,
    )


def _get_warm_engine(
    piper_model_path: Path,
    piper_config_path: Path,
    rvc_model_path: Path,
    rvc_index_path: Path,
    device: str,
) -> _WarmLocalTTSEngine:
    key = _engine_cache_key(
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        rvc_model_path=rvc_model_path,
        rvc_index_path=rvc_index_path,
        device=device,
    )
    with _ENGINE_CACHE_LOCK:
        existing = _ENGINE_CACHE.get(key)
        if existing is not None:
            return existing

        engine = _WarmLocalTTSEngine(
            piper_model_path=piper_model_path,
            piper_config_path=piper_config_path,
            rvc_model_path=rvc_model_path,
            rvc_index_path=rvc_index_path,
            device=device,
        )
        _ENGINE_CACHE[key] = engine
        return engine


def prewarm_local_tts_engine() -> float:
    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    _ensure_runtime_support()
    _, piper_model_path, piper_config_path = _resolve_piper_model_files()

    warm_started = time.perf_counter()
    engine = _get_warm_engine(
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        rvc_model_path=settings.TTS_RVC_MODEL_PATH,
        rvc_index_path=settings.TTS_RVC_INDEX_PATH,
        device=settings.TTS_RVC_DEVICE,
    )

    # Prime the first RVC inference so the first real line in interactive mode is faster.
    if settings.TTS_LOCAL_PRIME_ON_PREWARM:
        prime_temp_wav_path, _ = _build_paths()
        try:
            engine.synthesize_to_wav_bytes(
                text="BMO is ready.",
                mode="small",
                realtime=True,
                temp_wav_path=prime_temp_wav_path,
            )
        finally:
            prime_temp_wav_path.unlink(missing_ok=True)

    return time.perf_counter() - warm_started


def _run_subprocess_pipeline(
    text: str,
    piper_model_path: Path,
    piper_config_path: Path,
    temp_wav_path: Path,
    output_wav_path: Path,
    mode: PipelineMode,
) -> dict[str, float]:
    mode_settings = MODE_SETTINGS[mode]

    piper_started = time.perf_counter()
    _run_command(
        [
            sys.executable,
            "-m",
            "piper",
            "-m",
            str(piper_model_path),
            "-c",
            str(piper_config_path),
            "-f",
            str(temp_wav_path),
            text,
        ],
        "Piper synthesis",
    )
    piper_ended = time.perf_counter()

    _run_command(
        [
            sys.executable,
            "-m",
            "rvc_python",
            "cli",
            "-i",
            str(temp_wav_path),
            "-o",
            str(output_wav_path),
            "-mp",
            str(settings.TTS_RVC_MODEL_PATH),
            "-ip",
            str(settings.TTS_RVC_INDEX_PATH),
            "-de",
            settings.TTS_RVC_DEVICE,
            "-me",
            mode_settings["method"],
            "-ir",
            mode_settings["index_rate"],
            "-fr",
            mode_settings["filter_radius"],
            "-pr",
            mode_settings["protect"],
        ],
        "RVC conversion",
    )
    rvc_ended = time.perf_counter()

    return {
        "piper_s": piper_ended - piper_started,
        "rvc_s": rvc_ended - piper_ended,
        "inference_s": rvc_ended - piper_started,
    }


def run_local_tts_playback(
    text: str,
    mode: str | None = None,
    *,
    realtime: bool = False,
) -> LocalTTSPlaybackResult:
    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    total_started = time.perf_counter()
    requested_mode = resolve_mode(mode)
    resolved_mode: PipelineMode = "small" if realtime else requested_mode
    _ensure_runtime_support()
    _, piper_model_path, piper_config_path = _resolve_piper_model_files()

    temp_wav_path, _ = _build_paths()
    setup_ended = time.perf_counter()

    engine_prepare_started = time.perf_counter()
    engine = _get_warm_engine(
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        rvc_model_path=settings.TTS_RVC_MODEL_PATH,
        rvc_index_path=settings.TTS_RVC_INDEX_PATH,
        device=settings.TTS_RVC_DEVICE,
    )
    engine_prepare_seconds = time.perf_counter() - engine_prepare_started

    try:
        wav_bytes, timings_seconds, sample_rate = engine.synthesize_to_wav_bytes(
            text=text,
            mode=resolved_mode,
            realtime=realtime,
            temp_wav_path=temp_wav_path,
        )
    finally:
        temp_wav_path.unlink(missing_ok=True)

    timings_seconds["engine_prepare_s"] = engine_prepare_seconds
    timings_seconds["engine_load_s"] = engine.load_seconds
    timings_seconds["setup_s"] = setup_ended - total_started
    timings_seconds["total_s"] = time.perf_counter() - total_started

    pipeline_name = "piper-rvc-realtime" if realtime else "piper-rvc"
    return LocalTTSPlaybackResult(
        text=text,
        mode=resolved_mode,
        pipeline=pipeline_name,
        output_format="wav",
        wav_bytes=wav_bytes,
        sample_rate=sample_rate,
        timings_seconds=timings_seconds,
    )


def run_local_tts_pipeline(text: str, mode: str | None = None) -> LocalTTSPipelineResult:
    if not settings.TTS_LOCAL_ENABLED:
        raise LocalTTSPipelineError("Local TTS pipeline is disabled by configuration.")

    total_started = time.perf_counter()
    resolved_mode = resolve_mode(mode)
    _ensure_runtime_support()
    voice_id, piper_model_path, piper_config_path = _resolve_piper_model_files()

    temp_wav_path, output_wav_path = _build_paths()

    setup_ended = time.perf_counter()
    timings_seconds: dict[str, float]

    try:
        if settings.TTS_LOCAL_USE_WARM_ENGINE:
            engine_prepare_started = time.perf_counter()
            engine = _get_warm_engine(
                piper_model_path=piper_model_path,
                piper_config_path=piper_config_path,
                rvc_model_path=settings.TTS_RVC_MODEL_PATH,
                rvc_index_path=settings.TTS_RVC_INDEX_PATH,
                device=settings.TTS_RVC_DEVICE,
            )
            engine_prepare_seconds = time.perf_counter() - engine_prepare_started
            timings_seconds = engine.synthesize(
                text=text,
                mode=resolved_mode,
                temp_wav_path=temp_wav_path,
                output_wav_path=output_wav_path,
            )
            timings_seconds["engine_prepare_s"] = engine_prepare_seconds
            timings_seconds["engine_load_s"] = engine.load_seconds
        else:
            timings_seconds = _run_subprocess_pipeline(
                text=text,
                piper_model_path=piper_model_path,
                piper_config_path=piper_config_path,
                temp_wav_path=temp_wav_path,
                output_wav_path=output_wav_path,
                mode=resolved_mode,
            )
    finally:
        if temp_wav_path.exists() and not settings.TTS_LOCAL_KEEP_TMP:
            temp_wav_path.unlink(missing_ok=True)

    if not output_wav_path.exists():
        raise LocalTTSPipelineError(
            f"RVC conversion reported success but output file was not created: {output_wav_path}"
        )

    timings_seconds["setup_s"] = setup_ended - total_started
    timings_seconds["total_s"] = time.perf_counter() - total_started

    return LocalTTSPipelineResult(
        text=text,
        mode=resolved_mode,
        piper_model=voice_id,
        piper_model_path=piper_model_path,
        piper_config_path=piper_config_path,
        rvc_model_path=settings.TTS_RVC_MODEL_PATH,
        rvc_index_path=settings.TTS_RVC_INDEX_PATH,
        temp_wav_path=temp_wav_path,
        output_wav_path=output_wav_path,
        timings_seconds=timings_seconds,
    )
