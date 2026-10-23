from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_route_prefix(value: str, fallback: str) -> str:
    raw = (value or fallback).strip()
    if not raw:
        return fallback
    if not raw.startswith("/"):
        raw = f"/{raw}"
    return raw.rstrip("/") or "/"


def _get_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _get_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value in allowed:
        return value
    return default


class Settings:
    def __init__(self) -> None:
        self.APP_NAME = os.getenv("APP_NAME", "BMO System")
        self.APP_ENV = os.getenv("APP_ENV", "development")
        self.APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
        self.APP_PORT = _get_int("APP_PORT", 8000)
        self.APP_RELOAD = _get_bool("APP_RELOAD", True)
        self.API_PREFIX = _normalize_route_prefix(os.getenv("API_PREFIX", "/api/v1"), "/api/v1")
        self.ROOT_ROUTE = _normalize_route_prefix(os.getenv("ROOT_ROUTE", "/"), "/")
        self.ROOT_HEALTH_ROUTE = _normalize_route_prefix(os.getenv("ROOT_HEALTH_ROUTE", "/health"), "/health")

        self.ASSETS_DIR = _get_path("ASSETS_DIR", ".")
        self.ASSETS_UPLOADS_DIR = _get_path("ASSETS_UPLOADS_DIR", "STT/assets/uploads")
        self.ASSETS_STT_DIR = _get_path("ASSETS_STT_DIR", "STT/assets")
        self.ASSETS_CV_DIR = _get_path("ASSETS_CV_DIR", "CV/assets")
        self.ASSETS_TTS_DIR = _get_path("ASSETS_TTS_DIR", "TTS/assets")
        self.ASSETS_LLM_DIR = _get_path("ASSETS_LLM_DIR", "LLM/assets")

        self.STT_ROUTE_PREFIX = _normalize_route_prefix(os.getenv("STT_ROUTE_PREFIX", "/stt"), "/stt")
        self.STT_HEALTH_ROUTE = _normalize_route_prefix(os.getenv("STT_HEALTH_ROUTE", "/health"), "/health")
        self.STT_CONFIG_ROUTE = _normalize_route_prefix(os.getenv("STT_CONFIG_ROUTE", "/config"), "/config")
        self.STT_TRANSCRIBE_ROUTE = _normalize_route_prefix(
            os.getenv("STT_TRANSCRIBE_ROUTE", "/transcribe"),
            "/transcribe",
        )
        self.STT_PROVIDER = os.getenv("STT_PROVIDER", "local")
        self.STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")
        self.STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
        self.STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
        self.STT_DEFAULT_LANGUAGE = os.getenv("STT_DEFAULT_LANGUAGE", "en")
        self.STT_BEAM_SIZE = _get_int("STT_BEAM_SIZE", 5)
        self.STT_TEMPERATURE = _get_float("STT_TEMPERATURE", 0.0)

        self.CV_ROUTE_PREFIX = _normalize_route_prefix(os.getenv("CV_ROUTE_PREFIX", "/cv"), "/cv")
        self.CV_HEALTH_ROUTE = _normalize_route_prefix(os.getenv("CV_HEALTH_ROUTE", "/health"), "/health")
        self.CV_CONFIG_ROUTE = _normalize_route_prefix(os.getenv("CV_CONFIG_ROUTE", "/config"), "/config")
        self.CV_ANALYZE_ROUTE = _normalize_route_prefix(os.getenv("CV_ANALYZE_ROUTE", "/analyze"), "/analyze")
        self.CV_PROVIDER = os.getenv("CV_PROVIDER", "local")
        self.CV_MODEL_NAME = os.getenv("CV_MODEL_NAME", "yolo26n")
        self.CV_WEIGHTS_DIR = _get_path("CV_WEIGHTS_DIR", "CV/weights")
        self.CV_CONFIDENCE_THRESHOLD = _get_float("CV_CONFIDENCE_THRESHOLD", 0.35)

        self.LLM_ROUTE_PREFIX = _normalize_route_prefix(os.getenv("LLM_ROUTE_PREFIX", "/llm"), "/llm")
        self.LLM_HEALTH_ROUTE = _normalize_route_prefix(os.getenv("LLM_HEALTH_ROUTE", "/health"), "/health")
        self.LLM_CONFIG_ROUTE = _normalize_route_prefix(os.getenv("LLM_CONFIG_ROUTE", "/config"), "/config")
        self.LLM_CHAT_ROUTE = _normalize_route_prefix(os.getenv("LLM_CHAT_ROUTE", "/chat"), "/chat")
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
        self.LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.7)
        self.LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 512)
        self.LLM_HISTORY_MESSAGES = _get_int("LLM_HISTORY_MESSAGES", 20)
        self.LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "/chat/completions")

        self.TTS_ROUTE_PREFIX = _normalize_route_prefix(os.getenv("TTS_ROUTE_PREFIX", "/tts"), "/tts")
        self.TTS_HEALTH_ROUTE = _normalize_route_prefix(os.getenv("TTS_HEALTH_ROUTE", "/health"), "/health")
        self.TTS_CONFIG_ROUTE = _normalize_route_prefix(os.getenv("TTS_CONFIG_ROUTE", "/config"), "/config")
        self.TTS_SYNTHESIZE_ROUTE = _normalize_route_prefix(
            os.getenv("TTS_SYNTHESIZE_ROUTE", "/synthesize"),
            "/synthesize",
        )
        self.TTS_PROVIDER = os.getenv("TTS_PROVIDER", "local")
        self.TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "pocket-tts")
        self.TTS_DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "clip-small")
        self.TTS_OUTPUT_FORMAT = os.getenv("TTS_OUTPUT_FORMAT", "wav")
        self.TTS_USE_REMOTE = _get_bool("TTS_USE_REMOTE", False)
        self.TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "/audio/speech")
        self.TTS_LOCAL_ENABLED = _get_bool("TTS_LOCAL_ENABLED", True)
        self.TTS_LOCAL_DEFAULT_TEXT = os.getenv(
            "TTS_LOCAL_DEFAULT_TEXT",
            "BMO says: Adventure time! Let's explore and have fun.",
        )
        self.TTS_LOCAL_DEFAULT_MODE: Literal["small", "medium", "large"] = _get_choice(
            "TTS_LOCAL_DEFAULT_MODE",
            "small",
            {"small", "medium", "large"},
        )
        self.TTS_LOCAL_OUT_DIR = _get_path("TTS_LOCAL_OUT_DIR", "TTS/assets/out")
        self.TTS_LOCAL_USE_WARM_ENGINE = _get_bool("TTS_LOCAL_USE_WARM_ENGINE", True)
        self.TTS_POCKET_MODEL_CONFIG = os.getenv("TTS_POCKET_MODEL_CONFIG", "b6369a24")
        self.TTS_POCKET_TEMPERATURE = _get_float("TTS_POCKET_TEMPERATURE", 0.7)
        self.TTS_POCKET_LSD_DECODE_STEPS = _get_int("TTS_POCKET_LSD_DECODE_STEPS", 1)
        self.TTS_POCKET_NOISE_CLAMP = (
            _get_float("TTS_POCKET_NOISE_CLAMP", 0.0) if os.getenv("TTS_POCKET_NOISE_CLAMP") is not None else None
        )
        self.TTS_POCKET_EOS_THRESHOLD = _get_float("TTS_POCKET_EOS_THRESHOLD", -4.0)
        self.TTS_POCKET_MAX_TOKENS = _get_int("TTS_POCKET_MAX_TOKENS", 1200)
        self.TTS_POCKET_MIN_TOKENS = _get_int("TTS_POCKET_MIN_TOKENS", 120)
        self.TTS_POCKET_DYNAMIC_MAX_TOKENS = _get_bool("TTS_POCKET_DYNAMIC_MAX_TOKENS", True)
        self.TTS_POCKET_TOKENS_PER_CHAR = _get_float("TTS_POCKET_TOKENS_PER_CHAR", 4.0)
        self.TTS_POCKET_FRAMES_AFTER_EOS = _get_int("TTS_POCKET_FRAMES_AFTER_EOS", 4)
        self.TTS_POCKET_TORCH_THREADS = _get_int(
            "TTS_POCKET_TORCH_THREADS",
            max(1, (os.cpu_count() or 4) // 2),
        )
        self.TTS_POCKET_VOICE_DIR = _get_path("TTS_POCKET_VOICE_DIR", "TTS/assets/voice")
        self.TTS_POCKET_CLIP_SMALL_PATH = _get_path("TTS_POCKET_CLIP_SMALL_PATH", "TTS/assets/voice/1.wav")
        self.TTS_POCKET_CLIP_MEDIUM_PATH = _get_path(
            "TTS_POCKET_CLIP_MEDIUM_PATH",
            "TTS/assets/voice/2.wav",
        )
        self.TTS_POCKET_CLIP_LARGE_PATH = _get_path("TTS_POCKET_CLIP_LARGE_PATH", "TTS/assets/voice/3.wav")

        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.GROQ_TIMEOUT_SECONDS = _get_float("GROQ_TIMEOUT_SECONDS", 45.0)

    def ensure_asset_dirs(self) -> None:
        for path in (
            self.ASSETS_DIR,
            self.ASSETS_UPLOADS_DIR,
            self.ASSETS_STT_DIR,
            self.ASSETS_CV_DIR,
            self.ASSETS_TTS_DIR,
            self.ASSETS_LLM_DIR,
            self.TTS_LOCAL_OUT_DIR,
            self.TTS_POCKET_VOICE_DIR,
            self.TTS_POCKET_CLIP_SMALL_PATH.parent,
            self.TTS_POCKET_CLIP_MEDIUM_PATH.parent,
            self.TTS_POCKET_CLIP_LARGE_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_asset_dirs()
