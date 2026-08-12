from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from constants import settings
from TTS.local_pipeline import MODE_SETTINGS, LocalTTSPipelineError, run_local_tts_pipeline


router = APIRouter(prefix=settings.TTS_ROUTE_PREFIX, tags=["TTS"])


class TTSSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str | None = None
    output_format: str | None = None
    mode: Literal["small", "medium", "large"] | None = None


class TTSSynthesizeResponse(BaseModel):
    provider: str
    model: str
    voice: str
    output_format: str
    used_remote: bool
    mode: str | None
    pipeline: str | None
    file_path: str
    timestamp_utc: str
    timings_seconds: dict[str, float] | None = None


@router.get(settings.TTS_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "tts", "provider": settings.TTS_PROVIDER}


@router.get(settings.TTS_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.TTS_PROVIDER,
        "model": settings.TTS_MODEL_NAME,
        "default_voice": settings.TTS_DEFAULT_VOICE,
        "default_format": settings.TTS_OUTPUT_FORMAT,
        "route_prefix": settings.TTS_ROUTE_PREFIX,
        "local_enabled": settings.TTS_LOCAL_ENABLED,
        "local_default_mode": settings.TTS_LOCAL_DEFAULT_MODE,
        "local_use_warm_engine": settings.TTS_LOCAL_USE_WARM_ENGINE,
        "pocket_model_config": settings.TTS_POCKET_MODEL_CONFIG,
        "pocket_temperature": settings.TTS_POCKET_TEMPERATURE,
        "pocket_lsd_decode_steps": settings.TTS_POCKET_LSD_DECODE_STEPS,
        "pocket_noise_clamp": settings.TTS_POCKET_NOISE_CLAMP,
        "pocket_eos_threshold": settings.TTS_POCKET_EOS_THRESHOLD,
        "pocket_max_tokens": settings.TTS_POCKET_MAX_TOKENS,
        "pocket_frames_after_eos": settings.TTS_POCKET_FRAMES_AFTER_EOS,
        "pocket_clips": {
            "small": str(settings.TTS_POCKET_CLIP_SMALL_PATH),
            "medium": str(settings.TTS_POCKET_CLIP_MEDIUM_PATH),
            "large": str(settings.TTS_POCKET_CLIP_LARGE_PATH),
        },
        "pocket_voice_dir": str(settings.TTS_POCKET_VOICE_DIR),
        "local_modes": MODE_SETTINGS,
    }


@router.post(settings.TTS_SYNTHESIZE_ROUTE, response_model=TTSSynthesizeResponse)
async def synthesize(req: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty or whitespace only.")

    voice = req.voice or settings.TTS_DEFAULT_VOICE
    output_format = (req.output_format or settings.TTS_OUTPUT_FORMAT).lower()
    if output_format != "wav":
        raise HTTPException(status_code=422, detail="PocketTTS setup only supports wav output format.")
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        result = run_local_tts_pipeline(text=req.text.strip(), mode=req.mode, voice=req.voice)
    except LocalTTSPipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TTSSynthesizeResponse(
        provider="local",
        model=result.model_name,
        voice=voice,
        output_format=result.output_format,
        used_remote=False,
        mode=result.mode,
        pipeline=result.pipeline,
        file_path=str(result.output_wav_path),
        timestamp_utc=timestamp_utc,
        timings_seconds=result.timings_seconds,
    )
