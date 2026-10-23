from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from constants import settings
from TTS.local_pipeline import (
    MODE_SETTINGS,
    LocalTTSInputError,
    LocalTTSPipelineError,
    list_voice_samples,
    run_local_tts_pipeline,
)


router = APIRouter(prefix=settings.TTS_ROUTE_PREFIX, tags=["TTS"])

TTS_FILES_ROUTE = "/files"


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
    file_url: str
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
        "voice_samples": [path.name for path in list_voice_samples()],
        "local_modes": MODE_SETTINGS,
    }


@router.post(settings.TTS_SYNTHESIZE_ROUTE, response_model=TTSSynthesizeResponse)
async def synthesize(req: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty or whitespace only.")

    output_format = (req.output_format or settings.TTS_OUTPUT_FORMAT).lower()
    if output_format != "wav":
        raise HTTPException(status_code=422, detail="PocketTTS setup only supports wav output format.")
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        result = await run_in_threadpool(run_local_tts_pipeline, req.text.strip(), req.mode, req.voice)
    except LocalTTSInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LocalTTSPipelineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    file_url = f"{settings.API_PREFIX}{settings.TTS_ROUTE_PREFIX}{TTS_FILES_ROUTE}/{result.output_wav_path.name}"
    return TTSSynthesizeResponse(
        provider="local",
        model=result.model_name,
        voice=result.clip_path.name,
        output_format=result.output_format,
        used_remote=False,
        mode=result.mode,
        pipeline=result.pipeline,
        file_path=str(result.output_wav_path),
        file_url=file_url,
        timestamp_utc=timestamp_utc,
        timings_seconds=result.timings_seconds,
    )


@router.get(TTS_FILES_ROUTE + "/{file_name}")
async def download(file_name: str) -> FileResponse:
    out_dir = settings.TTS_LOCAL_OUT_DIR.resolve()
    path = (out_dir / file_name).resolve()
    if path.parent != out_dir or path.suffix != ".wav" or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No generated wav named '{file_name}'.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
