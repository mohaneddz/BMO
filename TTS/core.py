from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from constants import settings
from TTS.local_pipeline import MODE_SETTINGS, LocalTTSPipelineError, run_local_tts_pipeline


router = APIRouter(prefix=settings.TTS_ROUTE_PREFIX, tags=["TTS"])


class TTSSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str | None = None
    output_format: str | None = None
    use_remote: bool | None = None
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


async def _remote_groq_tts(req: TTSSynthesizeRequest, voice: str, output_format: str) -> bytes:
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")

    endpoint = f"{settings.GROQ_BASE_URL}{settings.TTS_ENDPOINT}"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    body = {
        "model": settings.TTS_MODEL_NAME,
        "voice": voice,
        "input": req.text,
        "format": output_format,
    }

    async with httpx.AsyncClient(timeout=settings.GROQ_TIMEOUT_SECONDS) as client:
        response = await client.post(endpoint, headers=headers, json=body)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"GROQ TTS request failed ({response.status_code}): {response.text}",
        )

    return response.content


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
        "use_remote": settings.TTS_USE_REMOTE,
        "route_prefix": settings.TTS_ROUTE_PREFIX,
        "endpoint": settings.TTS_ENDPOINT,
        "local_enabled": settings.TTS_LOCAL_ENABLED,
        "local_default_mode": settings.TTS_LOCAL_DEFAULT_MODE,
        "local_piper_model": settings.TTS_LOCAL_PIPER_MODEL,
        "local_use_warm_engine": settings.TTS_LOCAL_USE_WARM_ENGINE,
        "local_prime_on_prewarm": settings.TTS_LOCAL_PRIME_ON_PREWARM,
        "local_rvc_model_path": str(settings.TTS_RVC_MODEL_PATH),
        "local_rvc_index_path": str(settings.TTS_RVC_INDEX_PATH),
        "local_rvc_device": settings.TTS_RVC_DEVICE,
        "local_modes": MODE_SETTINGS,
    }


@router.post(settings.TTS_SYNTHESIZE_ROUTE, response_model=TTSSynthesizeResponse)
async def synthesize(req: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text input cannot be empty or whitespace only.")

    use_remote = req.use_remote if req.use_remote is not None else settings.TTS_USE_REMOTE
    voice = req.voice or settings.TTS_DEFAULT_VOICE
    output_format = (req.output_format or settings.TTS_OUTPUT_FORMAT).lower()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    if use_remote:
        audio_bytes = await _remote_groq_tts(req, voice, output_format)
        output_ext = output_format
        output_path = settings.ASSETS_TTS_DIR / f"tts_{uuid4().hex}.{output_ext}"
        output_path.write_bytes(audio_bytes)

        return TTSSynthesizeResponse(
            provider=settings.TTS_PROVIDER,
            model=settings.TTS_MODEL_NAME,
            voice=voice,
            output_format=output_format,
            used_remote=True,
            mode=req.mode,
            pipeline="remote-groq",
            file_path=str(output_path),
            timestamp_utc=timestamp_utc,
            timings_seconds=None,
        )

    try:
        result = run_local_tts_pipeline(text=req.text.strip(), mode=req.mode)
    except LocalTTSPipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TTSSynthesizeResponse(
        provider="local",
        model=result.rvc_model_path.name,
        voice=result.piper_model,
        output_format=result.output_format,
        used_remote=False,
        mode=result.mode,
        pipeline=result.pipeline,
        file_path=str(result.output_wav_path),
        timestamp_utc=timestamp_utc,
        timings_seconds=result.timings_seconds,
    )
