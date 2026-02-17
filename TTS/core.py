from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from constants import settings


router = APIRouter(prefix=settings.TTS_ROUTE_PREFIX, tags=["TTS"])


class TTSSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str | None = None
    output_format: str | None = None
    use_remote: bool | None = None


class TTSSynthesizeResponse(BaseModel):
    provider: str
    model: str
    voice: str
    output_format: str
    used_remote: bool
    placeholder: bool
    file_path: str
    timestamp_utc: str


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
    }


@router.post(settings.TTS_SYNTHESIZE_ROUTE, response_model=TTSSynthesizeResponse)
async def synthesize(req: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    use_remote = req.use_remote if req.use_remote is not None else settings.TTS_USE_REMOTE
    voice = req.voice or settings.TTS_DEFAULT_VOICE
    output_format = (req.output_format or settings.TTS_OUTPUT_FORMAT).lower()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    if use_remote:
        audio_bytes = await _remote_groq_tts(req, voice, output_format)
        output_ext = output_format
        placeholder = False
    else:
        audio_bytes = (
            "TTS placeholder file\n"
            f"text={req.text}\n"
            f"voice={voice}\n"
            f"model={settings.TTS_MODEL_NAME}\n"
        ).encode("utf-8")
        output_ext = "txt"
        output_format = "txt"
        placeholder = True

    output_path = settings.ASSETS_TTS_DIR / f"tts_{uuid4().hex}.{output_ext}"
    output_path.write_bytes(audio_bytes)

    return TTSSynthesizeResponse(
        provider=settings.TTS_PROVIDER,
        model=settings.TTS_MODEL_NAME,
        voice=voice,
        output_format=output_format,
        used_remote=use_remote,
        placeholder=placeholder,
        file_path=str(output_path),
        timestamp_utc=timestamp_utc,
    )
