from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from constants import settings


router = APIRouter(prefix=settings.STT_ROUTE_PREFIX, tags=["STT"])


class STTTranscriptionResponse(BaseModel):
    provider: str
    model: str
    language: str
    source_file: str
    saved_to: str
    transcription: str
    placeholder: bool = True


@router.get(settings.STT_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "stt", "provider": settings.STT_PROVIDER}


@router.get(settings.STT_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.STT_PROVIDER,
        "model": settings.STT_MODEL_NAME,
        "default_language": settings.STT_DEFAULT_LANGUAGE,
        "beam_size": settings.STT_BEAM_SIZE,
        "temperature": settings.STT_TEMPERATURE,
        "route_prefix": settings.STT_ROUTE_PREFIX,
    }


@router.post(settings.STT_TRANSCRIBE_ROUTE, response_model=STTTranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> STTTranscriptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio filename is required.")

    extension = Path(file.filename).suffix or ".bin"
    dest_name = f"stt_{uuid4().hex}{extension}"
    destination = settings.ASSETS_STT_DIR / dest_name

    file_bytes = await file.read()
    destination.write_bytes(file_bytes)

    chosen_language = language or settings.STT_DEFAULT_LANGUAGE
    text = (
        f"[STT placeholder] Received '{file.filename}' "
        f"({len(file_bytes)} bytes) with local model '{settings.STT_MODEL_NAME}'."
    )

    return STTTranscriptionResponse(
        provider=settings.STT_PROVIDER,
        model=settings.STT_MODEL_NAME,
        language=chosen_language,
        source_file=file.filename,
        saved_to=str(destination),
        transcription=text,
    )
