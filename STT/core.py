import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from constants import settings


router = APIRouter(prefix=settings.STT_ROUTE_PREFIX, tags=["STT"])

_MODEL = None
_MODEL_LOCK = threading.Lock()


class STTSegment(BaseModel):
    start: float
    end: float
    text: str


class STTTranscriptionResponse(BaseModel):
    provider: str
    model: str
    language: str
    language_probability: float | None
    duration_seconds: float
    source_file: str
    saved_to: str
    transcription: str
    segments: list[STTSegment]


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="STT needs faster-whisper. Install it with `pip install -r requirements.txt`.",
            ) from exc

        try:
            _MODEL = WhisperModel(
                settings.STT_MODEL_NAME,
                device=settings.STT_DEVICE,
                compute_type=settings.STT_COMPUTE_TYPE,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not load Whisper model '{settings.STT_MODEL_NAME}' "
                    f"(device={settings.STT_DEVICE}, compute_type={settings.STT_COMPUTE_TYPE}). "
                    "The first request downloads the weights from Hugging Face, so it needs internet access. "
                    f"Cause: {exc}"
                ),
            ) from exc
        return _MODEL


def _transcribe_file(path: Path, language: str | None) -> tuple[list[STTSegment], object]:
    model = _get_model()
    segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=settings.STT_BEAM_SIZE,
        temperature=settings.STT_TEMPERATURE,
    )
    # transcribe() returns a lazy generator; decoding happens while iterating it.
    return [STTSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments], info


@router.get(settings.STT_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "stt", "provider": settings.STT_PROVIDER, "model_loaded": _MODEL is not None}


@router.get(settings.STT_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.STT_PROVIDER,
        "model": settings.STT_MODEL_NAME,
        "device": settings.STT_DEVICE,
        "compute_type": settings.STT_COMPUTE_TYPE,
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

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    extension = Path(file.filename).suffix or ".bin"
    destination = settings.ASSETS_STT_DIR / f"stt_{uuid4().hex}{extension}"
    destination.write_bytes(file_bytes)

    requested_language = (language or settings.STT_DEFAULT_LANGUAGE).strip().lower()
    whisper_language = None if requested_language == "auto" else requested_language

    try:
        segments, info = await run_in_threadpool(_transcribe_file, destination, whisper_language)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode or transcribe '{file.filename}': {exc}",
        ) from exc

    return STTTranscriptionResponse(
        provider=settings.STT_PROVIDER,
        model=settings.STT_MODEL_NAME,
        language=info.language,
        language_probability=info.language_probability if whisper_language is None else None,
        duration_seconds=info.duration,
        source_file=file.filename,
        saved_to=str(destination),
        transcription=" ".join(segment.text for segment in segments).strip(),
        segments=segments,
    )
