from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from constants import settings


router = APIRouter(prefix=settings.CV_ROUTE_PREFIX, tags=["CV"])


class CVDetection(BaseModel):
    label: str
    confidence: float
    bbox: list[int]


class CVAnalyzeResponse(BaseModel):
    provider: str
    model: str
    task: str
    source_file: str
    saved_to: str
    detections: list[CVDetection]
    summary: str
    placeholder: bool = True


@router.get(settings.CV_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "cv", "provider": settings.CV_PROVIDER}


@router.get(settings.CV_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.CV_PROVIDER,
        "model": settings.CV_MODEL_NAME,
        "default_task": settings.CV_DEFAULT_TASK,
        "confidence_threshold": settings.CV_CONFIDENCE_THRESHOLD,
        "route_prefix": settings.CV_ROUTE_PREFIX,
    }


@router.post(settings.CV_ANALYZE_ROUTE, response_model=CVAnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    task: str | None = Form(default=None),
) -> CVAnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    extension = Path(file.filename).suffix or ".bin"
    dest_name = f"cv_{uuid4().hex}{extension}"
    destination = settings.ASSETS_CV_DIR / dest_name

    file_bytes = await file.read()
    destination.write_bytes(file_bytes)

    chosen_task = task or settings.CV_DEFAULT_TASK
    mock_detection = CVDetection(
        label="placeholder-object",
        confidence=max(settings.CV_CONFIDENCE_THRESHOLD, 0.51),
        bbox=[32, 48, 192, 220],
    )

    return CVAnalyzeResponse(
        provider=settings.CV_PROVIDER,
        model=settings.CV_MODEL_NAME,
        task=chosen_task,
        source_file=file.filename,
        saved_to=str(destination),
        detections=[mock_detection],
        summary=f"[CV placeholder] Processed '{file.filename}' using local model '{settings.CV_MODEL_NAME}'.",
    )
