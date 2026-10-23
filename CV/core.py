import io
import threading
from collections import Counter
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from constants import settings


router = APIRouter(prefix=settings.CV_ROUTE_PREFIX, tags=["CV"])

_MODEL = None
_MODEL_LOCK = threading.Lock()


class CVDetection(BaseModel):
    label: str
    confidence: float
    bbox: list[int]


class CVAnalyzeResponse(BaseModel):
    provider: str
    model: str
    source_file: str
    saved_to: str
    image_size: list[int]
    detections: list[CVDetection]
    summary: str


def _weights_path() -> Path:
    return settings.CV_WEIGHTS_DIR / f"{settings.CV_MODEL_NAME}.pt"


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="CV needs ultralytics. Install it with `pip install -r requirements.txt`.",
            ) from exc

        weights = _weights_path()
        weights.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Ultralytics downloads official weights (e.g. yolo26n.pt) to this path when it is missing.
            _MODEL = YOLO(str(weights))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not load YOLO weights '{weights}'. The first request downloads them from the "
                    f"Ultralytics GitHub releases, so it needs internet access. Cause: {exc}"
                ),
            ) from exc
        return _MODEL


def _detect(image) -> list[CVDetection]:
    model = _get_model()
    result = model.predict(image, conf=settings.CV_CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            CVDetection(
                label=result.names[class_id],
                confidence=round(float(box.conf.item()), 4),
                bbox=[int(round(v)) for v in box.xyxy[0].tolist()],
            )
        )
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def _summarize(detections: list[CVDetection]) -> str:
    if not detections:
        return "No objects detected above the confidence threshold."
    counts = Counter(d.label for d in detections)
    return "Detected " + ", ".join(f"{count} {label}" for label, count in counts.most_common()) + "."


@router.get(settings.CV_HEALTH_ROUTE)
async def health() -> dict:
    return {"status": "ok", "module": "cv", "provider": settings.CV_PROVIDER, "model_loaded": _MODEL is not None}


@router.get(settings.CV_CONFIG_ROUTE)
async def config() -> dict:
    return {
        "provider": settings.CV_PROVIDER,
        "model": settings.CV_MODEL_NAME,
        "weights_path": str(_weights_path()),
        "confidence_threshold": settings.CV_CONFIDENCE_THRESHOLD,
        "route_prefix": settings.CV_ROUTE_PREFIX,
    }


@router.post(settings.CV_ANALYZE_ROUTE, response_model=CVAnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> CVAnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    file_bytes = await file.read()
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="CV needs Pillow (installed with ultralytics).") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"'{file.filename}' is not a readable image: {exc}") from exc

    extension = Path(file.filename).suffix or ".bin"
    destination = settings.ASSETS_CV_DIR / f"cv_{uuid4().hex}{extension}"
    destination.write_bytes(file_bytes)

    detections = await run_in_threadpool(_detect, image.convert("RGB"))

    return CVAnalyzeResponse(
        provider=settings.CV_PROVIDER,
        model=settings.CV_MODEL_NAME,
        source_file=file.filename,
        saved_to=str(destination),
        image_size=[image.width, image.height],
        detections=detections,
        summary=_summarize(detections),
    )
