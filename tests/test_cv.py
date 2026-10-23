import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

import CV.core as cv_core
from conftest import api
from constants import settings


class FakeTensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=float)

    def item(self) -> float:
        return float(self._values.reshape(-1)[0])

    def tolist(self) -> list:
        return self._values.tolist()

    def __getitem__(self, index):
        return FakeTensor(self._values[index])


class FakeYOLO:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, image, conf=None, verbose=None):
        self.calls.append({"size": image.size, "mode": image.mode, "conf": conf})
        boxes = [
            SimpleNamespace(cls=FakeTensor([0]), conf=FakeTensor([0.62]), xyxy=FakeTensor([[1.2, 2.6, 30.4, 40.5]])),
            SimpleNamespace(cls=FakeTensor([16]), conf=FakeTensor([0.91]), xyxy=FakeTensor([[5, 6, 20, 25]])),
            SimpleNamespace(cls=FakeTensor([0]), conf=FakeTensor([0.4]), xyxy=FakeTensor([[0, 0, 8, 8]])),
        ]
        return [SimpleNamespace(boxes=boxes, names={0: "person", 16: "dog"})]


def make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (200, 30, 30, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_analyze_returns_detections(client, monkeypatch):
    fake = FakeYOLO()
    monkeypatch.setattr(cv_core, "_get_model", lambda: fake)

    response = client.post(
        api(settings.CV_ROUTE_PREFIX, settings.CV_ANALYZE_ROUTE),
        files={"file": ("scene.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["image_size"] == [64, 48]
    assert [d["label"] for d in body["detections"]] == ["dog", "person", "person"]
    assert body["detections"][1]["bbox"] == [1, 3, 30, 40]
    assert body["summary"] == "Detected 2 person, 1 dog."
    assert fake.calls == [{"size": (64, 48), "mode": "RGB", "conf": settings.CV_CONFIDENCE_THRESHOLD}]


def test_non_image_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(cv_core, "_get_model", FakeYOLO)

    response = client.post(
        api(settings.CV_ROUTE_PREFIX, settings.CV_ANALYZE_ROUTE),
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]
