# app/tests/test_no_double_detection.py
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_multiframe_detector_called_once_per_frame(monkeypatch):
    # IMPORTANTE: parcheamos el detector del predictor global
    import app.application.liveness_service as ls

    calls = {"n": 0}
    original = ls._predictor.detector.detect

    def counted_detect(img):
        calls["n"] += 1
        return original(img)

    monkeypatch.setattr(ls._predictor.detector, "detect", counted_detect)

    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    with img_path.open("rb") as f1, img_path.open("rb") as f2, img_path.open("rb") as f3:
        files = [
            ("files", ("f1.jpeg", f1, "image/jpeg")),
            ("files", ("f2.jpeg", f2, "image/jpeg")),
            ("files", ("f3.jpeg", f3, "image/jpeg")),
        ]
        res = client.post("/liveness/check-frames", files=files)

    assert res.status_code == 200

    # Si está “bien”, detect() se llama una vez por frame (3),
    # no dos veces por frame (6).
    assert calls["n"] == 3
