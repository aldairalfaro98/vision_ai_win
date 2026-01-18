# app/tests/test_blink_integration.py
import io
import numpy as np
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_multiframe_sets_blink_detected_true(monkeypatch):
    import app.application.liveness_service as ls

    # Serie EAR con blink (abierto -> cerrado -> abierto)
    ears = [
        (0.30, 0.30),
        (0.29, 0.29),
        (0.30, 0.30),
        (0.12, 0.12),
        (0.11, 0.11),
        (0.29, 0.29),
    ]
    idx = {"i": 0}

    class FakeLM:
        def __init__(self, el, er):
            self.ear_left = el
            self.ear_right = er
            self.rvec = np.zeros((3, 1), dtype=float)
            self.tvec = np.zeros((3, 1), dtype=float)

    def fake_analyze(image_bgr, face_bbox):
        i = idx["i"]
        idx["i"] += 1
        el, er = ears[min(i, len(ears) - 1)]
        return FakeLM(el, er)

    monkeypatch.setattr(ls._landmarks, "analyze", fake_analyze)

    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    img_bytes = img_path.read_bytes()

    # Mandamos 6 frames para cumplir baseline_frames(3) + 2
    files = [
        ("files", (f"f{i}.jpeg", io.BytesIO(img_bytes), "image/jpeg"))
        for i in range(6)
    ]

    res = client.post("/liveness/check-frames", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["checks"]["blink_detected"] is True
