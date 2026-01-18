#app/tests/test_head_movement_integration.py
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_multiframe_sets_head_movement_true(monkeypatch):
    import app.application.liveness_service as ls

    # Simulamos rvec “quieto” y luego “movimiento fuerte”
    rvecs = [
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        np.array([[0.0], [1.5], [0.0]], dtype=float),  # cambio grande
    ]
    idx = {"i": 0}

    class FakeLM:
        def __init__(self, rvec):
            self.ear_left = 0.30
            self.ear_right = 0.30
            self.rvec = rvec
            self.tvec = np.zeros((3, 1), dtype=float)

    def fake_analyze(image_bgr, face_bbox):
        i = idx["i"]
        idx["i"] += 1
        return FakeLM(rvecs[min(i, len(rvecs) - 1)])

    monkeypatch.setattr(ls._landmarks, "analyze", fake_analyze)

    img_path = Path(__file__).parent / "assets" / "face.jpeg"
    img_bytes = img_path.read_bytes()

    files = [
        ("files", (f"f{i}.jpeg", __import__("io").BytesIO(img_bytes), "image/jpeg"))
        for i in range(6)
    ]

    res = client.post("/liveness/check-frames", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["checks"]["head_movement"] is True
